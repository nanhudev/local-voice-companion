"""The realtime turn orchestrator.

Microphone -> VAD -> ASR -> transcript -> LLM stream -> clause chunker ->
TTS stream -> audio queue -> playback.

Three properties matter more than anything else here:

1. Streaming.  TTS starts before the LLM finishes writing.
2. Cancellation. A barge-in tears down every stage through one token, and no
   stale audio can reach playback.
3. Backpressure. Every queue between stages is bounded with a drop policy.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from ..core.cancellation import CancellationToken
from ..core.errors import CancelledTurn, ProviderUnavailable
from ..core.events import EventType, TurnTimeline
from ..core.session import Session
from ..core.state import TurnState
from ..core.types import AudioChunk, AudioFormat, ChatMessage
from ..pipeline.chunker import AdaptiveTextChunker
from ..pipeline.queue import BoundedQueue
from ..providers.base import ASRProvider, LLMProvider, ProviderHealth, TTSProvider, VADProvider, ensure_ready

MAX_TRANSCRIPT_RETRY_CHARS = 2

#: Sub-millisecond timing for per-stage latency. `time.monotonic()` on Windows
#: is tick-quantised and cannot resolve a single ASR call.
_perf_counter = time.perf_counter


@dataclass
class Pipeline:
    """The four loaded providers that will serve turns in this runtime."""

    asr: ASRProvider | None = None
    llm: LLMProvider | None = None
    tts: TTSProvider | None = None
    vad: VADProvider | None = None

    def require(self, *names: str) -> None:
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ProviderUnavailable(
                f"pipeline stages unavailable: {', '.join(missing)}", missing=missing
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in ("asr", "llm", "tts", "vad"):
            provider = getattr(self, name)
            if provider is None:
                continue
            descriptor = provider.descriptor()
            payload[name] = {
                "provider": descriptor.id,
                "streaming": descriptor.streaming,
                "state": provider.lifecycle.state.value,
            }
        return payload


@dataclass
class TurnRequest:
    """Either `audio` or `text` must be supplied."""

    audio: AudioChunk | None = None
    text: str = ""
    system_prompt: str = "You are a concise and friendly voice assistant."
    voice: str | None = None
    max_tokens: int = 64
    temperature: float = 0.35
    language: str = ""
    speak: bool = True
    chunk_min_chars: int = 8
    chunk_max_chars: int = 120

    def __post_init__(self) -> None:
        if not self.text and self.audio is None:
            raise ValueError("TurnRequest needs either text or audio")


@dataclass
class TurnResult:
    turn_id: str
    transcript: str = ""
    reply: str = ""
    audio_bytes: int = 0
    chunks_spoken: int = 0
    cancelled: bool = False
    timeline: TurnTimeline | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "transcript": self.transcript,
            "reply": self.reply,
            "audio_bytes": self.audio_bytes,
            "chunks_spoken": self.chunks_spoken,
            "cancelled": self.cancelled,
            "error": self.error,
            "timeline": self.timeline.to_dict() if self.timeline else None,
        }


@dataclass
class PlaybackStats:
    bytes_enqueued: int = 0
    chunks: int = 0
    started: bool = False


async def transcribe_turn(
    pipeline: Pipeline,
    audio: AudioChunk,
    language: str,
    *,
    timeline: TurnTimeline,
    token: CancellationToken,
    session: Session,
) -> str:
    """Run one-shot ASR with proper lifecycle guarding and timing."""

    pipeline.require("asr")
    assert pipeline.asr is not None
    ensure_ready(pipeline.asr)

    timeline.mark("vad_end")
    timeline.mark("asr_start")
    started = _perf_counter()
    transcript = ""
    try:
        transcript = await pipeline.asr.transcribe(audio, language=language, token=token)
    finally:
        timeline.mark("asr_end")
        session.emit(
            EventType.ASR_FINAL,
            {"text": transcript, "elapsed_ms": int((_perf_counter() - started) * 1000)},
            turn_id=timeline.turn_id,
        )
    token.raise_if_cancelled()
    return transcript.strip()


async def run_turn(session: Session, pipeline: Pipeline, request: TurnRequest) -> TurnResult:
    """Execute a complete conversational turn."""

    timeline, token = session.begin_turn()
    result = TurnResult(turn_id=timeline.turn_id, timeline=timeline)
    language = request.language or session.language

    try:
        if request.audio is not None:
            session.set_state(TurnState.TRANSCRIBING)
            transcript = await transcribe_turn(
                pipeline, request.audio, language, timeline=timeline, token=token, session=session
            )
            if len(transcript) < MAX_TRANSCRIPT_RETRY_CHARS:
                session.emit(
                    EventType.ASR_FINAL,
                    {"text": "", "too_short": True},
                    turn_id=timeline.turn_id,
                )
                session.set_state(TurnState.LISTENING)
                session.finish_turn("ignored")
                return result
            result.transcript = transcript
            session.last_transcript = transcript
            session.emit(EventType.ASR_FINAL, {"text": transcript}, turn_id=timeline.turn_id)
        else:
            timeline.mark("vad_end")
            timeline.mark("asr_start")
            timeline.mark("asr_end")
            transcript = request.text.strip()
            result.transcript = transcript
            session.emit(EventType.ASR_FINAL, {"text": transcript, "source": "text"}, turn_id=timeline.turn_id)

        session.append("user", transcript)

        if request.speak:
            reply = await _generate_and_speak(
                session, pipeline, request, transcript, timeline, token, result, language
            )
        else:
            reply = await _generate_text_only(session, pipeline, request, transcript, timeline, token, result)

        result.reply = reply
        session.last_reply = reply
        session.append("assistant", reply)
        timeline.mark("playback_end")
        session.set_state(TurnState.LISTENING)
        session.finish_turn("completed")
        return result

    except CancelledTurn as exc:
        result.cancelled = True
        result.error = str(exc)
        session.set_state(TurnState.LISTENING, detail=str(exc))
        session.finish_turn("cancelled")
        return result
    except asyncio.CancelledError:
        session.set_state(TurnState.LISTENING, detail="task cancelled")
        session.finish_turn("cancelled")
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed wire error below
        result.error = f"{type(exc).__name__}: {exc}"
        session.emit(
            EventType.ERROR,
            {"message": str(exc), "error_type": type(exc).__name__},
            turn_id=timeline.turn_id,
        )
        session.set_state(TurnState.ERROR, detail=str(exc))
        session.finish_turn("failed")
        return result


async def _generate_text_only(
    session: Session,
    pipeline: Pipeline,
    request: TurnRequest,
    transcript: str,
    timeline: TurnTimeline,
    token: CancellationToken,
    result: TurnResult,
) -> str:
    pipeline.require("llm")
    assert pipeline.llm is not None
    ensure_ready(pipeline.llm)

    session.set_state(TurnState.THINKING)
    timeline.mark("llm_start")
    # Emitted before the first token, not after: a client needs to know the
    # model was invoked so it can distinguish "thinking" from "idle" while the
    # first token is still in flight.
    session.emit(EventType.LLM_STARTED, {}, turn_id=timeline.turn_id)
    messages = session.messages(request.system_prompt, transcript)
    pieces: list[str] = []
    first = True
    async for delta in pipeline.llm.stream(
        messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        token=token,
    ):
        if first:
            timeline.mark("llm_first_token")
            first = False
        pieces.append(delta)
        session.emit(EventType.LLM_DELTA, {"text": delta, "full": "".join(pieces)}, turn_id=timeline.turn_id)
    timeline.mark("llm_end")
    reply = "".join(pieces).strip()
    session.emit(EventType.LLM_COMPLETED, {"text": reply}, turn_id=timeline.turn_id)
    token.raise_if_cancelled()
    return reply


async def _generate_and_speak(
    session: Session,
    pipeline: Pipeline,
    request: TurnRequest,
    transcript: str,
    timeline: TurnTimeline,
    token: CancellationToken,
    result: TurnResult,
    language: str,
) -> str:
    """Stream LLM -> chunker -> bounded TTS queue -> audio consumer."""

    pipeline.require("llm", "tts")
    assert pipeline.llm is not None and pipeline.tts is not None
    ensure_ready(pipeline.llm)
    ensure_ready(pipeline.tts)

    messages = session.messages(request.system_prompt, transcript)
    text_queue: BoundedQueue[str | None] = BoundedQueue(8, policy="oldest", name="tts-text")
    audio_queue: BoundedQueue[tuple[bytes, AudioFormat] | None] = BoundedQueue(
        8, policy="oldest", name="tts-audio"
    )

    synth_task = asyncio.create_task(
        _synthesis_worker(pipeline.tts, text_queue, audio_queue, request, timeline, token, session, language),
        name=f"tts-{timeline.turn_id}",
    )
    emit_task = asyncio.create_task(
        _audio_emitter(audio_queue, timeline, token, session, result),
        name=f"audio-{timeline.turn_id}",
    )

    session.set_state(TurnState.THINKING)
    timeline.mark("llm_start")
    session.emit(EventType.LLM_STARTED, {}, turn_id=timeline.turn_id)
    chunker = AdaptiveTextChunker(
        min_chars=request.chunk_min_chars,
        max_chars=request.chunk_max_chars,
        language=language,
    )
    pieces: list[str] = []
    spoken: list[str] = []
    first = True

    try:
        async for delta in pipeline.llm.stream(
            messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            token=token,
        ):
            token.raise_if_cancelled()
            if first:
                timeline.mark("llm_first_token")
                session.set_state(TurnState.SYNTHESIZING)
                first = False
            pieces.append(delta)
            session.emit(
                EventType.LLM_DELTA, {"text": delta, "full": "".join(pieces)}, turn_id=timeline.turn_id
            )
            for chunk in chunker.push(delta):
                spoken.append(chunk)
                await text_queue.put(chunk)
    finally:
        for chunk in chunker.flush():
            spoken.append(chunk)
            await text_queue.put(chunk)
        await text_queue.put(None)

    timeline.mark("llm_end")
    reply = "".join(pieces).strip()
    session.emit(EventType.LLM_COMPLETED, {"text": reply}, turn_id=timeline.turn_id)

    await synth_task
    await audio_queue.put(None)
    await emit_task

    result.chunks_spoken = len(spoken)
    token.raise_if_cancelled()
    return reply


async def _synthesis_worker(
    tts: TTSProvider,
    text_queue: BoundedQueue,
    audio_queue: BoundedQueue,
    request: TurnRequest,
    timeline: TurnTimeline,
    token: CancellationToken,
    session: Session,
    language: str,
) -> None:
    """Consume text chunks and produce audio, one chunk at a time."""

    started = False
    chunks_synthesized = 0
    while True:
        try:
            item = await text_queue.get()
        except asyncio.CancelledError:
            raise
        if item is None:
            if started:
                # Symmetry with `llm.completed`: a client tracking synthesis
                # needs a terminal frame, not just a stream that stops.
                session.emit(
                    EventType.TTS_COMPLETED,
                    {"chunks": chunks_synthesized},
                    turn_id=timeline.turn_id,
                )
            return
        token.raise_if_cancelled()
        if not started:
            timeline.mark("tts_start")
            started = True
            session.emit(EventType.TTS_STARTED, {}, turn_id=timeline.turn_id)
        audio, fmt = await tts.synthesize(
            item, voice=request.voice, language=language, token=token
        )
        if audio:
            chunks_synthesized += 1
            await audio_queue.put((audio, fmt))


async def _audio_emitter(
    audio_queue: BoundedQueue,
    timeline: TurnTimeline,
    token: CancellationToken,
    session: Session,
    result: TurnResult,
) -> None:
    """Deliver finished audio fragments and stop hard on cancellation."""

    started = False
    while True:
        item = await audio_queue.get()
        if item is None:
            if started:
                # Playback is client-side, so "finished" here means "every
                # fragment has been handed over", which is what a client needs
                # to release its audio device and leave SPEAKING.
                timeline.mark("playback_end")
                session.emit(
                    EventType.PLAYBACK_FINISHED,
                    {"bytes": result.audio_bytes, "chunks": result.chunks_spoken},
                    turn_id=timeline.turn_id,
                )
            return
        audio, fmt = item
        if token.cancelled:
            # Never let audio from an interrupted turn reach the client.
            audio_queue.drain()
            raise CancelledTurn("audio delivery interrupted")
        if not started:
            timeline.mark("tts_first_audio")
            timeline.mark("playback_start")
            session.set_state(TurnState.SPEAKING)
            session.emit(EventType.PLAYBACK_STARTED, {}, turn_id=timeline.turn_id)
            started = True
        result.audio_bytes += len(audio)
        session.emit(
            EventType.TTS_AUDIO,
            {
                "audio_base64": base64.b64encode(audio).decode("ascii"),
                "mime_type": fmt.mime_type,
                "bytes": len(audio),
            },
            turn_id=timeline.turn_id,
        )
