"""Streaming turns: `asr.partial` has to leave the runtime, not just exist.

PHASE 3A built the parts -- frames, a stability tracker, a provider that can
decode while audio arrives -- and left one claim unproven: that a hypothesis
reaches a client *before* the speaker finishes. `EventType.ASR_PARTIAL` was
declared and emitted by nobody, which is a claim worth exactly nothing.

These tests pin the emission. They deliberately use a provider whose hypotheses
are scripted, because a provider that knows its answer up front (``FakeASR``)
cannot fail the interesting way: it would pass an assertion about partials
by producing them all at the end.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from local_voice_companion.core.audio import AudioFrame, iter_frames
from local_voice_companion.core.orchestrator import Pipeline, TurnRequest, run_turn
from local_voice_companion.core.session import Session
from local_voice_companion.core.types import AudioChunk
from local_voice_companion.providers.fake import (
    FakeASR,
    FakeLLM,
    FakeTTS,
    ScriptedStreamingASR,
)

SAMPLE_RATE = 16000
FRAME_MS = 20
PARTIALS = ["今天", "今天天气", "今天天气怎么样"]
FINAL = "今天天气怎么样"


def _pcm(milliseconds: int = FRAME_MS, amplitude: int = 1000) -> bytes:
    count = int(SAMPLE_RATE * milliseconds / 1000)
    return b"".join(struct.pack("<h", amplitude) for _ in range(count))


def _frames(count: int, delay: float = 0.0):
    async def source():
        for index in range(count):
            if delay:
                await asyncio.sleep(delay)
            yield AudioFrame(
                pcm=_pcm(),
                sample_rate=SAMPLE_RATE,
                sequence=index,
                captured_at=0.0,
            )

    return source()


def _pipeline(asr) -> Pipeline:
    pipeline = Pipeline(asr=asr, llm=FakeLLM({}), tts=FakeTTS({}))
    for name in ("asr", "llm", "tts"):
        provider = getattr(pipeline, name)
        provider.lifecycle.transition("AVAILABLE")
        provider.lifecycle.transition("LOADING")
        provider.lifecycle.transition("READY")
    return pipeline


def _events(session: Session, event_type: str) -> list[dict]:
    return [item for item in session.bus.after(0) if item["type"] == event_type]


class TestStreamingTurnEmitsPartials:
    def test_partials_are_emitted_before_the_final(self) -> None:
        session = Session()
        pipeline = _pipeline(ScriptedStreamingASR({"partials": PARTIALS}))

        result = asyncio.run(
            run_turn(session, pipeline, TurnRequest(frames=_frames(6), speak=False))
        )

        partials = _events(session, "asr.partial")
        finals = _events(session, "asr.final")
        assert len(partials) == len(PARTIALS), [item["data"] for item in partials]
        assert finals, "a streaming turn must always terminate with asr.final"
        order = [item["type"] for item in session.bus.after(0)]
        assert order.index("asr.partial") < order.index("asr.final")
        # The last partial and the final must agree, or the client is being
        # shown a transcript that gets rewritten at the very end.
        assert partials[-1]["data"]["text"] == FINAL
        assert result.transcript == FINAL

    def test_timeline_records_time_to_first_partial(self) -> None:
        session = Session()
        pipeline = _pipeline(ScriptedStreamingASR({"partials": PARTIALS}))

        result = asyncio.run(
            run_turn(session, pipeline, TurnRequest(frames=_frames(6), speak=False))
        )

        assert "asr_first_partial" in result.timeline.marks
        assert result.timeline.asr_ttfp_ms is not None
        assert result.timeline.asr_ttfp_ms >= 0

    def test_committed_prefix_grows_and_never_exceeds_the_text(self) -> None:
        session = Session()
        pipeline = _pipeline(ScriptedStreamingASR({"partials": PARTIALS}))

        asyncio.run(run_turn(session, pipeline, TurnRequest(frames=_frames(6), speak=False)))

        committed: list[str] = []
        for event in _events(session, "asr.partial"):
            data = event["data"]
            assert data["text"].startswith(data["committed"]), data
            assert data["committed"] + data["unstable"] == data["text"], data
            committed.append(data["committed"])
        # The first hypothesis cannot be committed: there is nothing to compare
        # it against yet, so claiming stability would be inventing evidence.
        assert committed[0] == ""
        assert committed[-1] == "今天天气"

    def test_vad_end_is_marked_when_capture_ends_not_when_frames_run_out(self) -> None:
        session = Session()
        pipeline = _pipeline(ScriptedStreamingASR({"partials": PARTIALS}))
        stream = session.open_input_stream(SAMPLE_RATE)

        async def scenario():
            task = asyncio.create_task(
                run_turn(session, pipeline, TurnRequest(frames=stream.frames(), speak=False))
            )
            await asyncio.sleep(0)
            for _ in range(4):
                await stream.put_pcm(_pcm(), SAMPLE_RATE)
            await session.end_input_stream()
            return await task

        result = asyncio.run(scenario())

        assert "vad_end" in result.timeline.marks
        # Marked at capture close, so it must precede the end of recognition --
        # otherwise asr_latency_ms would measure queue drain, not final decode.
        assert result.timeline.marks["vad_end"] <= result.timeline.marks["asr_end"]
        assert result.transcript == FINAL

    def test_cancelled_stream_still_terminates_with_asr_final(self) -> None:
        session = Session()
        pipeline = _pipeline(ScriptedStreamingASR({"partials": PARTIALS}))

        async def scenario():
            task = asyncio.create_task(
                run_turn(session, pipeline, TurnRequest(frames=_frames(50, delay=0.001), speak=False))
            )
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            session.cancel(detail="test barge-in")
            return await task

        result = asyncio.run(scenario())

        assert result.cancelled is True
        assert _events(session, "asr.final"), "a cancelled turn must still close its transcript"


class TestOneShotAudioSharesTheStreamingPath:
    def test_finished_buffer_still_emits_partials_when_supported(self) -> None:
        session = Session()
        pipeline = _pipeline(ScriptedStreamingASR({"partials": PARTIALS}))
        audio = AudioChunk(pcm=_pcm(milliseconds=1000), sample_rate=SAMPLE_RATE)

        result = asyncio.run(run_turn(session, pipeline, TurnRequest(audio=audio, speak=False)))

        assert len(_events(session, "asr.partial")) == len(PARTIALS)
        assert result.transcript == FINAL
        # A finished utterance has no capture to close, so speech "ended" before
        # recognition started rather than during it.
        assert "vad_end" in result.timeline.marks

    def test_provider_without_partials_emits_none(self) -> None:
        session = Session()
        pipeline = _pipeline(FakeASR({}))
        audio = AudioChunk(pcm=_pcm(milliseconds=1000), sample_rate=SAMPLE_RATE)

        result = asyncio.run(run_turn(session, pipeline, TurnRequest(audio=audio, speak=False)))

        assert _events(session, "asr.partial") == [], "partials must never be manufactured"
        assert result.transcript, result
        assert _events(session, "asr.final")


class TestIterFrames:
    def test_splits_a_buffer_into_twenty_millisecond_frames(self) -> None:
        audio = AudioChunk(pcm=_pcm(milliseconds=100), sample_rate=SAMPLE_RATE)

        frames = list(iter_frames(audio))

        assert len(frames) == 5
        assert [frame.sequence for frame in frames] == [0, 1, 2, 3, 4]
        assert all(frame.sample_rate == SAMPLE_RATE for frame in frames)
        assert frames[-1].is_final is True
        assert all(not frame.is_final for frame in frames[:-1])

    def test_bytes_are_preserved_in_order(self) -> None:
        audio = AudioChunk(pcm=_pcm(milliseconds=60), sample_rate=SAMPLE_RATE)

        assert b"".join(frame.pcm for frame in iter_frames(audio)) == audio.pcm

    def test_rejects_a_non_positive_frame_length(self) -> None:
        audio = AudioChunk(pcm=_pcm(milliseconds=20), sample_rate=SAMPLE_RATE)

        with pytest.raises(ValueError, match="frame_ms"):
            list(iter_frames(audio, frame_ms=0))

    def test_empty_audio_yields_nothing(self) -> None:
        assert list(iter_frames(AudioChunk(pcm=b"", sample_rate=SAMPLE_RATE))) == []
