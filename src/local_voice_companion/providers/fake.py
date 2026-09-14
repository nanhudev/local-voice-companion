"""Deterministic reference providers used by CI and by the Phase 1 pipeline.

These are first-class citizens, not toys: every integration test exercises the
real orchestrator, real cancellation, real bounded queues and the real event
bus through these implementations. Nothing here may download anything.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from io import BytesIO
from typing import Any, AsyncIterator, Sequence

from ..core.cancellation import CancellationToken
from ..core.types import (
    AudioChunk,
    AudioFormat,
    ChatMessage,
    Device,
    ProviderKind,
    QualitySource,
)
from .base import (
    ASRProvider,
    LLMProvider,
    ModelRef,
    ProviderDescriptor,
    TTSProvider,
    VADProvider,
    VoiceRef,
)

FAKE_MODEL_ID = "fake-1"
DEFAULT_FAKE_LATIN = ["你", "好", "，", "我", "是", "本", "地", "语", "音", "运", "行", "时", "。"]

_LANGUAGES = ("zh", "en", "ja")


def _option_int(provider: Any, key: str, default: int) -> int:
    raw = provider.options.get(key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _option_float(provider: Any, key: str, default: float) -> float:
    raw = provider.options.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


class FakeASR(ASRProvider):
    """Deterministic transcription.

    Options:
        scripted_transcript: str -> exact text returned for any audio
        delay_ms: int        -> simulated processing time per call
        streaming_delay_ms   -> per-partial delay
        fail: bool           -> force ProviderUnavailable behaviour
    """

    kind = ProviderKind.ASR

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        return ProviderDescriptor(
            id="fake_asr",
            kind=ProviderKind.ASR,
            display_name="Fake ASR (deterministic test provider)",
            languages=_LANGUAGES,
            devices=(Device.CPU,),
            streaming=False,
            estimated_ram_mb=1,
            estimated_vram_mb=0,
            quality_tier=1,
            latency_tier=5,
            quality_score=None,
            quality_source=QualitySource.UNKNOWN,
            version="1.0.0",
            tags=("fake", "test", "ci", "no-download"),
            models=(ModelRef(id=FAKE_MODEL_ID, display_name="Fake ASR Model", languages=_LANGUAGES),),
        )

    async def load(self, model: str | None = None, device: str = Device.CPU.value) -> None:
        return await super().load(model, device)

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> str:
        if self.options.get("fail"):
            from ..core.errors import ProviderUnavailable

            raise ProviderUnavailable("fake_asr configured to fail", provider="fake_asr")

        delay_ms = _option_int(self, "delay_ms", 5)
        if token is not None:
            await token.sleep(delay_ms / 1000.0)
            token.raise_if_cancelled()

        scripted = self.options.get("scripted_transcript")
        if isinstance(scripted, str) and scripted.strip():
            return scripted.strip()

        duration = audio.duration_ms
        if duration <= 0:
            return ""
        return f"你好，这是 {int(duration)} 毫秒的测试语音。"


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class FakeLLM(LLMProvider):
    """Character-by-character streaming with a deterministic answer.

    Options:
        response: str   -> the full reply to stream
        delay_ms: int   -> delay per delta
        ttft_ms: int    -> delay before the first delta (models TTFT)
        fail: bool
    """

    kind = ProviderKind.LLM

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        return ProviderDescriptor(
            id="fake_llm",
            kind=ProviderKind.LLM,
            display_name="Fake LLM (deterministic test provider)",
            languages=_LANGUAGES,
            devices=(Device.CPU,),
            streaming=True,
            estimated_ram_mb=1,
            estimated_vram_mb=0,
            quality_tier=1,
            latency_tier=5,
            quality_source=QualitySource.UNKNOWN,
            version="1.0.0",
            tags=("fake", "test", "ci", "no-download"),
            models=(ModelRef(id=FAKE_MODEL_ID, display_name="Fake LLM", languages=_LANGUAGES, context_window=2048),),
        )

    def _reply(self, messages: Sequence[ChatMessage]) -> str:
        scripted = self.options.get("response")
        if isinstance(scripted, str) and scripted.strip():
            return scripted.strip()
        last_user = next(
            (message.content for message in reversed(messages) if message.role == "user"), ""
        )
        return f"我听到了：{last_user}。这是一个用于验证运行时链路的本地回复。"

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        token: CancellationToken | None = None,
    ) -> AsyncIterator[str]:
        if self.options.get("fail"):
            from ..core.errors import ProviderUnavailable

            raise ProviderUnavailable("fake_llm configured to fail", provider="fake_llm")

        if token is not None:
            await token.sleep(_option_int(self, "ttft_ms", 2) / 1000.0)
            token.raise_if_cancelled()

        reply = self._reply(messages)
        per_token = _option_int(self, "delay_ms", 1)
        for index, char in enumerate(reply):
            if max_tokens is not None and index >= max_tokens:
                break
            if token is not None:
                token.raise_if_cancelled()
                if per_token:
                    await token.sleep(per_token / 1000.0)
            yield char


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class FakeTTS(TTSProvider):
    """Generates a short PCM sine tone so playback length is measurable.

    Options:
        sample_rate: int  (default 24000)
        ms_per_char: int  (default 12)  -> deterministic duration
        tone_hz: float    (default 220.0)
        fail: bool
    """

    kind = ProviderKind.TTS

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        return ProviderDescriptor(
            id="fake_tts",
            kind=ProviderKind.TTS,
            display_name="Fake TTS (deterministic test provider)",
            languages=_LANGUAGES,
            devices=(Device.CPU,),
            streaming=True,
            estimated_ram_mb=1,
            estimated_vram_mb=0,
            quality_tier=1,
            latency_tier=5,
            quality_source=QualitySource.UNKNOWN,
            version="1.0.0",
            tags=("fake", "test", "ci", "no-download"),
            voices=(
                VoiceRef(id="test-voice", display_name="Test Voice", languages=_LANGUAGES),
                VoiceRef(id="vivian", display_name="Vivian", languages=("zh", "en")),
            ),
        )

    def _wav(self, duration_ms: float, tone_hz: float) -> bytes:
        sample_rate = _option_int(self, "sample_rate", 24000)
        frames = max(1, int(sample_rate * duration_ms / 1000.0))
        buffer = BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            frames_data = b"".join(
                struct.pack(
                    "<h",
                    int(12000 * math.sin(2 * math.pi * tone_hz * (i / sample_rate))),
                )
                for i in range(frames)
            )
            handle.writeframes(frames_data)
        return buffer.getvalue()

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> tuple[bytes, AudioFormat]:
        if self.options.get("fail"):
            from ..core.errors import ProviderUnavailable

            raise ProviderUnavailable("fake_tts configured to fail", provider="fake_tts")

        duration = max(60.0, len(text) * _option_int(self, "ms_per_char", 12))
        if token is not None:
            token.raise_if_cancelled()
        audio = self._wav(duration, _option_float(self, "tone_hz", 220.0))
        return audio, AudioFormat(mime_type="audio/wav", sample_rate=_option_int(self, "sample_rate", 24000))

    async def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> AsyncIterator[bytes]:
        blob, _fmt = await self.synthesize(
            text, voice=voice, language=language, model=model, token=token
        )
        yield blob


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


class FakeVAD(VADProvider):
    """Energy-threshold VAD used by tests and as a CPU fallback.

    Options:
        threshold_rms: int (default 300)
    """

    kind = ProviderKind.VAD

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        return ProviderDescriptor(
            id="fake_vad",
            kind=ProviderKind.VAD,
            display_name="Energy-threshold VAD (deterministic)",
            languages=(),
            devices=(Device.CPU,),
            streaming=True,
            estimated_ram_mb=1,
            quality_tier=2,
            latency_tier=5,
            version="1.0.0",
            tags=("fake", "energy", "no-download"),
        )

    def is_speech(self, frame: AudioChunk) -> bool:
        return pcm_rms(frame.pcm, frame.sample_width) >= _option_int(self, "threshold_rms", 300)


def pcm_rms(pcm: bytes, sample_width: int = 2) -> int:
    """RMS of little-endian signed PCM. Pure python so it works anywhere."""

    if not pcm:
        return 0
    try:
        import audioop  # type: ignore[attr-defined]

        return audioop.rms(pcm, sample_width)
    except Exception:  # noqa: BLE001 - audioop removed in 3.13
        pass
    count = len(pcm) // sample_width
    if count <= 0:
        return 0
    total = 0.0
    for index in range(0, count * sample_width, sample_width):
        value = int.from_bytes(pcm[index : index + sample_width], "little", signed=True)
        total += value * value
    return int(math.sqrt(total / count))


FAKE_PROVIDERS = (FakeASR, FakeLLM, FakeTTS, FakeVAD)


def dump_plan(plan: Any) -> str:  # pragma: no cover - helper for manual debugging
    return json.dumps(plan, ensure_ascii=False, indent=2, default=str)
