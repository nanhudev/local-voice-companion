"""Streaming ASR backed by sherpa-onnx (Zipformer transducer, int8, CPU).

Why this provider exists at all comes down to one number the previous phase could
not improve: PHASE 2's fastest turn-based ASR returned its only result in
**573 ms**, and no amount of orchestration can emit a partial before the engine
finishes the utterance. A recogniser that consumes frames while they arrive is
the only way to get a hypothesis to the user *during* speech.

Two properties are deliberately built in, because they are what make the
benchmark comparison meaningful rather than merely possible:

1. **The same weights serve both paths.** :meth:`transcribe` runs the identical
   recogniser over a complete utterance and :meth:`stream_transcribe` runs it over
   frames. Comparing faster-whisper against this provider is therefore a
   comparison of *architectures*, not of model quality -- two variables would make
   the result unattributable.
2. **sherpa-onnx carries its own ONNX Runtime.** The wheel statically links ORT
   into ``_sherpa_onnx.pyd``, so importing it cannot clash with the
   ``onnxruntime`` distribution Kokoro loads. Verified on this machine, not
   assumed.

It does not replace faster-whisper. faster-whisper stays the higher-quality
final transcript; this is the engine you listen *with*.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Mapping

from ...core.cancellation import CancellationToken
from ...core.errors import ProviderUnavailable
from ...core.lifecycle import ProviderState
from ...core.types import (
    AudioChunk,
    Device,
    ProviderKind,
    QualitySource,
)
from ..base import ASRProvider, ModelRef, ProviderDescriptor, ProviderHealth
from . import runtime_probe

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from ...core.audio import AudioFrame
    from ...core.transcriber import TranscriptUpdate

MODEL_ID = "streaming-zipformer-multi-zh-hans-int8"
MODEL_DIRNAME = MODEL_ID
NATIVE_SAMPLE_RATE = 16000

REQUIRED_FILES: tuple[str, ...] = (
    "encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx",
    "decoder-epoch-20-avg-1-chunk-16-left-128.onnx",
    "joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx",
    "tokens.txt",
)


def _option_int(options: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(options.get(key, default))
    except (TypeError, ValueError):
        return default


def model_root() -> Any:
    from pathlib import Path

    override = os.getenv("LVC_MODELS_DIR", "").strip()
    if override:
        return Path(override) / "sherpa" / MODEL_DIRNAME
    from ...config.paths import DEFAULT_LAYOUT

    return DEFAULT_LAYOUT.models_dir / "sherpa" / MODEL_DIRNAME


def _float32_samples(pcm: bytes) -> list[float]:
    """int16 little-endian PCM -> float32 in [-1, 1]. No numpy dependency."""

    count = len(pcm) // 2
    return [value / 32768.0 for value in memoryview(pcm).cast("h")[:count]]


def _resample(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    """Linear resampling to the rate the recogniser was built for.

    Linear interpolation is not a good resampler, and it is used anyway because
    it is dependency-free and the honest alternative -- refusing non-16 kHz audio
    -- would make every Kokoro-sourced 24 kHz buffer a hard error in a test
    fixture. The failure mode of a linear resampler (slightly soft high
    frequencies) does not silently produce wrong transcripts.
    """

    if source_rate == target_rate or not samples:
        return samples
    ratio = target_rate / source_rate
    length = int(len(samples) * ratio)
    out: list[float] = []
    for index in range(length):
        position = index / ratio
        left = int(position)
        if left + 1 >= len(samples):
            out.append(samples[-1])
            continue
        fraction = position - left
        out.append(samples[left] * (1.0 - fraction) + samples[left + 1] * fraction)
    return out


class SherpaStreamingASR(ASRProvider):
    """Frame-by-frame speech recognition with partial hypotheses."""

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        super().__init__(options)
        self._recognizer: Any = None

    # -- static description -------------------------------------------------

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        return ProviderDescriptor(
            id="sherpa_streaming_asr_cpu",
            kind=ProviderKind.ASR,
            display_name="sherpa-onnx Streaming Zipformer (int8, CPU)",
            languages=("zh", "en"),
            devices=(Device.CPU,),
            streaming=True,
            supports_streaming=True,
            supports_partial_results=True,
            estimated_ram_mb=260,
            estimated_vram_mb=0,
            estimated_disk_mb=74,
            requires_network=False,
            is_local=True,
            quality_tier=2,
            latency_tier=5,
            quality_score=None,
            # No measured WER exists for this machine. Claiming one would be
            # exactly the fabrication the quality_source field exists to prevent.
            quality_source=QualitySource.UNKNOWN,
            supports_cancellation=True,
            version=runtime_probe.package_version("sherpa-onnx") or "none",
            tags=("streaming", "onnx", "cpu", "partials"),
            models=(
                ModelRef(
                    id=MODEL_ID,
                    display_name="Streaming Zipformer multi zh-Hans (int8)",
                    languages=("zh", "en"),
                    quantization="int8",
                    disk_mb=74,
                    context_window=0,
                    requires_network=False,
                ),
            ),
        )

    # -- lifecycle ----------------------------------------------------------

    async def probe(self) -> ProviderHealth:
        if not runtime_probe.module_present("sherpa_onnx"):
            return ProviderHealth(
                ok=False,
                detail="sherpa_onnx is not installed: pip install sherpa-onnx sherpa-onnx-core",
            )
        root = model_root()
        missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
        if missing:
            return ProviderHealth(
                ok=False,
                detail=(
                    f"model {MODEL_ID} missing {len(missing)} file(s) under {root}: "
                    f"{', '.join(missing)}"
                ),
            )
        try:
            self.lifecycle.transition(ProviderState.AVAILABLE)
        except Exception:  # pragma: no cover - transition already validated elsewhere
            pass
        return ProviderHealth(ok=True, detail=f"{MODEL_ID} present", extra={"path": str(root)})

    async def load(self, model: str | None = None, device: str = Device.CPU.value) -> None:
        await self._ensure_probed()
        if self._recognizer is not None:
            return
        self.lifecycle.transition(ProviderState.LOADING)
        started = time.perf_counter()
        root = model_root()
        threads = _option_int(self.options, "threads", 4)
        try:
            self._recognizer = await asyncio.to_thread(self._build, str(root), threads)
        except Exception as exc:
            self.lifecycle.transition(ProviderState.ERROR, str(exc))
            raise ProviderUnavailable(
                f"could not build the streaming recogniser: {exc}",
                provider=self.descriptor().id,
            ) from exc
        self.lifecycle.transition(ProviderState.READY)
        self.lifecycle.detail = f"{MODEL_ID} loaded in {int((time.perf_counter() - started) * 1000)} ms"

    @staticmethod
    def _build(root: str, threads: int) -> Any:
        from pathlib import Path

        import sherpa_onnx

        base = Path(root)
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(base / REQUIRED_FILES[0]),
            decoder=str(base / REQUIRED_FILES[1]),
            joiner=str(base / REQUIRED_FILES[2]),
            tokens=str(base / REQUIRED_FILES[3]),
            num_threads=threads,
            provider="cpu",
            decoding_method="greedy_search",
            sample_rate=NATIVE_SAMPLE_RATE,
            feature_dim=80,
        )

    async def unload(self) -> None:
        self.lifecycle.transition(ProviderState.UNLOADING)
        self._recognizer = None
        self.lifecycle.transition(ProviderState.AVAILABLE)

    # -- recognition --------------------------------------------------------

    def _require(self) -> Any:
        if self._recognizer is None:
            raise ProviderUnavailable(
                "sherpa streaming ASR is not loaded", provider=self.descriptor().id
            )
        return self._recognizer

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> str:
        """Whole-utterance recognition over the *same* weights as the stream.

        This is the apples-to-apples counterpart to faster-whisper in the
        benchmark; it exists so the comparison has one variable, not two.
        """

        recognizer = self._require()
        samples = _resample(
            _float32_samples(audio.pcm), audio.sample_rate, NATIVE_SAMPLE_RATE
        )
        return await asyncio.to_thread(self._run_complete, recognizer, samples)

    @staticmethod
    def _run_complete(recognizer: Any, samples: list[float]) -> str:
        stream = recognizer.create_stream()
        stream.accept_waveform(NATIVE_SAMPLE_RATE, samples)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        result = recognizer.get_result(stream)
        return result if isinstance(result, str) else getattr(result, "text", "")

    async def stream_transcribe(
        self,
        frames: AsyncIterator["AudioFrame"],
        *,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> AsyncIterator["TranscriptUpdate"]:
        """Consume frames as they arrive, yielding stable partially-decoded text."""

        from ...core.transcriber import TranscriptStabilityTracker

        recognizer = self._require()
        tracker = TranscriptStabilityTracker()
        stream = recognizer.create_stream()
        decoder = _StreamDecoder(recognizer, stream)

        async for frame in frames:
            if token is not None and token.cancelled:
                return
            samples = _resample(
                _float32_samples(frame.pcm), frame.sample_rate, NATIVE_SAMPLE_RATE
            )
            text = await asyncio.to_thread(decoder.accept, samples)
            update = tracker.push(text, degraded=False)
            if update is not None:
                yield update

        text = await asyncio.to_thread(decoder.finish)
        final = tracker.finalize(text)
        if final.text:
            yield final

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<SherpaStreamingASR loaded={self._recognizer is not None}>"


class _StreamDecoder:
    """Thin stateful wrapper over one sherpa ``OnlineStream``.

    Kept separate from the provider because it is inherently synchronous and
    stateful, while every public surface above is async: mixing the two in one
    class is how a streaming provider acquires a race condition.
    """

    def __init__(self, recognizer: Any, stream: Any) -> None:
        self._recognizer = recognizer
        self._stream = stream

    def accept(self, samples: list[float]) -> str:
        self._stream.accept_waveform(NATIVE_SAMPLE_RATE, samples)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        result = self._recognizer.get_result(self._stream)
        return result if isinstance(result, str) else getattr(result, "text", "")

    def finish(self) -> str:
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        result = self._recognizer.get_result(self._stream)
        return result if isinstance(result, str) else getattr(result, "text", "")
