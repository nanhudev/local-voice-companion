"""Real streaming ASR: partial hypotheses have to actually arrive.

Everything in ``tests/unit/test_streaming_core.py`` is true by construction,
because the recogniser behind it is a stub that echoes byte counts. Those tests
pin the *plumbing*. These tests pin the claim the plumbing exists for: that a
real engine, fed a real wav one frame at a time, produces intermediate text
before the utterance is over.

They are ``hardware`` marked and skip loudly -- with the command that would fix
it -- when sherpa-onnx or its weights are absent. A skip here is an honest
statement about the machine. Inventing a passing result on a machine with no
engine would be the one thing these tests must never do.
"""

from __future__ import annotations

import asyncio
import time
import wave
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

pytestmark = pytest.mark.hardware

from local_voice_companion.core.audio import AudioFrame  # noqa: E402
from local_voice_companion.core.transcriber import TranscriptUpdate  # noqa: E402

FRAME_MS = 20


def _provider_class():
    from local_voice_companion.providers.local.runtime_probe import module_present
    from local_voice_companion.providers.local.sherpa_streaming_asr import (
        REQUIRED_FILES,
        SherpaStreamingASR,
        model_root,
    )

    if not module_present("sherpa_onnx"):
        pytest.skip("sherpa-onnx is not installed: pip install sherpa-onnx sherpa-onnx-core")
    root = model_root()
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        pytest.skip(
            f"streaming weights missing under {root}: {', '.join(missing)} "
            f"(lvc models fetch streaming-zipformer-multi-zh-hans-int8)"
        )
    return SherpaStreamingASR


def _wav() -> Path:
    from local_voice_companion.providers.local.sherpa_streaming_asr import model_root

    wavs = sorted((model_root() / "test_wavs").glob("*.wav"))
    if not wavs:
        pytest.skip("no test wav shipped with the streaming model")
    return wavs[0]


def _read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            pytest.skip(f"{path.name} is not 16-bit PCM")
        return handle.readframes(handle.getnframes()), handle.getframerate()


def _frames(pcm: bytes, rate: int) -> list[bytes]:
    size = rate * 2 * FRAME_MS // 1000
    return [pcm[i : i + size] for i in range(0, len(pcm), size)]


async def _stream(provider: Any, pcm: bytes, rate: int) -> tuple[list[TranscriptUpdate], float]:
    """Push the whole file as fast as the consumer takes it, timing each update."""

    async def source() -> AsyncIterator[AudioFrame]:
        for index, chunk in enumerate(_frames(pcm, rate)):
            yield AudioFrame(
                pcm=chunk,
                sample_rate=rate,
                sequence=index,
                captured_at=time.perf_counter(),
            )

    started = time.perf_counter()
    updates: list[TranscriptUpdate] = []
    first_ms = 0.0
    async for update in provider.stream_transcribe(source()):
        if not updates:
            first_ms = (time.perf_counter() - started) * 1000
        updates.append(update)
    return updates, first_ms


class TestSherpaStreamingASR:
    def test_partials_arrive_before_the_utterance_ends(self) -> None:
        """The whole point of 3A: text before the speaker stops talking."""

        cls = _provider_class()
        pcm, rate = _read_wav(_wav())

        async def run() -> tuple[list[TranscriptUpdate], float]:
            provider = cls({"threads": 4})
            await provider.load()
            try:
                return await _stream(provider, pcm, rate)
            finally:
                await provider.unload()

        updates, first_at = asyncio.run(run())
        partials = [u for u in updates if not u.is_final]

        assert partials, (
            "no partial was emitted at all -- the provider is buffering and "
            "behaving like a turn-based engine with extra steps"
        )
        assert updates[-1].is_final, "the stream must end with a final update"
        assert first_at > 0, "the first update must be timestamped"

    def test_the_streaming_final_matches_the_whole_utterance_result(self) -> None:
        """Streaming must not be a quiet quality downgrade.

        If the incremental decoder and the whole-utterance decoder disagree, the
        runtime is quietly giving the user a worse transcript in exchange for
        latency, and nothing in the metrics would reveal it.
        """

        cls = _provider_class()
        pcm, rate = _read_wav(_wav())

        async def run() -> tuple[str, str]:
            from local_voice_companion.core.types import AudioChunk

            provider = cls({"threads": 4})
            await provider.load()
            try:
                updates, _ = await _stream(provider, pcm, rate)
                whole = await provider.transcribe(
                    AudioChunk(pcm=pcm, sample_rate=rate, channels=1, sample_width=2)
                )
                return updates[-1].text, whole
            finally:
                await provider.unload()

        streamed, whole = asyncio.run(run())
        assert streamed == whole, f"streaming final {streamed!r} != whole-utterance {whole!r}"

    def test_committed_text_is_never_ahead_of_the_hypothesis(self) -> None:
        """Real recognisers retract. The tracker must retract with them.

        With a stub this is trivially true. A transducer decoder regularly
        revises the tail of its output, so this is where a naive "committed
        prefix only ever grows" implementation would be caught lying.
        """

        cls = _provider_class()
        pcm, rate = _read_wav(_wav())

        async def run() -> list[TranscriptUpdate]:
            provider = cls({"threads": 4})
            await provider.load()
            try:
                updates, _ = await _stream(provider, pcm, rate)
                return updates
            finally:
                await provider.unload()

        for update in asyncio.run(run()):
            assert update.text.startswith(update.committed), (
                f"committed {update.committed!r} is not a prefix of {update.text!r}"
            )
            assert update.committed + update.unstable == update.text, update

    def test_time_to_first_partial_beats_whole_utterance_latency(self) -> None:
        """The measured claim, not a slogan.

        Both numbers come from the same weights on the same audio, so this
        isolates architecture: the only difference is whether the caller gets
        text while the audio is still arriving.
        """

        cls = _provider_class()
        pcm, rate = _read_wav(_wav())

        async def run() -> tuple[float, float]:
            from local_voice_companion.core.types import AudioChunk

            provider = cls({"threads": 4})
            await provider.load()
            try:
                updates, ttfp_ms = await _stream(provider, pcm, rate)
                started = time.perf_counter()
                await provider.transcribe(
                    AudioChunk(pcm=pcm, sample_rate=rate, channels=1, sample_width=2)
                )
                whole_ms = (time.perf_counter() - started) * 1000
                return ttfp_ms, whole_ms, len(updates)
            finally:
                await provider.unload()

        ttfp_ms, whole_ms, count = asyncio.run(run())
        assert whole_ms > 0, "the whole-utterance path must actually run"
        assert ttfp_ms > 0, "no update was emitted at all"
        assert ttfp_ms < whole_ms, (
            f"first partial at {ttfp_ms:.1f} ms is not earlier than the "
            f"whole-utterance result at {whole_ms:.1f} ms ({count} updates)"
        )
