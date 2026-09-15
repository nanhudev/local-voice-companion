"""Unit tests for the streaming foundation.

These exercise logic only: no microphone, no recogniser, no model. Anything that
needs a real backend is intentionally absent here -- see RULE 12, which is why
there is not a single "sherpa is installed" assertion in this file.
"""

from __future__ import annotations

import asyncio

import pytest

from local_voice_companion.core.audio import (
    AudioFrame,
    InputAudioStream,
    concat_frames,
)
from local_voice_companion.core.transcriber import (
    TranscriptStabilityTracker,
    common_prefix,
)
from local_voice_companion.providers.base import ASRProvider, ProviderDescriptor
from local_voice_companion.core.types import AudioChunk, ProviderKind


def _pcm(frames: int = 320, value: int = 1000) -> bytes:
    import struct

    return struct.pack(f"<{frames}h", *([value] * frames))


class _EchoASR(ASRProvider):
    """Minimal turn-based provider, used to exercise the default contract."""

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        return ProviderDescriptor(
            id="echo-asr",
            kind=ProviderKind.ASR,
            display_name="Echo",
            languages=("zh",),
            streaming=False,
        )

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str = "",
        model: str | None = None,
        token=None,
    ) -> str:
        return f"bytes={len(audio.pcm)}"


# ---------------------------------------------------------------------------
# AudioFrame
# ---------------------------------------------------------------------------


class TestAudioFrame:
    def test_duration_is_derived_from_pcm_not_declared(self) -> None:
        frame = AudioFrame(pcm=_pcm(320), sample_rate=16000)
        # 320 samples at 16 kHz is 20 ms -- the standard frame this project uses.
        assert frame.duration_ms == pytest.approx(20.0)

    def test_age_ms_grows_from_capture_time(self) -> None:
        frame = AudioFrame(pcm=_pcm(10), sample_rate=16000, captured_at=0.0)
        assert frame.age_ms(now=0.250) == pytest.approx(250.0)

    def test_to_chunk_drops_the_streaming_fields(self) -> None:
        """The downgrade must be visible: sequence and capture time do not survive."""

        frame = AudioFrame(pcm=_pcm(4), sample_rate=16000, sequence=7, captured_at=1.5)
        chunk = frame.to_chunk()
        assert chunk.pcm == frame.pcm
        assert chunk.sample_rate == 16000
        assert not hasattr(chunk, "sequence")


class TestConcatFrames:
    def test_preserves_order_and_round_trips_to_bytes(self) -> None:
        first = AudioFrame(pcm=b"aa", sample_rate=16000, sequence=0)
        second = AudioFrame(pcm=b"bb", sample_rate=16000, sequence=1)
        joined = concat_frames([first, second])
        assert joined.pcm == b"aabb"
        assert joined.sample_rate == 16000
        assert joined.is_final is True

    def test_mixed_sample_rates_are_refused_not_resampled(self) -> None:
        """Silently resampling here would hide a real pipeline bug."""

        frames = [
            AudioFrame(pcm=b"aa", sample_rate=16000),
            AudioFrame(pcm=b"bb", sample_rate=48000),
        ]
        with pytest.raises(ValueError, match="mixed sample rates"):
            concat_frames(frames)


# ---------------------------------------------------------------------------
# InputAudioStream
# ---------------------------------------------------------------------------


class TestInputAudioStream:
    def test_frames_are_stamped_with_monotonic_sequence(self) -> None:
        async def run() -> list[int]:
            stream = InputAudioStream(capacity=8)
            seen = []
            for _ in range(3):
                await stream.put_pcm(_pcm(4), 16000)
                seen.append(stream.sequence)
            return seen

        assert asyncio.run(run()) == [1, 2, 3]

    def test_iteration_stops_on_close(self) -> None:
        async def run() -> int:
            stream = InputAudioStream(capacity=8)
            for _ in range(3):
                await stream.put_pcm(_pcm(4), 16000)
            await stream.close()
            count = 0
            async for _frame in stream.frames():
                count += 1
            return count

        assert asyncio.run(run()) == 3

    def test_full_buffer_drops_oldest_under_backpressure(self) -> None:
        """A live mic must never grow memory without bound."""

        async def run() -> tuple[int, int, int]:
            stream = InputAudioStream(capacity=2, policy="oldest")
            for _ in range(6):
                await stream.put_pcm(_pcm(4), 16000)
            await stream.close()
            sequences = [frame.sequence async for frame in stream.frames()]
            return min(sequences), max(sequences), stream.stats().dropped

        oldest, newest, dropped = asyncio.run(run())
        # Keeps the newest two frames; everything before them is gone, and the
        # loss is recorded rather than swallowed.
        assert newest == 5
        assert oldest == 4
        assert dropped > 0

    def test_a_missing_frame_is_visible_as_a_gap(self) -> None:
        async def run() -> int:
            stream = InputAudioStream(capacity=8)
            # Jump 0 -> 5: four frames vanished somewhere upstream.
            await stream.put(AudioFrame(pcm=_pcm(4), sample_rate=16000, sequence=5))
            await stream.close()
            async for _frame in stream.frames():
                pass
            return stream.stats().gaps

        assert asyncio.run(run()) == 1

    def test_drain_reports_what_was_lost(self) -> None:
        async def run() -> int:
            stream = InputAudioStream(capacity=8)
            for _ in range(4):
                await stream.put_pcm(_pcm(4), 16000)
            return stream.drain()

        assert asyncio.run(run()) == 4


# ---------------------------------------------------------------------------
# Transcript stability
# ---------------------------------------------------------------------------


class TestTranscriptStability:
    def test_common_prefix_helper(self) -> None:
        assert common_prefix("abcdef", "abcxyz") == "abc"
        assert common_prefix("abc", "xyz") == ""
        assert common_prefix("", "abc") == ""

    def test_the_first_result_commits_nothing(self) -> None:
        """With one observation nothing has stopped changing yet."""

        tracker = TranscriptStabilityTracker(started_at=0.0)
        update = tracker.push("你好")
        assert update is not None
        assert update.committed == ""
        assert update.unstable == "你好"
        assert update.stability == 0.0

    def test_agreement_between_consecutive_results_commits_the_prefix(self) -> None:
        tracker = TranscriptStabilityTracker(started_at=0.0)
        tracker.push("你好")
        second = tracker.push("你好吗")
        assert second is not None
        assert second.committed == "你好"
        assert second.unstable == "吗"

    def test_identical_results_are_suppressed_not_forwarded(self) -> None:
        """Engines repeat themselves while waiting for audio; floods are not events."""

        tracker = TranscriptStabilityTracker(started_at=0.0)
        assert tracker.push("你好") is not None
        assert tracker.push("你好") is None
        assert tracker.push("你好") is None
        assert tracker.suppressed == 2
        assert tracker.emitted == 1

    def test_a_retraction_clamps_committed_back_down(self) -> None:
        """Keeping a committed prefix the recogniser withdrew would be a lie."""

        tracker = TranscriptStabilityTracker(started_at=0.0)
        tracker.push("今天天气很好")
        tracker.push("今天天气不错")
        assert tracker.committed == "今天天气"
        tracker.push("今天我")
        assert tracker.committed == "今天"

    def test_finalize_settles_everything_and_marks_it_final(self) -> None:
        tracker = TranscriptStabilityTracker(started_at=0.0)
        tracker.push("你好")
        final = tracker.finalize()
        assert final.is_final is True
        assert final.committed == "你好"
        assert final.unstable == ""
        assert final.stability == 1.0

    def test_since_change_ms_supports_pause_detection(self) -> None:
        """Pause detection needs a controllable "how long has it been still".

        The clock is injected, not wall-clock: with a real clock this assertion
        could only pass by accident. It previously mixed an injected `started_at`
        of 0.0 with `perf_counter()` samples, producing ~-7.4e6 ms.
        """

        now = 0.0
        tracker = TranscriptStabilityTracker(started_at=0.0, clock=lambda: now)
        tracker.push("你好")  # the hypothesis exists from t=0
        now = 1.5
        assert tracker.since_change_ms() == pytest.approx(1500.0)

        # A change resets the stillness clock, which is what makes this a pause
        # signal instead of a stopwatch started at the beginning of the utterance.
        now = 2.0
        tracker.push("你好吗")
        now = 2.4
        assert tracker.since_change_ms() == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# The ASR contract default
# ---------------------------------------------------------------------------


class TestASRStreamContract:
    def test_a_turn_based_provider_yields_exactly_one_final(self) -> None:
        """No partials are manufactured from a whole-utterance result."""

        async def run() -> list[dict]:
            provider = _EchoASR()
            stream = InputAudioStream(capacity=8)
            for _ in range(3):
                await stream.put_pcm(_pcm(320), 16000)
            await stream.close()
            updates = []
            async for update in provider.stream_transcribe(stream.frames()):
                updates.append(update.to_dict())
            return updates

        updates = asyncio.run(run())
        assert len(updates) == 1, updates
        assert updates[0]["is_final"] is True
        # 3 frames * 320 samples * 2 bytes
        assert updates[0]["text"] == "bytes=1920"
        assert updates[0]["committed"] == updates[0]["text"]

    def test_an_empty_stream_yields_no_updates(self) -> None:
        async def run() -> list[dict]:
            provider = _EchoASR()
            stream = InputAudioStream(capacity=8)
            await stream.close()
            return [update.to_dict() async for update in provider.stream_transcribe(stream.frames())]

        assert asyncio.run(run()) == []
