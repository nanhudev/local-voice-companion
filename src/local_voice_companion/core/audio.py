"""Continuous input audio: frames, ordering, and backpressure.

`AudioChunk` (in :mod:`core.types`) models a **complete utterance** -- audio
whose length is known and which can be handed to ASR in one call. That is right
for a turn-based runtime and wrong for a duplex one, where audio arrives forever
and never "completes".

A duplex input path needs two things `AudioChunk` deliberately does not carry:

* **sequence** -- to detect gaps. If frames arrive 1,2,3,7,8 then frames 4..6
  were lost, and a partial transcript built from that audio is quietly wrong.
  With a bare `AudioChunk` the loss is invisible; with a sequence number the
  runtime can say "this hypothesis rests on audio with a hole in it".
* **captured_at** -- the moment the audio *entered the process*, not the moment a
  consumer happened to read it. Latency measured from a dequeue timestamp is
  really "queue wait + decode". Time To First Partial has to start at capture, or
  the headline metric of this phase quietly becomes a measure of scheduling.

Timing uses `time.perf_counter` for the same reason :mod:`core.events` does:
Windows `time.monotonic` is tick-quantised to roughly 15.6 ms, which is coarser
than a 20 ms frame's own processing time and would report many frames as 0 ms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable

from ..pipeline.queue import BoundedQueue
from .types import AudioChunk

__all__ = ["AudioFrame", "InputAudioStream", "StreamStats", "concat_frames"]

_perf_counter = time.perf_counter


@dataclass(frozen=True)
class AudioFrame:
    """One slice of an audio stream that is still open."""

    pcm: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2
    #: Monotonically increasing inside one stream. A gap means loss.
    sequence: int = 0
    #: `perf_counter` seconds at the moment this audio entered the runtime.
    captured_at: float = 0.0
    #: True for the last frame of a stream.
    is_final: bool = False

    @property
    def duration_ms(self) -> float:
        if not self.pcm:
            return 0.0
        frames = len(self.pcm) / (self.sample_width * self.channels)
        return frames / self.sample_rate * 1000.0

    def age_ms(self, now: float | None = None) -> float:
        """How long this frame has been sitting inside the runtime."""

        reference = _perf_counter() if now is None else now
        return max(0.0, (reference - self.captured_at) * 1000.0)

    def to_chunk(self) -> AudioChunk:
        """Downgrade for APIs that still speak in whole utterances.

        This loses `sequence` and `captured_at`, and that is accepted
        deliberately: the loss should be visible at the call site, not silent.
        """

        return AudioChunk(
            pcm=self.pcm,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
            is_final=self.is_final,
        )


def concat_frames(frames: Iterable[AudioFrame]) -> AudioChunk:
    """Fold frames back into one utterance, preserving order.

    Mixed sample rates are refused rather than silently resampled: a caller that
    needs resampling should ask for it, because doing it here would hide a real
    pipeline bug behind a plausible-looking transcript.
    """

    parts: list[bytes] = []
    sample_rate = 0
    channels = 1
    sample_width = 2
    for frame in frames:
        if frame.sample_rate and sample_rate and frame.sample_rate != sample_rate:
            raise ValueError(
                f"cannot concatenate mixed sample rates: {sample_rate} and {frame.sample_rate}"
            )
        if frame.sample_rate and not sample_rate:
            sample_rate = frame.sample_rate
            channels = frame.channels
            sample_width = frame.sample_width
        parts.append(frame.pcm)
    return AudioChunk(
        pcm=b"".join(parts),
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        is_final=True,
    )


@dataclass
class StreamStats:
    frames_in: int = 0
    frames_out: int = 0
    dropped: int = 0
    gaps: int = 0
    closed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_in": self.frames_in,
            "frames_out": self.frames_out,
            "dropped": self.dropped,
            "gaps": self.gaps,
            "closed": self.closed,
        }


class InputAudioStream:
    """A bounded, self-sequencing queue of microphone frames.

    The bound is the entire point. A microphone produces PCM frames forever; an
    unbounded queue turns any slow consumer -- a stalled websocket, a GC pause, a
    TTS worker holding the loop -- into unbounded memory growth.

    When the buffer fills, the **oldest** frame is dropped. For live audio the
    newest second matters more than the second before it, and preserving old
    audio at the cost of ever-growing latency is how duplex systems become laggy
    ones.

    Every drop and every sequence gap is counted and readable through `stats()`,
    because silent loss is what would let a partial transcript look authoritative
    while resting on audio with holes in it.
    """

    def __init__(
        self,
        capacity: int = 64,
        *,
        policy: str = "oldest",
        name: str = "input-audio",
    ) -> None:
        # One reserved slot for the end-of-stream marker. Closing a full stream
        # must not evict audio: the final frames are exactly the ones the
        # recogniser needs in order to turn a good hypothesis into a final.
        self._capacity = capacity
        self._queue: BoundedQueue[AudioFrame | None] = BoundedQueue(
            capacity, policy=policy, name=name, reserved=1  # type: ignore[arg-type]
        )
        self._sequence = 0
        self._gaps = 0
        self._frames_in = 0
        self._frames_out = 0
        self._closed = False

    # -- producer side ------------------------------------------------------

    def frame(
        self,
        pcm: bytes,
        sample_rate: int,
        *,
        channels: int = 1,
        sample_width: int = 2,
        captured_at: float | None = None,
        is_final: bool = False,
    ) -> AudioFrame:
        """Build a frame stamped with the *next* sequence number and capture time."""

        return AudioFrame(
            pcm=pcm,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            sequence=self._sequence,
            captured_at=_perf_counter() if captured_at is None else captured_at,
            is_final=is_final,
        )

    async def put(self, frame: AudioFrame) -> bool:
        """Enqueue a frame. Returns False when it was dropped under backpressure."""

        # Gap detection lives on the consumer side only. Counting here as well
        # reported the same hole twice: a frame that arrives late is missing from
        # the producer's view *and* from the consumer's, but it is one gap.
        self._sequence = max(self._sequence, frame.sequence) + 1
        self._frames_in += 1
        return await self._queue.put(frame)

    async def put_pcm(
        self,
        pcm: bytes,
        sample_rate: int,
        *,
        channels: int = 1,
        sample_width: int = 2,
        captured_at: float | None = None,
        is_final: bool = False,
    ) -> bool:
        """Convenience wrapper: build the frame, stamp it, enqueue it."""

        return await self.put(
            self.frame(
                pcm,
                sample_rate,
                channels=channels,
                sample_width=sample_width,
                captured_at=captured_at,
                is_final=is_final,
            )
        )

    async def close(self) -> None:
        """Signal end-of-stream exactly once."""

        if self._closed:
            return
        self._closed = True
        await self._queue.put(None, force=True)

    # -- consumer side ------------------------------------------------------

    async def frames(self) -> AsyncIterator[AudioFrame]:
        """Yield frames until the producer closes the stream."""

        expected = 0
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if item.sequence > expected:
                # Frames were dropped by the backpressure policy before reaching
                # us, so the sequence number itself reveals the hole.
                self._gaps += 1
            expected = item.sequence + 1
            self._frames_out += 1
            yield item

    def drain(self) -> int:
        """Discard everything buffered. Used on barge-in. Returns what was lost."""

        return len(self._queue.drain())

    # -- introspection ------------------------------------------------------

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def closed(self) -> bool:
        return self._closed

    def stats(self) -> StreamStats:
        """Live counts. `dropped`/`gaps` are what make a lossy stream honest."""

        return StreamStats(
            frames_in=self._frames_in,
            frames_out=self._frames_out,
            dropped=self._queue.stats.dropped,
            gaps=self._gaps,
            closed=self._closed,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.stats().to_dict()
        payload.update(
            {
                "name": self._queue.name,
                "capacity": self._queue.capacity,
                "depth": self._queue.qsize(),
                "policy": self._queue.policy,
            }
        )
        return payload
