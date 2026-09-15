"""Bounded queues with explicit drop policy.

`asyncio.Queue()` with no maxsize is forbidden on the streaming path: a stalled
consumer would otherwise grow memory without limit. Every queue here is bounded
and every overflow is counted, so telemetry can show real backpressure.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from ..core.cancellation import CancellationToken

T = TypeVar("T")

DropPolicy = Literal["oldest", "newest", "await"]


@dataclass
class QueueStats:
    capacity: int
    depth: int = 0
    enqueued: int = 0
    dequeued: int = 0
    dropped: int = 0
    last_overflow: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "depth": self.depth,
            "enqueued": self.enqueued,
            "dequeued": self.dequeued,
            "dropped": self.dropped,
        }


class BoundedQueue(Generic[T]):
    """Capacity-limited queue with a policy for what happens when it is full."""

    def __init__(
        self,
        capacity: int,
        *,
        policy: DropPolicy = "oldest",
        name: str = "",
        reserved: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if reserved < 0:
            raise ValueError("reserved must be >= 0")
        # `reserved` slots exist so a control item -- an end-of-stream marker --
        # can always be enqueued. Without them, closing a full queue makes the
        # drop policy treat the marker as one more payload frame and evict real
        # audio that had already arrived.
        self._capacity = capacity
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=capacity + reserved)
        self.policy: DropPolicy = policy
        self.name = name
        self.reserved = reserved
        self.stats = QueueStats(capacity=capacity)

    @property
    def capacity(self) -> int:
        return self._capacity

    def _payload_full(self) -> bool:
        """Full for payload purposes. Reserved slots are not payload space."""

        return self._queue.qsize() >= self._capacity

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    async def put(self, item: T, *, force: bool = False) -> bool:
        """Enqueue without blocking the producer. Returns False if dropped.

        ``force=True`` places the item in the reserved space, bypassing the drop
        policy. It exists so that *control* items -- end-of-stream markers,
        cancellation notices -- can never be the reason a payload item is lost.
        """

        if force or not self._payload_full():
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:  # pragma: no cover - only if force over-reserves
                self.stats.dropped += 1
                self.stats.last_overflow = time.monotonic()
                return False
            self.stats.enqueued += 1
            self._sync_depth()
            return True

        if self.policy == "newest":
            self.stats.dropped += 1
            self.stats.last_overflow = time.monotonic()
            return False

        if self.policy == "oldest":
            self._drop_one()
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:  # pragma: no cover - race window
                self.stats.dropped += 1
                return False
            self.stats.enqueued += 1
            return True

        # policy == "await": apply backpressure, but never forever.
        await self._queue.put(item)
        self.stats.enqueued += 1
        self._sync_depth()
        return True

    def put_nowait(self, item: T) -> bool:
        try:
            self._queue.put_nowait(item)
            self.stats.enqueued += 1
            self._sync_depth()
            return True
        except asyncio.QueueFull:
            if self.policy == "oldest":
                self._drop_one()
                try:
                    self._queue.put_nowait(item)
                    self.stats.enqueued += 1
                    return True
                except asyncio.QueueFull:  # pragma: no cover
                    self.stats.dropped += 1
                    return False
            self.stats.dropped += 1
            self.stats.last_overflow = time.monotonic()
            return False

    async def get(self) -> T:
        item = await self._queue.get()
        self.stats.dequeued += 1
        self._sync_depth()
        return item

    async def get_or_cancel(self, token: CancellationToken) -> T:
        """Take the next item, or fail the moment the turn is cancelled.

        A bare `await queue.get()` leaves a stage asleep until the next item
        happens to arrive. On barge-in that is precisely the wrong behaviour:
        the audio emitter would sit on a half-spoken sentence until the
        synthesizer happened to produce another chunk, and the interruption
        would be audible as a delay rather than felt as an interruption.

        Racing the queue against the token makes cancellation take effect
        within one event-loop turn, which is what the barge-in latency number
        measures.
        """

        # Imported here, not at module scope: `core.audio` imports this module,
        # so a top-level import of `core.*` would close an import cycle.
        from ..core.errors import CancelledTurn

        token.raise_if_cancelled()
        getter = asyncio.ensure_future(self._queue.get())
        waiter = asyncio.ensure_future(token.wait())
        try:
            done, pending = await asyncio.wait(
                {getter, waiter}, return_when=asyncio.FIRST_COMPLETED
            )
        except BaseException:  # pragma: no cover - outer cancellation
            getter.cancel()
            waiter.cancel()
            raise
        for task in pending:
            task.cancel()
        if getter in done:
            self.stats.dequeued += 1
            self._sync_depth()
            return getter.result()
        # The loser may have been cancelled mid-retrieval, so an item can be
        # lost here. That is accepted: the turn is over and its audio is stale.
        raise CancelledTurn(token.reason.detail if token.reason else "turn cancelled")

    def drain(self) -> list[T]:
        """Drop everything currently queued (used on barge-in)."""

        items: list[T] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        self.stats.dropped += len(items)
        self._sync_depth()
        return items

    def _drop_one(self) -> None:
        try:
            self._queue.get_nowait()
            self.stats.dropped += 1
            self.stats.last_overflow = time.monotonic()
        except asyncio.QueueEmpty:  # pragma: no cover - racing consumer
            pass

    def _sync_depth(self) -> None:
        self.stats.depth = self._queue.qsize()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "policy": self.policy, **self.stats.to_dict()}
