"""Bounded queues with explicit drop policy.

`asyncio.Queue()` with no maxsize is forbidden on the streaming path: a stalled
consumer would otherwise grow memory without limit. Every queue here is bounded
and every overflow is counted, so telemetry can show real backpressure.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

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

    def __init__(self, capacity: int, *, policy: DropPolicy = "oldest", name: str = "") -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=capacity)
        self.policy: DropPolicy = policy
        self.name = name
        self.stats = QueueStats(capacity=capacity)

    @property
    def capacity(self) -> int:
        return self._queue.maxsize

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    async def put(self, item: T) -> bool:
        """Enqueue without blocking the producer. Returns False if dropped."""

        if not self._queue.full():
            self._queue.put_nowait(item)
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
