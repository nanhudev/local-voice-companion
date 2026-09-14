"""Cancellation primitives.

Every streaming stage accepts a token. When barge-in happens the token fires,
all in-flight coroutines observe it, and no stale audio from the previous turn
can leak into playback.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .errors import CancelledTurn


@dataclass
class CancelReason:
    code: str
    detail: str = ""


class CancellationToken:
    """Thread/async safe one-shot cancellation flag."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: CancelReason | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, detail: str = "", code: str = "barge_in") -> None:
        if not self._event.is_set():
            self.reason = CancelReason(code=code, detail=detail)
            self._event.set()

    def raise_if_cancelled(self, detail: str = "") -> None:
        if self._event.is_set():
            raise CancelledTurn(detail or (self.reason.detail if self.reason else "turn cancelled"))

    async def wait(self) -> None:
        await self._event.wait()

    async def sleep(self, seconds: float) -> None:
        """Sleep that wakes up immediately when the turn is cancelled."""

        try:
            await asyncio.wait_for(self._event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return
        raise CancelledTurn(self.reason.detail if self.reason else "turn cancelled")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cancelled": self.cancelled,
            "reason": None if self.reason is None else vars(self.reason),
        }
