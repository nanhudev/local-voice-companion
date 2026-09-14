"""Conversation sessions.

A session owns its event bus, its turn state machine and the currently active
turn. Exactly one turn may be active; starting a new one cancels the old one
(barge-in), and cancellation is propagated through the shared token rather than
through string comparisons.
"""

from __future__ import annotations

import asyncio
import itertools
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..bots.schema import BotManifest
from ..core.cancellation import CancellationToken
from ..core.errors import ConfigurationError
from ..core.events import Event, EventBus, EventType, TurnTimeline
from ..core.state import InvalidTurnTransition, TurnState, can_transition
from ..core.types import ChatMessage
from .types import API_PREFIX


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Session:
    id: str = field(default_factory=new_session_id)
    bot_id: str = ""
    bot_name: str = ""
    language: str = "zh"
    bus: EventBus = field(default_factory=EventBus)
    created_at: float = field(default_factory=time.time)
    max_history: int = 24

    state: TurnState = TurnState.IDLE
    history: list[ChatMessage] = field(default_factory=list)
    active_turn: asyncio.Task | None = None
    active_token: CancellationToken | None = None
    active_timeline: TurnTimeline | None = None
    sequence: itertools.count = field(default_factory=itertools.count, repr=False)
    last_transcript: str = ""
    last_reply: str = ""
    closed: bool = False

    @classmethod
    def from_bot(cls, bot: BotManifest, **kwargs: Any) -> "Session":
        return cls(
            bot_id=bot.id,
            bot_name=bot.name,
            language=bot.language.primary,
            max_history=max(1, bot.conversation.history_turns),
            **kwargs,
        )

    # -- events -------------------------------------------------------------

    def emit(self, type: str, data: dict[str, Any] | None = None, **kwargs: Any) -> Event:
        return self.bus.emit(type, session_id=self.id, data=dict(data or {}), **kwargs)

    def set_state(self, target: TurnState | str, detail: str = "") -> bool:
        """Transition the turn state. Returns True when it actually changed."""

        target_state = TurnState(target)
        if target_state == self.state:
            return False
        if not can_transition(self.state, target_state):
            raise InvalidTurnTransition(f"{self.state.value} -> {target_state.value}")
        self.state = target_state
        payload: dict[str, Any] = {"state": target_state.value}
        if detail:
            payload["detail"] = detail
        self.bus.emit(
            EventType.TURN_STATE,
            session_id=self.id,
            turn_id=self.active_timeline.turn_id if self.active_timeline else "",
            data=payload,
        )
        return True

    # -- turn lifecycle -----------------------------------------------------

    def next_turn_id(self) -> str:
        return f"{self.id}-{next(self.sequence)}"

    def begin_turn(self) -> tuple[TurnTimeline, CancellationToken]:
        if self.closed:
            raise ConfigurationError("session is closed")
        timeline = TurnTimeline(turn_id=self.next_turn_id(), session_id=self.id)
        timeline.mark("turn_started")
        token = CancellationToken()
        self.active_timeline = timeline
        self.active_token = token
        self.bus.emit(
            EventType.TURN_STARTED,
            session_id=self.id,
            turn_id=timeline.turn_id,
            data={"turn_id": timeline.turn_id},
        )
        return timeline, token

    def finish_turn(self, status: str = "completed") -> None:
        timeline = self.active_timeline
        self.bus.emit(
            EventType.TURN_COMPLETED,
            session_id=self.id,
            turn_id=timeline.turn_id if timeline else "",
            data={"status": status, "timeline": timeline.to_dict() if timeline else None},
        )
        if timeline is not None:
            for name, value in timeline.metrics().items():
                self.bus.emit(
                    EventType.RUNTIME_METRIC,
                    session_id=self.id,
                    turn_id=timeline.turn_id,
                    data={"metric": name, "value_ms": value},
                )
        self.active_timeline = None
        self.active_token = None

    def cancel(self, detail: str = "barge-in", code: str = "barge_in") -> bool:
        """Interrupt the active turn. Idempotent and safe to call with none."""

        token = self.active_token
        if token is None or token.cancelled:
            return False
        token.cancel(detail=detail, code=code)
        self.bus.emit(
            EventType.TURN_CANCELLED,
            session_id=self.id,
            turn_id=self.active_timeline.turn_id if self.active_timeline else "",
            data={"reason": code, "detail": detail},
        )
        return True

    # -- history ------------------------------------------------------------

    def append(self, role: str, content: str) -> None:
        if not content:
            return
        self.history.append(ChatMessage(role=role, content=content))
        if len(self.history) > self.max_history * 2:
            del self.history[: -self.max_history * 2]

    def messages(
        self, system_prompt: str, user_text: str, extra_history: Sequence[ChatMessage] = ()
    ) -> list[ChatMessage]:
        messages = [ChatMessage(role="system", content=system_prompt)]
        messages.extend(self.history[-self.max_history :])
        messages.extend(extra_history)
        messages.append(ChatMessage(role="user", content=user_text))
        return messages

    def clear_history(self) -> None:
        self.history.clear()

    # -- misc ---------------------------------------------------------------

    def to_dict(self, include_history: bool = False) -> dict[str, Any]:
        """Session snapshot.

        History is omitted by default because `Session` is also serialised on
        hot paths (event fan-out, list endpoints). Callers that need to restore
        a conversation -- the session detail endpoint -- ask for it explicitly.
        """

        payload: dict[str, Any] = {
            "id": self.id,
            "bot_id": self.bot_id,
            "bot_name": self.bot_name,
            "language": self.language,
            "state": self.state.value,
            "created_at": self.created_at,
            "closed": self.closed,
            "turns": len(self.history) // 2,
            "last_transcript": self.last_transcript,
            "last_reply": self.last_reply,
            "active_turn": self.active_timeline.turn_id if self.active_timeline else None,
        }
        if include_history:
            payload["history"] = [
                {"role": message.role, "content": message.content} for message in self.history
            ]
        return payload

    async def close(self) -> None:
        self.cancel(detail="session closed", code="session_close")
        task = self.active_turn
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.closed = True
        self.bus.emit(EventType.SESSION_CLOSED, session_id=self.id, data={})

    @property
    def stream_path(self) -> str:
        return f"{API_PREFIX}/sessions/{self.id}/stream"
