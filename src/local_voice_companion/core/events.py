"""Unified runtime event schema.

Every headline-worthy thing that happens inside the runtime leaves through this
module. WebSocket, REST polling, Godot, and the web UI all consume the exact
same wire format, so no client ever reverse-engineers internals.

Wire format (stable, versioned by `v`):

    {
      "v": 1,
      "id": 17,
      "type": "llm.delta",
      "ts": 1712345678.123,
      "session_id": "...",
      "turn_id": "...",
      "data": {...}
    }
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

EVENT_SCHEMA_VERSION = 1


class EventType:
    # runtime / session
    RUNTIME_READY = "runtime.ready"
    SESSION_OPENED = "session.opened"
    SESSION_CLOSED = "session.closed"
    # turn lifecycle
    TURN_STARTED = "turn.started"
    TURN_STATE = "turn.state"
    TURN_CANCELLED = "turn.cancelled"
    TURN_COMPLETED = "turn.completed"
    # asr
    ASR_PARTIAL = "asr.partial"
    ASR_FINAL = "asr.final"
    # llm
    LLM_STARTED = "llm.started"
    LLM_DELTA = "llm.delta"
    LLM_COMPLETED = "llm.completed"
    # tts
    TTS_STARTED = "tts.started"
    TTS_AUDIO = "tts.audio"
    TTS_COMPLETED = "tts.completed"
    # playback
    PLAYBACK_STARTED = "playback.started"
    PLAYBACK_FINISHED = "playback.finished"
    # observability & errors
    RUNTIME_METRIC = "runtime.metric"
    PROVIDER_STATE = "provider.state"
    ERROR = "error"


ALL_EVENT_TYPES: tuple[str, ...] = tuple(
    sorted(
        value
        for name, value in vars(EventType).items()
        if not name.startswith("_") and isinstance(value, str)
    )
)


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    turn_id: str = ""
    id: int = 0
    ts: float = field(default_factory=time.time)

    def to_wire(self) -> dict[str, Any]:
        return {
            "v": EVENT_SCHEMA_VERSION,
            "id": self.id,
            "type": self.type,
            "ts": self.ts,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "data": self.data,
        }


class EventBus:
    """Bounded per-session event ring buffer plus live subscriber fan-out.

    The buffer is bounded on purpose: a disconnected browser must not be able to
    grow memory without limit. Old events are dropped, never piled up.
    """

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        self._events: list[Event] = []
        self._subscribers: list[Any] = []
        self._counter = 0

    def publish(self, event: Event) -> Event:
        self._counter += 1
        event.id = self._counter
        self._events.append(event)
        overflow = len(self._events) - self.capacity
        if overflow > 0:
            del self._events[:overflow]
        self._emit(event)
        return event

    def emit(self, type: str, **kwargs: Any) -> Event:
        """Convenience: build + publish in one call."""

        data = kwargs.pop("data", {})
        return self.publish(Event(type=type, data=data, **kwargs))

    def _emit(self, event: Event) -> None:
        stale: list[Any] = []
        wire = event.to_wire()
        for subscriber in list(self._subscribers):
            try:
                subscriber(wire)
            except Exception:  # a dead consumer must not poison the runtime
                stale.append(subscriber)
        for subscriber in stale:
            self.unsubscribe(subscriber)

    def subscribe(self, callback: Any) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Any) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def after(self, after_id: int = 0) -> list[dict[str, Any]]:
        return [event.to_wire() for event in self._events if event.id > after_id]

    def replay(self) -> Iterator[dict[str, Any]]:
        for event in self._events:
            yield event.to_wire()

    @property
    def last_id(self) -> int:
        return self._counter


# ---------------------------------------------------------------------------
# Turn timeline: the single place where latency is measured.
# ---------------------------------------------------------------------------

TIMELINE_STAGES: tuple[str, ...] = (
    "turn_started",
    "vad_end",
    "asr_start",
    # Present only when the recogniser produced a hypothesis before the
    # utterance ended. Its absence is meaningful: it means this turn waited for
    # the whole utterance, which is exactly the latency this phase measures.
    "asr_first_partial",
    "asr_end",
    "llm_start",
    "llm_first_token",
    "llm_end",
    "tts_start",
    "tts_first_audio",
    "playback_start",
    "playback_end",
)

#: Wall-clock clock used for turn timing.
#:
#: `time.monotonic()` on Windows is driven by the system tick (~15.6 ms), which
#: is coarser than most stages of a well-tuned voice turn. Timing a whole turn
#: with it collapses every stage to the same tick and reports 0 ms -- turning the
#: project's headline metric into a lie. `perf_counter` is high-resolution and
#: still monotonic, so it is the correct clock here.
try:  # pragma: no cover - platform capability probe
    _now = time.perf_counter
except AttributeError:  # pragma: no cover - no platform lacks this
    _now = time.monotonic

#: Below this many milliseconds a difference is not trusted as a real
#: measurement. Reported values stay as measured; this is only used to decide
#: whether a stage is fast enough that its number is clock-limited.
CLOCK_FLOOR_MS = 1.0


@dataclass
class TurnTimeline:
    """High-resolution timestamps for one turn. Missing values stay None."""

    turn_id: str
    session_id: str = ""
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, stage: str, when: float | None = None) -> None:
        if stage not in TIMELINE_STAGES:
            raise KeyError(f"unknown timeline stage: {stage}")
        if stage in self.marks:
            return
        self.marks[stage] = _now() if when is None else when

    def _delta_ms(self, start: str, end: str) -> int | None:
        if start not in self.marks or end not in self.marks:
            return None
        return int(round((self.marks[end] - self.marks[start]) * 1000))

    @property
    def clock_limited(self) -> bool:
        """True when the whole turn was faster than the clock can resolve.

        A caller must never present a 0 ms stage as "instant"; it means "not
        measurable at this resolution", usually because a stub provider was used.
        """

        total = self.total_turn_ms
        return total is not None and total < CLOCK_FLOOR_MS

    @property
    def asr_latency_ms(self) -> int | None:
        start = self.marks.get("vad_end")
        end = self.marks.get("asr_end")
        return None if start is None or end is None else int(round((end - start) * 1000))

    @property
    def asr_ttfp_ms(self) -> int | None:
        """Speech start -> first partial hypothesis.

        On the replay path (`iter_frames` over a finished buffer) this measures
        the recogniser alone, because `asr_start` is stamped when the replay
        begins rather than when the sound was captured. On the live path it is
        the number a user feels.
        """

        return self._delta_ms("asr_start", "asr_first_partial")

    @property
    def llm_ttft_ms(self) -> int | None:
        return self._delta_ms("llm_start", "llm_first_token")

    @property
    def tts_ttfa_ms(self) -> int | None:
        return self._delta_ms("tts_start", "tts_first_audio")

    @property
    def time_to_first_audio_ms(self) -> int | None:
        """The headline metric: speech end -> audible assistant audio."""

        return self._delta_ms("vad_end", "tts_first_audio")

    @property
    def total_turn_ms(self) -> int | None:
        """turn_started -> the last stage that actually happened.

        `playback_end` is the natural end of a spoken turn, but a text-only turn
        never plays audio. Falling back keeps the metric defined instead of
        silently returning None and hiding the turn from the metrics registry.
        """

        end = None
        for stage in ("playback_end", "llm_end", "asr_end"):
            if stage in self.marks:
                end = stage
                break
        if end is None:
            return None
        return self._delta_ms("turn_started", end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "stages": dict(self.marks),
            "asr_latency_ms": self.asr_latency_ms,
            "asr_ttfp_ms": self.asr_ttfp_ms,
            "llm_ttft_ms": self.llm_ttft_ms,
            "tts_ttfa_ms": self.tts_ttfa_ms,
            "time_to_first_audio_ms": self.time_to_first_audio_ms,
            "total_turn_ms": self.total_turn_ms,
            "clock_limited": self.clock_limited,
        }

    def metrics(self) -> dict[str, int]:
        derived = {
            "asr_latency": self.asr_latency_ms,
            "asr_ttfp": self.asr_ttfp_ms,
            "llm_ttft": self.llm_ttft_ms,
            "tts_ttfa": self.tts_ttfa_ms,
            "time_to_first_audio": self.time_to_first_audio_ms,
            "total_turn": self.total_turn_ms,
        }
        return {key: value for key, value in derived.items() if value is not None}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TurnTimeline({asdict(self)})"
