"""Provider lifecycle states.

A provider must never be described by a pile of booleans. There is exactly one
state per instance, transitions are validated, and every state has a meaning:

    DISCOVERED   -> the registry knows the implementation exists
    AVAILABLE    -> probe() succeeded; dependencies are installed
    LOADING      -> load() in progress
    READY        -> warm and idle, safe to serve a turn
    BUSY         -> currently serving a turn
    DEGRADED     -> still usable but not healthy (fallback device, slow, etc.)
    UNAVAILABLE  -> probe() failed or dependencies missing
    UNLOADING    -> unload() in progress
    ERROR        -> load/health failed; needs operator attention
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProviderState(str, Enum):
    DISCOVERED = "DISCOVERED"
    AVAILABLE = "AVAILABLE"
    LOADING = "LOADING"
    READY = "READY"
    BUSY = "BUSY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    UNLOADING = "UNLOADING"
    ERROR = "ERROR"


TRANSITIONS: dict[ProviderState, set[ProviderState]] = {
    ProviderState.DISCOVERED: {ProviderState.AVAILABLE, ProviderState.UNAVAILABLE, ProviderState.ERROR},
    ProviderState.AVAILABLE: {ProviderState.LOADING, ProviderState.UNAVAILABLE, ProviderState.ERROR},
    ProviderState.LOADING: {ProviderState.READY, ProviderState.DEGRADED, ProviderState.ERROR, ProviderState.UNLOADING},
    ProviderState.READY: {ProviderState.BUSY, ProviderState.DEGRADED, ProviderState.UNLOADING, ProviderState.ERROR},
    ProviderState.BUSY: {ProviderState.READY, ProviderState.DEGRADED, ProviderState.ERROR, ProviderState.UNLOADING},
    ProviderState.DEGRADED: {ProviderState.READY, ProviderState.BUSY, ProviderState.UNLOADING, ProviderState.ERROR},
    ProviderState.UNAVAILABLE: {ProviderState.AVAILABLE, ProviderState.ERROR},
    ProviderState.UNLOADING: {ProviderState.AVAILABLE, ProviderState.UNAVAILABLE, ProviderState.ERROR},
    ProviderState.ERROR: {ProviderState.AVAILABLE, ProviderState.UNAVAILABLE, ProviderState.LOADING},
}

# States that can serve a request. `DEGRADED` is intentionally inclusive so a
# CPU fallback keeps answering instead of collapsing into a 500.
SERVING_STATES = frozenset({ProviderState.READY, ProviderState.BUSY, ProviderState.DEGRADED})

TERMINAL_STATES = frozenset({ProviderState.UNAVAILABLE, ProviderState.ERROR})


class InvalidTransition(RuntimeError):
    pass


@dataclass
class Lifecycle:
    """Single source of truth for one provider instance's state."""

    provider_id: str
    state: ProviderState = ProviderState.DISCOVERED
    detail: str = ""

    def can_transition(self, target: ProviderState) -> bool:
        return target in TRANSITIONS.get(self.state, set())

    def transition(self, target: ProviderState, detail: str = "") -> ProviderState:
        target = ProviderState(target)
        if target != self.state and not self.can_transition(target):
            raise InvalidTransition(
                f"provider {self.provider_id}: cannot go {self.state.value} -> {target.value}"
            )
        self.state = target
        if detail:
            self.detail = detail
        return self.state

    @property
    def is_serving(self) -> bool:
        return self.state in SERVING_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "state": self.state.value,
            "detail": self.detail,
            "serving": self.is_serving,
        }
