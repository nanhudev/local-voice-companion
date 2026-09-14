"""Conversation turn state machine.

Clients consume these states verbatim. They must never guess meaning from
free-form strings produced by a specific provider or the web UI.
"""

from __future__ import annotations

from enum import Enum


class TurnState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    CAPTURING = "CAPTURING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SYNTHESIZING = "SYNTHESIZING"
    SPEAKING = "SPEAKING"
    CANCELLING = "CANCELLING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


TURN_TRANSITIONS: dict[TurnState, set[TurnState]] = {
    # IDLE may enter the pipeline mid-way: a text-only turn (the TTS/chat proxy
    # endpoints) has nothing to capture, and a turn submitted with a pre-recorded
    # audio buffer goes straight to TRANSCRIBING. Requiring IDLE -> LISTENING
    # first would force the API to fake a listening phase it never performed.
    TurnState.IDLE: {
        TurnState.LISTENING,
        TurnState.TRANSCRIBING,
        TurnState.THINKING,
        TurnState.SYNTHESIZING,
        TurnState.PAUSED,
        TurnState.ERROR,
    },
    TurnState.LISTENING: {
        TurnState.CAPTURING,
        TurnState.TRANSCRIBING,
        TurnState.THINKING,
        TurnState.SYNTHESIZING,
        TurnState.IDLE,
        TurnState.PAUSED,
        TurnState.ERROR,
    },
    TurnState.CAPTURING: {
        TurnState.TRANSCRIBING,
        TurnState.CANCELLING,
        TurnState.LISTENING,
        TurnState.ERROR,
    },
    TurnState.TRANSCRIBING: {
        TurnState.THINKING,
        TurnState.CANCELLING,
        TurnState.LISTENING,
        TurnState.ERROR,
    },
    TurnState.THINKING: {
        TurnState.SYNTHESIZING,
        TurnState.SPEAKING,
        TurnState.CANCELLING,
        TurnState.LISTENING,
        TurnState.IDLE,
        TurnState.ERROR,
    },
    TurnState.SYNTHESIZING: {
        TurnState.SPEAKING,
        TurnState.THINKING,
        TurnState.CANCELLING,
        TurnState.LISTENING,
        TurnState.IDLE,
        TurnState.ERROR,
    },
    TurnState.SPEAKING: {
        TurnState.LISTENING,
        TurnState.IDLE,
        TurnState.CANCELLING,
        TurnState.ERROR,
    },
    TurnState.CANCELLING: {TurnState.LISTENING, TurnState.IDLE, TurnState.ERROR},
    TurnState.PAUSED: {TurnState.LISTENING, TurnState.IDLE, TurnState.ERROR},
    TurnState.ERROR: {TurnState.LISTENING, TurnState.IDLE, TurnState.PAUSED},
}


class InvalidTurnTransition(RuntimeError):
    pass


def can_transition(current: TurnState | str, target: TurnState | str) -> bool:
    current = TurnState(current)
    target = TurnState(target)
    return target == current or target in TURN_TRANSITIONS.get(current, set())
