"""Barge-in: the user starts talking while the assistant is still talking.

This is the first half of duplex behaviour and the one that is measurable
end-to-end on this machine. The other half -- a *model* that listens and speaks
at the same instant -- needs 18-24 GB of VRAM (see `docs/DUPLEX_FEASIBILITY.md`)
and is not attempted. What is implemented here is pipeline-level arbitration:
speech detected on the microphone during playback tears the turn down.

Three things make this more than "call cancel when VAD fires":

* **Hysteresis.** A single 20 ms frame of noise must not interrupt a sentence.
  `min_speech_ms` of *continuous* speech is required; any silence longer than
  `reset_silence_ms` restarts the count.
* **Cooldown.** The instant playback starts, the tail of the previous
  utterance and the speaker's own attack transient are still in the signal. For
  headset use this barely matters; for speaker use it is the difference between
  a working system and one that interrupts itself forever. Real speaker barge-in
  additionally needs acoustic echo cancellation (step 3D); a cooldown window is
  a mitigation, not a substitute.
* **Honest absence.** With no VAD provider loaded there is no way to detect
  speech, so the watcher disables itself rather than guessing from energy. A
  disabled watcher must never be reported as an armed one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from ..core.audio import AudioFrame
    from ..core.session import Session
    from ..providers.base import VADProvider

from ..core.events import EventType
from ..core.state import TurnState

#: States in which the assistant is producing audio, so user speech is an
#: interruption rather than the turn's own input.
ARMED_STATES: frozenset[TurnState] = frozenset({TurnState.SPEAKING, TurnState.SYNTHESIZING})


@dataclass
class BargeInConfig:
    """Tuning for :class:`BargeInWatcher`."""

    enabled: bool = True
    #: Continuous speech required before interrupting. Below ~120 ms a door
    #: slam or a cough interrupts the assistant; above ~400 ms the user has
    #: already finished a word before anything stops.
    min_speech_ms: int = 180
    #: Silence that resets the accumulator. Shorter than the gap inside a
    #: sentence (breath, plosive) so normal speech is not split in two.
    reset_silence_ms: int = 200
    #: Grace period measured from `playback_start`. Echo and the speaker's own
    #: attack transient live here.
    cooldown_ms: int = 350


@dataclass
class BargeInDecision:
    """What one frame produced. Returned even when nothing was triggered."""

    triggered: bool = False
    speech_ms: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"triggered": self.triggered, "speech_ms": round(self.speech_ms, 1), "reason": self.reason}


@dataclass
class BargeInWatcher:
    """Watch the microphone while the assistant speaks, and interrupt on speech.

    The watcher is deliberately dumb about *what happens next*. It cancels the
    turn; deciding whether to start a new one from the interrupting audio is the
    arbiter's job (step 3C), not this module's.
    """

    session: "Session"
    vad: "VADProvider | None" = None
    config: BargeInConfig = field(default_factory=BargeInConfig)
    clock: Any = field(default=time.perf_counter)

    _speech_ms: float = field(default=0.0, init=False, repr=False)
    _silence_ms: float = field(default=0.0, init=False, repr=False)
    _fired_for: str | None = field(default=None, init=False, repr=False)

    # -- status -------------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether detection is possible at all on this machine.

        `lifecycle.is_serving` is checked rather than mere presence: a VAD that
        failed to load is not a VAD, and pretending otherwise would make the
        runtime claim a capability it does not have.
        """

        return self.config.enabled and self.vad is not None and self.vad.lifecycle.is_serving

    @property
    def armed(self) -> bool:
        """Whether a frame arriving right now can interrupt anything."""

        return self.available and TurnState(self.session.state) in ARMED_STATES

    @property
    def speech_ms(self) -> float:
        """Audio seen so far in the current run of consecutive speech."""

        return self._speech_ms

    @property
    def silence_ms(self) -> float:
        """Silence accumulated since the last speech frame."""

        return self._silence_ms

    # -- detection ----------------------------------------------------------

    def observe(self, frame: "AudioFrame") -> BargeInDecision:
        """Feed one frame. Triggers at most once per playback episode."""

        if not self.armed:
            self.reset()
            return BargeInDecision(reason="not_armed")
        assert self.vad is not None

        now = self.clock()
        # Measured in *audio* time, not wall-clock time: each frame contributes
        # its own duration. Timing this externally would let a queue stall or a
        # GC pause look like a long utterance, which is how a brief cough turns
        # into an interruption.
        if self._is_speaking(frame):
            self._speech_ms += frame.duration_ms
            self._silence_ms = 0.0
        else:
            if self._speech_ms:
                self._silence_ms += frame.duration_ms
                if self._silence_ms >= self.config.reset_silence_ms:
                    # A genuine gap, so the next speech starts a new run and
                    # must earn the threshold again from zero.
                    self._speech_ms = 0.0
                    self._silence_ms = 0.0

        if self._speech_ms <= 0.0:
            return BargeInDecision(reason="silence")

        if self._in_cooldown(now):
            return BargeInDecision(speech_ms=self._speech_ms, reason="cooldown")
        if self._speech_ms < self.config.min_speech_ms:
            return BargeInDecision(speech_ms=self._speech_ms, reason="below_threshold")

        elapsed = self._speech_ms
        return BargeInDecision(triggered=self._fire(elapsed), speech_ms=elapsed, reason="speech")

    def reset(self) -> None:
        """Forget accumulated speech. Called when a turn ends or restarts."""

        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._fired_for = None

    # -- internals ----------------------------------------------------------

    def _is_speaking(self, frame: "AudioFrame") -> bool:
        try:
            return bool(self.vad.is_speech(frame.to_chunk()))  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - a broken VAD must not kill the turn
            # Detection is best-effort. Silently disabling here would be worse
            # than not reporting it, so the failure is visible on the bus.
            self.config.enabled = False
            self.session.emit(
                EventType.ERROR,
                {"message": "barge-in VAD failed; barge-in disabled", "error_type": "vad_failure"},
            )
            return False

    def _in_cooldown(self, now: float) -> bool:
        timeline = self.session.active_timeline
        if timeline is None:
            return False
        started = timeline.marks.get("playback_start")
        if started is None:
            return False
        return (now - started) * 1000.0 < self.config.cooldown_ms

    def _fire(self, speech_ms: float) -> bool:
        turn_id = self.session.active_timeline.turn_id if self.session.active_timeline else ""
        if self._fired_for == turn_id:
            # One interruption per turn. Firing again would only produce a
            # second `bargein.detected` for an interruption already reported.
            return False
        self._fired_for = turn_id
        if self.session.active_timeline is not None:
            self.session.active_timeline.mark("bargein_detected")
        self.session.emit(
            EventType.BARGE_IN_DETECTED,
            {
                "speech_ms": round(speech_ms, 1),
                "min_speech_ms": self.config.min_speech_ms,
                "state": TurnState(self.session.state).value,
            },
            turn_id=turn_id,
        )
        self.session.cancel(detail=f"barge-in: {speech_ms:.0f} ms of speech during playback", code="barge_in")
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        return True
