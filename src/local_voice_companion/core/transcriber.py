"""Incremental transcript stability.

A streaming recogniser does not produce a transcript; it produces a *sequence of
guesses* about the same audio. Feeding every guess to a UI produces text that
visibly rewrites itself, and feeding every guess to a turn-completion heuristic
(produces a decision on noise. Both problems are the same problem: nothing has
told consumers which part of the text is **settled**.

This module splits every result into two parts:

* **committed** -- the prefix that has stopped changing. Safe to display, safe to
  reason about, and above all safe to *keep* if the user interrupts right now.
* **unstable** -- the tail still being revised. Display it greyed out; never let
  it decide anything.

How stability is decided
------------------------

The rule is deliberately simple and inspectable: when two consecutive results
agree on a prefix, that prefix is treated as settled. It is not a scorer: there
is no confidence threshold to tune, and nothing here needs a model. A heuristic
that has to be tuned per language is worse than one whose behaviour you can read
off a unit test.

Two edge cases matter and are handled explicitly rather than averaged away:

* **The first result cannot be committed.** With only one observation there is
  nothing to compare against, so nothing has "stopped changing" yet. Committing
  the first guess would be pretending to have evidence we do not have.
* **Recognisers revise backwards.** A result can get *shorter* than what was
  already committed. Silently keeping the longer committed prefix would emit text
  the recogniser has retracted; clamping it back down is the honest move, and the
  clamping is allowed to reduce `committed` because keeping it would be a lie.

Timing uses `time.perf_counter` and every update carries `since_change_ms`, which
is what pause detection needs later: "how long has this hypothesis been sitting
still" is the single most useful signal for deciding whether a speaker has
finished a thought.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["TranscriptUpdate", "TranscriptStabilityTracker"]

_perf_counter = time.perf_counter


def common_prefix(left: str, right: str) -> str:
    """Longest shared leading run of two strings."""

    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


@dataclass(frozen=True)
class TranscriptUpdate:
    """One hypothesis about the audio so far.

    `text` is always the recogniser's full current guess. `committed` and
    `unstable` are the derived split, and `committed + unstable == text` is an
    invariant callers may rely on.
    """

    text: str
    committed: str = ""
    unstable: str = ""
    is_final: bool = False
    #: Monotonic counter within one tracker, useful for gap detection on the wire.
    sequence: int = 0
    #: ms since the previous *changing* result; 0 for the first one.
    since_change_ms: float = 0.0
    #: ms from stream start to this result.
    elapsed_ms: float = 0.0
    #: True when frames were lost before this result (see core.audio).
    degraded: bool = False

    @property
    def stability(self) -> float:
        """Fraction of the text that has settled. 0.0 for an empty transcript."""

        if not self.text:
            return 0.0
        return len(self.committed) / len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "committed": self.committed,
            "unstable": self.unstable,
            "is_final": self.is_final,
            "sequence": self.sequence,
            "since_change_ms": round(self.since_change_ms, 1),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "stability": round(self.stability, 3),
            "degraded": self.degraded,
        }


class TranscriptStabilityTracker:
    """Turns a stream of full retranscriptions into a stream of stable updates.

    Construction takes a start time so that `elapsed_ms` is measured from the
    beginning of the utterance rather than from the first call, which is what
    makes the number comparable to the benchmark's own definition.
    """

    def __init__(
        self,
        started_at: float | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        # The clock is injectable because `since_change_ms` is the signal pause
        # detection will be built on, and a number you cannot control is a number
        # you cannot test. Taking `started_at` without a clock used to mix an
        # injected origin with wall-clock samples, so every elapsed figure came
        # out ~7.4e6 ms and negative -- a real defect, not a style preference.
        self._clock: Callable[[], float] = clock or _perf_counter
        self._started = self._clock() if started_at is None else started_at
        self._previous = ""
        self._committed = ""
        self._last_change = self._started
        self._sequence = 0
        self._emitted = 0
        self._suppressed = 0

    # -- feed ---------------------------------------------------------------

    def push(self, text: str, *, degraded: bool = False) -> TranscriptUpdate | None:
        """Fold one recogniser result into the stream.

        Returns ``None`` when the result changed nothing, which is how duplicate
        frames are suppressed: streaming engines commonly emit the same text many
        times while waiting for more audio, and forwarding those would flood the
        websocket and re-trigger every downstream decision for free.
        """

        now = self._clock()
        text = text or ""

        changed = text != self._previous
        if not changed:
            self._suppressed += 1
            return None

        # Growth: two consecutive results agree, so their shared prefix settled.
        candidate = common_prefix(self._previous, text) if self._previous else ""
        if len(candidate) > len(self._committed) and candidate.startswith(self._committed):
            self._committed = candidate

        # Retraction: the recogniser took back something we already committed.
        # Clamping keeps `committed` truthful instead of convenient.
        if self._committed and not text.startswith(self._committed):
            self._committed = common_prefix(self._committed, text)

        self._previous = text
        since_change = (now - self._last_change) * 1000.0
        self._last_change = now
        self._sequence += 1
        self._emitted += 1

        return TranscriptUpdate(
            text=text,
            committed=self._committed,
            unstable=text[len(self._committed) :],
            is_final=False,
            sequence=self._sequence,
            since_change_ms=since_change,
            elapsed_ms=(now - self._started) * 1000.0,
            degraded=degraded,
        )

    def finalize(self, text: str | None = None, *, degraded: bool = False) -> TranscriptUpdate:
        """Produce the terminal update and mark it final.

        Callers get a final frame even when nothing ever changed, because a client
        tracking the transcript needs a terminator, not a stream that merely stops.
        """

        now = self._clock()
        final_text = self._previous if text is None else (text or "")
        if text is not None and text != self._previous:
            self._previous = final_text
            self._last_change = now
        # At end-of-audio everything is settled by definition: there is no more
        # evidence that could revise it.
        self._committed = final_text
        self._sequence += 1
        self._emitted += 1
        return TranscriptUpdate(
            text=final_text,
            committed=final_text,
            unstable="",
            is_final=True,
            sequence=self._sequence,
            since_change_ms=(now - self._last_change) * 1000.0,
            elapsed_ms=(now - self._started) * 1000.0,
            degraded=degraded,
        )

    # -- introspection ------------------------------------------------------

    @property
    def text(self) -> str:
        return self._previous

    @property
    def committed(self) -> str:
        return self._committed

    @property
    def unstable(self) -> str:
        return self._previous[len(self._committed) :]

    @property
    def last_change_at(self) -> float:
        return self._last_change

    def since_change_ms(self, now: float | None = None) -> float:
        """How long the transcript has been sitting still."""

        reference = self._clock() if now is None else now
        return (reference - self._last_change) * 1000.0

    @property
    def emitted(self) -> int:
        return self._emitted

    @property
    def suppressed(self) -> int:
        return self._suppressed

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "committed": self.committed,
            "unstable": self.unstable,
            "emitted": self._emitted,
            "suppressed": self._suppressed,
            "since_change_ms": round(self.since_change_ms(), 1),
        }
