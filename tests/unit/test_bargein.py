"""Barge-in: the assistant stops talking because the user started.

The claims worth pinning here are the ones that are easy to get wrong and hard
to notice from the outside:

* **Hysteresis.** One 20 ms frame must not interrupt. Neither may a burst
  shorter than `min_speech_ms`, while a burst split by a breath must count as
  one utterance rather than two sub-threshold runs.
* **Honest absence.** With no usable VAD there is no way to detect speech, so
  the watcher must say it is unavailable instead of guessing from energy.
* **Once per turn.** A second `bargein.detected` for an interruption already
  reported is noise that would make any latency histogram meaningless.

The clock is injected *and* the timeline marks are written in that same clock,
because the cooldown window compares the two. Mixing a fake clock with
real-time timeline marks would place every playback start at the epoch, the
cooldown would never expire, and the test would pass while proving nothing.
"""

from __future__ import annotations

import asyncio
import struct

import pytest

from local_voice_companion.core.audio import AudioFrame
from local_voice_companion.core.bargein import (
    ARMED_STATES,
    BargeInConfig,
    BargeInWatcher,
)
from local_voice_companion.core.orchestrator import Pipeline, TurnRequest, run_turn
from local_voice_companion.core.session import Session
from local_voice_companion.core.state import TurnState
from local_voice_companion.providers.fake import FakeASR, FakeLLM, FakeTTS, FakeVAD

SAMPLE_RATE = 16000
FRAME_MS = 20


def _pcm(milliseconds: int = FRAME_MS, amplitude: int = 1000) -> bytes:
    count = int(SAMPLE_RATE * milliseconds / 1000)
    return b"".join(struct.pack("<h", amplitude) for _ in range(count))


def _frame() -> AudioFrame:
    return AudioFrame(pcm=_pcm(), sample_rate=SAMPLE_RATE, captured_at=0.0)


class _FakeClock:
    """A clock that moves only when a test says so."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, milliseconds: float) -> None:
        self.now += milliseconds / 1000.0


class ScriptedVAD(FakeVAD):
    """A VAD whose per-frame decisions are written down in advance."""

    def __init__(self, options=None, decisions=None) -> None:
        super().__init__(options or {})
        self.decisions: list[bool] = list(decisions or [])
        self.calls = 0

    def is_speech(self, frame) -> bool:  # noqa: ANN001 - provider contract
        self.calls += 1
        if not self.decisions:
            return False
        return self.decisions[min(self.calls - 1, len(self.decisions) - 1)]


class ExplodingVAD(FakeVAD):
    """A VAD that fails. The watcher must survive it and say so."""

    def is_speech(self, frame) -> bool:  # noqa: ANN001 - provider contract
        raise RuntimeError("vad backend crashed")


def _serving(vad: FakeVAD) -> FakeVAD:
    for state in ("AVAILABLE", "LOADING", "READY"):
        vad.lifecycle.transition(state)
    return vad


def _speaking(decisions, *, config=None, vad=None):
    """A session producing audio, with a VAD watching for interruption."""

    session = Session()
    session.set_state(TurnState.THINKING)
    session.set_state(TurnState.SPEAKING)
    timeline, _ = session.begin_turn()
    watcher = BargeInWatcher(
        session,
        vad if vad is not None else _serving(ScriptedVAD(decisions=decisions)),
        config or BargeInConfig(cooldown_ms=0),
        clock=_FakeClock(),
    )
    session.barge_in = watcher
    timeline.mark("playback_start", when=watcher.clock())
    return session, watcher, timeline


def _feed(watcher: BargeInWatcher, count: int, *, vad: ScriptedVAD | None = None) -> list:
    """Advance the clock one frame at a time and collect every decision."""

    decisions = []
    for _ in range(count):
        watcher.clock.advance(FRAME_MS)
        decisions.append(watcher.observe(_frame()))
    return decisions


def _events(session: Session, event_type: str) -> list[dict]:
    return [item for item in session.bus.after(0) if item["type"] == event_type]


class TestHysteresis:
    def test_a_single_frame_never_interrupts(self) -> None:
        session, watcher, timeline = _speaking([True])

        decision = watcher.observe(_frame())

        assert decision.triggered is False
        assert decision.reason == "below_threshold"
        assert _events(session, "bargein.detected") == []

    def test_speech_shorter_than_the_threshold_is_ignored(self) -> None:
        # 100 ms against a 180 ms threshold.
        session, watcher, timeline = _speaking([True] * 5)

        decisions = _feed(watcher, 5)

        assert [item.triggered for item in decisions] == [False] * 5
        assert decisions[-1].reason == "below_threshold"
        assert watcher.speech_ms == pytest.approx(100.0, abs=1e-6)

    def test_enough_continuous_speech_interrupts(self) -> None:
        session, watcher, timeline = _speaking([True] * 10)

        decisions = _feed(watcher, 10)
        fired = next((item for item in decisions if item.triggered), None)

        assert fired is not None, "sustained speech during playback must interrupt"
        assert fired.reason == "speech"
        assert fired.speech_ms >= BargeInConfig().min_speech_ms
        detected = _events(session, "bargein.detected")
        assert len(detected) == 1
        assert detected[0]["data"]["state"] == TurnState.SPEAKING.value
        assert "bargein_detected" in timeline.marks

    def test_a_breath_does_not_split_one_utterance_into_two(self) -> None:
        # 100 ms of speech, 80 ms of silence (shorter than `reset_silence_ms`),
        # then 100 ms more: together more than the threshold, so it must fire.
        # A pause inside a sentence is not the start of a new one.
        decisions = [True] * 5 + [False] * 4 + [True] * 5
        session, watcher, timeline = _speaking(decisions)

        fired = [item for item in _feed(watcher, len(decisions)) if item.triggered]

        assert fired, "a breath between two runs of speech must not reset them"
        assert timeline.marks.get("bargein_detected") is not None

    def test_a_real_gap_resets_the_accumulator(self) -> None:
        # Same speech, but the silence now exceeds `reset_silence_ms`.
        decisions = [True] * 5 + [False] * 15 + [True] * 5
        session, watcher, timeline = _speaking(decisions)

        fired = [item for item in _feed(watcher, len(decisions)) if item.triggered]

        assert fired == []
        assert timeline.marks.get("bargein_detected") is None


class TestGating:
    def test_silence_is_reported_as_silence(self) -> None:
        session, watcher, timeline = _speaking([False] * 4)

        decision = watcher.observe(_frame())

        assert decision.reason == "silence"
        assert watcher.speech_ms == 0.0

    def test_the_cooldown_suppresses_exactly_what_it_exists_for(self) -> None:
        session, watcher, timeline = _speaking([True] * 20)
        watcher.config.cooldown_ms = 350

        decisions = _feed(watcher, 8)  # 160 ms of speech, inside the window

        assert {item.reason for item in decisions} == {"cooldown"}
        assert timeline.marks.get("bargein_detected") is None

        watcher.clock.advance(400)
        assert watcher.observe(_frame()).triggered, "past the cooldown, speech must interrupt"

    def test_nothing_interrupts_while_the_assistant_is_not_speaking(self) -> None:
        quiet = (
            TurnState.IDLE,
            TurnState.LISTENING,
            TurnState.THINKING,
            TurnState.TRANSCRIBING,
        )
        for state in quiet:
            session = Session()
            session.state = state
            watcher = BargeInWatcher(
                session, _serving(ScriptedVAD(decisions=[True])), clock=_FakeClock()
            )

            fired = [item for item in _feed(watcher, 10) if item.triggered]

            assert fired == [], state
            assert watcher.armed is False, state

    def test_armed_states_are_the_output_states(self) -> None:
        assert TurnState.SPEAKING in ARMED_STATES
        assert TurnState.SYNTHESIZING in ARMED_STATES
        assert TurnState.LISTENING not in ARMED_STATES

    def test_an_interruption_fires_once_per_turn(self) -> None:
        session, watcher, timeline = _speaking([True] * 40)

        fired = len([item for item in _feed(watcher, 40) if item.triggered])

        assert fired == 1
        assert len(_events(session, "bargein.detected")) == 1


class TestHonestAbsence:
    def test_no_vad_means_no_barge_in_not_a_guess(self) -> None:
        session = Session()
        session.set_state(TurnState.THINKING)
        session.set_state(TurnState.SPEAKING)
        watcher = BargeInWatcher(session, None, clock=_FakeClock())

        assert watcher.available is False
        assert watcher.armed is False
        assert session.listening_for_barge_in is False
        assert [item for item in _feed(watcher, 10) if item.triggered] == []

    def test_a_vad_that_never_loaded_is_not_a_vad(self) -> None:
        session = Session()
        session.set_state(TurnState.THINKING)
        session.set_state(TurnState.SPEAKING)

        # Constructed but never transitioned to READY, like a load that failed.
        watcher = BargeInWatcher(session, ExplodingVAD({}), clock=_FakeClock())

        assert watcher.available is False

    def test_disabled_by_configuration_stays_disabled(self) -> None:
        session, watcher, timeline = _speaking(
            [True] * 20, config=BargeInConfig(enabled=False, cooldown_ms=0)
        )

        _feed(watcher, 20)

        assert timeline.marks.get("bargein_detected") is None
        assert watcher.armed is False

    def test_a_crashing_vad_reports_rather_than_killing_the_turn(self) -> None:
        session, watcher, timeline = _speaking([], vad=_serving(ExplodingVAD({})))

        decision = watcher.observe(_frame())

        assert decision.triggered is False
        assert watcher.config.enabled is False
        errors = _events(session, "error")
        assert errors and "barge-in VAD failed" in errors[-1]["data"]["message"]
        assert timeline.marks.get("bargein_detected") is None


class TestSessionRouting:
    def test_frames_are_still_examined_with_no_open_utterance(self) -> None:
        session, watcher, timeline = _speaking([True] * 12)

        async def scenario():
            accepted = []
            for _ in range(12):
                watcher.clock.advance(FRAME_MS)
                accepted.append(await session.push_audio(_pcm(), SAMPLE_RATE))
            return accepted

        accepted = asyncio.run(scenario())

        # Nobody was transcribing, but every frame still had a job: deciding
        # whether the user is talking over the assistant.
        assert accepted == [True] * 12
        assert len(_events(session, "bargein.detected")) == 1

    def test_a_frame_with_nothing_to_do_is_refused_not_buffered(self) -> None:
        session = Session()
        session.set_state(TurnState.LISTENING)

        assert asyncio.run(session.push_audio(_pcm(), SAMPLE_RATE)) is False

    def test_the_watcher_forgets_between_turns(self) -> None:
        session, watcher, timeline = _speaking([True] * 12)
        _feed(watcher, 12)

        session.finish_turn("cancelled", timeline)

        assert watcher.speech_ms == 0.0


class TestInterruptedTurn:
    """The point of the whole exercise: playback stops, and nothing leaks."""

    @staticmethod
    def _pipeline() -> Pipeline:
        pipeline = Pipeline(
            asr=FakeASR({}), llm=FakeLLM({"ttft_ms": 5, "delay_ms": 15}), tts=FakeTTS({})
        )
        for name in ("asr", "llm", "tts"):
            provider = getattr(pipeline, name)
            for state in ("AVAILABLE", "LOADING", "READY"):
                provider.lifecycle.transition(state)
        return pipeline

    @staticmethod
    async def _wait_for_playback(session: Session) -> bool:
        for _ in range(500):
            if any(item["type"] == "playback.started" for item in session.bus.after(0)):
                return True
            await asyncio.sleep(0.01)
        return False

    def test_cancelling_playback_reports_playback_stopped(self) -> None:
        session = Session()
        pipeline = self._pipeline()

        async def scenario():
            task = asyncio.create_task(
                run_turn(session, pipeline, TurnRequest(text="你好", speak=True))
            )
            # Waiting for playback matters: without it the assertion would be
            # vacuous, since nothing ever started playing.
            started = await self._wait_for_playback(session)
            session.cancel(detail="barge-in", code="barge_in")
            await task
            return started, task.result()

        started, result = asyncio.run(scenario())

        assert started, "the turn never reached playback"
        types = [item["type"] for item in session.bus.after(0)]
        assert "playback.stopped" in types, types
        assert "tts.audio" not in types[types.index("playback.stopped") :], (
            "stale audio leaked after playback was stopped"
        )
        assert "playback.finished" not in types
        assert result.cancelled is True
        assert "playback_stopped" in result.timeline.marks
        # A normal end and a cut short are different facts; both being present
        # would make the timeline ambiguous.
        assert "playback_end" not in result.timeline.marks

    def test_barge_in_latency_is_measured_when_a_detector_fired(self) -> None:
        session = Session()
        pipeline = self._pipeline()

        async def scenario():
            task = asyncio.create_task(
                run_turn(session, pipeline, TurnRequest(text="你好", speak=True))
            )
            await self._wait_for_playback(session)
            # Stand in for a detection that happened a moment ago. In real
            # operation the watcher marks this same stage, on this timeline.
            session.active_timeline.mark("bargein_detected")
            session.cancel(detail="barge-in", code="barge_in")
            await task
            return task.result()

        result = asyncio.run(scenario())

        assert result.timeline.barge_in_latency_ms is not None
        assert result.timeline.barge_in_latency_ms >= 0

    def test_a_superseded_turn_cannot_wipe_the_one_that_replaced_it(self) -> None:
        session = Session()
        timeline, token = session.begin_turn()
        token.cancel(detail="superseded", code="barge_in")
        # The replacement begins while the old turn is still unwinding -- which
        # is exactly what happens on barge-in.
        replacement, replacement_token = session.begin_turn()

        session.finish_turn("cancelled", timeline)

        assert session.active_timeline is replacement
        assert session.active_token is replacement_token
