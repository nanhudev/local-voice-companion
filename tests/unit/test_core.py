"""Core primitives: lifecycle, turn state, cancellation, events, timeline."""

from __future__ import annotations

import asyncio

import pytest

from local_voice_companion.core.cancellation import CancellationToken
from local_voice_companion.core.errors import CancelledTurn, LVCError
from local_voice_companion.core.events import EventBus, TurnTimeline
from local_voice_companion.core.lifecycle import (
    InvalidTransition,
    Lifecycle,
    ProviderState,
)
from local_voice_companion.core.state import TurnState, can_transition
from local_voice_companion.core.types import normalize_language


class TestLifecycle:
    def test_starts_discovered(self) -> None:
        assert Lifecycle("p").state is ProviderState.DISCOVERED

    def test_happy_path(self) -> None:
        life = Lifecycle("p")
        life.transition(ProviderState.AVAILABLE)
        life.transition(ProviderState.LOADING)
        life.transition(ProviderState.READY)
        assert life.is_serving

    def test_discovered_cannot_jump_to_loading(self) -> None:
        """A provider must be probed before it can be loaded."""

        with pytest.raises(InvalidTransition):
            Lifecycle("p").transition(ProviderState.LOADING)

    def test_degraded_still_serves(self) -> None:
        """A CPU fallback must keep answering rather than 500."""

        life = Lifecycle("p")
        for state in (ProviderState.AVAILABLE, ProviderState.LOADING, ProviderState.DEGRADED):
            life.transition(state)
        assert life.is_serving

    def test_self_transition_is_allowed(self) -> None:
        life = Lifecycle("p")
        life.transition(ProviderState.AVAILABLE)
        life.transition(ProviderState.AVAILABLE)
        assert life.state is ProviderState.AVAILABLE


class TestTurnState:
    def test_text_turn_may_start_from_idle(self) -> None:
        """A text-only turn has nothing to capture; IDLE -> THINKING is legal."""

        assert can_transition(TurnState.IDLE, TurnState.THINKING)

    def test_second_text_turn_from_listening(self) -> None:
        """After turn 1 the session sits in LISTENING; turn 2 must still work."""

        assert can_transition(TurnState.LISTENING, TurnState.THINKING)

    def test_prerecorded_audio_skips_capture(self) -> None:
        assert can_transition(TurnState.IDLE, TurnState.TRANSCRIBING)

    def test_voice_flow_is_still_walkable(self) -> None:
        path = [
            TurnState.IDLE,
            TurnState.LISTENING,
            TurnState.CAPTURING,
            TurnState.TRANSCRIBING,
            TurnState.THINKING,
            TurnState.SYNTHESIZING,
            TurnState.SPEAKING,
            TurnState.LISTENING,
        ]
        for current, following in zip(path, path[1:]):
            assert can_transition(current, following), f"{current} -> {following}"

    def test_every_state_can_recover(self) -> None:
        """No state may be a dead end: a session must always be able to reset."""

        for state in TurnState:
            assert can_transition(state, TurnState.IDLE) or can_transition(
                state, TurnState.LISTENING
            ), f"{state} cannot return to a resting state"

    def test_arbitrary_jump_is_rejected(self) -> None:
        assert not can_transition(TurnState.CAPTURING, TurnState.SPEAKING)


class TestCancellationToken:
    def test_starts_uncancelled(self) -> None:
        token = CancellationToken()
        assert not token.cancelled
        assert token.reason is None

    def test_cancel_is_idempotent(self) -> None:
        """A barge-in may fire several times; the first reason must win."""

        token = CancellationToken()
        token.cancel(detail="first", code="barge_in")
        token.cancel(detail="second", code="other")
        assert token.cancelled
        assert token.reason is not None
        assert token.reason.detail == "first"

    def test_raise_if_cancelled(self) -> None:
        token = CancellationToken()
        token.raise_if_cancelled()
        token.cancel()
        with pytest.raises(CancelledTurn):
            token.raise_if_cancelled()

    def test_reason_is_exposed_on_the_wire(self) -> None:
        token = CancellationToken()
        token.cancel(detail="user spoke", code="barge_in")
        assert token.to_dict()["reason"]["detail"] == "user spoke"

    def test_sleep_wakes_on_cancel(self) -> None:
        """A cancelled turn must not block for the full sleep duration."""

        token = CancellationToken()
        token.cancel()
        with pytest.raises(CancelledTurn):
            asyncio.run(token.sleep(5.0))


class TestEventBus:
    def test_ids_are_monotonic(self) -> None:
        bus = EventBus()
        first = bus.emit("a")
        second = bus.emit("b")
        assert second.id > first.id

    def test_after_returns_only_newer(self) -> None:
        bus = EventBus()
        bus.emit("a")
        marker = bus.last_id
        bus.emit("b")
        assert [e["type"] for e in bus.after(marker)] == ["b"]

    def test_ring_buffer_is_bounded(self) -> None:
        """An unbounded event log would grow without limit on a long call."""

        bus = EventBus(capacity=8)
        for index in range(50):
            bus.emit("evt", data={"n": index})
        assert len(bus.after(0)) <= 8
        assert bus.last_id == 50

    def test_subscriber_receives_wire_format(self) -> None:
        bus = EventBus()
        seen: list[dict] = []
        bus.subscribe(seen.append)
        bus.emit("x", data={"k": 1}, session_id="s")
        assert seen[0]["type"] == "x"
        assert seen[0]["data"] == {"k": 1}

    def test_dead_subscriber_is_dropped(self) -> None:
        """One broken consumer must not poison every later publish."""

        bus = EventBus()

        def explode(_event: dict) -> None:
            raise RuntimeError("consumer died")

        bus.subscribe(explode)
        bus.emit("before")
        bus.emit("after")  # must not raise

    def test_unsubscribe_stops_delivery(self) -> None:
        bus = EventBus()
        seen: list[dict] = []
        bus.subscribe(seen.append)
        bus.emit("a")
        bus.unsubscribe(seen.append)
        bus.emit("b")
        assert len(seen) == 1

    def test_wire_format_is_stable(self) -> None:
        bus = EventBus()
        wire = bus.emit("turn.started", data={"k": "v"}, session_id="s", turn_id="t").to_wire()
        assert set(wire) == {"v", "id", "type", "ts", "session_id", "turn_id", "data"}


class TestTurnTimeline:
    def test_unknown_stage_is_rejected(self) -> None:
        with pytest.raises(KeyError):
            TurnTimeline("t").mark("not_a_stage")

    def test_first_mark_wins(self) -> None:
        timeline = TurnTimeline("t")
        timeline.mark("llm_start", 100.0)
        timeline.mark("llm_start", 999.0)
        assert timeline.marks["llm_start"] == 100.0

    def test_derives_llm_ttft(self) -> None:
        timeline = TurnTimeline("t")
        timeline.mark("llm_start", 1.0)
        timeline.mark("llm_first_token", 1.25)
        assert timeline.llm_ttft_ms == 250

    def test_derives_headline_metric(self) -> None:
        timeline = TurnTimeline("t")
        timeline.mark("vad_end", 2.0)
        timeline.mark("tts_first_audio", 2.4)
        assert timeline.time_to_first_audio_ms == 400

    def test_missing_marks_stay_none(self) -> None:
        assert TurnTimeline("t").llm_ttft_ms is None

    def test_total_falls_back_when_no_audio_played(self) -> None:
        """A text-only turn never reaches playback_end but must still be timed."""

        timeline = TurnTimeline("t")
        timeline.mark("turn_started", 0.0)
        timeline.mark("llm_end", 0.5)
        assert timeline.total_turn_ms == 500

    def test_clock_limited_flag(self) -> None:
        """Sub-millisecond turns must be flagged, never reported as '0 ms instant'."""

        timeline = TurnTimeline("t")
        timeline.mark("turn_started", 0.0)
        timeline.mark("llm_end", 0.0002)
        assert timeline.clock_limited is True

    def test_metrics_omits_none(self) -> None:
        timeline = TurnTimeline("t")
        timeline.mark("llm_start", 0.0)
        timeline.mark("llm_first_token", 0.1)
        assert set(timeline.metrics()) == {"llm_ttft"}


class TestErrors:
    def test_wire_shape(self) -> None:
        wire = LVCError("boom", provider="p").to_wire()
        assert wire["error"]["code"] == LVCError.code
        assert wire["error"]["message"] == "boom"
        assert set(wire["error"]) == {"code", "message", "retryable", "context"}

    def test_subclass_codes_are_distinct(self) -> None:
        from local_voice_companion.core.errors import NotFound, ProviderUnavailable

        assert ProviderUnavailable.code != NotFound.code
        assert ProviderUnavailable("x", provider="p").to_wire()["error"]["retryable"] is True

    def test_details_are_preserved(self) -> None:
        wire = LVCError("boom", provider="p", state="READY").to_wire()
        assert wire["error"]["context"]["provider"] == "p"

    def test_http_status_is_carried(self) -> None:
        from local_voice_companion.core.errors import ValidationFailed

        assert ValidationFailed("bad").http_status == 422


class TestLanguage:
    @pytest.mark.parametrize(
        "raw,expected",
        [("zh-CN", "zh"), ("ZH", "zh"), ("en_US", "en"), ("ja", "ja")],
    )
    def test_normalises(self, raw: str, expected: str) -> None:
        assert normalize_language(raw) == expected
