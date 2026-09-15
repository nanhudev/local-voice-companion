"""Contract tests: the interfaces other code and clients depend on.

These tests are deliberately narrow and boring. They exist so that a refactor
cannot silently change a wire format, a state machine's legal moves, or the
shape of a provider's self-description -- the three things an external agent,
the Godot client or a stored bot manifest would break on.

A failure here is not a bug report about behaviour; it is a warning that a
public format changed and every consumer must be updated in the same commit.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.contract


# ---------------------------------------------------------------------------
# versioning
# ---------------------------------------------------------------------------


class TestVersioning:
    def test_schema_versions_are_integers(self) -> None:
        from local_voice_companion.core.events import EVENT_SCHEMA_VERSION
        from local_voice_companion.core.types import SCHEMA_VERSION

        assert isinstance(SCHEMA_VERSION, int) and SCHEMA_VERSION >= 1
        assert isinstance(EVENT_SCHEMA_VERSION, int) and EVENT_SCHEMA_VERSION >= 1

    def test_api_prefix_is_versioned(self) -> None:
        from local_voice_companion.core.types import API_PREFIX

        assert API_PREFIX == "/api/v1"


# ---------------------------------------------------------------------------
# provider descriptor
# ---------------------------------------------------------------------------


class TestProviderDescriptorContract:
    DESCRIPTOR_KEYS = {
        "id",
        "kind",
        "display_name",
        "languages",
        "devices",
        "streaming",
        "supports_streaming",
        "supports_partial_results",
        "estimated_ram_mb",
        "estimated_vram_mb",
        "estimated_disk_mb",
        "requires_network",
        "is_local",
        "quality_tier",
        "latency_tier",
        "quality_score",
        "quality_source",
        "supports_cancellation",
        "version",
        "tags",
        "models",
        "voices",
    }

    def test_serialised_keys_are_stable(self, isolated_registry) -> None:
        """Every provider must describe itself with exactly these keys.

        The selection engine reads this dict, and so does the UI. A provider
        that omits a field silently changes how it scores, so the key set is
        pinned rather than checked field by field at the call site.
        """

        for registration in isolated_registry:
            payload = registration.descriptor.to_dict()
            assert set(payload) == self.DESCRIPTOR_KEYS, registration.descriptor.id

    def test_lists_not_tuples_on_the_wire(self, isolated_registry) -> None:
        """JSON has no tuples. Serialisation must normalise them up front."""

        for registration in isolated_registry:
            payload = registration.descriptor.to_dict()
            for key in ("languages", "devices", "tags"):
                assert isinstance(payload[key], list), (registration.descriptor.id, key)
            assert isinstance(payload["kind"], str)
            assert isinstance(payload["quality_source"], str)

    def test_kind_and_quality_source_are_valid_enums(self, isolated_registry) -> None:
        from local_voice_companion.core.types import ProviderKind, QualitySource

        for registration in isolated_registry:
            payload = registration.descriptor.to_dict()
            assert ProviderKind(payload["kind"]) is registration.descriptor.kind
            assert QualitySource(payload["quality_source"]) is registration.descriptor.quality_source

    def test_quality_is_never_invented(self, isolated_registry) -> None:
        """An unmeasured provider must say so instead of guessing a number."""

        from local_voice_companion.core.types import QualitySource

        for registration in isolated_registry:
            descriptor = registration.descriptor
            if descriptor.quality_source is QualitySource.UNKNOWN:
                assert descriptor.quality_score is None, descriptor.id


# ---------------------------------------------------------------------------
# provider lifecycle
# ---------------------------------------------------------------------------


class TestProviderLifecycleContract:
    def test_serving_states(self) -> None:
        from local_voice_companion.core.lifecycle import SERVING_STATES, ProviderState

        assert SERVING_STATES == {
            ProviderState.READY,
            ProviderState.BUSY,
            ProviderState.DEGRADED,
        }
        # DEGRADED must serve: a CPU fallback that refuses requests is worse
        # than one that answers slowly.
        assert ProviderState.DEGRADED in SERVING_STATES

    def test_loading_is_only_reachable_from_available(self) -> None:
        """The legal path into LOADING is what keeps discovery honest.

        If DISCOVERED could jump straight to LOADING, a provider could load a
        model without ever having been probed, and a missing backend would
        surface as a crash mid-turn instead of a clean "unavailable".
        """

        from local_voice_companion.core.lifecycle import TRANSITIONS, ProviderState

        sources = {
            state for state, targets in TRANSITIONS.items() if ProviderState.LOADING in targets
        }
        assert sources == {ProviderState.AVAILABLE, ProviderState.ERROR}

    def test_illegal_transition_raises(self) -> None:
        from local_voice_companion.core.lifecycle import InvalidTransition, Lifecycle, ProviderState

        lifecycle = Lifecycle(provider_id="p-test")
        with pytest.raises(InvalidTransition):
            lifecycle.transition(ProviderState.READY)

    def test_happy_path_transitions(self) -> None:
        from local_voice_companion.core.lifecycle import Lifecycle, ProviderState

        lifecycle = Lifecycle(provider_id="p-test")
        for state in (
            ProviderState.AVAILABLE,
            ProviderState.LOADING,
            ProviderState.READY,
            ProviderState.BUSY,
            ProviderState.READY,
            ProviderState.UNLOADING,
            ProviderState.AVAILABLE,
        ):
            lifecycle.transition(state)
        assert lifecycle.state is ProviderState.AVAILABLE
        assert not lifecycle.is_serving

    def test_self_transition_is_allowed_and_idempotent(self) -> None:
        from local_voice_companion.core.lifecycle import Lifecycle, ProviderState

        lifecycle = Lifecycle(provider_id="p-test")
        lifecycle.transition(ProviderState.AVAILABLE)
        lifecycle.transition(ProviderState.AVAILABLE)
        assert lifecycle.state is ProviderState.AVAILABLE

    def test_to_dict_shape(self) -> None:
        from local_voice_companion.core.lifecycle import Lifecycle

        payload = Lifecycle(provider_id="p-test").to_dict()
        assert set(payload) == {"provider_id", "state", "detail", "serving"}


# ---------------------------------------------------------------------------
# turn state machine
# ---------------------------------------------------------------------------


class TestTurnStateContract:
    def test_state_names_are_stable(self) -> None:
        from local_voice_companion.core.state import TurnState

        assert {item.name for item in TurnState} == {
            "IDLE",
            "LISTENING",
            "CAPTURING",
            "TRANSCRIBING",
            "THINKING",
            "SYNTHESIZING",
            "SPEAKING",
            "CANCELLING",
            "PAUSED",
            "ERROR",
        }

    def test_every_state_is_reachable_from_idle(self) -> None:
        """No state may be an orphan: an unreachable state is dead weight."""

        from local_voice_companion.core.state import TURN_TRANSITIONS, TurnState

        reachable = set()
        frontier = [TurnState.IDLE]
        while frontier:
            state = frontier.pop()
            for target in TURN_TRANSITIONS.get(state, set()):
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        missing = {item for item in TurnState if item not in reachable and item is not TurnState.IDLE}
        assert not missing, f"unreachable turn states: {sorted(item.name for item in missing)}"

    def test_cancelling_can_return_to_idle(self) -> None:
        """Barge-in must be able to finish, not get stuck mid-cancel."""

        from local_voice_companion.core.state import TURN_TRANSITIONS, TurnState

        assert TurnState.IDLE in TURN_TRANSITIONS[TurnState.CANCELLING]

    def test_speaking_can_be_interrupted(self) -> None:
        from local_voice_companion.core.state import TURN_TRANSITIONS, TurnState

        assert TurnState.CANCELLING in TURN_TRANSITIONS[TurnState.SPEAKING]
        assert TurnState.LISTENING in TURN_TRANSITIONS[TurnState.SPEAKING]

    def test_illegal_transition_rejected(self) -> None:
        from local_voice_companion.core.state import TurnState, can_transition

        assert not can_transition(TurnState.IDLE, TurnState.SPEAKING)


# ---------------------------------------------------------------------------
# event wire format
# ---------------------------------------------------------------------------


class TestEventWireContract:
    WIRE_KEYS = {"v", "id", "type", "ts", "session_id", "turn_id", "data"}

    def test_event_to_wire_shape(self) -> None:
        from local_voice_companion.core.events import Event

        wire = Event(type="llm.delta", data={"text": "x"}, session_id="s1", turn_id="s1-0").to_wire()
        assert set(wire) == self.WIRE_KEYS

    def test_wire_is_json_serialisable(self) -> None:
        from local_voice_companion.core.events import Event

        wire = Event(type="turn.completed", data={"status": "completed", "nested": {"a": [1, 2]}}).to_wire()
        assert json.loads(json.dumps(wire)) == wire

    def test_all_event_types_are_namespaced(self) -> None:
        """Every type is either `domain.action` or a bare top-level domain.

        `error` is the single deliberate exception: it is the catch-all bucket
        every client must handle unconditionally, so prefixing it would invite
        clients to filter it out by domain and miss fatal frames.
        """

        from local_voice_companion.core.events import ALL_EVENT_TYPES

        exceptions = {"error"}
        for event_type in ALL_EVENT_TYPES:
            if event_type in exceptions:
                continue
            assert "." in event_type, event_type
            domain, _, action = event_type.partition(".")
            assert domain and action, event_type

    def test_required_event_types_present(self) -> None:
        from local_voice_companion.core.events import ALL_EVENT_TYPES

        required = {
            "runtime.ready",
            "runtime.metric",
            "session.opened",
            "turn.started",
            "turn.state",
            "turn.completed",
            "turn.cancelled",
            "asr.partial",
            "asr.final",
            "llm.started",
            "llm.delta",
            "llm.completed",
            "tts.started",
            "tts.audio",
            "tts.completed",
            "playback.started",
            "playback.finished",
            "provider.state",
            "error",
        }
        assert required <= set(ALL_EVENT_TYPES), required - set(ALL_EVENT_TYPES)

    def test_completed_timeline_wire_shape(self) -> None:
        """`turn.completed` carries the timeline; clients chart it directly."""

        from local_voice_companion.core.events import TurnTimeline

        timeline = TurnTimeline(turn_id="s1-0", session_id="s1")
        payload = timeline.to_dict()
        assert set(payload) == {
            "turn_id",
            "session_id",
            "stages",
            "asr_latency_ms",
            "asr_ttfp_ms",
            "llm_ttft_ms",
            "tts_ttfa_ms",
            "time_to_first_audio_ms",
            "total_turn_ms",
            "clock_limited",
        }

    def test_timeline_stage_names_are_stable(self) -> None:
        from local_voice_companion.core.events import TIMELINE_STAGES

        assert set(TIMELINE_STAGES) == {
            "turn_started",
            "vad_end",
            "asr_start",
            "asr_first_partial",
            "asr_end",
            "llm_start",
            "llm_first_token",
            "llm_end",
            "tts_start",
            "tts_first_audio",
            "playback_start",
            "playback_end",
        }

    def test_unknown_stage_is_rejected(self) -> None:
        from local_voice_companion.core.events import TurnTimeline

        with pytest.raises(KeyError):
            TurnTimeline(turn_id="t").mark("made_up_stage")


class TestEventBusContract:
    def test_subscribers_receive_wire_dicts(self) -> None:
        """Subscribers get the serialised form, not the internal dataclass.

        Handing out the live `Event` would let a subscriber mutate runtime
        state, and would force every consumer to know the internal class.
        """

        from local_voice_companion.core.events import EventBus

        bus = EventBus()
        received: list[object] = []
        bus.subscribe(received.append)
        bus.emit("turn.started", session_id="s1", turn_id="s1-0", data={"n": 1})

        assert len(received) == 1
        assert isinstance(received[0], dict)
        assert received[0]["type"] == "turn.started"
        assert received[0]["v"] >= 1

    def test_buffer_is_bounded(self) -> None:
        """A disconnected client must not be able to grow memory forever."""

        from local_voice_companion.core.events import EventBus

        bus = EventBus(capacity=8)
        for index in range(50):
            bus.emit("runtime.metric", data={"n": index})
        assert len(bus.after(0)) == 8

    def test_after_returns_only_newer_events(self) -> None:
        from local_voice_companion.core.events import EventBus

        bus = EventBus()
        for index in range(5):
            bus.emit("runtime.metric", data={"n": index})
        events = bus.after(0)
        assert [item["id"] for item in events] == [1, 2, 3, 4, 5]
        assert [item["id"] for item in bus.after(3)] == [4, 5]

    def test_a_dead_subscriber_does_not_break_fanout(self) -> None:
        from local_voice_companion.core.events import EventBus

        bus = EventBus()
        good: list[dict] = []

        def boom(_event: dict) -> None:
            raise RuntimeError("subscriber exploded")

        bus.subscribe(boom)
        bus.subscribe(good.append)
        bus.emit("runtime.metric", data={"n": 1})
        assert len(good) == 1


# ---------------------------------------------------------------------------
# selection decision + plan
# ---------------------------------------------------------------------------


class TestSelectionWireContract:
    DECISION_KEYS = {
        "fingerprint",
        "requested_policy",
        "effective_policy",
        "policy_reason",
        "plan",
        "candidates_considered",
        "rejected",
        "alternatives",
    }

    def test_recommendation_shape(self) -> None:
        from local_voice_companion.core.types import ProviderKind, SelectionPolicy
        from local_voice_companion.hardware.profile import synthetic_profile
        from local_voice_companion.providers.fake import FAKE_PROVIDERS
        from local_voice_companion.selection.engine import recommend_from_descriptors

        profile = synthetic_profile(cpu_threads=8, ram_mb=8192, vram_mb=0)
        descriptors = [cls.descriptor() for cls in FAKE_PROVIDERS]
        decision = _run(recommend_from_descriptors(profile, descriptors, policy="balanced"))
        assert set(decision.to_dict()) == self.DECISION_KEYS
        assert ProviderKind(decision.plan.assignments["llm"].candidate.kind) is ProviderKind.LLM
        assert SelectionPolicy(decision.effective_policy) is SelectionPolicy.BALANCED

    def test_alternatives_are_omitted_on_request(self) -> None:
        """The full candidate ranking is large; the API drops it by default."""

        from local_voice_companion.hardware.profile import synthetic_profile
        from local_voice_companion.providers.fake import FAKE_PROVIDERS
        from local_voice_companion.selection.engine import recommend_from_descriptors

        profile = synthetic_profile(cpu_threads=8, ram_mb=8192, vram_mb=0)
        descriptors = [cls.descriptor() for cls in FAKE_PROVIDERS]
        decision = _run(recommend_from_descriptors(profile, descriptors))
        trimmed = decision.to_dict(include_alternatives=False)
        assert "alternatives" not in trimmed
        assert set(trimmed) == self.DECISION_KEYS - {"alternatives"}

    def test_plan_shape(self) -> None:
        from local_voice_companion.hardware.profile import synthetic_profile
        from local_voice_companion.providers.fake import FAKE_PROVIDERS
        from local_voice_companion.selection.engine import recommend_from_descriptors

        profile = synthetic_profile(cpu_threads=8, ram_mb=8192, vram_mb=0)
        descriptors = [cls.descriptor() for cls in FAKE_PROVIDERS]
        decision = _run(recommend_from_descriptors(profile, descriptors))
        payload = decision.plan.to_dict()
        assert set(payload) == {"feasible", "assignments", "footprint", "budget", "notes"}
        assert set(payload["footprint"]) == {"ram_mb", "vram_mb"}
        assert set(payload["budget"]) == {"ram_mb", "vram_mb"}

    def test_plan_footprint_respects_budget(self) -> None:
        """A normal plan must fit. An over-budget one must say `feasible: False`."""

        from local_voice_companion.hardware.profile import synthetic_profile
        from local_voice_companion.providers.fake import FAKE_PROVIDERS
        from local_voice_companion.selection.engine import recommend_from_descriptors

        profile = synthetic_profile(cpu_threads=8, ram_mb=8192, vram_mb=0)
        descriptors = [cls.descriptor() for cls in FAKE_PROVIDERS]
        decision = _run(recommend_from_descriptors(profile, descriptors))
        plan = decision.plan
        if plan.feasible:
            assert plan.footprint.ram_mb <= plan.budget.ram_mb
            assert plan.footprint.vram_mb <= plan.budget.vram_mb


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# bot manifest
# ---------------------------------------------------------------------------


class TestBotManifestContract:
    MANIFEST_KEYS = {
        "schema_version",
        "id",
        "name",
        "persona",
        "language",
        "asr",
        "llm",
        "tts",
        "runtime",
        "conversation",
        "template",
        "tags",
        "created_at",
        "updated_at",
    }

    def test_manifest_shape(self) -> None:
        from local_voice_companion.bots.schema import BotManifest

        payload = BotManifest.new(id="c-bot", name="契约").to_dict()
        assert set(payload) == self.MANIFEST_KEYS

    def test_stage_binding_defaults_to_auto(self) -> None:
        """A manifest is portable: unset stages must resolve on the target host.

        Pinning a concrete provider id in the default would make a bot created
        on a GPU machine unusable on a laptop. `explicit` is a derived property,
        so it is not part of the serialised form -- it is asserted through the
        model, which is where consumers actually read it.
        """

        from local_voice_companion.bots.schema import BotManifest

        manifest = BotManifest.new(id="c-bot", name="契约")
        payload = manifest.to_dict()
        for stage in ("asr", "llm", "tts"):
            assert payload[stage]["provider"] == "auto", stage
            assert payload[stage]["model"] == "auto", stage
            assert getattr(manifest, stage).explicit is False, stage

    def test_explicit_binding_is_detected(self) -> None:
        from local_voice_companion.bots.schema import BotManifest

        manifest = BotManifest.new(id="c-bot", name="契约")
        manifest.llm.provider = "ollama"
        assert manifest.llm.explicit is True
        assert manifest.asr.explicit is False

    def test_manifest_never_contains_host_paths(self) -> None:
        from local_voice_companion.bots.schema import BotManifest

        serialised = json.dumps(BotManifest.new(id="c-bot", name="契约").to_dict())
        for marker in (":\\", "C:/", "/home/", "/Users/", "/usr/"):
            assert marker not in serialised, marker

    def test_round_trip_is_lossless(self) -> None:
        from local_voice_companion.bots.schema import BotManifest

        original = BotManifest.new(id="c-bot", name="契约", language="en", voice="v-1")
        restored = BotManifest.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_unknown_field_is_rejected(self) -> None:
        """A typo in a hand-edited manifest must fail loudly, not silently."""

        from pydantic import ValidationError

        from local_voice_companion.bots.schema import BotManifest

        document = BotManifest.new(id="c-bot", name="契约").to_dict()
        document["persoan"] = {"system_prompt": "typo"}
        with pytest.raises(ValidationError):
            BotManifest.from_dict(document)

    def test_invalid_id_is_rejected(self) -> None:
        """Ids become filenames, so separators and odd characters must not pass."""

        from pydantic import ValidationError

        from local_voice_companion.bots.schema import BotManifest

        for bad in ("-leading", "_leading", "has space", "a", "x" * 70, "中文", "a/b", "a\\b", "a.b"):
            with pytest.raises(ValidationError):
                BotManifest.new(id=bad, name="契约")

    def test_id_is_normalised_to_lowercase(self) -> None:
        """`Bot 1` and `bot 1` must not become two different bots."""

        from local_voice_companion.bots.schema import BotManifest

        assert BotManifest.new(id="  C-Bot  ", name="契约").id == "c-bot"

    def test_schema_version_is_current(self) -> None:
        from local_voice_companion.bots.schema import BotManifest

        assert BotManifest.new(id="c-bot", name="契约").schema_version == 2


# ---------------------------------------------------------------------------
# error wire format
# ---------------------------------------------------------------------------


class TestErrorWireContract:
    def test_error_envelope(self) -> None:
        from local_voice_companion.core.errors import LVCError, NotFound

        error = NotFound("bot not found", bot_id="x")
        wire = error.to_wire()
        assert set(wire) == {"error"}
        assert set(wire["error"]) == {"code", "message", "retryable", "context"}
        assert wire["error"]["code"]
        assert wire["error"]["message"] == "bot not found"
        assert wire["error"]["retryable"] is False
        assert wire["error"]["context"] == {"bot_id": "x"}
        assert error.http_status == 404

    def test_error_codes_are_snake_case(self) -> None:
        from local_voice_companion.core import errors as errors_module

        for name in dir(errors_module):
            obj = getattr(errors_module, name)
            if isinstance(obj, type) and issubclass(obj, errors_module.LVCError) and obj is not errors_module.LVCError:
                instance = _construct(obj)
                assert instance.code.islower(), (name, instance.code)
                assert " " not in instance.code, (name, instance.code)

    def test_http_statuses_are_sane(self) -> None:
        from local_voice_companion.core.errors import LVCError, NotFound, ValidationFailed

        assert 400 <= ValidationFailed("x").http_status < 500
        assert 400 <= NotFound("x").http_status < 500
        assert LVCError("x").http_status >= 400


def _construct(cls):
    """Instantiate an LVCError subclass without knowing its signature."""

    try:
        return cls("probe message")
    except TypeError:
        return cls()
