"""Selection: constraints, candidates, scoring, planning and policies.

These tests exist specifically to pin down the defects found while wiring the
engine up, so none of them can silently come back:

* a streaming requirement applied to every kind silently deleted all ASR
  candidates;
* `latency_tier` was not applied to TTFT, making every provider in a kind score
  identically;
* a remote provider was charged local VRAM, which made the low-memory policy
  demote the wrong stage.
"""

from __future__ import annotations

import pytest

from local_voice_companion.core.types import Device, ProviderKind, SelectionPolicy
from local_voice_companion.hardware.profile import synthetic_profile
from local_voice_companion.providers.base import (
    ModelRef,
    ProviderDescriptor,
    ProviderHealth,
)
from local_voice_companion.providers.fake import FakeASR, FakeLLM, FakeTTS, FakeVAD
from local_voice_companion.selection.benchmark import (
    BenchmarkSource,
    estimate_metrics,
    is_stub,
    simulate_benchmark,
)
from local_voice_companion.selection.candidates import (
    Candidate,
    ConstraintSet,
    constraints_from,
    expand_devices,
    generate_candidates,
    take_rejections,
)
from local_voice_companion.selection.policies import resolve_policy, weights_for
from local_voice_companion.selection.scoring import (
    ScoredCandidate,
    quality_score,
    rank,
    resource_safety_score,
    score_candidate,
)
from local_voice_companion.selection.selector import (
    PipelineResourcePlanner,
    missing_stages,
)


def _descriptor(
    id: str,
    kind: ProviderKind,
    *,
    devices=(Device.CPU,),
    languages=("zh", "en"),
    latency_tier: int = 3,
    quality_tier: int = 3,
    ram_mb: int = 512,
    vram_mb: int = 0,
    disk_mb: int = 0,
    streaming: bool = True,
    requires_network: bool = False,
    models=(),
) -> ProviderDescriptor:
    return ProviderDescriptor(
        id=id,
        kind=kind,
        display_name=id,
        devices=list(devices),
        languages=list(languages),
        latency_tier=latency_tier,
        quality_tier=quality_tier,
        estimated_ram_mb=ram_mb,
        estimated_vram_mb=vram_mb,
        estimated_disk_mb=disk_mb,
        streaming=streaming,
        requires_network=requires_network,
        models=list(models),
    )


class TestConstraints:
    def test_rejects_a_profile_passed_as_policy(self, synthetic_profile) -> None:
        """Guards the exact argument-order bug that broke the engine."""

        with pytest.raises(TypeError, match="selection policy"):
            constraints_from(synthetic_profile, synthetic_profile)  # type: ignore[arg-type]

    def test_streaming_only_constrains_the_llm(self, synthetic_profile) -> None:
        """ASR is overwhelmingly non-streaming; requiring it deletes every ASR."""

        constraints = constraints_from(synthetic_profile, SelectionPolicy.BALANCED)
        assert constraints.require_streaming
        assert constraints.streaming_kinds == (ProviderKind.LLM,)
        assert ProviderKind.ASR not in constraints.streaming_kinds

    def test_non_streaming_asr_survives_balanced(self, synthetic_profile) -> None:
        descriptor = _descriptor("asr-x", ProviderKind.ASR, streaming=False)
        candidate = Candidate("asr-x", ProviderKind.ASR, "m", Device.CPU.value, descriptor)
        ok, _ = constraints_from(synthetic_profile, "balanced").allows(
            candidate, synthetic_profile
        )
        assert ok

    def test_non_streaming_llm_is_rejected_by_balanced(self, synthetic_profile) -> None:
        descriptor = _descriptor("llm-x", ProviderKind.LLM, streaming=False)
        candidate = Candidate("llm-x", ProviderKind.LLM, "m", Device.CPU.value, descriptor)
        ok, reason = constraints_from(synthetic_profile, "balanced").allows(
            candidate, synthetic_profile
        )
        assert not ok
        assert "streaming" in reason

    def test_cpu_only_policy_excludes_accelerators(self, synthetic_profile) -> None:
        constraints = constraints_from(synthetic_profile, "cpu_only")
        assert constraints.allowed_devices == [Device.CPU.value]

    def test_cpu_only_flag_beats_device_preferences(self, synthetic_profile) -> None:
        constraints = constraints_from(
            synthetic_profile, "balanced", cpu_only=True, prefer_devices=["cuda", "cpu"]
        )
        assert constraints.allowed_devices == [Device.CPU.value]

    def test_cpu_is_always_available(self, cpu_only_profile) -> None:
        constraints = constraints_from(cpu_only_profile, "balanced")
        assert Device.CPU.value in constraints.allowed_devices

    def test_language_mismatch_is_rejected(self, synthetic_profile) -> None:
        descriptor = _descriptor("asr-en", ProviderKind.ASR, languages=("en",))
        candidate = Candidate("asr-en", ProviderKind.ASR, "m", Device.CPU.value, descriptor)
        ok, reason = constraints_from(synthetic_profile, "balanced", language="zh").allows(
            candidate, synthetic_profile
        )
        assert not ok
        assert "language" in reason

    def test_network_can_be_disabled(self, synthetic_profile) -> None:
        """A network-only provider must not survive ``allow_network=False``.

        Uses an explicitly remote-capable *and* device-permitted candidate so the
        rejection is attributable to the network policy rather than to the device
        allow-list, which is checked first and would otherwise mask it.
        """

        descriptor = _descriptor(
            "llm-api", ProviderKind.LLM, devices=(Device.REMOTE,), requires_network=True
        )
        candidate = Candidate("llm-api", ProviderKind.LLM, "m", Device.REMOTE.value, descriptor)
        constraints = constraints_from(synthetic_profile, "balanced", allow_network=False)
        constraints.allowed_devices = [Device.REMOTE.value]
        ok, reason = constraints.allows(candidate, synthetic_profile)
        assert not ok
        assert "network" in reason

    def test_network_is_allowed_by_default(self, synthetic_profile) -> None:
        descriptor = _descriptor(
            "llm-api", ProviderKind.LLM, devices=(Device.REMOTE,), requires_network=True
        )
        candidate = Candidate("llm-api", ProviderKind.LLM, "m", Device.REMOTE.value, descriptor)
        constraints = constraints_from(synthetic_profile, "balanced")
        constraints.allowed_devices = [Device.REMOTE.value]
        ok, reason = constraints.allows(candidate, synthetic_profile)
        assert ok, reason


class TestCandidateGeneration:
    def test_expands_every_supported_device(self, synthetic_profile) -> None:
        descriptor = _descriptor("p", ProviderKind.LLM, devices=(Device.CUDA, Device.CPU))
        assert expand_devices(descriptor, ["cuda", "cpu"]) == ["cuda", "cpu"]

    def test_expand_always_includes_cpu(self, synthetic_profile) -> None:
        descriptor = _descriptor("p", ProviderKind.LLM, devices=(Device.CUDA, Device.CPU))
        assert "cpu" in expand_devices(descriptor, ["cuda"])

    def test_generates_one_candidate_per_model(self, synthetic_profile) -> None:
        descriptor = _descriptor(
            "asr",
            ProviderKind.ASR,
            models=[ModelRef(id="small"), ModelRef(id="large")],
        )
        constraints = constraints_from(synthetic_profile, "balanced")
        candidates = generate_candidates([descriptor], synthetic_profile, constraints)
        assert {c.model_id for c in candidates} == {"small", "large"}

    def test_rejections_are_recorded_with_reasons(self, synthetic_profile) -> None:
        descriptor = _descriptor(
            "llm-slow", ProviderKind.LLM, latency_tier=1, streaming=False
        )
        take_rejections()
        constraints = constraints_from(synthetic_profile, "balanced")
        generate_candidates([descriptor], synthetic_profile, constraints)
        rejections = take_rejections()
        assert rejections and "streaming" in rejections[0]["reason"]

    def test_fake_asr_reaches_the_candidate_pool(self, synthetic_profile) -> None:
        """Regression: this returned zero candidates before the streaming fix."""

        constraints = constraints_from(synthetic_profile, "balanced")
        candidates = generate_candidates(
            [FakeASR.descriptor()], synthetic_profile, constraints
        )
        assert len(candidates) >= 1

    def test_all_fake_stages_are_selectable(self, synthetic_profile) -> None:
        descriptors = [
            FakeASR.descriptor(),
            FakeLLM.descriptor(),
            FakeTTS.descriptor(),
            FakeVAD.descriptor(),
        ]
        constraints = constraints_from(synthetic_profile, "balanced")
        candidates = generate_candidates(descriptors, synthetic_profile, constraints)
        kinds = {c.kind for c in candidates}
        assert kinds == {
            ProviderKind.ASR,
            ProviderKind.LLM,
            ProviderKind.TTS,
            ProviderKind.VAD,
        }


class TestBenchmarkHonesty:
    def test_simulated_is_always_labelled(self, synthetic_profile) -> None:
        descriptor = _descriptor("asr", ProviderKind.ASR)
        candidate = Candidate("asr", ProviderKind.ASR, "m", Device.CPU.value, descriptor)
        result = simulate_benchmark(candidate, synthetic_profile)
        assert result.source is BenchmarkSource.SIMULATED
        assert result.to_dict()["source"] == "simulated"

    def test_stub_detection(self) -> None:
        assert is_stub("fake_asr")
        assert is_stub("test_llm")
        assert not is_stub("ollama_llm")
        assert not is_stub("faster_whisper_asr")

    def test_latency_tier_changes_ttft(self, synthetic_profile) -> None:
        """Regression: tiers were decorative, so every candidate tied on latency."""

        fast = _descriptor("llm", ProviderKind.LLM, latency_tier=5)
        slow = _descriptor("llm", ProviderKind.LLM, latency_tier=1)
        fast_c = Candidate("llm", ProviderKind.LLM, "m", Device.CPU.value, fast)
        slow_c = Candidate("llm", ProviderKind.LLM, "m", Device.CPU.value, slow)
        _, fast_ttft, _ = estimate_metrics(fast_c, synthetic_profile)
        _, slow_ttft, _ = estimate_metrics(slow_c, synthetic_profile)
        assert fast_ttft < slow_ttft

    def test_stubs_are_not_rewarded_for_being_cheap(self, synthetic_profile) -> None:
        """A test double must not out-rank a real engine by doing no work."""

        stub = FakeLLM.descriptor()
        assert is_stub(stub.id)
        stub_c = Candidate(stub.id, ProviderKind.LLM, "m", Device.CPU.value, stub)
        reference = _descriptor("llm", ProviderKind.LLM, latency_tier=3)
        ref_c = Candidate("llm", ProviderKind.LLM, "m", Device.CPU.value, reference)
        _, stub_ttft, _ = estimate_metrics(stub_c, synthetic_profile)
        _, ref_ttft, _ = estimate_metrics(ref_c, synthetic_profile)
        assert stub_ttft == pytest.approx(ref_ttft)

    def test_remote_stage_reserves_no_local_memory(self, synthetic_profile) -> None:
        """A remote LLM must not be reported as occupying the local GPU."""

        descriptor = _descriptor(
            "llm-api",
            ProviderKind.LLM,
            devices=(Device.REMOTE,),
            vram_mb=8192,
            ram_mb=4096,
            requires_network=True,
        )
        candidate = Candidate("llm-api", ProviderKind.LLM, "m", Device.REMOTE.value, descriptor)
        result = simulate_benchmark(candidate, synthetic_profile)
        assert result.vram_peak_mb.value == 0
        assert result.ram_peak_mb.value == 0

    def test_remote_stage_pays_no_load_cost(self, synthetic_profile) -> None:
        descriptor = _descriptor(
            "llm-api", ProviderKind.LLM, devices=(Device.REMOTE,), requires_network=True
        )
        candidate = Candidate("llm-api", ProviderKind.LLM, "m", Device.REMOTE.value, descriptor)
        load, _, _ = estimate_metrics(candidate, synthetic_profile)
        assert load == 0


class TestScoring:
    def _scored(self, descriptor, device=Device.CPU, policy="balanced", profile=None):
        profile = profile or synthetic_profile()
        candidate = Candidate(descriptor.id, descriptor.kind, "m", device.value, descriptor)
        benchmark = simulate_benchmark(candidate, profile)
        score = score_candidate(
            candidate,
            benchmark,
            profile,
            weights=weights_for(policy),
            language="zh",
            ram_budget_mb=14000,
            vram_budget_mb=7000,
        )
        return ScoredCandidate(
            candidate=candidate,
            benchmark=benchmark,
            score=score,
            quality_source=quality_score(candidate)[1],
        )

    @staticmethod
    def _labels(scored: list[ScoredCandidate]) -> list[str]:
        """Provider ids in rank order.

        ``Candidate.id`` is the composite ``"{provider}:{model}@{device}"`` used
        for de-duplication, so identity assertions must not compare it to a bare
        provider id.
        """

        return [item.candidate.provider_id for item in scored]

    def test_total_is_bounded(self, synthetic_profile) -> None:
        scored = self._scored(_descriptor("asr", ProviderKind.ASR), profile=synthetic_profile)
        assert 0.0 <= scored.score.total <= 1.0

    def test_cuda_beats_cpu_for_latency(self, synthetic_profile) -> None:
        descriptor = _descriptor(
            "llm", ProviderKind.LLM, devices=(Device.CUDA, Device.CPU), latency_tier=3
        )
        on_gpu = self._scored(descriptor, Device.CUDA, profile=synthetic_profile)
        on_cpu = self._scored(descriptor, Device.CPU, profile=synthetic_profile)
        assert on_gpu.score.latency > on_cpu.score.latency

    def test_remote_is_not_penalised_for_memory(self, synthetic_profile) -> None:
        """Regression: remote scored resource_safety 0.0 purely for naming a device."""

        remote = _descriptor(
            "llm-api",
            ProviderKind.LLM,
            devices=(Device.REMOTE,),
            vram_mb=8192,
            requires_network=True,
        )
        candidate = Candidate("llm-api", ProviderKind.LLM, "m", Device.REMOTE.value, remote)
        assert resource_safety_score(candidate, synthetic_profile, 14000, 7000) == pytest.approx(1.0)

    def test_oversized_model_is_penalised(self, synthetic_profile) -> None:
        huge = _descriptor("llm", ProviderKind.LLM, devices=(Device.CUDA,), vram_mb=99999)
        candidate = Candidate("llm", ProviderKind.LLM, "m", Device.CUDA.value, huge)
        assert resource_safety_score(candidate, synthetic_profile, 14000, 7000) == 0.0

    def test_language_fit(self, synthetic_profile) -> None:
        matching = self._scored(
            _descriptor("a", ProviderKind.ASR, languages=("zh",)), profile=synthetic_profile
        )
        other = self._scored(
            _descriptor("b", ProviderKind.ASR, languages=("de",)), profile=synthetic_profile
        )
        assert matching.score.language_fit == 1.0
        assert other.score.language_fit == 0.0

    def test_rank_is_deterministic(self, synthetic_profile) -> None:
        items = [
            self._scored(_descriptor(f"asr{i}", ProviderKind.ASR), profile=synthetic_profile)
            for i in range(4)
        ]
        assert [s.candidate.id for s in rank(items)] == [
            s.candidate.id for s in rank(list(reversed(items)))
        ]

    def test_higher_score_ranks_first(self, synthetic_profile) -> None:
        good = self._scored(
            _descriptor("good", ProviderKind.LLM, latency_tier=5, quality_tier=5),
            profile=synthetic_profile,
        )
        poor = self._scored(
            _descriptor("poor", ProviderKind.LLM, latency_tier=1, quality_tier=1),
            profile=synthetic_profile,
        )
        assert self._labels(rank([poor, good]))[0] == "good"


class TestPolicyWeights:
    def test_auto_resolves_to_something_concrete(self, synthetic_profile) -> None:
        effective, reason = resolve_policy("auto", synthetic_profile)
        assert effective is not SelectionPolicy.AUTO
        assert reason

    def test_low_memory_weights_resources_heavily(self) -> None:
        assert (
            weights_for("low_memory")["resource_safety"]
            > weights_for("balanced")["resource_safety"]
        )

    def test_quality_weights_quality_heavily(self) -> None:
        assert weights_for("quality")["quality"] > weights_for("balanced")["quality"]

    def test_ull_weights_latency_heavily(self) -> None:
        assert (
            weights_for("ultra_low_latency")["latency"] > weights_for("balanced")["latency"]
        )

    def test_every_policy_has_full_weights(self) -> None:
        expected = {
            "latency",
            "quality",
            "resource_safety",
            "language_fit",
            "startup_cost",
            "stability",
        }
        for policy in SelectionPolicy:
            assert set(weights_for(policy)) == expected


class TestPipelinePlanner:
    def _scored_for(self, descriptor, device, profile, policy="balanced"):
        candidate = Candidate(descriptor.id, descriptor.kind, "m", device.value, descriptor)
        benchmark = simulate_benchmark(candidate, profile)
        score = score_candidate(
            candidate,
            benchmark,
            profile,
            weights=weights_for(policy),
            language="zh",
            ram_budget_mb=14000,
            vram_budget_mb=7000,
        )
        return ScoredCandidate(
            candidate=candidate,
            benchmark=benchmark,
            score=score,
            quality_source="test",
        )

    def test_plans_all_three_required_stages(self, synthetic_profile) -> None:
        kinds = {
            "asr": [self._scored_for(_descriptor("asr", ProviderKind.ASR), Device.CPU, synthetic_profile)],
            "llm": [self._scored_for(_descriptor("llm", ProviderKind.LLM), Device.CPU, synthetic_profile)],
            "tts": [self._scored_for(_descriptor("tts", ProviderKind.TTS), Device.CPU, synthetic_profile)],
        }
        plan = PipelineResourcePlanner(14000, 7000).plan(kinds)
        assert plan.feasible
        assert missing_stages(plan) == []

    def test_demotes_to_fit_vram(self, synthetic_profile) -> None:
        """A GPU plan that busts VRAM must fall back to the CPU variant."""

        gpu = _descriptor(
            "llm", ProviderKind.LLM, devices=(Device.CUDA, Device.CPU), vram_mb=6000
        )
        cpu_asr = _descriptor("asr", ProviderKind.ASR, devices=(Device.CPU,))
        cpu_tts = _descriptor("tts", ProviderKind.TTS, devices=(Device.CPU,))
        kinds = {
            "asr": [self._scored_for(cpu_asr, Device.CPU, synthetic_profile)],
            "llm": [
                self._scored_for(gpu, Device.CUDA, synthetic_profile),
                self._scored_for(gpu, Device.CPU, synthetic_profile),
            ],
            "tts": [self._scored_for(cpu_tts, Device.CPU, synthetic_profile)],
        }
        plan = PipelineResourcePlanner(14000, 4000).plan(kinds)
        assert plan.feasible
        assert plan.assignments["llm"].candidate.device == Device.CPU.value
        assert plan.fits_budget

    def test_reports_missing_stage_distinctly_from_over_budget(self, synthetic_profile) -> None:
        """A missing stage and an over-budget plan must not share one message."""

        kinds = {
            "llm": [self._scored_for(_descriptor("llm", ProviderKind.LLM), Device.CPU, synthetic_profile)]
        }
        plan = PipelineResourcePlanner(14000, 7000).plan(kinds)
        assert not plan.feasible
        assert missing_stages(plan) == ["asr", "tts"]
        # The missing stages must be named, and no budget complaint invented.
        assert any("missing stage" in note for note in plan.notes)
        assert not any("budget" in note for note in plan.notes)

    def test_empty_kind_is_reported(self, synthetic_profile) -> None:
        kinds = {
            "asr": [],
            "llm": [self._scored_for(_descriptor("llm", ProviderKind.LLM), Device.CPU, synthetic_profile)],
            "tts": [self._scored_for(_descriptor("tts", ProviderKind.TTS), Device.CPU, synthetic_profile)],
        }
        plan = PipelineResourcePlanner(14000, 7000).plan(kinds)
        assert not plan.feasible
        assert any("no usable candidate for asr" in note for note in plan.notes)

    def test_notes_say_over_budget_when_that_is_the_cause(self, synthetic_profile) -> None:
        gpu = _descriptor("llm", ProviderKind.LLM, devices=(Device.CUDA,), vram_mb=99999)
        kinds = {
            "asr": [self._scored_for(_descriptor("asr", ProviderKind.ASR), Device.CPU, synthetic_profile)],
            "llm": [self._scored_for(gpu, Device.CUDA, synthetic_profile)],
            "tts": [self._scored_for(_descriptor("tts", ProviderKind.TTS), Device.CPU, synthetic_profile)],
        }
        plan = PipelineResourcePlanner(14000, 1000).plan(kinds)
        assert not plan.feasible
        assert not plan.fits_budget
        assert any("budget" in note for note in plan.notes)

    def test_remote_stage_does_not_consume_budget(self, synthetic_profile) -> None:
        remote = _descriptor(
            "llm-api",
            ProviderKind.LLM,
            devices=(Device.REMOTE,),
            vram_mb=8192,
            requires_network=True,
        )
        kinds = {
            "asr": [self._scored_for(_descriptor("asr", ProviderKind.ASR), Device.CPU, synthetic_profile)],
            "llm": [self._scored_for(remote, Device.REMOTE, synthetic_profile)],
            "tts": [self._scored_for(_descriptor("tts", ProviderKind.TTS), Device.CPU, synthetic_profile)],
        }
        plan = PipelineResourcePlanner(14000, 2000).plan(kinds)
        assert plan.feasible, plan.notes
        assert plan.footprint.vram_mb == 0

    def test_guard_against_infinite_loop(self, synthetic_profile) -> None:
        kinds = {
            "llm": [
                self._scored_for(
                    _descriptor("llm", ProviderKind.LLM, devices=(Device.CUDA,), vram_mb=90000),
                    Device.CUDA,
                    synthetic_profile,
                )
            ]
        }
        plan = PipelineResourcePlanner(14000, 100).plan(kinds)
        assert not plan.feasible
