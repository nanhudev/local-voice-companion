"""The selection engine: hardware + policy + providers -> executable plan.

Nothing in this module knows any provider by name. It only reads descriptors,
benchmarks and candidates.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..config.schema import SystemConfig
from ..core.types import ProviderKind
from ..hardware.profile import HardwareProfile
from ..providers.base import ProviderDescriptor
from ..providers.discovery import discover
from ..providers.registry import ProviderRegistry, registry
from .benchmark import BenchmarkCache, simulate_benchmark
from .candidates import (
    Candidate,
    _record_rejection,
    collecting_rejections,
    constraints_from,
    generate_candidates,
    take_rejections,
)
from .policies import resolve_policy, weights_for
from .scoring import ScoredCandidate, rank, score_candidate
from .selector import PipelineResourcePlanner, SelectionDecision

KIND_ORDER: tuple[ProviderKind, ...] = (
    ProviderKind.ASR,
    ProviderKind.LLM,
    ProviderKind.TTS,
    ProviderKind.VAD,
)

REQUIRED_KINDS: tuple[ProviderKind, ...] = (ProviderKind.ASR, ProviderKind.LLM, ProviderKind.TTS)


def _policy_value(policy: Any) -> str:
    """Accept a SelectionPolicy or a plain string without assuming which.

    Config is validated, but the engine is also called directly from tests and
    from the CLI with literals, so it must not blow up on a bare string.
    """

    if isinstance(policy, str):
        return policy
    value = getattr(policy, "value", None)
    return value if isinstance(value, str) else str(policy)


def _apply_manual_overrides(
    candidates: Sequence[Candidate],
    overrides: Mapping[str, Any],
    language: str,
) -> Sequence[Candidate]:
    """Manual pinning wins outright; the selector never scores against it.

    This is the only place user pinning is consulted, and it lives in the
    selector rather than in core or in any provider.
    """

    pinned: list[Candidate] = []
    for kind in KIND_ORDER:
        kind_candidates = [item for item in candidates if item.kind is kind]
        if not kind_candidates:
            continue
        pin = None
        for candidate in kind_candidates:
            override = overrides.get(candidate.provider_id)
            if override is not None and not getattr(override, "enabled", True):
                continue
            if override is not None and (override.model or override.voice or override.device):
                pin = candidate
                break
        if pin is not None:
            pinned.append(pin)
        else:
            pinned.extend(kind_candidates)
    return pinned


async def recommend(
    profile: HardwareProfile,
    config: SystemConfig,
    *,
    reg: ProviderRegistry = registry,
    cache: BenchmarkCache | None = None,
    language: str = "zh",
    policy_override: str | None = None,
) -> SelectionDecision:
    """Produce a complete, explainable pipeline plan."""

    cache = cache or BenchmarkCache()
    runtime_policy = config.runtime

    requested = policy_override or _policy_value(runtime_policy.policy)
    effective, policy_reason = resolve_policy(
        requested, profile, cpu_only=runtime_policy.cpu_only
    )
    weights = weights_for(effective)

    constraints = constraints_from(
        profile,
        effective,
        language=language,
        prefer_devices=runtime_policy.prefer_devices,
        allow_network_llm=runtime_policy.allow_network_llm,
        cpu_only=runtime_policy.cpu_only,
        reserve_vram_mb=runtime_policy.vram_reserve_mb,
        reserve_ram_mb=runtime_policy.ram_reserve_mb,
    )

    probe_results = await discover(reg, options_by_id=_provider_options(config, reg))
    available = {item.provider_id for item in probe_results if item.available}
    probe_detail = {
        item.provider_id: item.detail for item in probe_results if not item.available
    }

    # Candidates come from *every* registered provider, not only the reachable
    # ones. Filtering by availability first has a subtle and harmful effect: a
    # provider that is merely down (Voicebox stopped, Ollama not running) never
    # reaches the constraint check, so it is absent from the plan with no
    # recorded reason -- and the user cannot tell "not installed" from
    # "excluded by policy". Generating from the full set lets each provider earn
    # an explicit rejection instead of a silent omission.
    descriptors = [reg.registration(provider_id).descriptor for provider_id in reg.ids()]

    fingerprint = profile.fingerprint()

    # Everything from candidate generation onwards belongs to *this* decision,
    # so its rejections are collected as a unit. Draining at the end instead
    # loses the explanation whenever anything in between touches the log.
    with collecting_rejections() as rejections:
        candidates = generate_candidates(descriptors, profile, constraints, kinds=KIND_ORDER)
        candidates = list(_apply_manual_overrides(candidates, config.providers, language))

        # A provider can pass the constraint check and still not be installable.
        # Record that here so the plan distinguishes "down right now" from
        # "rejected by policy"; both reasons are user-facing.
        for candidate in candidates:
            if candidate.provider_id in available:
                continue
            _record_rejection(
                candidate,
                "provider is not reachable right now: "
                + (probe_detail.get(candidate.provider_id) or "probe reported unavailable"),
            )

        ram_budget = constraints.max_ram_mb
        vram_budget = constraints.max_vram_mb

        ranked_by_kind: dict[str, list[ScoredCandidate]] = {}
        considered = 0
        for kind in KIND_ORDER:
            scored_items: list[ScoredCandidate] = []
            for candidate in candidates:
                if candidate.kind is not kind:
                    continue
                if candidate.provider_id not in available:
                    # It already has its rejection recorded above; scoring it
                    # would let a dead service win the plan.
                    continue
                considered += 1
                result = cache.get(fingerprint, candidate) or simulate_benchmark(candidate, profile)
                if cache.get(fingerprint, candidate) is None:
                    cache.put(fingerprint, candidate, result)
                breakdown = score_candidate(
                    candidate,
                    result,
                    profile,
                    weights=weights,
                    language=language,
                    ram_budget_mb=ram_budget,
                    vram_budget_mb=vram_budget,
                )
                scored_items.append(
                    ScoredCandidate(
                        candidate=candidate,
                        benchmark=result,
                        score=breakdown,
                        quality_source=candidate.descriptor.quality_source.value
                        if candidate.descriptor.quality_score is not None
                        else "quality_tier_ordinal",
                    )
                )
            if scored_items:
                ranked_by_kind[kind.value] = rank(scored_items)

        planner = PipelineResourcePlanner(
            ram_budget_mb=ram_budget, vram_budget_mb=vram_budget
        )
        plan = planner.plan(ranked_by_kind)
        missing = [kind.value for kind in REQUIRED_KINDS if kind.value not in plan.assignments]
        if missing:
            plan.feasible = False
            plan.notes.append(
                "no available provider for: "
                + ", ".join(missing)
                + "; install a local provider or configure an API provider"
            )

    cache.flush()

    return SelectionDecision(
        profile_fingerprint=fingerprint,
        requested_policy=str(requested),
        effective_policy=effective.value,
        policy_reason=policy_reason,
        plan=plan,
        candidates_considered=considered,
        rejected=list(rejections),
        alternatives={
            kind: [item.to_dict() for item in options[:4]]
            for kind, options in ranked_by_kind.items()
        },
    )


def _provider_options(config: SystemConfig, reg: ProviderRegistry) -> dict[str, dict[str, Any]]:
    """Provider options come from config; secrets stay as env references."""

    options: dict[str, dict[str, Any]] = {}
    for provider_id in reg.ids():
        override = config.providers.get(provider_id)
        if override is None:
            continue
        payload: dict[str, Any] = dict(override.options or {})
        if override.base_url:
            payload["base_url"] = override.base_url
        if override.model:
            payload["model"] = override.model
        if override.voice:
            payload["voice"] = override.voice
        if override.device:
            payload["device"] = override.device
        if payload:
            options[provider_id] = payload
    return options


async def recommend_from_descriptors(
    profile: HardwareProfile,
    descriptors: Sequence[ProviderDescriptor],
    *,
    language: str = "zh",
    policy: str = "balanced",
    allow_network: bool = True,
    cpu_only: bool = False,
) -> SelectionDecision:
    """Descriptor-driven recommendation, primarily for tests and the API."""

    from ..config.schema import RuntimePolicyConfig, SystemConfig

    runtime = RuntimePolicyConfig(policy=policy, allow_network_llm=allow_network, cpu_only=cpu_only)
    config = SystemConfig(runtime=runtime)
    effective, reason = resolve_policy(policy, profile, cpu_only=cpu_only)
    weights = weights_for(effective)
    constraints = constraints_from(
        profile, effective, language=language, allow_network_llm=allow_network, cpu_only=cpu_only
    )
    with collecting_rejections() as rejections:
        candidates = generate_candidates(descriptors, profile, constraints, kinds=KIND_ORDER)

        ranked_by_kind: dict[str, list[ScoredCandidate]] = {}
        considered = 0
        for kind in KIND_ORDER:
            scored_items: list[ScoredCandidate] = []
            for candidate in candidates:
                if candidate.kind is not kind:
                    continue
                considered += 1
                result = simulate_benchmark(candidate, profile)
                breakdown = score_candidate(
                    candidate,
                    result,
                    profile,
                    weights=weights,
                    language=language,
                    ram_budget_mb=constraints.max_ram_mb,
                    vram_budget_mb=constraints.max_vram_mb,
                )
                scored_items.append(
                    ScoredCandidate(
                        candidate=candidate,
                        benchmark=result,
                        score=breakdown,
                        quality_source=candidate.descriptor.quality_source.value,
                    )
                )
            if scored_items:
                ranked_by_kind[kind.value] = rank(scored_items)

        planner = PipelineResourcePlanner(
            ram_budget_mb=constraints.max_ram_mb, vram_budget_mb=constraints.max_vram_mb
        )
        plan = planner.plan(ranked_by_kind)

    return SelectionDecision(
        profile_fingerprint=profile.fingerprint(),
        requested_policy=str(policy),
        effective_policy=effective.value,
        policy_reason=reason,
        plan=plan,
        candidates_considered=considered,
        rejected=list(rejections),
        alternatives={
            kind: [item.to_dict() for item in options[:4]] for kind, options in ranked_by_kind.items()
        },
    )
