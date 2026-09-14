"""Candidate scoring.

Six normalised components, each in [0, 1]:

    latency          faster first token/audio is better
    quality          ordinal tier, with a citable provenance
    resource_safety  headroom left after this candidate's own footprint
    language_fit     exact / partial / no language support
    startup_cost     model load time or network round trip
    stability        historical reliability (defaults conservative, never 1.0
                     unless the provider has actually proven itself)

Weighted combination uses the policy weights from config/defaults.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from ..core.types import clamp01, normalize_language
from ..hardware.profile import HardwareProfile
from ..selection.benchmark import BenchmarkResult
from .candidates import Candidate

#: No provider starts with a perfect stability score; it must earn it.
DEFAULT_STABILITY = 0.80

#: Penalties for providers known to be fiddly. Empty until evidence exists --
#: being wrong here would be worse than being silent.
KNOWN_INSTABLE: dict[str, float] = {}


@dataclass
class ScoreBreakdown:
    latency: float = 0.0
    quality: float = 0.0
    resource_safety: float = 0.0
    language_fit: float = 0.0
    startup_cost: float = 0.0
    stability: float = DEFAULT_STABILITY
    total: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload = {k: round(v, 4) if isinstance(v, float) else v for k, v in payload.items()}
        return payload


def latency_score(result: BenchmarkResult, kind: str) -> float:
    """Reward low TTFT. Saturation points differ per stage."""

    ttft = result.ttft_ms.value if result.ttft_ms else None
    if ttft is None:
        return 0.5
    saturation = {"llm": 900.0, "tts": 1200.0, "asr": 1500.0, "vad": 50.0}.get(kind, 1200.0)
    return clamp01(1.0 - (ttft / saturation))


def quality_score(candidate: Candidate) -> tuple[float, str]:
    """Quality may only come from a declared source. Never invented."""

    descriptor = candidate.descriptor
    if descriptor.quality_score is not None:
        return clamp01(descriptor.quality_score / 100.0 if descriptor.quality_score > 1 else descriptor.quality_score), descriptor.quality_source.value
    # Fallback: ordinal tier mapped to the low half of the range.  Tier 5 is
    # therefore 1.0 but is clearly labelled as nothing more than a tier.
    return clamp01(max(0, min(descriptor.quality_tier, 5)) / 5.0), "quality_tier_ordinal"


def resource_safety_score(
    candidate: Candidate, profile: HardwareProfile, ram_budget_mb: int, vram_budget_mb: int
) -> float:
    """1.0 when the candidate fits comfortably, dropping sharply near the limit.

    Only *local* residency is scored. A remote/API stage consumes neither the
    local VRAM nor the local model RAM, so it must not be penalised for naming a
    device it does not actually occupy.
    """

    remote = candidate.device == "remote" or candidate.descriptor.requires_network
    if remote:
        needs_vram = 0
        needs_ram = 0
    else:
        needs_vram = candidate.descriptor.estimated_vram_mb if candidate.device != "cpu" else 0
        needs_ram = max(1, candidate.descriptor.estimated_ram_mb)

    vram_score = 1.0
    if needs_vram > 0:
        budget = vram_budget_mb if vram_budget_mb > 0 else 0
        if budget == 0:
            return 0.0
        ratio = needs_vram / budget
        if ratio > 1.0:
            return 0.0
        if ratio > 0.85:
            vram_score = clamp01((1.0 - ratio) / 0.15)
        else:
            vram_score = 1.0 - 0.25 * ratio

    ram_score = 1.0
    if needs_ram <= 0:
        # Nothing resident locally: no memory pressure to report.
        ram_score = 1.0
    elif ram_budget_mb > 0:
        ratio = needs_ram / ram_budget_mb
        if ratio > 1.0:
            ram_score = clamp01((1.0 - ratio) / 2.0)
        else:
            ram_score = 1.0 - 0.2 * ratio
    else:
        ram_score = 0.8 if needs_ram <= 4096 else 0.5

    return clamp01(min(vram_score, ram_score))


def language_fit_score(candidate: Candidate, language: str) -> float:
    supported = candidate.descriptor.languages
    if not supported or not language:
        return 1.0
    wanted = normalize_language(language)
    if any(normalize_language(item) == wanted for item in supported):
        return 1.0
    if any(item.startswith("*") for item in supported):
        return 0.7
    return 0.0


def startup_cost_score(result: BenchmarkResult) -> float:
    load = result.load_ms.value if result.load_ms else 0.0
    if load <= 0:
        return 1.0
    return clamp01(1.0 - (load / 6000.0))


def stability_score(candidate: Candidate) -> float:
    for provider_id, penalty in KNOWN_INSTABLE.items():
        if candidate.provider_id == provider_id:
            return clamp01(DEFAULT_STABILITY - penalty)
    return DEFAULT_STABILITY


def score_candidate(
    candidate: Candidate,
    result: BenchmarkResult,
    profile: HardwareProfile,
    *,
    weights: Mapping[str, float],
    language: str = "zh",
    ram_budget_mb: int = 0,
    vram_budget_mb: int = 0,
) -> ScoreBreakdown:
    """Weighted score. Returns every component so the UI can explain itself."""

    components = {
        "latency": latency_score(result, candidate.kind.value),
        "quality": quality_score(candidate)[0],
        "resource_safety": resource_safety_score(candidate, profile, ram_budget_mb, vram_budget_mb),
        "language_fit": language_fit_score(candidate, language),
        "startup_cost": startup_cost_score(result),
        "stability": stability_score(candidate),
    }
    total = sum(components[name] * weights.get(name, 0.0) for name in components)
    return ScoreBreakdown(
        total=round(clamp01(total), 4), weights=dict(weights), **components
    )


@dataclass
class ScoredCandidate:
    candidate: Candidate
    benchmark: BenchmarkResult
    score: ScoreBreakdown
    quality_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "score": self.score.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "quality_source": self.quality_source,
        }


def rank(scored: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """Deterministic ordering: total desc, then stable tie-breakers."""

    def key(item: ScoredCandidate) -> tuple:
        return (
            -item.score.total,
            -item.score.resource_safety,
            -item.score.latency,
            item.candidate.provider_id,
            item.candidate.model_id,
            item.candidate.device,
        )

    return sorted(scored, key=key)
