"""Selection decision + pipeline resource planning.

The interesting part is not "which provider is fastest", it is
"can ASR, LLM and TTS coexist on this machine at the same time?".

    GPU 8 GB, LLM needs 5.5 GB, ASR 2 GB, TTS 2 GB
    -> each fits alone, together they do not.

The planner therefore demotes whole candidates (or just their device) until the
resident set fits, always choosing the demotion that costs the least score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..core.types import Device, ProviderKind
from ..hardware.profile import HardwareProfile
from .benchmark import BenchmarkResult
from .candidates import Candidate
from .scoring import ScoredCandidate


@dataclass
class ResourceFootprint:
    ram_mb: int = 0
    vram_mb: int = 0

    def add(self, ram: int = 0, vram: int = 0) -> None:
        self.ram_mb += ram
        self.vram_mb += vram


@dataclass
class PipelineAssignment:
    kind: ProviderKind
    candidate: Candidate
    benchmark: BenchmarkResult
    score: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            **self.candidate.to_dict(),
            "score": round(self.score, 4),
            "reason": self.reason,
            "benchmark": self.benchmark.to_dict(),
        }


@dataclass
class PipelinePlan:
    assignments: dict[str, PipelineAssignment] = field(default_factory=dict)
    footprint: ResourceFootprint = field(default_factory=ResourceFootprint)
    budget: ResourceFootprint = field(default_factory=ResourceFootprint)
    notes: list[str] = field(default_factory=list)
    feasible: bool = True

    def assignment(self, kind: str | ProviderKind) -> PipelineAssignment | None:
        key = kind.value if isinstance(kind, ProviderKind) else kind
        return self.assignments.get(key)

    def primary_device(self, kind: str | ProviderKind) -> str:
        found = self.assignment(kind)
        return found.candidate.device if found else Device.CPU.value

    @property
    def vram_mb(self) -> int:
        return self.footprint.vram_mb

    @property
    def ram_mb(self) -> int:
        return self.footprint.ram_mb

    @property
    def fits_budget(self) -> bool:
        """Memory-only verdict. Says nothing about completeness."""

        if self.budget.vram_mb and self.footprint.vram_mb > self.budget.vram_mb:
            return False
        if self.budget.ram_mb and self.footprint.ram_mb > self.budget.ram_mb:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "assignments": {key: value.to_dict() for key, value in sorted(self.assignments.items())},
            "footprint": {"ram_mb": self.footprint.ram_mb, "vram_mb": self.footprint.vram_mb},
            "budget": {"ram_mb": self.budget.ram_mb, "vram_mb": self.budget.vram_mb},
            "notes": list(self.notes),
        }


def _footprint_of(scored: ScoredCandidate) -> tuple[int, int]:
    """Local residency of a stage: (ram_mb, vram_mb).

    A remote stage runs in someone else's process, so it costs no local VRAM
    and no local model RAM. Charging it here is what makes a pipeline planner
    believe an API-hosted LLM exhausted the GPU.
    """

    candidate = scored.candidate
    if candidate.device == Device.REMOTE.value or candidate.descriptor.requires_network:
        return 0, 0
    vram = candidate.descriptor.estimated_vram_mb if candidate.device != Device.CPU.value else 0
    return candidate.descriptor.estimated_ram_mb, vram


@dataclass
class SelectionDecision:
    profile_fingerprint: str
    requested_policy: str
    effective_policy: str
    policy_reason: str
    plan: PipelinePlan
    candidates_considered: int = 0
    rejected: list[dict[str, str]] = field(default_factory=list)
    alternatives: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.plan.feasible

    def to_dict(self, include_alternatives: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "fingerprint": self.profile_fingerprint,
            "requested_policy": self.requested_policy,
            "effective_policy": self.effective_policy,
            "policy_reason": self.policy_reason,
            "plan": self.plan.to_dict(),
            "candidates_considered": self.candidates_considered,
            "rejected": self.rejected,
        }
        if include_alternatives:
            payload["alternatives"] = self.alternatives
        return payload


class PipelineResourcePlanner:
    """Fit three residents into one budget without over-committing.

    Strategy: start from the best-scoring candidate per kind, then repeatedly
    demote the option with the best score-per-MB-freed until everything fits.
    Demotion first tries a cheaper device (GPU -> CPU) and then a lower-ranked
    alternative for the same kind.
    """

    def __init__(self, ram_budget_mb: int, vram_budget_mb: int) -> None:
        self.ram_budget = max(0, ram_budget_mb)
        self.vram_budget = max(0, vram_budget_mb)

    # -- public API ---------------------------------------------------------

    def plan(self, ranked_by_kind: Mapping[str, Sequence[ScoredCandidate]]) -> PipelinePlan:
        notes: list[str] = []
        chosen: dict[str, ScoredCandidate] = {}
        for kind, options in ranked_by_kind.items():
            if not options:
                notes.append(f"no usable candidate for {kind}")
                continue
            chosen[kind] = options[0]

        plan = self._build(chosen, notes)
        guard = 0
        while not plan.fits_budget and guard < 64:
            guard += 1
            movable = self._demotion_options(chosen, ranked_by_kind)
            if not movable:
                break
            kind, replacement, note = movable
            chosen[kind] = replacement
            plan = self._build(chosen, notes + [note])

        # `feasible` means "this plan can actually serve a turn": every required
        # stage present AND inside the memory budget. Reporting feasible=True for
        # a plan with no ASR would let the API claim readiness it cannot back up.
        if not plan.fits_budget:
            plan.notes.append(
                "no configuration fits the available VRAM/RAM budget; "
                "consider an API LLM or a lower-memory policy"
            )
        missing = missing_stages(plan)
        if missing:
            plan.notes.append(
                f"incomplete plan, missing stage(s): {', '.join(missing)}"
            )
        plan.feasible = plan.fits_budget and not missing and bool(plan.assignments)
        return plan

    # -- internals ----------------------------------------------------------

    def _build(
        self, chosen: Mapping[str, ScoredCandidate], notes: Sequence[str]
    ) -> PipelinePlan:
        assignments: dict[str, PipelineAssignment] = {}
        footprint = ResourceFootprint()
        for kind, scored in chosen.items():
            ram, vram = _footprint_of(scored)
            footprint.add(ram, vram)
            assignments[kind] = PipelineAssignment(
                kind=ProviderKind(kind),
                candidate=scored.candidate,
                benchmark=scored.benchmark,
                score=scored.score.total,
                reason=self._reason_for(scored),
            )
        return PipelinePlan(
            assignments=assignments,
            footprint=footprint,
            budget=ResourceFootprint(ram_mb=self.ram_budget, vram_mb=self.vram_budget),
            notes=list(notes),
            # Provisional: `plan()` decides the real verdict once it knows
            # whether the loop has settled. Never trusted by a caller.
            feasible=self._fits_from(footprint),
        )

    def _fits(self, plan: PipelinePlan) -> bool:
        return plan.fits_budget

    def _fits_from(self, footprint: ResourceFootprint) -> bool:
        if self.vram_budget and footprint.vram_mb > self.vram_budget:
            return False
        if self.ram_budget and footprint.ram_mb > self.ram_budget:
            return False
        return True

    def _demotion_options(
        self,
        chosen: Mapping[str, ScoredCandidate],
        ranked_by_kind: Mapping[str, Sequence[ScoredCandidate]],
    ):
        """Pick the single change that frees the most memory per score lost."""

        best = None
        for kind, scored in chosen.items():
            options = list(ranked_by_kind.get(kind, ()))

            # Option A: same provider/model, cheaper device.
            current_ram, current_vram = _footprint_of(scored)
            for replacement in options:
                replacement_ram, replacement_vram = _footprint_of(replacement)
                freed_vram = current_vram - replacement_vram
                freed_ram = current_ram - replacement_ram
                freed = freed_vram + freed_ram / 4
                lost = scored.score.total - replacement.score.total
                if freed <= 0:
                    continue
                efficiency = freed / max(lost, 0.001)
                note = self._demotion_note(kind, scored, replacement)
                candidate_entry = (efficiency, lost, kind, replacement, note)
                if best is None or (efficiency, -lost) > (best[0], -best[1]):
                    best = candidate_entry
        if best is None:
            return None
        _efficiency, _lost, kind, replacement, note = best
        if replacement.candidate.key == chosen[kind].candidate.key:
            return None
        return kind, replacement, note

    @staticmethod
    def _demotion_note(kind: str, old: ScoredCandidate, new: ScoredCandidate) -> str:
        old_device = old.candidate.device
        new_device = new.candidate.device
        if old.candidate.provider_id == new.candidate.provider_id and old.candidate.model_id == new.candidate.model_id:
            return (
                f"{kind}: moved {old.candidate.provider_id} from {old_device} to {new_device} "
                f"to stay inside the memory budget (score {old.score.total:.3f} -> {new.score.total:.3f})"
            )
        if old_device != new_device:
            return (
                f"{kind}: switched from {old.candidate.provider_id}@{old_device} to "
                f"{new.candidate.provider_id}@{new_device} to stay inside the memory budget"
            )
        return (
            f"{kind}: downgraded model from {old.candidate.model_id} to {new.candidate.model_id} "
            f"to stay inside the memory budget"
        )

    @staticmethod
    def _reason_for(scored: ScoredCandidate) -> str:
        breakdown = scored.score
        strongest = max(
            (
                (value, name)
                for name, value in (
                    ("latency", breakdown.latency),
                    ("quality", breakdown.quality),
                    ("resource_safety", breakdown.resource_safety),
                    ("language_fit", breakdown.language_fit),
                    ("startup_cost", breakdown.startup_cost),
                )
            ),
            default=(0.0, "stability"),
        )
        return f"best weighted score ({breakdown.total:.3f}), strongest component: {strongest[1]}"


def summarise_plan(plan: PipelinePlan) -> dict[str, Any]:
    return {
        "stages": {
            key: {
                "provider": value.candidate.provider_id,
                "model": value.candidate.model_id,
                "device": value.candidate.device,
            }
            for key, value in sorted(plan.assignments.items())
        },
        "estimated_vram_mb": plan.footprint.vram_mb,
        "estimated_ram_mb": plan.footprint.ram_mb,
        "feasible": plan.feasible,
    }


def missing_stages(plan: PipelinePlan, required: Iterable[str] = ("asr", "llm", "tts")) -> list[str]:
    return [stage for stage in required if stage not in plan.assignments]
