"""Candidate generation and constraint filtering.

Pipeline:  descriptor -> candidate -> constraint check -> benchmark -> score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..core.types import (
    Device,
    Device,
    ProviderKind,
    SelectionPolicy,
    normalize_language,
)
from ..hardware.profile import HardwareProfile
from ..providers.base import ProviderDescriptor

ALL_KINDS: tuple[ProviderKind, ...] = (
    ProviderKind.ASR,
    ProviderKind.LLM,
    ProviderKind.TTS,
    ProviderKind.VAD,
)


@dataclass(frozen=True)
class Candidate:
    provider_id: str
    kind: ProviderKind
    model_id: str
    device: str
    descriptor: ProviderDescriptor = field(compare=False, repr=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.provider_id, self.model_id, self.device)

    @property
    def id(self) -> str:
        return f"{self.provider_id}:{self.model_id}@{self.device}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider_id": self.provider_id,
            "kind": self.kind.value,
            "model_id": self.model_id,
            "device": self.device,
        }


@dataclass
class ConstraintSet:
    """Everything a candidate must satisfy before it is even scored."""

    allowed_devices: list[str] = field(default_factory=lambda: [Device.CPU.value])
    max_vram_mb: int = 0
    max_ram_mb: int = 0
    language: str = "zh"
    require_local: bool = False
    allow_network: bool = True
    require_streaming: bool = False
    # When non-empty, `require_streaming` only applies to these kinds.
    streaming_kinds: tuple[ProviderKind, ...] = ()
    budget_total_vram_mb: int = 0

    def allows(
        self, candidate: Candidate, profile: HardwareProfile
    ) -> tuple[bool, str]:
        """Return (ok, reason). Reasons are surfaced to the UI verbatim."""

        if candidate.device not in self.allowed_devices:
            return False, f"device {candidate.device} not allowed by policy/hardware"

        if candidate.device != Device.CPU.value and not profile.supports_device(candidate.device):
            return False, f"device {candidate.device} not present on this machine"

        descriptor = candidate.descriptor
        if self.require_local and descriptor.requires_network and not self.allow_network:
            return False, "network providers are disabled"

        if descriptor.requires_network and not self.allow_network:
            return False, "requires network access, which is disabled"

        if self.language and descriptor.languages:
            wanted = normalize_language(self.language)
            supported = {normalize_language(item) for item in descriptor.languages}
            if wanted not in supported:
                return False, f"does not support language {self.language}"

        if self.require_streaming and not descriptor.streaming:
            if not self.streaming_kinds or candidate.kind in self.streaming_kinds:
                return False, "streaming required but not supported"

        return True, "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_devices": list(self.allowed_devices),
            "max_vram_mb": self.max_vram_mb,
            "max_ram_mb": self.max_ram_mb,
            "language": self.language,
            "require_local": self.require_local,
            "allow_network": self.allow_network,
            "require_streaming": self.require_streaming,
            "streaming_kinds": [kind.value for kind in self.streaming_kinds],
        }


def budget_for(profile: HardwareProfile, reserve_vram_mb: int, reserve_ram_mb: int) -> tuple[int, int]:
    """Usable VRAM/RAM after leaving headroom for the OS and display."""

    vram = 0
    if profile.total_vram_mb:
        usable = profile.best_vram_available_mb or profile.total_vram_mb
        vram = max(0, usable - reserve_vram_mb)
    ram = max(0, profile.memory.total_mb - reserve_ram_mb) if profile.memory.total_mb else 4 * 1024
    return vram, ram


def constraints_from(
    profile: HardwareProfile,
    policy: SelectionPolicy | str = SelectionPolicy.BALANCED,
    *,
    language: str = "zh",
    prefer_devices: Sequence[str] | None = None,
    allow_network_llm: bool = True,
    cpu_only: bool = False,
    reserve_vram_mb: int = 512,
    reserve_ram_mb: int = 1024,
    require_streaming: bool | None = None,
    streaming_kinds_override: Sequence[ProviderKind] | None = None,
    allow_network: bool | None = None,
) -> ConstraintSet:
    """Derive the participation rules from hardware + user intent.

    ``policy`` and ``language`` are the only positional-ish inputs; everything
    else is keyword-only and has a default, so a caller can never accidentally
    pass a ``HardwareProfile`` where a policy belongs.
    """

    if isinstance(policy, HardwareProfile):
        raise TypeError(
            "constraints_from(profile, policy, ...) -- the second argument is the "
            "selection policy, not a hardware profile"
        )

    effective_policy: SelectionPolicy = policy
    if not isinstance(policy, SelectionPolicy):
        try:
            effective_policy = SelectionPolicy(policy)
        except ValueError as exc:
            raise ValueError(f"unknown selection policy: {policy!r}") from exc

    vram, ram = budget_for(profile, reserve_vram_mb, reserve_ram_mb)

    if cpu_only or effective_policy is SelectionPolicy.CPU_ONLY:
        allowed = [Device.CPU.value]
    else:
        requested = list(prefer_devices or ["cuda", "directml", "metal", "cpu"])
        allowed = [device for device in requested if device == Device.CPU.value or profile.supports_device(device)]
        if Device.CPU.value not in allowed:
            allowed.append(Device.CPU.value)

    # Only the LLM needs token streaming: it is what makes "first audio before
    # the full answer is generated" possible at all. ASR streaming is optional
    # (partial hypotheses) and TTS streaming is optional (progressive playback);
    # requiring it everywhere silently rejects every non-streaming ASR, which is
    # most of them.
    if require_streaming is None:
        require_streaming = effective_policy in {
            SelectionPolicy.ULTRA_LOW_LATENCY,
            SelectionPolicy.BALANCED,
        }

    streaming_kinds: tuple[ProviderKind, ...] = ()
    if require_streaming:
        streaming_kinds = (
            (ProviderKind.LLM,) if streaming_kinds_override is None else tuple(streaming_kinds_override)
        )

    network_ok = allow_network_llm if allow_network is None else allow_network
    return ConstraintSet(
        allowed_devices=allowed,
        max_vram_mb=vram,
        max_ram_mb=ram,
        language=language,
        allow_network=network_ok,
        require_streaming=bool(require_streaming),
        streaming_kinds=streaming_kinds,
        budget_total_vram_mb=vram,
    )


def expand_devices(descriptor: ProviderDescriptor, allowed: Sequence[str]) -> list[str]:
    """Every device this descriptor can actually use on this machine."""

    devices = [device.value for device in descriptor.devices]
    ordered = [device for device in allowed if device in devices]
    if Device.CPU.value in devices and Device.CPU.value not in ordered:
        ordered.append(Device.CPU.value)
    return ordered


def generate_candidates(
    descriptors: Iterable[ProviderDescriptor],
    profile: HardwareProfile,
    constraints: ConstraintSet,
    kinds: Sequence[ProviderKind] = ALL_KINDS,
) -> list[Candidate]:
    """Turn descriptors into (provider x model x device) candidates."""

    candidates: list[Candidate] = []
    for descriptor in descriptors:
        if descriptor.kind not in kinds:
            continue
        models = descriptor.models or (descriptor.id,)
        for device in expand_devices(descriptor, constraints.allowed_devices):
            for model in models:
                model_id = model.id if hasattr(model, "id") else str(model)
                candidate = Candidate(
                    provider_id=descriptor.id,
                    kind=descriptor.kind,
                    model_id=model_id,
                    device=device,
                    descriptor=descriptor,
                )
                ok, reason = constraints.allows(candidate, profile)
                if ok:
                    candidates.append(candidate)
                else:
                    _record_rejection(candidate, reason)
    return candidates


_REJECTIONS: list[dict[str, str]] = []


def _record_rejection(candidate: Candidate, reason: str) -> None:
    _REJECTIONS.append({"candidate": candidate.id, "reason": reason})


def take_rejections() -> list[dict[str, str]]:
    """Drain the rejection log (used to explain decisions in the UI)."""

    snapshot = list(_REJECTIONS)
    _REJECTIONS.clear()
    return snapshot
