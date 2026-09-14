"""Async provider discovery.

`discover()` probes every registered provider exactly once, records the outcome
in the provider's own lifecycle, and returns the descriptors that can actually
be loaded. Probing must be cheap and must never raise out of this module.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..core.lifecycle import ProviderState
from ..core.types import ProviderKind
from .base import BaseProvider, ProviderDescriptor, ProviderHealth
from .registry import ProviderRegistry, registry


@dataclass
class ProbeResult:
    provider_id: str
    kind: ProviderKind
    available: bool
    state: str
    detail: str = ""
    latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "kind": self.kind.value,
            "available": self.available,
            "state": self.state,
            "detail": self.detail,
            "latency_ms": self.latency_ms,
        }


def _instantiate(registration_options: Mapping[str, Any] | None, cls: type[BaseProvider]) -> BaseProvider:
    options = dict(registration_options or {})
    return cls(options)


async def probe_provider(
    cls: type[BaseProvider], options: Mapping[str, Any] | None = None
) -> ProbeResult:
    """Probe one provider instance. Errors are reported, not raised."""

    descriptor = cls.descriptor()
    provider = _instantiate(options, cls)
    started = time.monotonic()
    try:
        health: ProviderHealth = await provider.probe()
        available = bool(health.ok)
        state = (
            ProviderState.AVAILABLE.value if available and provider.lifecycle.state is ProviderState.DISCOVERED
            else provider.lifecycle.state.value
        )
        if not available:
            provider.lifecycle.transition(ProviderState.UNAVAILABLE, health.detail)
            state = ProviderState.UNAVAILABLE.value
        return ProbeResult(
            provider_id=descriptor.id,
            kind=descriptor.kind,
            available=available,
            state=state,
            detail=health.detail,
            latency_ms=int(round((time.monotonic() - started) * 1000)),
        )
    except Exception as exc:  # noqa: BLE001 - probe must not kill discovery
        try:
            provider.lifecycle.transition(ProviderState.UNAVAILABLE, f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            pass
        return ProbeResult(
            provider_id=descriptor.id,
            kind=descriptor.kind,
            available=False,
            state=ProviderState.UNAVAILABLE.value,
            detail=f"{type(exc).__name__}: {exc}",
            latency_ms=int(round((time.monotonic() - started) * 1000)),
        )


async def discover(
    reg: ProviderRegistry = registry,
    *,
    options_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    kinds: Sequence[ProviderKind] | None = None,
    concurrency: int = 8,
) -> list[ProbeResult]:
    """Probe all registered providers concurrently, preserving id order."""

    targets = [
        (reg.get_class(provider_id), provider_id)
        for provider_id in reg.ids()
        if kinds is None or reg.registration(provider_id).descriptor.kind in kinds
    ]

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(cls: type[BaseProvider], provider_id: str) -> ProbeResult:
        async with semaphore:
            return await probe_provider(cls, (options_by_id or {}).get(provider_id))

    results = await asyncio.gather(*(run(cls, pid) for cls, pid in targets))
    return sorted(results, key=lambda item: item.provider_id)


async def available_descriptors(
    reg: ProviderRegistry = registry,
    *,
    options_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ProviderDescriptor]:
    """Descriptors of everything that passed its probe."""

    results = await discover(reg, options_by_id=options_by_id)
    good = {item.provider_id for item in results if item.available}
    return [
        reg.registration(provider_id).descriptor
        for provider_id in reg.ids()
        if provider_id in good
    ]


def filter_available(
    descriptors: Iterable[ProviderDescriptor], available: Iterable[str]
) -> list[ProviderDescriptor]:
    allowed = set(available)
    return [descriptor for descriptor in descriptors if descriptor.id in allowed]
