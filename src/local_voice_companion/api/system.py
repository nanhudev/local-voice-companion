"""Runtime state: hardware profile, provider probes and the loaded pipeline.

One object owns the "what is this machine actually running" question so the API
layer never has to assemble it from parts.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..config.schema import SystemConfig
from ..core.events import EventBus, EventType
from ..core.orchestrator import Pipeline
from ..core.types import ProviderKind
from ..hardware.probe import probe_hardware
from ..hardware.profile import HardwareProfile
from ..providers.base import ProviderDescriptor
from ..providers.discovery import ProbeResult, discover
from ..providers.legacy import LEGACY_PROVIDERS, legacy_options
from ..providers.registry import ProviderRegistry, registry
from ..selection.benchmark import BenchmarkCache
from ..selection.engine import recommend
from ..selection.selector import missing_stages
from ..observability.metrics import MetricsRegistry


@dataclass
class RuntimeState:
    config: SystemConfig
    reg: ProviderRegistry = field(default_factory=lambda: registry)
    bus: EventBus = field(default_factory=EventBus)
    started_at: float = field(default_factory=time.monotonic)

    profile: HardwareProfile | None = None
    probe_results: list[ProbeResult] = field(default_factory=list)
    pipeline: Pipeline | None = None
    decision: Any = None
    missing_stages: list[str] = field(default_factory=list)
    detail: str = ""
    ready: bool = False

    @property
    def initialized(self) -> bool:
        """True once a pipeline has been planned and loaded at least once."""

        return self.pipeline is not None and bool(self.pipeline.to_dict())

    @property
    def readiness_detail(self) -> str:
        """Human-readable reason for the current readiness verdict.

        An uninitialised runtime is not a degraded pipeline, and reporting it as
        one (with no missing stages) reads like a bug to any client polling
        /readyz at startup.
        """

        if not self.initialized:
            return "pipeline has not been prepared yet; call POST /api/v1/selection/recommend or wait for lazy startup"
        if self.missing_stages:
            return f"missing stages: {', '.join(self.missing_stages)}"
        return self.detail or ("ready" if self.ready else "not ready")

    _profile_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _pipeline_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # -- hardware -----------------------------------------------------------

    async def refresh_profile(self, *, include_audio: bool = True) -> HardwareProfile:
        async with self._profile_lock:
            self.profile = await asyncio.to_thread(
                probe_hardware, include_audio=include_audio, include_gpu=True
            )
            return self.profile

    # -- providers ----------------------------------------------------------

    async def discover(self) -> list[ProbeResult]:
        options = legacy_options(self.config) if self.config.legacy.enabled else {}
        self.probe_results = await discover(self.reg, options_by_id=options)
        return self.probe_results

    def provider_descriptors(self, kind: ProviderKind | None = None) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.reg.descriptors(kind)]

    def instance_states(self) -> list[dict[str, Any]]:
        return [item.lifecycle.to_dict() for item in self.reg.instances()]

    def models(self, kind: ProviderKind | None = None) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for descriptor in self.reg.descriptors(kind):
            for model in descriptor.models:
                payload.append({"provider_id": descriptor.id, "kind": descriptor.kind.value, **model.to_dict()})
        return payload

    def voices(self, language: str | None = None) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for descriptor in self.reg.descriptors(ProviderKind.TTS):
            for voice in descriptor.voices:
                if language and voice.languages and language not in voice.languages:
                    continue
                payload.append({"provider_id": descriptor.id, **voice.to_dict()})
        return payload

    # -- pipeline -----------------------------------------------------------

    async def prepare_pipeline(self, *, policy_override: str | None = None) -> Pipeline:
        """Select, load and warm the stages of the current plan.

        Failures degrade: whatever loaded is still usable, and the missing
        stages are reported instead of raising a 500 into the UI.
        """

        async with self._pipeline_lock:
            if self.pipeline is not None and not self.missing_stages:
                return self.pipeline

            if self.profile is None:
                await self.refresh_profile()
            profile = self.profile
            assert profile is not None

            if not self.probe_results:
                await self.discover()

            options = legacy_options(self.config) if self.config.legacy.enabled else {}
            self.decision = await recommend(
                profile,
                self.config,
                reg=self.reg,
                cache=BenchmarkCache(),
                policy_override=policy_override,
            )
            plan = self.decision.plan
            self.missing_stages = missing_stages(plan)

            pipeline = Pipeline()
            errors: list[str] = []
            for kind in ("asr", "llm", "tts"):
                assignment = plan.assignment(kind)
                if assignment is None:
                    continue
                try:
                    provider = await self.reg.acquire(
                        assignment.candidate.provider_id, options.get(assignment.candidate.provider_id)
                    )
                    await provider.load(
                        assignment.candidate.model_id, device=assignment.candidate.device
                    )
                    setattr(pipeline, kind, provider)
                    self.bus.emit(
                        EventType.PROVIDER_STATE,
                        data={
                            "provider_id": assignment.candidate.provider_id,
                            "state": provider.lifecycle.state.value,
                            "stage": kind,
                            "model": assignment.candidate.model_id,
                            "device": assignment.candidate.device,
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - degrade, never crash
                    errors.append(f"{kind}:{assignment.candidate.provider_id}: {exc}")

            # VAD is optional and only used by the microphone path.
            for descriptor in self.reg.descriptors(ProviderKind.VAD):
                try:
                    provider = await self.reg.acquire(descriptor.id, options.get(descriptor.id))
                    if not provider.lifecycle.is_serving:
                        await provider.load()
                    pipeline.vad = provider
                    break
                except Exception:  # noqa: BLE001
                    continue

            self.pipeline = pipeline
            self.missing_stages = [name for name in ("asr", "llm", "tts") if getattr(pipeline, name) is None]
            self.ready = not self.missing_stages
            self.detail = "; ".join(errors) if errors else "pipeline ready"
            self.bus.emit(
                EventType.RUNTIME_READY,
                data={
                    "ready": self.ready,
                    "plan": self.decision.plan.to_dict(),
                    "missing_stages": self.missing_stages,
                    "errors": errors,
                },
            )
            return pipeline

    async def shutdown(self) -> None:
        if self.pipeline is not None:
            for name in ("asr", "llm", "tts", "vad"):
                provider = getattr(self.pipeline, name)
                if provider is None:
                    continue
                try:
                    await provider.unload()
                except Exception:  # noqa: BLE001
                    pass
        await self.reg.release_all()

    # -- reporting ----------------------------------------------------------

    def queue_stats(self) -> list[dict[str, Any]]:
        """Queue telemetry. Queues are created per turn, so this is a snapshot."""

        return list(getattr(self, "_queue_snapshots", []))

    def to_dict(self) -> dict[str, Any]:
        plan = self.decision.plan.to_dict() if self.decision is not None else None
        profile = self.profile
        return {
            "ready": self.ready,
            "detail": self.detail,
            "missing_stages": self.missing_stages,
            "uptime_s": round(time.monotonic() - self.started_at, 1),
            "hardware": {
                "fingerprint": profile.fingerprint() if profile else "",
                "os": f"{profile.os} {profile.os_version}" if profile else "unknown",
                "partial": profile.partial if profile else True,
                "accelerators": profile.accelerators if profile else [],
                "vram_mb": profile.total_vram_mb if profile else 0,
                "ram_mb": profile.memory.total_mb if profile else 0,
                "note": profile.notes if profile else [],
            },
            "pipeline": self.pipeline.to_dict() if self.pipeline else {},
            "plan_summary": summarise_plan(self.decision.plan) if self.decision else None,
            "plan": plan,
            "probes": [item.to_dict() for item in self.probe_results],
            "config": self.config.summary(),
        }

    def emit(self, type: str, data: Mapping[str, Any] | None = None) -> None:
        self.bus.emit(type, data=dict(data or {}))


def build_runtime_state(config: SystemConfig, *, reg: ProviderRegistry | None = None) -> RuntimeState:
    """Build a runtime state bound to an explicit registry.

    The registry is a constructor argument rather than a module global so that
    a test harness, a second app in the same process, or an embedded host can
    supply an isolated provider set without mutating global state.
    """

    state = RuntimeState(config=config)
    if reg is not None:
        state.reg = reg
    return state
