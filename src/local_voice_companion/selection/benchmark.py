"""Benchmark results, cache and the honest simulator.

HONESTY CONTRACT
----------------
A BenchmarkResult whose `source` is SIMULATED contains *derived estimates*,
not measured numbers. Nothing in this file may present simulated data as real
measurements, and any UI/API surface must carry the same `source` field.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..config.paths import DEFAULT_LAYOUT
from ..hardware.profile import HardwareProfile
from .candidates import Candidate


class BenchmarkSource(str, Enum):
    SIMULATED = "simulated"
    MEASURED = "measured"


@dataclass
class Metric:
    value: float
    source: BenchmarkSource = BenchmarkSource.SIMULATED
    unit: str = "ms"

    def to_dict(self) -> dict[str, Any]:
        return {"value": round(self.value, 3), "source": self.source.value, "unit": self.unit}


@dataclass
class BenchmarkResult:
    candidate_id: str
    provider_id: str
    model_id: str
    device: str
    source: BenchmarkSource = BenchmarkSource.SIMULATED

    load_ms: Metric | None = None
    ttft_ms: Metric | None = None
    real_time_factor: Metric | None = None
    ram_peak_mb: Metric | None = None
    vram_peak_mb: Metric | None = None
    supported_languages: list[str] = field(default_factory=list)
    error: str = ""
    taken_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return not self.error

    def cache_key(self, fingerprint: str) -> str:
        """Key must capture hardware + provider + model versions, nothing volatile."""

        return "|".join(
            [
                fingerprint,
                self.provider_id,
                self.model_id,
                self.device,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source.value
        for name in ("load_ms", "ttft_ms", "real_time_factor", "ram_peak_mb", "vram_peak_mb"):
            value = getattr(self, name)
            payload[name] = value.to_dict() if value else None
        payload.pop("candidate_id", None)
        payload["candidate_id"] = self.candidate_id
        return payload


# ---------------------------------------------------------------------------
# Deterministic estimator used when nothing has been measured yet.
# ---------------------------------------------------------------------------

#: Relative throughput multipliers. These are *ordinal* performance classes,
#: not measured factors, and are always reported as SIMULATED.
_DEVICE_SPEED: dict[str, float] = {
    "cpu": 1.0,
    "directml": 1.8,
    "rocm": 2.4,
    "metal": 2.6,
    "cuda": 3.2,
    "remote": 1.2,
}

_KIND_BASE_MS: dict[str, float] = {
    "asr": 420.0,
    "llm": 260.0,
    "tts": 380.0,
    "vad": 5.0,
}

#: `latency_tier` is an ordinal 1..5 speed class (5 = fastest). These are the
#: multipliers applied to the per-kind base cost. Without this, latency_tier is
#: decorative and every provider in a kind gets an identical TTFT estimate --
#: which silently destroys the ranking, because latency is the heaviest weight
#: in every policy.
_TIER_FACTOR: tuple[float, ...] = (0.0, 1.9, 1.35, 1.0, 0.78, 0.6)

#: Providers that do no model work at all (test doubles, stubs, mocks). They
#: must not out-rank a real engine merely by being cheap to run, so they are
#: held at an average-for-kind latency and flagged as such.
_STUB_PROVIDER_PREFIXES: tuple[str, ...] = ("fake_", "stub_", "dummy_", "test_")

#: The tier a stub is pinned to. 3 is deliberately mid-pack: a stub is not
#: "fast", it is *unmeasured*.
_STUB_TIER = 3


def is_stub(provider_id: str) -> bool:
    """True for test doubles that perform no real inference."""

    return provider_id.startswith(_STUB_PROVIDER_PREFIXES)


def _cpu_class_factor(profile: HardwareProfile) -> float:
    threads = profile.cpu.threads or 4
    if threads >= 16:
        return 0.75
    if threads >= 8:
        return 1.0
    if threads >= 4:
        return 1.4
    return 2.1


def estimate_metrics(
    candidate: Candidate, profile: HardwareProfile
) -> tuple[float, float, float]:
    """Return (load_ms, ttft_ms, rtf) as *simulated* values.

    A ``remote`` device means the heavy lifting happens somewhere else: it pays
    round-trip latency but consumes no local VRAM. Treating it like a local
    accelerator is what makes the planner think an API LLM is "out of memory".
    """

    descriptor = candidate.descriptor
    device = candidate.device
    is_remote = device == "remote" or descriptor.requires_network
    device_speed = _DEVICE_SPEED.get(device, 1.0)
    stub = is_stub(candidate.provider_id)
    tier = _STUB_TIER if stub else min(max(descriptor.latency_tier, 1), 5)
    tier_factor = _TIER_FACTOR[tier]
    cpu_factor = _cpu_class_factor(profile)
    heavy = max(1.0, descriptor.estimated_disk_mb / 700.0)

    if is_remote:
        # Round-trip overhead that no CPU/GPU speed can fix, plus the remote
        # engine's own tier. No local load, no local memory residency.
        base = _KIND_BASE_MS.get(candidate.kind.value, 300.0) * tier_factor
        return 0.0, max(1.0, base * 1.25), 0.35

    base = _KIND_BASE_MS.get(candidate.kind.value, 300.0) * tier_factor
    if device == "cpu":
        ttft = base * cpu_factor
    else:
        # Mixed CPU+accelerator path still pays CPU-side text handling.
        ttft = base / device_speed + 12.0 * cpu_factor

    load = heavy * (420.0 / device_speed if device != "cpu" else 420.0)
    rtf = round(0.42 / device_speed * tier_factor, 4) if device != "cpu" else round(0.42 * cpu_factor * tier_factor, 4)
    return max(1.0, load), max(1.0, ttft), max(0.01, rtf)


def simulate_benchmark(
    candidate: Candidate, profile: HardwareProfile
) -> BenchmarkResult:
    """Deterministic pseudo-benchmark. Always labelled SIMULATED."""

    load, ttft, rtf = estimate_metrics(candidate, profile)
    descriptor = candidate.descriptor
    simulated = BenchmarkSource.SIMULATED

    # A remote stage processes elsewhere: it holds no local VRAM, and a stub
    # holds nothing measurable at all. Reporting residency here would make the
    # resource planner reject stages that cost nothing.
    remote = candidate.device == "remote" or descriptor.requires_network
    stub = is_stub(candidate.provider_id)
    local_resident = not remote and not stub

    vram = descriptor.estimated_vram_mb if (local_resident and candidate.device != "cpu") else 0
    ram = descriptor.estimated_ram_mb if local_resident else 0

    return BenchmarkResult(
        candidate_id=candidate.id,
        provider_id=candidate.provider_id,
        model_id=candidate.model_id,
        device=candidate.device,
        source=simulated,
        load_ms=Metric(load, simulated),
        ttft_ms=Metric(ttft, simulated),
        real_time_factor=Metric(rtf, simulated, unit="ratio"),
        ram_peak_mb=Metric(ram, simulated, unit="mb"),
        vram_peak_mb=Metric(vram, simulated, unit="mb"),
        supported_languages=list(descriptor.languages),
    )


async def measure_benchmark(candidate: Candidate, profile: HardwareProfile) -> BenchmarkResult:
    """Real measurement hook.

    Phase 1 ships no heavy model, so this delegates to the simulator and keeps
    the source label honest. When real providers arrive they will replace this
    body with an actual timed call and set source=MEASURED.
    """

    return simulate_benchmark(candidate, profile)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class BenchmarkCache:
    """Filesystem-backed result cache keyed by hardware fingerprint + versions."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_LAYOUT.benchmark_db
        self._data: dict[str, dict[str, Any]] | None = None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._data is None:
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}
        assert self._data is not None
        return self._data

    def get(self, fingerprint: str, candidate: Candidate) -> BenchmarkResult | None:
        key = self._key(fingerprint, candidate)
        payload = self._load().get(key)
        if not payload:
            return None
        return self._from_payload(payload)

    def put(self, fingerprint: str, candidate: Candidate, result: BenchmarkResult) -> None:
        key = self._key(fingerprint, candidate)
        self._load()[key] = result.to_dict()

    def flush(self) -> None:
        if self._data is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            # A read-only cache must not break selection.
            pass

    def clear(self) -> None:
        self._data = {}
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _key(fingerprint: str, candidate: Candidate) -> str:
        return "|".join([fingerprint, candidate.provider_id, candidate.model_id, candidate.device])

    @staticmethod
    def _from_payload(payload: dict[str, Any]) -> BenchmarkResult:
        def metric(name: str, unit: str) -> Metric | None:
            raw = payload.get(name)
            if not raw:
                return None
            return Metric(value=float(raw["value"]), source=BenchmarkSource(raw["source"]), unit=unit)

        return BenchmarkResult(
            candidate_id=payload.get("candidate_id", ""),
            provider_id=payload["provider_id"],
            model_id=payload["model_id"],
            device=payload["device"],
            source=BenchmarkSource(payload["source"]),
            load_ms=metric("load_ms", "ms"),
            ttft_ms=metric("ttft_ms", "ms"),
            real_time_factor=metric("real_time_factor", "ratio"),
            ram_peak_mb=metric("ram_peak_mb", "mb"),
            vram_peak_mb=metric("vram_peak_mb", "mb"),
            supported_languages=list(payload.get("supported_languages") or []),
            error=payload.get("error", ""),
            taken_at=payload.get("taken_at", 0.0),
        )
