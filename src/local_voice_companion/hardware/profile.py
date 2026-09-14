"""Hardware and runtime capability model.

The probe must never take the process down. Every field has a "partial"
representation: when something cannot be detected we record it as unknown and
keep going, because selection already has enough to make a safe choice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

UNKNOWN = "unknown"

#: Packages whose presence/version actually changes how fast a model runs.
#: Everything else (web framework, HTTP client, audio glue) is excluded from the
#: fingerprint so that unrelated upgrades do not invalidate measured benchmarks.
_FINGERPRINT_RELEVANT_DEPS = frozenset(
    {
        "torch",
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-directml",
        "faster-whisper",
        "kokoro-onnx",
        "piper-tts",
    }
)


@dataclass
class CpuInfo:
    model: str = UNKNOWN
    cores: int = 0
    threads: int = 0
    instruction_sets: list[str] = field(default_factory=list)
    vendor: str = UNKNOWN
    base_freq_mhz: int = 0


@dataclass
class MemoryInfo:
    total_mb: int = 0
    available_mb: int = 0


@dataclass
class GpuInfo:
    vendor: str = UNKNOWN
    model: str = UNKNOWN
    index: int = 0
    vram_total_mb: int = 0
    vram_available_mb: int = 0
    cuda: bool = False
    cuda_version: str = ""
    rocm: bool = False
    directml: bool = False
    metal: bool = False
    vulkan: bool = False
    driver_version: str = ""
    detection_method: str = UNKNOWN


@dataclass
class AudioDeviceInfo:
    name: str
    index: int = -1
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_sample_rate: int = 0


@dataclass
class AudioInfo:
    available: bool = False
    backend: str = UNKNOWN
    input_devices: list[AudioDeviceInfo] = field(default_factory=list)
    output_devices: list[AudioDeviceInfo] = field(default_factory=list)
    default_input: str = ""
    default_output: str = ""
    supported_sample_rates: list[int] = field(default_factory=list)
    error: str = ""


@dataclass
class RuntimeInfo:
    python_version: str = ""
    platform: str = ""
    executable: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)
    missing_dependencies: list[str] = field(default_factory=list)
    #: Split by severity: the server cannot start without these.
    missing_required_dependencies: list[str] = field(default_factory=list)
    #: Missing inference backends. Limits provider choice, not correctness.
    missing_optional_dependencies: list[str] = field(default_factory=list)
    installed_services: list[str] = field(default_factory=list)


@dataclass
class HardwareProfile:
    os: str = UNKNOWN
    os_version: str = UNKNOWN
    architecture: str = UNKNOWN
    cpu: CpuInfo = field(default_factory=CpuInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    gpus: list[GpuInfo] = field(default_factory=list)
    audio: AudioInfo = field(default_factory=AudioInfo)
    runtime: RuntimeInfo = field(default_factory=RuntimeInfo)
    partial: bool = False
    notes: list[str] = field(default_factory=list)

    # -- derived helpers ----------------------------------------------------

    @property
    def primary_gpu(self) -> GpuInfo | None:
        return self.gpus[0] if self.gpus else None

    @property
    def total_vram_mb(self) -> int:
        return max((gpu.vram_total_mb for gpu in self.gpus), default=0)

    @property
    def best_vram_available_mb(self) -> int:
        usable = [gpu.vram_available_mb for gpu in self.gpus if gpu.vram_available_mb > 0]
        usable.extend(gpu.vram_total_mb for gpu in self.gpus if gpu.vram_available_mb <= 0)
        return max(usable, default=0)

    @property
    def accelerators(self) -> list[str]:
        """Sorted, de-duplicated list of usable accelerator backends."""

        found: list[str] = []
        for gpu in self.gpus:
            for name, active in (
                ("cuda", gpu.cuda),
                ("rocm", gpu.rocm),
                ("directml", gpu.directml),
                ("metal", gpu.metal),
                ("vulkan", gpu.vulkan),
            ):
                if active and name not in found:
                    found.append(name)
        return found

    @property
    def has_accelerator(self) -> bool:
        return bool(self.accelerators) and self.total_vram_mb > 0

    def supports_device(self, device: str) -> bool:
        """CPU and HTTP-backed services are always structurally available.

        Accelerators require at least one GPU that actually reports support.
        """

        name = str(device)
        if name in {"cpu", "remote"}:
            return True
        return any(_gpu_supports_device(gpu, name) for gpu in self.gpus)

    def fingerprint(self) -> str:
        """Stable hardware+toolchain fingerprint used as a benchmark cache key.

        Deliberately excludes volatile quantities such as free VRAM, otherwise
        every benchmark would invalidate itself.

        Only *runtime-relevant* dependencies are folded in. A benchmark result
        depends on the accelerator runtimes and inference backends actually
        present; it does not depend on the web framework version, so upgrading
        fastapi must not throw away every measured number.
        """

        payload = {
            "os": self.os,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "cpu": {"model": self.cpu.model, "cores": self.cpu.cores, "threads": self.cpu.threads},
            "memory_mb": self.memory.total_mb,
            "gpus": [
                {
                    "vendor": gpu.vendor,
                    "model": gpu.model,
                    "vram_total_mb": gpu.vram_total_mb,
                    "detection_method": gpu.detection_method,
                }
                for gpu in self.gpus
            ],
            "python": self.runtime.python_version,
            "deps": {
                name: version
                for name, version in sorted(self.runtime.dependencies.items())
                if name in _FINGERPRINT_RELEVANT_DEPS
            },
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gpu_supports_device(gpu: GpuInfo, device: str) -> bool:
    mapping = {
        "cuda": gpu.cuda,
        "rocm": gpu.rocm,
        "directml": gpu.directml,
        "metal": gpu.metal,
        "vulkan": gpu.vulkan,
        "cpu": True,
        "remote": True,
    }
    return bool(mapping.get(str(device), False))


def profile_from_dict(payload: dict[str, Any]) -> HardwareProfile:
    """Rebuild a profile, tolerating partial/unknown fields."""

    raw_cpu = payload.get("cpu") or {}
    raw_memory = payload.get("memory") or {}
    raw_audio = payload.get("audio") or {}
    raw_runtime = payload.get("runtime") or {}
    return HardwareProfile(
        os=payload.get("os", UNKNOWN),
        os_version=payload.get("os_version", UNKNOWN),
        architecture=payload.get("architecture", UNKNOWN),
        cpu=CpuInfo(
            model=raw_cpu.get("model", UNKNOWN),
            cores=int(raw_cpu.get("cores") or 0),
            threads=int(raw_cpu.get("threads") or 0),
            instruction_sets=list(raw_cpu.get("instruction_sets") or []),
        ),
        memory=MemoryInfo(
            total_mb=int(raw_memory.get("total_mb") or 0),
            available_mb=int(raw_memory.get("available_mb") or 0),
        ),
        gpus=[GpuInfo(**gpu) for gpu in payload.get("gpus") or []],
        audio=AudioInfo(
            available=bool(raw_audio.get("available")),
            input_devices=[AudioDeviceInfo(**item) for item in raw_audio.get("input_devices") or []],
            output_devices=[AudioDeviceInfo(**item) for item in raw_audio.get("output_devices") or []],
        ),
        runtime=RuntimeInfo(**{k: v for k, v in raw_runtime.items() if k in RuntimeInfo.__annotations__}),
        partial=bool(payload.get("partial", False)),
        notes=list(payload.get("notes") or []),
    )


def synthetic_profile(
    *,
    os_name: str = "Windows",
    cpu_threads: int = 8,
    ram_mb: int = 16384,
    vram_mb: int = 0,
    accelerator: str | None = None,
    audio: bool = True,
) -> HardwareProfile:
    """Deterministic profile factory for tests and selector scenarios."""

    gpus = []
    if vram_mb > 0:
        gpus.append(
            GpuInfo(
                vendor="NVIDIA" if accelerator in {"cuda", None} else UNKNOWN,
                model="Test GPU",
                vram_total_mb=vram_mb,
                vram_available_mb=vram_mb,
                cuda=accelerator == "cuda",
                directml=accelerator == "directml",
                metal=accelerator == "metal",
                rocm=accelerator == "rocm",
                detection_method="synthetic",
            )
        )
    return HardwareProfile(
        os=os_name,
        os_version="11",
        architecture="x86_64",
        cpu=CpuInfo(model="Test CPU", cores=max(1, cpu_threads // 2), threads=cpu_threads),
        memory=MemoryInfo(total_mb=ram_mb, available_mb=max(1024, ram_mb // 2)),
        gpus=gpus,
        audio=AudioInfo(available=audio),
        partial=False,
        notes=["synthetic profile for deterministic testing"],
    )
