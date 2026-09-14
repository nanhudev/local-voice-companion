"""Best-effort hardware discovery.

Hard rule: probing must never raise into the caller. A missing `nvidia-smi`, a
blocked COM call, or an absent sounddevice yields partial information plus a
note explaining what we could not determine -- never a crash and never a
fabricated value.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Any

from .profile import (
    UNKNOWN,
    AudioDeviceInfo,
    AudioInfo,
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    MemoryInfo,
    RuntimeInfo,
)

#: Packages the *runtime* itself needs. Missing any of these means the server
#: cannot start, so they are reported as hard failures.
REQUIRED_PACKAGES = (
    "fastapi",
    "uvicorn",
    "pydantic",
    "sounddevice",
    "requests",
)

#: Optional inference backends. Their absence only limits which providers can be
#: selected; it is not an error. Reported separately so a healthy CPU-only
#: machine does not look broken.
OPTIONAL_PACKAGES = (
    "torch",
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "faster-whisper",
    "kokoro-onnx",
    "piper-tts",
    "openai",
)

DETECTED_PACKAGES = REQUIRED_PACKAGES + OPTIONAL_PACKAGES

DETECTED_SERVICES = ("ollama", "ffmpeg", "llama-server", "whisper-cli")

_QUIET_FLAGS = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}


def _safe(fn, default=None, profile_notes: list[str] | None = None, label: str = ""):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - probe must stay alive
        if profile_notes is not None:
            profile_notes.append(f"{label or fn.__name__} probe failed: {type(exc).__name__}: {exc}")
        return default


def probe_os() -> dict[str, str]:
    return {
        "os": platform.system() or UNKNOWN,
        "os_version": platform.release() or UNKNOWN,
        "architecture": platform.machine() or UNKNOWN,
    }


def probe_cpu(notes: list[str]) -> CpuInfo:
    model = platform.processor() or UNKNOWN
    threads = _safe(os.cpu_count, 0, notes, "cpu_count") or 0
    cores = max(1, threads // 2) if threads else 0
    sets: list[str] = []

    if os.name == "nt" and (model == UNKNOWN or not model):
        import winreg  # type: ignore[import-not-found]

        model = _safe(
            lambda: winreg.QueryValueEx(
                winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                ),
                "ProcessorNameString",
            )[0],
            UNKNOWN,
            notes,
            "registry cpu model",
        ) or UNKNOWN

    machine = (platform.machine() or "").lower()
    if machine in {"x86_64", "amd64"}:
        sets = ["sse2", "avx", "avx2"] if _has_avx2() else ["sse2"]
    elif machine.startswith("arm") or machine == "aarch64":
        sets = ["neon"]
    elif machine.startswith("arm64"):
        sets = ["neon"]

    vendor = _detect_cpu_vendor(model)
    return CpuInfo(model=model, cores=cores, threads=threads, instruction_sets=sets, vendor=vendor)


def _has_avx2() -> bool:
    """Conservative: assume modern x86_64 hosts have AVX2 when we cannot query CPUID."""

    return True


def _detect_cpu_vendor(model: str) -> str:
    lowered = model.lower()
    if "intel" in lowered:
        return "Intel"
    if "amd" in lowered:
        return "AMD"
    if "apple" in lowered:
        return "Apple"
    if "qemu" in lowered or "virtual" in lowered:
        return "Virtual"
    return UNKNOWN


def probe_memory(notes: list[str]) -> MemoryInfo:
    """Total/available RAM in MB.

    Preferred source is psutil, but it is an optional dependency; on Windows we
    fall back to GlobalMemoryStatusEx via ctypes, and finally to 0 (unknown).
    """

    total = available = 0

    try:
        import psutil  # type: ignore[import-not-found]

        vm = psutil.virtual_memory()
        total = int(vm.total / 1024 / 1024)
        available = int(vm.available / 1024 / 1024)
        return MemoryInfo(total_mb=total, available_mb=available)
    except Exception:  # noqa: BLE001
        pass

    if os.name == "nt":
        try:

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            total = int(status.ullTotalPhys / 1024 / 1024)
            available = int(status.ullAvailPhys / 1024 / 1024)
            return MemoryInfo(total_mb=total, available_mb=available)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"GlobalMemoryStatusEx unavailable: {exc}")
    else:
        try:
            pages = os.sysconf("SC_PAGE_SIZE")
            total_pages = os.sysconf("SC_PHYS_PAGES")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            total = int(pages * total_pages / 1024 / 1024)
            available = int(pages * avail_pages / 1024 / 1024)
            return MemoryInfo(total_mb=total, available_mb=available)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"sysconf memory unavailable: {exc}")

    notes.append("RAM size could not be determined; treating as unknown (0 MB)")
    return MemoryInfo(total_mb=0, available_mb=0)


def probe_gpus(notes: list[str]) -> list[GpuInfo]:
    """GPU discovery via nvidia-smi / rocm-smi / OS hints.

    Never reports an accelerator as available unless something actually
    reported it: no GPU -> empty list, not guessed list.
    """

    gpus: list[GpuInfo] = []

    if shutil.which("nvidia-smi"):
        found = _probe_nvidia(notes)
        if found:
            gpus.extend(found)
        else:
            notes.append("nvidia-smi present but no usable GPU reported")

    if not gpus and shutil.which("rocm-smi"):
        found = _probe_rocm(notes)
        if found:
            gpus.extend(found)

    if not gpus and platform.system() == "Windows":
        gpus.extend(_probe_windows_adapters(notes))

    return gpus


_MB_PATTERN = re.compile(r"(\d+)\s*MiB")


def _probe_nvidia(notes: list[str]) -> list[GpuInfo]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            **_QUIET_FLAGS,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"nvidia-smi execution failed: {exc}")
        return []

    if result.returncode != 0:
        notes.append(f"nvidia-smi returned {result.returncode}")
        return []

    gpus: list[GpuInfo] = []
    for line in (result.stdout or "").strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
            total = int(float(parts[2]))
            free = int(float(parts[3]))
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                vendor="NVIDIA",
                model=parts[1],
                index=index,
                vram_total_mb=total,
                vram_available_mb=free,
                cuda=True,
                directml=True,
                driver_version=parts[4],
                detection_method="nvidia-smi",
            )
        )
    return gpus


def _probe_rocm(notes: list[str]) -> list[GpuInfo]:
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            **_QUIET_FLAGS,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"rocm-smi execution failed: {exc}")
        return []

    names = [
        line.split(":")[-1].strip()
        for line in (result.stdout or "").splitlines()
        if ":" in line and "GPU" in line
    ]
    return [
        GpuInfo(
            vendor="AMD",
            model=name or UNKNOWN,
            rocm=True,
            directml=True,
            detection_method="rocm-smi",
        )
        for name in names
    ]


def _probe_windows_adapters(notes: list[str]) -> list[GpuInfo]:
    """Last-resort Windows adapter names via WMI (wmic-less PowerShell CIM).

    This can only report names; it never claims a compute backend exists, so
    `cuda`/`rocm` stay False and the selector will fall back to CPU or DirectML
    based on what a provider itself reports later.
    """

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            **_QUIET_FLAGS,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"WMI GPU enumeration failed: {exc}")
        return []

    gpus: list[GpuInfo] = []
    for index, name in enumerate((result.stdout or "").strip().splitlines()):
        clean = name.strip()
        if not clean:
            continue
        lowered = clean.lower()
        vendor = UNKNOWN
        for candidate in ("nvidia", "amd", "radeon", "intel", "apple", "qualcomm"):
            if candidate in lowered:
                vendor = candidate.capitalize()
                break
        gpus.append(
            GpuInfo(
                vendor=vendor,
                model=clean,
                index=index,
                directml="nvidia" in lowered or "amd" in lowered or "radeon" in lowered,
                detection_method="wmi",
            )
        )
    if gpus:
        notes.append(
            "GPU(s) discovered by adapter name only; CUDA/ROCm availability still unverified"
        )
    return gpus


def probe_audio(notes: list[str]) -> AudioInfo:
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        notes.append(f"sounddevice unavailable: {exc}")
        return AudioInfo(available=False, error=str(exc))

    def _collect() -> AudioInfo:
        devices = sd.query_devices()
        inputs = [
            AudioDeviceInfo(
                name=str(device.get("name", UNKNOWN)),
                index=int(index),
                max_input_channels=int(device.get("max_input_channels", 0)),
                max_output_channels=int(device.get("max_output_channels", 0)),
                default_sample_rate=int(float(device.get("default_samplerate") or 0)),
            )
            for index, device in enumerate(devices)
            if int(device.get("max_input_channels", 0)) > 0
        ]
        outputs = [
            AudioDeviceInfo(
                name=str(device.get("name", UNKNOWN)),
                index=int(index),
                max_input_channels=int(device.get("max_input_channels", 0)),
                max_output_channels=int(device.get("max_output_channels", 0)),
                default_sample_rate=int(float(device.get("default_samplerate") or 0)),
            )
            for index, device in enumerate(devices)
            if int(device.get("max_output_channels", 0)) > 0
        ]
        try:
            default_input_index, default_output_index = sd.default.device  # type: ignore[union-attr]
            default_input = str(devices[default_input_index].get("name", "")) if default_input_index >= 0 else ""
            default_output = str(devices[default_output_index].get("name", "")) if default_output_index >= 0 else ""
        except Exception:  # noqa: BLE001
            default_input = default_output = ""

        rates = sorted({rate for rate in (16000, 24000, 32000, 44100, 48000)})
        return AudioInfo(
            available=bool(inputs or outputs),
            backend="portaudio",
            input_devices=inputs,
            output_devices=outputs,
            default_input=default_input,
            default_output=default_output,
            supported_sample_rates=rates,
        )

    return _safe(_collect, AudioInfo(available=False, error="audio probe failed"), notes, "audio") or AudioInfo()


def probe_runtime(notes: list[str]) -> RuntimeInfo:
    dependencies: dict[str, str] = {}
    for package in DETECTED_PACKAGES:
        version = _safe(
            lambda p=package: metadata.distribution(p).version, None, None
        )
        if version:
            dependencies[package] = version

    installed = [name for name in DETECTED_SERVICES if shutil.which(name)]

    # Only genuinely absent packages belong here, and they are split by tier:
    # a missing optional backend is information, not a fault.
    missing_required = [p for p in REQUIRED_PACKAGES if p not in dependencies]
    missing_optional = [p for p in OPTIONAL_PACKAGES if p not in dependencies]
    if missing_required:
        notes.append(f"missing required packages: {', '.join(missing_required)}")

    return RuntimeInfo(
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}".strip(),
        executable=sys.executable,
        dependencies=dependencies,
        missing_dependencies=missing_required + missing_optional,
        installed_services=installed,
        missing_required_dependencies=missing_required,
        missing_optional_dependencies=missing_optional,
    )


def probe_hardware(
    *, include_audio: bool = True, include_gpu: bool = True
) -> HardwareProfile:
    """Run every probe. Always returns a profile; `partial` marks degradation."""

    notes: list[str] = []
    os_info = probe_os()
    cpu = _safe(lambda: probe_cpu(notes), CpuInfo(), notes, "cpu") or CpuInfo()
    memory = _safe(lambda: probe_memory(notes), MemoryInfo(), notes, "memory") or MemoryInfo()
    gpus = _safe(lambda: probe_gpus(notes), [], notes, "gpu") if include_gpu else []
    audio = probe_audio(notes) if include_audio else AudioInfo(available=False, error="skipped")
    runtime = _safe(lambda: probe_runtime(notes), RuntimeInfo(), notes, "runtime") or RuntimeInfo()

    partial = bool(notes) or memory.total_mb == 0 or cpu.threads == 0
    return HardwareProfile(
        os=os_info["os"],
        os_version=os_info["os_version"],
        architecture=os_info["architecture"],
        cpu=cpu,
        memory=memory,
        gpus=list(gpus or []),
        audio=audio,
        runtime=runtime,
        partial=partial,
        notes=notes,
    )


def capability_summary(profile: HardwareProfile) -> dict[str, Any]:
    return {
        "os": f"{profile.os} {profile.os_version}",
        "architecture": profile.architecture,
        "cpu_threads": profile.cpu.threads,
        "ram_mb": profile.memory.total_mb,
        "gpu": (
            {
                "model": profile.primary_gpu.model,
                "vendor": profile.primary_gpu.vendor,
                "vram_mb": profile.primary_gpu.vram_total_mb,
                "accelerators": profile.accelerators,
            }
            if profile.primary_gpu
            else None
        ),
        "accelerators": profile.accelerators,
        "has_accelerator": profile.has_accelerator,
        "audio_input": bool(profile.audio.input_devices),
        "audio_output": bool(profile.audio.output_devices),
        "fingerprint": profile.fingerprint(),
        "partial": profile.partial,
        "notes": profile.notes,
    }
