"""Inference-runtime introspection shared by the native providers.

The point of this module is *granular failure reporting*. "Provider unavailable"
is not actionable; "onnxruntime is not installed" and "onnxruntime is installed
but reports no CPU execution provider" are. Each probe helper returns a
:class:`RuntimeStatus` that a provider can hand straight to ``ProviderHealth``.

Everything here is import-safe: no heavy module is imported at import time, and
every probe is wrapped so that a broken install produces a status object rather
than a traceback.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass, field
from typing import Any

#: Probe outcomes. Ordered roughly from "not our fault" to "works".
STATUS_READY = "ready"
STATUS_DEPENDENCY_MISSING = "dependency_missing"
STATUS_RUNTIME_UNBROKEN = "runtime_import_failed"
STATUS_RUNTIME_UNAVAILABLE = "runtime_unavailable"
STATUS_NO_PROVIDERS = "no_execution_providers"
STATUS_MODEL_MISSING = "model_missing"
STATUS_UNSUPPORTED_PLATFORM = "unsupported_platform"


@dataclass
class RuntimeStatus:
    """Result of inspecting an optional inference dependency."""

    status: str
    package: str = ""
    detail: str = ""
    version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_READY

    @property
    def actionable(self) -> bool:
        """True when the fix is a one-line install the user can just run."""

        return self.status in {
            STATUS_DEPENDENCY_MISSING,
            STATUS_MODEL_MISSING,
        }

    def hint(self, extra_name: str) -> str:
        if self.status == STATUS_DEPENDENCY_MISSING:
            return f"pip install 'local-voice-companion[{extra_name}]'"
        if self.status == STATUS_RUNTIME_IMPORT_FAILED:
            return (
                f"src/local_voice_companion/providers/local: importing {self.package} raised "
                f"{self.detail}. Reinstall it: pip install --force-reinstall {self.package}"
            )
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "package": self.package,
            "detail": self.detail,
            "version": self.version,
            **self.extra,
        }


def module_present(name: str) -> bool:
    """True when the module can be found, without importing it."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def package_version(module_name: str) -> str:
    """Best-effort version lookup that never raises."""

    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(module_name)
        except PackageNotFoundError:
            return ""
    except Exception:  # noqa: BLE001 - introspection must not break a probe
        return ""


def probe_ctranslate2() -> RuntimeStatus:
    """Inspect CTranslate2, the engine behind faster-whisper."""

    if not module_present("ctranslate2"):
        return RuntimeStatus(
            STATUS_DEPENDENCY_MISSING,
            package="ctranslate2",
            detail="ctranslate2 is not installed; faster-whisper cannot run",
        )
    try:
        import ctranslate2  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - a broken native wheel lands here
        return RuntimeStatus(
            STATUS_RUNTIME_IMPORT_FAILED,
            package="ctranslate2",
            detail=f"{type(exc).__name__}: {exc}",
        )

    cpu_types: list[str] = []
    cuda_count = 0
    try:
        cpu_types = sorted(ctranslate2.get_supported_compute_types("cpu"))
    except Exception as exc:  # noqa: BLE001
        return RuntimeStatus(
            STATUS_RUNTIME_UNAVAILABLE,
            package="ctranslate2",
            detail=f"compute-type enumeration failed: {type(exc).__name__}: {exc}",
        )
    try:
        cuda_count = int(ctranslate2.get_cuda_device_count())
    except Exception:  # noqa: BLE001 - no CUDA runtime is a normal outcome
        cuda_count = 0

    if not cpu_types:
        return RuntimeStatus(
            STATUS_NO_PROVIDERS,
            package="ctranslate2",
            detail="ctranslate2 reports no CPU compute types",
        )

    return RuntimeStatus(
        STATUS_READY,
        package="ctranslate2",
        detail="cpu compute types available",
        version=getattr(ctranslate2, "__version__", "") or package_version("ctranslate2"),
        extra={
            "cpu_compute_types": cpu_types,
            "cuda_device_count": cuda_count,
            "cuda_usable": cuda_count > 0,
            "cpu_count": getattr(ctranslate2, "get_cpu_count", lambda *_: None)() or None,
        },
    )


def probe_faster_whisper() -> RuntimeStatus:
    if not module_present("faster_whisper"):
        return RuntimeStatus(
            STATUS_DEPENDENCY_MISSING,
            package="faster-whisper",
            detail="faster-whisper is not installed",
        )
    engine = probe_ctranslate2()
    if not engine.ok:
        return engine
    try:
        import faster_whisper  # type: ignore[import-not-found]

        version = getattr(faster_whisper, "__version__", "") or package_version("faster-whisper")
    except Exception as exc:  # noqa: BLE001
        return RuntimeStatus(
            STATUS_RUNTIME_IMPORT_FAILED,
            package="faster-whisper",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return RuntimeStatus(
        STATUS_READY,
        package="faster-whisper",
        detail="faster-whisper importable",
        version=version,
        extra=dict(engine.extra),
    )


def probe_onnxruntime() -> RuntimeStatus:
    """Inspect onnxruntime and which execution providers it actually offers."""

    module_name = "onnxruntime"
    if not module_present(module_name):
        # A GPU-only install registers the same import name, so absence is
        # unambiguous: there is no ONNX runtime at all.
        return RuntimeStatus(
            STATUS_DEPENDENCY_MISSING,
            package="onnxruntime",
            detail="onnxruntime is not installed; ONNX-based TTS cannot run",
        )
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return RuntimeStatus(
            STATUS_RUNTIME_IMPORT_FAILED,
            package="onnxruntime",
            detail=f"{type(exc).__name__}: {exc}",
        )

    try:
        providers = list(ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001
        return RuntimeStatus(
            STATUS_RUNTIME_UNAVAILABLE,
            package="onnxruntime",
            detail=f"provider enumeration failed: {type(exc).__name__}: {exc}",
        )
    if not providers:
        return RuntimeStatus(
            STATUS_NO_PROVIDERS,
            package="onnxruntime",
            detail="onnxruntime reports no execution providers",
        )

    return RuntimeStatus(
        STATUS_READY,
        package="onnxruntime",
        detail=f"{len(providers)} execution provider(s)",
        version=getattr(ort, "__version__", "") or package_version("onnxruntime"),
        extra={
            "execution_providers": providers,
            "cpu_available": "CPUExecutionProvider" in providers,
            "cuda_available": "CUDAExecutionProvider" in providers,
            "directml_available": "DmlExecutionProvider" in providers,
        },
    )


def probe_kokoro_onnx() -> RuntimeStatus:
    if not module_present("kokoro_onnx"):
        return RuntimeStatus(
            STATUS_DEPENDENCY_MISSING,
            package="kokoro-onnx",
            detail="kokoro-onnx is not installed",
        )
    runtime = probe_onnxruntime()
    if not runtime.ok:
        return runtime
    try:
        import kokoro_onnx  # type: ignore[import-not-found]

        version = getattr(kokoro_onnx, "__version__", "") or package_version("kokoro-onnx")
    except Exception as exc:  # noqa: BLE001
        return RuntimeStatus(
            STATUS_RUNTIME_IMPORT_FAILED,
            package="kokoro-onnx",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return RuntimeStatus(
        STATUS_READY,
        package="kokoro-onnx",
        detail="kokoro-onnx importable",
        version=version,
        extra=dict(runtime.extra),
    )


def probe_sherpa_onnx() -> RuntimeStatus:
    """Streaming ASR + VAD runtime.

    Unlike the other engines this one is not probed through onnxruntime:
    sherpa-onnx statically links its own build of ORT into ``_sherpa_onnx``, so
    ``probe_onnxruntime()`` here would report on a copy of ORT that sherpa never
    loads and say nothing useful. What actually matters is that the extension
    imports and exposes the streaming entry point.
    """

    if not module_present("sherpa_onnx"):
        return RuntimeStatus(
            STATUS_DEPENDENCY_MISSING,
            package="sherpa-onnx",
            detail="sherpa-onnx is not installed",
        )
    try:
        import sherpa_onnx  # type: ignore[import-not-found]

        version = getattr(sherpa_onnx, "__version__", "") or package_version("sherpa-onnx")
        has_streaming = hasattr(sherpa_onnx, "OnlineRecognizer")
    except Exception as exc:  # noqa: BLE001
        return RuntimeStatus(
            STATUS_RUNTIME_IMPORT_FAILED,
            package="sherpa-onnx",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if not has_streaming:
        return RuntimeStatus(
            STATUS_RUNTIME_UNAVAILABLE,
            package="sherpa-onnx",
            detail="sherpa_onnx.OnlineRecognizer is missing; install sherpa-onnx-core too",
        )
    return RuntimeStatus(
        STATUS_READY,
        package="sherpa-onnx",
        detail="sherpa-onnx importable; streaming recogniser available",
        version=version,
    )


def probe_g2p(backend: str = "misaki") -> RuntimeStatus:
    """Chinese grapheme-to-phoneme front end.

    Kokoro is a phoneme model: it cannot be handed raw Chinese text. Two
    backends can produce the phoneme string, and they are not equivalent.

    ``misaki`` is the upstream reference front end and handles Chinese pinyin
    conversion, tone sandhi and heteronym selection. ``espeak`` is available on
    every platform because ``espeakng-loader`` ships the binary, but espeak-ng's
    Mandarin support is noticeably worse, so it is the fallback rather than the
    default -- and the descriptor records which one is in use.
    """

    if backend == "misaki":
        return _probe_misaki()
    return _probe_espeak()


def _probe_misaki() -> RuntimeStatus:
    if not module_present("misaki"):
        return RuntimeStatus(
            STATUS_DEPENDENCY_MISSING,
            package="misaki",
            detail="misaki is not installed; Chinese G2P falls back to espeak-ng",
        )
    try:
        from misaki import zh  # type: ignore[import-not-found]

        g2p = zh.ZHG2P(version="1.1")
    except Exception as exc:  # noqa: BLE001 - missing dictionaries land here
        return RuntimeStatus(
            STATUS_RUNTIME_IMPORT_FAILED,
            package="misaki",
            detail=f"{type(exc).__name__}: {exc}",
        )
    tone = getattr(g2p, "tone", False)
    return RuntimeStatus(
        STATUS_READY,
        package="misaki",
        detail=f"misaki zh ZHG2P (tone={tone})",
        version=package_version("misaki"),
        extra={"backend": "misaki", "tone": bool(tone)},
    )


def _probe_espeak() -> RuntimeStatus:
    if not module_present("phonemizer"):
        return RuntimeStatus(
            STATUS_DEPENDENCY_MISSING,
            package="phonemizer",
            detail="phonemizer is not installed; no G2P backend is available",
        )
    if not module_present("espeakng_loader"):
        return RuntimeStatus(
            STATUS_DEPENDENCY_MISSING,
            package="espeakng-loader",
            detail="espeakng-loader is not installed; the espeak-ng binary is missing",
        )
    try:
        import espeakng_loader  # type: ignore[import-not-found]
        from phonemizer import phonemize  # type: ignore[import-not-found]
        from phonemizer.backend import EspeakBackend  # type: ignore[import-not-found]

        # Point espeak at the bundled binary rather than whatever the OS has,
        # then prove it can actually phonemise before claiming readiness.
        library_path = espeakng_loader.get_library_path()
        EspeakBackend(
            language="cmn",
            library_path=library_path,
            data_path=espeakng_loader.get_data_path(),
        )
        sample = phonemize("你好", language="cmn", backend="espeak", strip=True)
        if not sample:
            raise RuntimeError("espeak produced an empty phoneme string for 你好")
    except Exception as exc:  # noqa: BLE001
        return RuntimeStatus(
            STATUS_RUNTIME_UNAVAILABLE,
            package="espeakng-loader",
            detail=f"bundled espeak-ng unusable: {type(exc).__name__}: {exc}",
        )
    return RuntimeStatus(
        STATUS_READY,
        package="espeakng-loader",
        detail="bundled espeak-ng phonemises Mandarin",
        version=package_version("espeakng-loader") or package_version("phonemizer"),
        extra={"backend": "espeak", "library_path": str(library_path)},
    )


def platform_summary() -> dict[str, Any]:
    """Small, dependency-free platform note attached to every probe result."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


__all__ = [
    "RuntimeStatus",
    "STATUS_DEPENDENCY_MISSING",
    "STATUS_MODEL_MISSING",
    "STATUS_NO_PROVIDERS",
    "STATUS_READY",
    "STATUS_RUNTIME_IMPORT_FAILED",
    "STATUS_RUNTIME_UNAVAILABLE",
    "STATUS_UNSUPPORTED_PLATFORM",
    "module_present",
    "package_version",
    "platform_summary",
    "probe_ctranslate2",
    "probe_espeak",
    "probe_faster_whisper",
    "probe_g2p",
    "probe_kokoro_onnx",
    "probe_misaki",
    "probe_onnxruntime",
    "probe_sherpa_onnx",
]
