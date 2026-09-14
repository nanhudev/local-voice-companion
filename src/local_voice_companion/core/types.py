"""Shared domain types for the Local Voice Companion runtime.

These types are intentionally dependency-free (stdlib dataclasses only) so that
any layer -- core, providers, selection, api, integrations -- can import them
without pulling heavy frameworks in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2

API_PREFIX = "/api/v1"


class ProviderKind(str, Enum):
    """The four extension points of the runtime."""

    ASR = "asr"
    TTS = "tts"
    LLM = "llm"
    VAD = "vad"


class Device(str, Enum):
    """Execution targets a provider can run on."""

    CPU = "cpu"
    CUDA = "cuda"
    DIRECTML = "directml"
    METAL = "metal"
    ROCM = "rocm"
    VULKAN = "vulkan"
    REMOTE = "remote"


class QualitySource(str, Enum):
    """Where a quality number came from. Never fabricate quality."""

    MEASURED = "measured"
    OFFLINE_EVAL = "offline_eval"
    CURATED_METADATA = "curated_metadata"
    PROVIDER_REPORTED = "provider_reported"
    USER_PREFERENCE = "user_preference"
    UNKNOWN = "unknown"


class SelectionPolicy(str, Enum):
    """User-facing selection intent. Expert knobs live behind `manual`."""

    AUTO = "auto"
    ULTRA_LOW_LATENCY = "ultra_low_latency"
    BALANCED = "balanced"
    QUALITY = "quality"
    LOW_MEMORY = "low_memory"
    CPU_ONLY = "cpu_only"
    MANUAL = "manual"


@dataclass(frozen=True)
class ResourceRequirements:
    """Estimated resource envelope of a provider/model combination."""

    ram_mb: int = 0
    vram_mb: int = 0
    cpu_threads: int = 1
    disk_mb: int = 0
    requires_network: bool = False


@dataclass(frozen=True)
class AudioChunk:
    """A block of PCM audio travelling through the pipeline."""

    pcm: bytes
    sample_rate: int
    channels: int = 1
    sample_width: int = 2
    is_final: bool = False

    @property
    def duration_ms(self) -> float:
        if not self.pcm:
            return 0.0
        frames = len(self.pcm) / (self.sample_width * self.channels)
        return frames / self.sample_rate * 1000.0


@dataclass(frozen=True)
class AudioFormat:
    """Container for synthesised audio returned by a TTS provider."""

    mime_type: str = "audio/wav"
    sample_rate: int = 24000
    codec: str = "pcm_s16le"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ProviderRuntimeState:
    """Mutable per-instance lifecycle record for a loaded provider."""

    provider_id: str
    state: str = "discovered"
    detail: str = ""
    loaded_model: str | None = None
    device: str = Device.CPU.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "state": self.state,
            "detail": self.detail,
            "loaded_model": self.loaded_model,
            "device": self.device,
        }


def normalize_language(language: str) -> str:
    """Collapse `zh-CN` and `zh_CN` to the bare `zh` used by descriptors."""

    cleaned = (language or "").strip().lower().replace("_", "-")
    if not cleaned:
        return ""
    return cleaned.split("-")[0]


def language_matches(desired: str, supported: Iterable[str]) -> bool:
    """True when `desired` is served by one of `supported`."""

    wanted = normalize_language(desired)
    if not wanted:
        return True
    return any(normalize_language(item) == wanted for item in supported)


def merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update({key: value for key, value in override.items() if value is not None})
    return merged


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def unique(items: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


@dataclass
class Mutable:
    """Marker base for mutable dataclasses that also expose a wire format."""

    extra: dict[str, Any] = field(default_factory=dict)
