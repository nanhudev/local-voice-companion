"""Version-aware configuration schemas.

Two documents, never one:

    system.json  -> host concerns: server, hardware policy, provider overrides, paths
    bots/*.yaml  -> portable BotManifests (persona + runtime *intent*, never host paths)

API keys never appear in either file. They are referenced by environment
variable name and resolved at call time.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.types import SCHEMA_VERSION, SelectionPolicy
from .paths import describe as describe_paths

CURRENT_SCHEMA_VERSION = SCHEMA_VERSION

TIMELINE_STAGE_NAMES: tuple[str, ...] = (
    "turn_started",
    "vad_end",
    "asr_start",
    "asr_end",
    "llm_start",
    "llm_first_token",
    "llm_end",
    "tts_start",
    "tts_first_audio",
    "playback_start",
    "playback_end",
)


class StrictModel(BaseModel):
    """Forbid unknown keys so typos surface as errors, not silent no-ops."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def merge(self, override: Mapping[str, Any]) -> "StrictModel":
        payload = self.model_dump()
        payload.update({k: v for k, v in override.items() if v is not None})
        return type(self).model_validate(payload)


class ServerConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = 17831
    # Binding outside loopback REQUIRES an auth token (checked at start).
    auth_token_env: str = "LVC_AUTH_TOKEN"

    @model_validator(mode="after")
    def _require_token_env_off_loopback(self) -> "ServerConfig":
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not self.auth_token_env:
            raise ValueError("auth_token_env is required when binding outside loopback")
        return self


class AudioConfig(StrictModel):
    sample_rate: int = 16000
    block_ms: int = 30
    pre_roll_ms: int = 180
    end_silence_ms: int = 360
    min_speech_ms: int = 270
    max_speech_seconds: float = 12.0
    minimum_rms: int = 420
    silence_after_playback_ms: int = 350


class PipelineConfig(StrictModel):
    """Bounded queues and cancellation behaviour. No unbounded asyncio.Queue."""

    max_concurrent_tts_items: int = 8
    max_pending_audio_chunks: int = 64
    drop_policy: Literal["oldest", "newest"] = "oldest"
    chunk_min_chars: int = 8
    chunk_max_chars: int = 120
    history_turns: int = 24
    max_tokens: int = 64
    temperature: float = 0.35


class RuntimePolicyConfig(StrictModel):
    policy: SelectionPolicy = SelectionPolicy.BALANCED
    prefer_devices: list[str] = Field(
        default_factory=lambda: ["cuda", "directml", "metal", "cpu", "remote"]
    )
    allow_network_llm: bool = True
    allow_local_llm: bool = True
    cpu_only: bool = False
    vram_reserve_mb: int = 512
    ram_reserve_mb: int = 1024
    # Secret reference only, never a literal API key.
    llm_api_key_env: str = "LVC_LLM_API_KEY"

    @model_validator(mode="after")
    def _validate_devices(self) -> "RuntimePolicyConfig":
        allowed = {"cpu", "cuda", "directml", "metal", "rocm", "vulkan", "remote"}
        for device in self.prefer_devices:
            if device not in allowed:
                raise ValueError(f"unknown device preference: {device}")
        return self


class ProviderOverride(StrictModel):
    """Manual pinning, surfaced only on the Advanced page."""

    enabled: bool = True
    model: str | None = None
    voice: str | None = None
    device: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    # Legacy Voicebox/Ollama endpoints live here as plain option strings so the
    # core stays ignorant of what Voicebox even is.
    base_url: str | None = None
    api_key_env: str | None = None


class LegacyConfig(StrictModel):
    """Adapter settings for the pre-2.0 Voicebox + Ollama path.

    Presence of this block enables compatibility. It is never the source of
    truth for anything else, and it must never become a core format.
    """

    enabled: bool = True
    label: str = "Voicebox + Ollama (legacy)"
    voicebox_url: str = "http://127.0.0.1:17493"
    voicebox_port: int = 17493
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:1.7b"
    asr_model: str = "base"
    tts_engine: str = "luxtts"
    tts_model_size: str = "0.6B"
    voice_profile_id: str = "default"
    voice_profile_name: str = "Default"
    language: str = "zh"
    system_prompt: str = "You are a concise and friendly voice assistant."
    relay_url: str = ""
    relay_provider: str = "whisper"
    asr_backend: Literal["local", "relay"] = "local"
    asr_worker_id: str = ""
    unload_unused_tts_models: bool = True
    relay_token_env: str = "AI_RELAY_TOKEN"


class ObservabilityConfig(StrictModel):
    structured_logging: bool = True
    log_level: str = "INFO"
    retain_turns: int = 200
    timeline_stages: list[str] = Field(default_factory=lambda: list(TIMELINE_STAGE_NAMES))


class SystemConfig(StrictModel):
    """Everything host-specific. Always validated, always versioned."""

    schema_version: int = CURRENT_SCHEMA_VERSION
    server: ServerConfig = Field(default_factory=ServerConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    runtime: RuntimePolicyConfig = Field(default_factory=RuntimePolicyConfig)
    providers: dict[str, ProviderOverride] = Field(default_factory=dict)
    legacy: LegacyConfig = Field(default_factory=LegacyConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    def provider_override(self, provider_id: str) -> ProviderOverride:
        return self.providers.get(provider_id, ProviderOverride())

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.runtime.policy.value,
            "host": self.server.host,
            "port": self.server.port,
            "prefer_devices": self.runtime.prefer_devices,
            "providers": sorted(self.providers),
            "legacy_enabled": self.legacy.enabled,
        }


def _scrub(node: Any) -> Any:
    if isinstance(node, dict):
        result: dict[str, Any] = {}
        for key, value in node.items():
            lowered = key.lower()
            if "api_key" in lowered or "token" in lowered or "secret" in lowered:
                result[key] = "<set>" if value else "<unset>"
            else:
                result[key] = _scrub(value)
        return result
    if isinstance(node, list):
        return [_scrub(item) for item in node]
    return node


def redacted(model: BaseModel) -> dict[str, Any]:
    """Wire-safe dump. Nothing here may reach a browser with a real key."""

    return _scrub(model.model_dump(mode="json"))


def paths_summary() -> dict[str, str]:
    return describe_paths()
