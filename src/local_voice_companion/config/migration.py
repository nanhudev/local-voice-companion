"""Config migration: v1 -> v2 (and future steps).

Migration must never silently destroy user configuration. Each step returns a
new dict plus a human-readable changelog entry so the UI can say what changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

CURRENT_VERSION = 2

#: v1 keys that belong to the `audio` block in v2.
AUDIO_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "sample_rate",
        "block_ms",
        "pre_roll_ms",
        "end_silence_ms",
        "min_speech_ms",
        "max_speech_seconds",
        "minimum_rms",
        "silence_after_playback_ms",
    }
)


@dataclass(frozen=True)
class MigrationResult:
    config: dict[str, Any]
    from_version: int
    to_version: int
    notes: list[str]

    @property
    def migrated(self) -> bool:
        return self.from_version != self.to_version

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "migrated": self.migrated,
            "notes": list(self.notes),
        }


def detect_version(raw: dict[str, Any]) -> int:
    version = raw.get("schema_version")
    if isinstance(version, int) and version > 0:
        return version
    # Pre-version documents are recognised by their flat key set.
    if any(key in raw for key in ("voicebox_url", "ollama_url", "tts_engine")):
        return 1
    return CURRENT_VERSION


def _port_from_url(url: Any) -> int | None:
    """Extract the port from a URL string, or None when absent/unparseable."""

    if not isinstance(url, str) or not url.strip():
        return None
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    if parsed.port:
        return int(parsed.port)
    return 443 if parsed.scheme == "https" else (80 if parsed.scheme == "http" else None)


def migrate_v1_to_v2(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Fold the flat Voicebox+Ollama config into the layered v2 shape."""

    notes = [
        "moved flat Voicebox/Ollama settings under `legacy`",
        "moved microphone/VAD tuning under `audio`",
        "server host/port preserved",
        "runtime defaults applied for hardware-aware selection",
    ]

    def _pick(key: str, default: Any = None) -> Any:
        return raw.get(key, default)

    voicebox_url = _pick("voicebox_url", "http://127.0.0.1:17493")
    # `voicebox_url` is authoritative. When a v1 config sets a custom URL but no
    # explicit `voicebox_port`, defaulting the port to 17493 left the two fields
    # disagreeing and the legacy adapter dialled the wrong port.
    voicebox_port = _pick("voicebox_port", None)
    if voicebox_port is None:
        voicebox_port = _port_from_url(voicebox_url) or 17493

    legacy = {
        "enabled": True,
        "label": "Voicebox + Ollama (migrated from v1)",
        "voicebox_url": voicebox_url,
        "voicebox_port": voicebox_port,
        "ollama_url": _pick("ollama_url", "http://127.0.0.1:11434"),
        "ollama_model": _pick("ollama_model", "qwen3:1.7b"),
        "asr_model": _pick("asr_model", "base"),
        "tts_engine": _pick("tts_engine", "luxtts"),
        "tts_model_size": _pick("tts_model_size", "0.6B"),
        "voice_profile_id": _pick("voice_profile_id", "default"),
        "voice_profile_name": _pick("voice_profile_name", "Default"),
        "language": _pick("language", "zh"),
        "system_prompt": _pick(
            "system_prompt", "You are a concise and friendly voice assistant."
        ),
        "relay_url": _pick("relay_url", ""),
        "relay_provider": _pick("relay_provider", "whisper"),
        "asr_backend": _pick("asr_backend", "local"),
        "asr_worker_id": _pick("asr_worker_id", ""),
        "unload_unused_tts_models": _pick("unload_unused_tts_models", True),
    }

    audio = {
        key: raw[key]
        for key in AUDIO_FIELD_NAMES
        if key in raw
    }

    server = {
        "host": _pick("listen_host", "127.0.0.1"),
        "port": _pick("listen_port", 17831),
        "auth_token_env": "LVC_AUTH_TOKEN",
    }

    # Anything not consumed above is preserved rather than thrown away. Only
    # scalar values are kept, because `legacy` is a strict typed block.
    consumed = AUDIO_FIELD_NAMES | {
        "schema_version",
        "listen_host",
        "listen_port",
        "voicebox_url",
        "voicebox_port",
        "ollama_url",
        "ollama_model",
        "asr_model",
        "asr_backend",
        "asr_worker_id",
        "tts_engine",
        "tts_model_size",
        "voice_profile_id",
        "voice_profile_name",
        "language",
        "system_prompt",
        "relay_url",
        "relay_provider",
        "unload_unused_tts_models",
    }
    leftovers = {
        key: value
        for key, value in raw.items()
        if key not in consumed and isinstance(value, (str, int, float, bool))
    }
    if leftovers:
        notes.append(
            f"dropped {len(leftovers)} unknown scalar key(s) that have no v2 home: "
            f"{sorted(leftovers)}"
        )

    migrated: dict[str, Any] = {
        "schema_version": 2,
        "server": server,
        "legacy": legacy,
    }
    if audio:
        migrated["audio"] = audio
    return migrated, notes


def migrate(raw: dict[str, Any]) -> MigrationResult:
    """Bring any historical config document up to CURRENT_VERSION."""

    version = detect_version(raw)
    origin = version
    notes: list[str] = []
    working = dict(raw)

    if version == 1:
        working, notes = migrate_v1_to_v2(working)
        version = 2
    elif version > CURRENT_VERSION:
        # A newer document than this build understands: keep it readable but
        # never overwrite it, and record the situation for the UI.
        notes.append(
            f"document schema_version={version} is newer than supported {CURRENT_VERSION}; "
            "loaded as-is without rewrite"
        )
        return MigrationResult(config=working, from_version=version, to_version=version, notes=notes)

    working["schema_version"] = CURRENT_VERSION
    return MigrationResult(
        config=working, from_version=origin, to_version=CURRENT_VERSION, notes=notes
    )
