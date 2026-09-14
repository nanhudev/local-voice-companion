"""Read the historical flat `config.json` through the new typed schema.

The old file is never rewritten by this module. It is read, migrated in memory
and exposed as both a dict (for old call sites) and a typed SystemConfig.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.loader import load_config, save_config
from ..config.migration import MigrationResult, migrate
from ..config.paths import LEGACY_CONFIG_EXAMPLE, LEGACY_CONFIG_PATH
from ..config.schema import SystemConfig


def load_legacy_config(path: Path | None = None) -> tuple[dict[str, Any], MigrationResult]:
    """Return (flat dict in the old shape, migration record)."""

    target = Path(path) if path else (
        LEGACY_CONFIG_PATH if LEGACY_CONFIG_PATH.is_file() else LEGACY_CONFIG_EXAMPLE
    )
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    result = migrate(raw)
    return raw, result


@dataclass
class LegacyConfigAdapter:
    """Bridges the flat v1 document and the layered v2 document.

    `as_flat()` produces exactly the key set the old `app.py` expects, so the
    legacy path can be deleted later without touching its call sites today.
    """

    path: Path | None = None

    def typed(self) -> SystemConfig:
        if self.path:
            return load_config(self.path)
        raw, _result = load_legacy_config()
        return load_config()

    def legacy_block(self):
        return self.typed().legacy

    def as_flat(self) -> dict[str, Any]:
        """The old flat document, reconstructed from the typed config."""

        config = self.typed()
        legacy = config.legacy
        flat: dict[str, Any] = {
            "schema_version": 1,
            "listen_host": config.server.host,
            "listen_port": config.server.port,
            "voicebox_url": legacy.voicebox_url,
            "relay_url": legacy.relay_url,
            "relay_provider": legacy.relay_provider,
            "asr_backend": legacy.asr_backend,
            "asr_worker_id": legacy.asr_worker_id,
            "ollama_url": legacy.ollama_url,
            "ollama_model": legacy.ollama_model,
            "voice_profile_id": legacy.voice_profile_id,
            "voice_profile_name": legacy.voice_profile_name,
            "language": legacy.language,
            "asr_model": legacy.asr_model,
            "tts_engine": legacy.tts_engine,
            "tts_model_size": legacy.tts_model_size,
            "unload_unused_tts_models": legacy.unload_unused_tts_models,
            "sample_rate": config.audio.sample_rate,
            "block_ms": config.audio.block_ms,
            "pre_roll_ms": config.audio.pre_roll_ms,
            "end_silence_ms": config.audio.end_silence_ms,
            "min_speech_ms": config.audio.min_speech_ms,
            "max_speech_seconds": config.audio.max_speech_seconds,
            "minimum_rms": config.audio.minimum_rms,
            "silence_after_playback_ms": config.audio.silence_after_playback_ms,
            "system_prompt": legacy.system_prompt,
        }
        return flat

    def persist_flat_patch(self, patch: dict[str, Any]) -> SystemConfig:
        """Apply old-style settings onto the typed config and save it."""

        answer = self.typed()
        legacy = answer.legacy.model_copy()
        mapping = {
            "voicebox_url": "voicebox_url",
            "relay_url": "relay_url",
            "relay_provider": "relay_provider",
            "asr_backend": "asr_backend",
            "asr_worker_id": "asr_worker_id",
            "ollama_url": "ollama_url",
            "ollama_model": "ollama_model",
            "voice_profile_id": "voice_profile_id",
            "voice_profile_name": "voice_profile_name",
            "language": "language",
            "asr_model": "asr_model",
            "tts_engine": "tts_engine",
            "tts_model_size": "tts_model_size",
            "system_prompt": "system_prompt",
            "unload_unused_tts_models": "unload_unused_tts_models",
        }
        audio = answer.audio.model_copy()
        audio_fields = {
            "sample_rate",
            "block_ms",
            "pre_roll_ms",
            "end_silence_ms",
            "min_speech_ms",
            "max_speech_seconds",
            "minimum_rms",
            "silence_after_playback_ms",
        }
        server = answer.server.model_copy()

        for key, value in patch.items():
            if key in mapping:
                setattr(legacy, mapping[key], value)
            elif key in audio_fields:
                setattr(audio, key, value)
            elif key == "listen_host":
                server.host = value
            elif key == "listen_port":
                server.port = int(value)

        updated = answer.model_copy(update={"legacy": legacy, "audio": audio, "server": server})
        if self.path:
            save_config(updated, self.path)
        return updated
