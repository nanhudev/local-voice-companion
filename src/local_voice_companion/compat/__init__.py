"""Legacy compatibility adapters.

These adapters exist so the pre-2.0 single-file runtime (`app.py`, `config.json`,
Voicebox, Ollama) keeps working while the new layered runtime grows. They must
never become the source of truth for anything else.
"""

from __future__ import annotations

from .legacy_config import LegacyConfigAdapter, load_legacy_config
from .voicebox import LegacyVoiceService, create_legacy_http_server

__all__ = [
    "LegacyConfigAdapter",
    "load_legacy_config",
    "LegacyVoiceService",
    "create_legacy_http_server",
]
