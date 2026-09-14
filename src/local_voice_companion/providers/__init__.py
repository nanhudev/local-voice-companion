"""Pluggable ASR / LLM / TTS / VAD implementations.

Importing this package does NOT touch the registry. Call
:func:`register_builtin_providers` explicitly (or use
:func:`ensure_builtin_providers`) so that tests can build an isolated registry
without the builtin set leaking in.
"""

from __future__ import annotations

from .base import (
    ASRProvider,
    BaseProvider,
    LLMProvider,
    ModelRef,
    ProviderDescriptor,
    ProviderHealth,
    TTSProvider,
    VADProvider,
    VoiceRef,
    descriptors_to_wire,
)
from .registry import (
    ProviderRegistry,
    Registration,
    get_registry,
    register,
    register_all,
    registry,
)

__all__ = [
    "ASRProvider",
    "BaseProvider",
    "LLMProvider",
    "ModelRef",
    "ProviderDescriptor",
    "ProviderHealth",
    "ProviderRegistry",
    "Registration",
    "TTSProvider",
    "VADProvider",
    "VoiceRef",
    "descriptors_to_wire",
    "get_registry",
    "register",
    "register_all",
    "register_builtin_providers",
    "ensure_builtin_providers",
    "registry",
]

_builtins_registered = False


def register_builtin_providers(target: ProviderRegistry | None = None) -> list[str]:
    """Register the built-in provider set (fake + legacy adapters).

    Idempotent: a provider id that is already registered with the same class is
    skipped silently. Returns the ids now present for the requested kinds.
    """

    from .fake import FAKE_PROVIDERS
    from .legacy import LEGACY_PROVIDERS

    reg = target or registry
    for cls in (*FAKE_PROVIDERS, *LEGACY_PROVIDERS):
        reg.register(cls)
    return reg.ids()


def ensure_builtin_providers(target: ProviderRegistry | None = None) -> ProviderRegistry:
    """Register builtins once, then return the registry to use."""

    global _builtins_registered
    reg = target or registry
    if not _builtins_registered or target is not None:
        register_builtin_providers(reg)
        _builtins_registered = True
    return reg
