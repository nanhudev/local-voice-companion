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
    """Register the built-in provider set.

    Three tiers, in increasing order of what they need from the machine:

    * ``fake``   -- deterministic doubles used by CI. No dependencies at all.
    * ``local``  -- native inference on this machine (faster-whisper, Kokoro).
    * ``legacy`` -- HTTP adapters for an existing Voicebox / Ollama install.

    The local tier is imported lazily and its import failure is swallowed on
    purpose: a user with neither onnxruntime nor CTranslate2 still gets a
    working runtime, just one that reports those providers as unavailable. An
    ImportError escaping here would take down the whole application because two
    optional extras are missing.
    """

    from .fake import FAKE_PROVIDERS
    from .legacy import LEGACY_PROVIDERS

    # Explicit None check, never `target or registry`: an empty registry is
    # falsy under __len__, so the `or` form would quietly redirect every
    # registration into the module singleton -- including the isolated
    # registries the tests build on purpose.
    reg = registry if target is None else target

    classes: list[type[BaseProvider]] = [*FAKE_PROVIDERS]
    try:
        from .local import NATIVE_PROVIDERS

        classes.extend(NATIVE_PROVIDERS)
    except Exception:  # noqa: BLE001 - optional tier, see docstring
        pass
    classes.extend(LEGACY_PROVIDERS)

    for cls in classes:
        reg.register(cls)
    return reg.ids()


def ensure_builtin_providers(target: ProviderRegistry | None = None) -> ProviderRegistry:
    """Register builtins once, then return the registry to use.

    With an explicit ``target`` the builtins are always (re-)registered there,
    so each caller's isolated registry gets a full set. With no argument the
    module singleton is populated at most once.
    """

    global _builtins_registered
    reg = registry if target is None else target
    if target is not None:
        register_builtin_providers(reg)
        return reg
    if not _builtins_registered:
        register_builtin_providers(reg)
        _builtins_registered = True
    return reg
