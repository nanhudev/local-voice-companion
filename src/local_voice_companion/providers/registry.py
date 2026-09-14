"""Unified registry.

The runtime resolves providers by id only. There is exactly one place that
knows the set of implementations, and adding a new provider means registering
it here -- never adding a branch in core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..core.lifecycle import ProviderState
from ..core.types import ProviderKind
from ..core.errors import ConfigurationError, NotFound
from .base import ASRProvider, BaseProvider, LLMProvider, ProviderDescriptor, TTSProvider, VADProvider

FAMILY_BASE: dict[ProviderKind, type[BaseProvider]] = {
    ProviderKind.ASR: ASRProvider,
    ProviderKind.TTS: TTSProvider,
    ProviderKind.LLM: LLMProvider,
    ProviderKind.VAD: VADProvider,
}


@dataclass(frozen=True)
class Registration:
    cls: type[BaseProvider]
    descriptor: ProviderDescriptor
    origin: str = "builtin"


class ProviderRegistry:
    """Class -> descriptor catalogue, plus instance pool."""

    def __init__(self) -> None:
        self._classes: dict[str, Registration] = {}
        self._instances: dict[str, BaseProvider] = {}

    # -- class catalogue ----------------------------------------------------

    def register(self, cls: type[BaseProvider], origin: str = "builtin") -> None:
        descriptor = cls.descriptor()
        if descriptor.id in self._classes:
            existing = self._classes[descriptor.id]
            if existing.cls is cls:
                return
            raise ConfigurationError(
                "duplicate provider id", provider_id=descriptor.id, existing=existing.cls.__name__
            )
        family = FAMILY_BASE.get(descriptor.kind)
        if family is None or not issubclass(cls, family):
            raise ConfigurationError(
                f"{cls.__name__} must subclass {family.__name__ if family else 'a provider family'}"
            )
        self._classes[descriptor.id] = Registration(cls=cls, descriptor=descriptor, origin=origin)

    def register_instance(self, provider: BaseProvider, origin: str = "runtime") -> None:
        descriptor = provider.descriptor()
        self._classes[descriptor.id] = Registration(cls=type(provider), descriptor=descriptor, origin=origin)
        self._instances[descriptor.id] = provider

    def ids(self, kind: ProviderKind | None = None) -> list[str]:
        return sorted(
            key
            for key, registration in self._classes.items()
            if kind is None or registration.descriptor.kind is kind
        )

    def get_class(self, provider_id: str) -> type[BaseProvider]:
        registration = self._classes.get(provider_id)
        if registration is None:
            raise NotFound(f"provider not registered: {provider_id}", provider_id=provider_id)
        return registration.cls

    def registration(self, provider_id: str) -> Registration:
        registration = self._classes.get(provider_id)
        if registration is None:
            raise NotFound(f"provider not registered: {provider_id}", provider_id=provider_id)
        return registration

    def descriptors(self, kind: ProviderKind | None = None) -> list[ProviderDescriptor]:
        return [
            registration.descriptor
            for key, registration in sorted(self._classes.items())
            if kind is None or registration.descriptor.kind is kind
        ]

    def __contains__(self, provider_id: object) -> bool:
        return isinstance(provider_id, str) and provider_id in self._classes

    def __iter__(self) -> Iterator[Registration]:
        for key in sorted(self._classes):
            yield self._classes[key]

    def __len__(self) -> int:
        return len(self._classes)

    # -- instance pool ------------------------------------------------------

    def create(self, provider_id: str, options: Mapping[str, Any] | None = None) -> BaseProvider:
        cls = self.get_class(provider_id)
        return cls(options)

    async def acquire(
        self, provider_id: str, options: Mapping[str, Any] | None = None
    ) -> BaseProvider:
        """Singleton-per-id instance pool. Loads on first acquisition."""

        existing = self._instances.get(provider_id)
        if existing is not None:
            return existing
        provider = self.create(provider_id, options)
        self._instances[provider_id] = provider
        return provider

    def instance(self, provider_id: str) -> BaseProvider | None:
        return self._instances.get(provider_id)

    def instances(self) -> Sequence[BaseProvider]:
        return [self._instances[key] for key in sorted(self._instances)]

    async def release(self, provider_id: str) -> None:
        provider = self._instances.pop(provider_id, None)
        if provider is not None and provider.lifecycle.state not in {
            ProviderState.UNLOADING,
            ProviderState.UNAVAILABLE,
        }:
            await provider.unload()

    async def release_all(self) -> None:
        for provider_id in list(self._instances):
            await self.release(provider_id)

    def clear_instances(self) -> None:
        self._instances.clear()

    def clear(self) -> None:
        self._classes.clear()
        self._instances.clear()


registry = ProviderRegistry()


def register(cls: type[BaseProvider], origin: str = "builtin") -> type[BaseProvider]:
    registry.register(cls, origin=origin)
    return cls


def get_registry() -> ProviderRegistry:
    return registry


def register_all(providers: Iterable[type[BaseProvider]], origin: str = "builtin") -> None:
    for cls in providers:
        registry.register(cls, origin=origin)
