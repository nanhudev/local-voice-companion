"""Provider contracts.

Core never says `if engine == "kokoro"`. It asks every registered provider for
its descriptor, filters by what the hardware can actually do, scores the
survivors, and talks to the winner only through these interfaces.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from ..core.audio import AudioFrame
    from ..core.transcriber import TranscriptUpdate

from ..core.cancellation import CancellationToken
from ..core.errors import ProviderLoadError, ProviderUnavailable
from ..core.lifecycle import SERVING_STATES, Lifecycle, ProviderState
from ..core.types import (
    AudioChunk,
    AudioFormat,
    ChatMessage,
    Device,
    ProviderKind,
    QualitySource,
    ResourceRequirements,
)


@dataclass(frozen=True)
class ProviderDescriptor:
    """Machine-readable capability description. This is the anti-branch contract."""

    id: str
    kind: ProviderKind
    display_name: str
    languages: tuple[str, ...] = ()
    devices: tuple[Device, ...] = (Device.CPU,)
    streaming: bool = False
    # Two finer-grained streaming facts. `streaming` only ever meant "produces
    # output progressively", which is not enough to answer the questions duplex
    # actually asks:
    #   supports_streaming       -> consumes audio that is still arriving
    #   supports_partial_results -> emits hypotheses before audio ends
    # A provider can have either without the other (a turn-based engine that is
    # fed frames still has no partials), so they are separate flags rather than
    # one overloaded one. Both default to False, which is additive: every
    # existing descriptor keeps its meaning.
    supports_streaming: bool = False
    supports_partial_results: bool = False
    # Resource envelope used by the PipelineResourcePlanner.
    estimated_ram_mb: int = 0
    estimated_vram_mb: int = 0
    estimated_disk_mb: int = 0
    requires_network: bool = False
    is_local: bool = True
    # 1..5 tiers. Tier numbers are ordinal, not percentages.
    quality_tier: int = 3
    latency_tier: int = 3
    # Quality is never invented. `quality_source` records provenance.
    quality_score: float | None = None
    quality_source: QualitySource = QualitySource.UNKNOWN
    supports_cancellation: bool = True
    version: str = "0.0.0"
    tags: tuple[str, ...] = ()
    models: tuple["ModelRef", ...] = ()
    voices: tuple["VoiceRef", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["devices"] = [item.value for item in self.devices]
        payload["quality_source"] = self.quality_source.value
        payload["models"] = [item.to_dict() for item in self.models]
        payload["voices"] = [item.to_dict() for item in self.voices]
        payload["languages"] = list(self.languages)
        payload["tags"] = list(self.tags)
        return payload


@dataclass(frozen=True)
class ModelRef:
    """Model identifier owned by a provider. The runtime treats these as opaque."""

    id: str
    display_name: str = ""
    languages: tuple[str, ...] = ()
    parameter_size: str = ""
    quantization: str = ""
    disk_mb: int = 0
    context_window: int = 0
    requires_network: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["languages"] = list(self.languages)
        return payload


@dataclass(frozen=True)
class VoiceRef:
    """A preset speaker. Voice cloning is explicitly out of scope for phase 1."""

    id: str
    display_name: str = ""
    languages: tuple[str, ...] = ()
    gender: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["languages"] = list(self.languages)
        return payload


@dataclass
class ProviderHealth:
    ok: bool
    detail: str = ""
    latency_ms: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "detail": self.detail, "latency_ms": self.latency_ms, **self.extra}


class BaseProvider(ABC):
    """Common lifecycle plumbing for every provider family."""

    kind: ProviderKind

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self.options: dict[str, Any] = dict(options or {})
        self.lifecycle = Lifecycle(self.descriptor().id)

    # -- static description -------------------------------------------------

    @staticmethod
    @abstractmethod
    def descriptor() -> ProviderDescriptor:
        """Return the capability descriptor. Must not perform IO."""

    # -- lifecycle ----------------------------------------------------------

    async def probe(self) -> ProviderHealth:
        """Cheap availability check: dependency present? endpoint reachable?

        Default implementation assumes the provider is dependency-free and
        therefore always available. Real providers override it.
        """

        try:
            self.lifecycle.transition(ProviderState.AVAILABLE)
        except Exception:  # pragma: no cover - defensive, transitions validated by tests
            pass
        return ProviderHealth(ok=True, detail="no external dependency")

    async def load(self, model: str | None = None, device: str = Device.CPU.value) -> None:
        """Bring the provider into READY.

        The only legal path into LOADING is from AVAILABLE, so if the provider
        has not been probed yet (still DISCOVERED) or was left in ERROR by an
        earlier attempt, probe implicitly first. Callers should not have to know
        the lifecycle protocol to load a provider -- `discover()` probes in bulk
        for the runtime, but a test or a CLI may just call `load()` directly.
        """

        await self._ensure_probed()
        try:
            self.lifecycle.transition(ProviderState.LOADING)
        except Exception as exc:
            raise ProviderLoadError(str(exc), provider=self.descriptor().id) from exc
        self.lifecycle.transition(ProviderState.READY)

    async def _ensure_probed(self) -> None:
        """Move DISCOVERED / ERROR into AVAILABLE so `load()` is always legal."""

        state = self.lifecycle.state
        if state in SERVING_STATES or state is ProviderState.LOADING:
            return
        if state in {ProviderState.DISCOVERED, ProviderState.ERROR}:
            health = await self.probe()
            if not health.ok:
                raise ProviderUnavailable(
                    health.detail or f"{self.descriptor().id} is unavailable",
                    provider=self.descriptor().id,
                    state=self.lifecycle.state.value,
                )
            return
        if state is ProviderState.UNAVAILABLE:
            raise ProviderUnavailable(
                self.lifecycle.detail or f"{self.descriptor().id} is unavailable",
                provider=self.descriptor().id,
                state=state.value,
            )

    async def unload(self) -> None:
        self.lifecycle.transition(ProviderState.UNLOADING)
        self.lifecycle.transition(ProviderState.AVAILABLE)

    async def health(self) -> ProviderHealth:
        started = time.monotonic()
        ok = self.lifecycle.is_serving
        return ProviderHealth(
            ok=ok,
            detail=self.lifecycle.detail,
            latency_ms=int(round((time.monotonic() - started) * 1000)),
            extra={"state": self.lifecycle.state.value},
        )

    # -- capability helpers -------------------------------------------------

    def capabilities(self) -> ProviderDescriptor:
        return self.descriptor()

    def resource_requirements(self, model: str | None = None) -> ResourceRequirements:
        descriptor = self.descriptor()
        return ResourceRequirements(
            ram_mb=descriptor.estimated_ram_mb,
            vram_mb=descriptor.estimated_vram_mb,
            disk_mb=descriptor.estimated_disk_mb,
            requires_network=descriptor.requires_network,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} id={self.descriptor().id}>"


class ASRProvider(BaseProvider):
    kind = ProviderKind.ASR

    @abstractmethod
    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> str:
        """Transcribe one complete utterance."""

    async def stream(
        self,
        frames: AsyncIterator[AudioChunk],
        *,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> AsyncIterator[str]:
        """Optional incremental transcription.

        Default implementation accumulates into a single final result, so a
        provider that cannot stream still satisfies the contract.
        """

        # `raise NotImplementedError` here would break the uniform interface;
        # instead we fold everything into one final transcript.
        if False:  # pragma: no cover - keeps the body async-typed
            yield ""
        raise RuntimeError(
            f"{self.descriptor().id} does not implement stream(); check streaming=False"
        )

    async def stream_transcribe(
        self,
        frames: AsyncIterator["AudioFrame"],
        *,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> AsyncIterator["TranscriptUpdate"]:
        """Transcribe audio that is still arriving, yielding intermediate results.

        The default implementation does not pretend to stream: it buffers every
        frame, calls :meth:`transcribe` once, and yields exactly one final update.
        That keeps the contract uniform -- callers never branch on provider type --
        while staying honest about capability. A provider whose descriptor says
        ``supports_partial_results = False`` yields no partials rather than
        manufacturing them by chopping up a whole-utterance result.

        Subclasses set `closed` implicitly: iteration ends when `frames` ends.
        """

        # Both imports are deliberately function-local, not TYPE_CHECKING-only:
        # the default implementation below *uses* TranscriptUpdate at runtime, and
        # putting it behind a typing guard turns a working path into NameError.
        from ..core.audio import concat_frames
        from ..core.transcriber import TranscriptUpdate

        collected: list[AudioFrame] = []
        async for frame in frames:
            if token is not None and token.cancelled:
                return
            collected.append(frame)
        if not collected:
            return
        text = await self.transcribe(
            concat_frames(collected), language=language, model=model, token=token
        )
        yield TranscriptUpdate(text=text, committed=text, unstable="", is_final=True)

    def supports(self, language: str) -> bool:
        from ..core.types import language_matches

        return language_matches(language, self.descriptor().languages)


class TTSProvider(BaseProvider):
    kind = ProviderKind.TTS

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> tuple[bytes, AudioFormat]:
        """Return one audio blob for the whole text."""

    async def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> AsyncIterator[bytes]:
        blob, _fmt = await self.synthesize(
            text, voice=voice, language=language, model=model, token=token
        )
        if blob:
            yield blob

    def list_voices(self) -> Sequence[VoiceRef]:
        return self.descriptor().voices

    def supports(self, language: str) -> bool:
        from ..core.types import language_matches

        return language_matches(language, self.descriptor().languages)


class LLMProvider(BaseProvider):
    kind = ProviderKind.LLM

    @abstractmethod
    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        token: CancellationToken | None = None,
    ) -> AsyncIterator[str]:
        """Stream text deltas."""

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        token: CancellationToken | None = None,
    ) -> str:
        chunks: list[str] = []
        async for delta in self.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            token=token,
        ):
            chunks.append(delta)
        return "".join(chunks)

    async def cancel(self, token: CancellationToken) -> None:
        token.cancel(code="provider_cancel", detail="cancelled by caller")

    async def list_models(self) -> Sequence[ModelRef]:
        return self.descriptor().models


class VADProvider(BaseProvider):
    kind = ProviderKind.VAD

    @abstractmethod
    def is_speech(self, frame: AudioChunk) -> bool:
        """Frame-level decision used by the capture loop."""


ProviderType = ASRProvider | TTSProvider | LLMProvider | VADProvider


def ensure_ready(provider: BaseProvider) -> None:
    """Guard used by the orchestrator before dispatching work."""

    if not provider.lifecycle.is_serving:
        raise ProviderUnavailable(
            f"{provider.descriptor().id} is {provider.lifecycle.state.value}, not READY",
            provider=provider.descriptor().id,
            state=provider.lifecycle.state.value,
        )


def descriptors_to_wire(providers: Iterable[BaseProvider]) -> list[dict[str, Any]]:
    return [provider.descriptor().to_dict() for provider in providers]
