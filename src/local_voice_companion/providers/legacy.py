"""Compatibility providers for the pre-2.0 Voicebox + Ollama stack.

IMPORTANT ARCHITECTURE RULE
---------------------------
Voicebox and Ollama are *providers* here, not infrastructure. Nothing outside
this module knows these names, no core code branches on them, and the runtime
works perfectly well if all three are unavailable.

Phase 1 keeps the exact wire calls of the old gateway (`/transcribe`,
`/generate/stream`, `/api/chat`) so existing installations keep working while
native providers land later.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator, Mapping

from ..core.errors import AuthenticationError, ProviderLoadError, ProviderUnavailable
from ..core.types import (
    AudioChunk,
    AudioFormat,
    ChatMessage,
    Device,
    ProviderKind,
    QualitySource,
)
from ..pipeline.wav import clean_model_text, wav_bytes
from .base import (
    ASRProvider,
    LLMProvider,
    ModelRef,
    ProviderDescriptor,
    ProviderHealth,
    TTSProvider,
    VoiceRef,
)

DEFAULT_TIMEOUT = 90


def _http():
    import requests  # imported lazily so the package imports without requests

    return requests.Session()


def _resolve_secret(env_name: str | None) -> str:
    if not env_name:
        return ""
    return os.getenv(env_name, "")


class _HttpBacked:
    """Mixin for HTTP-backed providers. Requests run in a worker thread."""

    default_base_url: str = "http://127.0.0.1:17493"

    def __init__(self, options: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(options, **kwargs)
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = _http()
        return self._session

    @property
    def base_url(self) -> str:
        return str(self.options.get("base_url") or self.default_base_url).rstrip("/")

    async def _request(self, method: str, url: str, **kwargs: Any):
        def call() -> Any:
            return self.session.request(method, url, timeout=kwargs.pop("timeout", DEFAULT_TIMEOUT), **kwargs)

        try:
            return await asyncio.to_thread(call)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated to a typed error
            raise ProviderUnavailable(
                f"{self.descriptor().id} request failed: {exc}", provider=self.descriptor().id
            ) from exc


# ---------------------------------------------------------------------------
# Voicebox ASR
# ---------------------------------------------------------------------------


class VoiceboxASR(_HttpBacked, ASRProvider):
    """Wraps POST /transcribe on a Voicebox-compatible service."""

    kind = ProviderKind.ASR
    default_base_url = "http://127.0.0.1:17493"

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        return ProviderDescriptor(
            id="voicebox_asr",
            kind=ProviderKind.ASR,
            display_name="Voicebox Whisper (compatibility)",
            languages=("zh", "en", "ja", "yue"),
            devices=(Device.REMOTE,),
            streaming=False,
            estimated_ram_mb=64,
            estimated_vram_mb=0,
            quality_tier=3,
            latency_tier=3,
            quality_source=QualitySource.CURATED_METADATA,
            version="1.0.0",
            # Honest even on loopback: the runtime's view of this provider is an
            # HTTP request with an HTTP request's failure modes. Leaving this at
            # the default `False` let it survive `allow_network=False` whenever
            # `cpu_only` was off, which contradicted ADR-0002 and quietly
            # weakened the offline guarantee.
            requires_network=True,
            tags=("http://127.0.0.1:17493", "voicebox", "legacy", "compatibility"),
            models=tuple(
                ModelRef(id=name, display_name=name, languages=("zh", "en"))
                for name in ("base", "small", "medium", "large", "turbo")
            ),
        )

    async def probe(self) -> ProviderHealth:
        try:
            response = await self._request("GET", f"{self.base_url}/health", timeout=4)
        except ProviderUnavailable as exc:
            self.lifecycle.transition("UNAVAILABLE", str(exc))
            return ProviderHealth(ok=False, detail=str(exc))
        ok = response.ok
        self.lifecycle.transition("AVAILABLE" if ok else "UNAVAILABLE")
        return ProviderHealth(ok=ok, detail=f"HTTP {response.status_code}")

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str = "",
        model: str | None = None,
        token: Any = None,
    ) -> str:
        payload_model = model or self.options.get("asr_model") or "base"
        blob = wav_bytes(audio.pcm, audio.sample_rate)
        try:
            response = await self._request(
                "POST",
                f"{self.base_url}/transcribe",
                files={"file": ("utterance.wav", blob, "audio/wav")},
                data={"language": language or self.options.get("language", "zh"), "model": payload_model},
                timeout=90,
            )
        except ProviderUnavailable:
            raise
        if response.status_code >= 400:
            raise ProviderUnavailable(
                f"voicebox ASR HTTP {response.status_code}: {response.text[:200]}",
                status=response.status_code,
            )
        return str(response.json().get("text", "")).strip()


# ---------------------------------------------------------------------------
# Voicebox TTS
# ---------------------------------------------------------------------------


class VoiceboxTTS(_HttpBacked, TTSProvider):
    """Wraps POST /generate/stream on a Voicebox-compatible service."""

    kind = ProviderKind.TTS
    default_base_url = "http://127.0.0.1:17493"

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        return ProviderDescriptor(
            id="voicebox_tts",
            kind=ProviderKind.TTS,
            display_name="Voicebox TTS (compatibility)",
            languages=("zh", "en", "ja"),
            devices=(Device.REMOTE,),
            streaming=True,
            estimated_ram_mb=64,
            quality_tier=4,
            latency_tier=3,
            quality_source=QualitySource.CURATED_METADATA,
            version="1.0.0",
            requires_network=True,
            tags=("http://127.0.0.1:17493", "voicebox", "legacy", "compatibility"),
            models=(
                ModelRef(id="luxtts", display_name="LuxTTS", languages=("zh", "en")),
                ModelRef(id="qwen-0.6B", display_name="Qwen3 TTS 0.6B", languages=("zh", "en")),
                ModelRef(id="qwen-1.7B", display_name="Qwen3 TTS 1.7B", languages=("zh", "en")),
            ),
            voices=(VoiceRef(id="default", display_name="Default"),),
        )

    async def probe(self) -> ProviderHealth:
        try:
            response = await self._request("GET", f"{self.base_url}/health", timeout=4)
        except ProviderUnavailable as exc:
            self.lifecycle.transition("UNAVAILABLE", str(exc))
            return ProviderHealth(ok=False, detail=str(exc))
        ok = response.ok
        self.lifecycle.transition("AVAILABLE" if ok else "UNAVAILABLE")
        return ProviderHealth(ok=ok, detail=f"HTTP {response.status_code}")

    async def list_voice_refs(self) -> tuple[VoiceRef, ...]:
        try:
            response = await self._request("GET", f"{self.base_url}/profiles", timeout=5)
        except ProviderUnavailable:
            return self.descriptor().voices
        if not response.ok:
            return self.descriptor().voices
        try:
            profiles = response.json()
        except ValueError:
            return self.descriptor().voices
        return tuple(
            VoiceRef(id=str(item.get("id")), display_name=str(item.get("name", item.get("id"))))
            for item in profiles
            if item.get("id")
        ) or self.descriptor().voices

    def list_voices(self):
        return self.descriptor().voices

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str = "",
        model: str | None = None,
        token: Any = None,
    ) -> tuple[bytes, AudioFormat]:
        engine = model or self.options.get("tts_engine") or "luxtts"
        model_size = self.options.get("tts_model_size") or "0.6B"
        payload = {
            "profile_id": voice or self.options.get("voice_profile_id") or "default",
            "text": text,
            "language": language or self.options.get("language", "zh"),
            "engine": engine,
            "model_size": model_size,
            "normalize": True,
            "max_chunk_chars": int(self.options.get("max_chunk_chars", 220)),
            "crossfade_ms": int(self.options.get("crossfade_ms", 25)),
        }
        response = await self._request(
            "POST", f"{self.base_url}/generate/stream", json=payload, timeout=180
        )
        if response.status_code >= 400:
            raise ProviderUnavailable(
                f"voicebox TTS HTTP {response.status_code}: {response.text[:200]}",
                status=response.status_code,
            )
        content = response.content or b""
        if not content.startswith(b"RIFF"):
            raise ProviderUnavailable("voicebox did not return WAV audio")
        return content, AudioFormat(
            mime_type=response.headers.get("Content-Type", "audio/wav").split(";")[0]
        )


# ---------------------------------------------------------------------------
# Ollama LLM
# ---------------------------------------------------------------------------


class OllamaLLM(_HttpBacked, LLMProvider):
    """Streams from POST /api/chat on an Ollama-compatible endpoint."""

    kind = ProviderKind.LLM
    default_base_url = "http://127.0.0.1:11434"

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        return ProviderDescriptor(
            id="ollama_llm",
            kind=ProviderKind.LLM,
            display_name="Ollama (local models)",
            languages=("zh", "en"),
            devices=(Device.CUDA, Device.CPU, Device.METAL, Device.REMOTE),
            streaming=True,
            estimated_ram_mb=512,
            estimated_vram_mb=4096,
            quality_tier=4,
            latency_tier=3,
            quality_source=QualitySource.CURATED_METADATA,
            version="1.0.0",
            # Same reasoning as the Voicebox pair: the inference may be local to
            # the host, but reaching it is still an HTTP call that can fail in
            # ways a truly in-process provider cannot.
            requires_network=True,
            tags=("http://127.0.0.1:11434", "ollama", "local"),
            models=(ModelRef(id="auto", display_name="Auto-discovered model"),),
        )

    async def probe(self) -> ProviderHealth:
        try:
            response = await self._request("GET", f"{self.base_url}/api/tags", timeout=4)
        except ProviderUnavailable as exc:
            self.lifecycle.transition("UNAVAILABLE", str(exc))
            return ProviderHealth(ok=False, detail=str(exc))
        ok = response.ok
        self.lifecycle.transition("AVAILABLE" if ok else "UNAVAILABLE")
        return ProviderHealth(ok=ok, detail=f"HTTP {response.status_code}")

    async def list_models(self) -> tuple[ModelRef, ...]:
        try:
            response = await self._request("GET", f"{self.base_url}/api/tags", timeout=5)
        except ProviderUnavailable:
            return self.descriptor().models
        try:
            names = [item.get("name", "") for item in response.json().get("models", [])]
        except ValueError:
            return self.descriptor().models
        return tuple(ModelRef(id=name, display_name=name) for name in names if name) or self.descriptor().models

    def _model(self, model: str | None) -> str:
        return model or self.options.get("ollama_model") or "qwen3:1.7b"

    async def stream(
        self,
        messages: Any,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        token: Any = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self._model(model),
            "messages": [message.to_dict() for message in messages],
            "stream": True,
            "think": False,
            "keep_alive": self.options.get("keep_alive", "30m"),
            "options": {
                "temperature": self.options.get("temperature", temperature if temperature is not None else 0.35),
                "num_predict": int(max_tokens or self.options.get("max_tokens", 64)),
                "num_ctx": int(self.options.get("num_ctx", 1024)),
            },
        }

        def _post():
            return self.session.post(
                f"{self.base_url}/api/chat", json=payload, stream=True, timeout=DEFAULT_TIMEOUT
            )

        try:
            response = await asyncio.to_thread(_post)
        except Exception as exc:  # noqa: BLE001
            raise ProviderUnavailable(f"ollama request failed: {exc}", provider="ollama_llm") from exc

        if not response.ok:
            raise ProviderUnavailable(
                f"ollama HTTP {response.status_code}: {response.text[:200]}",
                status=response.status_code,
            )

        def lines():
            for line in response.iter_lines(chunk_size=1):
                if not line:
                    continue
                try:
                    payload_line = json.loads(line)
                except ValueError:
                    continue
                piece = payload_line.get("message", {}).get("content", "")
                if piece:
                    yield piece

        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=64)
        loop = asyncio.get_running_loop()

        def pump() -> None:
            try:
                for piece in lines():
                    loop.call_soon_threadsafe(queue.put_nowait, piece)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        import threading

        threading.Thread(target=pump, daemon=True).start()

        while True:
            piece = await queue.get()
            if piece is None:
                break
            if token is not None and token.cancelled:
                response.close()
                raise ConnectionAbortedError("turn cancelled")
            yield piece

    async def warm(self, model: str | None = None) -> None:
        """Load the model so the first real turn does not pay the cold start."""

        payload = {
            "model": self._model(model),
            "prompt": "",
            "keep_alive": self.options.get("keep_alive", "30m"),
            "stream": False,
            "options": {"num_ctx": int(self.options.get("num_ctx", 1024))},
        }
        response = await self._request("POST", f"{self.base_url}/api/generate", json=payload, timeout=120)
        if response.status_code >= 400:
            raise ProviderLoadError(
                f"ollama warmup HTTP {response.status_code}", provider="ollama_llm"
            )


LEGACY_PROVIDERS = (VoiceboxASR, VoiceboxTTS, OllamaLLM)


def legacy_options(config: Any) -> dict[str, dict[str, Any]]:
    """Translate the compat block into per-provider options (secrets by env ref)."""

    legacy = getattr(config, "legacy", None)
    if legacy is None:
        return {}
    token_env = getattr(legacy, "relay_token_env", "AI_RELAY_TOKEN")
    common: dict[str, Any] = {"language": getattr(legacy, "language", "zh")}
    return {
        "voicebox_asr": {
            "base_url": legacy.voicebox_url,
            "asr_model": legacy.asr_model,
            **common,
        },
        "voicebox_tts": {
            "base_url": legacy.voicebox_url,
            "tts_engine": legacy.tts_engine,
            "tts_model_size": legacy.tts_model_size,
            "voice_profile_id": legacy.voice_profile_id,
            **common,
        },
        "ollama_llm": {
            "base_url": legacy.ollama_url,
            "ollama_model": legacy.ollama_model,
            "relay_token_env": token_env,
            **common,
        },
    }


def require_relay_token(env_name: str) -> str:
    value = _resolve_secret(env_name)
    if not value:
        raise AuthenticationError(f"{env_name} is not set", env=env_name)
    return value
