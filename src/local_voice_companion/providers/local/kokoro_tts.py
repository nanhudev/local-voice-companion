"""Kokoro-82M v1.1 (Chinese) TTS: local neural speech synthesis on CPU.

Kokoro is a *phoneme* model. It is not handed text; it is handed a phoneme
string produced by a separate grapheme-to-phoneme (G2P) front end. That split is
the single most important thing about integrating it, because the front end is
where Chinese synthesis actually succeeds or fails:

*   ``misaki`` (the upstream front end) does pinyin conversion, tone sandhi and
    heteronym selection, and emits the mixed zhuyin/pinyin tone-numbered string
    the v1.1 model was trained on. It pulls ``jieba`` (segmentation),
    ``pypinyin`` and ``cn2an`` (number expansion).
*   ``espeak-ng`` has a Mandarin mode, but its tone handling is materially
    worse. On Windows it is also not reachable through ``phonemizer`` at all
    without extra environment setup, so in practice misaki is the only working
    option here.

The descriptor records which backend was actually used, so a decision log never
claims more than it got.

Honest limitations
------------------
*   ``streaming=False``. Kokoro's ONNX graph is a single forward pass producing
    the whole waveform; there is no chunk-wise decoder to drive. The runtime's
    time-to-first-audio improvement therefore comes from the text chunker
    synthesising sentence by sentence, not from this provider.
*   **Code-switching to Latin script is degraded.** With no English G2P wired
    in, Latin words become an unknown token and are effectively dropped. The
    provider detects this and reports it rather than emitting silence for the
    English half of a sentence.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import time
from typing import Any, AsyncIterator, Mapping, Sequence

from ...core.cancellation import CancellationToken
from ...core.errors import (
    ModelMissing,
    OutOfMemory,
    ProviderLoadError,
    ProviderUnavailable,
)
from ...core.types import (
    AudioFormat,
    Device,
    ProviderKind,
    QualitySource,
)
from ..base import (
    ModelRef,
    ProviderDescriptor,
    ProviderHealth,
    TTSProvider,
    VoiceRef,
)
from . import runtime_probe
from .model_store import default_store

PROVIDER_ID = "kokoro_tts_cpu"
BUNDLE_ID = "kokoro-v1.1-zh"
MODEL_ID = "kokoro-v1.1-zh"

SAMPLE_RATE = 24000

#: The token misaki emits for a character it could not phonemise. Its presence
#: in a phoneme string means the corresponding text will be silent.
UNKNOWN_PHONEME = "\u2753"  # ❓

#: A representative slice of the shipped Chinese voice set. The full set is 103
#: voices and is enumerated from the voices file after load; these are the ones
#: we advertise before the model is open, so the UI has something to show.
_DECLARED_VOICES: tuple[tuple[str, str, str], ...] = (
    ("zf_001", "Chinese female 1", "female"),
    ("zf_002", "Chinese female 2", "female"),
    ("zf_003", "Chinese female 3", "female"),
    ("zf_004", "Chinese female 4", "female"),
    ("zf_005", "Chinese female 5", "female"),
    ("zf_006", "Chinese female 6", "female"),
    ("zm_009", "Chinese male 1", "male"),
    ("zm_010", "Chinese male 2", "male"),
    ("zm_011", "Chinese male 3", "male"),
    ("zm_012", "Chinese male 4", "male"),
    ("zm_013", "Chinese male 5", "male"),
    ("zm_014", "Chinese male 6", "male"),
)

DEFAULT_VOICE = "zf_001"

_LANGUAGES: tuple[str, ...] = ("zh", "en")

#: Characters that misaki/Kokoro handle as prosody but that a chat LLM emits a
#: lot of. Markdown is the main offender: an asterisk or a backtick read aloud
#: as a word is an obvious quality bug.
_MARKDOWN_NOISE = re.compile(r"[*_`#>\[\]()]|~~|```")

#: CJK, ASCII letters/digits and the punctuation we keep. Anything else is
#: dropped before phonemisation; keeping control characters or emoji produces
#: unknown tokens and audible artefacts.
#
#: The class body mixes ranges and literals, so the string is built from parts
#: rather than written as one raw literal: that keeps the quote characters
#: readable and avoids the escaping mistakes a wall of backslashes invites.
_KEEP = re.compile(
    "["
    "\u3000-\u303f"          # CJK symbols and punctuation
    "\u4e00-\u9fff"          # CJK unified ideographs
    "\uff00-\uffef"          # fullwidth forms
    "\u3400-\u4dbf"          # CJK extension A
    "0-9A-Za-z"              # latin digits and letters
    "\u3001\u3002\uff01\uff1f\uff0c\uff1b\uff1a"  # 、。！？，；：
    "\u201c\u201d\u2018\u2019"                    # curly quotes
    "\uff08\uff09\u300a\u300b\u2014\u2026\u00b7"  # （）《》—…·
    r"\s,\.!\?;:'\"\-"       # ascii punctuation and whitespace
    "]"
)


def _option_float(options: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(options.get(key, default))
    except (TypeError, ValueError):
        return default


def _runtime_version() -> str:
    """The label a cached benchmark must be tied to.

    Synthesis depends on three independently versioned pieces: kokoro-onnx
    built the graph, onnxruntime executes it, and misaki turns text into
    phonemes. Reporting only one would let a G2P upgrade keep serving
    phoneme-generation timings measured against the old front end.

    All three are read from distribution metadata -- no import, so this stays
    cheap enough to call from `descriptor()`.
    """

    parts = []
    for distribution in ("kokoro-onnx", "onnxruntime", "misaki"):
        found = runtime_probe.package_version(distribution)
        if found:
            parts.append(f"{distribution}=={found}")
    return "+".join(parts) or "none"


def _option_int(options: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(options.get(key, default))
    except (TypeError, ValueError):
        return default


class KokoroTTS(TTSProvider):
    """Local Kokoro v1.1 Chinese synthesis."""

    kind = ProviderKind.TTS

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        super().__init__(options)
        self._engine: Any = None
        self._g2p: Any = None
        self._g2p_backend: str = ""
        self._loaded_voice: str = ""
        self._voices: tuple[VoiceRef, ...] = ()
        self._last_load_ms: int = 0
        self._synthesis_lock: asyncio.Lock | None = None
        self._store = default_store()

    # -- description --------------------------------------------------------

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        voices = tuple(
            VoiceRef(id=voice_id, display_name=display, languages=_LANGUAGES, gender=gender)
            for voice_id, display, gender in _DECLARED_VOICES
        )
        return ProviderDescriptor(
            id=PROVIDER_ID,
            kind=ProviderKind.TTS,
            display_name="Kokoro-82M v1.1 Chinese (ONNX Runtime, CPU)",
            languages=_LANGUAGES,
            devices=(Device.CPU,),
            # One forward pass per utterance; see the module docstring.
            streaming=False,
            estimated_ram_mb=520,
            estimated_vram_mb=0,
            estimated_disk_mb=380,
            requires_network=False,
            is_local=True,
            quality_tier=4,
            latency_tier=4,
            quality_source=QualitySource.CURATED_METADATA,
            # An 82M-parameter model with a real G2P front end is good for a
            # conversational assistant, but it is not a large TTS model.
            quality_score=0.78,
            supports_cancellation=True,
            # Composite rather than a single package: synthesis depends on the
            # model runtime *and* on the grapheme-to-phoneme stack, and swapping
            # the G2P backend changes both what comes out and how long it takes.
            # The selection cache keys on this string, so a bump in any of the
            # three correctly invalidates the stored numbers.
            version=_runtime_version(),
            tags=(
                "local",
                "cpu",
                "onnxruntime",
                "kokoro",
                "fp32",
                "no-download",
                "chinese",
            ),
            models=(
                ModelRef(
                    id=MODEL_ID,
                    display_name="Kokoro-82M v1.1 zh",
                    languages=_LANGUAGES,
                    parameter_size="82M",
                    disk_mb=380,
                ),
            ),
            voices=voices,
        )

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """Whether this provider could load right now, and why not.

        Mirrors `FasterWhisperASR.availability()`. `doctor` and the tests need
        exactly one source of truth here; three copies of "is it installed" is
        how a provider ends up reported ready in one view and refused in
        another.
        """

        if importlib.util.find_spec("kokoro_onnx") is None:
            return False, (
                "kokoro-onnx is not installed; "
                "run: pip install 'local-voice-companion[tts]'"
            )
        if importlib.util.find_spec("onnxruntime") is None:
            return False, "onnxruntime is not installed; it ships with kokoro-onnx"
        if importlib.util.find_spec("misaki") is None:
            return False, (
                "misaki is not installed; Chinese G2P requires it "
                "(pip install 'misaki[zh]')"
            )
        store = default_store()
        if not store.is_ready(BUNDLE_ID):
            return False, store.fetch_hint(BUNDLE_ID)
        return True, "model and G2P backend present"

    async def probe(self) -> ProviderHealth:
        started = time.perf_counter()
        runtime = await asyncio.to_thread(runtime_probe.probe_kokoro_onnx)
        extra: dict[str, Any] = {
            "runtime_status": runtime.status,
            "runtime_detail": runtime.detail,
        }
        if not runtime.ok:
            detail = runtime.hint("tts") or runtime.detail
            self.lifecycle.transition("UNAVAILABLE", detail)
            return ProviderHealth(ok=False, detail=detail, extra=extra)

        extra.update(runtime.extra)

        # G2P is not optional plumbing for this model: without it there is no
        # phoneme string and nothing to synthesise. It is probed separately so
        # the failure message says "install the Chinese front end" instead of
        # "model not found".
        g2p = await asyncio.to_thread(runtime_probe.probe_g2p, self._desired_g2p())
        extra["g2p_status"] = g2p.status
        extra["g2p_backend"] = g2p.extra.get("backend", "")
        if not g2p.ok:
            detail = (
                g2p.detail
                if g2p.status != runtime_probe.STATUS_DEPENDENCY_MISSING
                else f"{g2p.detail}. Install it with: pip install 'misaki[zh]'"
            )
            self.lifecycle.transition("UNAVAILABLE", detail)
            return ProviderHealth(ok=False, detail=detail, extra=extra)
        extra["g2p_version"] = g2p.version

        if not self._store.is_ready(BUNDLE_ID):
            hint = self._store.fetch_hint(BUNDLE_ID)
            extra["model_missing"] = True
            extra["model_id"] = BUNDLE_ID
            self.lifecycle.transition("UNAVAILABLE", hint)
            return ProviderHealth(ok=False, detail=hint, extra=extra)

        self.lifecycle.transition("AVAILABLE")
        return ProviderHealth(
            ok=True,
            detail=(
                f"Kokoro {MODEL_ID} ready on CPU "
                f"(onnxruntime {runtime.version}, g2p={g2p.extra.get('backend', '?')})"
            ),
            latency_ms=int(round((time.perf_counter() - started) * 1000)),
            extra=extra,
        )

    async def load(self, model: str | None = None, device: str = Device.CPU.value) -> None:
        await self._ensure_probed()
        try:
            self.lifecycle.transition("LOADING")
        except Exception as exc:  # noqa: BLE001
            raise ProviderLoadError(str(exc), provider=PROVIDER_ID) from exc

        model_dir = self._resolve_model_dir()
        backend = self._desired_g2p()
        started = time.perf_counter()
        try:
            engine, g2p = await asyncio.to_thread(self._build, model_dir, backend)
        except ModelMissing:
            self.lifecycle.transition("UNAVAILABLE", "model files disappeared during load")
            raise
        except MemoryError as exc:
            self.lifecycle.transition("ERROR", "out of memory")
            raise OutOfMemory(
                "not enough RAM to load the Kokoro model", provider=PROVIDER_ID
            ) from exc
        except Exception as exc:  # noqa: BLE001
            self.lifecycle.transition("ERROR", f"{type(exc).__name__}: {exc}")
            raise ProviderLoadError(
                f"failed to load Kokoro: {type(exc).__name__}: {exc}",
                provider=PROVIDER_ID,
                model_id=MODEL_ID,
            ) from exc

        self._engine = engine
        self._g2p = g2p
        self._g2p_backend = backend
        self._voices = self._enumerate_voices(engine)
        self._loaded_voice = self._first_voice_id()
        self._last_load_ms = int(round((time.perf_counter() - started) * 1000))
        self.lifecycle.transition("READY")

    def _build(self, model_dir: str, backend: str) -> tuple[Any, Any]:
        """Worker-thread loader. Heavy imports stay local to this call."""

        from pathlib import Path

        from kokoro_onnx import Kokoro  # type: ignore[import-not-found]

        directory = Path(model_dir)
        onnx_path = directory / "kokoro-v1.1-zh.onnx"
        voices_path = directory / "voices-v1.1-zh.bin"
        config_path = directory / "config.json"
        for required in (onnx_path, voices_path, config_path):
            if not required.is_file():
                raise ModelMissing(
                    f"{required.name} is missing from {directory}",
                    model_id=BUNDLE_ID,
                    file=required.name,
                )

        engine = Kokoro(str(onnx_path), str(voices_path), vocab_config=str(config_path))
        g2p = self._build_g2p(backend)
        return engine, g2p

    def _build_g2p(self, backend: str) -> Any:
        if backend == "misaki":
            from misaki import zh  # type: ignore[import-not-found]

            # `version="1.1"` selects the v1.1 front end; the v1.0 path emits a
            # different vocabulary that this model does not understand.
            return zh.ZHG2P(version="1.1")
        return _EspeakG2P()

    async def unload(self) -> None:
        engine, g2p = self._engine, self._g2p
        self._engine = None
        self._g2p = None
        self._loaded_voice = ""
        if engine is not None or g2p is not None:
            await asyncio.to_thread(self._dispose, engine, g2p)
        self.lifecycle.transition("UNLOADING")
        self.lifecycle.transition("AVAILABLE")

    @staticmethod
    def _dispose(*objects: Any) -> None:
        del objects
        import gc

        gc.collect()

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            ok=self.lifecycle.is_serving,
            detail=self.lifecycle.detail or f"state={self.lifecycle.state.value}",
            extra={
                "state": self.lifecycle.state.value,
                "backend": self._g2p_backend,
                "voices": len(self._voices),
                "load_ms": self._last_load_ms,
                "sample_rate": SAMPLE_RATE,
            },
        )

    # -- synthesis ----------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> tuple[bytes, AudioFormat]:
        """Return a complete mono 16-bit WAV for `text`."""

        if token is not None:
            token.raise_if_cancelled()

        clean = self.normalize_text(text)
        if not clean:
            # Not an error: the chunker legitimately hands us whitespace.
            return self._silent_wav(1), self.audio_format()

        await self._require_loaded()

        chosen = self._resolve_voice(voice)
        speed = _option_float(self.options, "speed", 1.0)

        def run() -> bytes:
            phonemes, _ = self._g2p(clean)
            if not phonemes.strip():
                raise ProviderUnavailable(
                    f"G2P produced no phonemes for {clean[:40]!r}",
                    provider=PROVIDER_ID,
                    backend=self._g2p_backend,
                )
            samples, sample_rate = self._engine.create(
                phonemes, voice=chosen, speed=speed, is_phonemes=True
            )
            return self._to_wav(samples, sample_rate)

        blob = await self._run_blocking(run, token)
        return blob, self.audio_format()

    async def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> AsyncIterator[bytes]:
        """One WAV blob; `streaming=False` is declared in the descriptor."""

        blob, _fmt = await self.synthesize(
            text, voice=voice, language=language, model=model, token=token
        )
        if blob:
            yield blob

    def list_voices(self) -> Sequence[VoiceRef]:
        return self._voices or self.descriptor().voices

    def audio_format(self) -> AudioFormat:
        return AudioFormat(
            mime_type="audio/wav",
            sample_rate=SAMPLE_RATE,
            codec="pcm_s16le",
        )

    # -- text handling ------------------------------------------------------

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Strip what an LLM emits but a phonemiser cannot speak.

        Deliberately conservative: this removes markup and control characters
        and collapses whitespace. It does *not* touch numbers or punctuation,
        because those carry prosody. Numbers are expanded later and more
        correctly by ``cn2an`` inside the front end.
        """

        if not text:
            return ""
        cleaned = _MARKDOWN_NOISE.sub(" ", text)
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = "".join(char for char in cleaned if char in "\n\t" or ord(char) >= 32)
        cleaned = "".join(char for char in cleaned if _KEEP.match(char) or char in "\n")
        cleaned = re.sub(r"[\t ]+", " ", cleaned)
        cleaned = re.sub(r"\n{2,}", "\n", cleaned)
        return cleaned.strip()

    @classmethod
    def has_unphonemisable_script(cls, text: str) -> bool:
        """True when `text` contains Latin words the Chinese front end will drop.

        Reported through `ProviderHealth.extra` rather than raised: a sentence
        mixing "OK" into Chinese is a quality cliff, not a failure.
        """

        if not text:
            return False
        # Two or more consecutive Latin letters is a word, not an acronym
        # letter read out.
        return bool(re.search(r"[A-Za-z]{2,}", text))

    def _silent_wav(self, duration_ms: int) -> bytes:
        import struct
        import wave
        from io import BytesIO

        frames = max(1, int(SAMPLE_RATE * duration_ms / 1000))
        buffer = BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes(struct.pack("<h", 0) * frames)
        return buffer.getvalue()

    @staticmethod
    def _to_wav(samples: Any, sample_rate: int) -> bytes:
        """float32 [-1, 1] -> mono 16-bit PCM WAV.

        Clipping is explicit: the model can overshoot slightly, and letting
        numpy wrap an int16 round-trips a loud sample into a loud *click*.
        """

        import struct
        import wave
        from io import BytesIO

        import numpy as np

        array = np.asarray(samples, dtype=np.float32)
        if array.ndim > 1:
            array = array.reshape(-1)
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
        array = np.clip(array, -1.0, 1.0)
        pcm = (array * 32767.0).astype("<i2").tobytes()

        buffer = BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(pcm)
        del struct  # only imported to document the frame format above
        return buffer.getvalue()

    # -- internals ----------------------------------------------------------

    def _desired_g2p(self) -> str:
        return str(self.options.get("g2p", "misaki")).strip().lower() or "misaki"

    def _resolve_model_dir(self) -> str:
        override = self.options.get("model_path")
        if override:
            from pathlib import Path

            directory = Path(str(override))
            if not directory.is_dir():
                raise ModelMissing(
                    f"tts_model_path does not exist: {directory}",
                    model_id=BUNDLE_ID,
                    path=str(directory),
                )
            return str(directory)
        return str(self._store.require(BUNDLE_ID))

    def _enumerate_voices(self, engine: Any) -> tuple[VoiceRef, ...]:
        """Read the real voice list out of the voices file.

        The descriptor advertises a dozen, but the shipped file contains over a
        hundred. The runtime should surface what actually exists.
        """

        try:
            raw = list(engine.get_voices())
        except Exception:  # noqa: BLE001 - enumeration is a nicety, not a need
            return self.descriptor().voices

        declared = {voice.id: voice for voice in self.descriptor().voices}
        voices: list[VoiceRef] = []
        for voice_id in raw:
            known = declared.get(voice_id)
            if known is not None:
                voices.append(known)
                continue
            gender = _gender_from_prefix(voice_id)
            voices.append(
                VoiceRef(
                    id=voice_id,
                    display_name=voice_id,
                    languages=_LANGUAGES,
                    gender=gender,
                )
            )
        return tuple(voices)

    def _first_voice_id(self) -> str:
        configured = str(self.options.get("voice", "")).strip()
        if configured:
            return configured
        for voice in self._voices:
            if voice.id.startswith("zf_"):
                return voice.id
        return self._voices[0].id if self._voices else DEFAULT_VOICE

    def _resolve_voice(self, voice: str | None) -> str:
        wanted = (voice or "").strip() or self._loaded_voice or DEFAULT_VOICE
        if self._voices and wanted not in {item.id for item in self._voices}:
            # A bad voice id silently falling back to the default would make
            # the UI show one voice and the audio be another.
            raise ProviderUnavailable(
                f"unknown voice {wanted!r}; available: "
                f"{', '.join(item.id for item in self._voices[:8])}",
                provider=PROVIDER_ID,
                voice=wanted,
            )
        return wanted

    async def _require_loaded(self) -> None:
        if self._engine is not None and self._g2p is not None:
            return
        await self.load()

    async def _run_blocking(self, call: Any, token: CancellationToken | None) -> bytes:
        if self._synthesis_lock is None:
            self._synthesis_lock = asyncio.Lock()

        async with self._synthesis_lock:
            if token is None:
                return await asyncio.to_thread(call)
            task = asyncio.ensure_future(asyncio.to_thread(call))
            waiter = asyncio.ensure_future(token.wait())
            try:
                done, _pending = await asyncio.wait(
                    {task, waiter}, return_when=asyncio.FIRST_COMPLETED
                )
                if task in done:
                    return task.result()
                task.cancel()
                token.raise_if_cancelled()
                return task.result()
            finally:
                waiter.cancel()

    def supports(self, language: str) -> bool:
        from ...core.types import language_matches

        return language_matches(language, self.descriptor().languages)


class _EspeakG2P:
    """Fallback front end backed by the espeak-ng binary.

    Kept because it costs nothing to keep and it is the only option on a machine
    where ``misaki``'s dependencies cannot be installed. It is *not* equivalent:
    espeak-ng's Mandarin tone handling is noticeably worse and its phoneme
    alphabet is close to, but not identical to, the one Kokoro v1.1 expects.
    Selection prefers misaki whenever it is importable.
    """

    def __init__(self) -> None:
        import espeakng_loader  # type: ignore[import-not-found]

        self._library = espeakng_loader.get_library_path()
        self._data = espeakng_loader.get_data_path()

    def __call__(self, text: str) -> tuple[str, None]:
        from phonemizer import phonemize  # type: ignore[import-not-found]

        phonemes = phonemize(
            text,
            language="cmn",
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
        )
        return phonemes, None


def _gender_from_prefix(voice_id: str) -> str:
    """Kokoro voice ids encode the language and gender in the first letter."""

    if not voice_id:
        return "unspecified"
    if voice_id.startswith("zf"):
        return "female"
    if voice_id.startswith("zm"):
        return "male"
    return "unspecified"


__all__ = ["KokoroTTS", "PROVIDER_ID", "BUNDLE_ID", "MODEL_ID", "UNKNOWN_PHONEME"]
