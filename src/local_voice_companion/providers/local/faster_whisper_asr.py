"""faster-whisper ASR: a real, fully-local speech recogniser on CPU.

Why this engine
---------------
faster-whisper runs Whisper through CTranslate2, which is a C++ inference engine
with no Python-level tensor framework underneath. That matters more than raw
speed here: the alternative (torch + transformers) drags in a 2-3 GB CUDA
runtime and a PyTorch build, which on a machine with a nearly-full system drive
is the difference between "installable" and "not". CTranslate2 also supports
INT8 quantisation on CPU as a first-class compute type, and `base` INT8 runs at
roughly real time on four modern cores, which is enough for a conversation loop
where the LLM and TTS then overlap with the user's next utterance.

Honest limitations, recorded here rather than discovered later
--------------------------------------------------------------
* ``stream()`` is NOT incremental. Whisper is an attention encoder-decoder with
  no causal streaming path; producing partial hypotheses requires a chunked
  local-agreement scheme that this provider does not implement. The descriptor
  therefore declares ``streaming=False`` and ``stream()`` is deliberately left
  to raise, so a caller that expects partials fails loudly instead of silently
  receiving one final transcript.
* The language is auto-detected when not supplied, and detection costs a forward
  pass. Passing ``language="zh"`` skips it.
* Word timestamps are requested but only used for VAD-free trimming; nothing
  downstream depends on them.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from typing import Any, AsyncIterator, Mapping

from ...core.cancellation import CancellationToken
from ...core.errors import (
    DeviceUnavailable,
    ModelMissing,
    OutOfMemory,
    ProviderLoadError,
    ProviderUnavailable,
)
from ...core.types import (
    AudioChunk,
    Device,
    ProviderKind,
    QualitySource,
    normalize_language,
)
from ..base import (
    ASRProvider,
    ModelRef,
    ProviderDescriptor,
    ProviderHealth,
)
from . import runtime_probe
from .model_store import default_store

PROVIDER_ID = "faster_whisper_cpu"

#: CTranslate2 compute types that are valid on CPU, fastest first. INT8 needs
#: AVX2 for the fast integer kernels; the probe result is consulted at load time
#: rather than assumed, because a pre-AVX2 CPU silently falls back to a slow
#: emulated path instead of raising.
_CPU_COMPUTE_TYPES: tuple[str, ...] = ("int8", "int8_float32", "float32")

#: Whisper checkpoints worth offering. `base` is the default: it is the largest
#: model that still runs comfortably in real time on a 4-core CPU.
_SUPPORTED_MODELS: tuple[tuple[str, str, int], ...] = (
    ("tiny", "Whisper tiny", 75),
    ("base", "Whisper base", 142),
    ("small", "Whisper small", 466),
    ("medium", "Whisper medium", 1530),
    ("large-v3", "Whisper large-v3", 3090),
    ("large-v3-turbo", "Whisper large-v3 turbo", 1620),
)

DEFAULT_MODEL = "base"

#: Languages the base model handles acceptably. Whisper itself claims ~99
#: languages; this list is the subset we are willing to *advertise*, because
#: advertising a language the model transcribes badly is worse than rejecting
#: it in the selector with a clear reason.
_LANGUAGES: tuple[str, ...] = (
    "zh",
    "en",
    "ja",
    "ko",
    "yue",
    "fr",
    "de",
    "es",
    "ru",
)


def _option_int(options: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(options.get(key, default))
    except (TypeError, ValueError):
        return default


def _option_float(options: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(options.get(key, default))
    except (TypeError, ValueError):
        return default


class FasterWhisperASR(ASRProvider):
    """Local Whisper transcription via CTranslate2."""

    kind = ProviderKind.ASR

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        super().__init__(options)
        self._engine: Any = None
        self._loaded_model: str = ""
        self._loaded_device: str = ""
        self._compute_type: str = ""
        self._last_load_ms: int = 0
        self._inference_lock: asyncio.Lock | None = None
        self._store = default_store()

    # -- description --------------------------------------------------------

    @staticmethod
    def descriptor() -> ProviderDescriptor:
        models = tuple(
            ModelRef(
                id=name,
                display_name=display,
                languages=_LANGUAGES,
                quantization="int8",
                disk_mb=disk,
            )
            for name, display, disk in _SUPPORTED_MODELS
        )
        return ProviderDescriptor(
            id=PROVIDER_ID,
            kind=ProviderKind.ASR,
            display_name="faster-whisper (CTranslate2, CPU INT8)",
            languages=_LANGUAGES,
            devices=(Device.CPU,),
            # Whisper has no incremental decoder in this integration. Declaring
            # True here would make the selector hand it to a caller expecting
            # partial hypotheses, which would then fail at runtime.
            streaming=False,
            estimated_ram_mb=350,
            estimated_vram_mb=0,
            estimated_disk_mb=150,
            requires_network=False,
            is_local=True,
            quality_tier=3,
            latency_tier=3,
            quality_source=QualitySource.CURATED_METADATA,
            # Whisper base INT8 is a genuinely capable recogniser, but it is not
            # the same class as a large-v3 or a dedicated commercial engine.
            quality_score=0.72,
            supports_cancellation=True,
            # The *installed engine's* version, not a constant. A benchmark
            # measures one specific build, so the selection cache keys on this;
            # a hardcoded "1.0.0" would silently reuse stale numbers across a
            # faster-whisper upgrade. Resolved from distribution metadata, which
            # does not import the package and costs ~1-6 ms.
            version=runtime_probe.package_version("faster-whisper") or "none",
            tags=(
                "local",
                "cpu",
                "ctranslate2",
                "faster-whisper",
                "int8",
                "no-download",
            ),
            models=models,
            voices=(),
        )

    # -- lifecycle ----------------------------------------------------------

    async def probe(self) -> ProviderHealth:
        """Report readiness with a specific, actionable reason when not ready."""

        started = time.perf_counter()
        runtime = await asyncio.to_thread(runtime_probe.probe_faster_whisper)
        extra: dict[str, Any] = {
            "runtime_status": runtime.status,
            "runtime_detail": runtime.detail,
        }
        if not runtime.ok:
            detail = runtime.hint("asr") or runtime.detail
            self.lifecycle.transition("UNAVAILABLE", detail)
            return ProviderHealth(ok=False, detail=detail, extra=extra)

        extra.update(runtime.extra)
        model_id = self._model_id(None)

        # A caller-supplied path bypasses the catalogue entirely, so `probe`
        # must not reject a perfectly good local checkout merely because it has
        # no registered bundle.
        override = self.options.get("model_path")
        if override:
            from pathlib import Path

            directory = Path(str(override))
            if not directory.is_dir():
                detail = f"asr_model_path does not exist: {directory}"
                self.lifecycle.transition("UNAVAILABLE", detail)
                return ProviderHealth(ok=False, detail=detail, extra=extra)
            self.lifecycle.transition("AVAILABLE")
            return ProviderHealth(
                ok=True,
                detail=f"faster-whisper {runtime.version} reading {directory}",
                latency_ms=int(round((time.perf_counter() - started) * 1000)),
                extra={**extra, "model_path": str(directory)},
            )

        cached = await asyncio.to_thread(self._find_in_hf_cache, model_id)
        if cached is not None:
            self.lifecycle.transition("AVAILABLE")
            return ProviderHealth(
                ok=True,
                detail=(
                    f"faster-whisper {runtime.version} with model {model_id} "
                    f"from the local HuggingFace cache"
                ),
                latency_ms=int(round((time.perf_counter() - started) * 1000)),
                extra={**extra, "cached_snapshot": cached},
            )

        bundle_id = self._bundle_id_for(model_id)
        if bundle_id is None:
            detail = (
                f"Whisper model {model_id!r} has no registered download bundle "
                f"(known: {', '.join(self._store.ids())}). Set asr_model_path to "
                f"use weights you already have."
            )
            self.lifecycle.transition("UNAVAILABLE", detail)
            extra["model_missing"] = True
            extra["model_id"] = model_id
            return ProviderHealth(ok=False, detail=detail, extra=extra)

        if not self._store.is_ready(bundle_id):
            hint = self._store.fetch_hint(bundle_id)
            self.lifecycle.transition("UNAVAILABLE", hint)
            extra["model_missing"] = True
            extra["model_id"] = bundle_id
            return ProviderHealth(ok=False, detail=hint, extra=extra)

        self.lifecycle.transition("AVAILABLE")
        return ProviderHealth(
            ok=True,
            detail=f"faster-whisper {runtime.version} with model {model_id} on CPU",
            latency_ms=int(round((time.perf_counter() - started) * 1000)),
            extra=extra,
        )

        if not store.is_ready(bundle_id):
            hint = store.fetch_hint(bundle_id)
            self.lifecycle.transition("UNAVAILABLE", hint)
            extra["model_missing"] = True
            extra["model_id"] = bundle_id
            return ProviderHealth(ok=False, detail=hint, extra=extra)

        self.lifecycle.transition("AVAILABLE")
        latency = int(round((time.perf_counter() - started) * 1000))
        return ProviderHealth(
            ok=True,
            detail=f"faster-whisper {runtime.version} with model {model_id} on CPU",
            latency_ms=latency,
            extra=extra,
        )

    async def load(self, model: str | None = None, device: str = Device.CPU.value) -> None:
        """Instantiate the Whisper model.

        Loading is a blocking, multi-second operation (the ONNX/CTranslate2
        weight load plus graph setup), so it runs in a worker thread. It is also
        serialised behind a lock: two concurrent loads of the same model would
        double peak memory for no benefit.
        """

        if str(device) not in {Device.CPU.value, "cpu", ""}:
            raise DeviceUnavailable(
                f"{PROVIDER_ID} only runs on CPU; asked for {device!r}",
                provider=PROVIDER_ID,
                device=str(device),
            )

        await self._ensure_probed()
        try:
            self.lifecycle.transition("LOADING")
        except Exception as exc:  # noqa: BLE001
            raise ProviderLoadError(str(exc), provider=PROVIDER_ID) from exc

        model_id = self._model_id(model)

        if self._engine is not None and self._loaded_model == model_id:
            self.lifecycle.transition("READY")
            return

        store, model_dir = self._resolve_model_dir(model_id)
        compute_type = self._pick_compute_type()

        started = time.perf_counter()
        try:
            engine = await asyncio.to_thread(
                self._build_engine, model_dir, compute_type
            )
        except ModelMissing:
            self.lifecycle.transition("UNAVAILABLE", "model files disappeared during load")
            raise
        except MemoryError as exc:
            self.lifecycle.transition("ERROR", "out of memory loading model")
            raise OutOfMemory(
                f"not enough RAM to load Whisper {model_id}",
                provider=PROVIDER_ID,
                model_id=model_id,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - normalised to a typed error
            self.lifecycle.transition("ERROR", f"{type(exc).__name__}: {exc}")
            raise ProviderLoadError(
                f"failed to load Whisper {model_id}: {type(exc).__name__}: {exc}",
                provider=PROVIDER_ID,
                model_id=model_id,
            ) from exc

        self._engine = engine
        self._loaded_model = model_id
        self._loaded_device = Device.CPU.value
        self._compute_type = compute_type
        self._last_load_ms = int(round((time.perf_counter() - started) * 1000))
        del store  # only bound to make the catalogue lookup explicit
        self.lifecycle.transition("READY")


    def _build_engine(self, model_dir: str, compute_type: str) -> Any:
        """Worker-thread body of `load`. Imports heavy modules on first use."""

        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        return WhisperModel(
            model_dir,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=self._cpu_threads(),
            num_workers=1,
            download_root=None,  # never let faster-whisper fetch anything
            local_files_only=True,
        )

    async def unload(self) -> None:
        engine = self._engine
        self._engine = None
        self._loaded_model = ""
        self._compute_type = ""
        if engine is not None:
            # CTranslate2 models release their weights on GC; doing it in a
            # thread keeps the event loop responsive and makes the free
            # observable before we return.
            await asyncio.to_thread(self._dispose, engine)
        self.lifecycle.transition("UNLOADING")
        self.lifecycle.transition("AVAILABLE")

    @staticmethod
    def _dispose(engine: Any) -> None:
        del engine
        import gc

        gc.collect()

    async def health(self) -> ProviderHealth:
        ok = self.lifecycle.is_serving
        return ProviderHealth(
            ok=ok,
            detail=self.lifecycle.detail or f"state={self.lifecycle.state.value}",
            extra={
                "state": self.lifecycle.state.value,
                "model": self._loaded_model,
                "device": self._loaded_device,
                "compute_type": self._compute_type,
                "load_ms": self._last_load_ms,
            },
        )

    # -- inference ----------------------------------------------------------

    async def transcribe(
        self,
        audio: AudioChunk,
        *,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> str:
        """Transcribe one utterance of 16 kHz mono PCM."""

        if token is not None:
            token.raise_if_cancelled()

        if audio.duration_ms <= 0:
            return ""

        await self._require_loaded(model)

        pcm = self._prepare_pcm(audio)
        wanted = normalize_language(language) or self._default_language()

        def run() -> str:
            segments, _info = self._engine.transcribe(
                pcm,
                language=wanted or None,
                beam_size=_option_int(self.options, "beam_size", 1),
                # A conversation turn is short and the VAD already trimmed it;
                # the padding whisper uses by default costs latency for nothing.
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=False,
                temperature=0.0,
            )
            return "".join(segment.text for segment in segments).strip()

        return await self._run_blocking(run, token)

    async def stream(
        self,
        frames: AsyncIterator[AudioChunk],
        *,
        language: str = "",
        model: str | None = None,
        token: CancellationToken | None = None,
    ) -> AsyncIterator[str]:
        """Not supported -- see the module docstring.

        A caller that wants partial hypotheses must not be handed a single
        final transcript and told it is a stream. The base class default raises
        `RuntimeError`; we keep that behaviour and say why.
        """

        if False:  # pragma: no cover - keeps this an async generator
            yield ""
        raise ProviderUnavailable(
            f"{PROVIDER_ID} does not implement incremental transcription "
            "(descriptor declares streaming=False); use transcribe()",
            provider=PROVIDER_ID,
        )

    # -- internals ----------------------------------------------------------

    def _model_id(self, model: str | None) -> str:
        raw = model or self.options.get("asr_model") or DEFAULT_MODEL
        return str(raw)

    def _default_language(self) -> str:
        return str(self.options.get("language", "") or "")

    def _cpu_threads(self) -> int:
        import os

        configured = _option_int(self.options, "cpu_threads", 0)
        if configured > 0:
            return configured
        return max(1, min(8, (os.cpu_count() or 4)))

    def _pick_compute_type(self) -> str:
        """Choose the fastest supported CPU compute type.

        The probe already told us which types CTranslate2 accepts. Preferring a
        type it did not report would raise deep inside the native layer.
        """

        configured = str(self.options.get("compute_type", "")).strip()
        if configured:
            return configured
        runtime = runtime_probe.probe_ctranslate2()
        supported = set(runtime.extra.get("cpu_compute_types") or ())
        if not supported:
            return "int8"
        for candidate in _CPU_COMPUTE_TYPES:
            if candidate in supported:
                return candidate
        # Nothing we know is available; let CTranslate2 pick.
        return "default"

    @classmethod
    def availability(cls, model_id: str | None = None) -> tuple[bool, str]:
        """Whether this provider could load `model_id` right now, and why not.

        Public because the CLI (`lvc doctor`) and the hardware tests must reach
        the same verdict as `probe()`; duplicating the rule in three places is
        how a provider ends up reporting "ready" in one view and refusing to
        load in another.
        """

        wanted = str(model_id or DEFAULT_MODEL)
        if importlib.util.find_spec("faster_whisper") is None:
            return False, (
                "faster-whisper is not installed; "
                "run: pip install 'local-voice-companion[asr]'"
            )
        if importlib.util.find_spec("ctranslate2") is None:
            return False, "ctranslate2 is not installed; it ships with faster-whisper"

        store = default_store()
        if cls._find_in_hf_cache(wanted) is not None:
            return True, f"model {wanted} found in the local HuggingFace cache"

        bundle_id = cls._bundle_id_for(wanted)
        if bundle_id is None:
            return False, (
                f"Whisper model {wanted!r} has no registered download bundle "
                f"(known: {', '.join(store.ids())}); set asr_model_path or pick a known model"
            )
        if not store.is_ready(bundle_id):
            return False, store.fetch_hint(bundle_id)
        return True, f"model {bundle_id} present"

    def _resolve_model_dir(self, model_id: str) -> tuple[Any, str]:
        """Map a model id to an on-disk directory, or raise `ModelMissing`."""

        store = self._store

        # A locally-registered bundle is the normal path. A raw filesystem path
        # is accepted so an operator can point at weights they already have
        # without teaching the store about them.
        override = self.options.get("model_path")
        if override:
            from pathlib import Path

            directory = Path(str(override))
            if not directory.is_dir():
                raise ModelMissing(
                    f"asr_model_path does not exist: {directory}",
                    model_id=model_id,
                    path=str(directory),
                )
            return store, str(directory)

        # Weights fetched by faster-whisper itself (or by `huggingface-cli`) sit
        # in a `models--<org>--<name>/snapshots/<rev>/` tree, which is a
        # different shape from our flat bundle directory. Reusing them avoids a
        # second 145 MB download on a machine that already paid for one.
        cached = self._find_in_hf_cache(model_id)
        if cached is not None:
            return store, cached

        bundle_id = self._bundle_id_for(model_id)
        if bundle_id is None:
            raise ModelMissing(
                f"Whisper model {model_id!r} has no registered download bundle. "
                f"Known bundles: {', '.join(store.ids())}. Either pick one of "
                f"those, set asr_model_path, or pre-download the CTranslate2 "
                f"conversion yourself.",
                model_id=model_id,
            )
        return store, str(store.require(bundle_id))

    @staticmethod
    def _hf_cache_roots() -> list[Any]:
        """Candidate HuggingFace hub directories, most specific first."""

        import os
        from pathlib import Path

        from .model_store import MODEL_ROOT

        roots: list[Path] = []
        explicit = os.getenv("LVC_MODELS_DIR", "").strip()
        if explicit:
            roots.append(Path(explicit).parent / "hf" / "hub")
        roots.append(MODEL_ROOT.parent / "hf" / "hub")
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            value = os.getenv(name, "").strip()
            if value:
                roots.append(Path(value))
        hf_home = os.getenv("HF_HOME", "").strip()
        if hf_home:
            roots.append(Path(hf_home) / "hub")
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        return [root for root in roots if root.is_dir()]

    @classmethod
    def _find_in_hf_cache(cls, model_id: str) -> str | None:
        """Locate an existing snapshot directory for `model_id`."""

        import os
        from pathlib import Path

        repo = cls._hf_repo_for(model_id)
        if repo is None:
            return None
        folder = "models--" + repo.replace("/", "--")
        for root in cls._hf_cache_roots():
            snapshots = root / folder / "snapshots"
            if not snapshots.is_dir():
                continue
            for revision in sorted(snapshots.iterdir(), reverse=True):
                if not revision.is_dir():
                    continue
                # A snapshot that still has dangling symlinks is a partial
                # download; faster-whisper would fail on it deep inside the
                # native loader, so verify the file resolves before trusting it.
                blob = revision / "model.bin"
                if blob.is_file() and blob.stat().st_size > 0:
                    del os  # imported only to keep the env-var reads above obvious
                    return str(revision)
        return None

    @staticmethod
    def _hf_repo_for(model_id: str) -> str | None:
        """Map a short name to the Systran CTranslate2 repository."""

        name = model_id.strip().lower()
        if name.startswith("systran/"):
            return name
        known = {
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
            "large-v3-turbo",
            "turbo",
            "distil-large-v3",
        }
        if name in known:
            return f"Systran/faster-whisper-{name}"
        if name.startswith("faster-whisper-"):
            return f"Systran/{name}"
        return None

    @staticmethod
    def _bundle_id_for(model_id: str) -> str | None:
        """Only `base` ships a pinned bundle; see docs/PROVIDER_LICENSES.md."""

        mapping = {
            "base": "faster-whisper-base",
            # Aliases people reasonably type.
            "whisper-base": "faster-whisper-base",
        }
        return mapping.get(model_id.strip().lower())

    def _prepare_pcm(self, audio: AudioChunk) -> Any:
        """Convert to the float32 mono 16 kHz the model expects.

        Whisper is trained on 16 kHz mono float32. Handing it the browser's
        48 kHz 16-bit stereo works -- ffmpeg resamples internally -- but it
        costs an extra copy, and any sample format we get wrong would be
        silently interpreted as a different bit depth.
        """

        import numpy as np

        if audio.sample_width not in (1, 2, 3, 4):
            raise ProviderUnavailable(
                f"unsupported sample width {audio.sample_width}",
                provider=PROVIDER_ID,
            )
        dtype = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.int32}[audio.sample_width]
        samples = np.frombuffer(audio.pcm, dtype=dtype).astype(np.float32)

        if audio.sample_width == 1:
            samples = (samples - 128.0) / 128.0
        elif audio.sample_width == 2:
            samples /= 32768.0
        elif audio.sample_width == 3:
            # 24-bit little-endian is not a numpy dtype; the raw buffer above
            # would have mis-parsed it, so decode explicitly.
            raw = np.frombuffer(audio.pcm, dtype=np.uint8).reshape(-1, 3)
            packed = (
                raw[:, 0].astype(np.int32)
                | (raw[:, 1].astype(np.int32) << 8)
                | (raw[:, 2].astype(np.int8).astype(np.int32) << 16)
            )
            samples = packed.astype(np.float32) / 8388608.0
        else:
            samples /= 2147483648.0

        if audio.channels > 1:
            samples = samples.reshape(-1, audio.channels).mean(axis=1)

        if audio.sample_rate != 16000:
            samples = self._resample(samples, audio.sample_rate, 16000)

        return np.ascontiguousarray(samples, dtype=np.float32)

    @staticmethod
    def _resample(samples: Any, source_rate: int, target_rate: int) -> Any:
        """Linear resampling.

        Deliberately simple. Whisper's front end applies a mel filterbank that
        is far more lossy than the difference between linear interpolation and
        a polyphase filter, and `av` (already a dependency) is not worth the
        extra copy for a one-octave conversion. If quality ever matters here,
        the right fix is to capture at 16 kHz upstream.
        """

        import numpy as np

        if source_rate == target_rate or source_rate <= 0:
            return samples
        duration = samples.shape[0] / float(source_rate)
        count = max(1, int(round(duration * target_rate)))
        source_positions = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, duration, num=count, endpoint=False)
        return np.interp(target_positions, source_positions, samples).astype(np.float32)

    async def _require_loaded(self, model: str | None) -> None:
        wanted = self._model_id(model)
        if self._engine is not None and self._loaded_model == wanted:
            return
        await self.load(model)

    async def _run_blocking(self, call: Any, token: CancellationToken | None) -> str:
        """Offload a blocking inference call without losing cancellation.

        `asyncio.to_thread` cannot be cancelled: the worker thread runs to
        completion regardless. We therefore race the inference against the
        cancellation token so the *caller* returns promptly, and let the thread
        finish in the background. The alternative -- blocking the event loop
        for the whole inference -- would freeze every other session.
        """

        if self._inference_lock is None:
            self._inference_lock = asyncio.Lock()

        async with self._inference_lock:
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
                # Cancellation won the race. Abandon the thread rather than
                # blocking; its result is discarded when it completes.
                task.cancel()
                token.raise_if_cancelled()
                return task.result()
            finally:
                waiter.cancel()

    # -- capability helpers --------------------------------------------------

    def supports(self, language: str) -> bool:
        from ...core.types import language_matches

        return language_matches(language, self.descriptor().languages)


__all__ = ["FasterWhisperASR", "PROVIDER_ID", "DEFAULT_MODEL"]
