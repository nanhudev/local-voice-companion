"""Model artefact registry: where weights live, and how they get there.

Two rules drive this module.

**Nothing downloads silently.** A provider whose weights are absent raises
:class:`ModelMissing` naming the exact command that would fetch them. There is
no "convenience" fetch-on-first-use path, because a 300 MB download triggered by
an unrelated HTTP request is indistinguishable from a hang.

**Downloads are atomic.** Weights are fetched into ``<target>.part`` and
renamed only after the size check passes. A half-written ONNX file that looks
present is worse than no file at all: every future run then fails deep inside
onnxruntime with an opaque protobuf error instead of a clean "not downloaded".
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

from ...config.paths import DEFAULT_LAYOUT
from ...core.errors import ModelMissing, ProviderUnavailable

ProgressFn = Callable[[str, int, int], None]


def _free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


@dataclass(frozen=True)
class ModelArtifact:
    """One file that must exist before a provider can load.

    ``min_bytes`` is a sanity floor, not a checksum substitute. HuggingFace
    mirrors redirect and occasionally serve LFS pointer stubs (a few hundred
    bytes of text); a size floor catches that class of failure cheaply. ``sha256``
    is optional because pinning it requires a first successful fetch to learn it,
    and a wrong hash is worse than an honest absence.
    """

    filename: str
    url: str
    min_bytes: int = 1024
    sha256: str = ""
    optional: bool = False

    def size_hint_mb(self) -> int:
        return int(round(self.min_bytes / (1024 * 1024)))


@dataclass
class ModelBundle:
    """A named set of artefacts living in one directory."""

    model_id: str
    directory_name: str
    artifacts: tuple[ModelArtifact, ...]
    homepage: str = ""
    license_id: str = ""
    notes: str = ""

    def total_min_bytes(self) -> int:
        return sum(item.min_bytes for item in self.artifacts if not item.optional)

    def total_estimated_mb(self) -> int:
        # Byte floors understate the real size; providers declare the honest
        # figure in their descriptor and this only has to be a floor for the
        # disk-space precheck.
        return int(round(self.total_min_bytes() * 1.15 / (1024 * 1024)))


class ModelStore:
    """Resolves, verifies and (on explicit request) fetches model artefacts."""

    def __init__(self, root: Path | None = None, bundles: Iterable[ModelBundle] = ()) -> None:
        self.root = Path(root) if root else MODEL_ROOT
        self._bundles: dict[str, ModelBundle] = {bundle.model_id: bundle for bundle in bundles}

    # -- catalogue ----------------------------------------------------------

    def register(self, bundle: ModelBundle) -> None:
        self._bundles[bundle.model_id] = bundle

    def bundle(self, model_id: str) -> ModelBundle:
        bundle = self._bundles.get(model_id)
        if bundle is None:
            raise ProviderUnavailable(
                f"unknown model id {model_id!r}", model_id=model_id
            )
        return bundle

    def ids(self) -> list[str]:
        return sorted(self._bundles)

    def bundles(self) -> list[ModelBundle]:
        return [self._bundles[key] for key in sorted(self._bundles)]

    # -- paths --------------------------------------------------------------

    def directory(self, model_id: str) -> Path:
        return self.root / self.bundle(model_id).directory_name

    def path(self, model_id: str, filename: str) -> Path:
        return self.directory(model_id) / filename

    # -- verification -------------------------------------------------------

    def missing(self, model_id: str) -> list[ModelArtifact]:
        """Required artefacts that are absent or implausibly small."""

        bundle = self.bundle(model_id)
        directory = self.directory(model_id)
        absent: list[ModelArtifact] = []
        for artifact in bundle.artifacts:
            if artifact.optional:
                continue
            target = directory / artifact.filename
            try:
                size = target.stat().st_size
            except OSError:
                absent.append(artifact)
                continue
            if size < artifact.min_bytes:
                absent.append(artifact)
        return absent

    def is_ready(self, model_id: str) -> bool:
        return not self.missing(model_id)

    def status(self, model_id: str) -> dict[str, object]:
        bundle = self.bundle(model_id)
        directory = self.directory(model_id)
        present = [
            artifact
            for artifact in bundle.artifacts
            if (directory / artifact.filename).is_file()
        ]
        absent = self.missing(model_id)
        return {
            "model_id": model_id,
            "directory": str(directory),
            "ready": not absent,
            "present": [item.filename for item in present],
            "missing": [item.filename for item in absent],
            "license": bundle.license_id,
            "homepage": bundle.homepage,
            "fetch_command": self.fetch_hint(model_id) if absent else "",
        }

    # -- operator guidance --------------------------------------------------

    def fetch_hint(self, model_id: str) -> str:
        bundle = self.bundle(model_id)
        absent = self.missing(model_id)
        listing = " ".join(item.filename for item in absent) or "<all>"
        return (
            f"lvc models fetch {model_id}   "
            f"(downloads {listing} into {self.directory(model_id)}, "
            f"~{bundle.total_estimated_mb()} MB)"
        )

    def require(self, model_id: str) -> Path:
        """Return the model directory or raise with the exact fetch command."""

        absent = self.missing(model_id)
        if absent:
            names = ", ".join(item.filename for item in absent)
            raise ModelMissing(
                f"model {model_id!r} is not installed (missing: {names}). "
                f"Run: {self.fetch_hint(model_id)}",
                model_id=model_id,
                missing=[item.filename for item in absent],
            )
        return self.directory(model_id)

    # -- fetching -----------------------------------------------------------

    def fetch(
        self,
        model_id: str,
        *,
        dry_run: bool = False,
        progress: ProgressFn | None = None,
        force: bool = False,
        timeout: float = 120.0,
    ) -> list[dict[str, object]]:
        """Download missing artefacts atomically.

        Returns one record per artefact. Nothing is written outside the model
        directory, nothing is renamed until its size floor is satisfied, and a
        failure leaves the ``.part`` file in place for inspection rather than
        promoting a truncated file to a valid name.
        """

        import requests  # local import: the base install does not need requests

        bundle = self.bundle(model_id)
        directory = self.directory(model_id)
        wanted = bundle.artifacts if force else self.missing(model_id)
        records: list[dict[str, object]] = []

        if not wanted:
            return [
                {"file": item.filename, "status": "already-present", "enough_space": True}
                for item in bundle.artifacts
                if not item.optional
            ]

        needed = bundle.total_min_bytes() * 2 if force else sum(
            item.min_bytes * 2 for item in wanted
        )
        available = _free_bytes(directory.parent if directory.parent.exists() else self.root)

        if dry_run:
            # A dry run *reports* capacity; it must never fail on it. Raising
            # here made the one command a user runs before committing several
            # hundred megabytes unusable on exactly the machines that need the
            # warning -- and it did so before printing the plan that would have
            # explained the problem.
            enough = not available or available >= needed
            return [
                {
                    "file": item.filename,
                    "status": "would-download",
                    "url": item.url,
                    "min_bytes": item.min_bytes,
                    "free_bytes": available,
                    "needed_bytes": needed,
                    "enough_space": enough,
                }
                for item in wanted
            ]

        if available and available < needed:
            raise ProviderUnavailable(
                f"not enough free space for {model_id!r}: need ~{needed // 2**20} MB, "
                f"have {available // 2**20} MB under {self.root}",
                model_id=model_id,
            )

        directory.mkdir(parents=True, exist_ok=True)
        for artifact in wanted:
            target = directory / artifact.filename
            part = target.with_suffix(target.suffix + ".part")
            record: dict[str, object] = {"file": artifact.filename, "status": "pending"}
            try:
                with requests.get(artifact.url, stream=True, timeout=timeout) as response:
                    response.raise_for_status()
                    declared = int(response.headers.get("Content-Length") or 0)
                    written = 0
                    digest = hashlib.sha256()
                    with part.open("wb") as handle:
                        for block in response.iter_content(chunk_size=1 << 20):
                            if not block:
                                continue
                            handle.write(block)
                            digest.update(block)
                            written += len(block)
                            if progress is not None:
                                progress(artifact.filename, written, declared or written)
                if written < artifact.min_bytes:
                    raise ProviderUnavailable(
                        f"{artifact.filename} is only {written} bytes, expected at least "
                        f"{artifact.min_bytes}. The mirror may be serving an LFS pointer.",
                        model_id=model_id,
                        file=artifact.filename,
                    )
                if artifact.sha256 and digest.hexdigest() != artifact.sha256:
                    raise ProviderUnavailable(
                        f"{artifact.filename} sha256 mismatch", model_id=model_id
                    )
                if target.exists():
                    target.unlink()
                part.rename(target)
            except Exception as exc:  # noqa: BLE001 - normalised below
                # Keep the .part file: it is evidence, and the next run resumes
                # from a clean slate anyway because we never trust it.
                if isinstance(exc, ProviderUnavailable):
                    raise
                raise ProviderUnavailable(
                    f"failed to download {artifact.filename} for {model_id!r}: {exc}",
                    model_id=model_id,
                    url=artifact.url,
                ) from exc
            record["status"] = "downloaded"
            record["bytes"] = target.stat().st_size
            records.append(record)

        leftover = self.missing(model_id)
        if leftover:
            raise ProviderUnavailable(
                f"{model_id!r} still incomplete after fetch: "
                f"{', '.join(item.filename for item in leftover)}",
                model_id=model_id,
            )
        return records


# ---------------------------------------------------------------------------
# The default catalogue
# ---------------------------------------------------------------------------

#: Weights live on a secondary volume, never in the source tree and never on C:.
MODEL_ROOT: Path = DEFAULT_LAYOUT.models_dir

#: Artefact inventories. Sizes are byte floors used for the disk precheck and
#: the LFS-pointer guard; the honest download figure is in `notes`.
BUNDLES: tuple[ModelBundle, ...] = (
    ModelBundle(
        model_id="faster-whisper-base",
        directory_name="faster-whisper-base",
        homepage="https://huggingface.co/Systran/faster-whisper-base",
        license_id="MIT",
        notes=(
            "CTranslate2 conversion of OpenAI Whisper base. ~142 MB on disk. "
            "Systran's build is MIT; the original Whisper weights are MIT too."
        ),
        artifacts=(
            ModelArtifact(
                filename="model.bin",
                url="https://huggingface.co/Systran/faster-whisper-base/resolve/main/model.bin",
                # Observed 145,217,532 bytes. A floor this close to the real size
                # also catches a truncated transfer, not just an LFS pointer.
                min_bytes=140 * 1024 * 1024,
            ),
            ModelArtifact(
                filename="config.json",
                url="https://huggingface.co/Systran/faster-whisper-base/resolve/main/config.json",
                min_bytes=1024,
            ),
            ModelArtifact(
                filename="tokenizer.json",
                url="https://huggingface.co/Systran/faster-whisper-base/resolve/main/tokenizer.json",
                # Observed 2,203,239 bytes.
                min_bytes=2 * 1024 * 1024,
            ),
            ModelArtifact(
                filename="vocabulary.txt",
                url="https://huggingface.co/Systran/faster-whisper-base/resolve/main/vocabulary.txt",
                # Observed 459,861 bytes.
                min_bytes=400 * 1024,
            ),
        ),
    ),
    ModelBundle(
        model_id="kokoro-v1.1-zh",
        directory_name="kokoro-v1.1-zh",
        homepage="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh",
        license_id="Apache-2.0",
        notes=(
            "Kokoro-82M v1.1 Chinese. ~380 MB total. The ONNX export ships from "
            "the kokoro-onnx release page; config.json comes from the upstream "
            "repo and is required for the misaki G2P vocabulary."
        ),
        artifacts=(
            ModelArtifact(
                filename="kokoro-v1.1-zh.onnx",
                url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.1-zh.onnx",
                min_bytes=300 * 1024 * 1024,
            ),
            ModelArtifact(
                filename="voices-v1.1-zh.bin",
                url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.1-zh.bin",
                min_bytes=45 * 1024 * 1024,
            ),
            ModelArtifact(
                filename="config.json",
                url="https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json",
                min_bytes=1024,
            ),
        ),
    ),
)

_STORE: ModelStore | None = None


def default_store(root: Path | None = None) -> ModelStore:
    """Process-wide store. Tests build their own with a tmp root."""

    global _STORE
    if root is not None:
        return ModelStore(root=root, bundles=BUNDLES)
    if _STORE is None:
        _STORE = ModelStore(root=MODEL_ROOT, bundles=BUNDLES)
    return _STORE


def env_overrides() -> Mapping[str, str]:
    """Environment variables that redirect model storage.

    Documented so the behaviour is inspectable rather than folklore: setting
    ``LVC_MODELS_DIR`` moves every artefact, which matters on machines whose
    system drive has less free space than the model set.
    """

    return {
        "LVC_MODELS_DIR": "overrides the model directory entirely",
        "LVC_DATA_ROOT": "moves the data root that models/ lives under",
        "HF_HOME": "only affects HuggingFace's own cache, not LVC's model dir",
    }


def hf_cache_dir() -> Path:
    """Where HuggingFace-owned caches go. Kept off the system drive."""

    explicit = os.getenv("LVC_MODELS_DIR", "").strip()
    base = Path(explicit) if explicit else MODEL_ROOT
    return base.parent / "hf"


__all__ = [
    "BUNDLES",
    "MODEL_ROOT",
    "ModelArtifact",
    "ModelBundle",
    "ModelStore",
    "default_store",
    "env_overrides",
    "hf_cache_dir",
]
