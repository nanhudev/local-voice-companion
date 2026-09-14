"""Filesystem layout.

Rule enforced by this module: the source tree never accumulates heavy artefacts
(models, caches, bot manifests, benchmark database). Everything writable lives
under a data root that defaults to a large secondary volume.

Resolution order:

    1. LVC_DATA_ROOT env var
    2. D:\\AI_Workspace\\local-voice-companion   (when D: exists)
    3. <project>/.local-voice-companion

Nothing here may raise: a missing or read-only location degrades to a writable
subdirectory under the project and records the reason in `PathResolution.note`.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

APP_NAME = "local-voice-companion"

#: Where the source lives. ``paths.py`` may only READ from here.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

#: Candidate roots for writable data, most preferred first.
_PREFERRED_ROOTS: tuple[str, ...] = ("D:\\AI_Workspace", "E:\\AI_Workspace")

_SUBDIRS: tuple[str, ...] = ("bots", "models", "cache", "logs", "benchmarks", "tmp")


@dataclass(frozen=True)
class Layout:
    root: Path
    note: str = ""
    writable: bool = True

    @property
    def bots_dir(self) -> Path:
        return self.root / "bots"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def benchmark_db(self) -> Path:
        return self.root / "benchmarks" / "benchmarks.json"

    @property
    def system_config_path(self) -> Path:
        return self.root / "system.json"

    def ensure(self) -> "Layout":
        for name in _SUBDIRS:
            try:
                (self.root / name).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        return self

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "bots_dir": str(self.bots_dir),
            "models_dir": str(self.models_dir),
            "cache_dir": str(self.cache_dir),
            "logs_dir": str(self.logs_dir),
            "benchmark_db": str(self.benchmark_db),
            "system_config_path": str(self.system_config_path),
            "note": self.note,
            "writable": "true" if self.writable else "false",
        }


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def resolve_data_root(override: str | None = None) -> Layout:
    """Pick the best writable data root without ever raising."""

    candidates: list[tuple[Path, str]] = []

    env_root = override or os.getenv("LVC_DATA_ROOT", "").strip()
    if env_root:
        candidates.append((Path(env_root), "LVC_DATA_ROOT"))

    for base in _PREFERRED_ROOTS:
        drive = base[:2]
        if os.path.exists(drive + "\\") or Path(drive).exists():
            candidates.append((Path(base) / APP_NAME, f"preferred volume {drive}"))

    candidates.append((PROJECT_ROOT / f".{APP_NAME}", "project fallback"))

    best: Path | None = None
    best_free = -1
    best_source = ""
    for path, source in candidates:
        free = _free_bytes(path) if path.parent.exists() else 0
        if free > best_free:
            best, best_free, best_source = path, free, source

    assert best is not None
    writable = _writable(best)
    note = f"selected via {best_source}"
    if not writable:
        fallback = PROJECT_ROOT / f".{APP_NAME}"
        if fallback != best and _writable(fallback):
            return Layout(fallback, note=f"{note}; original root unwritable", writable=True)
        return Layout(best, note=f"{note}; ROOT NOT WRITABLE", writable=False)
    return Layout(best, note=note, writable=True)


#: Legacy compatibility: the historical single-file config lived next to app.py.
LEGACY_CONFIG_PATH: Path = PROJECT_ROOT / "config.json"
LEGACY_CONFIG_EXAMPLE: Path = PROJECT_ROOT / "config.example.json"
LEGACY_WEB_DIR: Path = PROJECT_ROOT / "web"

DEFAULT_LAYOUT: Layout = resolve_data_root()


def describe() -> dict[str, str]:
    payload = DEFAULT_LAYOUT.to_dict()
    payload["project_root"] = str(PROJECT_ROOT)
    free = _free_bytes(DEFAULT_LAYOUT.root)
    payload["free_bytes"] = str(free)
    payload["free_gb"] = f"{free / 2 ** 30:.1f}"
    return payload


def iter_data_files(pattern: str = "*") -> Iterable[Path]:
    if not DEFAULT_LAYOUT.root.is_dir():
        return []
    return sorted(DEFAULT_LAYOUT.root.rglob(pattern))
