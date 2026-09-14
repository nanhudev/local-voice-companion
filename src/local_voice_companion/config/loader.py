"""Config loading, validation and atomic persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..core.errors import ConfigurationError
from .defaults import DEFAULT_SYSTEM_CONFIG
from .migration import MigrationResult, migrate
from .paths import (
    DEFAULT_LAYOUT,
    LEGACY_CONFIG_EXAMPLE,
    LEGACY_CONFIG_PATH,
    resolve_data_root,
)
from .schema import SystemConfig, redacted

_CONFIG_CACHE: dict[Path, SystemConfig] = {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in {path}: {exc}", path=str(path)) from exc
    except OSError as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}", path=str(path)) from exc


def load_source_document(layout=DEFAULT_LAYOUT) -> tuple[dict[str, Any], MigrationResult, Path]:
    """Return (migrated dict at CURRENT_VERSION, migration record, source path).

    The first element is always schema-current: callers must never have to
    remember to migrate before validating.
    """

    candidates = [
        layout.system_config_path,
        LEGACY_CONFIG_PATH,
        LEGACY_CONFIG_EXAMPLE,
    ]
    for candidate in candidates:
        if candidate.is_file():
            raw = _read_json(candidate)
            result = migrate(raw)
            return result.config, result, candidate
    return dict(DEFAULT_SYSTEM_CONFIG), MigrationResult({}, 2, 2, ["defaults"]), layout.system_config_path


def load_config(
    path: Path | None = None, layout=DEFAULT_LAYOUT, use_cache: bool = True
) -> SystemConfig:
    """Load + validate the host config. Falls back to defaults, never raises on missing file."""

    target = Path(path) if path else layout.system_config_path
    if use_cache and target in _CONFIG_CACHE:
        return _CONFIG_CACHE[target]

    if path is not None:
        raw = _read_json(target)
        raw = migrate(raw).config
        source: Path = target
    else:
        raw, _result, source = load_source_document(layout)

    try:
        config = SystemConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(
            f"system config validation failed for {source}", detail=json.dumps(exc.errors())
        ) from exc

    _CONFIG_CACHE[target] = config
    return config


def validate_config(raw: dict[str, Any]) -> SystemConfig:
    """Validate an arbitrary document. Used by tests and the API."""

    return SystemConfig.model_validate(migrate(raw).config)


def save_config(config: SystemConfig, path: Path | None = None, layout=DEFAULT_LAYOUT) -> Path:
    """Atomic write. A crash mid-write must never truncate the user's config."""

    target = Path(path) if path else layout.system_config_path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(payload + "\n", encoding="utf-8")
    temp.replace(target)
    _CONFIG_CACHE.pop(target, None)
    return target


def migration_report(layout=DEFAULT_LAYOUT) -> dict[str, Any]:
    try:
        _raw, result, source = load_source_document(layout)
    except ConfigurationError:
        return {"ok": False, "source": str(layout.system_config_path)}
    return {"ok": True, "source": str(source), **result.to_dict()}


def safe_summary(config: SystemConfig) -> dict[str, Any]:
    """Redacted wire form. No API key ever leaves through this."""

    return redacted(config)


def reset_cache() -> None:
    _CONFIG_CACHE.clear()


def ensure_layout(data_root: str | None = None):
    layout = resolve_data_root(data_root) if data_root else DEFAULT_LAYOUT
    return layout.ensure()
