"""Filesystem bot store.

Storage lives under the resolved data root, never inside the source tree and
never at a machine-absolute path embedded in the manifest itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.errors import ConfigurationError, NotFound
from ..config.paths import DEFAULT_LAYOUT
from .schema import BotManifest
from .yamlio import dump as yaml_dump

MANIFEST_SUFFIX = ".json"


class BotStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else DEFAULT_LAYOUT.bots_dir
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- paths --------------------------------------------------------------

    def _path(self, bot_id: str) -> Path:
        return self.directory / f"{bot_id}{MANIFEST_SUFFIX}"

    # -- crud ---------------------------------------------------------------

    def list(self) -> list[BotManifest]:
        manifests: list[BotManifest] = []
        for path in sorted(self.directory.glob(f"*{MANIFEST_SUFFIX}")):
            manifest = self._read(path)
            if manifest is not None:
                manifests.append(manifest)
        return sorted(manifests, key=lambda item: item.id)

    def get(self, bot_id: str) -> BotManifest:
        path = self._path(bot_id)
        if not path.is_file():
            raise NotFound(f"bot not found: {bot_id}", bot_id=bot_id)
        manifest = self._read(path)
        if manifest is None:
            raise NotFound(f"bot unreadable: {bot_id}", bot_id=bot_id)
        return manifest

    def exists(self, bot_id: str) -> bool:
        return self._path(bot_id).is_file()

    def create(self, manifest: BotManifest, *, overwrite: bool = False) -> BotManifest:
        if self.exists(manifest.id) and not overwrite:
            raise ConfigurationError(f"bot already exists: {manifest.id}", bot_id=manifest.id)
        return self._write(manifest)

    def update(self, bot_id: str, patch: dict[str, Any]) -> BotManifest:
        manifest = self.get(bot_id).merge(patch)
        if manifest.id != bot_id and self.exists(manifest.id):
            raise ConfigurationError(f"cannot rename to existing id: {manifest.id}")
        old_path = self._path(bot_id)
        new = self._write(manifest)
        if manifest.id != bot_id:
            old_path.unlink(missing_ok=True)
        return new

    def upsert(self, manifest: BotManifest) -> BotManifest:
        return self._write(manifest)

    def delete(self, bot_id: str) -> bool:
        path = self._path(bot_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    # -- portable exchange --------------------------------------------------

    def export_yaml(self, bot_id: str) -> str:
        manifest = self.get(bot_id)
        return yaml_dump(
            manifest.to_dict(),
            header_comment=(
                f"Local Voice Companion bot manifest\n"
                f"Portable across machines: importing this file re-runs hardware-aware selection.\n"
                f"bot: {manifest.id}"
            ),
        )

    def import_document(self, text: str, *, overwrite: bool = False) -> BotManifest:
        """Import a manifest. Accepts our own YAML export or plain JSON."""

        text = text.strip()
        if not text:
            raise ConfigurationError("empty manifest")

        payload: dict[str, Any] | None = None
        if text.lstrip().startswith("{"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"invalid JSON manifest: {exc}") from exc
        if payload is None:
            payload = self._parse_yaml_subset(text)
        if payload is None:
            raise ConfigurationError(
                "unsupported manifest format (expected the YAML this tool exports, or JSON)"
            )

        manifest = BotManifest.from_dict(payload)
        return self.create(manifest, overwrite=overwrite)

    # -- internals ----------------------------------------------------------

    def _read(self, path: Path) -> BotManifest | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return BotManifest.from_dict(payload)
        except (OSError, json.JSONDecodeError):
            return None
        except Exception:  # noqa: BLE001 - a corrupt file must not break listing
            return None

    def _write(self, manifest: BotManifest) -> BotManifest:
        path = self._path(manifest.id)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temp.replace(path)
        return manifest

    @staticmethod
    def _parse_yaml_subset(text: str) -> dict[str, Any] | None:
        """Parse only the restricted YAML this project writes.

        Supports nested mappings, block sequences, inline [lists] and quoted
        scalars plus the `|` block scalar. Anything else is rejected rather
        than guessed at, and no Python objects are constructed.
        """

        lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not lines:
            return None
        try:
            value, _index = _parse_block(lines, 0, 0)
        except ValueError:
            return None
        if not isinstance(value, dict):
            return None
        return value


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _scalar(text: str) -> Any:
    text = text.strip()
    if text == "" or text in {"null", "~"}:
        return None
    if text[0] in "\"'":
        quote = text[0]
        return text[1:].rsplit(quote, 1)[0]
    if text[0] == "[" and text[-1] == "]":
        inner = text[1:-1].strip()
        if not inner:
            return []
        parts: list[Any] = []
        current = ""
        in_quotes = False
        for char in inner:
            if char in "\"'":
                in_quotes = not in_quotes
                current += char
            elif char == "," and not in_quotes:
                parts.append(_scalar(current))
                current = ""
            else:
                current += char
        if current.strip():
            parts.append(_scalar(current))
        return parts
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _parse_block(lines: list[str], index: int, indent: int) -> tuple[Any, int]:
    """Return (value, next_index) for the block starting at `index`."""

    if index >= len(lines):
        return None, index
    first = lines[index]
    if first[indent:].startswith("- "):
        return _parse_sequence(lines, index, indent)
    return _parse_mapping(lines, index, indent)


def _parse_mapping(lines: list[str], index: int, indent: int) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line = lines[index]
        line_indent = _indent_of(line)
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError("unexpected indent")
        if ":" not in line:
            raise ValueError("expected key: value")
        key, _, rest = line.strip().partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest == "|":
            block: list[str] = []
            index += 1
            while index < len(lines) and _indent_of(lines[index]) > indent:
                block.append(lines[index].strip())
                index += 1
            result[key] = "\n".join(block)
            continue
        if rest == "":
            index += 1
            if index < len(lines) and (
                _indent_of(lines[index]) > indent
                or lines[index][line_indent:].startswith("- ")
            ):
                child_indent = _indent_of(lines[index])
                value, index = _parse_block(lines, index, child_indent)
                result[key] = value
            else:
                result[key] = None
            continue
        result[key] = _scalar(rest)
        index += 1
    return result, index


def _parse_sequence(lines: list[str], index: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines):
        line = lines[index]
        line_indent = _indent_of(line)
        if line_indent < indent or not line[line_indent:].startswith("-"):
            break
        content = line[line_indent + 1:].strip()
        if not content:
            index += 1
            if index < len(lines) and _indent_of(lines[index]) > indent:
                value, index = _parse_mapping(lines, index, _indent_of(lines[index]))
                items.append(value)
            continue
        if ":" in content and not content.startswith(("\"", "'", "[")):
            # Inline first key of a nested mapping.
            synthetic = [" " * (indent + 2) + content]
            consumed = index + 1
            while consumed < len(lines) and _indent_of(lines[consumed]) > indent:
                synthetic.append(lines[consumed])
                consumed += 1
            value, _ = _parse_mapping(synthetic, 0, indent + 2)
            items.append(value)
            index = consumed
            continue
        items.append(_scalar(content))
        index += 1
    return items, index


def store_summary(store: BotStore) -> dict[str, Any]:
    bots: Sequence[BotManifest] = store.list()
    return {
        "directory": str(store.directory),
        "count": len(bots),
        "bots": [
            {
                "id": bot.id,
                "name": bot.name,
                "language": bot.language.primary,
                "voice": bot.tts.voice or "auto",
                "policy": bot.runtime.policy.value,
                "template": bot.template,
            }
            for bot in bots
        ],
    }
