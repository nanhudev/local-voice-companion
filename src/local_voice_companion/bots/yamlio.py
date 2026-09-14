"""Minimal YAML emitter for portable bot manifests.

Only the subset this project emits is supported: mappings, sequences and
scalars (str/int/float/bool/null). Import goes through JSON instead, so there
is no third-party YAML dependency and no arbitrary object construction.
"""

from __future__ import annotations

from typing import Any

_RESERVED_TRUE = {"true", "yes", "on"}
_RESERVED_FALSE = {"false", "no", "off"}
_RESERVED_NULL = {"null", "~"}


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _needs_quotes(value: str) -> bool:
    lowered = value.strip().lower()
    if value == "" or lowered in _RESERVED_TRUE or lowered in _RESERVED_FALSE or lowered in _RESERVED_NULL:
        return True
    if _looks_numeric(value):
        return True
    specials = set(":{}#&*!|>'\"%@`-")
    if value[0] in specials or value.strip() != value:
        return True
    if "\n" in value or ":" in value or "#" in value:
        return True
    return False


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    text = str(value)
    if "\n" in text:
        # Block scalar keeps multi-line system prompts readable.
        lines = text.splitlines()
        body = "\n".join("  " + line for line in lines)
        return "|\n" + body
    return f'"{text}"' if _needs_quotes(text) else text


def dumps(data: Any, indent: int = 0) -> str:
    pad = "  " * indent
    lines: list[str] = []

    if isinstance(data, dict):
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, dict):
                if not value:
                    lines.append(f"{pad}{key}: {{}}")
                else:
                    lines.append(f"{pad}{key}:")
                    lines.append(dumps(value, indent + 1))
            elif isinstance(value, (list, tuple)):
                if not value:
                    lines.append(f"{pad}{key}: []")
                else:
                    lines.append(f"{pad}{key}:")
                    lines.extend(
                        f"{pad}  - {_list_item(item, indent + 1)}" for item in value
                    )
            else:
                rendered = _scalar(value)
                if rendered.startswith("|"):
                    lines.append(f"{pad}{key}: {rendered}")
                else:
                    lines.append(f"{pad}{key}: {rendered}")
    elif isinstance(data, (list, tuple)):
        for item in data:
            lines.append(f"{pad}- {_list_item(item, indent)}")
    else:
        lines.append(f"{pad}{_scalar(data)}")
    return "\n".join(lines)


def _list_item(item: Any, indent: int) -> str:
    if isinstance(item, dict):
        block = dumps(item, indent + 1)
        first, *rest = block.splitlines()
        return first.strip() + ("\n" + "\n".join(rest) if rest else "")
    if isinstance(item, (list, tuple)):
        inner = ", ".join(_scalar(value) for value in item)
        return f"[{inner}]"
    return _scalar(item)


def dump(data: Any, *, header_comment: str = "") -> str:
    body = dumps(data)
    if header_comment:
        prefix = "\n".join(f"# {line}" for line in header_comment.splitlines())
        return f"{prefix}\n{body}\n"
    return f"{body}\n"
