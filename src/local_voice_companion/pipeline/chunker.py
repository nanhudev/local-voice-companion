"""Adaptive sentence/clause chunker.

Waiting for a full LLM reply before synthesising anything is what makes voice
assistants feel slow. Waiting for one character at a time makes them stutter.
This chunker cuts at natural phrase boundaries under multiple conditions:

    * terminal punctuation (。！？!?；\\n)
    * clause punctuation (，,、：) once enough text accumulated
    * hard length cap, split at the last space-like boundary
    * time cap, so a slow first token still yields audio quickly

All parameters are explicit and language-aware.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable

_TERMINAL = "。！？!?；;\n"
_CLAUSE = "，,、：:；;—…"
_SPACE_LIKE = " \t　"


def _language_punctuation(language: str) -> tuple[str, str]:
    normalized = (language or "").lower().split("-")[0]
    if normalized in {"zh", "ja", "yue", "wuu"}:
        return _TERMINAL, _CLAUSE
    # Latin-script languages treat '.' as terminal too, but must not break on
    # decimals and abbreviations, so '.' requires a following space or end.
    return _TERMINAL + ".", _CLAUSE


@dataclass
class AdaptiveTextChunker:
    min_chars: int = 8
    max_chars: int = 120
    max_wait_ms: int = 350
    language: str = "zh"
    _buffer: str = field(default="", init=False)
    _last_flush: float = field(default_factory=time.monotonic, init=False)

    def __post_init__(self) -> None:
        self.min_chars = max(1, self.min_chars)
        self.max_chars = max(self.min_chars + 1, self.max_chars)
        self._terminal, self._clause = _language_punctuation(self.language)

    # -- public API ---------------------------------------------------------

    def push(self, delta: str, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now
        if not delta:
            return []
        self._buffer += delta
        return self._maybe_flush(final=False, now=now)

    def flush(self, now: float | None = None) -> list[str]:
        now = time.monotonic() if now is None else now
        chunks = self._maybe_flush(final=True, now=now)
        leftover = self._buffer.strip()
        self._buffer = ""
        self._last_flush = now
        return chunks + ([leftover] if leftover else [])

    def reset(self) -> None:
        self._buffer = ""
        self._last_flush = time.monotonic()

    @property
    def pending(self) -> str:
        return self._buffer

    # -- internals ----------------------------------------------------------

    def _maybe_flush(self, *, final: bool, now: float) -> list[str]:
        chunks: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            candidate = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:]
            self._last_flush = now
            if self._acceptable(candidate, final):
                chunks.append(candidate)
            elif final:
                chunks.append(candidate)
        return chunks

    def _acceptable(self, text: str, final: bool) -> bool:
        """Drop fragments too short to sound like speech (unless final)."""

        if not text:
            return False
        if final:
            return True
        return len(text.strip()) >= self.min_chars

    def _find_cut(self) -> int | None:
        if not self._buffer:
            return None

        # A hard cap comes first. Punctuation is the *preferred* place to cut,
        # but it is not a licence to overshoot the cap: `max_chars` exists to
        # bound how long the listener waits before the first audio, and a long
        # unpunctuated run (or one very long sentence) would otherwise defeat it.
        if len(self._buffer) >= self.max_chars:
            cut = self._cut_within(self.max_chars)
            if cut is not None:
                return cut

        # 1. terminal punctuation -> definitely a boundary
        for index, char in enumerate(self._buffer):
            if char in self._terminal:
                candidate = index + 1
                # Only accept a boundary that stays inside the cap; otherwise
                # fall through to the hard cap below.
                if candidate <= self.max_chars:
                    return candidate
                break

        # 2. clause punctuation once the accumulator is comfortably long
        if len(self._buffer) >= self.min_chars * 2:
            for index, char in enumerate(self._buffer):
                if char in self._clause and index + 1 >= self.min_chars:
                    if index + 1 > self.max_chars:
                        break
                    # Keep an ellipsis together.
                    if char == "…" and self._buffer[index + 1 : index + 2] == "…":
                        continue
                    return index + 1

        # 3. hard length cap
        if len(self._buffer) >= self.max_chars:
            return self._cut_within(self.max_chars) or self.max_chars

        return None

    def _cut_within(self, limit: int) -> int | None:
        """Best boundary at or before `limit`, preferring punctuation then space.

        Returns None when no natural boundary exists, so the caller can decide
        whether a hard cut at `limit` is appropriate.
        """

        horizon = min(limit, len(self._buffer))
        for index in range(horizon - 1, max(self.min_chars - 1, 0), -1):
            if self._buffer[index] in self._terminal or self._buffer[index] in self._clause:
                return index + 1
        for index in range(horizon - 1, max(self.min_chars - 1, 0), -1):
            if self._buffer[index] in _SPACE_LIKE:
                return index + 1
        return None

    def tick(self, now: float | None = None) -> list[str]:
        """Time-based flush for idle buffers (used by the streaming loop)."""

        now = time.monotonic() if now is None else now
        buffer = self._buffer.strip()
        if not buffer or (now - self._last_flush) * 1000 < self.max_wait_ms:
            return []
        if len(buffer) < self.min_chars:
            return []
        self._buffer = ""
        self._last_flush = now
        return [buffer]


def chunk_text(text: str, language: str = "zh", max_chars: int = 120, min_chars: int = 8) -> list[str]:
    """One-shot convenience wrapper (used by tests and non-streaming TTS)."""

    chunker = AdaptiveTextChunker(
        min_chars=min_chars, max_chars=max_chars, language=language
    )
    result: list[str] = []
    for char in text:
        result.extend(chunker.push(char))
    result.extend(chunker.flush())
    return [item for item in result if item.strip()]


def iter_chunks(deltas: Iterable[str], **kwargs: object) -> Iterable[str]:
    chunker = AdaptiveTextChunker(**kwargs)  # type: ignore[arg-type]
    for delta in deltas:
        yield from chunker.push(delta)
    yield from chunker.flush()
