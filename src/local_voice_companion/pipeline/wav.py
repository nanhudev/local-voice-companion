"""WAV container helpers and shared text utilities.

These two small utilities used to live inline in `app.py`. They are extracted
verbatim (same behaviour, same semantics) so the legacy launcher and the new
runtime share one implementation instead of drifting apart.
"""

from __future__ import annotations

import io
import re
import wave
from typing import Tuple

SENTENCE_END = re.compile(r"[。！？!?；;\n]")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
UNOPENED_THINK = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
ROLE_PREFIX = re.compile(r"^(?:助手|机器人|assistant)\s*[:：]\s*", re.IGNORECASE)


def wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw little-endian PCM in a single-channel WAV container."""

    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return output.getvalue()


def wav_duration_ms(blob: bytes) -> float:
    try:
        with wave.open(io.BytesIO(blob), "rb") as handle:
            return handle.getnframes() / handle.getframerate() * 1000.0
    except (wave.Error, EOFError):
        return 0.0


def ready_sentences(buffer: str, minimum: int = 4) -> Tuple[list[str], str]:
    """Split completed sentences out of a streaming buffer.

    Returns (complete_sentences, remaining_pending_text).
    """

    chunks: list[str] = []
    start = 0
    for match in SENTENCE_END.finditer(buffer):
        end = match.end()
        chunk = buffer[start:end].strip()
        if len(chunk) >= minimum:
            chunks.append(chunk)
            start = end
    return chunks, buffer[start:]


def clean_model_text(text: str) -> str:
    """Strip reasoning blocks and role prefixes that some local runtimes emit."""

    text = THINK_BLOCK.sub("", text)
    # Some runtimes stream an unclosed thought block before disconnecting.
    text = UNOPENED_THINK.sub("", text)
    text = ROLE_PREFIX.sub("", text)
    return text.strip()
