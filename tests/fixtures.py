"""Small binary fixtures shared by the test tiers.

Kept out of ``conftest.py`` so test modules can import it with a normal import
statement instead of relying on pytest's rootdir insertion.
"""

from __future__ import annotations

import math
import struct


def make_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit PCM in a minimal RIFF/WAVE container.

    Built by hand rather than imported from the runtime so the API tier proves
    it can strip a real third-party container, not merely re-parse its own
    writer's output.
    """

    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits
    )
    header += b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def make_tone(milliseconds: int = 1000, sample_rate: int = 16000, amplitude: int = 5000) -> bytes:
    """A deterministic sine tone as raw 16-bit mono PCM.

    Deterministic on purpose: a recorded fixture would be one more binary in
    the repo, and the fake providers only care about the length.
    """

    frames = int(sample_rate * milliseconds / 1000)
    return b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 220.0 * index / sample_rate)))
        for index in range(frames)
    )


__all__ = ["make_wav", "make_tone"]
