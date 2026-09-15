"""Native, fully-local inference providers.

Everything in this package runs on the machine it is installed on: no HTTP
gateway, no API keys, no telemetry. That is the whole point of the phase they
were added in -- before them, every provider that could actually hear or speak
was a remote adapter.

Design rules these modules follow (see docs/PROVIDER_AUTHORING.md):

1.  Heavy imports happen inside methods, never at module import time. Importing
    the package must stay fast and must not fail on a machine without onnxruntime.
2.  `probe()` reports *why* it is unavailable, not merely that it is. The
    distinction between "dependency missing" and "model missing" is the
    difference between a one-line pip hint and a multi-hundred-megabyte
    download.
3.  Nothing downloads silently. A missing model is an error with an explicit
    fetch instruction.
4.  Blocking inference is offloaded off the event loop.
5.  `descriptor()` may touch nothing slow or remote: no network, no model load,
    no filesystem walk, no heavy import. Reading a *distribution version* from
    `importlib.metadata` is allowed and deliberate -- it costs ~1-6 ms and has
    no failure mode, and reporting the real installed version is what makes a
    cached benchmark truthfully invalidatable.
"""

from __future__ import annotations

from . import runtime_probe
from .faster_whisper_asr import FasterWhisperASR
from .kokoro_tts import KokoroTTS
from .model_store import MODEL_ROOT, ModelArtifact, ModelBundle, ModelStore, default_store

NATIVE_PROVIDERS = (
    FasterWhisperASR,
    KokoroTTS,
)

__all__ = [
    "FasterWhisperASR",
    "KokoroTTS",
    "MODEL_ROOT",
    "ModelArtifact",
    "ModelBundle",
    "ModelStore",
    "NATIVE_PROVIDERS",
    "default_store",
    "runtime_probe",
]
