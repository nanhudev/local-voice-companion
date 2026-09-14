"""Local Voice Companion 2.0 -- an adaptive local voice runtime.

The package is layered and the dependencies only ever point inward:

    core        -> pure runtime logic (no provider names, no HTTP, no FastAPI)
    config      -> typed, versioned documents + filesystem layout
    hardware    -> best-effort capability probing
    providers   -> pluggable ASR / LLM / TTS / VAD implementations
    selection   -> candidate generation, benchmarking, scoring, resource planning
    pipeline    -> bounded queues, chunking, WAV helpers
    bots        -> portable BotManifest definition and store
    api         -> FastAPI adapters (REST, WebSocket, OpenAI compatibility)
    observability -> metrics derived from turn timelines
    compat      -> adapters for the pre-2.0 single-file runtime
"""

from __future__ import annotations

__version__ = "2.0.0"

SCHEMA_VERSION = 2

__all__ = ["__version__", "SCHEMA_VERSION"]
