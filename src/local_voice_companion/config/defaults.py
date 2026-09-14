"""Default values and policy weights.

Every tunable number that influences selection lives here. Scoring weights must
never be sprinkled across the selector implementation.
"""

from __future__ import annotations

from typing import Any

from ..core.types import Device, SelectionPolicy

#: Score weights per policy. Each row sums to 1.0 (asserted by a unit test).
POLICY_WEIGHTS: dict[str, dict[str, float]] = {
    SelectionPolicy.ULTRA_LOW_LATENCY.value: {
        "latency": 0.50,
        "quality": 0.10,
        "resource_safety": 0.15,
        "language_fit": 0.10,
        "startup_cost": 0.10,
        "stability": 0.05,
    },
    SelectionPolicy.BALANCED.value: {
        "latency": 0.30,
        "quality": 0.25,
        "resource_safety": 0.20,
        "language_fit": 0.15,
        "startup_cost": 0.05,
        "stability": 0.05,
    },
    SelectionPolicy.QUALITY.value: {
        "latency": 0.12,
        "quality": 0.50,
        "resource_safety": 0.15,
        "language_fit": 0.15,
        "startup_cost": 0.03,
        "stability": 0.05,
    },
    SelectionPolicy.LOW_MEMORY.value: {
        "latency": 0.15,
        "quality": 0.15,
        "resource_safety": 0.50,
        "language_fit": 0.10,
        "startup_cost": 0.05,
        "stability": 0.05,
    },
    SelectionPolicy.CPU_ONLY.value: {
        "latency": 0.25,
        "quality": 0.20,
        "resource_safety": 0.30,
        "language_fit": 0.15,
        "startup_cost": 0.05,
        "stability": 0.05,
    },
    SelectionPolicy.AUTO.value: {
        # Identical to BALANCED; kept distinct so `auto` can diverge later and
        # so the UI can show that AUTO resolved to a concrete policy.
        "latency": 0.30,
        "quality": 0.25,
        "resource_safety": 0.20,
        "language_fit": 0.15,
        "startup_cost": 0.05,
        "stability": 0.05,
    },
    SelectionPolicy.MANUAL.value: {
        "latency": 0.0,
        "quality": 0.0,
        "resource_safety": 0.0,
        "language_fit": 0.0,
        "startup_cost": 0.0,
        "stability": 0.0,
    },
}

#: AUTO resolves to a concrete policy using the machine's own envelope.
AUTO_RESOLUTION_RULES: tuple[tuple[str, str], ...] = (
    # (condition name, resolved policy)
    ("cpu_only_machine", SelectionPolicy.CPU_ONLY.value),
    ("tight_vram", SelectionPolicy.LOW_MEMORY.value),
    ("weak_cpu", SelectionPolicy.LOW_MEMORY.value),
    ("high_end_gpu", SelectionPolicy.QUALITY.value),
)

AUTO_THRESHOLDS: dict[str, Any] = {
    "tight_vram_mb": 4096,
    "high_end_vram_mb": 12288,
    "weak_cpu_threads": 8,
    "weak_ram_mb": 8192,
}

DEVICE_PREFERENCE_ORDER: tuple[str, ...] = (
    Device.CUDA.value,
    Device.DIRECTML.value,
    Device.METAL.value,
    Device.ROCM.value,
    Device.REMOTE.value,
    Device.CPU.value,
)

#: Device ordering the selector walks when applying `prefer_devices`.
DEFAULT_SYSTEM_CONFIG: dict[str, Any] = {
    "schema_version": 2,
    "server": {"host": "127.0.0.1", "port": 17831, "auth_token_env": "LVC_AUTH_TOKEN"},
    "audio": {
        "sample_rate": 16000,
        "block_ms": 30,
        "pre_roll_ms": 180,
        "end_silence_ms": 360,
        "min_speech_ms": 270,
        "max_speech_seconds": 12.0,
        "minimum_rms": 420,
        "silence_after_playback_ms": 350,
    },
    "pipeline": {
        "max_concurrent_tts_items": 8,
        "max_pending_audio_chunks": 64,
        "drop_policy": "oldest",
        "chunk_min_chars": 8,
        "chunk_max_chars": 120,
        "history_turns": 24,
        "max_tokens": 64,
        "temperature": 0.35,
    },
    "runtime": {
        "policy": SelectionPolicy.BALANCED.value,
        "prefer_devices": ["cuda", "directml", "metal", "cpu", "remote"],
        "allow_network_llm": True,
        "allow_local_llm": True,
        "cpu_only": False,
        "vram_reserve_mb": 512,
        "ram_reserve_mb": 1024,
        "llm_api_key_env": "LVC_LLM_API_KEY",
    },
    "providers": {},
    "legacy": {},
    "observability": {},
}


def policy_weights(policy: str | SelectionPolicy) -> dict[str, float]:
    key = policy.value if isinstance(policy, SelectionPolicy) else str(policy)
    if key not in POLICY_WEIGHTS:
        raise KeyError(f"unknown selection policy: {key}")
    return dict(POLICY_WEIGHTS[key])
