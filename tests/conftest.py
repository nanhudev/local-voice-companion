"""Shared pytest fixtures.

Every test in this suite must run on a machine with no GPU, no models, no
Ollama and no Voicebox. Anything that needs hardware is skipped or served by a
fake provider -- see RULE 12: never claim a hardware path was validated when it
was not exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture(autouse=True)
def _quiet_registry():
    """Keep the global provider registry clean between tests."""

    from local_voice_companion.providers import registry

    registry.clear_instances()
    yield
    registry.clear_instances()


@pytest.fixture
def synthetic_profile():
    """A deterministic RTX-2070-class profile. No probing, no hardware needed."""

    from local_voice_companion.hardware.profile import synthetic_profile

    return synthetic_profile(cpu_threads=12, ram_mb=16384, vram_mb=8192, accelerator="cuda")


@pytest.fixture
def cpu_only_profile():
    from local_voice_companion.hardware.profile import synthetic_profile

    return synthetic_profile(cpu_threads=8, ram_mb=16384, vram_mb=0)


@pytest.fixture
def fake_pipeline():
    """A fully loaded FakeASR -> FakeLLM -> FakeTTS -> FakeVAD pipeline."""

    from local_voice_companion.core.orchestrator import Pipeline
    from local_voice_companion.providers.fake import FakeASR, FakeLLM, FakeTTS, FakeVAD

    pipeline = Pipeline(asr=FakeASR({}), llm=FakeLLM({}), tts=FakeTTS({}), vad=FakeVAD({}))
    for name in ("asr", "llm", "tts", "vad"):
        provider = getattr(pipeline, name)
        provider.lifecycle.transition("AVAILABLE")
        provider.lifecycle.transition("LOADING")
        provider.lifecycle.transition("READY")
    return pipeline


@pytest.fixture
def isolated_registry():
    """A registry containing only the fake providers."""

    from local_voice_companion.providers.fake import FAKE_PROVIDERS
    from local_voice_companion.providers.registry import ProviderRegistry

    reg = ProviderRegistry()
    for cls in FAKE_PROVIDERS:
        reg.register(cls)
    return reg
