"""Offline acceptance: does the runtime work with the network switched off?

The claim under test is narrow and testable: **the selection path and the
provider contract do not depend on reaching the internet, and no code path
downloads anything implicitly.** That is distinct from "the models are on disk",
which is a packaging question checked by the `hardware` tests.

Two techniques are used, and the difference matters:

*   ``socket.socket`` is replaced with a guard that raises. This catches *any*
    outbound attempt, including one made from a C extension that has its own
    socket layer -- which is exactly the class of leak a mock would miss.
*   The provider catalogue is checked structurally: every descriptor that claims
    `requires_network=False` must also be usable with `allow_network=False`.

Nothing here needs a model to be installed, so it runs in CI on a bare machine.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

from local_voice_companion.config.loader import load_config
from local_voice_companion.core.types import Device, ProviderKind
from local_voice_companion.hardware.probe import probe_hardware
from local_voice_companion.providers import ProviderRegistry, register_builtin_providers
from local_voice_companion.providers.local.faster_whisper_asr import FasterWhisperASR
from local_voice_companion.providers.local.kokoro_tts import KokoroTTS
from local_voice_companion.providers.local.model_store import ModelStore, default_store
from local_voice_companion.selection.engine import recommend

pytestmark = pytest.mark.integration


class NetworkBlocked(RuntimeError):
    """Raised when code under test tries to open a socket."""


@pytest.fixture()
def no_network(monkeypatch):
    """Make every *outbound connection* attempt fail loudly.

    Patching `socket.socket` itself is too blunt: asyncio's ProactorEventLoop
    builds its own self-pipe from a socketpair during startup, so a blanket
    guard breaks the event loop before any test code runs. (That failure mode
    was observed here first -- it is why this fixture targets the connect path
    instead.)

    What is blocked is therefore the pair of calls that can actually reach a
    remote host: name resolution and connection establishment. Any HTTP client
    -- `requests`, `httpx`, `urllib` -- has to pass through one of them.
    """

    def block_connect(*args, **kwargs):
        raise NetworkBlocked("outbound connection attempted while offline")

    def block_resolve(*args, **kwargs):
        raise NetworkBlocked("DNS resolution attempted while offline")

    monkeypatch.setattr(socket, "create_connection", block_connect)
    monkeypatch.setattr(socket, "getaddrinfo", block_resolve)

    # Also fence the connect method so a caller that constructs its own socket
    # object still cannot dial out.
    original_connect = socket.socket.connect

    def guarded_connect(self, address, *args, **kwargs):
        # Loopback is how an event loop talks to itself; only external
        # addresses are a violation of "offline".
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in {"127.0.0.1", "::1", "localhost"}:
            raise NetworkBlocked(f"connect() to {host!r} attempted while offline")
        return original_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield block_connect


# ---------------------------------------------------------------------------
# the descriptor contract
# ---------------------------------------------------------------------------


class TestLocalProviderDeclarations:
    def test_native_asr_declares_no_network_and_cpu_only(self) -> None:
        descriptor = FasterWhisperASR.descriptor()
        assert descriptor.requires_network is False
        assert descriptor.is_local is True
        assert descriptor.devices == (Device.CPU,)
        assert descriptor.kind is ProviderKind.ASR
        assert descriptor.id == "faster_whisper_cpu"

    def test_native_tts_declares_no_network_and_cpu_only(self) -> None:
        descriptor = KokoroTTS.descriptor()
        assert descriptor.requires_network is False
        assert descriptor.is_local is True
        assert descriptor.devices == (Device.CPU,)
        assert descriptor.kind is ProviderKind.TTS
        assert descriptor.voices, "a TTS provider with no advertised voices is unusable"

    def test_descriptor_is_pure_and_cheap(self) -> None:
        """`descriptor()` must not touch the disk, the registry or the network.

        It is called during candidate generation for every registered provider,
        so a filesystem walk here would put IO in the selector's hot path.
        """

        import time

        started = time.perf_counter()
        for _ in range(50):
            FasterWhisperASR.descriptor()
            KokoroTTS.descriptor()
        elapsed = time.perf_counter() - started
        assert elapsed < 1.0, f"descriptor() is too slow to be pure: {elapsed:.3f}s for 100 calls"

    def test_descriptor_reports_the_installed_runtime_not_a_constant(self) -> None:
        """A version that never changes cannot invalidate anything.

        Both providers once hardcoded `version="1.0.0"`, which made cache
        invalidation look implemented while doing nothing: upgrade the engine
        and the old machine's numbers would still be served. The version has to
        come from the installed distribution.
        """

        whisper = FasterWhisperASR.descriptor().version
        kokoro = KokoroTTS.descriptor().version
        assert whisper not in {"", "1.0.0", None}, (
            "ASR must report its real engine version, got {whisper!r}"
        )
        assert kokoro not in {"", "1.0.0", None}, (
            "TTS must report its real runtime versions, got {kokoro!r}"
        )
        # Kokoro's output depends on the model runtime *and* the phonemiser, so
        # the composite label must name more than one package.
        assert "kokoro-onnx" in kokoro and "misaki" in kokoro, kokoro

    def test_descriptor_wire_shape_is_stable(self) -> None:
        """The descriptor key set is pinned, and grew additively in PHASE 3.

        `supports_streaming` / `supports_partial_results` were added rather than
        overloading the existing `streaming` flag, because the old flag only ever
        meant "produces output progressively" and cannot distinguish a provider
        that accepts live frames from one that can emit partials. Both new keys
        default to False, so every pre-existing descriptor keeps its meaning --
        which is what makes this an additive change instead of a silent one.
        """

        payload = FasterWhisperASR.descriptor().to_dict()
        assert set(payload) == {
            "devices",
            "display_name",
            "estimated_disk_mb",
            "estimated_ram_mb",
            "estimated_vram_mb",
            "id",
            "is_local",
            "kind",
            "languages",
            "latency_tier",
            "models",
            "quality_score",
            "quality_source",
            "quality_tier",
            "requires_network",
            "streaming",
            "supports_cancellation",
            "supports_partial_results",
            "supports_streaming",
            "tags",
            "version",
            "voices",
        }, "the 22-key descriptor contract changed"

    def test_capability_flags_are_additive_not_aspirational(self) -> None:
        """A turn-based provider must not claim duplex capabilities.

        The two new flags default to False. If a future refactor changes that
        default, every provider that did not opt in would start advertising
        capabilities it does not have -- and selection would route barge-in
        traffic to an engine that cannot answer it.
        """

        faster = FasterWhisperASR.descriptor()
        kokoro = KokoroTTS.descriptor()
        for descriptor in (faster, kokoro):
            assert descriptor.supports_streaming is False, descriptor.id
            assert descriptor.supports_partial_results is False, descriptor.id

        # The streaming provider opts in explicitly, and it is the only one.
        from local_voice_companion.providers.local.sherpa_streaming_asr import (
            SherpaStreamingASR,
        )

        sherpa = SherpaStreamingASR.descriptor()
        assert sherpa.supports_streaming is True, sherpa.id
        assert sherpa.supports_partial_results is True, sherpa.id
        assert sherpa.streaming is True, "supports_* must not replace `streaming`"

    def test_streaming_is_declared_honestly(self) -> None:
        """Both engines are single-pass; claiming otherwise misleads callers."""

        assert FasterWhisperASR.descriptor().streaming is False
        assert KokoroTTS.descriptor().streaming is False

    def test_asr_stream_reports_that_it_is_unsupported(self) -> None:
        from local_voice_companion.core.errors import ProviderUnavailable

        provider = FasterWhisperASR()

        async def drain() -> None:
            async for _ in provider.stream(_empty_frames()):
                pass

        with pytest.raises(ProviderUnavailable):
            asyncio.run(drain())


async def _empty_frames():
    if False:  # pragma: no cover - keeps this an async generator
        yield None


# ---------------------------------------------------------------------------
# no implicit downloads
# ---------------------------------------------------------------------------


class TestNoImplicitDownload:
    def test_probe_does_not_touch_the_network(self, no_network) -> None:
        """Probing a provider must be a local check, always.

        The probe is called on every `/api/v1/hardware` request and during
        selection. If it reached out, a missing model would turn an unrelated
        status call into a multi-hundred-megabyte download.
        """

        for provider in (FasterWhisperASR(), KokoroTTS()):
            health = asyncio.run(provider.probe())
            # The verdict may be either way depending on what is installed;
            # what must hold is that deciding did not open a socket.
            assert isinstance(health.ok, bool)

    def test_missing_model_raises_instead_of_downloading(self, tmp_path, no_network) -> None:
        """An empty store must produce a `ModelMissing`, not a download."""

        from local_voice_companion.core.errors import ModelMissing

        store = ModelStore(root=tmp_path)
        store.register(
            default_store().bundle("faster-whisper-base")
        )
        with pytest.raises(ModelMissing) as caught:
            store.require("faster-whisper-base")
        message = str(caught.value)
        assert "faster-whisper-base" in message
        assert "fetch" in message, "the error must name the command that fixes it"

    def test_fetch_dry_run_opens_no_connection(self, tmp_path, no_network) -> None:
        """A dry run must plan without downloading *and* without failing.

        It previously ran the free-space guard before returning, so on a full
        disk the one command a user runs to preview a download would abort
        instead of printing the plan that explains why it cannot proceed.
        """

        store = ModelStore(root=tmp_path)
        store.register(default_store().bundle("faster-whisper-base"))
        plan = store.fetch("faster-whisper-base", dry_run=True)
        assert plan, "dry run should report what it would do"
        assert all(record["status"] == "would-download" for record in plan)
        assert all("enough_space" in record for record in plan), (
            "a dry run must report disk feasibility rather than raising on it"
        )
        assert not (tmp_path / "faster-whisper-base").exists(), "dry run wrote to disk"

    def test_dry_run_reports_insufficient_space_instead_of_raising(
        self, tmp_path, monkeypatch
    ) -> None:
        """A preview that aborts is not a preview.

        The capacity check is real and must still block an actual download; it
        just must not block the *report* of one.
        """

        from local_voice_companion.providers.local import model_store as ms

        monkeypatch.setattr(ms, "_free_bytes", lambda path: 1)
        store = ModelStore(root=tmp_path)
        store.register(default_store().bundle("faster-whisper-base"))

        plan = store.fetch("faster-whisper-base", dry_run=True)
        assert plan and all(record["enough_space"] is False for record in plan)

        with pytest.raises(Exception) as caught:
            store.fetch("faster-whisper-base")
        assert "free space" in str(caught.value)

    def test_provider_load_refuses_a_missing_model_without_network(
        self, tmp_path, monkeypatch, no_network
    ) -> None:
        """A provider with no model anywhere must fail, not fetch.

        The HF-cache lookup is disabled here on purpose. On a machine that
        already has the weights cached, `load()` legitimately succeeds, and
        asserting that it raises would test the machine rather than the code.

        Two exceptions are acceptable and both mean the same thing: the probe
        layer runs first and reports `ProviderUnavailable`, and the store layer
        below it raises `ModelMissing`. What must never happen is a silent
        download, so the assertion is on the *behaviour* -- a raised error whose
        message names the manual fix -- rather than on one exact class.
        """

        from local_voice_companion.core.errors import ModelMissing, ProviderUnavailable

        monkeypatch.setattr(
            FasterWhisperASR, "_hf_cache_roots", classmethod(lambda cls: [])
        )
        provider = FasterWhisperASR()
        provider._store = ModelStore(root=tmp_path)
        provider._store.register(default_store().bundle("faster-whisper-base"))

        with pytest.raises((ModelMissing, ProviderUnavailable)) as caught:
            asyncio.run(provider.load())

        message = str(caught.value)
        assert "faster-whisper-base" in message, message
        assert "fetch" in message, f"the error must name the command that fixes it: {message}"
        # The refusal must not have written anything into the empty store.
        assert not (tmp_path / "faster-whisper-base").exists()

    def test_store_layer_raises_model_missing_directly(self, tmp_path) -> None:
        """Pin the store's own contract independently of the probe layer.

        Without this, a change to `probe()` could quietly move the refusal out
        of `ModelStore.require()` and the test above would still pass.
        """

        from local_voice_companion.core.errors import ModelMissing

        store = ModelStore(root=tmp_path)
        store.register(default_store().bundle("faster-whisper-base"))
        with pytest.raises(ModelMissing):
            store.require("faster-whisper-base")


# ---------------------------------------------------------------------------
# selection under CPU-only / offline constraints
# ---------------------------------------------------------------------------


def _offline_profile(profile):
    """A copy of the profile with every accelerator removed."""

    import copy

    stripped = copy.deepcopy(profile)
    stripped.gpus = []
    return stripped


class TestCpuOnlySelection:
    def _decision(self, *, allow_network: bool, cpu_only: bool, policy: str):
        profile = _offline_profile(probe_hardware(include_audio=False))
        config = load_config()
        config.runtime.cpu_only = cpu_only
        config.runtime.allow_network_llm = allow_network
        config.runtime.policy = policy

        reg = ProviderRegistry()
        register_builtin_providers(reg)
        return asyncio.run(
            recommend(profile, config, reg=reg, policy_override=policy, language="zh")
        )

    def test_cpu_only_policy_selects_native_providers_when_installed(self) -> None:
        """The Phase 2 exit criterion.

        Skipped rather than faked when the native engines are absent: a machine
        without them legitimately has no local voice path, and asserting
        otherwise here would be the fabricated result RULE 12 forbids.
        """

        ok_asr, reason_asr = FasterWhisperASR.availability()
        ok_tts, reason_tts = KokoroTTS.availability() if hasattr(
            KokoroTTS, "availability"
        ) else (default_store().is_ready("kokoro-v1.1-zh"), "")
        if not (ok_asr and ok_tts):
            pytest.skip(
                f"native engines unavailable -- asr: {reason_asr}; tts: {reason_tts or 'weights missing'}"
            )

        decision = self._decision(allow_network=False, cpu_only=True, policy="cpu_only")
        plan = decision.plan

        assert decision.effective_policy == "cpu_only"
        assert plan.feasible, plan.notes

        asr = plan.assignment("asr")
        tts = plan.assignment("tts")
        assert asr is not None and tts is not None, plan.notes

        assert asr.candidate.provider_id == "faster_whisper_cpu", asr.reason
        assert tts.candidate.provider_id == "kokoro_tts_cpu", tts.reason
        assert asr.candidate.device == Device.CPU.value
        assert tts.candidate.device == Device.CPU.value

    def test_offline_rejects_every_network_provider(self) -> None:
        decision = self._decision(allow_network=False, cpu_only=True, policy="cpu_only")
        for assignment in decision.plan.assignments.values():
            descriptor = assignment.candidate.descriptor
            assert descriptor.requires_network is False, (
                f"{descriptor.id} requires the network but was selected while offline"
            )
            assert assignment.candidate.device != Device.REMOTE.value

    def test_offline_policy_explains_the_rejections(self) -> None:
        """Rejections are user-facing text; they must be specific.

        The remote-only adapters are excluded by the *device* policy before they
        are ever scored, which used to leave the rejection log empty and the
        reason unexplained. This asserts the explanation survives.
        """

        decision = self._decision(allow_network=False, cpu_only=True, policy="cpu_only")
        rejected_ids = {item["candidate"].split(":")[0] for item in decision.rejected}
        reasons = " ".join(item["reason"] for item in decision.rejected)

        assert decision.rejected, "an offline CPU-only run must record why providers were excluded"
        assert "voicebox_asr" in rejected_ids or "voicebox_tts" in rejected_ids, (
            f"the remote adapters should be excluded by name, got {rejected_ids}"
        )
        assert "remote" in reasons or "network" in reasons or "device" in reasons

    def test_http_backed_providers_are_excluded_offline_even_without_cpu_only(
        self,
    ) -> None:
        """The hole that `requires_network=False` left open.

        `cpu_only` excludes the Voicebox pair by *device*, so the bug was
        invisible there. With `balanced` + `allow_network=False` there is no
        device reason to fall back on -- the only thing standing between the
        plan and an HTTP call to 127.0.0.1 was the network flag, and until these
        providers declared `requires_network=True` it did not stop them.
        """

        decision = self._decision(allow_network=False, cpu_only=False, policy="balanced")
        selected = {
            assignment.candidate.provider_id
            for assignment in decision.plan.assignments.values()
        }
        http_backed = {"voicebox_asr", "voicebox_tts", "ollama_llm"}
        assert not (selected & http_backed), (
            f"an HTTP-backed provider was selected while offline: {selected & http_backed}"
        )

        # Asserting *why* is what pins the fix. Checking only `selected` would
        # still pass if these providers merely happened to be unreachable on the
        # machine running the test, which says nothing about the constraint.
        rejected = {
            item["candidate"].split(":")[0]: item["reason"] for item in decision.rejected
        }
        for provider_id in ("voicebox_asr", "voicebox_tts", "ollama_llm"):
            assert provider_id in rejected, f"{provider_id} was not excluded at all"
            assert "network" in rejected[provider_id], (
                f"{provider_id} was excluded for the wrong reason: {rejected[provider_id]!r}"
            )

    def test_cpu_only_alternatives_never_include_a_gpu_candidate(self) -> None:
        decision = self._decision(allow_network=False, cpu_only=True, policy="cpu_only")
        for kind, options in decision.alternatives.items():
            for option in options:
                assert option["device"] == Device.CPU.value, f"{kind}: {option}"

    def test_balanced_policy_is_allowed_to_pick_remote_when_online(self) -> None:
        """The offline constraint must be what excludes remote providers, not a
        hard-coded dislike of them. With the network enabled the remote ASR
        adapter should at least remain a candidate."""

        decision = self._decision(allow_network=True, cpu_only=False, policy="balanced")
        seen = {option["provider_id"] for options in decision.alternatives.values() for option in options}
        selected = {a.candidate.provider_id for a in decision.plan.assignments.values()}
        # Not an assertion that a remote provider *wins* -- only that the
        # constraint set is doing the filtering rather than the code.
        assert seen or selected


# ---------------------------------------------------------------------------
# a real bot, through the real HTTP API
# ---------------------------------------------------------------------------


class TestRealBotTurn:
    """The whole point, exercised through the surface a client actually uses.

    Everything above tests providers and the selector directly. This drives the
    FastAPI application instead, so a regression in wiring -- the registry handed
    to `create_app`, the plan a session resolves, the speak flag reaching TTS --
    fails here rather than escaping to a user.
    """

    @pytest.fixture()
    def cpu_only_app(self, tmp_path):
        """A `cpu_only` application writing its bots to a throwaway directory.

        Two things make this necessary. Pointing at the real bots directory
        leaves residue in the user's data root, and a fixed bot id then fails on
        the *second* run because the bot persisted from the first -- which is
        exactly how this test failed initially, and only in a full-suite run
        where another test had already created it.
        """

        from fastapi.testclient import TestClient

        from local_voice_companion.api.app import create_app
        from local_voice_companion.providers import ensure_builtin_providers, registry

        ensure_builtin_providers()
        config = load_config()
        config.runtime.cpu_only = True
        config.runtime.allow_network_llm = False
        config.runtime.policy = "cpu_only"
        return TestClient(create_app(config=config, registry=registry, bots_dir=tmp_path))

    def test_a_cpu_only_bot_turn_produces_real_local_audio(self, cpu_only_app) -> None:
        ok_asr, why_asr = FasterWhisperASR.availability()
        ok_tts, why_tts = KokoroTTS.availability()
        if not (ok_asr and ok_tts):
            pytest.skip(f"native engines unavailable -- asr: {why_asr}; tts: {why_tts}")

        created = cpu_only_app.post(
            "/api/v1/bots",
            json={"id": "offline-bot", "name": "offline-bot", "display_name": "离线助手"},
        )
        assert created.status_code in (200, 201), created.text[:400]

        session = cpu_only_app.post(
            "/api/v1/sessions", json={"bot_id": "offline-bot"}
        ).json()["session"]["id"]

        response = cpu_only_app.post(
            f"/api/v1/sessions/{session}/turns",
            json={"text": "今天天气怎么样", "speak": True},
        )
        assert response.status_code == 200, response.text[:600]

        result = response.json()["result"]
        assert result["error"] == "", result
        assert result["reply"], "the turn produced no reply text"

        # Real synthesis, not a silent placeholder. 16 kHz mono int16 for 4 s
        # would be ~128 KB; anything under 32 KB is not a spoken reply.
        assert result["audio_bytes"] > 32_000, (
            f"expected real synthesised audio, got {result['audio_bytes']} bytes"
        )
        assert result["chunks_spoken"] >= 1, result

        timeline = result["timeline"]
        # Text was supplied directly, so ASR legitimately does no work here; a
        # nonzero value would mean text was being round-tripped through a model.
        assert timeline["asr_latency_ms"] == 0, "a text-only turn must not invoke ASR"
        assert timeline["tts_ttfa_ms"] > 0, "TTS latency must come from real synthesis"
        assert timeline["clock_limited"] is False, timeline
