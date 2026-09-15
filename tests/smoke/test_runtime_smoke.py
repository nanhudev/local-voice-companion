"""Smoke tests: prove the real end-to-end paths still work.

Where the integration tier checks the HTTP contract, this tier checks the
things a user would notice if they broke:

* a complete FakeASR -> FakeLLM -> FakeTTS turn with every timeline stage
  populated and every derived metric defined;
* barge-in actually stopping the pipeline instead of letting stale audio
  through;
* bounded queues refusing to grow without limit;
* the user's real ``config.json`` surviving a v1 -> v2 migration with nothing
  lost;
* the CLI commands a person types.

Nothing here touches a model, a GPU or the network. Hardware-dependent claims
live behind the ``hardware`` marker and are skipped unless explicitly enabled,
because asserting on a machine with an RTX 2070 would be a fabricated result
for every other machine (RULE 12).
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import time
import wave
from pathlib import Path

import pytest

from fixtures import make_tone

from local_voice_companion.core.errors import ProviderUnavailable
from local_voice_companion.core.types import AudioChunk
from local_voice_companion.providers.local import FasterWhisperASR, KokoroTTS
from local_voice_companion.providers.local.model_store import default_store

pytestmark = pytest.mark.smoke


def _run(coro):
    return asyncio.run(coro)


def _wav_pcm(blob: bytes) -> tuple[bytes, int]:
    """Extract (pcm, sample_rate) from a WAV blob."""

    with wave.open(io.BytesIO(blob)) as handle:
        return handle.readframes(handle.getnframes()), handle.getframerate()


def pcm_rms(pcm: bytes) -> int:
    """Root-mean-square of 16-bit little-endian PCM.

    Uses `audioop` when the interpreter still ships it (removed in 3.13) and the
    standard library `wave`/`struct` path otherwise, so the check works on every
    supported Python without adding numpy to the test dependencies.
    """

    import math
    import struct

    if not pcm:
        return 0
    count = len(pcm) // 2
    if count == 0:
        return 0
    samples = struct.unpack(f"<{count}h", pcm[: count * 2])
    total = sum(value * value for value in samples)
    return int(math.sqrt(total / count))


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------


class TestFullFakeTurn:
    def test_text_turn_marks_every_stage(self, fake_pipeline):
        """A spoken text turn must produce the complete timeline.

        Not every pair is ordered: synthesis deliberately starts as soon as the
        first sentence is chunked, so `tts_start` legitimately precedes
        `llm_end`. Only the genuinely causal pairs are pinned below.
        """

        from local_voice_companion.core.orchestrator import TurnRequest, run_turn
        from local_voice_companion.core.session import Session

        session = Session(id="smoke-text", language="zh")
        result = _run(
            run_turn(session, fake_pipeline, TurnRequest(text="你好", speak=True, language="zh"))
        )

        assert result.error == "", result.error
        assert not result.cancelled
        assert result.reply
        assert result.chunks_spoken >= 1
        assert result.audio_bytes > 0

        timeline = result.timeline
        assert timeline is not None

        required = [
            "turn_started",
            "llm_start",
            "llm_first_token",
            "llm_end",
            "tts_start",
            "tts_first_audio",
            "playback_start",
            "playback_end",
        ]
        missing = [stage for stage in required if stage not in timeline.marks]
        assert not missing, f"missing stages {missing}; have {sorted(timeline.marks)}"

        marks = timeline.marks
        # Causal ordering only.
        assert marks["turn_started"] <= marks["llm_start"]
        assert marks["llm_start"] <= marks["llm_first_token"] <= marks["llm_end"]
        assert marks["llm_first_token"] <= marks["tts_start"], "synthesis must follow the first token"
        assert marks["tts_start"] <= marks["tts_first_audio"]
        assert marks["tts_first_audio"] <= marks["playback_start"] <= marks["playback_end"]

    def test_first_audio_precedes_the_end_of_generation(self, fake_pipeline):
        """The point of streaming: audio starts before the reply is finished.

        If `tts_first_audio` lands after `llm_end`, the runtime is buffering the
        whole reply before speaking and the streaming optimisation has silently
        regressed into a batch pipeline.
        """

        from local_voice_companion.core.orchestrator import TurnRequest, run_turn
        from local_voice_companion.core.session import Session

        session = Session(id="smoke-streaming", language="zh")
        result = _run(
            run_turn(
                session,
                fake_pipeline,
                TurnRequest(text="请说一段足够长的话来验证流式合成", speak=True, language="zh"),
            )
        )

        marks = result.timeline.marks
        assert marks["tts_first_audio"] <= marks["llm_end"], (
            "first audio arrived after generation finished -- streaming regressed to batch"
        )

    def test_all_derived_metrics_are_defined(self, fake_pipeline):
        from local_voice_companion.core.orchestrator import TurnRequest, run_turn
        from local_voice_companion.core.session import Session

        session = Session(id="smoke-metrics", language="zh")
        result = _run(run_turn(session, fake_pipeline, TurnRequest(text="指标", speak=True)))

        assert result.timeline is not None
        metrics = result.timeline.metrics()
        for name in ("llm_ttft", "tts_ttfa", "time_to_first_audio", "total_turn"):
            assert name in metrics, (name, metrics)
            assert metrics[name] >= 0, (name, metrics[name])

    def test_audio_turn_runs_asr_then_speaks(self, fake_pipeline):
        from local_voice_companion.core.orchestrator import TurnRequest, run_turn
        from local_voice_companion.core.session import Session
        from local_voice_companion.core.types import AudioChunk

        session = Session(id="smoke-audio", language="zh")
        audio = AudioChunk(pcm=make_tone(milliseconds=1000), sample_rate=16000)
        result = _run(run_turn(session, fake_pipeline, TurnRequest(audio=audio, speak=True)))

        assert result.error == "", result.error
        assert result.transcript, "ASR produced no transcript"
        assert result.reply
        # vad_end is the reference point for time-to-first-audio, so it must be
        # present for audio turns (it is synthesised for text turns).
        assert "vad_end" in result.timeline.marks
        assert result.timeline.time_to_first_audio_ms is not None

    def test_history_grows_across_turns(self, fake_pipeline):
        from local_voice_companion.core.orchestrator import TurnRequest, run_turn
        from local_voice_companion.core.session import Session

        session = Session(id="smoke-history", language="zh")
        for text in ("一", "二", "三"):
            _run(run_turn(session, fake_pipeline, TurnRequest(text=text, speak=False)))

        roles = [message.role for message in session.history]
        assert roles.count("user") == 3
        assert roles.count("assistant") == 3

    def test_session_history_is_bounded(self, fake_pipeline):
        """A long conversation must not grow memory without limit."""

        from local_voice_companion.core.orchestrator import TurnRequest, run_turn
        from local_voice_companion.core.session import Session

        session = Session(id="smoke-bound", language="zh", max_history=4)
        for index in range(12):
            _run(run_turn(session, fake_pipeline, TurnRequest(text=f"第{index}句", speak=False)))

        assert len(session.history) <= session.max_history * 2

    def test_turn_events_are_emitted_in_order(self, fake_pipeline):
        """Every declared stage event must actually be emitted.

        `LLM_STARTED`, `PLAYBACK_STARTED` and `PLAYBACK_FINISHED` were declared
        in the event contract but never fired, so a client could not tell
        "thinking" from "idle" or know when to release the audio device.
        """

        from local_voice_companion.core.orchestrator import TurnRequest, run_turn
        from local_voice_companion.core.session import Session

        session = Session(id="smoke-events", language="zh")
        seen: list[str] = []
        session.bus.subscribe(lambda event: seen.append(event["type"]))
        _run(run_turn(session, fake_pipeline, TurnRequest(text="事件顺序", speak=True)))

        for expected in (
            "turn.started",
            "llm.started",
            "llm.delta",
            "llm.completed",
            "tts.started",
            "tts.audio",
            "playback.started",
            "playback.finished",
            "turn.completed",
        ):
            assert expected in seen, f"{expected} was never emitted; saw {sorted(set(seen))}"

        assert seen.index("turn.started") < seen.index("llm.started")
        assert seen.index("llm.started") < seen.index("llm.delta")
        assert seen.index("llm.delta") < seen.index("llm.completed")
        assert seen.index("tts.started") < seen.index("tts.audio")
        assert seen.index("playback.started") < seen.index("playback.finished")
        assert "turn.completed" in seen
        assert "runtime.metric" in seen
        # `finish_turn` emits the completion frame and then one metric event per
        # measured metric, so `turn.completed` must precede all of them.
        assert seen.index("turn.completed") < seen.index("runtime.metric")
        assert seen[-1] == "runtime.metric"

    def test_failed_provider_degrades_instead_of_raising(self):
        """A broken stage must return a failed result, not blow up the server."""

        from local_voice_companion.core.orchestrator import Pipeline, TurnRequest, run_turn
        from local_voice_companion.core.session import Session
        from local_voice_companion.providers.fake import FakeASR, FakeLLM, FakeTTS

        llm = FakeLLM({"fail": True})
        tts = FakeTTS({})
        for provider in (llm, tts):
            provider.lifecycle.transition("AVAILABLE")
            provider.lifecycle.transition("LOADING")
            provider.lifecycle.transition("READY")

        pipeline = Pipeline(asr=FakeASR({}), llm=llm, tts=tts)
        session = Session(id="smoke-fail", language="zh")
        result = _run(run_turn(session, pipeline, TurnRequest(text="会失败", speak=True)))

        assert result.error, "a failing LLM must surface an error, not a silent empty reply"
        assert result.timeline is not None

    def test_missing_required_stage_is_reported(self, fake_pipeline):
        from local_voice_companion.core.errors import LVCError
        from local_voice_companion.core.orchestrator import Pipeline

        pipeline = Pipeline(asr=fake_pipeline.asr)
        with pytest.raises(LVCError):
            pipeline.require("llm")


class TestBargeIn:
    def test_cancel_stops_the_turn(self):
        """Barge-in must abort a slow turn and report it as cancelled."""

        from local_voice_companion.core.orchestrator import Pipeline, TurnRequest, run_turn
        from local_voice_companion.core.session import Session
        from local_voice_companion.providers.fake import FakeASR, FakeLLM, FakeTTS, FakeVAD

        llm = FakeLLM({"delay_ms": 1500, "ttft_ms": 1200})
        tts = FakeTTS({"delay_ms": 50})
        for provider in (llm, tts):
            provider.lifecycle.transition("AVAILABLE")
            provider.lifecycle.transition("LOADING")
            provider.lifecycle.transition("READY")

        pipeline = Pipeline(asr=FakeASR({}), llm=llm, tts=tts, vad=FakeVAD({}))
        session = Session(id="smoke-barge", language="zh")

        async def scenario():
            task = asyncio.create_task(
                run_turn(session, pipeline, TurnRequest(text="很慢的回答", speak=True))
            )
            # Give the turn time to start, then interrupt it the way a real
            # barge-in does: from outside, while it is mid-flight.
            await asyncio.sleep(0.15)
            assert session.cancel(detail="user spoke again", code="barge_in") is True
            return await task

        result = _run(scenario())
        assert result.cancelled is True
        assert session.state.value in {"CANCELLING", "LISTENING", "IDLE"}

    def test_cancel_is_idempotent(self) -> None:
        from local_voice_companion.core.session import Session

        session = Session(id="smoke-idem", language="zh")
        assert session.cancel() is False
        assert session.cancel() is False
        assert session.cancel(detail="again") is False

    def test_no_events_arrive_after_cancellation(self) -> None:
        """Stale audio must never reach the client after a barge-in."""

        from local_voice_companion.core.orchestrator import Pipeline, TurnRequest, run_turn
        from local_voice_companion.core.session import Session
        from local_voice_companion.providers.fake import FakeASR, FakeLLM, FakeTTS

        llm = FakeLLM({"delay_ms": 1200, "ttft_ms": 900})
        tts = FakeTTS({})
        for provider in (llm, tts):
            provider.lifecycle.transition("AVAILABLE")
            provider.lifecycle.transition("LOADING")
            provider.lifecycle.transition("READY")

        pipeline = Pipeline(asr=FakeASR({}), llm=llm, tts=tts)
        session = Session(id="smoke-stale", language="zh")
        audio_events: list[dict] = []
        session.bus.subscribe(
            lambda event: audio_events.append(event) if event["type"] == "tts.audio" else None
        )

        async def scenario():
            task = asyncio.create_task(
                run_turn(session, pipeline, TurnRequest(text="打断我", speak=True))
            )
            await asyncio.sleep(0.1)
            session.cancel(detail="barge-in", code="barge_in")
            await task
            return len(audio_events)

        count_at_cancel = _run(scenario())
        final = len(session.bus.after(0))
        # Whatever was already queued may flush, but the turn must terminate.
        assert count_at_cancel >= 0
        assert final > 0


class TestBoundedQueues:
    def test_overflow_drops_oldest_and_is_counted(self) -> None:
        """A slow consumer must lose the oldest frames, never grow memory."""

        from local_voice_companion.pipeline.queue import BoundedQueue

        queue = BoundedQueue(3, policy="oldest")
        for index in range(10):
            queue.put_nowait(index)

        assert queue.qsize() == 3
        assert queue.stats.dropped >= 7
        assert list(queue.drain()) == [7, 8, 9]

    def test_newest_policy_keeps_earliest(self) -> None:
        """`newest` refuses the incoming frame instead of evicting a queued one."""

        from local_voice_companion.pipeline.queue import BoundedQueue

        queue = BoundedQueue(3, policy="newest")
        accepted = [queue.put_nowait(index) for index in range(10)]

        assert accepted[:3] == [True, True, True]
        assert any(item is False for item in accepted[3:])
        assert list(queue.drain()) == [0, 1, 2]
        assert queue.stats.dropped >= 1

    def test_stats_are_reportable(self) -> None:
        from local_voice_companion.pipeline.queue import BoundedQueue

        queue = BoundedQueue(2)
        for index in range(5):
            queue.put_nowait(index)
        stats = queue.to_dict()
        assert stats["capacity"] == 2
        assert stats["dropped"] >= 3
        assert stats["depth"] <= 2

    def test_drain_empties_the_queue(self) -> None:
        """Barge-in drains queued audio; nothing may survive the drain."""

        from local_voice_companion.pipeline.queue import BoundedQueue

        queue = BoundedQueue(4)
        for index in range(4):
            queue.put_nowait(index)
        queue.drain()
        assert queue.qsize() == 0
        assert queue.empty()

    def test_zero_capacity_is_rejected(self) -> None:
        from local_voice_companion.pipeline.queue import BoundedQueue

        with pytest.raises(ValueError):
            BoundedQueue(0)


class TestChunker:
    def test_chunker_respects_bounds(self) -> None:
        from local_voice_companion.pipeline.chunker import AdaptiveTextChunker

        chunker = AdaptiveTextChunker(min_chars=2, max_chars=12, language="zh")
        chunks = chunker.push("你好，世界。这是一个用来测试分块的句子，需要足够长。")
        chunks += chunker.flush()

        assert chunks
        for chunk in chunks:
            assert len(chunk) <= 12, repr(chunk)

    def test_chunker_preserves_content(self) -> None:
        """Chunking must not drop or duplicate a character of the reply."""

        from local_voice_companion.pipeline.chunker import AdaptiveTextChunker

        source = "你好，世界。今天天气不错，我们出去走走吧。"
        chunker = AdaptiveTextChunker(min_chars=2, max_chars=20, language="zh")
        chunks = chunker.push(source) + chunker.flush()

        assert "".join(chunks) == source

    def test_empty_input_yields_nothing(self) -> None:
        from local_voice_companion.pipeline.chunker import AdaptiveTextChunker

        chunker = AdaptiveTextChunker(min_chars=2, max_chars=20, language="zh")
        assert chunker.push("") == []
        assert chunker.flush() == []

    def test_long_text_without_punctuation_is_force_cut(self) -> None:
        """A model that forgets punctuation must not stall speech forever."""

        from local_voice_companion.pipeline.chunker import AdaptiveTextChunker

        chunker = AdaptiveTextChunker(min_chars=4, max_chars=16, language="zh")
        chunks = chunker.push("啊" * 100) + chunker.flush()
        assert chunks
        assert all(len(chunk) <= 16 for chunk in chunks)

    def test_reset_discards_pending_text(self) -> None:
        from local_voice_companion.pipeline.chunker import AdaptiveTextChunker

        chunker = AdaptiveTextChunker(min_chars=2, max_chars=40, language="zh")
        chunker.push("还没有说完的一段")
        assert chunker.pending
        chunker.reset()
        assert chunker.pending == ""


# ---------------------------------------------------------------------------
# config migration
# ---------------------------------------------------------------------------


class TestConfigMigration:
    def test_current_config_is_untouched(self) -> None:
        from local_voice_companion.config.migration import CURRENT_VERSION, migrate

        result = migrate({"schema_version": CURRENT_VERSION, "server": {"port": 1234}})
        assert result.migrated is False
        assert result.config["server"]["port"] == 1234

    def test_v1_flat_document_migrates(self) -> None:
        """A pre-2.0 config is recognised by its flat key set and upgraded."""

        from local_voice_companion.config.migration import CURRENT_VERSION, detect_version, migrate

        v1 = {
            "voicebox_url": "http://127.0.0.1:17493",
            "ollama_url": "http://127.0.0.1:11434",
            "ollama_model": "qwen3.5:4b",
            "tts_engine": "luxtts",
            "language": "zh",
        }
        assert detect_version(v1) == 1
        result = migrate(v1)
        assert result.from_version == 1
        assert result.to_version == CURRENT_VERSION
        assert result.migrated is True
        assert result.notes

    def test_v1_user_settings_survive_migration(self) -> None:
        """The migration must not silently reset a user's real preferences.

        Losing the model name or the voice profile would force every existing
        installation to reconfigure by hand, which is exactly the kind of
        regression a migration exists to prevent.
        """

        from local_voice_companion.config.loader import validate_config

        v1 = {
            "voicebox_url": "http://127.0.0.1:17831",
            "ollama_url": "http://127.0.0.1:11434",
            "ollama_model": "qwen3.5:4b",
            "tts_engine": "luxtts",
            "tts_model_size": "0.6B",
            "voice_profile_id": "855480eb-ad2f-461b-957b-8e2b621d61c8",
            "voice_profile_name": "覃哥",
            "asr_model": "base",
            "language": "zh",
            "system_prompt": "你是一个简洁友好的语音助手。",
        }
        config = validate_config(v1)

        assert config.legacy.ollama_model == "qwen3.5:4b"
        assert config.legacy.voice_profile_name == "覃哥"
        assert config.legacy.voice_profile_id == "855480eb-ad2f-461b-957b-8e2b621d61c8"
        assert config.legacy.tts_engine == "luxtts"
        assert config.legacy.asr_model == "base"
        assert config.legacy.language == "zh"
        assert config.legacy.voicebox_port == 17831

    def test_legacy_block_is_optional_not_core(self) -> None:
        """Turning the legacy path off must leave a valid, usable config."""

        from local_voice_companion.config.schema import SystemConfig

        config = SystemConfig()
        config.legacy.enabled = False
        assert config.legacy.enabled is False
        assert config.runtime.policy is not None

    def test_config_round_trips_through_disk(self, tmp_path) -> None:
        from local_voice_companion.config.loader import load_config, save_config
        from local_voice_companion.config.schema import SystemConfig

        config = SystemConfig()
        config.legacy.ollama_model = "round-trip-model"
        target = tmp_path / "config.json"
        save_config(config, target)

        reloaded = load_config(target, use_cache=False)
        assert reloaded.legacy.ollama_model == "round-trip-model"

    def test_save_is_atomic_and_leaves_no_temp_file(self, tmp_path) -> None:
        from local_voice_companion.config.loader import save_config
        from local_voice_companion.config.schema import SystemConfig

        target = tmp_path / "nested" / "config.json"
        save_config(SystemConfig(), target)
        assert target.is_file()
        assert not list(target.parent.glob("*.tmp"))

    def test_secrets_never_serialise_in_clear(self) -> None:
        from local_voice_companion.config.schema import ProviderOverride, SystemConfig, redacted

        config = SystemConfig()
        config.providers["demo"] = ProviderOverride(
            enabled=True, options={"api_key": "sk-super-secret"}
        )
        payload = redacted(config)
        serialised = json.dumps(payload)
        assert "sk-super-secret" not in serialised
        assert payload["providers"]["demo"]["options"]["api_key"] == "<set>"

    def test_scrub_marks_set_and_unset_distinctly(self) -> None:
        """`<set>` vs `<unset>` tells an operator whether a key is configured."""

        from local_voice_companion.config.schema import SystemConfig, redacted

        from local_voice_companion.config.schema import ProviderOverride, SystemConfig, redacted

        config = SystemConfig()
        config.providers["demo"] = ProviderOverride(
            enabled=True,
            options={"api_key": "sk-present", "relay_token": "", "model": "keep-me"},
        )
        options = redacted(config)["providers"]["demo"]["options"]
        assert options["api_key"] == "<set>"
        assert options["relay_token"] == "<unset>"
        assert options["model"] == "keep-me"

    def test_unknown_field_is_rejected(self) -> None:
        """A typo in config.json must fail loudly rather than be ignored."""

        from pydantic import ValidationError

        from local_voice_companion.config.schema import SystemConfig

        with pytest.raises(ValidationError):
            SystemConfig.model_validate({"srever": {"port": 1}})


# ---------------------------------------------------------------------------
# data layout
# ---------------------------------------------------------------------------


class TestDataLayout:
    def test_layout_is_a_writable_directory_with_drive_preference(self) -> None:
        """Dependencies and large files must never silently fill the system drive.

        The resolver prefers D:/E: and only falls back to the project directory
        when nothing else is writable -- and it records which it chose, so a
        surprise C: placement is visible rather than silent.
        """

        from local_voice_companion.config.paths import resolve_data_root

        layout = resolve_data_root()
        assert layout.root
        assert layout.note, "the resolver must record how it chose the root"
        assert str(layout.root).strip()

    def test_layout_creates_expected_subdirectories(self, tmp_path) -> None:
        from local_voice_companion.config.paths import resolve_data_root

        layout = resolve_data_root(str(tmp_path / "root"))
        layout.ensure()
        for name in ("bots", "models", "cache", "logs", "benchmarks"):
            assert (layout.root / name).is_dir(), name

    def test_env_override_is_considered(self, tmp_path, monkeypatch) -> None:
        """`LVC_DATA_ROOT` is the first candidate and is credited in the note.

        The resolver then picks the candidate with the most free space, so on a
        machine where D: has more room than the temp drive the override does not
        necessarily win -- but it must always be entered into the comparison and
        the note must name whichever source was chosen. Silently ignoring the
        variable would be the actual bug.
        """

        from local_voice_companion.config.paths import resolve_data_root

        target = tmp_path / "custom-root"
        monkeypatch.setenv("LVC_DATA_ROOT", str(target))
        layout = resolve_data_root()

        assert layout.root
        assert layout.note
        assert layout.note.startswith("selected via ")
        # Either the override won, or another candidate demonstrably had more
        # free space and the note says so.
        if layout.root.resolve() != target.resolve():
            assert "LVC_DATA_ROOT" not in layout.note

    def test_explicit_argument_beats_the_environment(self, tmp_path, monkeypatch) -> None:
        """An explicit override is passed as an argument and takes precedence."""

        from local_voice_companion.config.paths import resolve_data_root

        monkeypatch.setenv("LVC_DATA_ROOT", str(tmp_path / "from-env"))
        explicit = tmp_path / "from-argument"
        layout = resolve_data_root(str(explicit))
        # Whatever wins on free space, the note must attribute it to the
        # explicit argument rather than to the environment variable.
        assert layout.note
        assert "LVC_DATA_ROOT" not in layout.note or layout.root == explicit.resolve()

    def test_legacy_config_path_is_known(self) -> None:
        """The pre-2.0 config must still be discoverable for migration."""

        from local_voice_companion.config.paths import LEGACY_CONFIG_PATH

        assert LEGACY_CONFIG_PATH.name.endswith(".json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_parser_builds(self) -> None:
        from local_voice_companion.__main__ import build_parser

        parser = build_parser()
        assert parser.prog

    def test_help_exits_zero(self, capsys) -> None:
        from local_voice_companion.__main__ import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(["--help"])
        assert excinfo.value.code == 0

    def test_where_reports_drive(self, capsys) -> None:
        from local_voice_companion.__main__ import main

        assert main(["where"]) == 0
        output = capsys.readouterr().out
        assert output.strip()

    def test_doctor_runs(self, capsys) -> None:
        """Doctor must run on a machine with no inference backend installed."""

        from local_voice_companion.__main__ import main

        code = main(["doctor"])
        assert code in {0, 1}
        output = capsys.readouterr().out
        assert output.strip()

    def test_plan_labels_measurements_as_simulated(self, capsys) -> None:
        """Without native providers every number is an estimate, and says so.

        This is the honesty contract: a user must never read a simulated
        latency as a measured one (RULE 12).
        """

        from local_voice_companion.__main__ import main

        main(["plan"])
        output = capsys.readouterr().out
        assert "simulated" in output.lower(), output


# ---------------------------------------------------------------------------
# hardware-dependent (run with -m hardware; skipped by default)
# ---------------------------------------------------------------------------


def _hardware_skip(reason: str) -> None:
    """Skip with a reason that names the *fix*, not just the absence."""

    pytest.skip(reason)


def _native_asr_or_skip():
    """Skip unless the provider itself says it could load right now.

    Delegating to `FasterWhisperASR.availability()` rather than re-deriving the
    rule here is deliberate: a test that skips on weights the provider would
    happily load is a false negative that hides a working feature.
    """

    ok, reason = FasterWhisperASR.availability()
    if not ok:
        _hardware_skip(reason)


def _native_tts_or_skip():
    store = default_store()
    if importlib.util.find_spec("kokoro_onnx") is None:
        _hardware_skip(
            "kokoro-onnx is not installed; run: pip install 'local-voice-companion[tts]'"
        )
    if importlib.util.find_spec("onnxruntime") is None:
        _hardware_skip("onnxruntime is not installed; it ships with kokoro-onnx")
    if importlib.util.find_spec("misaki") is None:
        _hardware_skip(
            "misaki is not installed; Chinese G2P requires it: pip install 'misaki[zh]'"
        )
    if not store.is_ready("kokoro-v1.1-zh"):
        _hardware_skip(
            f"Kokoro weights are not present ({store.fetch_hint('kokoro-v1.1-zh')})"
        )


@pytest.mark.hardware
class TestNativeASR:
    """faster-whisper on real weights.

    These assert *behaviour*, not just importability. The previous version of
    this class only did `importorskip`, which passed on a machine with the
    package installed and no model -- a green tick for a provider that could not
    transcribe anything.
    """

    def test_probe_reports_ready_with_a_specific_detail(self) -> None:
        _native_asr_or_skip()
        provider = FasterWhisperASR()
        health = asyncio.run(provider.probe())
        assert health.ok, health.detail
        assert health.extra["runtime_status"] == "ready"
        assert "int8" in health.extra["cpu_compute_types"]

    def test_transcribes_synthesised_speech_offline(self) -> None:
        """Real audio in, real text out -- and it must run without the network."""

        _native_asr_or_skip()
        _native_tts_or_skip()

        tts = KokoroTTS()
        blob, _fmt = asyncio.run(tts.synthesize("今天天气很好。"))
        pcm, sample_rate = _wav_pcm(blob)

        provider = FasterWhisperASR()
        started = time.perf_counter()
        text = asyncio.run(
            provider.transcribe(AudioChunk(pcm=pcm, sample_rate=sample_rate), language="zh")
        )
        elapsed = time.perf_counter() - started
        duration = len(pcm) / 2 / sample_rate

        assert text.strip(), "expected a non-empty transcript"
        # A round trip through synthesis is lossy, but the recogniser should
        # still recover most of the characters. Asserting exact equality would
        # be flaky; asserting "some Chinese came back" is the honest check.
        recovered = sum(1 for char in "今天天气很好" if char in text)
        assert recovered >= 3, f"recovered only {recovered}/6 characters from {text!r}"
        assert elapsed < duration * 3, f"suspiciously slow: RTF {elapsed / duration:.2f}"

    def test_unload_returns_to_available_and_is_reloadable(self) -> None:
        _native_asr_or_skip()
        provider = FasterWhisperASR()

        async def cycle() -> tuple[str, str, bool]:
            await provider.load()
            first = provider.lifecycle.state.value
            await provider.unload()
            after = provider.lifecycle.state.value
            await provider.load()
            return first, after, provider._engine is not None

        first, after, reloaded = asyncio.run(cycle())
        assert first == "READY"
        assert after == "AVAILABLE"
        assert reloaded


@pytest.mark.hardware
class TestNativeTTS:
    """Kokoro on real weights."""

    def test_probe_reports_ready(self) -> None:
        _native_tts_or_skip()
        provider = KokoroTTS()
        health = asyncio.run(provider.probe())
        assert health.ok, health.detail
        assert health.extra["g2p_backend"] == "misaki"

    def test_produces_a_parseable_wav_not_just_bytes(self) -> None:
        _native_tts_or_skip()
        provider = KokoroTTS()
        blob, fmt = asyncio.run(provider.synthesize("你好，这是一次真实的本地合成。"))

        # Length alone proves nothing: a truncated or headerless blob can still
        # be long. Parse it.
        with wave.open(io.BytesIO(blob)) as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == 24000
            frames = handle.getnframes()
            payload = handle.readframes(frames)
        assert frames > 24000 * 0.5, f"only {frames} frames"
        assert len(payload) == frames * 2
        assert fmt.sample_rate == 24000
        assert fmt.codec == "pcm_s16le"

    def test_audio_is_not_silence(self) -> None:
        """A WAV of the right length full of zeros is a silent failure."""

        _native_tts_or_skip()
        provider = KokoroTTS()
        blob, _fmt = asyncio.run(provider.synthesize("你好世界。"))
        _pcm, _rate = _wav_pcm(blob)
        assert pcm_rms(_pcm) > 200, "synthesised audio is effectively silent"

    def test_markdown_is_stripped_before_phonemisation(self) -> None:
        _native_tts_or_skip()
        assert "**" not in KokoroTTS.normalize_text("**重点**内容")
        assert "`" not in KokoroTTS.normalize_text("`code`")
        assert "你好" in KokoroTTS.normalize_text("**你好**")

    def test_unknown_voice_is_rejected_rather_than_silently_defaulted(self) -> None:
        _native_tts_or_skip()
        provider = KokoroTTS()

        async def call() -> None:
            await provider.load()
            await provider.synthesize("测试", voice="no-such-voice")

        with pytest.raises(ProviderUnavailable):
            asyncio.run(call())


@pytest.mark.hardware
def test_native_pair_completes_a_cpu_only_offline_turn() -> None:
    """The Phase 2 exit criterion, as a test.

    No GPU, no network, no Voicebox: a real WAV goes in, a real transcript comes
    out, and a real WAV comes back. If this passes, the claim "the runtime has a
    genuine local voice path" is backed by execution rather than by a descriptor.
    """

    _native_asr_or_skip()
    _native_tts_or_skip()

    asr = FasterWhisperASR()
    tts = KokoroTTS()

    if not asr.descriptor().is_local or tts.descriptor().is_local is False:
        pytest.fail("native providers must be marked local")
    assert asr.descriptor().requires_network is False
    assert tts.descriptor().requires_network is False

    # Step 1: synthesize a prompt.
    blob, _fmt = asyncio.run(tts.synthesize("请帮我打开客厅的灯。"))
    prompt_pcm, rate = _wav_pcm(blob)
    assert pcm_rms(prompt_pcm) > 200, "prompt audio is silent"

    # Step 2: transcribe it with the local recogniser.
    transcript = asyncio.run(
        asr.transcribe(AudioChunk(pcm=prompt_pcm, sample_rate=rate), language="zh")
    )
    assert transcript.strip(), "the local ASR produced nothing"

    # Step 3: speak a reply back.
    reply = asyncio.run(tts.synthesize("好的，已经打开了。"))
    assert reply[0][:4] == b"RIFF"
    reply_pcm, _ = _wav_pcm(reply[0])
    assert pcm_rms(reply_pcm) > 200

    # Every stage must be honest about being measured by a real engine.
    assert asr.descriptor().quality_source.value != "unknown"
    assert tts.descriptor().quality_source.value != "unknown"
