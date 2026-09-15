"""Barge-in over the WebSocket: speech during playback stops playback.

The unit tier proves the watcher fires. This tier proves the *socket* can be
interrupted, which is a different claim and the one that actually matters:

* the frames arrive on one coroutine (the websocket receive loop) while the
  turn runs on another, so a frame sent during SPEAKING has to find its way to
  the detector without an utterance stream to belong to;
* `playback.stopped` has to reach the client, because the client is holding an
  audio device that will keep talking until told otherwise;
* the turn has to end, not hang.

Everything here is deterministic: a slow LLM gives enough time to interrupt, and
a scripted VAD decides when the user "starts talking".
"""

from __future__ import annotations

import asyncio
import base64
import struct
from dataclasses import replace

import pytest

SAMPLE_RATE = 16000


def _tone(milliseconds: int = 20, amplitude: int = 4000) -> bytes:
    """A frame loud enough that the energy-threshold VAD calls it speech."""

    count = int(SAMPLE_RATE * milliseconds / 1000)
    return b"".join(struct.pack("<h", amplitude) for _ in range(count))


def _frame_b64(milliseconds: int = 20) -> str:
    return base64.b64encode(_tone(milliseconds)).decode("ascii")


def _loud_vad_class():
    """A VAD that hears speech in anything above silence.

    `FakeVAD` already does this by energy, so it is reused rather than
    reinvented -- what is needed here is a response long enough to interrupt.
    """

    from local_voice_companion.providers.fake import FakeVAD

    class LoudFakeVAD(FakeVAD):
        def __init__(self, options=None) -> None:
            super().__init__({"threshold_rms": 50, **(options or {})})

        @staticmethod
        def descriptor():
            return replace(FakeVAD.descriptor(), id="itest_vad")

    return LoudFakeVAD


def _slow_llm_class():
    """An LLM that takes long enough for a test to interrupt mid-playback."""

    from local_voice_companion.providers.fake import FakeLLM

    class SlowFakeLLM(FakeLLM):
        def __init__(self, options=None) -> None:
            super().__init__({"ttft_ms": 5, "delay_ms": 25, **(options or {})})

        @staticmethod
        def descriptor():
            return replace(FakeLLM.descriptor(), id="itest_slow_llm")

    return SlowFakeLLM


@pytest.fixture
def duplex_app(host_config, tmp_path):
    """An app wired for speaking turns that can be interrupted."""

    from local_voice_companion.api.app import create_app
    from local_voice_companion.providers.fake import FakeASR, FakeTTS
    from local_voice_companion.providers.registry import ProviderRegistry

    registry = ProviderRegistry()
    for provider_class in (FakeASR, _slow_llm_class(), FakeTTS, _loud_vad_class()):
        registry.register(provider_class)
    config = host_config
    # Short enough that a test can send the frames and still assert promptly.
    config.audio.barge_in_min_speech_ms = 100
    config.audio.silence_after_playback_ms = 0
    return create_app(
        config=config,
        registry=registry,
        bots_dir=tmp_path / "bots",
        register_default_providers=False,
    )


@pytest.fixture
def duplex_client(duplex_app):
    from fastapi.testclient import TestClient

    with TestClient(duplex_app) as client:
        yield client


@pytest.fixture
def duplex_session(duplex_client) -> str:
    created = duplex_client.post(
        "/api/v1/bots", json={"id": "duplex-bot", "name": "Duplex Bot", "language": "zh"}
    )
    assert created.status_code == 201, created.text
    opened = duplex_client.post("/api/v1/sessions", json={"bot_id": "duplex-bot"})
    assert opened.status_code == 201, opened.text
    return opened.json()["session"]["id"]


def _wait_for(ws, event_type: str, limit: int = 300) -> dict | None:
    """Read frames until `event_type` shows up, keeping everything seen."""

    seen: list[dict] = []
    for _ in range(limit):
        frame = ws.receive_json()
        seen.append(frame)
        if frame["type"] == event_type:
            frame["_seen"] = seen  # type: ignore[assignment]
            return frame
        if frame["type"] == "turn.completed":
            frame["_seen"] = seen  # type: ignore[assignment]
            return frame
    return {"_seen": seen}  # type: ignore[return-value]


class TestBargeInOverWebSocket:
    def test_speech_during_playback_stops_it(self, duplex_client, duplex_session) -> None:
        with duplex_client.websocket_connect(
            f"/api/v1/sessions/{duplex_session}/stream"
        ) as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_json({"type": "text", "text": "请说一段很长的话", "speak": True})

            reached_playback = _wait_for(ws, "playback.started")
            assert reached_playback["type"] == "playback.started"

            # Six 20 ms frames = 120 ms of speech, just past the configured
            # 100 ms threshold.
            for _ in range(6):
                ws.send_json(
                    {
                        "type": "audio.frame",
                        "audio_base64": _frame_b64(),
                        "sample_rate": SAMPLE_RATE,
                    }
                )
            stopped = _wait_for(ws, "playback.stopped")
            seen = stopped.pop("_seen")

        types = [frame["type"] for frame in seen] + [stopped["type"]]
        assert "bargein.detected" in types, types
        assert types.index("bargein.detected") < types.index("playback.stopped"), types
        assert stopped["type"] == "playback.stopped", stopped
        assert stopped["data"]["reason"] == "barge_in", stopped

    def test_no_audio_is_delivered_after_the_stop(self, duplex_client, duplex_session) -> None:
        with duplex_client.websocket_connect(
            f"/api/v1/sessions/{duplex_session}/stream"
        ) as ws:
            ws.receive_json()
            ws.send_json({"type": "text", "text": "请继续说下去", "speak": True})
            assert _wait_for(ws, "playback.started")["type"] == "playback.started"
            for _ in range(6):
                ws.send_json(
                    {
                        "type": "audio.frame",
                        "audio_base64": _frame_b64(),
                        "sample_rate": SAMPLE_RATE,
                    }
                )
            stopped = _wait_for(ws, "playback.stopped")
            seen = stopped.pop("_seen")
            tail = _wait_for(ws, "turn.completed")
            seen.extend(tail.pop("_seen"))

        types = [frame["type"] for frame in seen]
        cut = types.index("playback.stopped")
        assert "tts.audio" not in types[cut:], "stale audio leaked past the stop"
        assert "playback.finished" not in types
        assert types[-1] == "turn.completed"

    def test_the_interrupted_turn_reports_a_barge_in_latency(
        self, duplex_client, duplex_session
    ) -> None:
        with duplex_client.websocket_connect(
            f"/api/v1/sessions/{duplex_session}/stream"
        ) as ws:
            ws.receive_json()
            ws.send_json({"type": "text", "text": "再说一点", "speak": True})
            assert _wait_for(ws, "playback.started")["type"] == "playback.started"
            for _ in range(6):
                ws.send_json(
                    {
                        "type": "audio.frame",
                        "audio_base64": _frame_b64(),
                        "sample_rate": SAMPLE_RATE,
                    }
                )
            _wait_for(ws, "playback.stopped")
            completed = _wait_for(ws, "turn.completed")

        timeline = completed["data"]["timeline"]
        assert timeline["stages"].get("bargein_detected") is not None
        assert timeline["stages"].get("playback_stopped") is not None
        assert timeline["stages"].get("playback_end") is None
        assert timeline["barge_in_latency_ms"] is not None
        assert timeline["barge_in_latency_ms"] >= 0

    def test_the_session_accepts_a_new_turn_after_the_interruption(
        self, duplex_client, duplex_session
    ) -> None:
        with duplex_client.websocket_connect(
            f"/api/v1/sessions/{duplex_session}/stream"
        ) as ws:
            ws.receive_json()
            ws.send_json({"type": "text", "text": "第一句", "speak": True})
            assert _wait_for(ws, "playback.started")["type"] == "playback.started"
            for _ in range(6):
                ws.send_json(
                    {
                        "type": "audio.frame",
                        "audio_base64": _frame_b64(),
                        "sample_rate": SAMPLE_RATE,
                    }
                )
            _wait_for(ws, "playback.stopped")
            _wait_for(ws, "turn.completed")

            # The replacement turn must run to completion on a timeline of its
            # own, not inherit the interrupted one's.
            ws.send_json({"type": "text", "text": "第二句", "speak": False})
            completed = _wait_for(ws, "turn.completed")

        assert completed["data"]["status"] == "completed", completed
        assert completed["data"]["timeline"]["barge_in_latency_ms"] is None

    def test_quiet_frames_do_not_interrupt(self, duplex_client, duplex_session) -> None:
        with duplex_client.websocket_connect(
            f"/api/v1/sessions/{duplex_session}/stream"
        ) as ws:
            ws.receive_json()
            ws.send_json({"type": "text", "text": "安静地讲完", "speak": True})
            assert _wait_for(ws, "playback.started")["type"] == "playback.started"
            # A single frame: below the threshold, so nothing should stop.
            ws.send_json(
                {
                    "type": "audio.frame",
                    "audio_base64": base64.b64encode(b"\x00\x00" * 320).decode("ascii"),
                    "sample_rate": SAMPLE_RATE,
                }
            )
            outcome = _wait_for(ws, "playback.stopped")
            seen = outcome.pop("_seen")

        assert "bargein.detected" not in [frame["type"] for frame in seen]

    def test_frames_are_still_rejected_when_nothing_is_listening(
        self, duplex_client, duplex_session
    ) -> None:
        """The old contract survives: no stream, no turn, no silent buffering."""

        with duplex_client.websocket_connect(
            f"/api/v1/sessions/{duplex_session}/stream"
        ) as ws:
            ws.receive_json()
            ws.send_json(
                {
                    "type": "audio.frame",
                    "audio_base64": _frame_b64(),
                    "sample_rate": SAMPLE_RATE,
                }
            )
            frame = ws.receive_json()

        assert frame["type"] == "error"
        assert "audio.start" in frame["data"]["message"]
