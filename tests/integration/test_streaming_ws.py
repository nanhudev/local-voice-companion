"""Live audio over the WebSocket: partials must arrive while the user talks.

The unit tier proves the orchestrator emits `asr.partial`. This tier proves the
*socket* carries it, which is a different claim: the frame arrives on one
coroutine (the websocket receive loop) and is consumed on another (the turn
task), and the handoff is a bounded queue that drops under backpressure. A bug
in that handoff -- a queue never closed, a turn started too late, a final frame
swallowed -- is invisible to unit tests.

The protocol under test:

    {"type": "audio.start", "sample_rate": 16000}
    {"type": "audio.frame", "audio_base64": "...", "sample_rate": 16000}
    {"type": "audio.end"}

`speak` is false throughout so the assertions stay about transcription.
"""

from __future__ import annotations

import base64
from dataclasses import replace

import pytest

SAMPLE_RATE = 16000
PARTIALS = ["今天", "今天天气", "今天天气怎么样"]
FINAL = "今天天气怎么样"


def _scripted_provider_class():
    """A `ScriptedStreamingASR` whose hypotheses are baked in.

    Provider options travel from the runtime's own configuration, and the app
    hands none to a non-legacy provider. Baking the script into a subclass is
    therefore how the HTTP tier gets a recogniser with known partials, instead
    of adding a test-only configuration knob to the runtime.
    """

    from local_voice_companion.providers.fake import ScriptedStreamingASR

    class IntegrationStreamingASR(ScriptedStreamingASR):
        def __init__(self, options=None) -> None:
            merged = {"partials": list(PARTIALS), "final_transcript": FINAL}
            merged.update(options or {})
            super().__init__(merged)

        @staticmethod
        def descriptor():
            base = ScriptedStreamingASR.descriptor()
            return replace(base, id="itest_streaming_asr")

    return IntegrationStreamingASR


@pytest.fixture
def streaming_app(host_config, tmp_path):
    """An app whose only ASR is the scripted streaming provider.

    ``FakeASR`` is deliberately left out: with both registered, selection could
    pick either, and a test that depends on which one won would fail for reasons
    unrelated to streaming.
    """

    from local_voice_companion.api.app import create_app
    from local_voice_companion.providers.fake import FakeLLM, FakeTTS, FakeVAD
    from local_voice_companion.providers.registry import ProviderRegistry

    registry = ProviderRegistry()
    for provider_class in (_scripted_provider_class(), FakeLLM, FakeTTS, FakeVAD):
        registry.register(provider_class)
    return create_app(
        config=host_config,
        registry=registry,
        bots_dir=tmp_path / "bots",
        register_default_providers=False,
    )


@pytest.fixture
def streaming_client(streaming_app):
    from fastapi.testclient import TestClient

    with TestClient(streaming_app) as client:
        yield client


@pytest.fixture
def streaming_session(streaming_client) -> str:
    created = streaming_client.post(
        "/api/v1/bots", json={"id": "stream-bot", "name": "Stream Bot", "language": "zh"}
    )
    assert created.status_code == 201, created.text
    opened = streaming_client.post("/api/v1/sessions", json={"bot_id": "stream-bot"})
    assert opened.status_code == 201, opened.text
    return opened.json()["session"]["id"]


def _frame_b64(milliseconds: int = 20) -> str:
    from fixtures import make_tone

    return base64.b64encode(make_tone(milliseconds=milliseconds)).decode("ascii")


def _drain(ws, limit: int = 400) -> list[dict]:
    seen: list[dict] = []
    for _ in range(limit):
        frame = ws.receive_json()
        seen.append(frame)
        if frame["type"] == "turn.completed":
            break
    return seen


class TestStreamingAudioProtocol:
    def test_partials_arrive_before_the_final_transcript(self, streaming_client, streaming_session) -> None:
        with streaming_client.websocket_connect(
            f"/api/v1/sessions/{streaming_session}/stream"
        ) as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_json({"type": "audio.start", "sample_rate": SAMPLE_RATE, "speak": False})
            for _ in range(6):
                ws.send_json(
                    {
                        "type": "audio.frame",
                        "audio_base64": _frame_b64(),
                        "sample_rate": SAMPLE_RATE,
                    }
                )
            ws.send_json({"type": "audio.end"})

            seen = _drain(ws)

        types = [frame["type"] for frame in seen]
        assert "asr.partial" in types, types
        assert types.index("asr.partial") < types.index("asr.final"), types
        partials = [frame for frame in seen if frame["type"] == "asr.partial"]
        assert len(partials) == len(PARTIALS), [item["data"] for item in partials]

        final = next(frame for frame in seen if frame["type"] == "asr.final")
        assert final["data"]["text"] == FINAL

    def test_partials_carry_the_stability_split(self, streaming_client, streaming_session) -> None:
        with streaming_client.websocket_connect(
            f"/api/v1/sessions/{streaming_session}/stream"
        ) as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_json({"type": "audio.start", "sample_rate": SAMPLE_RATE, "speak": False})
            for _ in range(6):
                ws.send_json({"type": "audio.frame", "audio_base64": _frame_b64(), "sample_rate": SAMPLE_RATE})
            ws.send_json({"type": "audio.end"})
            seen = _drain(ws)

        for frame in (item for item in seen if item["type"] == "asr.partial"):
            data = frame["data"]
            assert {"text", "committed", "unstable", "is_final"} <= set(data), data
            assert data["is_final"] is False
            assert data["committed"] + data["unstable"] == data["text"], data

    def test_turn_completes_and_leaves_no_open_stream(self, streaming_client, streaming_session) -> None:
        with streaming_client.websocket_connect(
            f"/api/v1/sessions/{streaming_session}/stream"
        ) as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_json({"type": "audio.start", "sample_rate": SAMPLE_RATE, "speak": False})
            for _ in range(3):
                ws.send_json({"type": "audio.frame", "audio_base64": _frame_b64(), "sample_rate": SAMPLE_RATE})
            ws.send_json({"type": "audio.end"})
            seen = _drain(ws)

        assert seen[-1]["type"] == "turn.completed"
        detail = streaming_client.get(f"/api/v1/sessions/{streaming_session}")
        assert detail.status_code == 200
        assert detail.json()["session"]["active_turn"] is None

    def test_frame_before_start_is_reported_not_silently_dropped(
        self, streaming_client, streaming_session
    ) -> None:
        with streaming_client.websocket_connect(
            f"/api/v1/sessions/{streaming_session}/stream"
        ) as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_json({"type": "audio.frame", "audio_base64": _frame_b64(), "sample_rate": SAMPLE_RATE})
            frame = ws.receive_json()

        assert frame["type"] == "error"
        assert "audio.start" in frame["data"]["message"]

    def test_end_without_start_is_reported(self, streaming_client, streaming_session) -> None:
        with streaming_client.websocket_connect(
            f"/api/v1/sessions/{streaming_session}/stream"
        ) as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_json({"type": "audio.end"})
            frame = ws.receive_json()

        assert frame["type"] == "error"
        assert "audio stream" in frame["data"]["message"]
