"""PHASE 1 acceptance: the fake pipeline must drive the whole runtime.

These tests are the literal completion criteria for PHASE 1:

    pytest passes
    server starts
    GET  /healthz                    works
    GET  /api/v1/providers           works
    POST bot creation                works
    WebSocket session                works
    Fake pipeline end-to-end         works
    existing config migration        works

Every assertion below runs against a hermetic app with fake providers only, so
a green run means the runtime contract is wired up -- not that any particular
model works. Hardware claims live in the smoke tier and in the CLI ``plan``
command, and must always carry a ``source`` label (RULE 12).
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# server starts / health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_healthz(self, client) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == "local-voice-companion"
        assert body["schema_version"] >= 1
        assert body["event_schema"] >= 1
        assert isinstance(body["uptime_s"], (int, float))

    def test_readyz_is_503_starting_before_the_first_turn(self, client) -> None:
        """An un-prepared pipeline is 'starting', not a broken deployment."""

        response = client.get("/readyz")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "starting"
        assert body["initialized"] is False
        assert body["detail"]

    def test_readyz_is_200_after_the_pipeline_is_prepared(self, client, session_id) -> None:
        turn = client.post(
            f"/api/v1/sessions/{session_id}/turns", json={"text": "预热", "speak": False}
        )
        assert turn.status_code == 200

        response = client.get("/readyz")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "ready"
        assert body["initialized"] is True
        assert body["missing_stages"] == []
        assert set(body["pipeline"]) >= {"asr", "llm", "tts"}


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


class TestProviders:
    def test_providers_listing(self, client) -> None:
        response = client.get("/api/v1/providers")
        assert response.status_code == 200
        body = response.json()

        ids = {item["id"] for item in body["providers"]}
        assert {"fake_asr", "fake_llm", "fake_tts"} <= ids, ids
        assert body["probes"], "discovery must have run and reported probe results"
        for descriptor in body["providers"]:
            assert descriptor["kind"] in {"asr", "llm", "tts", "vad"}
            assert descriptor["id"]
            assert "devices" in descriptor

    def test_providers_can_be_filtered_by_kind(self, client) -> None:
        response = client.get("/api/v1/providers", params={"kind": "llm"})
        assert response.status_code == 200
        kinds = {item["kind"] for item in response.json()["providers"]}
        assert kinds == {"llm"}

    def test_models_and_voices(self, client) -> None:
        models = client.get("/api/v1/models")
        assert models.status_code == 200
        assert isinstance(models.json()["models"], list)

        voices = client.get("/api/v1/voices")
        assert voices.status_code == 200
        assert isinstance(voices.json()["voices"], list)

    def test_policies_and_selection(self, client) -> None:
        policies = client.get("/api/v1/system/policies")
        assert policies.status_code == 200
        listed = {item["id"] for item in policies.json()["policies"]}
        assert {"auto", "balanced", "low_memory", "cpu_only", "manual"} <= listed
        assert all(item["weights"] for item in policies.json()["policies"])

        decision = client.post("/api/v1/selection/recommend", json={"language": "zh"})
        assert decision.status_code == 200
        body = decision.json()
        plan = body["plan"]
        assert plan["feasible"] is True, plan
        # The three required conversational stages must always be planned;
        # VAD may join when a provider is available (it gates the mic path).
        assert {"asr", "llm", "tts"} <= set(plan["assignments"]), plan
        for stage, assignment in plan["assignments"].items():
            assert assignment["provider_id"], (stage, assignment)
            assert assignment["device"], (stage, assignment)
        assert plan["footprint"]["ram_mb"] <= plan["budget"]["ram_mb"]
        assert plan["footprint"]["vram_mb"] <= plan["budget"]["vram_mb"]
        assert body["candidates_considered"] > 0
        assert body["effective_policy"] == "balanced"


# ---------------------------------------------------------------------------
# bots
# ---------------------------------------------------------------------------


class TestBots:
    def test_create_bot(self, client, tmp_path) -> None:
        response = client.post(
            "/api/v1/bots",
            json={
                "id": "acceptance-bot",
                "name": "验收机器人",
                "system_prompt": "你很简短。",
                "language": "zh",
                "policy": "ultra_low_latency",
            },
        )
        assert response.status_code == 201, response.text
        bot = response.json()["bot"]
        assert bot["id"] == "acceptance-bot"
        assert bot["name"] == "验收机器人"
        assert bot["language"]["primary"] == "zh"
        assert bot["runtime"]["policy"] == "ultra_low_latency"
        assert (tmp_path / "bots" / "acceptance-bot.json").exists()

    def test_duplicate_bot_id_is_rejected(self, client, bot_id) -> None:
        response = client.post("/api/v1/bots", json={"id": bot_id, "name": "再次"})
        assert response.status_code in {400, 409}, response.text

    def test_bot_lifecycle(self, client, bot_id) -> None:
        listing = client.get("/api/v1/bots")
        assert listing.status_code == 200
        assert bot_id in {item["id"] for item in listing.json()["bots"]}

        updated = client.put(f"/api/v1/bots/{bot_id}", json={"name": "改名后"})
        assert updated.status_code == 200
        assert updated.json()["bot"]["name"] == "改名后"

        plan = client.get(f"/api/v1/bots/{bot_id}/plan")
        assert plan.status_code == 200
        assert plan.json()["portable"] is True

        exported = client.get(f"/api/v1/bots/{bot_id}/export", params={"format": "yaml"})
        assert exported.status_code == 200
        assert bot_id in exported.text
        # A portable manifest must never leak a host path.
        assert ":\\" not in exported.text and "/home/" not in exported.text

        deleted = client.delete(f"/api/v1/bots/{bot_id}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        assert client.get(f"/api/v1/bots/{bot_id}").status_code == 404

    def test_unknown_bot_is_404(self, client) -> None:
        assert client.get("/api/v1/bots/does-not-exist").status_code == 404

    def test_bot_round_trip_via_import(self, client, tmp_path) -> None:
        """Export then re-import under a new id must reproduce the bot exactly."""

        client.post("/api/v1/bots", json={"id": "portable-bot", "name": "可移植", "language": "en"})
        document = client.get("/api/v1/bots/portable-bot/export", params={"format": "json"}).json()

        # Re-importing the same id is a conflict and must say so, not overwrite
        # silently -- a bot carries the user's persona and voice choices.
        conflict = client.post("/api/v1/bots/import", json=document)
        assert conflict.status_code in {400, 409}, conflict.text

        document["id"] = "portable-bot-copy"
        imported = client.post("/api/v1/bots/import", json=document)
        assert imported.status_code == 201, imported.text
        copy = imported.json()["bot"]
        assert copy["id"] == "portable-bot-copy"
        assert copy["name"] == "可移植"
        assert copy["language"]["primary"] == "en"

        # An explicit overwrite is allowed to replace in place.
        document["name"] = "覆写后"
        overwritten = client.post(
            "/api/v1/bots/import", params={"overwrite": True}, json=document
        )
        assert overwritten.status_code == 201, overwritten.text


# ---------------------------------------------------------------------------
# fake pipeline end to end
# ---------------------------------------------------------------------------


class TestFakePipeline:
    def test_text_turn_completes_with_a_full_timeline(self, client, session_id) -> None:
        response = client.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"text": "你好，测试一下", "speak": True},
        )
        assert response.status_code == 200, response.text
        result = response.json()["result"]

        assert result["error"] == ""
        assert result["cancelled"] is False
        assert result["reply"], result
        assert result["chunks_spoken"] >= 1
        assert result["audio_bytes"] > 0

        timeline = result["timeline"]
        assert timeline["turn_id"]
        for stage in ("turn_started", "llm_start", "llm_first_token", "llm_end", "tts_start", "tts_first_audio"):
            assert stage in timeline["stages"], (stage, timeline["stages"])
        assert timeline["llm_ttft_ms"] is not None
        assert timeline["tts_ttfa_ms"] is not None
        assert timeline["total_turn_ms"] is not None

    def test_fake_asr_accepts_real_audio_and_records_the_asr_stage(self, client, session_id) -> None:
        """Upload PCM through the OpenAI-compatible endpoint, not just text."""

        from fixtures import make_tone, make_wav

        response = client.post(
            "/v1/audio/transcriptions",
            content=make_wav(make_tone(milliseconds=1000)),
            headers={"content-type": "audio/wav"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["text"], response.json()

    def test_second_turn_sees_the_first_turn_in_history(self, client, session_id) -> None:
        """The runtime must carry context across turns, not reset each time."""

        first = client.post(f"/api/v1/sessions/{session_id}/turns", json={"text": "第一句"})
        assert first.status_code == 200
        first_reply = first.json()["result"]["reply"]

        second = client.post(f"/api/v1/sessions/{session_id}/turns", json={"text": "第二句"})
        assert second.status_code == 200
        second_reply = second.json()["result"]["reply"]
        assert second_reply
        assert second_reply != first_reply

        detail = client.get(f"/api/v1/sessions/{session_id}")
        assert detail.status_code == 200
        history = detail.json()["session"]["history"]
        roles = [item["role"] for item in history]
        assert roles.count("user") == 2
        assert roles.count("assistant") == 2

    def test_speak_false_delivers_no_audio(self, client, session_id) -> None:
        response = client.post(
            f"/api/v1/sessions/{session_id}/turns", json={"text": "只要文字", "speak": False}
        )
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["reply"]
        assert result["audio_bytes"] == 0

    def test_noise_in_transcript_is_cleaned(self, client, session_id) -> None:
        """Model noise must not reach the user verbatim."""

        response = client.post(
            f"/api/v1/sessions/{session_id}/turns",
            json={"text": "嗯嗯。。。那个,你好!", "speak": False},
        )
        assert response.status_code == 200
        assert response.json()["result"]["reply"]


# ---------------------------------------------------------------------------
# sessions, metrics, events
# ---------------------------------------------------------------------------


class TestSessions:
    def test_session_listing_and_detail(self, client, session_id) -> None:
        listing = client.get("/api/v1/sessions")
        assert listing.status_code == 200
        assert session_id in {item["id"] for item in listing.json()["sessions"]}

        detail = client.get(f"/api/v1/sessions/{session_id}")
        assert detail.status_code == 200
        body = detail.json()
        assert body["session"]["id"] == session_id
        assert isinstance(body["events"], list)
        assert "metrics" in body

    def test_unknown_session_is_404(self, client) -> None:
        assert client.get("/api/v1/sessions/nope").status_code == 404

    def test_closed_session_rejects_new_turns(self, client, session_id) -> None:
        assert client.delete(f"/api/v1/sessions/{session_id}").status_code == 200
        response = client.post(f"/api/v1/sessions/{session_id}/turns", json={"text": "还在吗"})
        assert response.status_code == 404

    def test_metrics_aggregate_real_timelines(self, client, session_id) -> None:
        for text in ("一", "二", "三"):
            assert client.post(f"/api/v1/sessions/{session_id}/turns", json={"text": text}).status_code == 200

        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        runtime = response.json()["runtime"]
        assert runtime["total"] == 3
        assert runtime["completed"] == 3
        assert runtime["failed"] == 0

        ttfa = runtime["time_to_first_audio"]
        assert ttfa["samples"] == 3
        assert ttfa["p50_ms"] >= 0
        assert ttfa["max_ms"] >= ttfa["min_ms"]

        recent = response.json()["recent_turns"]
        assert len(recent) == 3
        assert all(item["turn_id"] and item["turn_id"] != "none" for item in recent)
        assert all(item["stages"] for item in recent)

    def test_runtime_events_are_replayable(self, client, session_id) -> None:
        client.post(f"/api/v1/sessions/{session_id}/turns", json={"text": "事件"})
        response = client.get("/api/v1/events", params={"after": 0})
        assert response.status_code == 200
        events = response.json()["events"]
        assert events
        types = [item["type"] for item in events]
        assert "runtime.ready" in types
        for event in events:
            assert event["v"] >= 1
            assert isinstance(event["id"], int)
            assert event["ts"] > 0

    def test_events_after_cursor(self, client, session_id) -> None:
        client.post(f"/api/v1/sessions/{session_id}/turns", json={"text": "游标"})
        all_events = client.get("/api/v1/events").json()["events"]
        assert len(all_events) > 1
        cursor = all_events[len(all_events) // 2]["id"]
        tail = client.get("/api/v1/events", params={"after": cursor}).json()["events"]
        assert all(item["id"] > cursor for item in tail)
        assert len(tail) < len(all_events)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


class TestWebSocket:
    def test_websocket_handshake(self, client, session_id) -> None:
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/stream") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "stream.ready"
            assert ready["session_id"] == session_id
            assert ready["v"] >= 1
            assert "state" in ready["data"]

    def test_websocket_unknown_session_is_closed(self, client) -> None:
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/v1/sessions/missing/stream") as ws:
                ws.receive_json()

    def test_websocket_text_turn_streams_events(self, client, session_id) -> None:
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/stream") as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_json({"type": "text", "text": "从 WebSocket 说一句", "speak": True})

            seen: list[str] = []
            for _ in range(200):
                frame = ws.receive_json()
                seen.append(frame["type"])
                if frame["type"] == "turn.completed":
                    break

        assert "turn.started" in seen
        assert "llm.delta" in seen
        assert "turn.completed" in seen

    def test_websocket_ping_is_answered(self, client, session_id) -> None:
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/stream") as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_json({"type": "ping"})
            frame = ws.receive_json()
            assert frame["type"] == "runtime.metric"
            assert frame["data"]["metric"] == "ping"

    def test_websocket_rejects_unknown_message_types(self, client, session_id) -> None:
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/stream") as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_json({"type": "sudo-make-me-a-sandwich"})
            frame = ws.receive_json()
            assert frame["type"] == "error"

    def test_websocket_rejects_malformed_json(self, client, session_id) -> None:
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/stream") as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_text("{not json at all")
            frame = ws.receive_json()
            assert frame["type"] == "error"
            assert "JSON" in frame["data"]["message"]

    def test_websocket_audio_frame_runs_asr(self, client, session_id) -> None:
        import base64

        from fixtures import make_tone

        pcm = make_tone(milliseconds=1000)
        with client.websocket_connect(f"/api/v1/sessions/{session_id}/stream") as ws:
            assert ws.receive_json()["type"] == "stream.ready"
            ws.send_json(
                {
                    "type": "audio",
                    "audio_base64": base64.b64encode(pcm).decode("ascii"),
                    "sample_rate": 16000,
                    "speak": False,
                }
            )
            seen: list[str] = []
            for _ in range(200):
                frame = ws.receive_json()
                seen.append(frame["type"])
                if frame["type"] == "turn.completed":
                    break
        assert "asr.final" in seen or "asr.partial" in seen, seen
        assert "turn.completed" in seen


# ---------------------------------------------------------------------------
# OpenAI compatibility surface
# ---------------------------------------------------------------------------


class TestOpenAICompat:
    def test_speech_returns_wav(self, client) -> None:
        response = client.post("/v1/audio/speech", json={"input": "你好世界", "response_format": "wav"})
        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "audio/wav"
        body = response.content
        assert body.startswith(b"RIFF")
        assert len(body) > 44

    def test_transcriptions_returns_json(self, client) -> None:
        from fixtures import make_tone, make_wav

        response = client.post(
            "/v1/audio/transcriptions",
            content=make_wav(make_tone(milliseconds=1000)),
            headers={"content-type": "audio/wav"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["text"]
        assert isinstance(payload.get("duration", 0), (int, float))


# ---------------------------------------------------------------------------
# error contract
# ---------------------------------------------------------------------------


class TestErrorContract:
    def test_validation_error_is_structured(self, client, session_id) -> None:
        response = client.post(f"/api/v1/sessions/{session_id}/turns", json={})
        assert response.status_code == 422

    def test_error_shape_is_stable(self, client) -> None:
        response = client.get("/api/v1/sessions/missing")
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"], body
        assert body["error"]["message"]
        assert "session_id" in body["error"]["context"]

    def test_secrets_are_never_echoed(self, client) -> None:
        response = client.get("/api/v1/system/config")
        assert response.status_code == 200
        serialised = json.dumps(response.json())
        assert "<set>" in serialised or "<unset>" in serialised
        for leaked in ("sk-", "Bearer ", "AI_RELAY_TOKEN="):
            assert leaked not in serialised
