"""FastAPI application: versioned REST + realtime WebSocket + OpenAI compat.

The API is a thin adapter. It must not contain provider-specific logic; it
validates input, calls the runtime, and serialises events that the runtime
already produced.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from ..bots.schema import BotManifest
from ..bots.store import BotStore, store_summary
from ..config.loader import load_config, migration_report, save_config
from ..config.paths import DEFAULT_LAYOUT
from ..config.schema import SystemConfig, redacted
from ..core.errors import LVCError, NotFound, ValidationFailed
from ..core.events import EVENT_SCHEMA_VERSION, EventType
from ..core.orchestrator import Pipeline, TurnRequest, run_turn
from ..core.session import Session
from ..core.types import API_PREFIX, SCHEMA_VERSION, AudioChunk, ProviderKind, SelectionPolicy
from ..hardware.probe import capability_summary, probe_hardware
from ..observability.metrics import MetricsRegistry
from ..pipeline.wav import wav_bytes
from ..providers.discovery import discover
from ..providers.fake import FAKE_PROVIDERS
from ..providers.legacy import LEGACY_PROVIDERS, legacy_options
from ..providers.base import ensure_ready
from ..providers.registry import ProviderRegistry, get_registry
from ..selection.benchmark import BenchmarkCache
from ..selection.engine import recommend
from ..selection.policies import describe_policy
from ..selection.selector import missing_stages, summarise_plan
from .system import RuntimeState, build_runtime_state


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------


class BotCreateRequest(BaseModel):
    id: str
    name: str
    system_prompt: str = "You are a concise and friendly voice assistant."
    description: str = ""
    language: str = "zh"
    voice: str | None = None
    policy: str | None = None
    template: str = ""
    tags: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)


class BotUpdateRequest(BaseModel):
    name: str | None = None
    system_prompt: str | None = None
    description: str | None = None
    language: str | None = None
    voice: str | None = None
    policy: str | None = None
    tags: list[str] | None = None
    conversation: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    asr: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
    tts: dict[str, Any] | None = None


class SessionCreateRequest(BaseModel):
    bot_id: str


class TextTurnRequest(BaseModel):
    text: str
    speak: bool = True


class SelectionRequest(BaseModel):
    policy: str | None = None
    language: str | None = None
    bot_id: str | None = None


class BenchmarkRequest(BaseModel):
    kinds: list[str] = Field(default_factory=lambda: ["asr", "llm", "tts"])
    force: bool = False


class SpeechRequest(BaseModel):
    model: str = "bot"
    input: str = ""
    voice: str | None = None
    response_format: str = "wav"
    session_id: str | None = None


# ---------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    config: SystemConfig | None = None,
    state: RuntimeState | None = None,
    bots_dir: Path | None = None,
    registry: ProviderRegistry | None = None,
    register_default_providers: bool = True,
) -> FastAPI:
    cfg = config or load_config()
    runtime_state = state or build_runtime_state(cfg, reg=registry or get_registry())
    app = FastAPI(
        title="Local Voice Companion",
        version="2.0.0-phase2",
        description=(
            "Adaptive local voice runtime. Provider-agnostic ASR/LLM/TTS selection "
            "with a stable event protocol for any agent, game or application."
        ),
    )
    app.state.config = cfg
    app.state.runtime = runtime_state
    app.state.sessions = {}
    app.state.metrics = MetricsRegistry(retain=cfg.observability.retain_turns)
    app.state.bots = BotStore(bots_dir or DEFAULT_LAYOUT.bots_dir)

    # Providers are registered into the runtime's own registry, never into the
    # module-level singleton. Registering in the singleton meant an app built
    # with a custom `state.reg` (an isolated registry in tests, or two apps in
    # one process) reported an empty provider list and could not plan a
    # pipeline, even though everything was wired correctly.
    provider_registry = runtime_state.reg
    if register_default_providers:
        for cls in (*FAKE_PROVIDERS, *LEGACY_PROVIDERS):
            if cls.descriptor().id not in provider_registry:
                provider_registry.register(cls, origin="builtin")

    # -- error mapping ------------------------------------------------------

    @app.exception_handler(LVCError)
    async def _lvc_error(_request: Request, exc: LVCError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_wire())

    # -- health -------------------------------------------------------------

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "local-voice-companion",
            "schema_version": SCHEMA_VERSION,
            "event_schema": EVENT_SCHEMA_VERSION,
            "uptime_s": round(time.monotonic() - runtime_state.started_at, 1),
        }

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        ready = runtime_state.ready
        payload = {
            "status": "ready" if ready else ("starting" if not runtime_state.initialized else "degraded"),
            "initialized": runtime_state.initialized,
            "pipeline": runtime_state.pipeline.to_dict() if runtime_state.pipeline else {},
            "missing_stages": runtime_state.missing_stages,
            "detail": runtime_state.readiness_detail,
        }
        return JSONResponse(status_code=200 if ready else 503, content=payload)

    # -- system -------------------------------------------------------------

    @app.get(f"{API_PREFIX}/system/profile")
    async def system_profile(refresh: bool = False) -> dict[str, Any]:
        if refresh or runtime_state.profile is None:
            await runtime_state.refresh_profile()
        profile = runtime_state.profile
        assert profile is not None
        return {
            "profile": profile.to_dict(),
            "capabilities": capability_summary(profile),
            "partial": profile.partial,
            "notes": profile.notes,
        }

    @app.get(f"{API_PREFIX}/system/runtime")
    async def system_runtime() -> dict[str, Any]:
        return runtime_state.to_dict()

    @app.get(f"{API_PREFIX}/system/config")
    async def system_config() -> dict[str, Any]:
        """Redacted host config. Secrets appear only as <set>/<unset>."""

        return {
            "config": redacted(app.state.config),
            "migration": migration_report(DEFAULT_LAYOUT),
            "paths": DEFAULT_LAYOUT.to_dict(),
        }

    @app.put(f"{API_PREFIX}/system/config")
    async def update_system_config(patch: dict[str, Any]) -> dict[str, Any]:
        merged = app.state.config.model_validate(
            _deep_merge(app.state.config.model_dump(mode="json"), patch)
        )
        save_config(merged)
        app.state.config = merged
        runtime_state.config = merged
        return {"config": redacted(merged)}

    @app.get(f"{API_PREFIX}/system/policies")
    async def system_policies() -> dict[str, Any]:
        return {
            "policies": [
                describe_policy(policy)
                for policy in (
                    SelectionPolicy.AUTO,
                    SelectionPolicy.ULTRA_LOW_LATENCY,
                    SelectionPolicy.BALANCED,
                    SelectionPolicy.QUALITY,
                    SelectionPolicy.LOW_MEMORY,
                    SelectionPolicy.CPU_ONLY,
                    SelectionPolicy.MANUAL,
                )
            ],
            "current": app.state.config.runtime.policy.value,
        }

    # -- providers ----------------------------------------------------------

    @app.get(f"{API_PREFIX}/providers")
    async def list_providers(kind: str | None = None, refresh: bool = False) -> dict[str, Any]:
        if refresh or not runtime_state.probe_results:
            await runtime_state.discover()
        provider_kind = ProviderKind(kind) if kind else None
        return {
            "providers": runtime_state.provider_descriptors(provider_kind),
            "probes": [item.to_dict() for item in runtime_state.probe_results],
            "instances": runtime_state.instance_states(),
        }

    @app.get(f"{API_PREFIX}/models")
    async def list_models(kind: str | None = None) -> dict[str, Any]:
        provider_kind = ProviderKind(kind) if kind else None
        return {"models": runtime_state.models(provider_kind)}

    @app.get(f"{API_PREFIX}/voices")
    async def list_voices(language: str | None = None) -> dict[str, Any]:
        return {"voices": runtime_state.voices(language)}

    # -- selection & benchmark ---------------------------------------------

    @app.post(f"{API_PREFIX}/selection/recommend")
    async def selection_recommend(body: SelectionRequest) -> dict[str, Any]:
        if runtime_state.profile is None:
            await runtime_state.refresh_profile()
        profile = runtime_state.profile
        assert profile is not None

        language = body.language or "zh"
        policy = body.policy
        bots = app.state.bots
        if body.bot_id:
            bot = bots.get(body.bot_id)
            language = bot.language.primary
            policy = policy or bot.runtime.policy.value

        # The registry must be passed explicitly: `recommend()` defaults to the
        # module-level singleton, which is empty when the app was built with an
        # isolated registry, and would silently report "no available provider".
        decision = await recommend(
            profile,
            app.state.config,
            reg=runtime_state.reg,
            language=language,
            policy_override=policy,
        )
        return decision.to_dict()

    @app.post(f"{API_PREFIX}/benchmark")
    async def run_benchmark(body: BenchmarkRequest) -> dict[str, Any]:
        """Run selection with a fresh cache.

        Phase 1 ships only simulated measurements, so every result is labelled
        `source=simulated`. Real numbers arrive with the native providers.
        """

        if runtime_state.profile is None:
            await runtime_state.refresh_profile()
        profile = runtime_state.profile
        assert profile is not None
        cache = BenchmarkCache()
        if body.force:
            cache.clear()
        decision = await recommend(
            profile, app.state.config, reg=runtime_state.reg, cache=cache
        )
        sources = {
            item.get("benchmark", {}).get("source")
            for item in (decision.alternatives or {}).values()
            for item in item
            if isinstance(item, dict)
        }
        return {
            "decision": decision.to_dict(include_alternatives=False),
            "measurement_sources": sorted(source for source in sources if source),
            "note": (
                "Phase 1 has no native model providers installed, so all figures are "
                "Simulated estimates derived from provider metadata, not measurements."
            ),
        }

    # -- bots ---------------------------------------------------------------

    @app.get(f"{API_PREFIX}/bots")
    async def list_bots() -> dict[str, Any]:
        return store_summary(app.state.bots)

    @app.post(f"{API_PREFIX}/bots", status_code=201)
    async def create_bot(body: BotCreateRequest) -> dict[str, Any]:
        manifest = BotManifest.new(
            id=body.id,
            name=body.name,
            system_prompt=body.system_prompt,
            language=body.language,
            voice=body.voice,
            policy=body.policy or SelectionPolicy.AUTO.value,
            template=body.template,
            tags=body.tags,
        )
        manifest.persona.description = body.description
        if body.overrides:
            manifest = manifest.merge(body.overrides)
        created = app.state.bots.create(manifest)
        runtime_state.emit(
            EventType.RUNTIME_READY, {"event": "bot.created", "bot_id": created.id}
        )
        return {"bot": created.to_dict()}

    @app.get(f"{API_PREFIX}/bots/{{bot_id}}")
    async def get_bot(bot_id: str) -> dict[str, Any]:
        return {"bot": app.state.bots.get(bot_id).to_dict()}

    @app.put(f"{API_PREFIX}/bots/{{bot_id}}")
    async def update_bot(bot_id: str, body: BotUpdateRequest) -> dict[str, Any]:
        patch = body.model_dump(exclude_none=True)
        if "system_prompt" in patch:
            patch["persona"] = {"system_prompt": patch.pop("system_prompt")}
        if "description" in patch:
            patch.setdefault("persona", {})["description"] = patch.pop("description")
        if "language" in patch:
            patch["language"] = {"primary": patch.pop("language")}
        if "voice" in patch:
            patch["tts"] = {**(patch.get("tts") or {}), "voice": patch.pop("voice")}
        if "policy" in patch:
            patch["runtime"] = {**(patch.get("runtime") or {}), "policy": patch.pop("policy")}
        updated = app.state.bots.update(bot_id, patch)
        return {"bot": updated.to_dict()}

    @app.delete(f"{API_PREFIX}/bots/{{bot_id}}")
    async def delete_bot(bot_id: str) -> dict[str, Any]:
        deleted = app.state.bots.delete(bot_id)
        if not deleted:
            raise NotFound(f"bot not found: {bot_id}", bot_id=bot_id)
        return {"deleted": True, "bot_id": bot_id}

    @app.get(f"{API_PREFIX}/bots/{{bot_id}}/export")
    async def export_bot(bot_id: str, format: str = "yaml") -> Response:
        if format == "json":
            return JSONResponse(content=app.state.bots.get(bot_id).to_dict())
        text = app.state.bots.export_yaml(bot_id)
        return Response(
            content=text,
            media_type="application/x-yaml",
            headers={"Content-Disposition": f'attachment; filename="{bot_id}.yaml"'},
        )

    @app.post(f"{API_PREFIX}/bots/import", status_code=201)
    async def import_bot(body: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
        text = body.get("manifest") if isinstance(body.get("manifest"), str) else json.dumps(body)
        manifest = app.state.bots.import_document(str(text), overwrite=overwrite)
        return {"bot": manifest.to_dict()}

    @app.get(f"{API_PREFIX}/bots/{{bot_id}}/plan")
    async def bot_plan(bot_id: str) -> dict[str, Any]:
        """Which runtime this bot would resolve to on THIS machine."""

        bot = app.state.bots.get(bot_id)
        if runtime_state.profile is None:
            await runtime_state.refresh_profile()
        profile = runtime_state.profile
        assert profile is not None
        decision = await recommend(
            profile,
            app.state.config,
            reg=runtime_state.reg,
            language=bot.language.primary,
            policy_override=bot.runtime.policy.value,
        )
        payload = decision.to_dict(include_alternatives=False)
        payload["bot_id"] = bot.id
        payload["portable"] = True
        return payload

    # -- sessions -----------------------------------------------------------

    @app.post(f"{API_PREFIX}/sessions", status_code=201)
    async def create_session(body: SessionCreateRequest) -> dict[str, Any]:
        bot = app.state.bots.get(body.bot_id)
        session = Session.from_bot(bot)
        app.state.sessions[session.id] = session
        session.emit(
            EventType.SESSION_OPENED,
            {"bot_id": bot.id, "bot_name": bot.name, "stream": session.stream_path},
        )
        return {
            "session": session.to_dict(),
            "stream_path": session.stream_path,
            "protocol": {
                "events": "runtime events, schema v" + str(EVENT_SCHEMA_VERSION),
                "client_messages": [
                    "text",
                    "audio",
                    "cancel",
                    "state",
                    "ping",
                    "close",
                ],
            },
        }

    @app.get(f"{API_PREFIX}/sessions")
    async def list_sessions() -> dict[str, Any]:
        return {"sessions": [item.to_dict() for item in app.state.sessions.values()]}

    @app.get(f"{API_PREFIX}/sessions/{{session_id}}")
    async def get_session(session_id: str) -> dict[str, Any]:
        session = _get_session(app, session_id)
        return {
            "session": session.to_dict(include_history=True),
            "events": session.bus.after(0),
            "metrics": app.state.metrics.summary(),
        }

    @app.delete(f"{API_PREFIX}/sessions/{{session_id}}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        session = _get_session(app, session_id)
        await session.close()
        app.state.sessions.pop(session_id, None)
        return {"closed": True, "session_id": session_id}

    @app.post(f"{API_PREFIX}/sessions/{{session_id}}/turns")
    async def run_text_turn(session_id: str, body: TextTurnRequest) -> dict[str, Any]:
        session = _get_session(app, session_id)
        if session.closed:
            raise ValidationFailed("session is closed", session_id=session_id)
        await _ensure_pipeline(app, runtime_state)

        pipeline = runtime_state.pipeline
        assert pipeline is not None
        bot = app.state.bots.get(session.bot_id) if session.bot_id else None
        request = TurnRequest(
            text=body.text,
            speak=body.speak,
            language=session.language,
            system_prompt=bot.persona.system_prompt if bot else "",
            voice=bot.tts.voice if bot else None,
            max_tokens=bot.conversation.max_tokens if bot else app.state.config.pipeline.max_tokens,
            temperature=bot.conversation.temperature if bot else app.state.config.pipeline.temperature,
            chunk_min_chars=app.state.config.pipeline.chunk_min_chars,
            chunk_max_chars=app.state.config.pipeline.chunk_max_chars,
        )
        session.cancel(detail="superseded by a new turn", code="superseded")
        task = asyncio.create_task(run_turn(session, pipeline, request), name=f"turn-{session.id}")
        session.active_turn = task
        result: Any = None
        try:
            result = await task
        except asyncio.CancelledError:
            raise ValidationFailed("turn cancelled", session_id=session_id)
        finally:
            # `session.active_timeline` is already cleared by `finish_turn()`, so
            # reading it here recorded an empty timeline (turn_id "none",
            # stages {}). The authoritative timeline is the one the result
            # carries back from the turn that actually ran.
            timeline = getattr(result, "timeline", None) or session.active_timeline
            if timeline is not None:
                status = "completed"
                if getattr(result, "cancelled", False):
                    status = "cancelled"
                elif getattr(result, "error", ""):
                    status = "failed"
                app.state.metrics.record(timeline, status)
        return {"result": result.to_dict()}

    @app.post(f"{API_PREFIX}/sessions/{{session_id}}/cancel")
    async def cancel_turn(session_id: str) -> dict[str, Any]:
        session = _get_session(app, session_id)
        cancelled = session.cancel(detail="explicit cancel", code="client_cancel")
        if cancelled and session.active_turn and not session.active_turn.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(asyncio.shield(session.active_turn), timeout=2)
        return {"cancelled": cancelled, "session": session.to_dict()}

    # -- observability ------------------------------------------------------

    @app.get(f"{API_PREFIX}/events")
    async def recent_events(after: int = 0, limit: int = 200) -> dict[str, Any]:
        return {"events": runtime_state.bus.after(after)[: max(1, limit)]}

    @app.get(f"{API_PREFIX}/metrics")
    async def metrics(session_id: str | None = None) -> dict[str, Any]:
        if session_id:
            session = _get_session(app, session_id)
            return {"session_id": session_id, "events": session.bus.after(0)[-50:]}
        return {
            "runtime": app.state.metrics.summary(),
            "recent_turns": app.state.metrics.recent(20),
            "queues": runtime_state.queue_stats(),
        }

    # -- OpenAI compatibility ----------------------------------------------

    @app.post("/v1/audio/speech")
    async def openai_speech(body: SpeechRequest) -> Response:
        """Mirror of OpenAI's speech endpoint so existing tools work unchanged."""

        await _ensure_pipeline(app, runtime_state)
        pipeline = runtime_state.pipeline
        assert pipeline is not None and pipeline.tts is not None
        ensure_ready(pipeline.tts)
        audio, fmt = await pipeline.tts.synthesize(
            body.input, voice=body.voice, language=app.state.config.legacy.language
        )
        media = "audio/wav" if body.response_format in {"wav", "pcm"} else f"audio/{body.response_format}"
        return Response(content=audio, media_type=media)

    @app.post("/v1/audio/transcriptions")
    async def openai_transcriptions(request: Request) -> dict[str, Any]:
        """Accepts a raw audio body or multipart `file` upload."""

        await _ensure_pipeline(app, runtime_state)
        pipeline = runtime_state.pipeline
        assert pipeline is not None and pipeline.asr is not None
        ensure_ready(pipeline.asr)

        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                raise ValidationFailed("multipart request is missing the `file` field")
            blob = await upload.read()  # type: ignore[union-attr]
        else:
            blob = await request.body()
        if not blob:
            raise ValidationFailed("empty audio payload")

        chunk = AudioChunk(pcm=_strip_wav_header(blob), sample_rate=app.state.config.audio.sample_rate)
        text = await pipeline.asr.transcribe(chunk, language=app.state.config.legacy.language)
        return {"text": text}

    # -- realtime websocket -------------------------------------------------

    @app.websocket(f"{API_PREFIX}/sessions/{{session_id}}/stream")
    async def session_stream(websocket: WebSocket, session_id: str) -> None:
        session = app.state.sessions.get(session_id)
        if session is None:
            await websocket.close(code=4404, reason="unknown session")
            return
        await websocket.accept()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)

        def forward(event: dict[str, Any]) -> None:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer: drop the oldest frame rather than the socket.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)

        session.bus.subscribe(forward)
        pump = asyncio.create_task(_pump(websocket, queue), name=f"ws-pump-{session_id}")
        try:
            await websocket.send_json(
                {
                    "v": EVENT_SCHEMA_VERSION,
                    "type": "stream.ready",
                    "session_id": session_id,
                    "data": {"state": session.state.value, "replay": len(session.bus.after(0))},
                }
            )
            while True:
                raw = await websocket.receive_text()
                await _handle_client_message(app, runtime_state, session, raw)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - never leak a stack into the socket
            with contextlib.suppress(Exception):
                await websocket.send_json(
                    {"type": EventType.ERROR, "data": {"message": str(exc)}}
                )
        finally:
            pump.cancel()
            session.bus.unsubscribe(forward)

    return app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _get_session(app: FastAPI, session_id: str) -> Session:
    session = app.state.sessions.get(session_id)
    if session is None:
        raise NotFound(f"session not found: {session_id}", session_id=session_id)
    return session


async def _ensure_pipeline(app: FastAPI, state: RuntimeState) -> None:
    if state.pipeline is not None:
        return
    await state.prepare_pipeline()


async def _pump(websocket: WebSocket, queue: "asyncio.Queue[dict[str, Any]]") -> None:
    while True:
        event = await queue.get()
        await websocket.send_json(event)


async def _handle_client_message(
    app: FastAPI, state: RuntimeState, session: Session, raw: str
) -> None:
    try:
        message = json.loads(raw)
    except json.JSONDecodeError:
        session.emit(EventType.ERROR, {"message": "invalid JSON frame"})
        return
    if not isinstance(message, dict):
        session.emit(EventType.ERROR, {"message": "frame must be a JSON object"})
        return

    kind = str(message.get("type", "")).lower()

    if kind == "ping":
        session.emit(EventType.RUNTIME_METRIC, {"metric": "ping", "value_ms": 0})
        return

    if kind == "cancel":
        session.cancel(detail=str(message.get("detail", "client cancel")), code="client_cancel")
        return

    if kind == "close":
        await session.close()
        return

    if kind in {"text", "audio"}:
        await _ensure_pipeline(app, state)
        pipeline = state.pipeline
        assert pipeline is not None
        if session.active_turn is not None and not session.active_turn.done():
            session.cancel(detail="barge-in: new user input", code="barge_in")

        bot = app.state.bots.get(session.bot_id) if session.bot_id else None
        request = TurnRequest(
            text=str(message.get("text", "")) if kind == "text" else "",
            audio=_decode_audio(message) if kind == "audio" else None,
            speak=bool(message.get("speak", True)),
            language=str(message.get("language") or session.language),
            system_prompt=bot.persona.system_prompt if bot else "",
            voice=bot.tts.voice if bot else None,
            max_tokens=bot.conversation.max_tokens if bot else app.state.config.pipeline.max_tokens,
            temperature=bot.conversation.temperature if bot else app.state.config.pipeline.temperature,
            chunk_min_chars=app.state.config.pipeline.chunk_min_chars,
            chunk_max_chars=app.state.config.pipeline.chunk_max_chars,
        )
        if request.audio is None and not request.text:
            session.emit(EventType.ERROR, {"message": "frame carries neither text nor audio"})
            return
        task = asyncio.create_task(run_turn(session, pipeline, request), name=f"turn-{session.id}")
        session.active_turn = task
        return

    if kind == "state":
        session.emit(
            EventType.TURN_STATE,
            {"state": session.state.value, "detail": "client requested snapshot"},
        )
        return

    session.emit(EventType.ERROR, {"message": f"unsupported client message: {kind}"})


def _decode_audio(message: dict[str, Any]) -> AudioChunk:
    payload = message.get("audio_base64") or message.get("audio") or ""
    try:
        blob = base64.b64decode(str(payload), validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailed(f"audio_base64 is not valid base64: {exc}") from exc
    if blob.startswith(b"RIFF"):
        blob = _strip_wav_header(blob)
    return AudioChunk(
        pcm=blob,
        sample_rate=int(message.get("sample_rate", 16000)),
        channels=int(message.get("channels", 1)),
    )


def _strip_wav_header(blob: bytes) -> bytes:
    """Extract the `data` chunk payload from a RIFF/WAVE container."""

    if not blob.startswith(b"RIFF") or len(blob) < 44:
        return blob
    index = 12
    while index + 8 <= len(blob):
        chunk_id = blob[index : index + 4]
        size = int.from_bytes(blob[index + 4 : index + 8], "little")
        if chunk_id == b"data":
            return blob[index + 8 : index + 8 + size]
        index += 8 + size
    return blob[44:]


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _empty_timeline(session: Session):
    from ..core.events import TurnTimeline

    return TurnTimeline(turn_id="none", session_id=session.id)


def default_app() -> FastAPI:
    """Entry point used by uvicorn when no explicit app is constructed."""

    return create_app()


# Re-export the runtime state builder for callers that build an app manually.
__all__ = ["create_app", "default_app", "RuntimeState", "build_runtime_state"]
