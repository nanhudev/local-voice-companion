"""Legacy Voicebox + Ollama service kept alive behind the new architecture.

This is a *thin* compatibility layer, not a second implementation. It reuses
the same provider classes the new runtime uses (`VoiceboxASR`, `VoiceboxTTS`,
`OllamaLLM`) and the legacy config adapter, so the old 4-endpoint HTTP surface
(`/health`, `/events`, `/control`, `/options`, `/settings`, `/text`) keeps
working for existing browser UIs, Godot samples and the Windows relay worker.

What changed relative to the pre-2.0 monolith:
    * no duplicated HTTP calls to Voicebox/Ollama
    * no provider-specific branching outside the provider classes
    * config handling goes through the typed schema (v1 -> v2 migration)
    * state names come from the shared turn state machine
"""

from __future__ import annotations

import io
import json
import queue
import threading
import time
import wave
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..config.paths import LEGACY_WEB_DIR, PROJECT_ROOT
from ..core.state import TurnState
from ..core.types import AudioChunk, ChatMessage
from ..pipeline.wav import clean_model_text, ready_sentences, wav_bytes
from ..providers.legacy import OllamaLLM, VoiceboxASR, VoiceboxTTS
from .legacy_config import LegacyConfigAdapter

#: Map the new canonical states onto the strings the old web UI renders.
_STATE_ALIASES: dict[str, str] = {
    TurnState.IDLE.value: "paused",
    TurnState.LISTENING.value: "listening",
    TurnState.CAPTURING.value: "hearing",
    TurnState.TRANSCRIBING.value: "transcribing",
    TurnState.THINKING.value: "thinking",
    TurnState.SYNTHESIZING.value: "speaking",
    TurnState.SPEAKING.value: "speaking",
    TurnState.CANCELLING.value: "listening",
    TurnState.PAUSED.value: "paused",
    TurnState.ERROR.value: "error",
}


def _legacy_state(state: str) -> str:
    return _STATE_ALIASES.get(state, state.lower())


def _run(coro):
    """Run a coroutine from a synchronous legacy thread."""

    import asyncio

    return asyncio.run(coro)


SSENTENCE_TERMINATORS = "。！？!?；;\n"


class LegacyVoiceService:
    """Synchronous facade over the async provider classes."""

    def __init__(self, adapter: LegacyConfigAdapter | None = None) -> None:
        self.adapter = adapter or LegacyConfigAdapter()
        self.config: dict[str, Any] = self.adapter.as_flat()
        self._typed = None

        self.enabled = True
        self.state = TurnState.IDLE.value
        self.last_error = ""
        self.events: deque[dict[str, Any]] = deque(maxlen=240)
        self.event_id = 0
        self.lock = threading.Lock()
        self.turn_lock = threading.Lock()
        self.turn_id = 0
        self.speaking = False
        self.history: list[dict[str, str]] = []
        self.running = True
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self.tts_queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self.playback_finished_at = 0.0

        self._asr: VoiceboxASR | None = None
        self._tts: VoiceboxTTS | None = None
        self._llm: OllamaLLM | None = None

    # -- providers ----------------------------------------------------------

    def _provider_options(self, provider_id: str) -> dict[str, Any]:
        cfg = self.config
        base = {
            "language": cfg.get("language", "zh"),
        }
        if provider_id == "voicebox_asr":
            return {**base, "base_url": cfg["voicebox_url"], "asr_model": cfg["asr_model"]}
        if provider_id == "voicebox_tts":
            return {
                **base,
                "base_url": cfg["voicebox_url"],
                "tts_engine": cfg["tts_engine"],
                "tts_model_size": cfg["tts_model_size"],
                "voice_profile_id": cfg["voice_profile_id"],
            }
        if provider_id == "ollama_llm":
            return {**base, "base_url": cfg["ollama_url"], "ollama_model": cfg["ollama_model"]}
        raise KeyError(provider_id)

    def asr(self) -> VoiceboxASR:
        if self._asr is None:
            self._asr = VoiceboxASR(self._provider_options("voicebox_asr"))
            _run(self._asr.load())
        return self._asr

    def tts(self) -> VoiceboxTTS:
        if self._tts is None:
            self._tts = VoiceboxTTS(self._provider_options("voicebox_tts"))
            _run(self._tts.load())
        return self._tts

    def llm(self) -> OllamaLLM:
        if self._llm is None:
            self._llm = OllamaLLM(self._provider_options("ollama_llm"))
            _run(self._llm.load())
        return self._llm

    # -- events -------------------------------------------------------------

    def emit(
        self,
        event_type: str,
        text: str = "",
        state: str = "",
        detail: str = "",
        elapsed_ms: int = 0,
    ) -> None:
        with self.lock:
            self.event_id += 1
            self.events.append(
                {
                    "id": self.event_id,
                    "type": event_type,
                    "text": text,
                    "state": _legacy_state(state) if state else "",
                    "detail": detail,
                    "elapsed_ms": elapsed_ms,
                }
            )
            if state:
                self.state = state

    def events_after(self, after: int) -> list[dict[str, Any]]:
        with self.lock:
            return [event for event in self.events if event["id"] > after]

    def health(self) -> dict[str, Any]:
        return {
            "ok": not bool(self.last_error),
            "enabled": self.enabled,
            "state": _legacy_state(self.state),
            "error": self.last_error,
            "ollama_model": self.config["ollama_model"],
            "voice": self.config["voice_profile_name"],
            "tts": self.config["tts_engine"],
            "asr": self.config["asr_model"],
            "asr_backend": self.config.get("asr_backend", "local"),
            "asr_worker": self.config.get("asr_worker_id", ""),
            "event_id": self.event_id,
            "compat_layer": "2.0",
        }

    # -- upstream probing ---------------------------------------------------

    def probe_upstreams(self) -> dict[str, Any]:
        asr_health = _run(self.asr().probe())
        llm_health = _run(self.llm().probe())
        if not asr_health.ok:
            raise RuntimeError(f"voicebox unavailable: {asr_health.detail}")
        if not llm_health.ok:
            raise RuntimeError(f"ollama unavailable: {llm_health.detail}")
        return {
            "voicebox": asr_health.ok,
            "ollama": llm_health.ok,
            "model": self.config["ollama_model"],
        }

    def options(self) -> dict[str, Any]:
        from ..providers.base import ensure_ready

        asr = self.asr()
        llm = self.llm()
        tts = self.tts()
        ensure_ready(asr)
        ensure_ready(llm)
        ensure_ready(tts)

        models = _run(llm.list_models())
        profiles = _run(tts.list_voice_refs())
        return {
            "current": {
                "voice_profile_id": self.config["voice_profile_id"],
                "ollama_model": self.config["ollama_model"],
                "asr_model": self.config["asr_model"],
                "tts_engine": self.config["tts_engine"],
                "tts_model_size": self.config["tts_model_size"],
                "system_prompt": self.config.get("system_prompt", ""),
            },
            "profiles": [{"id": item.id, "name": item.display_name} for item in profiles],
            "ollama_models": [{"id": item.id, "name": item.display_name} for item in models],
            "asr_models": [
                {"id": item.id, "name": item.display_name}
                for item in asr.descriptor().models
            ],
            "asr_workers": [],
            "tts_engines": [
                {"id": "luxtts", "name": "LuxTTS（最快）", "sizes": []},
                {"id": "qwen", "name": "Qwen3 TTS（中文质量）", "sizes": ["0.6B", "1.7B"]},
            ],
        }

    def apply_settings(self, requested: dict[str, Any]) -> dict[str, Any]:
        if self.turn_lock.locked() or self.speaking:
            raise RuntimeError("请等当前回答结束后再切换模型")
        prompt = str(requested.get("system_prompt", self.config.get("system_prompt", ""))).strip()
        if len(prompt) > 2000:
            raise ValueError("提示词不能超过 2000 个字符")
        patch = {
            key: requested[key]
            for key in (
                "voice_profile_id",
                "voice_profile_name",
                "ollama_model",
                "asr_model",
                "tts_engine",
                "tts_model_size",
                "language",
            )
            if key in requested
        }
        patch["system_prompt"] = prompt or "你是一个自然、友好的中文聊天伙伴。请简短、直接地回答用户。"
        updated = self.adapter.persist_flat_patch(patch)
        self.config = self.adapter.as_flat()
        for provider_attr in ("_asr", "_tts", "_llm"):
            setattr(self, provider_attr, None)
        self.history.clear()
        self.emit("settings", text="模型设置已保存", state=TurnState.LISTENING.value)
        return self.health()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        try:
            self.probe_upstreams()
        except Exception as exc:  # noqa: BLE001 - reported through events
            self.last_error = str(exc)
            self.emit("error", detail=self.last_error, state=TurnState.ERROR.value)
            return
        self._warmup()
        threading.Thread(target=self._tts_worker, name="voice-tts", daemon=True).start()
        threading.Thread(target=self._microphone_worker, name="voice-mic", daemon=True).start()
        self.emit("status", text="语音机器人已就绪", state=TurnState.LISTENING.value)

    def _warmup(self) -> None:
        started = time.monotonic()
        self.emit("status", text="正在暖机", state=TurnState.THINKING.value)
        try:
            silence = wav_bytes(
                b"\0\0" * (int(self.config["sample_rate"]) // 3), int(self.config["sample_rate"])
            )
            asr = self.asr()
            chunk = AudioChunk(pcm=silence[44:], sample_rate=int(self.config["sample_rate"]))
            _run(asr.transcribe(chunk, language=self.config["language"]))
        except Exception as exc:  # noqa: BLE001
            self.emit("audio_warning", detail=f"asr warmup: {exc}")
        try:
            _run(self.llm().warm(self.config["ollama_model"]))
        except Exception as exc:  # noqa: BLE001
            self.emit("audio_warning", detail=f"llm warmup: {exc}")
        try:
            tts = self.tts()
            _run(
                tts.synthesize(
                    "准备好了。",
                    voice=self.config["voice_profile_id"],
                    language=self.config["language"],
                    model=self.config["tts_engine"],
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.emit("audio_warning", detail=f"tts warmup: {exc}")
        self.emit(
            "status",
            text="暖机完成",
            state=TurnState.LISTENING.value,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.emit(
            "status",
            text="正在聆听" if enabled else "语音已暂停",
            state=TurnState.LISTENING.value if enabled else TurnState.PAUSED.value,
        )

    def submit_text(self, text: str) -> bool:
        text = text.strip()
        if not text or self.speaking or self.turn_lock.locked():
            return False
        threading.Thread(target=self._conversation, args=(text,), name="voice-turn", daemon=True).start()
        return True

    # -- audio capture ------------------------------------------------------

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            self.emit("audio_warning", detail=str(status))
        try:
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _microphone_worker(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:  # noqa: BLE001 - headless hosts have no audio
            self.emit("audio_warning", detail=f"audio unavailable: {exc}")
            return

        from ..providers.fake import pcm_rms

        rate = int(self.config["sample_rate"])
        block_ms = int(self.config["block_ms"])
        block_frames = rate * block_ms // 1000
        pre_blocks = max(1, int(self.config["pre_roll_ms"]) // block_ms)
        end_blocks = max(1, int(self.config["end_silence_ms"]) // block_ms)
        min_blocks = max(1, int(self.config["min_speech_ms"]) // block_ms)
        max_blocks = max(1, int(float(self.config["max_speech_seconds"]) * 1000) // block_ms)
        pre: deque[bytes] = deque(maxlen=pre_blocks)
        speech: list[bytes] = []
        silent = 0
        hot = 0
        noise_rms = 100.0

        try:
            with sd.RawInputStream(
                samplerate=rate,
                blocksize=block_frames,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            ):
                while self.running:
                    block = self.audio_queue.get()
                    busy = (
                        not self.enabled
                        or self.speaking
                        or self.turn_lock.locked()
                        or time.monotonic() - self.playback_finished_at
                        < self.config["silence_after_playback_ms"] / 1000
                    )
                    if busy:
                        pre.clear()
                        speech.clear()
                        silent = 0
                        hot = 0
                        continue
                    rms = pcm_rms(block)
                    threshold = max(float(self.config["minimum_rms"]), noise_rms * 2.8)
                    if not speech:
                        pre.append(block)
                        if rms > threshold:
                            hot += 1
                        else:
                            hot = 0
                            noise_rms = noise_rms * 0.96 + rms * 0.04
                        if hot >= 2:
                            speech = list(pre)
                            self.emit("status", text="听到了，请继续说", state=TurnState.CAPTURING.value)
                    else:
                        speech.append(block)
                        silent = silent + 1 if rms < threshold * 0.78 else 0
                        if (silent >= end_blocks and len(speech) >= min_blocks) or len(speech) >= max_blocks:
                            usable = speech[:-silent] if silent and len(speech) > silent else speech
                            pcm = b"".join(usable)
                            speech = []
                            pre.clear()
                            silent = 0
                            hot = 0
                            threading.Thread(
                                target=self._transcribe_then_chat,
                                args=(pcm,),
                                name="voice-asr",
                                daemon=True,
                            ).start()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"microphone: {exc}"
            self.emit("error", detail=self.last_error, state=TurnState.ERROR.value)

    def _transcribe_then_chat(self, pcm: bytes) -> None:
        if self.turn_lock.locked():
            return
        started = time.monotonic()
        self.emit("status", text="正在识别", state=TurnState.TRANSCRIBING.value)
        try:
            chunk = AudioChunk(pcm=pcm, sample_rate=int(self.config["sample_rate"]))
            text = _run(self.asr().transcribe(chunk, language=self.config["language"])).strip()
            if len(text) < 2:
                self.emit("status", text="没有听清，请再说一次", state=TurnState.LISTENING.value)
                return
            self.emit(
                "user",
                text=text,
                state=TurnState.THINKING.value,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            self._conversation(text)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"asr: {exc}"
            self.emit("error", detail=self.last_error, state=TurnState.LISTENING.value)

    # -- conversation -------------------------------------------------------

    def _conversation(self, user_text: str) -> None:
        if not self.turn_lock.acquire(blocking=False):
            return
        try:
            self._conversation_locked(user_text)
        finally:
            self.turn_lock.release()

    def _conversation_locked(self, user_text: str) -> None:
        self.turn_id += 1
        turn = self.turn_id
        started = time.monotonic()
        self.emit("status", text="正在思考", state=TurnState.THINKING.value)

        messages = [ChatMessage(role="system", content=self.config["system_prompt"])]
        messages.extend(
            ChatMessage(role=item["role"], content=item["content"]) for item in self.history[-4:]
        )
        messages.append(ChatMessage(role="user", content=user_text))

        full = ""
        pending = ""
        last_partial = 0.0
        try:
            first = True
            buffer = ""
            for delta in _iter_sync(
                self.llm().stream(messages, max_tokens=48, temperature=0.35)
            ):
                if first:
                    first = False
                full += delta
                buffer += delta
                chunks, buffer = ready_sentences(buffer)
                if chunks:
                    full = clean_model_text(chunks[0])
                    break
                now = time.monotonic()
                if now - last_partial >= 0.08:
                    self.emit(
                        "assistant_partial",
                        text=clean_model_text(full),
                        state=TurnState.THINKING.value,
                    )
                    last_partial = now
                del pending
            full = clean_model_text(full)
            if not full:
                raise RuntimeError("LLM returned empty text")
            self.history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": full},
                ]
            )
            self.history = self.history[-10:]
            self.emit(
                "assistant",
                text=full,
                state=TurnState.SPEAKING.value,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            self.tts_queue.put((turn, full))
            self.tts_queue.put((turn, ""))
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"llm: {exc}"
            self.emit("error", detail=self.last_error, state=TurnState.LISTENING.value)

    def _tts_worker(self) -> None:
        active_turn = 0
        tts = self.tts()
        while self.running:
            item = self.tts_queue.get()
            if item is None:
                return
            turn, text = item
            if not text:
                if turn == active_turn:
                    self.speaking = False
                    self.playback_finished_at = time.monotonic()
                    self.emit(
                        "status",
                        text="正在聆听" if self.enabled else "语音已暂停",
                        state=TurnState.LISTENING.value if self.enabled else TurnState.PAUSED.value,
                    )
                continue
            active_turn = turn
            self.speaking = True
            self.emit("status", text="正在说话", state=TurnState.SPEAKING.value)
            try:
                audio, _fmt = _run(
                    tts.synthesize(
                        text,
                        voice=self.config["voice_profile_id"],
                        language=self.config["language"],
                        model=self.config["tts_engine"],
                    )
                )
                self._play(audio)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"tts: {exc}"
                self.emit("error", detail=self.last_error, state=TurnState.LISTENING.value)

    @staticmethod
    def _play(audio: bytes) -> None:
        """Play a WAV blob through the default output device."""

        try:
            import sounddevice as sd
            import numpy as np

            with wave.open(io.BytesIO(audio), "rb") as handle:
                rate = handle.getframerate()
                frames = handle.readframes(handle.getnframes())
            samples = np.frombuffer(frames, dtype="<i2")
            sd.play(samples, rate)
            sd.wait()
            return
        except Exception:  # noqa: BLE001 - fall back to the platform player
            pass
        try:
            import winsound

            temp = Path(PROJECT_ROOT) / ".voicebox-playback.wav"
            temp.write_bytes(audio)
            try:
                winsound.PlaySound(str(temp), winsound.SND_FILENAME)
            finally:
                temp.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"audio playback unavailable: {exc}") from exc


def _iter_sync(async_iterator) -> Any:
    """Drive an async iterator from a worker thread."""

    import asyncio

    loop = asyncio.new_event_loop()
    try:
        iterator = async_iterator.__aiter__()
        while True:
            try:
                yield loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# HTTP surface (identical paths to the pre-2.0 gateway)
# ---------------------------------------------------------------------------


def create_handler(service: LegacyVoiceService):
    class ApiHandler(BaseHTTPRequestHandler):
        companion = service

        def log_message(self, fmt: str, *args) -> None:
            print("VOICE_HTTP", fmt % args, flush=True)

        def _json(self, status: int, payload: Any) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass

        def _static(self, filename: str) -> None:
            import mimetypes

            path = LEGACY_WEB_DIR / filename
            if not path.is_file():
                self._json(404, {"error": "not_found"})
                return
            data = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type == "application/javascript":
                content_type += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = {
                "/": lambda: self._static("index.html"),
                "/app.js": lambda: self._static("app.js"),
                "/styles.css": lambda: self._static("styles.css"),
                "/health": lambda: self._json(200, self.companion.health()),
                "/events": lambda: self._json(
                    200,
                    {
                        "events": self.companion.events_after(
                            int(parse_qs(parsed.query).get("after", ["0"])[0])
                        )
                    },
                ),
            }
            handler = route.get(parsed.path)
            if handler is None:
                if parsed.path == "/options":
                    try:
                        self._json(200, self.companion.options())
                    except Exception as exc:  # noqa: BLE001
                        self._json(503, {"error": str(exc)})
                    return
                self._json(404, {"error": "not_found"})
                return
            handler()

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "invalid_json"})
                return
            if self.path == "/control":
                self.companion.set_enabled(bool(body.get("enabled", True)))
                self._json(200, self.companion.health())
            elif self.path == "/text":
                accepted = self.companion.submit_text(str(body.get("text", "")))
                self._json(202 if accepted else 409, {"accepted": accepted})
            elif self.path == "/settings":
                try:
                    self._json(200, self.companion.apply_settings(body))
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                except RuntimeError as exc:
                    self._json(409, {"error": str(exc)})
            else:
                self._json(404, {"error": "not_found"})

    return ApiHandler


def create_legacy_http_server(service: LegacyVoiceService) -> ThreadingHTTPServer:
    handler = create_handler(service)
    host = service.config.get("listen_host", "127.0.0.1")
    port = int(service.config.get("listen_port", 17831))
    return ThreadingHTTPServer((host, port), handler)
