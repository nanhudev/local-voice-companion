"""Local low-latency speech -> Ollama -> Voicebox gateway."""

from __future__ import annotations

import argparse
import audioop
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import queue
import re
import tempfile
import threading
import time
import mimetypes
from urllib.parse import parse_qs, urlparse
import wave
import winsound

import requests
import sounddevice as sd


CONFIG_PATH = Path(__file__).with_name("config.json")
CONFIG_EXAMPLE_PATH = Path(__file__).with_name("config.example.json")
WEB_DIR = Path(__file__).with_name("web")
SENTENCE_END = re.compile(r"[。！？!?；;\n]")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def load_config() -> dict:
    source = CONFIG_PATH if CONFIG_PATH.exists() else CONFIG_EXAMPLE_PATH
    return json.loads(source.read_text(encoding="utf-8"))


def resolve_available_defaults(config: dict) -> dict:
    """Select installed defaults when the example placeholders are unavailable."""
    changed = False
    session = requests.Session()
    try:
        response = session.get(f'{config["ollama_url"].rstrip("/")}/api/tags', timeout=4)
        response.raise_for_status()
        names = [item.get("name") for item in response.json().get("models", []) if item.get("name")]
        if names and config.get("ollama_model") not in names:
            config["ollama_model"] = names[0]
            changed = True
    except requests.RequestException:
        pass
    try:
        response = session.get(f'{config["voicebox_url"].rstrip("/")}/profiles', timeout=4)
        response.raise_for_status()
        profiles = [item for item in response.json() if item.get("id")]
        ids = {item["id"] for item in profiles}
        if profiles and config.get("voice_profile_id") not in ids:
            config["voice_profile_id"] = profiles[0]["id"]
            config["voice_profile_name"] = profiles[0].get("name", "Default")
            changed = True
    except requests.RequestException:
        pass
    if changed:
        CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def ready_sentences(buffer: str) -> tuple[list[str], str]:
    chunks: list[str] = []
    start = 0
    for match in SENTENCE_END.finditer(buffer):
        end = match.end()
        chunk = buffer[start:end].strip()
        if len(chunk) >= 4:
            chunks.append(chunk)
            start = end
    return chunks, buffer[start:]


def clean_model_text(text: str) -> str:
    text = THINK_BLOCK.sub("", text)
    # Some runtimes may stream an unclosed thought block before disconnecting.
    text = re.sub(r"<think>.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(?:助手|机器人|assistant)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


@dataclass
class Event:
    id: int
    type: str
    text: str = ""
    state: str = ""
    detail: str = ""
    elapsed_ms: int = 0


class VoiceCompanion:
    def __init__(self, config: dict):
        self.config = config
        self.enabled = True
        self.state = "starting"
        self.last_error = ""
        self.events: deque[dict] = deque(maxlen=240)
        self.event_id = 0
        self.lock = threading.Lock()
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=256)
        self.tts_queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self.turn_lock = threading.Lock()
        self.turn_id = 0
        self.speaking = False
        self.playback_finished_at = 0.0
        self.history: list[dict] = []
        self.running = True
        self.session = requests.Session()

    def emit(self, event_type: str, text: str = "", state: str = "", detail: str = "", elapsed_ms: int = 0) -> None:
        with self.lock:
            self.event_id += 1
            event = Event(self.event_id, event_type, text, state, detail, elapsed_ms)
            self.events.append(event.__dict__)
            if state:
                self.state = state

    def health(self) -> dict:
        return {
            "ok": not bool(self.last_error),
            "enabled": self.enabled,
            "state": self.state,
            "error": self.last_error,
            "ollama_model": self.config["ollama_model"],
            "voice": self.config["voice_profile_name"],
            "tts": self.config["tts_engine"] + (f'-{self.config["tts_model_size"]}' if self.config["tts_engine"] == "qwen" else ""),
            "asr": self.config["asr_model"],
            "asr_backend": self.config.get("asr_backend", "local"),
            "asr_worker": self.config.get("asr_worker_id", ""),
            "event_id": self.event_id,
        }

    def events_after(self, after: int) -> list[dict]:
        with self.lock:
            return [event for event in self.events if event["id"] > after]

    def options(self) -> dict:
        profiles_response = self.session.get(f'{self.config["voicebox_url"]}/profiles', timeout=5)
        profiles_response.raise_for_status()
        models_response = self.session.get(f'{self.config["voicebox_url"]}/models/status', timeout=5)
        models_response.raise_for_status()
        ollama_response = self.session.get(f'{self.config["ollama_url"]}/api/tags', timeout=5)
        ollama_response.raise_for_status()
        voicebox_models = models_response.json().get("models", [])
        downloaded = {item["model_name"] for item in voicebox_models if item.get("downloaded")}
        tts_engines = []
        if "luxtts" in downloaded:
            tts_engines.append({"id": "luxtts", "name": "LuxTTS（最快）", "sizes": []})
        qwen_sizes = [size for size, model in (("0.6B", "qwen-tts-0.6B"), ("1.7B", "qwen-tts-1.7B")) if model in downloaded]
        if qwen_sizes:
            tts_engines.append({"id": "qwen", "name": "Qwen3 TTS（中文质量）", "sizes": qwen_sizes})
        asr_models = [
            {"id": short, "name": label}
            for short, model, label in (
                ("base", "whisper-base", "Whisper Base（最快）"),
                ("small", "whisper-small", "Whisper Small"),
                ("medium", "whisper-medium", "Whisper Medium"),
                ("large", "whisper-large", "Whisper Large"),
                ("turbo", "whisper-turbo", "Whisper Turbo"),
            ) if model in downloaded
        ]
        relay_workers = []
        relay_token = os.getenv("AI_RELAY_TOKEN", "")
        relay_url = self.config.get("relay_url", "").rstrip("/")
        if relay_url and relay_token:
            try:
                response = self.session.get(
                    f"{relay_url}/workers",
                    headers={"Authorization": f"Bearer {relay_token}"},
                    timeout=4,
                )
                response.raise_for_status()
                relay_workers = [
                    item for item in response.json().get("workers", [])
                    if "asr" in item.get("capabilities", [])
                ]
                asr_models.extend(
                    {
                        "id": f"relay:{item['worker_id']}",
                        "name": f"{item['worker_id']}（CloudBase）",
                    }
                    for item in relay_workers
                )
            except requests.RequestException:
                pass
        current_asr = self.config.get("asr_model", "base")
        if self.config.get("asr_backend", "local") == "relay":
            current_asr = f"relay:{self.config.get('asr_worker_id', '')}"
        return {
            "current": {
                "voice_profile_id": self.config["voice_profile_id"],
                "ollama_model": self.config["ollama_model"],
                "asr_model": current_asr,
                "tts_engine": self.config["tts_engine"],
                "tts_model_size": self.config["tts_model_size"],
                "system_prompt": self.config.get("system_prompt", ""),
            },
            "profiles": [{"id": item["id"], "name": item["name"]} for item in profiles_response.json()],
            "ollama_models": [{"id": item["name"], "name": item["name"]} for item in ollama_response.json().get("models", [])],
            "asr_models": asr_models,
            "asr_workers": relay_workers,
            "tts_engines": tts_engines,
        }

    def apply_settings(self, requested: dict) -> dict:
        if self.turn_lock.locked() or self.speaking:
            raise RuntimeError("请等当前回答结束后再切换模型")
        options = self.options()
        allowed_profiles = {item["id"]: item["name"] for item in options["profiles"]}
        allowed_ollama = {item["id"] for item in options["ollama_models"]}
        allowed_asr = {item["id"] for item in options["asr_models"]}
        allowed_tts = {item["id"]: item for item in options["tts_engines"]}
        profile_id = str(requested.get("voice_profile_id", self.config["voice_profile_id"]))
        ollama_model = str(requested.get("ollama_model", self.config["ollama_model"]))
        asr_choice = str(requested.get("asr_model", self.config["asr_model"]))
        asr_backend = "relay" if asr_choice.startswith("relay:") else "local"
        asr_worker_id = asr_choice.split(":", 1)[1] if asr_backend == "relay" else ""
        asr_model = self.config["asr_model"] if asr_backend == "relay" else asr_choice
        tts_engine = str(requested.get("tts_engine", self.config["tts_engine"]))
        tts_size = str(requested.get("tts_model_size", self.config["tts_model_size"]))
        system_prompt = str(requested.get("system_prompt", self.config.get("system_prompt", ""))).strip()
        if profile_id not in allowed_profiles or ollama_model not in allowed_ollama or asr_choice not in allowed_asr or tts_engine not in allowed_tts:
            raise ValueError("所选模型或声线当前不可用")
        sizes = allowed_tts[tts_engine].get("sizes", [])
        if sizes and tts_size not in sizes:
            raise ValueError("所选 TTS 模型大小不可用")
        if len(system_prompt) > 2000:
            raise ValueError("提示词不能超过 2000 个字符")
        if not system_prompt:
            system_prompt = "你是一个自然、友好的中文聊天伙伴。请简短、直接地回答用户。"
        self.config.update({
            "voice_profile_id": profile_id,
            "voice_profile_name": allowed_profiles[profile_id],
            "ollama_model": ollama_model,
            "asr_model": asr_model,
            "asr_backend": asr_backend,
            "asr_worker_id": asr_worker_id,
            "tts_engine": tts_engine,
            "tts_model_size": tts_size,
            "system_prompt": system_prompt,
        })
        temp_path = CONFIG_PATH.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(self.config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(CONFIG_PATH)
        self.history.clear()
        self.emit("settings", text="模型设置已保存", state="warming")
        threading.Thread(target=self._rewarm_after_settings, name="voice-rewarm", daemon=True).start()
        return self.health()

    def _rewarm_after_settings(self) -> None:
        self._prepare_models()
        if self.enabled:
            self.emit("status", text="设置已生效，正在聆听", state="listening")
        else:
            self.emit("status", text="设置已生效，语音已暂停", state="paused")

    def probe_upstreams(self) -> dict:
        vb = self.session.get(f'{self.config["voicebox_url"]}/health', timeout=4)
        vb.raise_for_status()
        tags = self.session.get(f'{self.config["ollama_url"]}/api/tags', timeout=4)
        tags.raise_for_status()
        names = [item.get("name") for item in tags.json().get("models", [])]
        if self.config["ollama_model"] not in names:
            raise RuntimeError("configured Ollama model is not installed")
        profiles = self.session.get(f'{self.config["voicebox_url"]}/profiles', timeout=4)
        profiles.raise_for_status()
        profile_ids = {item.get("id") for item in profiles.json()}
        if self.config["voice_profile_id"] not in profile_ids:
            raise RuntimeError("configured Voicebox profile is missing")
        return {"voicebox": True, "ollama": True, "model": self.config["ollama_model"]}

    def start(self) -> None:
        try:
            self.probe_upstreams()
        except Exception as exc:
            self.last_error = str(exc)
            self.emit("error", detail=self.last_error, state="error")
            return
        self._prepare_models()
        threading.Thread(target=self._tts_worker, name="voice-tts", daemon=True).start()
        threading.Thread(target=self._microphone_worker, name="voice-mic", daemon=True).start()
        self.emit("status", text="语音机器人已就绪", state="listening")

    def _prepare_models(self) -> None:
        """Pay cold-start costs before gameplay so the first spoken turn is fast."""
        self.state = "warming"
        try:
            # Keep only the selected ASR model resident; Voicebox may otherwise
            # leave Whisper Turbo loaded and starve Ollama of VRAM.
            asr_names = {
                "base": "whisper-base", "small": "whisper-small",
                "medium": "whisper-medium", "large": "whisper-large",
                "turbo": "whisper-turbo",
            }
            selected_asr = asr_names.get(self.config.get("asr_model"))
            if self.config.get("asr_backend", "local") == "local":
                for model_name in asr_names.values():
                    if model_name != selected_asr:
                        self.session.post(f'{self.config["voicebox_url"]}/models/{model_name}/unload', timeout=15)
            if self.config.get("unload_unused_tts_models") and self.config["tts_engine"] != "qwen":
                for model_name in ("qwen-tts-0.6B", "qwen-tts-1.7B"):
                    self.session.post(f'{self.config["voicebox_url"]}/models/{model_name}/unload', timeout=15)
            if self.config.get("asr_backend", "local") == "local":
                silence = wav_bytes(b"\0\0" * (int(self.config["sample_rate"]) // 3), int(self.config["sample_rate"]))
                self.session.post(
                    f'{self.config["voicebox_url"]}/transcribe',
                    files={"file": ("warmup.wav", silence, "audio/wav")},
                    data={"language": self.config["language"], "model": self.config["asr_model"]},
                    timeout=90,
                ).raise_for_status()
            # Load Ollama last so its GPU allocation reflects the final Voicebox
            # footprint and the first real dialogue does not pay a model reload.
            self.session.post(
                f'{self.config["ollama_url"]}/api/generate',
                json={
                    "model": self.config["ollama_model"], "prompt": "", "keep_alive": "30m",
                    "stream": False, "options": {"num_ctx": 2048},
                },
                timeout=120,
            ).raise_for_status()
            self.session.post(
                f'{self.config["voicebox_url"]}/generate/stream',
                json={
                    "profile_id": self.config["voice_profile_id"], "text": "准备好了。",
                    "language": self.config["language"], "engine": self.config["tts_engine"],
                    "model_size": self.config["tts_model_size"], "normalize": True,
                },
                timeout=90,
            ).raise_for_status()
        except Exception as exc:
            self.emit("audio_warning", detail=f"warmup: {exc}")

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.emit("status", text="正在聆听" if enabled else "语音已暂停", state="listening" if enabled else "paused")

    def submit_text(self, text: str) -> bool:
        text = text.strip()
        if not text or self.speaking or self.state in {"thinking", "transcribing"}:
            return False
        threading.Thread(target=self._conversation, args=(text,), name="voice-turn", daemon=True).start()
        return True

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            self.emit("audio_warning", detail=str(status))
        try:
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _microphone_worker(self) -> None:
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
            with sd.RawInputStream(samplerate=rate, blocksize=block_frames, channels=1, dtype="int16", callback=self._audio_callback):
                while self.running:
                    block = self.audio_queue.get()
                    if (not self.enabled or self.speaking or self.turn_lock.locked() or self.state in {"warming", "transcribing", "thinking", "speaking"}
                            or time.monotonic() - self.playback_finished_at < self.config["silence_after_playback_ms"] / 1000):
                        pre.clear(); speech.clear(); silent = 0; hot = 0
                        continue
                    rms = audioop.rms(block, 2)
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
                            self.emit("status", text="听到了，请继续说", state="hearing")
                    else:
                        speech.append(block)
                        silent = silent + 1 if rms < threshold * 0.78 else 0
                        if (silent >= end_blocks and len(speech) >= min_blocks) or len(speech) >= max_blocks:
                            usable = speech[:-silent] if silent and len(speech) > silent else speech
                            pcm = b"".join(usable)
                            speech = []; pre.clear(); silent = 0; hot = 0
                            threading.Thread(target=self._transcribe_then_chat, args=(pcm,), name="voice-asr", daemon=True).start()
        except Exception as exc:
            self.last_error = f"microphone: {exc}"
            self.emit("error", detail=self.last_error, state="error")

    def _transcribe_then_chat(self, pcm: bytes) -> None:
        if self.turn_lock.locked() or self.state in {"transcribing", "thinking", "speaking"}:
            return
        started = time.monotonic()
        self.emit("status", text="正在识别", state="transcribing")
        try:
            audio = wav_bytes(pcm, int(self.config["sample_rate"]))
            if self.config.get("asr_backend", "local") == "relay":
                relay_url = self.config["relay_url"].rstrip("/")
                relay_token = os.getenv("AI_RELAY_TOKEN", "")
                if not relay_token:
                    raise RuntimeError("AI_RELAY_TOKEN 未配置")
                response = self.session.post(
                    f'{relay_url}/asr',
                    headers={"Authorization": f"Bearer {relay_token}"},
                    files={"file": ("utterance.wav", audio, "audio/wav")},
                    data={
                        "language": self.config["language"],
                        "provider": self.config.get("relay_provider", "mlx_whisper"),
                        "worker_id": self.config.get("asr_worker_id", ""),
                    },
                    timeout=75,
                )
            else:
                response = self.session.post(
                    f'{self.config["voicebox_url"]}/transcribe',
                    files={"file": ("utterance.wav", audio, "audio/wav")},
                    data={"language": self.config["language"], "model": self.config["asr_model"]},
                    timeout=90,
                )
            response.raise_for_status()
            text = response.json().get("text", "").strip()
            if len(text) < 2:
                self.emit("status", text="没有听清，请再说一次", state="listening")
                return
            self.emit("user", text=text, state="thinking", elapsed_ms=int((time.monotonic() - started) * 1000))
            self._conversation(text)
        except Exception as exc:
            self.last_error = f"asr: {exc}"
            self.emit("error", detail=self.last_error, state="listening")

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
        self.emit("status", text="正在思考", state="thinking")
        messages = [{"role": "system", "content": self.config["system_prompt"]}]
        messages.extend(self.history[-4:])
        messages.append({"role": "user", "content": user_text})
        full = ""
        pending = ""
        last_partial = 0.0
        try:
            with self.session.post(
                f'{self.config["ollama_url"]}/api/chat',
                json={
                    "model": self.config["ollama_model"], "messages": messages,
                    "stream": True, "think": False, "keep_alive": "30m",
                    "options": {"temperature": 0.35, "num_predict": 48, "num_ctx": 1024},
                },
                stream=True, timeout=120,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1):
                    if not line:
                        continue
                    payload = json.loads(line)
                    token = payload.get("message", {}).get("content", "")
                    if token:
                        full += token; pending += token
                        chunks, pending = ready_sentences(pending)
                        now = time.monotonic()
                        if chunks or now - last_partial >= 0.08:
                            self.emit("assistant_partial", text=full, state="speaking" if chunks else "thinking")
                            last_partial = now
                        if chunks:
                            full = chunks[0]
                            break
            full = clean_model_text(full)
            if not full:
                raise RuntimeError("Ollama returned empty text")
            self.history.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": full}])
            self.history = self.history[-10:]
            self.emit("assistant", text=full, state="speaking", elapsed_ms=int((time.monotonic() - started) * 1000))
            self.tts_queue.put((turn, full))
            self.tts_queue.put((turn, ""))
        except Exception as exc:
            self.last_error = f"llm: {exc}"
            self.emit("error", detail=self.last_error, state="listening")

    def _tts_worker(self) -> None:
        active_turn = 0
        while self.running:
            item = self.tts_queue.get()
            if item is None:
                return
            turn, text = item
            if not text:
                if turn == active_turn:
                    self.speaking = False
                    self.playback_finished_at = time.monotonic()
                    if self.enabled:
                        self.emit("status", text="正在聆听", state="listening")
                    else:
                        self.emit("status", text="语音已暂停", state="paused")
                continue
            active_turn = turn
            self.speaking = True
            self.emit("status", text="正在说话", state="speaking")
            try:
                response = self.session.post(
                    f'{self.config["voicebox_url"]}/generate/stream',
                    json={
                        "profile_id": self.config["voice_profile_id"], "text": text,
                        "language": self.config["language"], "engine": self.config["tts_engine"],
                        "model_size": self.config["tts_model_size"], "normalize": True,
                        "max_chunk_chars": 220, "crossfade_ms": 25,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(prefix="voicebot_", suffix=".wav", delete=False) as handle:
                    handle.write(response.content)
                    audio_path = Path(handle.name)
                try:
                    winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
                finally:
                    audio_path.unlink(missing_ok=True)
            except Exception as exc:
                self.last_error = f"tts: {exc}"
                self.emit("error", detail=self.last_error, state="listening")


class ApiHandler(BaseHTTPRequestHandler):
    companion: VoiceCompanion

    def log_message(self, fmt: str, *args) -> None:
        print("VOICE_HTTP", fmt % args, flush=True)

    def _json(self, status: int, payload: dict | list) -> None:
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
        path = WEB_DIR / filename
        if not path.is_file():
            self._json(404, {"error": "not_found"})
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") or content_type == "application/javascript" else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._static("index.html")
        elif parsed.path == "/app.js":
            self._static("app.js")
        elif parsed.path == "/styles.css":
            self._static("styles.css")
        elif parsed.path == "/health":
            self._json(200, self.companion.health())
        elif parsed.path == "/events":
            after = int(parse_qs(parsed.query).get("after", ["0"])[0])
            self._json(200, {"events": self.companion.events_after(after)})
        elif parsed.path == "/options":
            try:
                self._json(200, self.companion.options())
            except Exception as exc:
                self._json(503, {"error": str(exc)})
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid_json"}); return
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--auto-start", action="store_true", help="start locally installed Ollama and Voicebox backends")
    args = parser.parse_args()
    config = load_config()
    if args.auto_start:
        from backends import ensure_backends

        ensure_backends(config)
    config = resolve_available_defaults(config)
    companion = VoiceCompanion(config)
    if args.probe:
        try:
            print(json.dumps(companion.probe_upstreams(), ensure_ascii=False))
            return 0
        except Exception as exc:
            print(f"VOICE_PROBE_FAIL {exc}")
            return 1
    ApiHandler.companion = companion
    server = ThreadingHTTPServer((config["listen_host"], int(config["listen_port"])), ApiHandler)
    print(f'VOICE_GATEWAY_READY http://{config["listen_host"]}:{config["listen_port"]}', flush=True)
    threading.Thread(target=companion.start, name="voice-startup", daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        companion.running = False
        companion.tts_queue.put(None)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
