from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import uuid
import wave

import websockets


WORKER_ID = os.getenv("AI_WORKER_ID", "windows-main")
RELAY_WS_URL = os.getenv(
    "AI_RELAY_WS_URL",
    "ws://127.0.0.1:17832/worker",
)
VOICEBOX_URL = os.getenv("VOICEBOX_URL", "http://127.0.0.1:17493").rstrip("/")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_LLM_MODEL = os.getenv("AI_WORKER_LLM_MODEL", "qwen3:1.7b")
DEFAULT_ASR_MODEL = os.getenv("AI_WORKER_ASR_MODEL", "base")
DEFAULT_TTS_PROFILE = os.getenv("AI_WORKER_TTS_PROFILE", "")
FFMPEG_PATH = Path(os.getenv(
    "FFMPEG_PATH",
    "ffmpeg",
))
VOICEBOX_EXE = Path(os.getenv(
    "VOICEBOX_EXE",
    "voicebox-server-cuda.exe",
))
VOICEBOX_DATA_DIR = Path(os.getenv(
    "VOICEBOX_DATA_DIR",
    ".",
))
GPU_LOCK = asyncio.Lock()
VOICEBOX_START_LOCK = threading.Lock()
voicebox_process = None


def http_json(url: str, payload: dict | None = None, timeout: int = 180) -> dict | list:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def ensure_voicebox() -> None:
    global voicebox_process
    try:
        http_json(f"{VOICEBOX_URL}/health", timeout=3)
        return
    except Exception:
        pass
    with VOICEBOX_START_LOCK:
        try:
            http_json(f"{VOICEBOX_URL}/health", timeout=3)
            return
        except Exception:
            pass
        if not VOICEBOX_EXE.is_file():
            raise RuntimeError(f"Voicebox backend is missing: {VOICEBOX_EXE}")
        voicebox_process = subprocess.Popen(
            [str(VOICEBOX_EXE), "--data-dir", str(VOICEBOX_DATA_DIR), "--port", "17493",
             "--parent-pid", str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for _ in range(50):
            time.sleep(1)
            try:
                http_json(f"{VOICEBOX_URL}/health", timeout=3)
                return
            except Exception:
                if voicebox_process.poll() is not None:
                    break
        raise RuntimeError("Voicebox did not become ready")


def post_multipart(url: str, audio: bytes, mime_type: str, language: str, model: str) -> dict:
    boundary = "----xiaoqi" + uuid.uuid4().hex
    parts = []
    for name, value in (("language", language), ("model", model)):
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
        f"Content-Type: {mime_type}\r\n\r\n".encode()
    )
    body = b"".join(parts) + audio + f"\r\n--{boundary}--\r\n".encode()
    request = Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def get_runtime_options() -> tuple[list[str], list[dict]]:
    models = []
    voices = []
    try:
        tags = http_json(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [item["name"] for item in tags.get("models", []) if item.get("name")]
    except Exception:
        pass
    try:
        ensure_voicebox()
        profiles = http_json(f"{VOICEBOX_URL}/profiles", timeout=5)
        voices = [{"id": item.get("id"), "name": item.get("name")} for item in profiles if item.get("id")]
    except Exception:
        pass
    return models, voices


def compress_for_web(audio: bytes, mime_type: str) -> tuple[bytes, str]:
    if not FFMPEG_PATH.is_file():
        return audio, mime_type
    process = subprocess.run(
        [str(FFMPEG_PATH), "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
         "-vn", "-ac", "1", "-ar", "24000", "-b:a", "48k", "-f", "mp3", "pipe:1"],
        input=audio, capture_output=True, timeout=30, check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return (process.stdout, "audio/mpeg") if process.returncode == 0 and process.stdout else (audio, mime_type)


def warmup_runtime() -> None:
    ensure_voicebox()
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 4800)
    try:
        post_multipart(f"{VOICEBOX_URL}/transcribe", wav_buffer.getvalue(), "audio/wav", "zh", DEFAULT_ASR_MODEL)
    except Exception:
        pass
    try:
        http_json(f"{OLLAMA_URL}/api/generate", {
            "model": DEFAULT_LLM_MODEL, "prompt": "", "stream": False,
            "think": False, "keep_alive": "30m", "options": {"num_ctx": 1024},
        }, 180)
    except Exception:
        pass
    try:
        profiles = http_json(f"{VOICEBOX_URL}/profiles", timeout=5)
        profile_id = DEFAULT_TTS_PROFILE or (profiles[0]["id"] if profiles else "")
        if profile_id:
            payload = json.dumps({
                "profile_id": profile_id, "text": "准备好了。", "language": "zh",
                "engine": "luxtts", "model_size": "0.6B", "normalize": True,
                "max_chunk_chars": 220, "crossfade_ms": 25,
            }, ensure_ascii=False).encode("utf-8")
            request = Request(f"{VOICEBOX_URL}/generate/stream", data=payload, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=180) as response:
                response.read()
    except Exception:
        pass


async def handle_request(message: dict) -> dict:
    request_id = message.get("request_id")
    request_type = message.get("type")
    started = time.monotonic()
    if request_type in {"asr", "tts"}:
        await asyncio.to_thread(ensure_voicebox)
    async with GPU_LOCK:
        if request_type == "asr":
            audio = base64.b64decode(message["audio_base64"], validate=True)
            result = await asyncio.to_thread(
                post_multipart,
                f"{VOICEBOX_URL}/transcribe",
                audio,
                message.get("mime_type") or "audio/wav",
                message.get("language") or "zh",
                message.get("model") or DEFAULT_ASR_MODEL,
            )
            return {
                "type": "asr_result", "request_id": request_id,
                "text": result.get("text", ""), "language": message.get("language") or "zh",
                "provider": "voicebox", "model": message.get("model") or DEFAULT_ASR_MODEL,
                "latency_ms": int((time.monotonic() - started) * 1000),
            }

        if request_type == "chat":
            messages = message.get("messages") or [{"role": "user", "content": message.get("prompt", "")}]
            model = message.get("model") or DEFAULT_LLM_MODEL
            result = await asyncio.to_thread(
                http_json,
                f"{OLLAMA_URL}/api/chat",
                {
                    "model": model, "messages": messages, "stream": False, "think": False,
                    "keep_alive": "30m", "options": {
                        "temperature": message.get("temperature", 0.35),
                        "num_predict": message.get("max_tokens", 128), "num_ctx": 1024,
                    },
                },
                180,
            )
            return {
                "type": "llm_result", "request_id": request_id,
                "text": result.get("message", {}).get("content", ""),
                "provider": "ollama", "model": model,
                "latency_ms": int((time.monotonic() - started) * 1000),
            }

        if request_type == "tts":
            profiles = await asyncio.to_thread(http_json, f"{VOICEBOX_URL}/profiles", None, 5)
            requested_profile = message.get("profile_id") or message.get("voice")
            profile_ids = {item.get("id") for item in profiles}
            profile_id = requested_profile if requested_profile in profile_ids else DEFAULT_TTS_PROFILE
            if profile_id not in profile_ids:
                profile_id = profiles[0]["id"] if profiles else None
            if not profile_id:
                raise RuntimeError("No Voicebox profile is available")
            payload = {
                "profile_id": profile_id, "text": message.get("text", ""),
                "language": message.get("language") or "zh",
                "engine": message.get("engine") or "luxtts",
                "model_size": message.get("model_size") or "0.6B",
                "normalize": True, "max_chunk_chars": 220, "crossfade_ms": 25,
            }
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = Request(f"{VOICEBOX_URL}/generate/stream", data=data, headers={"Content-Type": "application/json"})
            def synthesize() -> tuple[bytes, str]:
                try:
                    with urlopen(request, timeout=180) as response:
                        return response.read(), response.headers.get_content_type()
                except HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
            audio, mime_type = await asyncio.to_thread(synthesize)
            audio, mime_type = await asyncio.to_thread(compress_for_web, audio, mime_type)
            return {
                "type": "tts_result", "request_id": request_id,
                "audio_base64": base64.b64encode(audio).decode("ascii"), "mime_type": mime_type,
                "provider": "voicebox", "latency_ms": int((time.monotonic() - started) * 1000),
            }

    raise RuntimeError(f"Unsupported request type: {request_type}")


async def heartbeat(ws) -> None:
    while True:
        await asyncio.sleep(15)
        models, voices = await asyncio.to_thread(get_runtime_options)
        await ws.send(json.dumps({
            "type": "heartbeat", "capabilities": ["asr", "llm", "tts"],
            "providers": ["voicebox", "ollama"], "models": models, "voices": voices,
        }, ensure_ascii=False))


async def run_worker() -> None:
    token = os.getenv("AI_RELAY_TOKEN", "")
    if not token:
        raise RuntimeError("AI_RELAY_TOKEN is not configured")
    await asyncio.to_thread(warmup_runtime)
    url = f"{RELAY_WS_URL}?token={quote(token, safe='')}"
    while True:
        try:
            async with websockets.connect(
                url, max_size=40 * 1024 * 1024, ping_interval=20,
                ping_timeout=30, open_timeout=30,
            ) as ws:
                models, voices = await asyncio.to_thread(get_runtime_options)
                await ws.send(json.dumps({
                    "type": "register", "worker_id": WORKER_ID, "platform": "windows-x64-rtx2070",
                    "capabilities": ["asr", "llm", "tts"], "providers": ["voicebox", "ollama"],
                    "models": models, "voices": voices,
                }, ensure_ascii=False))
                heartbeat_task = asyncio.create_task(heartbeat(ws))
                try:
                    async for raw in ws:
                        message = json.loads(raw)
                        if message.get("type") in {"registered", "heartbeat"}:
                            continue
                        try:
                            result = await handle_request(message)
                        except Exception as exc:
                            result = {"type": "error", "request_id": message.get("request_id"), "error": str(exc)[:500]}
                        await ws.send(json.dumps(result, ensure_ascii=False))
                finally:
                    heartbeat_task.cancel()
        except Exception as exc:
            print(f"WINDOWS_WORKER_RECONNECT {type(exc).__name__}: {exc}", flush=True)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(run_worker())
