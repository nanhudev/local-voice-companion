"""Real upstream smoke test without playing audio through the speakers."""

from __future__ import annotations

import io
import json
from pathlib import Path
import time
import wave

import requests


CONFIG_PATH = Path(__file__).with_name("config.json")
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).with_name("config.example.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def main() -> int:
    started = time.monotonic()
    speech = requests.post(
        CONFIG["voicebox_url"] + "/generate/stream",
        json={
            "profile_id": CONFIG["voice_profile_id"],
            "text": "你好，这是本地语音链路测试。",
            "language": "zh",
            "engine": CONFIG["tts_engine"],
            "model_size": CONFIG["tts_model_size"],
            "normalize": True,
        },
        timeout=180,
    )
    speech.raise_for_status()
    if not speech.content.startswith(b"RIFF"):
        raise RuntimeError("Voicebox did not return WAV audio")
    tts_ms = int((time.monotonic() - started) * 1000)
    with wave.open(io.BytesIO(speech.content), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()

    asr_started = time.monotonic()
    transcript = requests.post(
        CONFIG["voicebox_url"] + "/transcribe",
        files={"file": ("smoke.wav", speech.content, "audio/wav")},
        data={"language": "zh", "model": CONFIG["asr_model"]},
        timeout=180,
    )
    if not transcript.ok:
        raise RuntimeError(f"ASR HTTP {transcript.status_code}: {transcript.text}")
    asr_ms = int((time.monotonic() - asr_started) * 1000)
    text = transcript.json().get("text", "").strip()
    if not text:
        raise RuntimeError("ASR returned empty text")
    print(f"VOICE_RUNTIME_PASS tts_ms={tts_ms} audio_seconds={duration:.2f} asr_ms={asr_ms} transcript={text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
