"""Read-only diagnostics for Local Voice Studio."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import requests


ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = ROOT / "config.example.json"


def check_json(url: str) -> tuple[bool, str]:
    try:
        response = requests.get(url, timeout=4)
        response.raise_for_status()
        response.json()
        return True, f"HTTP {response.status_code}"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    checks = [
        ("Python 3.11+", sys.version_info >= (3, 11), sys.version.split()[0]),
        ("Voicebox", *check_json(config["voicebox_url"].rstrip("/") + "/health")),
        ("Ollama", *check_json(config["ollama_url"].rstrip("/") + "/api/tags")),
    ]
    try:
        import sounddevice as sd

        inputs = [d for d in sd.query_devices() if int(d.get("max_input_channels", 0)) > 0]
        checks.append(("Microphone", bool(inputs), f"{len(inputs)} input device(s)"))
    except Exception as exc:
        checks.append(("Microphone", False, str(exc)))

    print("Local Voice Studio doctor / 本地语音工作台诊断")
    print(f"Config: {CONFIG_PATH}")
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")
    required = checks[:3]
    return 0 if all(item[1] for item in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
