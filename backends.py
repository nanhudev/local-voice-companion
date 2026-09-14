"""Optional discovery and startup for locally installed voice backends."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

import requests


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def healthy(url: str) -> bool:
    try:
        return requests.get(url, timeout=2).ok
    except requests.RequestException:
        return False


def first_file(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if Path(candidate).is_file():
                return Path(candidate)
        except OSError:
            continue
    return None


def start_ollama(config: dict) -> bool:
    health = config["ollama_url"].rstrip("/") + "/api/tags"
    if healthy(health):
        return True
    local_app_data = Path(os.getenv("LOCALAPPDATA", ""))
    executable = first_file(
        os.getenv("OLLAMA_EXE"),
        shutil.which("ollama"),
        local_app_data / "Programs" / "Ollama" / "ollama.exe",
    )
    if not executable:
        return False
    subprocess.Popen(
        [str(executable), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
    )
    return wait_until(health, 20)


def start_voicebox(config: dict) -> bool:
    health = config["voicebox_url"].rstrip("/") + "/health"
    if healthy(health):
        return True
    app_data = Path(os.getenv("APPDATA", ""))
    data_dir = Path(os.getenv("VOICEBOX_DATA_DIR", str(app_data / "sh.voicebox.app")))
    executable = first_file(
        os.getenv("VOICEBOX_EXE"),
        data_dir / "backends" / "cuda" / "voicebox-server-cuda.exe",
    )
    if not executable:
        return False
    port = str(config.get("voicebox_port", 17493))
    subprocess.Popen(
        [str(executable), "--data-dir", str(data_dir), "--port", port, "--parent-pid", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
    )
    return wait_until(health, 45)


def wait_until(url: str, seconds: int) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if healthy(url):
            return True
        time.sleep(0.5)
    return False


def ensure_backends(config: dict) -> dict[str, bool]:
    print("Starting local backends when available...", flush=True)
    result = {"ollama": start_ollama(config), "voicebox": start_voicebox(config)}
    print(f"Backend status: {result}", flush=True)
    return result
