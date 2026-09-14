"""Compatibility launcher for the pre-2.0 entry point.

`python app.py` used to boot the original Voicebox + Ollama gateway. That
gateway now lives inside the package, and this file is a thin shim so existing
shortcuts, scripts and muscle memory keep working:

    python app.py                       -> legacy gateway (identical behaviour)
    python app.py --probe               -> probe upstream backends and exit
    python app.py --auto-start          -> start local Ollama/Voicebox first
    python app.py --runtime             -> the new adaptive runtime (uvicorn)

The three text helpers stay here rather than moving: they are small, pure,
dependency-free, and ``tests/legacy/test_app_py.py`` covers them directly. They
are not the old monolith, they are three functions.

The rule this file must respect: it must never grow back into a monolith.
Anything new belongs in the package (RULE 15).
"""

from __future__ import annotations

import io
import re
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


SENTENCE_END = re.compile(r"[。！？!?；;\n]")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# text helpers (pure, tiny, and directly tested)
# ---------------------------------------------------------------------------


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container."""

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def ready_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off completed sentences, returning them plus the trailing remainder."""

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
    """Strip reasoning blocks and role prefixes a model may leak into its reply."""

    text = THINK_BLOCK.sub("", text)
    # Some runtimes may stream an unclosed thought block before disconnecting.
    text = re.sub(r"<think>.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(?:助手|机器人|assistant)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if any(item in {"-h", "--help"} for item in args):
        print(__doc__.strip())
        print()
        print("Routing:")
        print("  --runtime      start the adaptive runtime (FastAPI + uvicorn)")
        print("  --legacy       start the pre-2.0 Voicebox + Ollama gateway (default)")
        print("  --probe        probe upstream backends and exit")
        print("  --auto-start   start locally installed backends before serving")
        return 0

    if "--runtime" in args:
        return _run_runtime([item for item in args if item != "--runtime"])

    # Strip the explicit flag so the package CLI receives a clean argument list.
    forwarded = [item for item in args if item != "--legacy"]
    return _run_legacy(forwarded)


def _run_runtime(argv: list[str]) -> int:
    """Serve the adaptive runtime through the package CLI.

    Arguments are forwarded verbatim: the `serve` subcommand already accepts
    ``--host`` / ``--port`` / ``--log-level``, so rewriting them here would only
    introduce a translation step that can disagree with the real parser. (It
    did: stripping the leading dashes turned `--port 18771` into `port 18771`.)
    """

    from local_voice_companion.__main__ import main as runtime_main

    return runtime_main(["serve", *argv])


def _run_legacy(argv: list[str]) -> int:
    """Serve the packaged legacy gateway.

    Mapped onto the package CLI so there is exactly one implementation of the
    old gateway. Keeping a second copy here is how a compat shim silently
    drifts away from the code it is meant to preserve.
    """

    from local_voice_companion.__main__ import main as runtime_main

    return runtime_main(["legacy", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
