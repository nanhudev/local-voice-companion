"""Command line entry point for the adaptive voice runtime.

    python -m local_voice_companion serve            # FastAPI runtime
    python -m local_voice_companion legacy           # pre-2.0 gateway surface
    python -m local_voice_companion probe            # hardware + provider report
    python -m local_voice_companion doctor           # read-only diagnostics
    python -m local_voice_companion plan             # show the selected pipeline
    python -m local_voice_companion where            # data root resolution
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Sequence


def _json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from local_voice_companion.api.app import create_app
    from local_voice_companion.config.loader import load_config
    from local_voice_companion.config.paths import DEFAULT_LAYOUT

    config = load_config()
    DEFAULT_LAYOUT.ensure()

    host = args.host or config.server.host
    port = args.port or config.server.port

    if host not in {"127.0.0.1", "localhost", "::1"}:
        import os

        token = os.getenv(config.server.auth_token_env, "")
        if not token:
            print(
                f"REFUSING TO START: binding to {host} requires "
                f"{config.server.auth_token_env} to be set",
                file=sys.stderr,
            )
            return 2

    app = create_app(config=config)
    print(f"LVC_RUNTIME_READY http://{host}:{port}")
    print(f"LVC_DOCS          http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return 0


def cmd_legacy(args: argparse.Namespace) -> int:
    """Preserve the exact pre-2.0 startup surface."""

    from local_voice_companion.compat.legacy_config import LegacyConfigAdapter
    from local_voice_companion.compat.voicebox import LegacyVoiceService, create_legacy_http_server

    adapter = LegacyConfigAdapter()
    if args.auto_start:
        from local_voice_companion.config.paths import PROJECT_ROOT

        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            import backends  # type: ignore[import-not-found]

            backends.ensure_backends(adapter.as_flat())
        except Exception as exc:  # noqa: BLE001 - optional helper
            print(f"VOICE_BACKEND_DISCOVERY_SKIPPED {exc}", file=sys.stderr)

    service = LegacyVoiceService(adapter)
    if args.probe:
        try:
            print(json.dumps(service.probe_upstreams(), ensure_ascii=False))
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"VOICE_PROBE_FAIL {exc}")
            return 1

    server = create_legacy_http_server(service)
    host, port = server.server_address[:2]
    print(f"VOICE_GATEWAY_READY http://{host}:{port}", flush=True)
    import threading

    threading.Thread(target=service.start, name="voice-startup", daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        service.running = False
        service.tts_queue.put(None)
        server.server_close()
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from local_voice_companion.config.loader import load_config
    from local_voice_companion.hardware.probe import capability_summary, probe_hardware

    profile = probe_hardware(include_audio=not args.no_audio)
    config = load_config()
    payload = {
        "capabilities": capability_summary(profile),
        "profile": profile.to_dict(),
        "config": config.summary(),
    }
    if not args.no_providers:
        from local_voice_companion.providers.fake import FAKE_PROVIDERS
        from local_voice_companion.providers.legacy import LEGACY_PROVIDERS, legacy_options
        from local_voice_companion.providers.discovery import discover
        from local_voice_companion.providers.registry import registry

        for cls in (*FAKE_PROVIDERS, *LEGACY_PROVIDERS):
            if cls.descriptor().id not in registry:
                registry.register(cls)

        results = asyncio.run(
            discover(registry, options_by_id=legacy_options(config) if config.legacy.enabled else {})
        )
        payload["providers"] = [item.to_dict() for item in results]
    _json(payload)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from local_voice_companion.config.loader import load_config
    from local_voice_companion.hardware.probe import probe_hardware
    from local_voice_companion.providers.fake import FAKE_PROVIDERS
    from local_voice_companion.providers.legacy import LEGACY_PROVIDERS, legacy_options
    from local_voice_companion.providers.registry import registry
    from local_voice_companion.selection.engine import recommend

    config = load_config()
    for cls in (*FAKE_PROVIDERS, *LEGACY_PROVIDERS):
        if cls.descriptor().id not in registry:
            registry.register(cls)

    profile = probe_hardware(include_audio=not args.no_audio)
    decision = asyncio.run(
        recommend(
            profile,
            config,
            policy_override=args.policy,
        )
    )
    _json(decision.to_dict(include_alternatives=args.alternatives))
    return 0 if decision.feasible else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    from local_voice_companion.config.loader import load_config, migration_report
    from local_voice_companion.config.paths import DEFAULT_LAYOUT
    from local_voice_companion.hardware.probe import OPTIONAL_PACKAGES, capability_summary, probe_hardware

    print("Local Voice Companion doctor")
    checks: list[tuple[str, bool, str]] = [
        ("Python 3.11+", sys.version_info >= (3, 11), sys.version.split()[0]),
    ]

    profile = probe_hardware()
    capabilities = capability_summary(profile)
    checks.append(("Hardware probe", not profile.partial, "; ".join(profile.notes[:2]) or "complete"))
    checks.append(
        (
            "Microphone",
            bool(profile.audio.input_devices),
            f"{len(profile.audio.input_devices)} input device(s)",
        )
    )
    checks.append(
        (
            "Audio output",
            bool(profile.audio.output_devices),
            f"{len(profile.audio.output_devices)} output device(s)",
        )
    )

    for package in ("fastapi", "uvicorn", "pydantic", "requests", "sounddevice"):
        present = package in profile.runtime.dependencies
        checks.append((f"Dependency {package}", present, profile.runtime.dependencies.get(package, "missing")))

    # Optional inference backends are reported as information, never as
    # failures: a healthy CPU-only install legitimately has none of them.
    optional_present = [
        name for name in profile.runtime.dependencies
        if name in set(OPTIONAL_PACKAGES)
    ]
    checks.append(
        (
            "Inference backends",
            True,
            f"{len(optional_present)} installed"
            + (f": {', '.join(sorted(optional_present))}" if optional_present else " (CPU/stub providers only)"),
        )
    )

    config = load_config()
    migration = migration_report()
    checks.append(("Config schema", True, f"v{config.schema_version} ({config.server.host}:{config.server.port})"))
    checks.append(("Data root", DEFAULT_LAYOUT.writable, str(DEFAULT_LAYOUT.root)))

    print(f"Data root: {DEFAULT_LAYOUT.root} ({DEFAULT_LAYOUT.note})")
    print(f"Fingerprint: {capabilities['fingerprint']}")
    if capabilities["gpu"]:
        gpu = capabilities["gpu"]
        print(f"GPU: {gpu['model']} ({gpu['vram_mb']} MB) accelerators={gpu['accelerators']}")
    else:
        print("GPU: none detected (CUDA/ROCm code paths will not be exercised here)")
    if migration.get("migrated"):
        print(f"Config migrated: v{migration['from_version']} -> v{migration['to_version']}")
        for note in migration.get("notes", []):
            print(f"  - {note}")

    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")

    # Honesty rule: never claim accelerator validation we did not perform.
    if not profile.has_accelerator:
        print("[INFO] No usable accelerator was detected on this machine.")
        print("       CUDA/DirectML code paths are implemented but NOT validated here.")
    if args.json:
        _json({"checks": [{"name": n, "ok": o, "detail": d} for n, o, d in checks]})
    return 0 if failed == 0 else 1


def cmd_where(args: argparse.Namespace) -> int:
    from local_voice_companion.config.paths import describe

    payload = describe()
    if args.json:
        _json(payload)
        return 0
    for key, value in payload.items():
        print(f"{key}: {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_voice_companion",
        description="Adaptive local voice runtime (ASR + LLM + TTS).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the FastAPI runtime")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    legacy = sub.add_parser("legacy", help="run the pre-2.0 Voicebox+Ollama gateway")
    legacy.add_argument("--probe", action="store_true")
    legacy.add_argument("--auto-start", action="store_true")
    legacy.set_defaults(func=cmd_legacy)

    probe = sub.add_parser("probe", help="print hardware and provider capabilities")
    probe.add_argument("--no-audio", action="store_true")
    probe.add_argument("--no-providers", action="store_true")
    probe.set_defaults(func=cmd_probe)

    plan = sub.add_parser("plan", help="show the pipeline this machine would select")
    plan.add_argument("--policy")
    plan.add_argument("--no-audio", action="store_true")
    plan.add_argument("--alternatives", action="store_true")
    plan.set_defaults(func=cmd_plan)

    doctor = sub.add_parser("doctor", help="read-only diagnostics")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    where = sub.add_parser("where", help="show resolved filesystem layout")
    where.add_argument("--json", action="store_true")
    where.set_defaults(func=cmd_where)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
