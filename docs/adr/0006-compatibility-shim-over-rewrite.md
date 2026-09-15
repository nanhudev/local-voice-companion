# ADR-0006 — Shim `app.py`, do not rewrite the client surface

**Status:** Accepted (PHASE 1)

## Context

`app.py` was 668 lines mixing argparse, config loading, text cleanup, sentence
chunking, WAV encoding, the HTTP server, ASR/TTS proxying, and relay worker
bootstrapping. It is the file the README tells people to run. Three clients
depend on the surface it exposes: `web/index.html`, `godot/voice_companion.gd`,
and whatever the user has wired up themselves.

The obvious move — delete it and point the README at the new package — breaks
all three clients, all existing configs, and every setup script at the same
moment. That is not a migration; it is a rewrite wearing a migration's
paperwork, and RULE 13 forbids it.

## Decision

Three separate decisions, each narrow:

**1. `app.py` becomes a 141-line shim.** It keeps the three pure helpers the old
tests covered directly (`wav_bytes`, `ready_sentences`, `clean_model_text`) and
forwards everything else:

```python
python app.py --runtime ...   →  local_voice_companion.__main__.serve
python app.py ...             →  local_voice_companion.__main__.legacy
```

Arguments are forwarded **verbatim** for `--runtime`. The `serve` subcommand
already accepts `--host`/`--port`/`--log-level`, and re-parsing them in the shim
is how `--port 18771` silently became `unrecognized arguments: port 18771`.

**2. The old HTTP surface is preserved.** The new API is additive under
`/api/v1`. `web/` and `godot/` continue to work unmodified.

**3. The original file is preserved.** At
`.workbuddy/tmp/app.py.original-backup` until the legacy path is fully ported.

## Consequences

**Good:**

- `python app.py` keeps working, so existing users upgrade without noticing.
- The old helpers remain importable, so `tests/legacy` still exercises them and
  their behaviour is pinned while the new code is built.
- Both clients keep working, which means the migration can proceed at whatever
  pace is convenient rather than all at once.
- The shim is small enough to read in full, which is the point: a 141-line
  forwarding layer is auditable, a 668-line monolith is not.

**Costs:**

- **Two entry points exist at once.** `--runtime` and the legacy path diverge,
  and a change to shared text handling must be made in both places until the
  legacy path is retired.
- **The backup file lives outside version control.** It will be lost if the
  working directory is cleaned. This is intentional — it is a temporary artefact
  and committing it would create a second source of truth — but it means the
  legacy behaviour's only permanently recorded form is `tests/legacy`.
- **`--runtime` is a mode switch rather than the default.** Making the new
  runtime the default would be cleaner and would break every existing script.
  That flip is a deliberate, separate decision for a later phase.

## Alternatives rejected

**Delete `app.py` and update the README.** Rejected: breaks `web/`, `godot/`,
all setup scripts, and every existing config simultaneously.

**Leave `app.py` untouched and add the package alongside.** Rejected: the
monolith would keep accumulating changes as bugs are fixed in the new code, and
the two would drift apart invisibly.

**Make `app.py` import the package at module scope and keep its own HTTP
server.** Rejected: it would preserve the old surface at the cost of keeping the
old server alive, which means two HTTP implementations to maintain and two sets
of bugs.
