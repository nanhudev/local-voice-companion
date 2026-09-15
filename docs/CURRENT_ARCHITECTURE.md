# CURRENT_ARCHITECTURE.md — PHASE 0 Repository Audit

Snapshot taken **before** any PHASE 1 change, at commit `9e94969`
("Make English the default README language"). Every claim below was read out
of the code, not inferred from the README.

## 1. What the repository actually contained

| Path | Lines | Role |
| --- | --- | --- |
| `app.py` | 668 | Everything: argparse, config load, text cleanup, sentence chunking, WAV encoding, HTTP server, ASR/TTS proxy, relay worker bootstrap |
| `backends.py` | 92 | Backend discovery helpers (probe Ollama / speech service) |
| `windows_worker.py` | 305 | Standalone GPU relay worker for split-machine setups |
| `doctor.py` | 52 | Environment diagnostics CLI |
| `smoke_test.py` | 59 | Single end-to-end script, prints PASS/FAIL, no assertions |
| `config.example.json` | 28 | Flat v1 config |
| `web/` | 90 | `index.html` + 18-line `app.js` + 9-line CSS |
| `godot/` | 202 | `voice_companion.gd` + a scene, calls the HTTP API |
| `tests/` | — | Absent |

Total application code ≈ **1,500 lines**, of which 45% lived in one file.

## 2. What worked

Verified by running it, not by reading it:

- **The server boots.** `app.py` starts an HTTP server on `127.0.0.1:17831`.
- **The browser UI functions.** `web/index.html` opens a session, streams
  microphone audio, plays returned audio, and renders settings.
- **Ollama integration works** when a model is present — `/api/chat` with
  streaming NDJSON, correctly reassembled.
- **The Voicebox-compatible path works** when that service is running —
  `/health`, `/profiles`, `/transcribe`, `/generate/stream`.
- **The Godot sample compiles** and speaks to the same HTTP surface.
- **`doctor.py` gives honest diagnostics** — it reports a missing backend
  rather than failing silently.

## 3. What was coupled

This is the list that justified PHASE 1. Each item is a place where changing
one thing forced a change somewhere unrelated.

### 3.1 Every decision was a string comparison

```python
if backend == "voicebox": ...
if model_name.startswith("kokoro"): ...
```

There was no representation of *what a backend can do* — only *what it is
called*. Adding a second TTS engine meant editing conditionals inside the
turn loop. A provider could not describe its own device, sample rate, voice
list, or streaming behaviour, so the calling code had to know it instead.

### 3.2 The turn loop was the only orchestration primitive

ASR → LLM → TTS ran inside one linear function with no state machine, no
cancellation token, and no way to observe progress. Consequences:

- **Barge-in could not be expressed.** A user speaking mid-reply had no path
  to cancel synthesis; the loop was already past the decision point.
- **No queue bounds.** Audio and text chunks accumulated in unbounded lists,
  so a fast producer and slow consumer grew memory without limit.
- **No latency instrumentation.** There was no timestamp anywhere in the turn,
  so "why does it feel slow" was unanswerable.

### 3.3 Hardware was assumed, never probed

GPU presence, VRAM, thread count and available RAM were configuration values a
human typed. `windows_worker.py` hard-coded CUDA assumptions. On a machine with
no GPU the failure mode was a runtime error deep inside the model call rather
than a declined plan.

### 3.4 Configuration had no schema

`config.json` was read with `.get()` at each use site, so a typo produced a
silent default rather than an error, and there was no version field. Secrets
were interpolated directly into requests and appeared in debug output.

### 3.5 The API had no contract

Routes returned ad-hoc dicts shaped at each call site. The Godot client and the
browser UI each depended on a slightly different implicit shape, and nothing in
the repository could tell you when that shape changed.

## 4. Migration risks identified

| Risk | Why it mattered |
| --- | --- |
| `app.py` executed on import | Nothing could be tested without starting a server; the old helpers had no home |
| Config had no version field | A schema change would silently misread existing user configs |
| Voicebox was load-bearing | Deleting it would break every existing deployment; it had to become *optional* first |
| Ollama was load-bearing | Same problem, and Ollama is a valid choice, so "remove" was wrong — "demote" was right |
| `windows_worker.py` spoke a private protocol | Rewriting it in PHASE 1 would have made the migration un-reviewable |
| No tests existed | Every refactor was unverifiable, so correctness had to be established *before* moving code |
| `web/` and `godot/` consumed the old HTTP shape | The new API had to be additive first, or both clients would break simultaneously |

## 5. What PHASE 1 changed, and what it deliberately did not

**Changed:** the architecture. A modular package, typed config with migration,
a provider contract, a selection engine, a turn state machine, bounded queues,
an event stream, a bot manifest, and a test suite were added alongside the old
code.

**Not changed:** `web/`, `godot/`, `windows_worker.py`, `backends.py`,
`doctor.py`, `smoke_test.py`, and the installation scripts. They continue to
work against the same files they always did. `app.py` keeps its three pure
helpers and forwards to the appropriate entry point.

**Not attempted:** downloading models, rewriting the legacy paths, UI work,
voice cloning, accounts, cloud sync, databases, RAG, or Live2D.

The guiding constraint was RULE 13: *keep the old thing working*. A migration
that breaks the running deployment is not a migration, it is a rewrite with a
migration's paperwork.
