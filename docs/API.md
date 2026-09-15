# API.md

Base path: `/api/v1`. Versioned at the path so a breaking change is a new
prefix rather than a silent break for `web/` and `godot/`.

All request and response bodies are JSON. Errors use one envelope:

```json
{
  "error": {
    "code": "configuration_error",
    "message": "human readable",
    "detail": {}
  }
}
```

`code` is the stable part. Branch on it, not on `message`.

## Health

### `GET /healthz`
Liveness. Always 200 while the process is up — this is what a supervisor polls.

```json
{"status": "ok", "schema_version": 2, "event_schema": 1}
```

### `GET /readyz`
Readiness. **503 until a pipeline has been prepared.** A runtime that is up but
has not decided what to run cannot serve a turn, and reporting 200 here would
make that indistinguishable from healthy.

## System

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/system/profile` | The `HardwareProfile` of this machine |
| `GET` | `/system/runtime` | Runtime state: prepared pipeline, loaded providers |
| `GET` | `/system/config` | Effective config, **secrets redacted** |
| `PUT` | `/system/config` | Replace config; re-validates and re-plans |
| `GET` | `/system/policies` | The seven selection policies with weights and summaries |

`/system/profile` reports what was probed, not what was assumed:

```json
{
  "cpu": {"threads": 12, "model": "..."},
  "memory": {"total_mb": 16280},
  "gpus": [{"name": "NVIDIA GeForce RTX 2070", "vram_total_mb": 8192,
            "vram_free_mb": 7235, "driver": "581.29"}],
  "accelerators": ["cuda", "directml"],
  "services": {"installed": ["ollama", "ffmpeg"]},
  "fingerprint": "dd77d292336c5834"
}
```

`fingerprint` is what benchmark cache entries are keyed against.

`/system/policies` entries are keyed by `id`:

```json
{"id": "balanced", "weights": {...}, "summary": "..."}
```

## Providers

### `GET /providers`
Every registered provider. Ids use underscores, not hyphens.

```json
{"providers": [
  {"id": "fake_asr", "kind": "asr", "display_name": "Fake ASR",
   "state": "available", "devices": ["cpu"], ...}
]}
```

Each entry is the full `ProviderDescriptor` — 20 keys, pinned by
`tests/contract/test_contracts.py::TestProviderDescriptorContract`. Lists are
always lists, never tuples, on the wire.

### `GET /models`
Models discovered across providers.

### `GET /voices`
Voices across providers that declare them.

## Selection

### `POST /selection/recommend`

```json
{"policy": "balanced", "include_alternatives": true}
```

Response:

```json
{
  "effective_policy": "balanced",
  "plan": {
    "assignments": [
      {"kind": "asr", "id": "...", "provider_id": "fake_asr",
       "model_id": null, "device": "cpu"},
      {"kind": "llm", "id": "...", "provider_id": "fake_llm",
       "model_id": null, "device": "cpu"},
      {"kind": "tts", "id": "...", "provider_id": "fake_tts",
       "model_id": null, "device": "cpu"}
    ]
  },
  "scores": [...],
  "alternatives": [...],
  "notes": [...]
}
```

Three things worth knowing:

- The key is `assignments`, not `stages`, and each assignment is **flat** —
  there is no nested `candidate` object.
- **`vad` may appear in `assignments`.** If a VAD provider is registered, it
  joins the plan; asserting exactly `{asr, llm, tts}` is wrong.
- The response key is `effective_policy`. `detailed_policy` in the request is a
  *request*; the response reports what was actually applied, which can differ
  when the requested policy cannot be satisfied on this hardware.

`alternatives` is omitted when `include_alternatives` is false.

### `POST /benchmark`
Runs or records a benchmark for a provider/model/device.

A result carries its provenance. `BenchmarkSource.SIMULATED` and
`BenchmarkSource.MEASURED` are distinct values, and simulated results are never
returned as measured ones. On a machine that has never run the model, the
honest answer is `SIMULATED`.

## Bots

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/bots` | List |
| `POST` | `/bots` | Create → 201; duplicate id → 409 |
| `GET` | `/bots/{bot_id}` | Fetch |
| `PUT` | `/bots/{bot_id}` | Partial update, deep-merges nested objects |
| `DELETE` | `/bots/{bot_id}` | Delete |
| `GET` | `/bots/{bot_id}/export` | Portable manifest, no host paths |
| `POST` | `/bots/import` | Create from manifest → 201 |
| `GET` | `/bots/{bot_id}/plan` | The plan this bot resolves to on **this** machine |

`bot_id` is normalised to lowercase with surrounding whitespace stripped, so
`"UPPER"` is accepted and stored as `upper`. Genuinely invalid ids
(`-leading`, `_leading`, `a/b`, `a\b`, `a.b`) are rejected with 400.

Re-importing an existing id returns **400** (`configuration_error`), not 409.
Pass `?overwrite=true` to replace, which returns 201.

A manifest forbids unknown keys. A typo in a persona field is an error rather
than a field that silently never applies.

## Sessions

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/sessions` | Open a session → 201 |
| `GET` | `/sessions` | List |
| `GET` | `/sessions/{session_id}` | Fetch, including current turn state |
| `DELETE` | `/sessions/{session_id}` | Close |
| `POST` | `/sessions/{session_id}/turns` | Run a turn |
| `POST` | `/sessions/{session_id}/cancel` | Cancel the active turn (barge-in) |

A session is bound to a bot and, on open, resolves that bot against this
machine's plan. Opening a session on hardware where no viable plan exists fails
loudly here rather than halfway through the first turn.

## Events

### `GET /events`
Server-sent stream of every runtime event. Useful for a terminal tail and for
debugging a turn that behaved unexpectedly.

### `GET /metrics`
Counters and histograms, including per-stage latency distributions and queue
overflow counts.

### `WS /sessions/{session_id}/stream`
The interactive channel. Client sends audio chunks and control messages; the
server sends the event stream plus audio frames.

Barge-in: the client sends a cancel message, which cancels the session's
`CancellationToken`. The active turn moves to `CANCELLING` — not `ERROR` — and
the next turn begins from `LISTENING`.

## OpenAI-compatible endpoints

| Method | Path |
| --- | --- |
| `POST` | `/v1/audio/speech` |
| `POST` | `/v1/audio/transcriptions` |

These exist outside `/api/v1` because they mirror an external specification
rather than this runtime's own shape. They are a compatibility surface: useful
for pointing an existing client at a local TTS/ASR, and explicitly **not** the
primary API.

## Event wire format

```json
{
  "v": 1, "id": "...", "type": "turn.state", "ts": 1757...,
  "session_id": "...", "turn_id": "...", "data": {}
}
```

| Type | Emitted when |
| --- | --- |
| `runtime.ready` | Runtime finished preparing |
| `session.opened` / `session.closed` | Session lifecycle |
| `turn.started` | Turn accepted |
| `turn.state` | Every state transition |
| `turn.cancelled` | Turn ended via cancellation |
| `turn.completed` | Turn finished normally |
| `asr.partial` / `asr.final` | Recognition output |
| `llm.started` | Generation began |
| `llm.delta` | Token or chunk produced |
| `llm.completed` | Generation finished |
| `tts.started` | First synthesis request issued |
| `tts.audio` | Audio chunk produced |
| `tts.completed` | Synthesis finished |
| `playback.started` | First frame delivered |
| `playback.finished` | Playback drained |
| `runtime.metric` | A measurement was recorded |
| `provider.state` | Provider lifecycle change |
| `error` | Something failed |

`turn.completed` is **not** the last event of a turn — `runtime.metric` events
follow it. A test asserting "the last event is turn.completed" is asserting the
wrong thing.

Ordering is causal, not globally monotonic: `tts.started` legitimately precedes
`llm.completed` because synthesis streams concurrently with generation.

`error` is the one type without a namespace prefix. It is the catch-all bucket,
and an invented category name would be less honest than its absence.

All 20 types are enumerable at runtime via
`local_voice_companion.core.events.ALL_EVENT_TYPES`, and
`tests/contract/test_contracts.py` pins both the list and the namespacing rule.
