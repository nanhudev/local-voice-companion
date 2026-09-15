# ARCHITECTURE.md

Local Voice Companion is an **adaptive local voice runtime**. It probes the
machine it is on, decides which speech recognition, language model, and speech
synthesis to use, and then runs conversations over that plan.

It is not a wrapper around a specific engine. Voicebox and Ollama are providers
here, exactly like a fake provider or a future local model is a provider.

## 1. The shape of a decision

Everything the runtime does is downstream of one pipeline:

```
HardwareProbe
     ↓
CapabilityDetection        what can this machine actually run
     ↓
ProviderDiscovery          what is installed and reachable
     ↓
CandidateGeneration        provider × model × device combinations
     ↓
Benchmark                  measured, or explicitly simulated
     ↓
PolicyEngine               what the user asked to optimise for
     ↓
PipelineSelection          one ASR + one LLM + one TTS that fit
     ↓
Runtime                    serves conversations over that plan
```

There is no path from "user opens the app" to "model runs" that skips this
pipeline. That is deliberate: a decision that cannot be inspected cannot be
debugged, and "it picked something weird" is the single most common complaint
about adaptive systems.

## 2. Layers

The package is layered, and dependencies point in one direction only.

```
api/          HTTP + WebSocket surface          depends on everything below
core/         runtime state, sessions, turn orchestration, events
selection/    candidate → constraint → score → decision
providers/    the provider contract and its implementations
  local/        native in-process inference (ASR + TTS)
hardware/     machine probing and the hardware profile
pipeline/     chunking, queues, WAV assembly
bots/         portable bot manifests
config/       typed config, paths, migration
compat/       adapters that let old code keep working
observability/ metrics
```

`core/` never imports from `api/`. `selection/` never imports a concrete
provider. `hardware/` imports nothing from the runtime. This is what makes
`FakeASR → FakeLLM → FakeTTS` a real test of the runtime rather than a mock of
it — the fakes go through the identical code path as real engines.

### The native provider tier

`providers/local/` holds providers that run inference inside this process:
`faster_whisper_cpu` and `kokoro_tts_cpu`. They reach the runtime through the
same registration path as the fakes, and the selection engine contains no
special case for them — which is the point. Adding a local engine changed
`register_builtin_providers` and nothing above it.

Three properties distinguish this tier, and all three are load-bearing:

* **Nothing downloads implicitly.** `ModelStore` raises `ModelMissing` naming
  the exact command that fixes it. A probe never opens a socket.
* **Heavy imports are lazy and per-method.** Importing `local_voice_companion`
  pulls in neither onnxruntime nor CTranslate2, so startup stays fast and a
  machine without those packages still imports cleanly.
* **Blocking inference is offloaded.** Vocoder work runs in a worker thread via
  `asyncio.to_thread`, raced against the cancellation token, so a long synthesis
  cannot stall the event loop or outlive a barge-in.

## 3. The provider contract

A provider is described by a single `ProviderDescriptor`. This is the
anti-branch contract: it exists so that no code in `core/` ever needs to know
which engine it is talking to.

```python
ProviderDescriptor(
    id="fake_tts",                    # stable, lowercase, underscores
    kind=ProviderKind.TTS,            # ASR | LLM | TTS | VAD
    display_name="Fake TTS",
    version="0.1.0",
    capabilities=...,                 # streaming, languages, sample rates
    resources=ResourceRequirements(...),  # ram_mb, vram_mb, cpu_threads, disk_mb
    quality=QualityEstimate(...),     # with a QualitySource
    devices=(DeviceKind.CPU, ...),
    requires_network=False,
    install_hint=...,
)
```

Two fields carry most of the design weight:

**`quality` comes with a `quality.source`.** The value is one of `measured`,
`offline_eval`, `curated_metadata`, `provider_reported`, `user_preference`, or
`unknown`. A number that was never measured must never be presented as though
it was. The scoring engine weights curated metadata below measured results for
exactly this reason.

**`resources` is an estimate, and is treated as one.** It feeds the planner,
which is allowed to be wrong in a recoverable direction (demoting a plan) and
not in an unrecoverable one (loading a model that does not fit).

### Lifecycle

```
DISCOVERED → AVAILABLE → LOADING → READY ⇄ BUSY
                            ↓        ↓
                          ERROR   DEGRADED
                            ↓        ↓
                       UNLOADING → UNAVAILABLE
```

`SERVING_STATES = {READY, BUSY, DEGRADED}` — only these can serve a turn.
The only legal entries into `LOADING` are from `AVAILABLE` or `ERROR`;
a provider cannot be loaded twice, and cannot be loaded before it is known to
be available. Illegal transitions raise rather than being silently coerced,
because a provider that is "kind of ready" produces bugs that look like model
failures.

## 4. Selection

### Candidates

A candidate is one concrete way to fill one stage: `(provider, model, device)`.
The LLM stage produces candidates per model; ASR and TTS per provider, since
their models are usually bundled.

### Constraints filter first

Hard constraints remove candidates before any scoring happens:

- the policy forbids the device (`cpu_only` rejects every GPU candidate)
- the resource envelope exceeds the budget
- a language is required that the provider does not declare
- the provider is unreachable

Filtering before scoring matters because scoring a candidate that cannot run
wastes time, and worse, makes the logs look like a real evaluation happened.

### Scoring

Each surviving candidate receives a `ScoreBreakdown` with named components —
latency, quality, resource fit, reliability, preference — rather than one opaque
number. The breakdown is returned to the caller, so "why was this chosen" is a
question with an answer in the response body.

### The planner demotes, it does not fail

`PipelineResourcePlanner` holds the ASR + LLM + TTS plan against the machine's
budget. When the plan does not fit, it **demotes** candidates by best
score-per-megabyte-freed — typically GPU → CPU — and re-checks, repeating until
the plan fits or nothing further can be demoted.

A plan that fits on paper but cannot be loaded is the failure mode this exists
to prevent. A slightly worse plan that runs beats an optimal plan that OOMs.

### Cache and honesty

Benchmark results are cached against a key built from the hardware fingerprint,
provider, model, and device. A cache entry records whether it was `MEASURED` or
`SIMULATED`. Simulated values are permitted — most machines have not run every
model — but they never masquerade as measurements, and the selection response
says which it used.

That is RULE 12 expressed in the type system: it is not possible to build a
response that claims an unmeasured benchmark was measured without deliberately
lying in one specific place.

### Policies

| Policy | Optimises for |
| --- | --- |
| `auto` | Whatever the hardware suggests; the default |
| `ultra_low_latency` | Time to first audio above all |
| `balanced` | The middle |
| `quality` | Output quality, latency secondary |
| `low_memory` | Smallest resident footprint |
| `cpu_only` | No GPU, for shared or headless machines |
| `manual` | The user pinned providers; the engine validates but does not override |

## 5. The turn state machine

```
IDLE → LISTENING → CAPTURING → TRANSCRIBING → THINKING → SYNTHESIZING → SPEAKING
          ↑                                                                    │
          └────────────────────────────────────────────────────────────────────┘
                                     (loop)

CANCELLING and PAUSED are reachable from any active state.
ERROR is terminal for the turn.
```

Every transition is validated. A turn that is `SPEAKING` cannot jump to
`TRANSCRIBING`; it goes through `CANCELLING` first. This is what makes barge-in
correct rather than approximate.

### Barge-in

A `CancellationToken` wraps an `asyncio.Event`. When the user speaks during
playback, the token is cancelled:

- `cancel()` is idempotent, and the **first** reason wins — so the recorded
  cause is the real one, not whichever cleanup handler ran last.
- Cancellation propagates to the synthesis worker and the audio emitter through
  the same token, so no task is orphaned.
- The turn ends in `CANCELLING`, not `ERROR`. A user interrupting is normal
  behaviour, not a fault.

### Bounded queues

Audio and text flow through queues with an explicit capacity and an explicit
overflow policy (`oldest` or `newest`). Overflow is counted and reported as
telemetry, never silently dropped. An unbounded queue converts a slow consumer
into an out-of-memory crash, and a silent drop converts it into a mystery.

### Streaming synthesis

Generation and synthesis run **concurrently**. The `AdaptiveTextChunker`
releases a phrase as soon as it is a natural boundary, the synthesis worker
picks it up immediately, and audio starts playing while the LLM is still
writing.

This means `tts.started` legitimately precedes `llm.completed`. The timeline is
not globally monotonic by design — it is causally ordered, and the derived
metrics (`llm_ttft_ms`, `tts_ttfa_ms`, `time_to_first_audio_ms`) are computed
from the causal pairs, not from the raw sequence.

## 6. Time To First Audio

**Time To First Audio is the headline metric.** Not tokens per second, not
total turn duration.

A voice assistant that answers in four seconds but starts speaking at 0.8s feels
fast. One that answers in three seconds and starts speaking at 2.9s feels
broken. The architecture is arranged to move that number: streaming generation,
early chunking, and concurrent synthesis all exist to reduce the gap between the
user finishing a sentence and hearing a reply begin.

Timeline stages, in causal order:

```
turn_started → vad_end → asr_start → asr_end
             → llm_start → llm_first_token → llm_end
             → tts_start → tts_first_audio → playback_start → playback_end
```

Derived: `asr_latency_ms`, `llm_ttft_ms`, `tts_ttfa_ms`,
`time_to_first_audio_ms`, `total_turn_ms`, and `clock_limited` when the clock
resolution is too coarse to trust the number.

## 7. Events

One wire format for everything:

```json
{
  "v": 1,
  "id": "...",
  "type": "turn.state",
  "ts": 1757...,
  "session_id": "...",
  "turn_id": "...",
  "data": {}
}
```

Event types are namespaced: `runtime.*`, `session.*`, `turn.*`, `asr.*`,
`llm.*`, `tts.*`, `playback.*`, plus `runtime.metric`, `provider.state`, and
`error`. `error` is the single deliberate exception to namespacing — it is the
catch-all bucket, and pretences about its category would be worse than an
unprefixed name.

Subscribers receive the **wire dict**, not the internal `Event` object. The
internal representation can change; the wire format is a contract with
`tests/contract`.

`EventBus` keeps a bounded ring buffer so a late subscriber can catch up
without the buffer growing with uptime.

## 8. Bots

A `BotManifest` is portable: it carries config, persona, and runtime intent,
and **no host paths**. Export from one machine, import on another, and the
manifest is re-planned against the new hardware rather than failing because a
path or a device changed.

Manifests are `extra="forbid"` — an unknown key is an error, not a silently
ignored field. A typo in a persona field should not be discovered by noticing
that the persona never applied.

YAML is emitted and parsed by a hand-written subset in `bots/yamlio.py`,
keeping the runtime dependency-free for its core contract.

## 9. Configuration

Layered: defaults → config file → environment → explicit override. Every model
is Pydantic v2 with a schema version, and a v1 config is migrated forward on
load. Migration is one-way and idempotent.

Secrets are referenced **by environment variable name only**. Never a value.
`redacted()` and `_scrub()` replace any key containing `api_key`, `token`, or
`secret` with `<set>`/`<unset>`, so a config dump is safe to paste into an
issue.

## 10. Data layout

```
LVC_DATA_ROOT            (env, explicit)
  → D:\AI_Workspace\local-voice-companion   (preferred)
    → project directory                      (fallback)
```

Resolved into a `Layout` with `bots/`, `models/`, `cache/`, `logs/`, and
`benchmarks/` subdirectories. The resolver picks the candidate with the most
free space and records **which one it chose and why** in the returned note.

Models are tens of gigabytes. Putting them on a full system drive is the most
common way a local-first tool becomes unusable, so the default assumption is
that they belong somewhere else.

## 11. Compatibility

The runtime coexists with the original code rather than replacing it.

- `app.py` is a 141-line shim: three pure helpers retained, everything else
  forwarded to `serve` or `legacy`.
- `compat/voicebox.py` exposes the old Voicebox service as an ordinary
  provider. It is optional — disable it and nothing breaks.
- Ollama is a provider too. It is a good choice on some machines; it is not
  the architecture.
- The old HTTP shape is preserved so `web/` and `godot/` keep working while the
  new API is adopted at whatever pace is convenient.

## 12. What is deliberately absent

No voice cloning. No accounts. No cloud sync. No database. No RAG. No Live2D.
No UI redesign.

Each of these is defensible in isolation and destructive in aggregate: they
would each add a dependency edge into `core/`, and the value of this
architecture is entirely in which edges do not exist.
