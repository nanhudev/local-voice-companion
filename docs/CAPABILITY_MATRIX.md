# CAPABILITY_MATRIX.md

What the runtime can actually run **today**, as opposed to what it is designed
to accommodate. Read this before concluding that something is broken.

Legend: ✅ works · ⚠️ works with caveats · ❌ not implemented

## Providers

| Provider id | Kind | Device | Network | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| `fake_asr` | ASR | CPU | no | ✅ | Deterministic. Returns canned text. For tests. |
| `fake_llm` | LLM | CPU | no | ✅ | Deterministic token stream. For tests. |
| `fake_tts` | TTS | CPU | no | ✅ | Synthesises a tone. For tests. |
| `fake_vad` | VAD | CPU | no | ✅ | Deterministic speech/silence. For tests. |
| `voicebox_asr` | ASR | CPU | **yes** | ⚠️ | Legacy adapter. Requires a Voicebox-compatible service on another process or host. |
| `voicebox_tts` | TTS | CPU/GPU | **yes** | ⚠️ | Same requirement. |
| `ollama_llm` | LLM | CPU/GPU | **yes** | ⚠️ | Requires a running Ollama daemon. |

### The gap this table is built to make visible

**There is no CPU-capable real ASR or TTS implementation in the builtin set.**

Every real speech provider in this table is a *remote compatibility adapter*: it
speaks HTTP to a service that owns the actual model. `voicebox_asr` and
`voicebox_tts` are marked `requires_network=True` because from this runtime's
point of view they are network calls — even when that network is loopback and
the service is on the same machine.

The practical consequences:

1. **`cpu_only` produces a degraded plan.** Excluding GPU removes most real
   options, and the builtin set has nothing to promote in their place. The
   planner does not fail — it returns its best remaining plan and says so in
   `notes` — but "best remaining" is not "good".
2. **A machine with no Voicebox and no Ollama has no real conversation path.**
   It can run the fake pipeline end to end, which is precisely what makes the
   runtime testable, but it cannot actually hear or speak.
3. **`@pytest.mark.hardware` tests skip here.** That is a truthful report of
   missing local backends, not a passing test. Per RULE 12, a skip is the
   correct outcome; a fabricated pass is not.

Closing this gap is the top item of the next milestone — see `ROADMAP.md`.
Shipping a selection engine that can only select between "fake" and "remote"
is a skeleton, and it would be dishonest to describe it as a working local
voice runtime.

## Selection policies

| Policy | Status | What actually happens |
| --- | --- | --- |
| `auto` | ✅ | Probes hardware, picks a policy |
| `balanced` | ✅ | The default weighting |
| `ultra_low_latency` | ⚠️ | Weights are applied correctly, but with only remote providers available the achievable latency floor is the remote service's |
| `quality` | ⚠️ | Same — quality is bounded by the most capable reachable provider |
| `low_memory` | ✅ | Works; favouring small footprints is meaningful even with remote providers |
| `cpu_only` | ⚠️ | Produces a valid but degraded plan (see above) |
| `manual` | ✅ | Validates pinned choices without overriding them |

The ⚠️ entries are not bugs. The policy machinery is correct; the *inputs* are
thin. A scoring engine with two mediocre options produces a mediocre
recommendation, and that is the right answer to give.

## Runtime features

| Feature | Status | Notes |
| --- | --- | --- |
| Turn state machine | ✅ | 10 states, validated transitions |
| Barge-in | ✅ | `CancellationToken`, idempotent, first reason wins |
| Bounded queues | ✅ | `oldest`/`newest` drop policies, overflow counted |
| Streaming chunking | ✅ | Language-aware phrase boundaries |
| Concurrent synthesis | ✅ | Synthesis overlaps generation |
| Time To First Audio | ✅ | Headline metric, `clock_limited` when resolution is too coarse |
| Event stream | ✅ | 20 types, one wire format, versioned |
| Bot manifests | ✅ | Portable, no host paths |
| Config migration | ✅ | v1 → v2, one-way, idempotent |
| Hardware probing | ✅ | CPU, RAM, GPU, VRAM, accelerators, services |
| Benchmark cache | ✅ | Keyed on hardware fingerprint; sim vs measured is explicit |
| FastAPI REST + WS | ✅ | 31 route handlers |
| Web UI | ❌ | Untouched. The new API is additive; the old UI still uses the old shape |
| Godot client | ❌ | Untouched, same reason |
| Relay worker | ❌ | `windows_worker.py` unmodified in PHASE 1 |

## Hardware used for verification

Real numbers from the machine these tests ran on. Recorded so that no claim of
hardware validation rests on a machine nobody can inspect.

| | |
| --- | --- |
| CPU | 12 threads |
| RAM | 16280 MB |
| GPU | NVIDIA GeForce RTX 2070, 8192 MB total / ≈7235 MB free |
| Driver | 581.29 |
| Accelerators | `cuda`, `directml` |
| Fingerprint | `dd77d292336c5834` |
| Services installed | `ollama`, `ffmpeg` |
| Local inference backends | **none** |

That last row is the important one. This machine has a capable GPU and no
model runtime installed, which is exactly why the runtime is designed to report
an honest degraded plan rather than to assume capability it cannot confirm.
