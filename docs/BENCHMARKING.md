# BENCHMARKING.md

How the runtime measures a machine, and how it avoids inventing numbers.

## The honesty contract

A benchmark result carries a `BenchmarkSource`:

| Source | Meaning |
| --- | --- |
| `MEASURED` | Observed on this machine, this run |
| `SIMULATED` | Derived from declared resources and curated metadata |

**A simulated value is never returned as measured.** This is enforced by the
type, not by discipline: the two are distinct enum values and the API reports
which one produced the selection.

This matters more than it sounds. Most machines have not run most models, so an
adaptive runtime is *always* partly guessing. The failure mode to avoid is not
guessing — it is guessing silently, so that the user cannot tell a measured
number from an invented one and therefore cannot tell when the guess is wrong.

## Cache key

```
hardware fingerprint × provider id × model id × device × provider version
```

The hardware fingerprint is a stable hash over CPU model, thread count, memory,
GPU model, VRAM, and driver version. Change a GPU and every cached entry for
that device is invalidated automatically, because the key no longer matches.

This is why `fingerprint` appears in `GET /api/v1/system/profile`: every cached
number is traceable to the machine that produced it.

**The version term was added after it was noticed to be missing.** A benchmark
measures one specific build, but the key ignored which build. Providers that
reported a hardcoded `"1.0.0"` therefore made invalidation *look* implemented
while guaranteeing that an upgraded engine would keep serving the previous
engine's numbers. The native providers now report their real installed version —
`faster-whisper` from distribution metadata, and Kokoro as a composite of
`kokoro-onnx`, `onnxruntime` and `misaki`, since synthesis depends on all three.
Reads go through `importlib.metadata`, which does not import the heavy package
and costs ~1–6 ms, so `descriptor()` stays cheap enough to call in the selector's
hot path.

## What is measured

Per stage, from the turn timeline:

| Metric | From | Meaning |
| --- | --- | --- |
| `asr_latency_ms` | `asr_start` → `asr_end` | Recognition time |
| `llm_ttft_ms` | `llm_start` → `llm_first_token` | Time to first token |
| `tts_ttfa_ms` | `tts_start` → `tts_first_audio` | Time to first audio frame |
| `time_to_first_audio_ms` | `vad_end` → `playback_start` | **The headline number** |
| `total_turn_ms` | `turn_started` → `playback_end` | Whole turn |

## Clock resolution

`time.perf_counter` is used, not `time.monotonic`.

On Windows, `time.monotonic` advances in steps of roughly 15.6 ms. A 40 ms
synthesis measurement taken with that clock can be off by nearly 40% in either
direction, and it would still print as a confident-looking number.

The turn timeline sets `clock_limited` when the recorded resolution is too coarse
for the reported value to be trustworthy. **A limited number is reported as
limited.** Presenting a three-significant-figure latency derived from a
15.6 ms tick would be a lie told with decimals.

## Running a benchmark

```
lvc benchmark --policy cpu_only            # human-readable
lvc benchmark --policy cpu_only --json     # machine-readable
```

Each stage is run once as warmup (loading weights, priming caches) and then
`runs` times, defaulting to three. The reported value is the **median**, with
min and max alongside — the median because outlier suppression matters on a
shared OS, and min/max because a median alone hides jitter that a user will
still hear.

On a machine with no local backend, the honest result is `SIMULATED` — the
runtime has the provider's declared resources and no observation of its own.
That is a correct answer. The incorrect answers are refusing to return anything
and returning a fabricated `MEASURED`.

## Current state — measured

The instrumentation is complete **and** real models are now timed on real
hardware. Both stages below are `MEASURED`, not estimated.

### Reference machine

| | |
|---|---|
| CPU | 12 threads |
| Memory | 16,280 MB |
| GPU | RTX 2070, 8,192 MB (disabled for these runs) |
| Accelerators | `cuda`, `directml` |
| Fingerprint | `d7fc5fcfb3419e08` |
| Disk caution | system drive was under 2 GB free; models live on another volume |

### Results — `policy=cpu_only`, GPU disabled, 1 warmup + 3 runs

| Stage | Provider | Model | Median | Min | Max | RTF | Source |
|---|---|---|---|---|---|---|---|
| ASR | `faster_whisper_cpu` | `base`, INT8 | **573.6 ms** | 567.8 | 581.3 | 0.287 | measured |
| TTS | `kokoro_tts_cpu` | `kokoro-v1.1-zh` | **1101.5 ms** | 1078.8 | 1110.8 | 0.337 | measured |
| TTFA | end to end | — | **2077.8 ms** | — | — | — | measured |

RTF is *synthesis/recognition time ÷ produced audio duration*, so **below 1.0
means faster than real time** — both stages comfortably clear that, which is what
makes a CPU-only conversational loop practical rather than merely possible.

**What the numbers do and do not cover.** They are single-turn, single-speaker,
short-utterance measurements on one machine. They say nothing about sustained
load about behaviour with concurrent requests. The 2.08 s TTFA includes load-free
steady-state inference; a cold start adds model loading (several seconds for
Kokoro's 325 MB graph) and is not represented here.

Numbers in this table come from a generated report, not from memory. Refresh it
by running the command above.

## Gaps still open

* **No LLM stage is measured.** Offline the honest default is deterministic stub
  text, which has no latency worth reporting. A measured LLM number requires a
  local LLM that is actually installed, and none is — see
  `CAPABILITY_MATRIX.md`.
* **Cold-start cost is not benchmarked.** Only warm-turn latency is.
* **No streaming.** Both native providers are single-pass by design
  (ADR-0007), so TTFA cannot come below a full ASR-plus-synthesis pass yet.
