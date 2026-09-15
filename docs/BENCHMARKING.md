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
hardware fingerprint × provider id × model id × device
```

The hardware fingerprint is a stable hash over CPU model, thread count, memory,
GPU model, VRAM, and driver version. Change a GPU and every cached entry for
that device is invalidated automatically, because the key no longer matches.

This is why `fingerprint` appears in `GET /api/v1/system/profile`: every cached
number is traceable to the machine that produced it.

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
POST /api/v1/benchmark
```

On a machine with no local backend, the honest result is `SIMULATED` — the
runtime has the provider's declared resources and no observation of its own.
That is a correct answer. The incorrect answers are refusing to return anything
and returning a fabricated `MEASURED`.

## Current state

**The instrumentation is complete and the measurements are not.** Every stage is
timed on every turn; no real model has been timed on this hardware, because none
is installed (see `CAPABILITY_MATRIX.md`).

Consequently the selection engine currently scores against largely simulated
input. The structured breakdown is real; the numbers inside it are informed
estimates. Replacing them is the first task of PHASE 2.

Anyone reading a selection result should check the source field before treating
the ranking as evidence.
