# CAPABILITY_MATRIX.md

What the runtime can actually run **today**, as opposed to what it is designed
to accommodate. Read this before concluding that something is broken.

Legend: ✅ works · ⚠️ works with caveats · ❌ not implemented

## Providers

| Provider id | Kind | Device | Network | Licence | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `fake_asr` | ASR | CPU | no | — | ✅ | Deterministic. Returns canned text. For tests only. |
| `fake_llm` | LLM | CPU | no | — | ✅ | Deterministic token stream. For tests only. |
| `fake_tts` | TTS | CPU | no | — | ✅ | Synthesises a tone. For tests only. |
| `fake_vad` | VAD | CPU | no | — | ✅ | Deterministic speech/silence. For tests only. |
| **`faster_whisper_cpu`** | ASR | CPU | **no** | MIT | ✅ | Real local recognition. CTranslate2 INT8. Weights fetched explicitly. |
| **`kokoro_tts_cpu`** | TTS | CPU | **no** | Apache-2.0 | ✅ | Real local synthesis. 103 zh voices via `misaki` G2P. |
| `voicebox_asr` | ASR | REMOTE | **yes** | — | ⚠️ | Legacy adapter. Requires a Voicebox-compatible service. |
| `voicebox_tts` | TTS | REMOTE | **yes** | — | ⚠️ | Same requirement. |
| `ollama_llm` | LLM | CPU/GPU/REMOTE | **yes** | MIT (server) | ⚠️ | Requires a running Ollama daemon. Model weights carry their own licences. |

Licences govern **redistribution**, and they decide what may enter the core
distribution. See `docs/PROVIDER_LICENSES.md` and ADR-0009.

### The gap that was closed

**There is now a CPU-capable real ASR and real TTS in the builtin set**, and both
run entirely offline. That was the single largest product gap: every real speech
provider used to be an HTTP client, so no claim about performance was a claim
about this process.

Verified end to end as a test rather than a manual demo — see
`tests/integration/test_offline_acceptance.py` and
`tests/smoke/test_runtime_smoke.py::test_native_pair_completes_a_cpu_only_offline_turn`.

Measured on the reference machine below, GPU disabled, `policy=cpu_only`:

| Stage | Provider | Median | RTF |
|---|---|---|---|
| ASR | `faster_whisper_cpu` / `base` INT8 | 573.6 ms | 0.287 |
| TTS | `kokoro_tts_cpu` / `kokoro-v1.1-zh` | 1101.5 ms | 0.337 |
| TTFA | end to end | 2077.8 ms | — |

See `BENCHMARKING.md` for the full table with min/max, and for what these numbers
do *not* cover.

### The gaps that remain

Stated here rather than buried, because a capability matrix that only lists
capabilities is marketing:

1. **No local LLM.** This is the largest remaining hole. Offline turns use
   deterministic stub text, which means the *reasoning* in a full local turn is
   not real even though the hearing and speaking are. `fake_llm` and `ollama_llm`
   are the only options, and neither is both local and real: one is a test
   double, the other needs a daemon that is not part of this project.
2. **No streaming.** Both native providers are single-pass (`streaming=False`).
   ASR gives no partial hypotheses; TTS gives no progressive audio. This sets a
   floor on achievable latency that no amount of tuning removes — see
   `docs/DUPLEX_FEASIBILITY.md` for what genuine full duplex would require.
3. **One working Chinese G2P backend on Windows.** `misaki` is the only one;
   `phonemizer`/espeak fails on this platform even though the DLL ships. Details
   in ADR-0008.
4. **Mixed-language synthesis is degraded.** misaki's Chinese front end cannot
   phonemise English; unphonemisable characters are dropped rather than spoken.
5. **Cold start is seconds.** Loading Kokoro's 325 MB graph dominates first-turn
   latency and is not reflected in the warm numbers above.
6. **The web UI and Godot client are still on the legacy wire format** —
   untouched, working, and not yet migrated.

## Selection policies

| Policy | Status | What actually happens |
| --- | --- | --- |
| `auto` | ✅ | Probes hardware, picks a policy |
| `balanced` | ✅ | The default weighting |
| `ultra_low_latency` | ⚠️ | Reachable floor is set by single-pass ASR + TTS (see gaps above) |
| `quality` | ⚠️ | Bounded by the most capable local provider: whisper-`base` and Kokoro-82M are competent, not state of the art |
| `low_memory` | ✅ | Meaningful: Kokoro is 325 MB and whisper-`base` 148 MB, so footprint now actually differentiates |
| `cpu_only` | ✅ | **Now genuinely feasible** — selects both native providers and completes a real turn offline |
| `manual` | ✅ | Validates pinned choices without overriding them |

`cpu_only` moved from ⚠️ to ✅ because it finally has real candidates to promote.
`low_memory` became meaningful for the same reason: when every provider was
remote or fake, footprint was not a distinguishing axis.

`ultra_low_latency` and `quality` remain ⚠️ honestly — not because the weighting
is wrong, but because the best available local models bound what can be achieved.
Swapping in a larger whisper checkpoint or a heavier TTS changes this without any
engine change, which is the point of keeping them as providers.

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
| Benchmark cache | ✅ | Keyed on hardware fingerprint **and provider version**; sim vs measured is explicit |
| Model fetch + doctor | ✅ | `python lvc.py models {list,status,fetch}` and `python lvc.py doctor`; nothing downloads implicitly |
| Real local ASR | ✅ | faster-whisper INT8 on CPU |
| Real local TTS | ✅ | Kokoro-82M zh on CPU, 103 voices |
| Local LLM | ❌ | The remaining hole — see gaps above |
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
| Fingerprint | `dd77d292336c5834` (benchmark runs: `d7fc5fcfb3419e08`) |
| Services installed | `ollama`, `ffmpeg` |
| Local inference backends | faster-whisper 1.2.1 / CTranslate2 4.8.2, kokoro-onnx 0.6.1 / onnxruntime 1.30.0, misaki 0.9.4 |
| Compute capability | 7.5 (Turing — no BF16) |

### Disk, which constrained the design more than the GPU did

This machine's system drive has repeatedly sat **under 2 GB free**, sometimes as
low as 1.7 GB. A PyTorch-with-CUDA install (~5–6 GB) is therefore not merely
slow, it fails. That constraint is why the native path went through CTranslate2
and onnxruntime rather than torch, and why model weights live on a different
volume:

| Path | Contents |
| --- | --- |
| `D:\AI_Workspace\venvs\lvc` | Python environment (~527 MB) |
| `E:\AI_Workspace\local-voice-companion\models\kokoro-v1.1-zh\` | Kokoro weights (~380 MB) |
| `E:\AI_Workspace\local-voice-companion\hf\hub\` | faster-whisper weights (~148 MB) |

None of that is hardcoded — the data root is resolved by free space via
`LVC_DATA_ROOT` → D: → E: → project fallback, so this layout is a consequence of
the machine rather than an assumption in the code.
