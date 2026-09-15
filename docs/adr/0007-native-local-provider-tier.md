# ADR-0007 — A native local provider tier: faster-whisper + Kokoro

**Status:** Accepted (PHASE 2)

## Context

At the end of PHASE 1 the runtime could plan and select competently, but every
provider that could actually hear or speak was an HTTP client. The set was:

* `voicebox_asr` / `voicebox_tts` — HTTP to a Voicebox service.
* `ollama_llm` — HTTP to Ollama.
* `fake_*` — test doubles that emit plausible audio and plausible transcripts.

That meant every claim about performance was a claim about somebody else's
server, and the honest answer to "does this work offline?" was no. The largest
product gap was not architecture; it was the absence of a single real ASR and a
single real TTS inside the process.

Two constraints shaped the search:

1. The reference machine has **7.2 GB usable VRAM** and, at the time, about
   **1.7 GB free on the system disk**. A solution that required a multi-gigabyte
   PyTorch install was not installable, let alone pleasant.
2. Whatever entered the **core** distribution had to survive being shipped in a
   commercial product later. See ADR-0009.

## Decision

Two providers were added as a new `providers/local/` tier:

* `faster_whisper_cpu` — faster-whisper 1.2.1 on CTranslate2, CPU, INT8.
* `kokoro_tts_cpu` — Kokoro-82M v1.1 zh through onnxruntime, CPU, fp32.

They are ordinary providers. Nothing outside their own modules knows they exist;
they were registered through the same `register_builtin_providers` path every
other provider uses, and the selection engine has no special case for them.

**Piper was evaluated and deliberately excluded.** See ADR-0009.

### Why these two specifically

faster-whisper over ONNX/CTranslate2 rather than vanilla `whisper`:

* native CPU INT8 quantisation, official and supported;
* no PyTorch dependency, so no CUDA runtime dragged along;
* MIT licensed.

Kokoro rather than an alternative:

* 82M parameters, ~325 MB fp32 — small enough that a full download is survivable;
* v1.1 ships **103 Chinese voices**, so it is not a single-speaker curiosity;
* Apache-2.0, including the weights;
* onnxruntime again avoids PyTorch.

## What this cost

Stated plainly, because the alternative would have looked simpler:

* **Chinese G2P is now a hard dependency, and on Windows there is exactly one
  working option.** `phonemizer` fails on Windows with `espeak not installed on
  your system` even though `espeakng-loader` ships the DLL. Only `misaki` works.
  The TTS path therefore depends on a single G2P backend. See ADR-0008.
* **Neither provider streams.** Both declare `streaming=False`, which caps how
  low end-to-end latency can go and is the main thing PHASE 3 should attack.
* **Model weights are ~380 MB for Kokoro and ~148 MB for whisper-base**, fetched
  separately from the Python install. Disk planning is a real user-facing step.
* **English text is not fully synthesisable** through the misaki Chinese front
  end; unphonemisable characters become a marker rather than sound. This is a
  known limitation, not a bug that was overlooked.

## Rejected alternatives

* **PyTorch + a torch-based ASR/TTS.** Rejected on install size and licence-free
  reasons alone: it would not fit on the reference disk, and it would make the
  base install several gigabytes for users who may only need CPU.
* **`whisper.cpp` via ctypes.** Lower-level control but no Python-native
  model management, and substantially more Windows build friction for equivalent
  benefit at this stage.
* **Waiting for a streaming model (Moshi-style).** Already analysed and rejected
  for this hardware in `docs/DUPLEX_FEASIBILITY.md`: 24 GB VRAM required
  against 7.2 GB available.

## Consequences

* With the GPU disabled and the network off, `policy=cpu_only` selects both
  native providers and completes a real turn. This is asserted by a test, not
  merely demonstrated once by hand.
* Latency numbers in `docs/BENCHMARKING.md` are now **measured** rather than
  simulated for the ASR and TTS stages.
* The base install remains small: none of these packages are in
  `requirements.txt`. They live in `requirements-local-asr.txt` and
  `requirements-local-tts.txt`.
