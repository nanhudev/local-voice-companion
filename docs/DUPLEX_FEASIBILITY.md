# DUPLEX_FEASIBILITY.md

Assessment of GPT-style "listen while speaking" realtime voice, and whether it
can be built locally on this hardware.

**Verdict: feasible as a full-duplex *pipeline*, not as a full-duplex *model*.**
The distinction is the whole document. A locally-hosted Moshi or PersonaPlex
class model does not fit in 8 GB of VRAM. A duplex-*orchestrated* cascade does,
and it delivers most of what people actually notice about GPT-Live.

This is a research and design document. Nothing here is implemented yet.

## 1. What GPT-Live actually does

Three architectures are worth separating, because they are routinely conflated:

| Architecture | How it works | Example |
| --- | --- | --- |
| **Cascaded** | STT → LLM → TTS, one direction at a time | Original ChatGPT Voice |
| **Turn-based single model** | One model does audio-in and audio-out, still in discrete turns | ChatGPT Advanced Voice Mode |
| **Full-duplex** | Continuously processes input *while* generating output | GPT-Live-1 |

GPT-Live's own description of the change is precise: the model "continuously
processes input while generating output" and "makes interaction decisions many
times per second: whether to speak, continue listening, pause, interrupt, or
invoke a tool."

The concrete user-visible differences, in order of how much they matter:

1. **Pauses stop being turn boundaries.** A cascaded system infers "user has
   finished" from a silence timer — it only knows that *sound stopped*, not that
   the *thought* finished. That is why these systems either interrupt a thinking
   pause or leave dead air. A duplex model can reason about whether you are done.
   Speak (the language-learning app) reports an ~80% reduction in interruptions
   during thinking pauses after moving to GPT-Live.
2. **Backchanneling.** "mhmm", "yeah", "right" *while* you are still talking.
   Requires generating output while input continues.
3. **Graceful interruption.** Not just cancelling playback, but truncating the
   conversation memory to what the user actually heard, so the model's next
   reply is coherent with what you did and did not hear.
4. **Overlap tolerance.** A cough or a background voice gets ignored rather than
   becoming a turn.

Note that (1) and (4) are **decisions**, and (3) is mostly **bookkeeping**. Only
(2) strictly requires simultaneous generation. That decomposition is what makes a
local plan possible at all.

## 2. The local full-duplex models, and why they do not fit

Two open models genuinely do full duplex at the model level:

| Model | Params | VRAM | Latency | License |
| --- | --- | --- | --- | --- |
| **Moshi** (Kyutai) | 7B + Mimi codec | **24 GB** (PyTorch, no quantization); Rust/Candle q8 path is lower | 160 ms theoretical, ~200 ms practical on an L4 | MIT code / CC-BY-4.0 weights |
| **PersonaPlex** (NVIDIA) | ~7B, Moshi-based | **~18 GB** loaded; `--cpu-offload` supported | low-latency, full-duplex | MIT code, NVIDIA Open Model License |

This machine has an **RTX 2070 with 8192 MiB total, ~7248 MiB free**, compute
capability **7.5** (Turing).

| Requirement | Available | Gap |
| --- | --- | --- |
| Moshi PyTorch | 24 GB VRAM | **~3× short** |
| PersonaPlex | ~18 GB VRAM | **~2.2× short** |

Both are out of reach. `--cpu-offload` exists for PersonaPlex but offloading a
7B duplex model to CPU does not produce conversational latency on a 12-thread
desktop — it produces a model that runs, which is a different claim.

Two further blockers, independent of VRAM:

- **Turing (7.5) has no BF16.** Moshi's checkpoints are bf16. Emulating bf16 on
  Turing is slower than fp16 and not a supported path.
- **PersonaPlex is Linux-only in practice** — it needs `libopus-dev` and builds
  Moshi from source. This is a Windows machine.

So: **a local full-duplex model is not feasible here, and the honest report is
that it is not feasible**, not that it is "challenging". RULE 12 applies to
model capability the same way it applies to test results.

## 3. What *is* feasible: duplex orchestration over a cascade

Here is the key observation. Look again at what GPT-Live's own documentation says
it delegates:

> GPT-Live is a voice layer, not a frontier reasoning model. It delegates
> reasoning and tool calls to a backend text model.

OpenAI's flagship duplex voice model is **itself a cascade** — a duplex voice
layer in front of a separate reasoning model. Full duplex at the interaction
layer does not require full duplex at the reasoning layer.

That is the gap this machine can close.

### The architecture

```
  ┌──────────────────── always-on input path ─────────────────────┐
  │  mic → AEC → VAD → streaming-ASR partials → turn/pause manager │
  └───────────────────────────┬───────────────────────────────────┘
                              │  decisions many times per second
                   ┌──────────▼──────────┐
                   │  interaction layer  │  ← the thing that is duplex
                   │  (arbitration point)│
                   └──────────┬──────────┘
                              │
  ┌───────────────────────────▼───────────────────────────────────┐
  │  speaker ← playback ← streaming-TTS ← chunker ← streaming-LLM  │
  └───────────────────────────────────────────────────────────────┘
```

The **interaction layer** is the whole point: a single arbitration point where
"should I keep speaking", "is the user finishing a thought or just pausing", and
"should I backchannel" are decided. Duplex-ness lives here, not in the model.

**What PHASE 1 already provides** — this is not a rewrite:

| Component needed | Exists today |
| --- | --- |
| Interrupt mid-speech | `CancellationToken`, `CANCELLING` state |
| Bounded buffers under continuous streaming | `BoundedQueue` with drop policies |
| Incremental text release | `AdaptiveTextChunker` |
| Concurrent synthesis and generation | already the case — `tts.started` precedes `llm.completed` |
| Per-stage latency accounting | `TurnTimeline`, `time_to_first_audio_ms` |
| Partial ASR events | `asr.partial` event type declared, **no provider emits it** |
| "Speaking while listening" state | **does not exist** — states are exclusive |

So the skeleton is right and the gap is specific: a streaming ASR that emits
partials, a non-exclusive speaking/listening state, and the interaction layer
that arbitrates between them.

### Hardware budget

A cascade has to share 7.2 GB of free VRAM, or run partly on CPU. Realistic
options for this machine:

| Component | Choice | Device | VRAM |
| --- | --- | --- | --- |
| ASR | `faster-whisper` base/small | GPU (CUDA) | ~0.5–1 GB |
| ASR (alt) | `sherpa-onnx` streaming Zipformer | CPU | 0 — ~50 MB RAM |
| LLM | 4B-class quantized (Q4) | GPU | ~2.5–3 GB |
| LLM (alt) | 1–3B quantized | GPU | ~1–2 GB |
| TTS | Piper | **CPU** | 0 — ~30 MB RAM |
| TTS (alt) | Kokoro-82M ONNX | GPU or CPU | ~0.3–0.5 GB / ~1 GB RAM |
| VAD | Silero | CPU | ~5 MB RAM |

Budget: **~3.5–4.5 GB VRAM** for ASR + LLM, leaving headroom. TTS on CPU is
correct here — Piper synthesises ~10 words in 100–150 ms on CPU, which is well
within the latency budget, and moving it off the GPU frees VRAM for the LLM.
Kokoro on GPU is 50–100 ms/sentence versus 200–500 ms on CPU, a real gain in
quality per millisecond if VRAM allows.

Reference latencies for a GPU cascade on comparable hardware: ~0.5–1.5 s ASR,
~0.3–2 s LLM first token, ~0.1–0.3 s TTS. With streaming everywhere and TTS
overlapping generation, a sub-second time-to-first-audio is plausible. That is
the target, and PHASE 1's instrumentation is what would prove or disprove it.

### The genuinely hard part: acoustic echo cancellation

This is the item that decides whether duplex is usable, and it is not a
modelling problem.

With an always-on microphone, the runtime will hear its own speech. Without
correction the ASR transcribes the TTS output as user speech and the assistant
responds to itself — sometimes in a runaway loop. This is the failure mode that
makes duplex demos embarrassing.

The standard answer is **browser-native AEC**, and the mechanism matters:

- `getUserMedia({audio: {echoCancellation: true}})` cancels echo **only for
  audio the browser knows is a remote participant** — which in practice means
  audio arriving over WebRTC.
- Locally generated audio played through an `<audio>` element is *not* known to
  the AEC, so it is not cancelled, even with `echoCancellation: true`.
- The workaround is a **WebRTC loopback**: route the TTS audio through a pair of
  loopback peer connections so the browser classifies it as remote-participant
  audio and cancels it from the mic. This is a known, documented technique.

Two rules follow, and both are counter-intuitive enough to be worth stating:

- **Do not toggle the microphone track.** `track.enabled = false` during playback
  is the common instinct and it **breaks AEC** — the canceller needs a continuous
  mic signal to adapt. Industry practice (ChatGPT voice, Meet, Zoom, Teams) is a
  continuously-enabled track with `echoCancellation: true`.
- **Do not add artificial delays or custom DSP** around playback as an echo
  workaround. Both interfere with the native canceller.

Chrome's `echoCancellationType: 'system'` (native OS canceller) is **not** a
recommendation here: Chrome's own measurements on Windows were described as
disappointing, with a caution against adopting it at scale. Stick with the
browser's software canceller.

**Honest fallback:** headphones. With a headset, the echo path does not exist and
duplex works without any of the above. That is not a cop-out — it is the
configuration most local duplex setups actually ship with, and it should be
labelled as a distinct mode rather than presented as equivalent to speaker-mode
duplex. Speaker mode is the stretch goal.

## 4. What cannot be replicated locally

Worth stating so the goal is not misrepresented:

- **Native audio understanding.** GPT-Live processes audio natively, so it
  perceives tone and emotion and can lose or keep paralinguistic information at
  will. A cascade flattens everything to text at the STT boundary; prosody is
  gone. A duplex *pipeline* cannot recover this.
- **Sub-200 ms model-level latency.** Moshi's 160 ms theoretical figure comes
  from a single streaming model. A cascade pays at least three model hops.
- **Semantic end-of-turn detection.** GPT-Live reasons about whether you are
  done. A local cascade approximates this with acoustic heuristics (pause length,
  prosodic contour, whether the partial ends on a clause boundary). It will be
  measurably worse at thinking-pause discrimination — likely the most noticeable
  remaining difference.

The claim worth making is bounded: **a local cascade can be made to feel duplex
for interruption, overlap and backchannel; it cannot be made to hear like a
natively-multimodal model.** Those are different achievements and should not be
described with the same word.

## 5. Recommendation

**Do not attempt a local full-duplex model.** 7.2 GB free VRAM against an
18–24 GB requirement is not a tuning problem.

**Do build duplex orchestration over the cascade**, in this order:

1. **PHASE 2 first — real local backends.** Duplex needs a streaming ASR and a
   real TTS. Neither exists yet, and duplex built on fakes would prove nothing.
   This is the prerequisite, not a detour.
2. **Streaming ASR with partials.** `asr.partial` is already an event type; make
   a provider emit it.
3. **A non-exclusive listening-while-speaking state.** The phase where the
   current machine models 10 mutually exclusive states becomes a duplex phase
   that runs the input path continuously.
4. **The interaction layer as a single arbitration point.** All turn-taking
   decisions in one place; never scattered across handlers, which is how race
   conditions show up as an assistant talking over itself.
5. **Browser AEC via WebRTC loopback**, with headphones as a documented mode.
6. **Measure the thinking-pause discrimination rate.** This is the metric that
   answers whether it feels natural. Count false interruptions during deliberate
   pauses against a labelled set.

Each of these is independently useful, and the runtime stays working at every
step. If step 6 shows the cascade cannot beat a silence timer meaningfully, that
is a finding worth publishing — and the turn-based runtime is still intact.
