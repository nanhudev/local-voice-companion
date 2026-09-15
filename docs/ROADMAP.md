# ROADMAP.md

## Where the project is

PHASE 1 built the skeleton: a selection engine, a provider contract, a turn
state machine, an event stream, and a test suite that runs `FakeASR → FakeLLM →
FakeTTS` through the real runtime.

What it does not have is a real voice. Every actual speech provider is a remote
compatibility adapter, so a machine with no Voicebox service can hold a
conversation with a fake and with nothing else.

The next phases close that gap and then use the skeleton for what it was built
for.

---

## PHASE 2 — Real local backends

**Goal:** `cpu_only` stops being a degraded path.

- Wire at least one genuinely CPU-capable ASR (`faster-whisper` or equivalent)
  as a first-class provider with honest resource estimates.
- Wire at least one CPU-capable TTS (Piper, Kokoro, or equivalent).
- Run benchmarks on real models and replace simulated entries with `MEASURED`.
- Publish real Time To First Audio numbers per policy on the reference machine.

**Exit criteria:** on a machine with no GPU and no network, `auto` selects a real
ASR and a real TTS, completes a turn, and reports `MEASURED` latency for each
stage.

**Why first:** until this lands, every claim the project makes about being a local
voice runtime is aspirational. The selection engine has been validated against
fakes; it has not been validated against a model that takes real time and real
memory.

---

## PHASE 3 — Realtime duplex voice

**Goal:** listen and speak at the same time.

The current runtime is turn-based: it listens, then thinks, then speaks. GPT-style
realtime voice does all three concurrently, and the felt difference is large — you
can interrupt mid-sentence and the reply adjusts, you hear a response beginning
before the model has finished composing, and pauses stop being turn boundaries.

**Feasibility has been assessed — see [DUPLEX_FEASIBILITY.md](DUPLEX_FEASIBILITY.md).**
The conclusion, in one line: a local full-duplex *model* does not fit on this
machine (7.2 GB free VRAM against Moshi's 24 GB / PersonaPlex's ~18 GB), but a
full-duplex *pipeline* over a cascade does, and that is what this phase builds.

The PHASE 1 skeleton was built with this in mind, and the pieces are already
there:

| Needed | Already exists |
| --- | --- |
| Interrupt mid-speech | `CancellationToken`, `CANCELLING` state |
| Concurrent listen + speak | Concurrent synthesis overlapping generation |
| Bounded buffers under streaming load | `BoundedQueue` with drop policies |
| Incremental output | `AdaptiveTextChunker` |
| Latency accounting | Full turn timeline, TTFA |

What is genuinely missing:

- **A streaming ASR that emits partials continuously** rather than one
  utterance per turn. The event type (`asr.partial`) exists; no provider emits
  them.
- **A state that means "speaking while listening"** — currently the ten states
  are mutually exclusive.
- **The interaction layer** — one arbitration point deciding whether to keep
  speaking, whether a pause means the thought finished, and whether to
  backchannel. Duplex-ness lives here, not in the model.
- **Acoustic echo handling.** Without it, the runtime transcribes its own voice.
  The mechanism is browser-native AEC via a WebRTC loopback; headphones are a
  documented fallback mode, not an equivalent one.
- **A local model capable of streaming inference at conversational latency on
  8 GB of VRAM.** Budgeted at ~3.5–4.5 GB for ASR + a 4B-class quantized LLM,
  with Piper TTS on CPU.

**Exit criteria:** with a headset, the user can interrupt mid-sentence and the
runtime responds to the interruption rather than to its own output; and a
measured thinking-pause discrimination rate that beats a silence timer.

**What will not be achieved:** native audio understanding. A cascade flattens
prosody to text at the STT boundary. A local cascade can feel duplex for
interruption, overlap and backchannel; it cannot hear like a natively-multimodal
model. Those are different achievements and this roadmap does not conflate them.

**Committed constraint:** this phase must not weaken the turn-based path. If
duplex proves infeasible on the target hardware, the turn-based runtime remains
the product and the duplex work is documented as an evaluated dead end. A
half-working duplex mode that degrades the working one is the worst outcome
available.

---

## PHASE 4 — Client integration

- Migrate `web/` to `/api/v1` and drive the UI from the event stream.
- Replace the Godot polling loop with the WebSocket event stream.
- Port `windows_worker.py` into a provider so the relay path stops being a
  separate protocol.
- Retire the legacy HTTP shape once both clients have moved.

---

## PHASE 5 — Distribution

- One-command install that puts models on a non-system drive by default.
- A model catalogue with declared sizes, so a user can choose before downloading
  eight gigabytes.
- Packaging for the desktop.

---

## Standing constraints

These apply to every phase:

1. **Read the code before designing against it.** Assumptions about structure
   that were never checked produce plans that do not survive first contact.
2. **Architecture before models.** A new model is never a reason to bypass the
   provider contract.
3. **No engine is core.** Voicebox and Ollama are providers, and so is anything
   that replaces them.
4. **Fakes and tests from the start.** A component that cannot be exercised
   deterministically is a component that will be debugged in production.
5. **Never claim to have tested hardware that does not exist.** A skip is a
   truthful result. A fabricated pass is not.
6. **Keep the old path working.** Existing deployments matter more than
   architectural tidiness.
7. **No new monoliths.**
8. **Large files go where there is space.** Models, caches, and virtual
   environments default to a data root outside the system drive, because a
   local-first tool that fills the system drive is unusable.
