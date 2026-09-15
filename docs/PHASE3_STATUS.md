# PHASE 3 — Realtime Duplex Conversation: status

This file is the honest state of PHASE 3. It is written to be read by the next
session before touching anything, because "PHASE 3 is in progress" is not a
state you can act on.

Baseline for this phase: `44f6988` (PHASE 2 close, 238 passed / 0 skipped).

---

## 3A — Streaming foundation: DONE (measured)

### What landed

| Piece | File | Why it exists |
|---|---|---|
| `AudioFrame` | `core/audio.py` | Carries `sequence` + `captured_at`. Without a sequence, a dropped frame is invisible and a partial transcript resting on audio with a hole in it looks just as authoritative as one that does not. |
| `InputAudioStream` | `core/audio.py` | Bounded, self-sequencing. Drops the **oldest** frame under backpressure — preserving old audio at the cost of latency is how duplex systems become laggy ones. |
| `TranscriptStabilityTracker` | `core/transcriber.py` | Splits every result into a **committed** prefix and an **unstable** tail. |
| `stream_transcribe()` | `providers/base.py` | Uniform contract: providers that cannot stream buffer and yield exactly one final, instead of manufacturing partials by chopping up a whole-utterance result. |
| `supports_streaming` / `supports_partial_results` | `providers/base.py` | Added, not overloading `streaming`. Both default False, so no existing descriptor changes meaning. |
| `SherpaStreamingASR` | `providers/local/sherpa_streaming_asr.py` | sherpa-onnx Streaming Zipformer, int8, CPU. |

### Measured, not asserted

Same wavs throughout (shipped with the model, 3.1–5.6 s of Chinese speech).
CPU only, 20 ms frames, 3 repeats, medians.

| Path | First text | Complete | RTF |
|---|---|---|---|
| **sherpa streaming** | **after 480–800 ms of audio** (compute 40–91 ms) | 0.44–0.79 s | 0.13–0.15 |
| sherpa whole-utterance (identical weights) | — | 0.47–0.85 s | 0.14–0.15 |
| faster-whisper whole-utterance (PHASE 2 engine) | — | **1.79–1.91 s** | 0.33–0.62 |

Streaming vs whole-utterance isolates **architecture** (same weights).
The two whole-utterance rows isolate **engine**.

**Read the first column carefully.** 40–91 ms is the pure compute time when
frames are supplied as fast as the engine consumes them. In live capture frames
arrive at real time, so the first partial cannot exist until enough speech has
happened — measured at **480–800 ms of audio**. That is the number comparable to
faster-whisper's 1.8–1.9 s, and quoting 40 ms as "time to first partial" would
misrepresent it.

Reproduce:

```bash
PYTHONPATH=src .workbuddy/tmp/bench_stream_asr.py --frame-ms 20 --repeats 3
```

### Defects found and fixed

1. **`BoundedQueue` evicted audio on close.** The end-of-stream marker was
   treated as payload, so closing a full buffer dropped the newest frame — at
   exactly the moment the final hypothesis needs it. Queues now take `reserved`
   slots and `put(..., force=True)` for control items.
2. **Sequence gaps were counted twice** — once by the producer, once by the
   consumer, for the same hole. Detection now lives only on the consumer, which
   sees both externally-missing frames and backpressure drops.
3. **The tracker mixed time bases.** Passing `started_at=0.0` still sampled
   `perf_counter()`, so `since_change_ms` returned −7.4e6 ms. The clock is now
   injectable, because a pause-detection signal you cannot control is one you
   cannot test.
4. **`faster-whisper` could not be installed at all.** `model.bin` is
   145,217,532 bytes (138 MiB) but the catalog floored it at 140 MiB, so every
   fetch downloaded everything and then failed validation. Floor corrected;
   error message no longer blames only the mirror.

### Test state

```
253 passed, 9 skipped   (3A foundation, two consecutive runs: 53.0 s / 54.3 s)
278 passed,  0 skipped  (after `asr.partial` landed; the 9 skips were Kokoro
                         weights that had finished downloading, not fixes)
```

`tests/integration/test_streaming_asr.py` holds 4 `hardware`-marked tests that
load the real engine. They assert that partials actually arrive, that the
streaming final equals the whole-utterance result (streaming must not be a
silent quality downgrade), and that committed text is never ahead of the
hypothesis — which is where a naive "committed only grows" tracker gets caught.

---

## 3A (part 2) — `asr.partial` leaves the runtime: DONE

`EventType.ASR_PARTIAL` existed and was emitted by nobody. It now is, on two
paths, and both are covered by tests that fail if it stops.

| Piece | File | What it does |
|---|---|---|
| `transcribe_stream()` | `core/orchestrator.py` | Runs `stream_transcribe()` and publishes every non-final update as `asr.partial`, then exactly one `asr.final` — always, including on cancellation, because a client needs a terminator and not a stream that merely stops. |
| `iter_frames()` | `core/audio.py` | Re-slices a finished buffer into 20 ms frames so a one-shot upload travels the *same* path as live audio. One path, not a second near-identical one that can drift. |
| `TurnRequest.frames` | `core/orchestrator.py` | The live-microphone shape. Set it and transcription runs while audio arrives. |
| `open_input_stream` / `push_audio` / `end_input_stream` | `core/session.py` | The handoff between the websocket that receives frames and the turn task that consumes them. |
| `audio.start` / `audio.frame` / `audio.end` | `api/app.py` (WS) | The client protocol. Frames arriving before `audio.start` are reported as an error, not silently dropped. |
| `asr_first_partial` stage | `core/events.py` | New timeline stage plus `asr_ttfp_ms`. Its *absence* is meaningful: it means the turn waited for the whole utterance. |
| `ScriptedStreamingASR` | `providers/fake.py` | A fake whose hypotheses are scripted, so "partials are emitted" is testable. `FakeASR` cannot fail this test: it knows its only answer up front. |

Two deliberate decisions, both about not lying with numbers:

* **`vad_end` is stamped when capture closes, not when frames run out.** Marking
  it inside ASR would silently redefine `asr_latency_ms` from "speech end →
  result" into "last frame dequeued → result".
* **`asr_ttfp_ms` on the replay path excludes capture and transport.** Frames
  produced by `iter_frames` are stamped when sliced, not when the sound
  happened. The live path is the one whose number a user feels; quoting the
  replay number as if it were live would understate it.

---

## 3B — Barge-in: DONE (measured end to end)

The user starts talking while the assistant is still talking, and playback stops.

This is **pipeline-level** interruption, not a duplex model. A model that listens
and speaks at the same instant needs 18–24 GB of VRAM
(`docs/DUPLEX_FEASIBILITY.md`); this machine has 7.2 GB free.

### How it works

`core/bargein.py` holds a `BargeInWatcher` fed one frame at a time from
`Session.push_audio`. It fires when:

1. the session is in `SPEAKING` or `SYNTHESIZING` — the only states in which
   user speech is an interruption rather than the turn's own input;
2. the loaded VAD is actually serving. **A VAD that is absent, failed to load,
   or disabled does not fall back to guessing from energy.** The watcher reports
   itself unavailable, and `Session.listening_for_barge_in` is False.
3. `min_speech_ms` of *continuous* speech has been seen, measured in **audio
   time** (each frame contributes its own duration) rather than wall-clock time.
   Wall-clock would let a queue stall or a GC pause look like a long utterance,
   which is how a cough becomes an interruption.
4. the cooldown window since `playback_start` has expired. This exists because
   the speaker's own attack transient and its echo live there. **A cooldown is a
   mitigation for speaker use, not a substitute for AEC** — that is 3D.

One interruption per turn. A second `bargein.detected` for something already
reported would make any latency histogram meaningless.

### What the client sees

```
bargein.detected  {"speech_ms": 100.0, "state": "SPEAKING"}
turn.cancelled    {"reason": "barge_in"}
playback.stopped  {"bytes": 6956, "discarded": 0, "reason": "barge_in"}
turn.completed    {"status": "cancelled"}
```

`playback.stopped` is deliberately **not** `turn.cancelled`. A client watching
only the latter cannot distinguish "nothing was playing, nothing to stop" from
"the speaker is mid-sentence, stop it now", and it holds the audio device.

New timeline stages: `bargein_detected`, `playback_stopped`. An interrupted turn
has `playback_stopped` and **no** `playback_end`; a normal one has `playback_end`
and no `playback_stopped`. New metric: `barge_in_latency_ms`.

### Three real defects found while building it

These were not hypothetical; each one was reproduced first.

1. **Cancellation did not stop playback.** Both pipeline stages used
   `await queue.get()`, so a cancelled stage stayed asleep until the next item
   happened to arrive — the interruption was audible as a delay.
   `BoundedQueue.get_or_cancel()` now races the queue against the token.
2. **The audio emitter was left running when the LLM was still streaming.**
   A barge-in lands during generation far more often than between chunks, and
   that path skipped the cleanup entirely: no `playback.stopped`, an orphaned
   task, and "Task exception was never retrieved" logged far from the cause.
   Every exit now funnels through one handler that drains both workers.
3. **A late-unwinding turn wiped the turn that replaced it.** After a barge-in
   the new turn begins while the old coroutine is still unwinding, and
   `finish_turn()` cleared `active_timeline` and `active_token` — belonging to
   the *new* turn. After that, `cancel()` and `vad_end` silently stopped
   working for the turn the user was actually in. `finish_turn` now takes the
   caller's own timeline and only clears it if it is still current; state
   changes go through the same identity check.

### Note on getting here

A full-suite run took **12+ minutes instead of ~2** during development. The
cause was a `NameError` in the websocket handler from a bad rename: the broad
`except Exception` turned it into an `error` frame, the client kept waiting for
`turn.completed`, and every websocket test then hung until timeout. Lesson:
`pytest … | tail` reports `tail`'s exit code, so a *timed-out* file looked like
a pass, and a per-file bisect using it reported every file green.

Tests: **304 passed, 0 skipped, two consecutive runs** (was 278). The 26 new
cases cover the watcher's hysteresis, its gating, its honest absence, and a real
interruption over the websocket, including that no `tts.audio` arrives after
`playback.stopped`.

---

## What is NOT done

| Step | State | Note |
|---|---|---|
| 3A — `asr.partial` into the API/WS | **DONE** | Emitted on both the live-frame path and the one-shot buffer path. |
| 3A — real TTFP measurement with sherpa on live audio | NOT DONE | The 480–800 ms figure above is "audio consumed before first text" from the benchmark, not a websocket round trip. |
| 3B — barge-in | **DONE** | Headset path. Cooldown mitigates speaker echo; real speaker barge-in still needs AEC (3D). |
| 3C — duplex arbitration | NOT STARTED | Today an interruption *stops* the turn. Deciding whether to start a new one from the interrupting audio is next. |
| 3D — AEC | NOT STARTED | Headset first; speaker AEC must not block the phase. |
| 3E — pause / backchannel | NOT STARTED | |
| 3F — performance | NOT STARTED | |

### Environment notes for whoever continues

* The working clone is **`D:\AI_Workspace\projects\local-voice-companion`**.
  The E: drive no longer exists on this machine; the data root resolves to
  `D:\AI_Workspace\local-voice-companion`, and PHASE 2's weights had to be
  re-downloaded there. If skips suddenly multiply, check the weights before
  suspecting a regression.
* sherpa-onnx needs **two** wheels: `sherpa-onnx` (metapackage) and
  `sherpa-onnx-core` (the binary). It statically links its own ONNX Runtime, so
  it does not clash with the `onnxruntime` Kokoro uses — verified, not assumed.
* `runtime_probe` exposes **`module_present()`**, not `has_module()`.
* VAD for 3B is available from sherpa already (`VadModel`, Silero or Ten) — no
  new dependency needed.
