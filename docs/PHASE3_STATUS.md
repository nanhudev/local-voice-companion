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
253 passed, 9 skipped   (two consecutive runs: 53.0 s / 54.3 s)
```

The 9 skips are all Kokoro weights still downloading. They are skips, not
passes: on a machine without them there is legitimately no local voice path.

`tests/integration/test_streaming_asr.py` holds 4 `hardware`-marked tests that
load the real engine. They assert that partials actually arrive, that the
streaming final equals the whole-utterance result (streaming must not be a
silent quality downgrade), and that committed text is never ahead of the
hypothesis — which is where a naive "committed only grows" tracker gets caught.

---

## What is NOT done

| Step | State | Note |
|---|---|---|
| 3A — wire `asr.partial` into the API/WS | **NOT STARTED** | `EventType.ASR_PARTIAL` is declared and still **emits nothing**. This is the remaining half of 3A. |
| 3B — barge-in | NOT STARTED | |
| 3C — duplex arbitration | NOT STARTED | |
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
