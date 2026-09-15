# ADR-0008 — Treat grapheme-to-phoneme as a first-class dependency

**Status:** Accepted (PHASE 2)

## Context

Before this decision, "TTS" was modelled as one black box: text in, audio out.
That model is wrong in a way that matters operationally, because a neural
vocoder cannot consume text. Something has to turn Chinese characters into
phonemes first, and that something is a separate package with its own version,
its own dictionary, its own platform quirks, and its own failure modes.

Treating G2P as invisible had two consequences:

* When it broke, the error surfaced as "TTS failed" rather than "the Chinese
  phonemiser is unavailable", so the fix was not obvious from the error.
* Its cost was hidden inside the TTS measurement, making it impossible to say
  whether synthesis was slow because of the model or because of the front end.

## Decision

G2P is probed, reported and benchmarked as its own concern:

* `runtime_probe.probe_g2p(backend)` is called **separately** from the weight
  check and the runtime check. A provider can therefore be reported as
  unreachable specifically because of its G2P backend, with an actionable
  message, rather than generically unavailable.
* `python lvc.py doctor` reports **Native Chinese G2P (misaki)** as its own line.
* Every TTS benchmark records `g2p_backend` and includes G2P time in the
  measured synthesis duration — not because that inflates the number, but
  because leaving it out would understate what the user actually waits for.

## The Windows constraint

This decision is heavier on Windows than anywhere else, and pretending otherwise
would be dishonest:

`phonemizer` raises `RuntimeError: espeak not installed on your system` on
Windows **even though `espeakng-loader` correctly ships the DLL.** The library's
platform detection does not resolve the bundled binary. So the espeak fallback
path exists in the code, is retained for Linux/macOS, and is known to be
non-functional on this platform — which leaves **`misaki` as the single working
Chinese G2P backend on Windows.**

That stated supported matrix is therefore:

| Backend | Windows | Linux/macOS |
|---|---|---|
| `misaki` | working | working |
| `espeak` (phonemizer) | **not working** | working |

A single-backend dependency is a real risk. It is accepted here because the
alternative is no Chinese TTS at all, and it is recorded rather than hidden so
that the risk can be acted on later.

## Known limitation

misaki's `ZHG2P` uses a Chinese front end. English words passed through it are
not phonemisable and are replaced by a marker (`❓`) rather than synthesised.
Mixed Chinese/English text is therefore partially degraded. This is upstream
behaviour, not a defect introduced locally, and it is why mixed-language input
should be normalised before synthesis at a later phase.

## Consequences

* `misaki[zh]` is a required dependency for Chinese TTS, pulling `jieba`,
  `pypinyin`, `cn2an` and friends. It is declared explicitly in
  `requirements-local-tts.txt` rather than hidden under `kokoro-onnx`.
* The TTS descriptor's `version` is a **composite** of kokoro-onnx, onnxruntime
  and misaki versions, so an upgrade to any one of them correctly invalidates a
  cached benchmark. This was introduced after it was noticed that a version of
  `"1.0.0"` made cache invalidation appear implemented while doing nothing.
* If misaki becomes unmaintained, Chinese TTS stops being installable on Windows.
  That is the accepted risk, and it belongs on the roadmap rather than in a
  comment.
