# Provider licences

Every model and inference runtime this project can load, with the licence that
governs **redistribution**. This file exists because "it works on my machine"
and "we are allowed to ship this" are different questions, and the answer to the
second one changes what may enter the core distribution.

The rule followed here: **a permissive licence (MIT / Apache-2.0 / BSD) may be
part of the default install; anything copyleft stays optional and out of the
core.** The distinction is drawn at *distribution*, not at *use* — running a
GPL-licensed model privately is fine, bundling it in a product we distribute is
not.

## In the core install

| Component | Role | Licence | Notes |
|---|---|---|---|
| `faster-whisper` | ASR engine | **MIT** | Bundles CTranslate2 (MIT). No PyTorch dependency. |
| CTranslate2 | inference runtime | **MIT** | |
| Whisper weights (`Systran/faster-whisper-base` etc.) | ASR model | **MIT** | OpenAI's original Whisper checkpoints are MIT too. |
| `kokoro-onnx` | TTS engine | **Apache-2.0** | |
| Kokoro-82M v1.1 zh weights | TTS model | **Apache-2.0** | ~325 MB model + ~54 MB voices. 103 speakers. |
| `onnxruntime` | inference runtime | **MIT** | CPU execution provider here; no CUDA/DirectML runtime pulled in. |
| `misaki` | Chinese G2P | **Apache-2.0** | Required for Chinese phonemisation. |

All six are permissive. That is why they are the default local path.

## Optional / experimental — **not** in the core distribution

| Component | Role | Licence | Why it is excluded |
|---|---|---|---|
| **Piper** | TTS engine | **GPL-3.0** ⚠ | See below. |

### Why Piper is not in the core distribution

Piper's weights are permissive, but the **current core TTS engine repository is
GPL-3.0**, and linking against it would put the GPL's reciprocal obligation on
anything distributed alongside it. That is a licensing boundary we do not want
to cross in a project intended to be embeddable in a commercial product later.

This is a judgement about *distribution risk*, not about quality: Piper is a
good engine. If either of these changes, the decision should be revisited:

* upstream relicenses to MIT / Apache-2.0, or
* the topology changes so the GPL component is reached over a process boundary
  with its own clearly separated licence boundary (see below).

### The boundary that would make a GPL engine acceptable

Running the engine in a separate process and talking to it over a socket does
**not** automatically resolve the obligation — that is an implementation detail,
not a licence argument, and the FSF's position on aggregation-vs-derivation is
well documented. Treat "we put it in a subprocess" as insufficient on its own.
What would work is shipping it as an **opt-in, separately downloaded component
that the user installs themselves**, with this project merely able to talk to
it. That is a defensible boundary. It is also more work, which is why it has not
been done.

**Nothing in this repository currently depends on Piper.**

## Remote / services

These are not redistributed at all — they are existing services this project can
talk to. Their licences constrain the service, not our code.

| Service | Role | Notes |
|---|---|---|
| Ollama | LLM (local or LAN) | MIT (server). Model weights carry their own licences, which vary per model — check before shipping. |
| Voicebox | ASR/TTS (HTTP) | External service; configured by the user. |

Model weights pulled through Ollama are the operator's responsibility. Some
popular weights have commercial-use restrictions that this project cannot detect.

## Verification

Licence claims here were checked against upstream sources when this file was
written. Re-verify before changing what enters the core install — upstream projects
do relicense, and an out-of-date row is worse than no row.

If you are evaluating this project for a commercial product, do not rely on this
table alone; have counsel review it. This file records engineering judgement, not
legal advice.
