# ADR-0005 — Time To First Audio as the headline metric

**Status:** Accepted (PHASE 1)

## Context

There were several candidate headline numbers: total turn duration, tokens per
second, real-time factor, time to first audio. Choosing one determines what the
architecture optimises, because a project optimises what it measures.

Total turn duration is the intuitive choice and it is the wrong one. It averages
over the part of the turn the model spends *talking*, which the user is not
waiting on. Tokens per second measures throughput, not responsiveness. Real-time
factor measures the synthesiser against a clock the user cannot perceive.

## Decision

**Time To First Audio is the headline metric.** It is the interval from the user
finishing a sentence to the first audio frame being delivered.

Everything else is a component of it: `asr_latency_ms`, `llm_ttft_ms`,
`tts_ttfa_ms`. The turn timeline records each stage, and the derived metric is
computed from the causal pairs.

This choice is why three architectural features exist:

- **Streaming generation** — start the LLM producing immediately rather than
  waiting for a complete reply.
- **Adaptive chunking** — release a phrase the moment it is a natural boundary,
  not when the paragraph ends.
- **Concurrent synthesis** — synthesise while generation is still running, so
  `tts.started` legitimately precedes `llm.completed`.

## Consequences

**Good:**

- The metric matches perception. A reply that starts in 0.8s and finishes in 4s
  feels fast; one that starts at 2.9s and finishes in 3s feels broken. Total
  duration cannot tell those apart; this can.
- It has a clear owner in the architecture. Any latency work reduces to one
  number, which makes experiments comparable.
- It forces the streaming path to be real. A runtime that batched everything and
  measured first-audio-at-the-end would report an honest but useless number.

**Costs:**

- **The event timeline is not globally monotonic.** `tts.started` can precede
  `llm.completed`. This surprised a test writer during PHASE 1 and it will
  surprise contributors again; the timeline is causally ordered, not
  sequentially ordered, and the tests assert causal pairs for that reason.
- **A large chunk after a long silence is not punished enough.** Time to first
  audio says nothing about the second audio frame. A provider that emits one
  fast frame and then stalls looks excellent on this metric and is not. This is a
  real blind spot and it is why `tts_ttfa_ms` and `total_turn_ms` are retained
  alongside it rather than replaced.
- **Clock resolution matters.** On Windows, `time.monotonic` ticks at roughly
  15.6 ms, which is coarse enough to distort sub-100ms measurements. The runtime
  uses `time.perf_counter` and sets `clock_limited` when the resolution is too
  coarse to trust the value. Reporting a precise-looking number computed from a
  coarse clock would be worse than reporting the limitation.

## Alternatives rejected

**Total turn duration.** Rejected: it hides the number the user actually
experiences inside an average weighted by reply length.

**Tokens per second.** Rejected: throughput is a server-side concern. A user
cannot perceive tokens per second; they perceive silence.

**Real-time factor.** Rejected: meaningful for batch synthesis, not for a
conversational turn where the answer is not known in advance.
