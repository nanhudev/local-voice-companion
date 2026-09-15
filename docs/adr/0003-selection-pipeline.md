# ADR-0003 — Selection as Candidate → Constraint → Score → Decision

**Status:** Accepted (PHASE 1)

## Context

The runtime must choose an ASR, an LLM, and a TTS for the machine it is on. The
naive implementation is a lookup table or a chain of conditionals: "if VRAM > 8
GB use X, else use Y". That shape has three problems that all appear at once as
soon as a third provider exists:

1. It cannot explain itself. `if` chains produce an answer with no reasons.
2. It cannot degrade. A table either has an entry for the situation or it does
   not, and "no entry" becomes a crash.
3. It cannot be extended without editing core.

## Decision

Selection is a four-stage pipeline with an explicit intermediate value at each
stage:

```
Candidate  →  ConstraintSet  →  BenchmarkResult  →  ScoreBreakdown  →  Decision
```

- **Candidates** generate the space: provider × model × device.
- **Constraints** filter hard, *before* scoring. A candidate that cannot run is
  removed, not penalised.
- **Benchmarks** supply latency and resource measurements, each tagged with
  whether it was measured or simulated.
- **Scores** are a named breakdown — latency, quality, resource fit,
  reliability, preference — not one opaque number.
- **Decisions** carry the effective policy, the plan, the alternatives, and
  notes.

A `PipelineResourcePlanner` sits across the whole plan and **demotes** candidates
by best score-per-megabyte-freed when ASR + LLM + TTS do not fit together.

## Consequences

**Good:**

- "Why was this chosen" is answerable from the response body. The `notes` and
  `scores` fields exist for this.
- Degradation is a first-class outcome. `cpu_only` returns a valid, worse plan
  and says so, rather than failing.
- The engine scores providers it has never seen, because it reads declared
  properties.
- Filters run before scoring, so a rejected candidate never appears as a low
  score. This matters: a low score suggests a real evaluation happened.

**Costs:**

- Significant surface area. `selection/` is six modules, and each has to stay
  coherent with the others.
- Resource estimates from providers are approximations, and the planner's
  arithmetic is only as good as they are. The mitigation is that errors are made
  in the recoverable direction — demote too eagerly rather than load something
  that does not fit.
- Benchmark data starts out mostly simulated. Until real measurements exist, the
  scores are informed guesses wearing a structured costume. This is a genuine
  weakness, and it is recorded rather than papered over: `BenchmarkSource`
  distinguishes the two, and the roadmap's next milestone is to replace
  simulated entries with measured ones.

## Alternatives rejected

**A weight table per hardware tier.** Rejected: it hard-codes the answer for
hardware that does not exist yet, and it cannot degrade.

**Scoring everything and letting low scores lose.** Rejected: scoring a
candidate that cannot run wastes work and produces misleading output. A
candidate that cannot execute is not a bad candidate, it is not a candidate.

**A single composite score.** Rejected: it makes the result unexplainable at
exactly the moment explanation matters — when the user disagrees with it.
