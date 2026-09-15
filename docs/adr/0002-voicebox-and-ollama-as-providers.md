# ADR-0002 — Demote Voicebox and Ollama to providers

**Status:** Accepted (PHASE 1)

## Context

The project began as a Voicebox-dependent voice chat page. Voicebox was not a
dependency in the ordinary sense — it was the spine. The turn loop existed in
its current shape because of how Voicebox's API is arranged, and Ollama sat
alongside it as the assumed LLM.

Both are legitimate choices and will remain useful for a long time. The problem
is not that they are bad. The problem is that the architecture could not express
"run without them", so the runtime's capability was capped by whatever those two
services happened to expose.

## Decision

Voicebox and Ollama become **optional compatibility providers** —
`voicebox_asr`, `voicebox_tts`, `ollama_llm` — registered alongside the fakes
and removable without breaking anything.

Concretely:

- They are marked `requires_network=True`. This is honest even when the service
  is on loopback, because from the runtime's perspective it is an HTTP call with
  an HTTP call's failure modes.
- Disabling both leaves a runtime that still starts, still plans, and still
  serves the fake pipeline.
- Their defaults live in `compat/`, not in `core/`.

## Consequences

**Good:**

- Existing users keep working: the adapters behave as the old code did.
- The architecture can now be reasoned about without reference to any specific
  engine.
- Adding a genuinely local model later is an addition, not a renegotiation.
- The legacy preference is expressed as configuration, which means it can be
  inspected, exported, and explained.

**Costs:**

- An adapter layer sits between the runtime and services that were previously
  called directly. That layer is a place for bugs to hide, and it must be
  maintained as those APIs evolve.
- The remote providers look local in configuration but behave like remote ones
  under failure. A user who has Voicebox running locally may be surprised that
  it is classified as networked. The classification is correct; the surprise is
  a documentation problem, which `CAPABILITY_MATRIX.md` addresses.
- The builtin set now contains no real local speech engine at all. Demoting
  Voicebox made that gap **visible** rather than created it — previously the gap
  was hidden behind the assumption that Voicebox was always there.

## Alternatives rejected

**Delete the Voicebox path and start clean.** Rejected: it breaks every existing
deployment at once, which RULE 13 forbids, and it discards a working integration
that is still the most capable option available today.

**Keep Voicebox as the default and add others beside it.** Rejected: "default"
in an adaptive runtime means "the thing chosen when nothing better is found",
and a hard-coded default defeats the selection engine before it starts.
