# ADR-0001 — One `ProviderDescriptor` as the anti-branch contract

**Status:** Accepted (PHASE 1)

## Context

The original code decided everything by comparing strings:

```python
if backend == "voicebox": ...
if model_name.startswith("kokoro"): ...
```

There was no representation of what a backend *can do* — only what it is
called. Adding a second TTS engine meant editing conditionals inside the turn
loop, and the turn loop therefore had to know every engine by name.

That is the actual disease. The symptom is `if engine == "x"` scattered through
core; the cause is that capability lives in the caller's head instead of in the
provider.

## Decision

Every provider — fake, local, or remote — is described by a single
`ProviderDescriptor` carrying id, kind, version, capabilities, resource
requirements, quality estimate, supported devices, network requirement, and
install hint.

No code in `core/` may branch on provider identity. It reads the descriptor and
decides from the declared facts.

## Consequences

**Good:**

- Adding a provider requires touching one new file and no core file.
- The selection engine can score providers it has never heard of, because it
  reads declared properties rather than a hard-coded table.
- The fake providers exercise the identical path as real ones, so the test
  suite tests the runtime rather than a mock of it.
- `GET /providers` is a complete answer to "what can this runtime do here",
  generated rather than maintained.

**Costs:**

- Every provider must fill in a descriptor honestly, including resource
  estimates it may not know precisely. A provider that guesses badly will be
  scored badly, and the fix is at the provider rather than in the engine.
- The descriptor is a 20-key structure that must stay stable; changing it is a
  contract change. This is why `tests/contract` pins it.
- Providers with genuinely unusual behaviour (a model that only works at one
  sample rate) must express that in the shared vocabulary, which occasionally
  fits awkwardly.

## Alternatives rejected

**A capability base class per kind.** Rejected: an inheritance hierarchy does
not survive contact with a provider that is *nearly* a TTS but not quite, and it
still leaves the caller deciding which class it is.

**A plugin discovery protocol with a registration callback.** Rejected as
over-engineering for the number of providers involved, and it does not solve the
underlying problem — knowing what a provider can do — it only moves where that
knowledge is missing.
