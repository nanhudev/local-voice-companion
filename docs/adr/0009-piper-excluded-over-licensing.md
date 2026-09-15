# ADR-0009 — Keep Piper out of the core distribution

**Status:** Accepted (PHASE 2)
**Superseded by:** — revisit if upstream relicenses

## Context

Piper was the first candidate considered for local Chinese TTS. It has a real
engineering advantage over Kokoro for low-resource deployment: it is fast, it is
small, and it is well established in the Raspberry Pi and hobbyist community. On
purely technical grounds it was competitive.

The user raised the objection that decided this, and it is worth quoting in
substance because it is the kind of thing that is easy to wave away and
expensive to discover later: **Piper's core repository is GPL-3.0.** Kokoro's is
Apache-2.0 and faster-whisper's is MIT.

There is no technical problem with *running* GPL-licensed code. The problem is
specific to *distributing* it. This project is being built toward being
embeddable — shipped inside something else, potentially commercially — and a
GPL dependency changes what distribution would require. Deciding that at the
point where it costs nothing is much cheaper than discovering it after adopting
the engine.

## Decision

Piper is **not** part of the core distribution. It may exist as an **optional,
experimental provider** that a user installs themselves, but:

* it is not in any default requirement set;
* nothing in the shipped code imports it or depends on it;
* the default local TTS path must not require it.

Nothing in this repository currently depends on Piper at all. This ADR records
the boundary so the next person who evaluates Piper does not have to rediscover
it — and so that "why isn't Piper supported?" has an answer attached.

The general rule is stated in `docs/PROVIDER_LICENSES.md`:

> A permissive licence (MIT / Apache-2.0 / BSD) may be part of the default
> install; anything copyleft stays optional and out of the core. The distinction
> is drawn at *distribution*, not at *use*.

## What would change this decision

Genuinely, not rhetorically:

1. **Upstream relicenses** the engine to MIT or Apache-2.0. This happens; it is
   worth re-checking periodically.
2. **A clear separation is established**, where Piper is not distributed by us at
   all and the user installs it independently, with this project only able to
   speak its interface. That is a defensible boundary and is the only realistic
   path to supporting it.

What does **not** count: putting it in a subprocess. Running GPL code in a
separate process and talking over a socket does not by itself resolve the
obligation — that is an implementation detail, not a licence argument. Recording
this explicitly because it is the most likely thing someone will propose.

## Cost of this decision

* Piper's low-resource performance advantage is given up. Kokoro at 82M is
  heavier than Piper on constrained hardware.
* If Piper is later adopted via the separation route, we keep a compatibility
  surface for an engine we do not control.
* Kokoro's Apache-2.0 licence covers *these* weights. If a future Kokoro release
  changed licence, the same analysis would need to be redone — this is not a
  permanent property of the project.

## Not legal advice

This is engineering judgement, recorded so the reasoning survives. The table in
`docs/PROVIDER_LICENSES.md` records what was verified against upstream sources
at the time of writing. Anyone evaluating this project for a commercial product
should have counsel confirm; do not rely on this ADR alone.
