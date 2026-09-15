# ADR-0004 — Explicit turn state machine with token-based cancellation

**Status:** Accepted (PHASE 1)

## Context

The original turn loop was a linear function:

```python
text = await transcribe(audio)
reply = await generate(text)
await speak(reply)
```

It worked. It could not do three things, and none of them were fixable by
patching the function:

1. **Barge-in was inexpressible.** A user speaking mid-reply had nowhere to
   interrupt *to*. The loop had already committed to `speak(reply)`, and there
   was no state that meant "stop the current thing".
2. **Progress was invisible.** With no states, there was nothing to report, so
   the UI could only show "working" from the first token to the last audio
   frame.
3. **Failure was indistinguishable from slowness.** No state meant no way to
   tell a hung synthesis from a slow one.

## Decision

A ten-state machine: `IDLE, LISTENING, CAPTURING, TRANSCRIBING, THINKING,
SYNTHESIZING, SPEAKING, CANCELLING, PAUSED, ERROR`. Every transition is
validated; illegal ones raise.

Cancellation is a `CancellationToken` wrapping an `asyncio.Event`, with three
properties that matter:

- **Idempotent.** Cancelling twice is not an error.
- **First reason wins.** Later cleanup does not overwrite the recorded cause, so
  the reported reason is the real one.
- **Propagated.** The synthesis worker and audio emitter observe the same token,
  so no task is orphaned on cancellation.

A user interruption ends the turn in `CANCELLING`, **not** `ERROR`.

## Consequences

**Good:**

- Barge-in is a state transition rather than a special case threaded through the
  turn function.
- `turn.state` events give the UI something real to display and give a debugger a
  trace of where a turn went.
- Cancellation is testable deterministically: cancel the token, assert the turn
  lands in `CANCELLING`, assert no task is left running. The tests in
  `tests/smoke/test_runtime_smoke.py::TestBargeIn` do exactly this.
- The state machine is the foundation for PHASE 3's duplex mode, which needs a
  first-class notion of "listening while speaking".

**Costs:**

- More code than a linear function, and every new behaviour has to find a legal
  path through the graph. Sometimes that is friction; the friction is the point,
  because the alternative is an unreachable state discovered in production.
- Illegal transitions raise, which means a caller that reasons sloppily about
  ordering gets an exception rather than a best-effort answer. This is
  deliberate — a provider that silently tolerates an impossible transition
  produces bugs that look like model failures.
- Interrupting a user is normal, so `CANCELLING` is a *successful* terminal
  state and every consumer of the state machine must understand that. Any client
  treating non-`IDLE` terminals as errors will misreport ordinary usage.

## Alternatives rejected

**A boolean `is_speaking` flag.** Rejected: two states cannot distinguish
transcribing from thinking, which is precisely the distinction needed to
diagnose latency.

**Python task cancellation (`asyncio.Task.cancel()`) alone.** Rejected as
insufficient on its own: `Task.cancel()` is not idempotent under pending
cleanup, does not record *why* cancellation happened, and raises
`CancelledError` at an unpredictable await point. The token is checked at
defined points instead, which makes the behaviour deterministic.
