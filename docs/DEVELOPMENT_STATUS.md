# DEVELOPMENT_STATUS.md

Honest state of the repository. Updated at the end of each phase.

**As of: PHASE 1 complete.**

## Phase status

| Phase | Scope | Status |
| --- | --- | --- |
| PHASE 0 | Repository audit → `CURRENT_ARCHITECTURE.md` | ✅ Complete |
| PHASE 1 | Architecture foundation | ✅ Complete |
| PHASE 2 | Real local backends | ⬜ Not started |
| PHASE 3 | Realtime duplex voice | ⬜ Not started |
| PHASE 4 | Client integration | ⬜ Not started |
| PHASE 5 | Distribution | ⬜ Not started |

## PHASE 1 completion criteria

The stated bar was: `FakeASR → FakeLLM → FakeTTS` must traverse the **real**
runtime, reachable from both browser and HTTP, producing a complete fake
conversation.

| Criterion | Status | Evidence |
| --- | --- | --- |
| `pytest` passes | ✅ | `203 passed, 3 skipped` |
| Server starts | ✅ | `python app.py --runtime --port 18771` boots uvicorn |
| `GET /healthz` | ✅ | `{"status":"ok","schema_version":2,"event_schema":1}` |
| `GET /api/v1/providers` | ✅ | 7 providers listed |
| Bot creation | ✅ | `POST /api/v1/bots` → 201 |
| WebSocket session | ✅ | `WS /api/v1/sessions/{id}/stream` |
| Fake pipeline end to end | ✅ | `tests/smoke/test_runtime_smoke.py::TestFullFakeTurn` |
| Existing config migration | ✅ | `tests/smoke/test_runtime_smoke.py::TestConfigMigration` |

The three skips are `@pytest.mark.hardware` tests asserting no native inference
backend is installed. They report the truth. RULE 12 says never pretend to have
tested hardware that does not exist, and a skip is what that looks like in a
test report.

## Test suite

```
tests/unit          pure logic, no I/O
tests/integration   real FastAPI app over HTTP and WebSocket, isolated registry
tests/contract      pins public wire formats
tests/smoke         end-to-end runtime paths and CLI entry points
tests/legacy        old app.py helpers, via the shim
```

Tiered by *what they protect*, not by size. `tests/contract` exists so that a
change to the provider descriptor or the event envelope fails loudly instead of
reaching a client.

## Defects found and fixed

15, all confirmed by a failing test or a reproduction before the fix.

| # | Defect | Impact if shipped |
| --- | --- | --- |
| 1–9 | Found during PHASE 1 construction | Various |
| 10 | `create_app` registered providers into the module-level singleton while `recommend()` read the runtime's registry | An app with an isolated registry listed **zero** providers and could not plan a pipeline |
| 11 | Three `recommend()` call sites omitted `reg=` | Same class of failure, different call path |
| 12 | `BotManifest.merge()` used `cls` inside an instance method | `PUT /api/v1/bots/{id}` raised `NameError` on **every** call |
| 13 | `llm.started`, `tts.completed`, `playback.started`, `playback.finished` declared but never emitted | Subscribers saw a timeline with holes at exactly the interesting transitions |
| 14 | `AdaptiveTextChunker` returned on terminal punctuation before checking `max_chars` | One long sentence defeated the time-to-first-audio bound the cap exists to enforce |
| 15 | v1→v2 migration dropped the port embedded in `voicebox_url` | A custom URL silently dialled port 17493 |

Defects 10–15 were found by *writing tests*, not by reading code. That is the
argument for the test suite, made concretely.

## What works end to end

Verified as a real process, not only through `TestClient`:

- `python app.py --runtime --port 18771` boots and serves the full API.
- `python app.py --probe` delegates to the legacy gateway and **fails
  gracefully** when no backend is present:
  `VOICE_PROBE_FAIL voicebox_asr request failed: ... [WinError 10061]`
  A clear failure beats a hang.
- `tests/legacy` passes against the shim, so the original helpers still work.
- The fake pipeline completes a full turn over both HTTP and WebSocket.

## What does not work yet

Stated plainly, because a status document that only lists successes is
marketing.

1. **No real local ASR or TTS.** Every real speech provider is a remote
   compatibility adapter. A machine with no Voicebox service cannot hear or
   speak. See `CAPABILITY_MATRIX.md`.
2. **`cpu_only` yields a degraded plan.** Correct behaviour given the available
   providers; still a gap in capability.
3. **`web/` and `godot/` are untouched.** They still use the old HTTP shape.
   The new API is additive precisely so this did not have to happen at once.
4. **No duplex/realtime streaming.** The runtime is turn-based. Barge-in works,
   but the model does not listen and speak simultaneously.
5. **Latency is unmeasured on real models.** The instrumentation is complete;
   it has not been pointed at a real backend.
6. **`windows_worker.py` is unmodified.** Still speaks its own protocol; not
   yet a provider.

## Next milestone

**PHASE 2 — Real local backends.** Wire at least one genuine CPU-capable ASR and
TTS so that `cpu_only` stops being a degraded path, then measure Time To First
Audio on real models and replace the simulated benchmark entries with measured
ones.

See `ROADMAP.md`.
