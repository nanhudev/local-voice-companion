# DEVELOPMENT_STATUS.md

Honest state of the repository. Updated at the end of each phase.

**As of: PHASE 2 complete.**

## Phase status

| Phase | Scope | Status |
| --- | --- | --- |
| PHASE 0 | Repository audit → `CURRENT_ARCHITECTURE.md` | ✅ Complete |
| PHASE 1 | Architecture foundation | ✅ Complete |
| PHASE 2 | Real local backends | ✅ **Complete** |
| PHASE 3 | Realtime duplex voice | ⬜ Not started |
| PHASE 4 | Client integration | ⬜ Not started |
| PHASE 5 | Distribution | ⬜ Not started |

## PHASE 2 completion criteria

The stated bar was: with **GPU disabled, network disabled, Voicebox
unavailable**, `policy=cpu_only` must select a real local ASR and a real local
TTS and complete `real WAV → ASR → transcript → LLM → text → TTS → valid WAV`,
with ASR, LLM, TTS latency and TTFA all **measured** rather than simulated.

| Criterion | Status | Evidence |
| --- | --- | --- |
| Real local ASR on CPU | ✅ | `faster_whisper_cpu` — faster-whisper 1.2.1, INT8 |
| Real local TTS on CPU | ✅ | `kokoro_tts_cpu` — Kokoro-82M v1.1 zh, 103 voices |
| Selected by `cpu_only` offline | ✅ | `test_cpu_only_policy_selects_native_providers_when_installed` |
| Full offline turn completes | ✅ | `test_native_pair_completes_a_cpu_only_offline_turn` |
| Valid WAV out | ✅ | `TestNativeTTS::test_synthesise_returns_a_parseable_wav` |
| Round trip through real audio | ✅ | TTS → ASR recovers ≥3/6 characters of the spoken phrase |
| ASR/TTFA/TTFA all `MEASURED` | ✅ | `lvc benchmark --policy cpu_only --json` → `"all_measured": true` |
| No implicit downloads | ✅ | `TestNoImplicitDownload`, socket-level network block |
| Provider abstraction intact | ✅ | No engine change knows these providers by name |
| Base install still light | ✅ | Both packages are in optional requirement files, not `requirements.txt` |
| Licence boundary recorded | ✅ | `docs/PROVIDER_LICENSES.md`, ADR-0009 |

Measured, `policy=cpu_only`, GPU disabled, 1 warmup + 3 runs:

| Stage | Median | Min | Max | RTF |
| --- | --- | --- | --- | --- |
| ASR (`base`, INT8) | 573.6 ms | 567.8 | 581.3 | 0.287 |
| TTS (`kokoro-v1.1-zh`) | 1101.5 ms | 1078.8 | 1110.8 | 0.337 |
| TTFA (end to end) | 2077.8 ms | — | — | — |

**The LLM row is deliberately absent.** No local LLM is installed, so offline
turns use deterministic stub text. Reporting a measured LLM latency from a stub
would be exactly the fabricated number the acceptance criteria forbid. Full
detail in `BENCHMARKING.md`.

### Test suite this phase

```
pytest                238 passed, 0 failed, 0 skipped
pytest -m hardware      9 passed, 229 deselected
```

Verified stable across consecutive runs, because a suite that only passes once
is hiding shared state.

There are **no longer any skips.** The three PHASE 1 skips were
`@pytest.mark.hardware` cases asserting no native backend existed; they now run
for real and pass. Nothing is skipped and nothing is fabricated.

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

The three skips were `@pytest.mark.hardware` tests asserting no native inference
backend was installed. They reported the truth at the time, and they now execute
for real — see the PHASE 2 section.

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

1. **No local LLM.** This is now the largest gap. Offline turns are heard and
   spoken for real, but the *reply* is generated by deterministic stub text.
   Everything except the reasoning is genuinely local.
2. **No streaming.** `streaming=False` on both native providers, so there are no
   partial ASR hypotheses and no progressive TTS playback. This puts a floor
   under latency that tuning cannot remove.
3. **Cold start is seconds.** Kokoro loads a 325 MB graph on first use. The
   benchmark above measures warm turns only.
4. **One working Chinese G2P on Windows.** `phonemizer`/espeak fails on this
   platform despite shipping the DLL; `misaki` is the sole backend. See ADR-0008.
5. **Mixed-language synthesis is degraded.** misaki's Chinese front end cannot
   phonemise English; unphonemisable characters are dropped, not spoken.
6. **`web/` and `godot/` are untouched.** Still on the old HTTP shape; the new
   API was built additively so this could wait.
7. **No duplex/realtime streaming.** The runtime remains turn-based. Barge-in
   works; simultaneous listening and speaking does not — see
   `DUPLEX_FEASIBILITY.md` for what it would actually take.
8. **`windows_worker.py` is unmodified.**

## Next milestone

**PHASE 3 — Realtime duplex voice.** The remaining numbers to improve are TTFA
and perceived responsiveness, and both are bounded by single-pass inference.
PHASE 3 should be planned from the *measured* figures above — ASR RTF 0.287,
TTS RTF 0.337, TTFA 2077.8 ms, cold start several seconds — rather than from
estimates, and should decide whether a streaming ASR brings more than a
streaming TTS.

See `ROADMAP.md`.
