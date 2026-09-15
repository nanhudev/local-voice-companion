# SECURITY.md

## Model

This runtime is designed to run **on the user's own machine, for the user**. It
is not a multi-tenant service and it does not attempt to be one.

That assumption is stated first because it determines everything below. A runtime
that binds to loopback and serves one person has a different threat model from a
shared server, and pretending otherwise would produce a document full of controls
that do not match the deployment.

## Secrets

**Secrets are referenced by environment variable name only. Never a value.**

```json
{"token_env": "MY_SPEECH_TOKEN"}
```

not

```json
{"token": "hunter2"}
```

`redacted()` and `_scrub()` replace any key containing `api_key`, `token`, or
`secret` with `<set>` or `<unset>` — presence indicated, value never emitted.
`GET /api/v1/system/config` and every log line pass through this.

The practical test: a config dump should be safe to paste into a public issue.
If it is not, that is a bug worth reporting.

`.gitignore` excludes `config.json` and `.env`. The repository ships
`config.example.json` with no real endpoints.

## Network exposure

The runtime binds to `127.0.0.1` by default.

Binding elsewhere is possible and is the user's decision, but it is worth being
explicit about what changes: **there is no authentication.** Anything that can
reach the port can open sessions, run turns, read and modify bots, and read the
config (redacted, but still a map of the machine). The OpenAI-compatible
endpoints are equally open.

Do not expose this port to a network you do not control. If a remote client
needs access, the relay worker's split-machine arrangement or an authenticated
reverse proxy is the appropriate answer, not a wider bind.

## Audio data

Microphone audio is processed in memory and forwarded to whichever ASR provider
is selected in the turn's plan.

**It is not persisted by default.** The `data/` layout reserves space for models,
cache, logs, and benchmark results; recorded audio is not among them.

The caveat that matters: if the selected ASR provider is remote, the audio leaves
the machine. `voicebox_asr` is marked `requires_network=True` for exactly this
reason, and `GET /api/v1/system/profile` and the selection response both make the
chosen provider visible. A user who wants to be certain that speech never leaves
the machine should select a local provider — which, per `CAPABILITY_MATRIX.md`,
the builtin set does not yet offer. That gap is a privacy limitation, not only a
capability one.

## Filesystem

Bots live under the resolved data root. `bot_id` is validated — normalised to
lowercase and restricted to a safe character set, with `a/b`, `a\b`, and `a.b`
rejected — so a crafted id cannot escape the bots directory.

Manifests are `extra="forbid"`, so an unexpected key is an error rather than a
silently ignored field. A typo in a security-relevant field is caught at import
rather than discovered later.

Bots are plain files, readable by anyone who can read the directory. Do not put
secrets in a persona.

## Model downloads

The runtime does not download models. It probes for what is installed.

This is a security property, not only a resource decision: a runtime that fetches
multi-gigabyte artefacts on demand is a runtime that fetches multi-gigabyte
artefacts from somewhere, and that somewhere becomes trusted input. Installing
models is left to the user, who can see where the bytes come from.

Related, and deliberate: the runtime never claims a backend works because it
downloaded something. It reports what it can probe.

## Dependency surface

The core contract — config, events, bots, selection, provider descriptors — has
no third-party dependencies beyond Pydantic. YAML is a hand-written subset in
`bots/yamlio.py` rather than a PyYAML dependency, so that the manifest format does
not add a parser to the trusted set.

FastAPI and uvicorn are required for the API layer. They are optional if the
runtime is embedded.

## Reporting

Open an issue. Do not include a full config dump without checking it first — it
is redacted, but the scrubber protects against known-sensitive key names and
cannot know about a field that encodes a secret under an unexpected name.
