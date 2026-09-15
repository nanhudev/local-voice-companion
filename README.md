# Local Voice Companion

English | [简体中文](#简体中文)

An **adaptive local voice runtime**. It probes the machine it is running on,
decides which speech recognition, language model, and speech synthesis to use,
and then serves conversations over that plan to any client — browser, game
engine, or another agent.

It is not a wrapper around one engine. Voicebox and Ollama are providers here,
exactly like the fake providers used in tests are providers.

## Highlights

- **Hardware probing** — CPU, RAM, GPU, VRAM, accelerators, and installed
  services, reported rather than assumed
- **Adaptive selection** — candidates are filtered, scored, and planned into one
  ASR + LLM + TTS combination that fits the machine
- **Time To First Audio** as the headline metric, with a full per-stage timeline
- **Turn state machine** with proper barge-in via cancellation tokens
- **Bounded queues** with counted overflow, so a slow consumer cannot become an
  out-of-memory crash
- **Streaming** generation and synthesis, running concurrently
- **Portable bot manifests** — export from one machine, import on another
- **Structured event stream** over HTTP and WebSocket
- Microphone voice activity detection and streaming turn handling
- Ollama-compatible and Voicebox-compatible providers (optional)
- Optional Windows GPU worker for split-machine setups
- Minimal Godot integration example

## Documentation

| Document | Contents |
| --- | --- |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the runtime works and why |
| [API.md](docs/API.md) | Every endpoint, with the real response shapes |
| [DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md) | What works, what does not, current test results |
| [CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) | Which providers actually run today |
| [BENCHMARKING.md](docs/BENCHMARKING.md) | How latency is measured, and how it avoids inventing numbers |
| [ROADMAP.md](docs/ROADMAP.md) | Phases, including realtime duplex voice |
| [SECURITY.md](docs/SECURITY.md) | Threat model and secret handling |
| [CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md) | PHASE 0 audit of the original codebase |
| [docs/adr/](docs/adr/) | Architecture decision records |

## Status, stated plainly

The architecture is complete and tested: **203 passed, 3 skipped** across unit,
integration, contract, and smoke tiers.

The three skips are `@pytest.mark.hardware` tests asserting that **no native
inference backend is installed on this machine**. They report the truth rather
than fabricating a pass.

**There is currently no real local ASR or TTS implementation in the builtin
provider set.** Every real speech provider is a remote compatibility adapter, so
`cpu_only` produces a valid but degraded plan. Closing this is PHASE 2 — see
[CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) for the full picture.

## Quick start on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
Copy-Item config.example.json config.json
.\.venv\Scripts\python app.py
```

Open `http://127.0.0.1:17831`. For the guided setup, run `setup.ps1` once and use `start.ps1` afterward. Run `doctor.ps1` when a model endpoint, speech service, or microphone is not detected.

### Running the new runtime

The adaptive runtime is available now, alongside the original gateway. It will
become the default in a later phase; today it is opt-in:

```powershell
.\.venv\Scripts\python app.py --runtime
```

Then:

```
GET  /healthz                     liveness
GET  /readyz                      readiness — 503 until a pipeline is prepared
GET  /api/v1/system/profile       what this machine actually has
GET  /api/v1/providers            what is registered and in what state
POST /api/v1/selection/recommend  the plan, with scores and reasons
```

Interact with it:

```powershell
# What does this machine look like?
curl http://127.0.0.1:17831/api/v1/system/profile

# What would it choose?
curl -X POST http://127.0.0.1:17831/api/v1/selection/recommend `
     -H "Content-Type: application/json" -d '{}'

# Create a bot, open a session, run a turn
curl -X POST http://127.0.0.1:17831/api/v1/bots `
     -H "Content-Type: application/json" -d '{"id":"demo","name":"Demo"}'
```

See [API.md](docs/API.md) for the full surface and [DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md)
for what currently works end to end.

### Tests

```powershell
.\.venv\Scripts\python -m pytest
```

Tiers: `tests/unit`, `tests/integration`, `tests/contract`, `tests/smoke`,
`tests/legacy`. Hardware-dependent tests are marked and skipped by default.

### Data location

Models are tens of gigabytes, so the runtime defaults its data root to
`D:\AI_Workspace\local-voice-companion` rather than the system drive, and picks
the candidate with the most free space. Override with `LVC_DATA_ROOT`.

## Backends

Providers are registered, discovered, and selected — not configured by name in
the code.

- **LLM:** `ollama_llm` (Ollama or a compatible endpoint), `fake_llm`
- **ASR:** `voicebox_asr` (Voicebox-compatible service exposing `/health`,
  `/profiles`, `/transcribe`), `fake_asr`
- **TTS:** `voicebox_tts` (Voicebox-compatible `/generate/stream`), `fake_tts`
- **VAD:** `fake_vad`
- **Relay worker:** configured with `AI_RELAY_WS_URL` and `AI_RELAY_TOKEN`

Copy `config.example.json` to `config.json` and adjust endpoints for your machine. Secrets are read from environment variables; do not commit tokens or machine-specific configuration.

The compatibility providers are optional. Disable them and the runtime still
starts, still plans, and still completes a turn with the fake providers — which
is what makes the whole pipeline testable.

## Requirements

Windows 10/11, Python 3.11+, and a supported local model or speech backend. A
microphone is required only for hands-free voice input; the text interface can be
tested without one.

## Project status

The adaptive runtime, gateway, browser UI, backend discovery, diagnostics, smoke
test, and Godot sample are included. Actual speech quality and latency depend on
the ASR/TTS models installed on the host machine.

The runtime's own architecture is finished and tested. What it lacks is a real
local speech engine — see [CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) rather
than taking this paragraph's word for it.

## License

MIT

## 简体中文

这是一个**自适应本地语音运行时**。它会先探测所在机器的硬件，决定使用哪套语音识别、
语言模型与语音合成，然后基于这份方案对外提供对话能力，供浏览器、游戏引擎或其它智能体调用。

它不是一个特定引擎的外壳。Voicebox 和 Ollama 在这里只是 provider，与测试用的 fake
provider 地位相同。

项目包含：硬件探测、自适应选择（候选 → 约束 → 评分 → 决策）、完整回合状态机与打断
（barge-in）、有界队列、流式生成与合成、可移植的 bot 清单、结构化事件流，以及四层测试
（unit / integration / contract / smoke）。

文档入口：[架构](docs/ARCHITECTURE.md)、[API](docs/API.md)、
[开发状态](docs/DEVELOPMENT_STATUS.md)、[能力矩阵](docs/CAPABILITY_MATRIX.md)、
[路线图](docs/ROADMAP.md)、[安全](docs/SECURITY.md)、[ADR](docs/adr/)。

**当前状态（如实说明）**：架构已完成，测试 `203 passed, 3 skipped`。跳过的是
`@pytest.mark.hardware` 标记的用例——它们断言本机没有安装原生推理后端，如实报告而不是伪造通过。

**内置 provider 中目前还没有真正可用的本地 ASR / TTS**，真实的语音能力都来自远程兼容
适配器，因此 `cpu_only` 只能给出降级方案。补齐这一环是 PHASE 2 的目标。

Windows 首次使用可运行 `setup.ps1`，之后使用 `start.ps1`；后端或麦克风未识别时运行
`doctor.ps1`。打开 `http://127.0.0.1:17831` 即可使用。新的自适应运行时可加 `--runtime`
参数启用。请从 `config.example.json` 创建本机配置，令牌通过环境变量提供，不要提交私密凭据。

模型体积很大，默认数据目录放在 `D:\AI_Workspace\local-voice-companion`（可自动选择剩余空间
最大的盘），避免占满系统盘；可用 `LVC_DATA_ROOT` 覆盖。
