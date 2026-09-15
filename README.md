# Local Voice Companion

English | [简体中文](#简体中文)

An **adaptive local voice runtime**. It probes the machine it is running on,
decides which speech recognition, language model, and speech synthesis to use,
and then serves conversations over that plan to any client — browser, game
engine, or another agent.

It is not a wrapper around one engine. Voicebox and Ollama are providers here,
exactly like the fake providers used in tests are providers — and so are the two
native engines that actually run inference in-process.

## Highlights

- **Real local ASR and TTS** — `faster_whisper_cpu` and `kokoro_tts_cpu` run
  inference on the CPU with no network and no GPU [1]
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

[1] Both are optional extras so the base install stays light. Neither is the default
LLM — see the status section.

## Documentation

| Document | Contents |
| --- | --- |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the runtime works and why |
| [API.md](docs/API.md) | Every endpoint, with the real response shapes |
| [DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md) | What works, what does not, current test results |
| [CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) | Which providers actually run today |
| [BENCHMARKING.md](docs/BENCHMARKING.md) | How latency is measured, and how it avoids inventing numbers |
| [ROADMAP.md](docs/ROADMAP.md) | Phases, including realtime duplex voice |
| [DUPLEX_FEASIBILITY.md](docs/DUPLEX_FEASIBILITY.md) | Whether GPT-style listen-while-speaking can run locally, and what it costs |
| [SECURITY.md](docs/SECURITY.md) | Threat model and secret handling |
| [CURRENT_ARCHITECTURE.md](docs/CURRENT_ARCHITECTURE.md) | PHASE 0 audit of the original codebase |
| [docs/adr/](docs/adr/) | Architecture decision records |
| [docs/PROVIDER_LICENSES.md](docs/PROVIDER_LICENSES.md) | Licence of every model and runtime, and why Piper is excluded |

## Status, stated plainly

**238 tests pass, none skipped or fabricated**, across unit, integration,
contract, and smoke tiers. Verified stable across consecutive runs.

Local voice now works offline. On a machine with the GPU disabled, the network
off and no Voicebox, `policy=cpu_only` selects `faster_whisper_cpu` +
`kokoro_tts_cpu` and completes `real WAV → ASR → transcript → LLM → text → TTS →
valid WAV`, with every stage genuinely measured:

| Stage | Provider | Median | RTF | Source |
| --- | --- | --- | --- | --- |
| ASR | `faster_whisper_cpu` (`base`, INT8) | 573.6 ms | 0.287 | measured |
| TTS | `kokoro_tts_cpu` (`kokoro-v1.1-zh`) | 1101.5 ms | 0.337 | measured |
| TTFA | end to end | 2077.8 ms | — | measured |

Measured on the reference machine with `--policy cpu_only`. RTF below 1.0 means
faster than real time; both stages clear it comfortably, which is what makes a
CPU-only conversation practical rather than merely possible.

**What is still missing, stated as plainly:** there is **no local LLM**, so an
offline reply is deterministic stub text. The runtime hears and speaks through
real local models and reasons from a template. There is deliberately no measured
LLM latency above, because inventing one from a stub is exactly the dishonesty
this project avoids. Both engines are also **single-pass** — no partial
transcripts, no progressive audio — which sets the latency floor PHASE 3 has to
attack. Cold start adds several seconds to load Kokoro's 325 MB graph and is not
in the warm numbers above.

Full detail, including what these numbers do *not* cover, in
[BENCHMARKING.md](docs/BENCHMARKING.md) and
[CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md).

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
- **ASR:** `faster_whisper_cpu` (**local**, CPU INT8), `voicebox_asr`
  (Voicebox-compatible `/health`, `/profiles`, `/transcribe`), `fake_asr`
- **TTS:** `kokoro_tts_cpu` (**local**, 103 zh voices), `voicebox_tts`
  (Voicebox-compatible `/generate/stream`), `fake_tts`
- **VAD:** `fake_vad`
- **Relay worker:** configured with `AI_RELAY_WS_URL` and `AI_RELAY_TOKEN`

Copy `config.example.json` to `config.json` and adjust endpoints for your machine. Secrets are read from environment variables; do not commit tokens or machine-specific configuration.

The local providers need one extra step, and only one. They are **optional
extras** — nothing is installed into the base environment, and nothing is
downloaded on import:

```powershell
# CPU-only local speech (~530 MB of packages, no PyTorch, no CUDA runtime)
pip install -r requirements-local.txt

# Then fetch weights explicitly. Nothing downloads silently, ever.
lvc models fetch
lvc models list          # what is present, what is missing
lvc doctor               # runtime, weights and Chinese G2P checks
```

Or take them one at a time with `requirements-local-asr.txt` and
`requirements-local-tts.txt`. From here, `policy=cpu_only` works with the
network switched off.

The compatibility providers are optional too. Disable them and the runtime still
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

Local hearing and speaking now work without network or GPU — see the status
section above for the measured numbers, and
[CAPABILITY_MATRIX.md](docs/CAPABILITY_MATRIX.md) for what is still missing.

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

**当前状态（如实说明）**：测试 `238 passed`，无跳过、无伪造通过，连续运行结果稳定。

**本地语音已真正可用（离线、纯 CPU）**：在没有 GPU、没有网络、没有 Voicebox 的机器上，
`policy=cpu_only` 会选中 `faster_whisper_cpu` 与 `kokoro_tts_cpu`，完成
`真实 WAV → ASR → 文本 → LLM → 文本 → TTS → 合法 WAV` 全链路，且每个阶段都是**实测**
而非估算：ASR 中位 573.6 ms（RTF 0.287）、TTS 中位 1101.5 ms（RTF 0.337）、TTFA
2077.8 ms。

需要额外安装一次（不会影响基础安装体积，也不会在导入时偷偷下载模型）：

```powershell
pip install -r requirements-local.txt
lvc models fetch
```

**仍需如实说明的缺口**：没有本地 LLM，离线回合的回复仍是确定性占位文本——它真的在听、
真的在说，但还不能真的思考；两个引擎都是单趟推理，没有流式部分结果，因此延迟存在下限；
冷启动加载 Kokoro 的 325 MB 图需要数秒，未计入上文的 warm 数字。

Windows 首次使用可运行 `setup.ps1`，之后使用 `start.ps1`；后端或麦克风未识别时运行
`doctor.ps1`。打开 `http://127.0.0.1:17831` 即可使用。新的自适应运行时可加 `--runtime`
参数启用。请从 `config.example.json` 创建本机配置，令牌通过环境变量提供，不要提交私密凭据。

模型体积很大，默认数据目录放在 `D:\AI_Workspace\local-voice-companion`（可自动选择剩余空间
最大的盘），避免占满系统盘；可用 `LVC_DATA_ROOT` 覆盖。
