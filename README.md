# Local Voice Companion

English | [简体中文](#简体中文)

A local-first voice conversation gateway that connects speech recognition, an LLM, and speech synthesis behind one browser interface. It is designed as a practical starting point for private desktop assistants, game characters, kiosks, and voice-enabled prototypes.

## Highlights

- Microphone voice activity detection and streaming turn handling
- Ollama-compatible local language-model integration
- Pluggable local or relay-backed speech recognition
- Voicebox-compatible speech synthesis with queued playback
- Browser-based runtime settings and health checks
- Optional Windows GPU worker for split-machine setups
- Minimal Godot integration example

## Quick start on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
Copy-Item config.example.json config.json
.\.venv\Scripts\python app.py
```

Open `http://127.0.0.1:17831`. For the guided setup, run `setup.ps1` once and use `start.ps1` afterward. Run `doctor.ps1` when a model endpoint, speech service, or microphone is not detected.

## Backends

- **LLM:** Ollama or another compatible local endpoint
- **ASR/TTS:** a Voicebox-compatible service exposing `/health`, `/profiles`, `/transcribe`, and `/generate/stream`
- **Relay worker:** configured with `AI_RELAY_WS_URL` and `AI_RELAY_TOKEN`

Copy `config.example.json` to `config.json` and adjust endpoints for your machine. Secrets are read from environment variables; do not commit tokens or machine-specific configuration.

## Requirements

Windows 10/11, Python 3.11+, and a supported local model or speech backend. A microphone is required only for hands-free voice input; the text interface can be tested without one.

## Project status

The gateway, browser UI, backend discovery, diagnostics, smoke test, and Godot sample are included. Actual speech quality and latency depend on the ASR/TTS models installed on the host machine.

## License

MIT

## 简体中文

这是一个本地优先的语音对话网关，通过统一的浏览器界面连接语音识别、语言模型和语音合成，可作为私人桌面助手、游戏角色、展台或语音原型的开发起点。

项目支持麦克风 VAD、流式对话、本地 Ollama 模型、可插拔 ASR、Voicebox 兼容 TTS、运行状态检查、Windows GPU 工作节点和 Godot 接入示例。

Windows 首次使用可运行 `setup.ps1`，之后使用 `start.ps1`；后端或麦克风未识别时运行 `doctor.ps1`。打开 `http://127.0.0.1:17831` 即可使用。请从 `config.example.json` 创建本机配置，令牌通过环境变量提供，不要提交私密凭据。
