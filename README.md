# Local Voice Companion / 本地语音伙伴

一个低延迟、可本地运行的 ASR → LLM → TTS 语音对话网关，提供浏览器控制台、Windows GPU worker 与 Godot 客户端示例。

A low-latency, local-first ASR → LLM → TTS voice gateway with a browser console, an optional Windows GPU worker, and a Godot client example.

## Highlights / 特性

- Microphone VAD and streaming turn handling / 麦克风 VAD 与流式轮次处理
- Local Ollama-compatible LLM integration / 本地 Ollama 兼容模型
- Local or relay-backed ASR / 本地或中继 ASR
- Voicebox-compatible TTS and queued playback / Voicebox 兼容 TTS 与队列播放
- Runtime settings UI and health endpoints / 运行时设置页面与健康检查
- Godot integration sample / Godot 接入示例

## Quick start / 快速开始

```powershell
py -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
Copy-Item config.example.json config.json
.\.venv\Scripts\python app.py
```

Open `http://127.0.0.1:17831`. Configure model, ASR, and TTS endpoints in `config.json`. Remote relay credentials are read from `AI_RELAY_TOKEN`; never commit the token.

打开 `http://127.0.0.1:17831`。在 `config.json` 中配置模型、ASR 与 TTS。远端中继凭据通过 `AI_RELAY_TOKEN` 读取，请勿提交密钥。

## Worker / Windows 工作节点

`windows_worker.py` supports environment-based configuration: `AI_RELAY_WS_URL`, `AI_RELAY_TOKEN`, `FFMPEG_PATH`, `VOICEBOX_EXE`, and `VOICEBOX_DATA_DIR`.

## License

MIT
