# Local Voice Companion

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Runs offline](https://img.shields.io/badge/offline-capable-brightgreen.svg)

English | [简体中文](#简体中文)

**A voice assistant that runs on your own computer.**

You talk. It writes down what you said, asks a language model for an answer,
and speaks the answer back — all three steps on your machine, with nothing
uploaded and no account to sign up for.

```
   🎙 you speak
        ↓   speech → text        your machine
        ↓   think                a model you choose
        ↓   text → speech        your machine
   🔊 it answers
```

It runs as a small local server with a page you open in your browser. The same
server speaks plain HTTP and WebSocket, so a game, a script, a shortcut key, or
another agent can use the assistant too.

## Why you might want it

- **Yours.** Audio and transcripts never leave the machine. No cloud account, no
  usage billing, no telemetry.
- **No graphics card required.** Everything runs on the CPU, and it picks a
  combination that actually fits your machine when it starts.
- **Keep talking with the network off.** Once the models are downloaded, it
  works offline.
- **Bring your own brain.** Point it at Ollama, or at any compatible service. You
  choose who does the thinking.
- **Characters you can carry around.** Personality, voice and settings save to a
  single file you can move to another machine.
- **Open source, MIT licensed.** Read it, change it, ship it.

## Requirements

- Windows 10/11 with Python 3.11 or newer
- A microphone — optional, you can type instead
- Roughly 1 GB of free disk if you want speech to run locally (models plus their
  runtime, downloaded once)

## Install

Open PowerShell in the folder where you want it:

```powershell
git clone https://github.com/nanhudev/local-voice-companion.git
cd local-voice-companion
py -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
Copy-Item config.example.json config.json
```

Prefer a guided install? Run `setup.ps1` once instead — it does exactly these
steps and creates `config.json` for you.

### Turn on speech that stays local

This extra step puts the listening and speaking on your machine, so nothing has
to go outside for either. It is downloaded once, and only when you ask:

```powershell
.\.venv\Scripts\pip install -r requirements-local.txt
.\.venv\Scripts\python lvc.py models fetch
.\.venv\Scripts\python lvc.py doctor          # checks everything is in place
```

## Run it

```powershell
.\start.ps1
```

Then open **<http://127.0.0.1:17831>**.

> `.\.venv\Scripts\python app.py` does the same thing without the wrapper script.

## Use it

1. Open the page above and allow microphone access.
2. **Talk.** It notices when you stop speaking and answers out loud.
3. **Or type** in the box at the bottom — useful for checking the whole chain
   without saying a word.
4. On the right-hand panel, pick a different voice, model or personality. Save
   and warm up when you change something.

### From the command line

| Command | What it tells you |
| --- | --- |
| `python lvc.py doctor` | What is missing, and how to fix it |
| `python lvc.py models list` | Every model it knows about |
| `python lvc.py models status` | What is already downloaded |
| `python lvc.py probe` | What your machine has |
| `python lvc.py plan` | Which combination it picked, and why |
| `python lvc.py where` | Where files are stored |

Replace `python` with `.\.venv\Scripts\python` if the virtual environment is not
active.

### From your own program

Create a session, then send a message. Everything answers over the same server:

```bash
# 1. a character (optional — there is a usable default)
curl -X POST http://127.0.0.1:17831/api/v1/bots \
     -H "Content-Type: application/json" -d '{"id":"demo","name":"Demo"}'

# 2. a conversation
curl -X POST http://127.0.0.1:17831/api/v1/sessions \
     -H "Content-Type: application/json" -d '{"bot_id":"demo"}'

# 3. one line of dialogue, spoken audio returned with the reply
curl -X POST http://127.0.0.1:17831/api/v1/sessions/<session-id>/turns \
     -H "Content-Type: application/json" -d '{"text":"你好","speak":true}'
```

For live conversation — microphone frames going in, partial transcripts and audio
coming back — connect to `ws://127.0.0.1:17831/api/v1/sessions/<session-id>/stream`
and send `{"type":"audio.start"}`, `{"type":"audio.frame","audio_base64":"…"}`,
`{"type":"audio.end"}`.

There is a ready-made Godot example in `godot/`, and the complete endpoint
reference in [docs/API.md](docs/API.md).

## Common questions

**Does it need the internet?** After the models are fetched, no. Until then, only
to download.

**Where did it put those big model files?** `python lvc.py where` prints every
path. It avoids the system drive when it can; set `LVC_DATA_ROOT` to put them
somewhere specific.

**Microphone or voices not detected?** Run `python lvc.py doctor`.

**It answers weirdly.** The reply comes from whatever language model you pointed
it at. Change it in the panel or in `config.json`.

## Where the project is heading

- **A brain that needs no external service** — so a fully offline install really
  is fully offline. Today the reply comes from a model you supply, usually
  Ollama.
- **Seeing your words appear while you speak**, instead of waiting for you to
  finish.
- **Talking over it.** Interrupting mid-sentence already works; making that feel
  natural in every setup is the current work.
- **Correct speaker support** — right now headphones give the best result.

Details are tracked in [docs/ROADMAP.md](docs/ROADMAP.md) and
[docs/DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md).

## Contributing

This is an open source project and issues and pull requests are welcome. If you
change behaviour, please run the suite first — it is honest, and skipped tests
mean something is genuinely missing:

```powershell
.\.venv\Scripts\python -m pytest
```

Design decisions are archived in [docs/adr/](docs/adr/), which is the fastest way
to understand why something is built the way it is.

## License

MIT — see [LICENSE](LICENSE).

Third-party model and runtime licences are listed in
[docs/PROVIDER_LICENSES.md](docs/PROVIDER_LICENSES.md). Everything bundled is
permissively licensed; Piper was evaluated and left out on licensing grounds.

---

## 简体中文

![许可证：MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![可离线运行](https://img.shields.io/badge/offline-capable-brightgreen.svg)

**一个跑在你自己电脑上的语音助手。**

你说话，它把你说的转成文字，交给语言模型思考，再把回答合成成语音播放出来。
三步都在本机完成，音频不上传，也不需要注册任何账号。

```
   🎙 你说话
        ↓   语音 → 文字        在你电脑上完成
        ↓   思考               由你指定的模型完成
        ↓   文字 → 语音        在你电脑上完成
   🔊 它回答
```

它以一个本地小服务的形式运行，附带一个浏览器页面可以直接用；同一个服务同时提供
标准 HTTP 与 WebSocket 接口，所以游戏、脚本、快捷键或者别的智能体也能接进来。

## 为什么值得一试

- **东西是自己的。** 音频和转写结果不出本机，没有云账号、没有按量计费、没有数据回传。
- **不需要独立显卡。** 全部跑在 CPU 上，启动时会自动挑一套适合你这台机器的组合。
- **断网也能继续用。** 模型下载完成后，断网照样对话。
- **大脑由你指定。** 可以指向 Ollama 或任何兼容服务，由你决定谁来思考。
- **角色可以带走。** 性格、声音和设置能存成一个文件，换台机器直接导入。
- **MIT 开源。** 可以读、可以改、可以发布。

## 环境要求

- Windows 10/11，Python 3.11 及以上
- 麦克风（可选，打字也能用）
- 想让语音完全跑在本地，留出约 1 GB 磁盘（模型及其运行时，只下载一次）

## 安装

在你准备放项目的目录里打开 PowerShell：

```powershell
git clone https://github.com/nanhudev/local-voice-companion.git
cd local-voice-companion
py -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
Copy-Item config.example.json config.json
```

想要引导式安装，可以直接运行一次 `setup.ps1`，它做的就是上面这几步，并顺便生成
`config.json`。

### 开启本地语音（推荐）

这一步把"听"和"说"都放到本机完成，两件事都不再需要外部服务。只下载一次，而且
只在你明确执行时才下载：

```powershell
.\.venv\Scripts\pip install -r requirements-local.txt
.\.venv\Scripts\python lvc.py models fetch
.\.venv\Scripts\python lvc.py doctor          # 检查是否都已就位
```

## 启动

```powershell
.\start.ps1
```

然后打开 **<http://127.0.0.1:17831>**。

> 不用封装脚本的话，`.\.venv\Scripts\python app.py` 效果完全一样。

## 怎么用

1. 打开上面的网址，允许使用麦克风。
2. **说话。** 它会自己判断你说完了，然后出声回答。
3. **也可以打字**——底部的输入框走的是同一条链路，不说话也能验证整体是否正常。
4. 右侧面板可以换声音、换模型、换性格；改完点保存并暖机。

### 命令行

| 命令 | 告诉你什么 |
| --- | --- |
| `python lvc.py doctor` | 缺什么、怎么补 |
| `python lvc.py models list` | 它认识哪些模型 |
| `python lvc.py models status` | 哪些已经下载好了 |
| `python lvc.py probe` | 你这台机器的配置 |
| `python lvc.py plan` | 它选中了什么组合，原因是什么 |
| `python lvc.py where` | 各类文件的存放位置 |

如果虚拟环境没激活，把 `python` 换成 `.\.venv\Scripts\python`。

### 在自己的程序里调用

先建会话，再发消息，都对着同一个服务：

```bash
# 1. 建一个角色（可选，有可用默认值）
curl -X POST http://127.0.0.1:17831/api/v1/bots \
     -H "Content-Type: application/json" -d '{"id":"demo","name":"Demo"}'

# 2. 开一个会话
curl -X POST http://127.0.0.1:17831/api/v1/sessions \
     -H "Content-Type: application/json" -d '{"bot_id":"demo"}'

# 3. 说一句话，回话里带上合成好的语音
curl -X POST http://127.0.0.1:17831/api/v1/sessions/<会话id>/turns \
     -H "Content-Type: application/json" -d '{"text":"你好","speak":true}'
```

要做实时对话——麦克风持续送帧、边识别边出音频——连接
`ws://127.0.0.1:17831/api/v1/sessions/<会话id>/stream`，依次发送
`{"type":"audio.start"}`、`{"type":"audio.frame","audio_base64":"…"}`、
`{"type":"audio.end"}`。

`godot/` 里有一份可直接用的 Godot 示例，完整接口清单见
[docs/API.md](docs/API.md)。

## 常见问题

**需要联网吗？** 模型下载完之后不需要；在那之前只在下载时要。

**那些大模型文件放哪了？** `python lvc.py where` 会列出全部路径。默认会避开
系统盘，也可以用 `LVC_DATA_ROOT` 指定位置。

**找不到麦克风或声音？** 运行 `python lvc.py doctor`。

**回答不太对。** 回答来自你指定的语言模型，可以在面板或 `config.json` 里换一个。

## 接下来的方向

- **不再依赖外部服务的本地大脑** —— 现在思考环节由你提供的模型完成（通常是
  Ollama），还没做到"装完就真的什么都不用配"。
- **边说边出字** —— 不用等你说完才显示。
- **可以随时插话打断** —— 中途打断已经能用，目前在把它打磨到各种环境下都自然。
- **完善外放支持** —— 现阶段戴耳机的效果最好。

进展记录在 [docs/ROADMAP.md](docs/ROADMAP.md) 与
[docs/DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md)。

## 参与贡献

这是一个开源项目，欢迎提 issue 和 PR。改动行为前请先跑测试——这里的测试不会放水，
出现 skip 说明确实缺了东西：

```powershell
.\.venv\Scripts\python -m pytest
```

设计决策都归档在 [docs/adr/](docs/adr/) 里，想搞清某处为什么这样写，看它最快。

## 许可证

MIT，见 [LICENSE](LICENSE)。

第三方模型与运行时的许可证见 [docs/PROVIDER_LICENSES.md](docs/PROVIDER_LICENSES.md)。
收录的模型均为宽松许可；Piper 经评估因许可问题没有纳入。
