# PHASE 4 — Product Shell & Universal Agent/Game Integration

> **状态：BLOCKED，未开工。**
> 起始条件（START CONDITION）要求 PHASE 3 达到
> `complete / tests green / pushed / clean` 之后才能开始。
> 截至 2026-09-15 的门槛核查：**未满足**，见 [第 0 节](#0-门槛状态)。

PHASE 0–3 解决的是 Runtime / Providers / Hardware Selection / Local ASR-TTS /
Realtime Duplex。**PHASE 4 第一次真正解决：普通用户到底怎么用它，以及其他
Agent、网页、游戏到底怎么接它。**

正式名称：**Product Shell & Universal Integration（产品外壳 + 通用语音 Agent 接入层）**

最终目标一句话：**Create a voice once. Use it anywhere.**
（创建一个会说话的 AI，然后把它接进任何地方。）

核心定位转变：从「语音聊天网页」变成 **Voice Runtime for Any Agent**。

---

## 0. 门槛状态（每次开始 PHASE 4 前都要重查）

不要相信本文档或聊天里的旧 commit SHA，必须自行跑：

```bash
git status
git log
git pull
read docs/DEVELOPMENT_STATUS.md
read docs/ROADMAP.md
read docs/ARCHITECTURE.md
read docs/API.md
read latest PHASE 3 docs
run full test suite
```

### 2026-09-15 的核查结果

| 门槛项 | 实测值 | 判定 |
| --- | --- | --- |
| `git log` 最新提交 | `44f6988 models list/status: ...`（PHASE 2 收尾） | ❌ PHASE 3 零提交 |
| `git status` | 仅未跟踪 `docs/PHASE3_STARTUP.md` | 树干净但无代码变更 |
| `origin/main` | `44f69887ad7cb55571e02ea6d637f558baddd20d`，与本地一致 | ❌ 无新推送 |
| PHASE 3 完成度 | 仅 3A 前置：sherpa-onnx 1.13.8 已装、模型已下到 E 盘、`src/` 未改 | ❌ 远未完成 |
| 测试 | 238 passed（PHASE 2 基线，非 PHASE 3 测试） | ⚠️ 是旧基线 |

**结论：PHASE 4 blocked。** PHASE 3 需要的 3A→3F（流式地基 / 打断 / 仲裁 / AEC /
停顿插话 / 性能优化）几乎都没做，`asr.partial` 仍零发射，多轴状态仍不存在。

---

## 1. 四大核心任务

```text
A. Product Frontend          ← 优先级最高
B. Bot / Voice Creation UX
C. Universal API + SDK
D. One-command / One-click usability
```

---

## PART A — Product Frontend

- **旧 `web/` 不再作为正式前端。** 若仍绑 legacy（`/options`、`/settings`、Voicebox、
  Ollama），明确标记为 **legacy UI**，不要在旧 UI 上无限补丁。建新目录 `frontend/` 或 `ui/`。
- **技术**：TypeScript + Vite + Preact（或 React，若生态明显更优）。
  原则：轻量、快启、易静态打包、易嵌入 Python backend。
  **不要** Next.js / SSR / 复杂 Node server / 大型全家桶。Backend 已存在，Frontend 是 client。
- **驱动方式**：前端**只能**通过 `/api/v1/*` 与 WebSocket 访问 Runtime。
  禁止直连 Voicebox / Ollama / Kokoro / Whisper，禁止知道 provider implementation。
- **信息架构**（可合并，能力必须都在）：

  ```text
  Home / Talk / Bots / Voices / Runtime / Connect / Settings
  ```

### Home

首页**不是 Dashboard**，只回答三个问题：电脑准备好了吗 / 当前 Bot 是谁 / 能直接说话了吗。

```text
Your computer is ready.
Balanced Runtime
  ✓ Speech Recognition  ✓ AI Model  ✓ Voice
Current Bot: Luna
[ Talk Now ]  [ Create Bot ]
```

**普通模式首屏禁止出现**：CTranslate2、RTF、Quantization、CUDA、ProviderDescriptor、
VRAM budget —— 这些属于 `Advanced`。

### First Run / Setup Wizard

```text
Welcome → Hardware Check → Runtime Recommendation → Voice Setup
       → LLM Setup → Create First Bot → Test Conversation → Done
```

- **Hardware Check** 显示 CPU / GPU / RAM / Microphone / Speaker，
  状态只有 `Ready / Recommended / Limited / Unavailable`。**不要**整页 JSON。
- **Runtime Recommendation** 显示 `Recommended Mode: Balanced` +
  `Speech Recognition Local / AI Model Local-API / Voice Local` +
  折叠的 `Why this setup?`（里面才放 provider / model / RAM / VRAM / latency）。

### Talk（最重要页面）

- 结构：Bot identity / 状态 / 波形 / Conversation / Controls / Latency。
- 状态必须清晰：`Listening / Thinking / Speaking / Interrupted / Paused / Offline`；
  若 PHASE 3 已有 duplex，还要有 **Listening while speaking**（用用户语言，不是术语）。
- 输入：**Microphone 是一级入口**，另支持 Text / Audio file。
- 输出：Text + Audio，并显示 live partial transcript 与 assistant streaming text（若 PHASE 3 已实现）。
- 控制至少：`Mute / Stop / Interrupt / Replay / Clear Conversation`；
  若 duplex 再加 `Turn-based / Duplex` 模式切换。

---

## PART B — Bots

- 用户创建的是 **Bot**，不是 Pipeline。一级产品对象。
- **Create Bot** 字段：Name / Avatar(optional) / Description(optional) /
  Personality / Language / Voice / AI Model preference / Conversation mode。
- **Persona**：先给模板 `Friendly Companion / Game NPC / Assistant / Tutor / Custom`，
  高级模式才编辑完整 System Prompt。不要让普通用户先面对巨大编辑框。
- **继续复用现有 `BotManifest`**，前端不新建第二套 Bot schema。
- **Bot 卡片示例**：

  ```text
  Luna
  Chinese / Female 03 / Balanced / Duplex
  [ Talk ] [ Edit ] [ Connect ]
  ```

- **Export / Import** 必须有，格式用现有 `bot.yaml`。
- **Bot 不绑定硬件**：bot 文件禁止写 `C:\...`、`RTX2070`、`specific provider required`。
  默认 `provider = auto`，换电脑重新选 runtime。

---

## PART C — Voice System

- 建立 **Voice Library**，用户看到 `Female 01 / Female 02 / Male 01 / Anime Soft / Calm /
  Energetic`，**不是** `zf_001 / zm_009`。底层 ID 保留，UI 用 `display_name`。
- 第一阶段至少：多个中文女声、多个中文男声、英语、日语（实际数量取决于现有模型，
  **不能虚构不存在的 voice**）。
- **Voice Preview**：每个 voice 有 `▶ Preview`，试听固定短句。
- **Preview Cache**：按 `voice + language + provider version` 缓存并失效，不要每次重生成。
- **Voice Clone（optional，非强制）**：用户可选择 `Preset` 或 `Clone Voice`；
  输入 WAV/MP3/M4A 或麦克风录制。
- **架构约束**：不要把 cloning 写进 Core，继续用 TTS Provider Capability：
  `supports_voice_clone / supports_voice_design / supports_preset_voices`。
- **候选 provider**：Qwen TTS / CosyVoice / F5-TTS / XTTS。**必须 benchmark，不要一次全集成。**
- **`VoiceManifest`**：`id / display_name / language / provider / type(preset|cloned|designed) /
  reference_audio / metadata`。
- **Portability**：cloned voice 依赖某 provider 时，导出记录 `preferred voice`；
  另一台机器没有该 provider 时 runtime **fallback 并告知**，bot 不能直接失效。
- **Consent**：Clone 页面必须明示 `Only clone voices you have permission to use.`
- **Storage**：reference audio 进 `LVC_DATA_ROOT`，**不进 Git**。

---

## PART D — Multilingual

- 一级功能，优先 **Chinese / English / Japanese**，后续视 provider 扩展。
- 语言能力**只能**来自 `ProviderDescriptor.languages`，**禁止 Core 硬编码**「Kokoro supports X」。
- `BotManifest.primary_language`，可选 `allowed_languages`。
- 支持 `language = auto`（ASR provider 能检测时）。
- **Fallback**：Bot 中文时用户突然说英文 → 当前 TTS 支持就继续，不支持就选兼容 voice/provider。
- **必须专门测中英混**：「我今天准备去 Starbucks 写 code。」
- **UI 语言**（简体中文 / English）与 **Bot 语言**是两件事，不要混。

---

## PART E — Universal Integration

- **Connect 页面**：每个 Bot 点击 `Connect`，选择
  `Web / JavaScript / Python / Godot / Unity / Unreal / REST / WebSocket / Agent / MCP`。
- 所有平台接入都围绕 **Bot ID / Session / Events**。
- **极简 API**：一个 Agent `POST text → audio`，最少一次调用。
- **高层端点**（基于现有 API 判断，**不要重复已有接口**）：

  ```text
  POST /api/v1/bots/{bot_id}/chat
  {"text": "你好", "speak": true}
  → {"text": "你好呀", "audio_url": "...", "session_id": "..."}
  ```

- **Realtime API**：WebSocket 用于 streaming text / streaming audio / interrupt /
  partial ASR / events。
- **Python SDK**：`local_voice_companion.client` 或 `lvc_client`。

  ```python
  from lvc import Client
  client = Client()
  luna = client.bot("luna")
  luna.say("你好")
  async for event in luna.talk(): ...
  ```

- **JavaScript SDK**：`@local-voice-companion/client` 或 `sdk/js`。
  第一阶段不一定要发 npm，但 API 要稳定。

  ```javascript
  const lvc = new LVCClient()
  const bot = lvc.bot("luna")
  await bot.say("Hello")
  ```

- **Web Embed（非常重要）**：`<script src=".../lvc.js">` → `const bot = LVC.bot("luna")`，
  任何网页都能嵌。
- **Web Widget（optional）**：`<lvc-voice bot="luna"></lvc-voice>`。
  **不能**变成「必须用 widget 才能接入」。
- **CORS**：支持 localhost 网页游戏与本地 dev server，但**默认安全**：
  config 里 `allowed_origins`，**不要无条件 `*`**。

---

## PART F — Games（一级功能）

目标：**任意 2D/3D/Web 游戏都能把 LVC 当 NPC Voice Runtime。**
游戏**完全不需要知道 provider**，只知道 `bot_id / npc_id(optional) / session_id`。

- **Godot**：升级现有集成 →

  ```gdscript
  var bot = LVC.bot("tinglan")
  bot.say("欢迎回来。")
  bot.start_voice_session()
  ```

  正式做 `addons/local_voice_companion/`（可直接拷进项目）。
  信号：`transcript_partial / transcript_final / assistant_text /
  audio_started / audio_finished / interrupted / state_changed`。
- **Unity**：提供 C# 示例（HTTP + WebSocket + PCM/WAV playback），
  第一版不必发 UPM package。
- **Unreal**：最小 integration example，Blueprint-friendly HTTP/WS 或 C++ helper，
  **不要过度开发 Plugin**。
- **Ren'Py / Python 游戏**：直接用 Python SDK 示例。
- **Web 游戏**（Phaser / Three.js / Babylon / Canvas / 普通 HTML5）：走 JavaScript SDK。
- **禁止建立「2D API / 3D API」**——这是错误抽象。Voice Runtime 与渲染引擎完全解耦。
- **NPC Session**：多个 NPC 共享 `bot_id` persona，每次交互独立 `session_id`；支持多 NPC 并存。
- **3D spatial audio 由游戏端负责**，Runtime 只返回 audio，**Core 不实现 3D sound**。
- **Lip Sync**：Runtime 可提供 audio timing（未来 phoneme/viseme），**PHASE 4 不强制**。
- **Game Event / Context API**：允许游戏注入临时 context
  （`player_entered_room`、Player HP / Location / Quest / Relationship / Time）。
- **明确不做**：NPC planning / world model / behavior tree / memory graph / RAG。
  只提供 **context injection**。

---

## PART G — Agent Integration

- 任何 Agent 可调用 `voice.speak() / voice.listen() / voice.converse()`。
- **MCP**：`voice.speak / voice.transcribe / voice.converse /
  voice.list_bots / voice.get_bot / voice.create_session`。
  MCP **仍然只是 Adapter，不应成为 Core**。
- **OpenAI-compatible**：保留并完善 `/v1/audio/speech`、`/v1/audio/transcriptions`。
- `/api/v1/voice` 只有在**明显改善使用**时才做，不要为了 API 数量造重复接口。

---

## PART H — Auth & Integration Security

- 默认 `127.0.0.1`，无需复杂账号系统。
- 若 `0.0.0.0`（LAN），**必须** API Token。
- 考虑 Per-client Integration Token；权限未来 `speak / listen / bots:read / bots:write`，
  第一版可简化为 `read-only / voice / admin`。
- Connect 页面让用户复制：`Endpoint / Bot ID / Token`。
- **UI 不显示 LLM API key 原始值**，只显示 `Configured / Not configured`。

---

## PART I / J — Runtime 页 & LLM Setup

- **Runtime 页**（高级用户）显示 ASR/LLM/TTS/VAD 的 `Provider / Model / Device /
  Latency / Memory / Status`；默认 `Auto`，允许 Manual Override。
  Mode：`Fast / Balanced / Quality / Low Memory / CPU Only / Manual`（PHASE 3 再加 Realtime/Duplex）。
- **LLM 页**：Local ↔ API 二选一。
  Local 显示 `Ollama detected` + 模型列表；API 支持 OpenAI-compatible
  （Base URL / Model / API Key）。
  API Key 只提交一次，之后显示 `••••••••`；必须有 `Test Connection` 按钮。

---

## PART K — One-command usability

**PHASE 4 必须解决当前最大产品问题**：不能再要求用户理解
venv / requirements-local / config.json / models fetch / `app.py --runtime`。

- Windows 目标：`.\setup.ps1` → `.\start.ps1` 即可。
- **`start.ps1` 必须启动新的 Adaptive Runtime**，不再默认 legacy；
  legacy 若保留则放 `start-legacy.ps1`。
- **`setup.ps1` 至少自动**：detect Python / create venv / install base / probe hardware /
  recommend dependencies / install local voice extras / download recommended models /
  create config / run doctor。
- setup 结束后直接 start server + open browser，或清晰提示下一步，但体验要顺。
- **不要求用户手改 config.json**：普通用户配置全走 UI，config.json 保留给高级用户。
- **Browser Auto Open**：server ready → `http://127.0.0.1:17831`。
- System tray 不是 PHASE 4 blocker（留到 packaging）。

---

## PART L / M — 前端 UX & 从 UI 安装模型

- 设计方向是 **AI product**，不是开发者 dashboard：clean / minimal / modern / audio-centric。
  技术数据收进 Advanced，**不要满屏表格**；状态优先（Ready / Listening / Thinking / Speaking）。
- 波形：实时 mic/playback waveform，不需要复杂 DSP visualizer。
- Empty State：`Create your first voice companion.` + `[ Create Bot ]`。
- **错误体验禁止把异常类丢给用户**。要翻译成可行动文案：
  - `ProviderUnavailable` → “Speech model isn't installed. [ Install ]”
  - 缺 LLM → “No AI model is configured. [ Use Local Ollama ] [ Connect API ]”
  - 缺 voice → “No local voice model installed. [ Install recommended voice ]”
- **UI Model Manager**：至少显示 `Installed / Available / Download size`；
  点击 Install 由 backend 调 Model Manager；必须有 progress（最好 speed + remaining）；
  下载可取消。
- **继承 PHASE 2 原则**：模型**不可静默下载**，用户必须知道 model / size / destination。

---

## PART N — Voice Clone UX

- Voice 页结构：`Preset / My Voices / Create Voice`。
- Create Voice 选择 `Clone` 或 `Design`（Design 仅 provider 支持时）。
- Clone 完成立即 `Preview`。
- Clone metadata 记录 `created_at / language / provider / reference duration`，
  **不多存隐私信息**。

---

## PART O / P — SDK 结构与示例

```text
sdk/
    python/
    javascript/
integrations/
    godot/ unity/ unreal/ renpy/ web/
```

- **SDK 不复制业务逻辑**，只做 `transport / typing / events / helpers`；
  Runtime 逻辑全在 server。
- OpenAPI 生成可以利用，但不要产生巨大难维护 client。
- **每个平台必须有 <50 行能跑的最小示例**：Web（一个 `Talk to Luna` 按钮）、
  Godot（NPC 按 E 说话）、Unity（按键发文本播放语音）、Unreal（最小 actor/component）、
  Python Agent（`bot.say(agent_output)`）。

---

## PART Q — Performance

- 前端静态 build 由 backend serve，**不能拖慢 Runtime**。
- WebSocket reconnect 必须可靠。
- 浏览器播放**不要等整个回答音频**；PHASE 3 有 streaming 就直接流式播。
- Client jitter buffer 要小，防断音，但**不能引入 500ms+ 额外延迟**。

---

## PART R — Testing

- 前端至少：`unit / API contract / E2E critical path`。
- **Critical E2E 必须覆盖**：First Run / Create Bot / Talk / Change Voice / Connect API。
- Browser E2E 建议 Playwright，只覆盖核心 flow。
- **现有 pytest 全部继续通过**（backend regression）。
- **SDK Contract Tests 必须针对真实 FastAPI Test server**。
- Godot：至少 protocol fixture；CI 无 Godot 要**清楚标记**。

---

## PART S — Packaging boundary

- PHASE 4 **不要求完整 installer**，但必须达到 `clone → setup → start → browser` 真正顺畅。
- PHASE 5 再做：`.exe` / MSI / Tauri shell / system tray / auto update。

---

## PART T — README

- 顶部**不再是 architecture**，而是 `1-command setup + screenshot + what it does`。
- 第一屏示例：

  ```text
  Local Voice Companion
  Give any AI agent a voice.
  ✓ Local ✓ Realtime ✓ Web / Game / Agent ✓ Preset or cloned voice ✓ Multilingual
  ```

- Quick Start 第一段必须是 `git clone → cd → .\setup.ps1`；第二段 `http://127.0.0.1:17831`。
- README 直接给 `bot.say(...)` 的 Python 与 JavaScript 例子。

---

## PART U — Quality bar

- **产品**：从没见过项目的人 `clone → setup → browser → create bot → talk`，
  **不读 ARCHITECTURE.md 也能成功**。
- **开发者**：游戏开发者 5–20 分钟能把 Bot 接进自己的项目。
- **Agent**：任意 Agent 只需 HTTP / WS / SDK 即可获得语音。
- **Voice**：preset 至少能正常选与用；Voice Clone 若实现必须是 optional。
- **Multilingual**：中/英/日 UI 能正确选择兼容的 ASR / TTS / Voice。

---

## PART V — 明确不做

```text
RAG / long-term memory / full NPC cognition / Live2D / 3D avatar
emotion recognition / wake word / account cloud / billing
social network / model training / complex plugin marketplace / mobile app
```

**PHASE 4 只做四个词：usable / visible / connectable / portable。**

---

## PART W — 推荐开发顺序

```text
1.  PHASE 3 merge 后 verify baseline
2.  迁移 start.ps1 → 默认新 Runtime
3.  new frontend skeleton
4.  First Run / Home / Talk / Bots
5.  Runtime / Voices / Settings
6.  Voice preview / Preset voices
7.  Connect 页面
8.  Python SDK
9.  JavaScript SDK
10. Godot addon
11. Unity / Web examples
12. Voice Clone spike（latency/dependency/license/distribution 合理才正式集成，
    否则标 experimental，不能拖住主线）
13. Multilingual matrix
14. setup 一键化
15. E2E tests
```

---

## PART X — 最终验收测试

```text
171. 尽可能 fresh checkout 重装
172. 只看 README 执行 setup.ps1 成功
173. start.ps1 自动 server + browser
174. 用户 Create Bot 成功
175. 真实麦克风 → Bot → Voice 全链路
176. 切换 Preset Voice，下一句声音真的变了
177. 中/英/日切换至少现实验证通过
178. Python bot.say("hello") 成功
179. JS bot.say("hello") 成功
180. Godot 最小 demo 播放成功
181. Web Game 最小 demo 成功
182. 多端同时连接，session 不互相污染
183. 非 localhost 无 Token 必须拒绝
184. Voice Clone：成熟才算 Complete，不成熟不要标 Complete
```

---

## PART Y — 最终报告章节

```text
A. Starting baseline（HEAD / PHASE 3 status / tests）
B. Frontend（pages / framework / build size）
C. First Run（真实 fresh install 时间）
D. One-command（setup.ps1 是否真的一键完成）
E. Talk（真实语音测试）
F. Bots（Create/Edit/Export/Import）
G. Voices（preset 数量，按语言）
H. Voice Clone（READY / EXPERIMENTAL / NOT IMPLEMENTED + provider）
I. Languages（真实测试矩阵）
J. Python SDK 示例    K. JS SDK 示例
L. Godot   M. Unity   N. Web Game
O. API/WebSocket 稳定性   P. Multi-client
Q. Security（Token / CORS）
R. Tests（backend / frontend / SDK / E2E）
S. Bugs found（真实 defect）
T. Remaining gaps（不要营销）
U. Git（final HEAD / commits / origin sync / working tree）
```

---

## 关键的一票否决项（动工前再确认一遍）

1. **PHASE 3 必须先 complete + green + pushed + clean**，否则不开始。
2. 前端**不得**绕过 `/api/v1/*` 与 WS 直连任何 provider。
3. Bot / BotManifest **不得**绑定硬件或具体 provider（默认 `auto`）。
4. 语言能力只能来自 `ProviderDescriptor.languages`，**不得**在 Core 硬编码。
5. 模型下载**不得静默**，必须有 size / destination / progress。
6. CORS **不得**默认 `*`。
7. SDK **不得**复制业务逻辑。
8. 不得建立「2D API / 3D API」抽象；Core **不得**实现 3D spatial audio。
