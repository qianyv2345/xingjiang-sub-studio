<div align="center">
  <img src="assets/app_icon.png" alt="星酱声幕 Logo" width="112" />
  <h1>星酱声幕 · XingJiangSubStudio</h1>
  <p><strong>v1.1.0</strong> · AI 短剧智能音频与字幕修正助手</p>
  <p>专为 <strong>Windows 11 + DaVinci Resolve Studio 20 + 剪映</strong> AI 短剧后期工作流设计的本地离线桌面工具（支持 NVIDIA CUDA GPU 加速与 CPU 自动回退）</p>
</div>

---

## 一、核心能力与亮点

1. **三种灵活工作流模式（支持视频 / 剧本台词文稿 / SRT 多集自动配对）**：
   - **文稿匹配（剧本匹配模式）**：拖入粗剪视频（`.mp4 / .mov / .mkv`）与台词文稿（`.txt / .md`），通过语音识别与本地序列匹配自动对齐时间轴、绑定说话人轮次，并稳健应对删减词、顺位调整与重复台词；
   - **自动生成字幕（纯视频直出）**：无需先行在剪映出字，直接基于 `Silero VAD v6` + `Qwen3-ASR-0.6B / 1.7B` 或 `FunASR Paraformer-ZH` 自动识别中英对白并完成强制对齐；
   - **修正已有 SRT（剪映对齐精修）**：拖入视频与剪映导出的 `.srt`，重算字词级起止帧并按模板重新断句，绝不擅自改写原台词文本；
   - 内置 `FFmpeg / ffprobe` 自动检测分辨率、帧率、画幅比例（竖屏/横屏/方屏）、多音轨与音画起始 PTS 偏移。

2. **中英双语强制对齐与自定义排版模板**：
   - 支持 **Qwen3-ForcedAligner-0.6B**（约 1.76 GB，中英多语高精字/词级对齐）与 **FunASR `fa-zh`**（约 160 MB，轻量高速中文对齐）双引擎，并结合局部声学包络能量与短语音尾音校准边界；
   - 内置 **竖屏 9:16**（默认 9 字/行）、**横屏 16:9**（默认 20 字/行）、**EN 26字**（英文词数与连词保护规则）预设，支持 **1 行 / 2 行** 切换、每行字数滑块实时预览、小数/千分位/连字符（`3.14`、`1,000`、`AI-2026`）及角色名保护；
   - **安全防抢跑预留 + 同句衔接防闪烁**：前后预留支持 `按帧数 (随视频帧率)`（默认前后各 3 帧）或固定毫秒自定义，且**仅向真实静音/停顿区间扩展**，0ms 连续说话断句边界绝不入侵上一条尾字；配合同句 1ms 无缝衔接消除达芬奇闪黑帧。

3. **工作室级 48kHz 人声音量均衡**：
   - 平滑动态增益曲线 + 情绪动态保留 + 静音防泵音门限 + 5ms 前瞻 True-Peak 真峰值限制器，解决片段响度不均、句首句尾忽大忽小与破音问题；
   - 提供 **轻度 / 标准 / 强力** 三档与高级参数调优，支持 **A · 原片 / B · 均衡后** 实时切换对比试听、波形图上人工增益节点编辑（支持 `Ctrl+Z / Ctrl+Y` 撤销重做）与仅音频重算（**重新应用均衡曲线**）。

4. **断点阶段缓存与高级单步控制**：
   - 基于媒体指纹与参数哈希的分阶段持久化缓存（提取音频 / VAD / ASR / 文稿匹配 / 强制对齐 / 排版 / 音频均衡），修改排版或参数仅重算受影响阶段；
   - 工具栏提供 **高级单步** 菜单（支持仅重跑识别、仅重新文稿匹配、仅重新对齐、仅重新排版、仅重新均衡、重新导出与强制重跑）及 **纯净白 / 曜石黑 / 浅樱粉** 三套主题。

5. **DaVinci Resolve Studio 20 无缝导出**：
   - 自动输出到视频同级 `智能修正输出/`（绝不覆盖原始视频与原 SRT），包含 `48kHz 24-bit/16-bit WAV`、`UTF-8 无 BOM + LF 换行且严格无重叠 SRT`、`JSON 检测报告` 以及可选无损视频流替换音轨 `.mp4`。

---

## 二、快速使用指南（独立工作流 / 剪映 → 星酱声幕 → 达芬奇）

### 第一步：准备离线模型
1. 启动程序（桌面打包版 `XingJiangSubStudio.exe` 或源码运行 `.venv\Scripts\python.exe -m drama_studio.app`）。
2. 点击右上工具栏 **「模型与缓存」**：
   - **默认通用中英推荐**：下载 **Qwen3-ASR-0.6B (~1.85 GB)** + **Qwen3-ForcedAligner-0.6B (~1.76 GB)**（RTX 3050 6GB 及以上显卡可批处理高速运行）；
   - **轻量纯中文高速方案**：下载 **FunASR Paraformer-ZH (~850 MB)** + **FunASR `fa-zh` (~160 MB)**；
   - **仅粗估语音分布**：内置 **Silero VAD v6** 随包集成，无需额外下载即可启用基础分段兜底。

### 第二步：导入素材与选择模式
- 拖入单集或批量剧集（支持 `.mp4 / .mov / .mkv`，可选同时拖入同名 `.txt / .md` 台词文稿或 `.srt` 字幕），或点击左侧 **「导入视频/文稿/SRT」** / **「导入剧集文件夹」**。
- 可使用 **「绑定/切换文稿」**、**「绑定/切换 SRT」** 调整配对，或选中剧集音轨（多音轨视频可指定对白轨）。

### 第三步：调整排版与均衡规则
- 右侧排版栏自动按画面横竖屏匹配预设，可切换 **1 行 / 2 行**、拖动字数滑块、设置 **前预留 / 后预留** 与 **同句衔接防闪烁**（修改后可点表格上方重新断句即时重排）。
- 中上方选择均衡档位（**轻度 / 标准 / 强力**）或展开 **高级参数** 微调。

### 第四步：一键处理、审听对比与局部精修
- 在右上角选择 **当前集** 或 **批量全部**，点击 **「✦ 开始处理」**；若只想调整某一步骤，可直接展开 **「高级单步 ▾」**。
- 处理完成后可在波形图上 **A · 原片 / B · 均衡后** 实时对比试听、拖拽人工增益节点或微调字幕边界，确认无误后通过 **异常检查** 复核低置信度区间。

### 第五步：导回 DaVinci Resolve Studio 20
1. 输出目录默认位于原视频旁 `智能修正输出/`（也可指定自定义导出目录）。
2. 将 `*_修正音频.wav` 放在新音轨 `00:00:00:00` 并静音原音轨，将 `*_修正字幕.srt` 拖入字幕轨即可（已严格保证时间线严格递增、无重叠且与首帧同步）。

---

## 三、开发运行、测试与增量打包

### 从源码运行
```powershell
.venv\Scripts\python.exe -m drama_studio.app
```
> 若从源码运行且尚未打包，请将 LGPL FFmpeg/ffprobe 放入 `vendor/ffmpeg/bin/`（包含 `ffmpeg.exe` 与 `ffprobe.exe`），或配置系统 `PATH` / `DRAMA_STUDIO_FFMPEG_DIR` 环境变量；也可直接使用 GitHub Releases 中的 Windows 独立发行包（已内置）。

### 运行全量自动化测试
```powershell
.venv\Scripts\python.exe -m pytest -q
```

### 打包为 Windows 独立可执行程序
```powershell
# 日常开发增量重打包（复用已分析依赖图，改业务代码快速重链接并自动同步变更资源与 EXE 图标）
.venv\Scripts\python.exe scripts\build_exe.py --incremental --require-qwen

# 完整全新清理打包（正式发版推荐）
.venv\Scripts\python.exe scripts\build_exe.py --require-qwen
```
> 产物位于 `dist\XingJiangSubStudio\XingJiangSubStudio.exe`（内置 LGPL FFmpeg/ffprobe）。可使用 `--smoke-test` 或 `--smoke-e2e <video.mp4> <sub.srt> <out_dir>` 验证可执行文件闭环。

---

## 四、已知限制与注意事项

1. **已混音 BGM / 音效分离限制**：若粗剪视频中人声已与大音量背景音乐或爆炸音效混为同一音轨，任何无源分离均无法做到 100% 无损人声分离；程序会自动探测较高非语音电平并给出风险提示，建议优先使用「轻度」或「标准」均衡模式并结合 A/B 试听确认。
2. **达芬奇与本地模型共用显存**：若与 DaVinci Resolve Studio 20 同时开启，显存占用较高时程序会自动控制并发 batch 或在 OOM / 低显存下平滑回退至更小 batch / CPU 执行。
3. **不擅自改写原台词原则**：在「修正已有 SRT」与「文稿匹配」模式下，工具不会篡改用户给定的同音字/人名，仅做声学时间戳对齐、断句与规范化标点排版。

---

## 五、版本更新记录

### v1.1.0 (2026-11-19)
- **全新 Logo 与多主题体验**：更换全新星空角色圆角图标（`assets/app_icon.png` / `app_icon.ico`），支持 `纯净白` / `曜石黑` / `浅樱粉` 三套界面主题与主窗口版本号展示。
- **ASR 直出与剧本台词匹配工作流**：新增 `Qwen3-ASR-0.6B / 1.7B` 与 `FunASR Paraformer-ZH` 自动识词生成字幕，新增 `.txt / .md` 剧本台词匹配引擎（支持顺位偏移、删改句与角色轮次绑定），并补齐 **英文 (`EN 26字`)** 分段预设。
- **字级对齐与静音安全预留优化**：新增声学包络能量与短首字/尾音校准，修复连续说话边界（0ms 间隙）开启前/后预留时抢跑“吃字”问题；预留缓冲严格只在真实静音停顿内扩展。
- **端到端性能与增量打包加速**：优化 `Silero VAD` 缓冲复用、`Qwen3-ASR` 按长排序自适应显存批推理、`Qwen3-ForcedAligner` 批量对齐、K 加权响度向量化与稀疏 4x True-Peak 扫描，并在 `build_exe.py` 中新增 `--incremental` 安全增量重打包与图标更新自动重嵌机制。
- **稳定化与阶段缓存**：引入分阶段断点缓存与 **高级单步** 菜单，修复 Windows 下覆盖写音频时的文件锁 `[WinError 5]` 冲突。

### v1.0.0 (2026-11-17)
- 首个公开版本：实现 `FunASR fa-zh` + `Qwen3-ForcedAligner-0.6B` 字幕对齐、DP 语义断句、48kHz 人声动态均衡与 DaVinci Resolve 20 导出闭环。

---

## 六、开源许可与第三方组件致谢

### 开源许可
- **本项目源码**：采用 [MIT License](LICENSE) 开源许可协议。
- **FFmpeg / ffprobe**：随发行包内置或作为外部子进程调用的 FFmpeg（[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) LGPL 构建）遵循 [LGPL v3](vendor/ffmpeg/LICENSE.txt) / LGPL v2.1 协议。
- **语音模型权重**：首次使用时按需下载缓存的第三方语音模型（FunASR `fa-zh` / `paraformer-zh`、`Qwen3-ASR`、`Qwen3-ForcedAligner-0.6B` 等）遵循各自上游模型许可协议。

### 第三方开源组件致谢
本项目依赖并受益于以下优秀开源项目（完整版本、许可证与职责说明请参阅 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)，技术选型对比见 [docs/开源方案调研与技术选型.md](docs/开源方案调研与技术选型.md)）：

- **媒体探测与封装**：[FFmpeg / ffprobe](https://ffmpeg.org/)（[BtbN LGPL Builds](https://github.com/BtbN/FFmpeg-Builds)）
- **桌面图形界面**：[PySide6 (Qt for Python)](https://www.qt.io/qt-for-python)（LGPL-3.0）
- **VAD、识别与强制对齐**：[Silero VAD](https://github.com/snakers4/silero-vad)、[ONNX Runtime](https://onnxruntime.ai/)、[FunASR](https://github.com/modelscope/FunASR)、[Qwen3-ASR (`Qwen3-ASR` / `Qwen3-ForcedAligner-0.6B`)](https://github.com/QwenLM/Qwen3-ASR)、[PyTorch](https://pytorch.org/)
- **响度计量与音频信号处理**：[pyloudnorm](https://github.com/csteinmetz1/pyloudnorm)、[NumPy](https://numpy.org/)、[SciPy](https://scipy.org/)、[python-soundfile](https://github.com/bastibe/python-soundfile)
- **字幕解析与中文分词**：[pysubs2](https://github.com/tkarabela/pysubs2)、[jieba](https://github.com/fxsjy/jieba)
- **模型下载与打包分发**：[ModelScope](https://modelscope.cn/)、[Hugging Face Hub](https://github.com/huggingface/huggingface_hub)、[PyInstaller](https://pyinstaller.org/)
