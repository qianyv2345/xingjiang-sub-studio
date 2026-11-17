# 第三方开源组件与许可证清单

本软件使用并集成了以下经过实际调研与验证的成熟开源技术组件。所有组件均按照对应开源许可证条款合规集成或作为独立命令行进程调用：

| 组件名称 | 验证版本 | 许可证 | 项目地址 | 在本软件中的职责 |
| :--- | :--- | :--- | :--- | :--- |
| **FFmpeg / ffprobe** | n8.1.3-35 (BtbN LGPL) | LGPL v2.1 / v3 | https://ffmpeg.org/ | 视频/音轨探测、时间基准偏移校验、48kHz WAV 提取、可选无损视频流 MP4 音轨封装 |
| **PySide6** (`PySide6-Essentials`, `PySide6-Addons`) | 6.12.0 | LGPL-3.0 | https://www.qt.io/qt-for-python | Windows 11 原生图形界面、HiDPI 波形时间轴绘制、`QtMultimedia` A/B 试听 |
| **Silero VAD** | 6.2.3 | MIT | https://github.com/snakers4/silero-vad | ONNX 语音活动检测，精准分离人声、停顿与底噪区间 |
| **ONNX Runtime** | 1.32.0 | MIT | https://onnxruntime.ai/ | 轻量推理运行时，免 PyTorch 依赖极速执行 Silero VAD |
| **FunASR** (`fa-zh` / `paraformer-zh`) | 1.4.26 | MIT (工具链) / FunASR Model License (模型) | https://github.com/modelscope/FunASR | 轻量高速中文字符级强制对齐及可选全局时间轴 ASR 锚点 |
| **Qwen3-ASR (`Qwen3-ForcedAligner-0.6B`)** | qwen-asr 0.0.6 | Apache-2.0 | https://github.com/QwenLM/Qwen3-ASR | 高精度中英混排非自回归强制对齐模型（Apache-2.0 模型与代码） |
| **pyloudnorm** | 0.2.0 | MIT | https://github.com/csteinmetz1/pyloudnorm | ITU-R BS.1770-4 / EBU R128 综合响度（LUFS）与动态范围计量 |
| **pysubs2** | 1.9.1 | MIT | https://github.com/tkarabela/pysubs2 | SRT 字幕文件结构化解析与格式基础支持 |
| **jieba / jieba_pyfast** | 0.42.1 / 4.0.1 | MIT | https://github.com/fxsjy/jieba | 中文分词与词性标注，用于智能语义断句边界与词组保护 |
| **PyTorch / torchaudio** | 2.8.0+cu126 | BSD-3-Clause | https://pytorch.org/ | GPU CUDA / CPU 神经网络强制对齐计算引擎 |
| **NumPy / SciPy / soundfile** | 2.5.4 / 1.18.1 / 0.14.0 | BSD-3-Clause | https://scipy.org/ | 48kHz 多声道重采样、滤波、软拐点压缩、前瞻真峰值限制与 WAV 读写 |
| **ModelScope / huggingface-hub** | 1.40.3 / 0.36.2 | Apache-2.0 | https://modelscope.cn/ | 首次运行按需下载并缓存本地开源语音模型 |
| **PyInstaller** | 6.22.3 | GPL v2 (含商业/独立应用打包例外) | https://pyinstaller.org/ | Windows 11 独立可执行程序打包分发 |

## 调研对比但未作为核心默认路径的参考开源项目

- **voxlevel** (`0.2.0`, MIT): 提供了 `Silero VAD ONNX + 滑窗 RMS + 静音插值 + Lookahead Limiter` 的优秀设计参考，但因其硬性限制仅支持 `8000/16000 Hz 单声道` 音频且不测量 EBU R128 LUFS，未满足达芬奇 48kHz 影视后期质量需求，故在其思路基础上自研升级为 48kHz 多声道、字幕窗感知、情绪保留与真峰值限幅引擎。
- **ffmpeg-normalize** (`1.45.0`, MIT): 适合整片线性响度标准化，但难以消除单集内部前小后大、耳语与喊叫或一两字突变的局部响度异常，适合作为参考基线或全局辅助。
- **WhisperX / sub-align** (`3.8.6` BSD-2-Clause / `0.3.0` MIT): 基于 `wav2vec2-large-xlsr-53-chinese-zh-cn` (~1.2GB) 与 `faster-whisper`，中文字符级边界精度与依赖体积在短剧场景下不如 `FunASR fa-zh` (~160MB) 与 `Qwen3-ForcedAligner-0.6B` (~1.76GB)。
- **HanLP** (`2.1.7` / `2.1.5`, Apache-2.0): 深度学习中文 NLP 精度高，但模型体积和启动开销大，在当前软件中定位为可选增强后端，默认采用 `jieba + 词性标注 + 保护词典 + 动态规划` 毫秒级离线引擎。
