"""Local model cache discovery, download (ModelScope / HuggingFace), and cleanup."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Callable


@dataclass
class ModelSpec:
    model_id: str
    display_name: str
    backend: str
    approx_size_mb: int
    license_name: str
    description: str
    ms_repo: str = ""
    hf_repo: str = ""
    is_bundled: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "silero_vad": ModelSpec(
        model_id="silero_vad",
        display_name="Silero VAD v6 (语音活动检测 ONNX)",
        backend="vad",
        approx_size_mb=2,
        license_name="MIT",
        description="随安装包内置，纯 CPU 极速推理，用于精准区分人声与静音/停顿边界。",
        is_bundled=True,
    ),
    "funasr_fa": ModelSpec(
        model_id="funasr_fa",
        display_name="FunASR 中文强制对齐模型 (fa-zh)",
        backend="funasr",
        approx_size_mb=160,
        license_name="FunASR Model License (MIT Toolkit)",
        description="轻量高速中文字符级强制对齐模型（约 160 MB），低显存/CPU 均可 1~3 秒完成单集对齐。",
        ms_repo="iic/speech_timestamp_prediction-v1-16k-offline",
        hf_repo="funasr/fa-zh",
    ),
    "funasr_asr": ModelSpec(
        model_id="funasr_asr",
        display_name="FunASR Paraformer-ZH + FSMN-VAD (全局时间轴锚点)",
        backend="funasr",
        approx_size_mb=850,
        license_name="Apache-2.0 / MIT",
        description="用于全片快速中文识别，自动检测剪映 SRT 与达芬奇视频剪辑引起的整体或分段偏移。",
        ms_repo="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        hf_repo="funasr/paraformer-zh",
    ),
    "qwen3_aligner": ModelSpec(
        model_id="qwen3_aligner",
        display_name="Qwen3-ForcedAligner-0.6B (高精度多语种强制对齐)",
        backend="qwen3",
        approx_size_mb=1760,
        license_name="Apache-2.0",
        description="阿里 Qwen3 非自回归对齐模型（约 1.76 GB），支持中英混排、人名/短句高精度时间定位（推荐 GPU >= 4GB 可用显存）。",
        ms_repo="Qwen/Qwen3-ForcedAligner-0.6B",
        hf_repo="Qwen/Qwen3-ForcedAligner-0.6B",
    ),
}


def get_model_local_dir(model_id: str, cache_dir: Path) -> Path:
    return cache_dir / model_id


def is_model_cached(model_id: str, cache_dir: Path) -> bool:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        return False
    if spec.is_bundled:
        return True
    mdir = get_model_local_dir(model_id, cache_dir)
    if not mdir.is_dir():
        return False
    # Check that at least one substantial weight file exists
    for f in mdir.rglob("*"):
        if f.is_file() and f.suffix in {".pt", ".bin", ".safetensors", ".onnx"} and f.stat().st_size > 1_000_000:
            return True
    return False


def get_cached_model_size_mb(model_id: str, cache_dir: Path) -> float:
    spec = MODEL_REGISTRY.get(model_id)
    if spec and spec.is_bundled:
        return float(spec.approx_size_mb)
    mdir = get_model_local_dir(model_id, cache_dir)
    if not mdir.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in mdir.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


def delete_cached_model(model_id: str, cache_dir: Path) -> bool:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None or spec.is_bundled:
        return False
    mdir = get_model_local_dir(model_id, cache_dir)
    if mdir.exists():
        shutil.rmtree(mdir, ignore_errors=True)
        return True
    return False


def download_model(
    model_id: str,
    cache_dir: Path,
    *,
    use_modelscope: bool = True,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """Download a model cleanly into `cache_dir / model_id` for future offline execution."""
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        raise ValueError(f"未知模型 ID: {model_id}")
    if spec.is_bundled:
        return cache_dir

    target_dir = get_model_local_dir(model_id, cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if is_model_cached(model_id, cache_dir):
        if progress_cb:
            progress_cb(f"{spec.display_name} 已在本地缓存，跳过下载。")
        return target_dir

    # Try ModelScope first in China, then fall back to HuggingFace Hub
    last_err: Exception | None = None
    if use_modelscope and spec.ms_repo:
        try:
            if progress_cb:
                progress_cb(f"正在从 ModelScope 下载 {spec.display_name} ({spec.ms_repo})...")
            from modelscope.hub.snapshot_download import snapshot_download

            snapshot_download(
                spec.ms_repo,
                local_dir=str(target_dir),
            )
            if is_model_cached(model_id, cache_dir):
                return target_dir
        except Exception as exc:
            last_err = exc
            if progress_cb:
                progress_cb(f"ModelScope 下载未成功 ({exc})，尝试切换到 HuggingFace...")

    if spec.hf_repo:
        try:
            if progress_cb:
                progress_cb(f"正在从 HuggingFace 下载 {spec.display_name} ({spec.hf_repo})...")
            from huggingface_hub import snapshot_download as hf_snapshot_download

            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            hf_snapshot_download(
                repo_id=spec.hf_repo,
                local_dir=str(target_dir),
            )
            if is_model_cached(model_id, cache_dir):
                return target_dir
        except Exception as exc:
            last_err = exc

    raise RuntimeError(f"下载模型 {spec.display_name} 失败: {last_err}")
