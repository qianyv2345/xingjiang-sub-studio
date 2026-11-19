"""Local model cache discovery, integrity verification, download (ModelScope / HuggingFace), and cleanup."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Callable

from drama_studio.align.runtime import MODEL_POOL, get_disk_free_mb, get_gpu_vram_info


@dataclass
class ModelSpec:
    model_id: str
    display_name: str
    backend: str
    role: str  # "vad" | "asr" | "aligner"
    languages: str  # "zh" | "zh,en"
    approx_size_mb: int
    min_vram_mb: int
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
        role="vad",
        languages="zh,en",
        approx_size_mb=2,
        min_vram_mb=0,
        license_name="MIT",
        description="随安装包内置，纯 CPU 极速推理，用于精准区分人声与静音/停顿边界及 ASR 分段。",
        is_bundled=True,
    ),
    "qwen3_asr_0.6b": ModelSpec(
        model_id="qwen3_asr_0.6b",
        display_name="Qwen3-ASR-0.6B (默认中英通用识别)",
        backend="qwen3_asr",
        role="asr",
        languages="zh,en",
        approx_size_mb=1850,
        min_vram_mb=2400,
        license_name="Apache-2.0",
        description="默认中英文通用语音识别模型（约 1.85 GB），兼顾 RTX 3050/4060 显存与识别准确度，支持剧本匹配与自动出字。",
        ms_repo="Qwen/Qwen3-ASR-0.6B",
        hf_repo="Qwen/Qwen3-ASR-0.6B",
    ),
    "qwen3_asr_1.7b": ModelSpec(
        model_id="qwen3_asr_1.7b",
        display_name="Qwen3-ASR-1.7B (高精度可选识别)",
        backend="qwen3_asr",
        role="asr",
        languages="zh,en",
        approx_size_mb=4600,
        min_vram_mb=4800,
        license_name="Apache-2.0",
        description="高精度中英文识别模型（约 4.6 GB，需 ≥ 5GB 空闲显存），不随包下载，RTX 3050 6GB 建议谨慎选用。",
        ms_repo="Qwen/Qwen3-ASR-1.7B",
        hf_repo="Qwen/Qwen3-ASR-1.7B",
    ),
    "qwen3_aligner": ModelSpec(
        model_id="qwen3_aligner",
        display_name="Qwen3-ForcedAligner-0.6B (中英多语强制对齐)",
        backend="qwen3",
        role="aligner",
        languages="zh,en",
        approx_size_mb=1760,
        min_vram_mb=2200,
        license_name="Apache-2.0",
        description="阿里 Qwen3 非自回归时间戳强制对齐模型（约 1.76 GB），支持中文字符级与英文词级高精度定位。",
        ms_repo="Qwen/Qwen3-ForcedAligner-0.6B",
        hf_repo="Qwen/Qwen3-ForcedAligner-0.6B",
    ),
    "funasr_fa": ModelSpec(
        model_id="funasr_fa",
        display_name="FunASR 中文强制对齐模型 (fa-zh)",
        backend="funasr",
        role="aligner",
        languages="zh",
        approx_size_mb=160,
        min_vram_mb=700,
        license_name="FunASR Model License (MIT Toolkit)",
        description="轻量高速中文专用字符级强制对齐模型（约 160 MB，不可用于英文），低显存/CPU 均可 1~3 秒完成。",
        ms_repo="iic/speech_timestamp_prediction-v1-16k-offline",
        hf_repo="funasr/fa-zh",
    ),
    "funasr_asr": ModelSpec(
        model_id="funasr_asr",
        display_name="FunASR Paraformer-ZH (中文快速识别/锚点)",
        backend="funasr",
        role="asr",
        languages="zh",
        approx_size_mb=850,
        min_vram_mb=1500,
        license_name="Apache-2.0 / MIT",
        description="中文高速识别与时间轴锚点模型（约 850 MB，仅中文），适合中文快速模式及大跨度剪辑对齐。",
        ms_repo="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        hf_repo="funasr/paraformer-zh",
    ),
}

_MARKER_FILENAME = ".model_verified.json"
_WEIGHT_EXTS = {".pt", ".bin", ".safetensors", ".onnx"}


def get_model_local_dir(model_id: str, cache_dir: Path) -> Path:
    return cache_dir / model_id


def _has_incomplete_artifacts(mdir: Path) -> bool:
    for f in mdir.rglob("*"):
        if not f.is_file():
            continue
        name_l = f.name.lower()
        if name_l.endswith((".incomplete", ".tmp", ".downloading", ".lock")) or "._____temp" in str(f):
            return True
    return False


def _verify_hf_sharded_weights(mdir: Path) -> bool:
    idx_file = mdir / "model.safetensors.index.json"
    if not idx_file.is_file():
        return True
    try:
        payload = json.loads(idx_file.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") or {}
        shards = {str(v) for v in weight_map.values() if v}
        return all((mdir / shard).is_file() and (mdir / shard).stat().st_size > 0 for shard in shards)
    except Exception:
        return False


def _write_verified_marker(model_id: str, mdir: Path) -> None:
    weights = [
        {
            "rel": str(f.relative_to(mdir)).replace("\\", "/"),
            "size": f.stat().st_size,
        }
        for f in sorted(mdir.rglob("*"))
        if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS and f.stat().st_size > 1_000_000
    ]
    marker = {
        "model_id": model_id,
        "verified_at": int(time.time()),
        "weights": weights,
    }
    (mdir / _MARKER_FILENAME).write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")


def is_model_cached(model_id: str, cache_dir: Path) -> bool:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        return False
    if spec.is_bundled:
        return True
    mdir = get_model_local_dir(model_id, cache_dir)
    if not mdir.is_dir():
        return False
    if _has_incomplete_artifacts(mdir):
        return False

    marker_p = mdir / _MARKER_FILENAME
    if marker_p.is_file():
        try:
            marker = json.loads(marker_p.read_text(encoding="utf-8"))
            weights = marker.get("weights") or []
            if weights and all(
                (mdir / w["rel"]).is_file() and (mdir / w["rel"]).stat().st_size == w["size"]
                for w in weights
            ):
                return True
            return False
        except Exception:
            return False

    # Legacy pre-existing cache verification (preserves models downloaded by v1.0.0)
    if spec.backend.startswith("qwen"):
        if not (mdir / "config.json").is_file():
            return False
        if not _verify_hf_sharded_weights(mdir):
            return False
    elif spec.backend == "funasr":
        if not ((mdir / "config.yaml").is_file() or (mdir / "configuration.json").is_file()):
            return False

    for f in mdir.rglob("*"):
        if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS and f.stat().st_size > 1_000_000:
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


def get_total_models_size_mb(cache_dir: Path) -> float:
    total = 0.0
    for mid, spec in MODEL_REGISTRY.items():
        if not spec.is_bundled:
            total += get_cached_model_size_mb(mid, cache_dir)
    return round(total, 1)


def delete_cached_model(model_id: str, cache_dir: Path) -> bool:
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None or spec.is_bundled:
        return False
    # Release from active GPU/CPU memory before deleting files on Windows
    MODEL_POOL.unload_all()
    mdir = get_model_local_dir(model_id, cache_dir)
    if mdir.exists():
        shutil.rmtree(mdir, ignore_errors=True)
        return not mdir.exists()
    return False


def check_model_vram_risk(model_id: str) -> str | None:
    """Return a user-facing warning if selected model may exceed available GPU VRAM."""
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None or spec.min_vram_mb <= 0:
        return None
    vram = get_gpu_vram_info()
    if not vram["available"]:
        if spec.approx_size_mb >= 1800:
            return f"未检测到 CUDA GPU，{spec.display_name} 在纯 CPU 下运行较慢。"
        return None
    free_mb = vram["free_mb"]
    total_mb = vram["total_mb"]
    if model_id == "qwen3_asr_1.7b" and (total_mb < 6500 or free_mb < spec.min_vram_mb):
        return (
            f"当前显卡 {vram['device_name']} 可用显存 {free_mb} MB / 总 {total_mb} MB，"
            f"运行 Qwen3-ASR-1.7B（推荐 ≥ {spec.min_vram_mb} MB）存在显存不足风险；"
            "若显存不足将自动尝试降级（缩小切片或回退 0.6B/CPU）。"
        )
    if free_mb < spec.min_vram_mb:
        return (
            f"当前可用显存 {free_mb} MB 低于 {spec.display_name} 推荐值 ({spec.min_vram_mb} MB)，"
            "运行阶段将自动启用低显存保护或 CPU 回退。"
        )
    return None


def _cleanup_download_temp(mdir: Path) -> None:
    for dname in ("._____temp", ".mscache", ".cache"):
        cand = mdir / dname
        if cand.exists():
            shutil.rmtree(cand, ignore_errors=True)


def download_model(
    model_id: str,
    cache_dir: Path,
    *,
    use_modelscope: bool = True,
    progress_cb: Callable[[str], None] | None = None,
    percent_cb: Callable[[int], None] | None = None,
) -> Path:
    """Download a model cleanly into `cache_dir / model_id` without duplicate copies."""
    spec = MODEL_REGISTRY.get(model_id)
    if spec is None:
        raise ValueError(f"未知模型 ID: {model_id}")
    if spec.is_bundled:
        if percent_cb:
            percent_cb(100)
        return cache_dir

    target_dir = get_model_local_dir(model_id, cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if is_model_cached(model_id, cache_dir):
        if progress_cb:
            progress_cb(f"{spec.display_name} 已在本地缓存，跳过下载。")
        if percent_cb:
            percent_cb(100)
        return target_dir

    # Remove stale verified marker if retrying after a broken download
    marker_p = target_dir / _MARKER_FILENAME
    if marker_p.exists():
        marker_p.unlink(missing_ok=True)

    # Check disk free space before downloading (need ~1.25x safety margin)
    existing_mb = get_cached_model_size_mb(model_id, cache_dir)
    needed_mb = max(200.0, spec.approx_size_mb * 1.25 - existing_mb)
    free_mb = get_disk_free_mb(target_dir)
    if free_mb > 0 and free_mb < needed_mb:
        raise RuntimeError(
            f"磁盘剩余空间不足：目标目录剩余 {free_mb:.0f} MB，下载 {spec.display_name} 建议至少预留 {needed_mb:.0f} MB。"
        )

    stop_monitor = threading.Event()

    def _monitor_dir_size() -> None:
        expected = max(1.0, float(spec.approx_size_mb))
        last_pct = -1
        while not stop_monitor.is_set():
            cur_mb = get_cached_model_size_mb(model_id, cache_dir)
            pct = min(98, max(1, int(round(cur_mb / expected * 100))))
            if pct != last_pct:
                last_pct = pct
                if percent_cb:
                    percent_cb(pct)
                if progress_cb:
                    progress_cb(f"正在下载 {spec.display_name}：{cur_mb:.1f} / ~{spec.approx_size_mb} MB ({pct}%)")
            stop_monitor.wait(0.8)

    mon = threading.Thread(target=_monitor_dir_size, daemon=True)
    mon.start()

    last_err: Exception | None = None
    try:
        if use_modelscope and spec.ms_repo:
            try:
                if progress_cb:
                    progress_cb(f"正在从 ModelScope 下载 {spec.display_name} ({spec.ms_repo})...")
                from modelscope.hub.snapshot_download import snapshot_download

                snapshot_download(
                    spec.ms_repo,
                    local_dir=str(target_dir),
                )
                _cleanup_download_temp(target_dir)
                if is_model_cached(model_id, cache_dir):
                    _write_verified_marker(model_id, target_dir)
                    if percent_cb:
                        percent_cb(100)
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
                _cleanup_download_temp(target_dir)
                if is_model_cached(model_id, cache_dir):
                    _write_verified_marker(model_id, target_dir)
                    if percent_cb:
                        percent_cb(100)
                    return target_dir
            except Exception as exc:
                last_err = exc
    finally:
        stop_monitor.set()
        mon.join(timeout=1.5)

    raise RuntimeError(f"下载模型 {spec.display_name} 失败: {last_err}")
