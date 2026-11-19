"""Qwen3-ASR (0.6B / 1.7B) backend with Windows CJK-path safeguard, stage VRAM pool, and OOM recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

import numpy as np

from drama_studio.align.base import ASRSegment
from drama_studio.align.model_manager import is_model_cached
from drama_studio.align.runtime import (
    MODEL_POOL,
    ensure_qwen_windows_safe,
    get_gpu_vram_info,
    is_oom_error,
    release_torch_memory,
    resolve_torch_device,
)


def _to_qwen_lang_arg(language_pref: str) -> str | None:
    low = (language_pref or "auto").strip().lower()
    if low in {"zh", "chinese", "cn"}:
        return "Chinese"
    if low in {"en", "eng", "english"}:
        return "English"
    return None


def _from_qwen_lang_name(raw_lang: str, text: str) -> str:
    low = (raw_lang or "").strip().lower()
    if "english" in low:
        return "en"
    if "chinese" in low or "cantonese" in low:
        return "zh"
    # Fallback heuristic from recognized characters
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return "en" if latin > cjk * 2 else "zh"


def _is_hallucinated_text(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return True
    # Pure punctuation/symbols
    alnum = [c for c in cleaned if c.isalnum() or ("\u4e00" <= c <= "\u9fff")]
    if not alnum:
        return True
    # Degenerate single-char repetition e.g. "啊啊啊啊啊啊啊" or "uh uh uh uh uh"
    if len(alnum) >= 8 and len(set(alnum)) == 1:
        return True
    words = re.findall(r"[A-Za-z']+", cleaned.lower())
    if len(words) >= 6 and len(set(words)) == 1 and len(words[0]) <= 3:
        return True
    return False


@dataclass
class QwenASRBatchResult:
    segments: list[ASRSegment]
    model_used: str
    warnings: list[str]


class Qwen3ASRBackend:
    """Wrapper around Qwen3-ASR-0.6B / Qwen3-ASR-1.7B without co-loading ForcedAligner."""

    def __init__(self, model_id: str, cache_dir: Path) -> None:
        if model_id not in {"qwen3_asr_0.6b", "qwen3_asr_1.7b"}:
            raise ValueError(f"Unsupported Qwen3-ASR model_id: {model_id}")
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.display_name = "Qwen3-ASR-1.7B" if model_id == "qwen3_asr_1.7b" else "Qwen3-ASR-0.6B"

    def is_available(self) -> bool:
        return is_model_cached(self.model_id, self.cache_dir)

    def _recommended_vram_mb(self) -> int:
        return 4500 if self.model_id == "qwen3_asr_1.7b" else 2300

    def _recommended_batch_size(self, used_dev: str) -> int:
        if not used_dev.startswith("cuda"):
            return 1
        vram = get_gpu_vram_info()
        free_mb = vram.get("free_mb", 0)
        loaded_key, _ = MODEL_POOL.loaded_roles().get("asr", ("", ""))
        already_loaded = loaded_key.startswith(f"{self.model_id}:{self.cache_dir}:{used_dev}")
        if self.model_id == "qwen3_asr_1.7b":
            if already_loaded:
                return 4 if free_mb >= 3200 else (2 if free_mb >= 1800 else 1)
            return 2 if free_mb >= 6200 else 1
        if already_loaded:
            return 8 if free_mb >= 2200 else (4 if free_mb >= 1200 else 1)
        return 8 if free_mb >= 4000 else (4 if free_mb >= 3000 else 1)

    def _load_model(self, device_pref: str = "auto", *, batch_size_override: int | None = None):
        ensure_qwen_windows_safe()
        import torch
        from qwen_asr import Qwen3ASRModel

        dev = resolve_torch_device(device_pref, min_free_vram_mb=self._recommended_vram_mb())
        batch_sz = batch_size_override or self._recommended_batch_size(dev)
        key = f"{self.model_id}:{self.cache_dir}:{dev}"
        cached = MODEL_POOL.get("asr", key)
        if cached is not None:
            model_cached, dev_cached = cached
            if hasattr(model_cached, "max_inference_batch_size"):
                model_cached.max_inference_batch_size = batch_sz
            return model_cached, dev_cached

        model_dir = str(self.cache_dir / self.model_id)
        if dev.startswith("cuda"):
            try:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                # Explicitly leave forced_aligner=None so ASR & Aligner never co-reside in VRAM
                model = Qwen3ASRModel.from_pretrained(
                    model_dir,
                    forced_aligner=None,
                    dtype=dtype,
                    device_map=dev,
                    max_inference_batch_size=batch_sz,
                    max_new_tokens=192,
                )
                MODEL_POOL.put("asr", key, model, dev, exclusive_gpu=True)
                return model, dev
            except Exception as exc:
                MODEL_POOL.unload_role("asr")
                release_torch_memory()
                if device_pref == "cuda" and not is_oom_error(exc):
                    raise
                # Fall through to CPU fallback

        model = Qwen3ASRModel.from_pretrained(
            model_dir,
            forced_aligner=None,
            dtype=torch.float32,
            device_map="cpu",
            max_inference_batch_size=1,
            max_new_tokens=192,
        )
        cpu_key = f"{self.model_id}:{self.cache_dir}:cpu"
        MODEL_POOL.put("asr", cpu_key, model, "cpu", exclusive_gpu=False)
        return model, "cpu"

    def transcribe_windows(
        self,
        audio_16k: np.ndarray,
        windows_ms: list[tuple[int, int]],
        *,
        language_pref: str = "auto",
        device: str = "auto",
        allow_vram_downgrade: bool = True,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> QwenASRBatchResult:
        """Transcribe VAD speech windows in length-sorted batches with full OOM protection."""
        if not windows_ms:
            return QwenASRBatchResult(segments=[], model_used=self.display_name, warnings=[])

        warnings_out: list[str] = []
        qwen_lang = _to_qwen_lang_arg(language_pref)

        if device != "cpu" and self.model_id == "qwen3_asr_1.7b":
            vram = get_gpu_vram_info()
            if vram["available"] and vram["free_mb"] < 4300:
                warnings_out.append(
                    f"显存警告：当前空闲显存 {vram['free_mb']} MB 偏低（1.7B 建议 ≥ 4500 MB）。"
                )

        try:
            model, used_dev = self._load_model(device)
        except Exception as exc:
            if is_oom_error(exc) and allow_vram_downgrade and self.model_id == "qwen3_asr_1.7b":
                if is_model_cached("qwen3_asr_0.6b", self.cache_dir):
                    warnings_out.append("Qwen3-ASR-1.7B 加载显存不足，已自动降级为 Qwen3-ASR-0.6B。")
                    fb = Qwen3ASRBackend("qwen3_asr_0.6b", self.cache_dir)
                    res = fb.transcribe_windows(
                        audio_16k,
                        windows_ms,
                        language_pref=language_pref,
                        device=device,
                        allow_vram_downgrade=True,
                        progress_cb=progress_cb,
                    )
                    res.warnings = warnings_out + res.warnings
                    return res
            raise

        batch_sz = self._recommended_batch_size(used_dev)
        if hasattr(model, "max_inference_batch_size"):
            model.max_inference_batch_size = batch_sz

        # Sort windows by length before batching to minimize padding & preserve ultra-short clip accuracy
        indexed_wins = sorted(
            enumerate(windows_ms),
            key=lambda item: item[1][1] - item[1][0],
        )
        segments_by_idx: dict[int, ASRSegment] = {}
        total_w = len(indexed_wins)
        orig_max_tokens = getattr(model, "max_new_tokens", None)

        i = 0
        try:
            while i < total_w:
                batch_items = indexed_wins[i : i + batch_sz]
                audio_inputs: list[tuple[np.ndarray, int]] = []
                valid_items: list[tuple[int, int, int]] = []
                for orig_idx, (ws, we) in batch_items:
                    s_idx = max(0, int(round(ws * 16.0)))
                    e_idx = min(len(audio_16k), int(round(we * 16.0)))
                    chunk = audio_16k[s_idx:e_idx].astype(np.float32, copy=False)
                    if len(chunk) >= 1600:
                        audio_inputs.append((chunk, 16000))
                        valid_items.append((orig_idx, ws, we))

                if not audio_inputs:
                    i += len(batch_items)
                    continue

                if progress_cb:
                    progress_cb(
                        i / max(1, total_w),
                        f"{self.display_name} [{used_dev}] 识别语音片段 ({min(total_w, i + len(batch_items))}/{total_w})...",
                    )

                if orig_max_tokens is not None:
                    max_dur_s = max((we - ws) / 1000.0 for _, ws, we in valid_items)
                    model.max_new_tokens = max(96, min(224, int(round(max_dur_s * 22.0)) + 32))

                try:
                    lang_arg = [qwen_lang] * len(audio_inputs) if qwen_lang is not None else None
                    outs = model.transcribe(
                        audio=audio_inputs,
                        language=lang_arg,
                        return_time_stamps=False,
                    )
                except Exception as exc:
                    if is_oom_error(exc) and used_dev.startswith("cuda"):
                        release_torch_memory()
                        if batch_sz > 1:
                            batch_sz = max(1, batch_sz // 2)
                            if hasattr(model, "max_inference_batch_size"):
                                model.max_inference_batch_size = batch_sz
                            warnings_out.append(f"识别阶段 GPU 显存紧张，已自动收紧批大小 (batch={batch_sz})。")
                            continue
                        MODEL_POOL.unload_role("asr")
                        release_torch_memory()
                        if (
                            allow_vram_downgrade
                            and self.model_id == "qwen3_asr_1.7b"
                            and is_model_cached("qwen3_asr_0.6b", self.cache_dir)
                        ):
                            warnings_out.append("Qwen3-ASR-1.7B 推理显存不足，已自动降级到 Qwen3-ASR-0.6B。")
                            rem_indices = [idx for idx, _ in indexed_wins[i:]]
                            rem_windows = [windows_ms[idx] for idx in sorted(rem_indices)]
                            fb = Qwen3ASRBackend("qwen3_asr_0.6b", self.cache_dir)
                            rem = fb.transcribe_windows(
                                audio_16k,
                                rem_windows,
                                language_pref=language_pref,
                                device=device,
                                allow_vram_downgrade=True,
                                progress_cb=progress_cb,
                            )
                            ordered_done = [segments_by_idx[k] for k in sorted(segments_by_idx)]
                            ordered_done.extend(rem.segments)
                            warnings_out.extend(rem.warnings)
                            return QwenASRBatchResult(
                                segments=ordered_done,
                                model_used=f"{self.display_name} -> {rem.model_used}",
                                warnings=warnings_out,
                            )
                        warnings_out.append(f"{self.display_name} GPU 显存不足，已自动回退至 CPU 继续。")
                        model, used_dev = self._load_model("cpu", batch_size_override=1)
                        batch_sz = 1
                        continue
                    raise

                for (orig_idx, ws, we), tr in zip(valid_items, outs):
                    txt = (tr.text or "").strip()
                    if _is_hallucinated_text(txt):
                        continue
                    seg_lang = (
                        "en"
                        if language_pref == "en"
                        else ("zh" if language_pref == "zh" else _from_qwen_lang_name(tr.language, txt))
                    )
                    segments_by_idx[orig_idx] = ASRSegment(
                        start_ms=ws,
                        end_ms=we,
                        text=txt,
                        language=seg_lang,
                        confidence=0.90 if self.model_id == "qwen3_asr_1.7b" else 0.86,
                        model_used=f"{self.display_name} [{used_dev}]",
                        src_win_start_ms=ws,
                        src_win_end_ms=we,
                    )

                i += len(batch_items)
        finally:
            if orig_max_tokens is not None and hasattr(model, "max_new_tokens"):
                model.max_new_tokens = orig_max_tokens

        segments = [segments_by_idx[k] for k in sorted(segments_by_idx)]
        return QwenASRBatchResult(
            segments=segments,
            model_used=f"{self.display_name} [{used_dev}]",
            warnings=warnings_out,
        )
