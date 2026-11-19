"""Qwen3-ForcedAligner-0.6B backend with Windows CJK-path safeguard, stage VRAM pool, and multilingual support."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from drama_studio.align.base import AlignerOutput
from drama_studio.align.funasr_aligner import _split_to_char_mapping
from drama_studio.align.model_manager import is_model_cached
from drama_studio.align.runtime import (
    MODEL_POOL,
    ensure_qwen_windows_safe,
    get_gpu_vram_info,
    is_oom_error,
    release_torch_memory,
    resolve_torch_device,
)
from drama_studio.models import CharTimestamp


def _normalize_qwen_language(language: str) -> str:
    low = (language or "").strip().lower()
    if low in {"en", "eng", "english"}:
        return "English"
    return "Chinese"


class Qwen3AlignerBackend:
    """Forced aligner backed by Alibaba Qwen3-ForcedAligner-0.6B (Apache-2.0)."""

    name = "Qwen3-ForcedAligner-0.6B"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def is_available(self) -> bool:
        return is_model_cached("qwen3_aligner", self.cache_dir)

    def _get_model(self, device_pref: str = "auto"):
        ensure_qwen_windows_safe()
        import torch
        from qwen_asr import Qwen3ForcedAligner

        dev = resolve_torch_device(device_pref, min_free_vram_mb=2200)
        key = f"qwen3_aligner:{self.cache_dir}:{dev}"
        cached = MODEL_POOL.get("aligner", key)
        if cached is not None:
            return cached

        model_dir = str(self.cache_dir / "qwen3_aligner")
        if dev.startswith("cuda"):
            try:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                aligner = Qwen3ForcedAligner.from_pretrained(
                    model_dir,
                    dtype=dtype,
                    device_map=dev,
                )
                MODEL_POOL.put("aligner", key, aligner, dev, exclusive_gpu=True)
                return aligner, dev
            except Exception:
                release_torch_memory()
                dev = "cpu"
                key = f"qwen3_aligner:{self.cache_dir}:cpu"
                cached_cpu = MODEL_POOL.get("aligner", key)
                if cached_cpu is not None:
                    return cached_cpu

        aligner = Qwen3ForcedAligner.from_pretrained(
            model_dir,
            dtype=torch.float32,
            device_map="cpu",
        )
        MODEL_POOL.put("aligner", key, aligner, "cpu", exclusive_gpu=False)
        return aligner, "cpu"

    def _prepare_query(
        self,
        audio_16k: np.ndarray,
        window_start_ms: int,
        window_end_ms: int,
        spoken_text: str,
        language: str,
    ) -> tuple[np.ndarray | None, str, str, list[str]]:
        s_idx = max(0, int(round(window_start_ms * 16000 / 1000.0)))
        e_idx = min(len(audio_16k), int(round(window_end_ms * 16000 / 1000.0)))
        segment = audio_16k[s_idx:e_idx].astype(np.float32, copy=False)
        qwen_lang = _normalize_qwen_language(language)
        if len(segment) < 800:
            return None, "", qwen_lang, ["音频切片过短"]

        if qwen_lang == "Chinese":
            acoustic_query, mapping = _split_to_char_mapping(spoken_text)
            if not acoustic_query.strip() or not mapping:
                return None, "", qwen_lang, []
        elif not spoken_text.strip():
            return None, "", qwen_lang, []

        return segment, spoken_text.strip(), qwen_lang, []

    @staticmethod
    def _items_to_char_timestamps(
        items: list,
        window_start_ms: int,
        window_end_ms: int,
    ) -> list[CharTimestamp]:
        raw_chars: list[CharTimestamp] = []
        for it in items:
            tok = it.text
            t_s = max(window_start_ms, int(round(it.start_time * 1000.0)) + window_start_ms)
            t_e = min(window_end_ms, max(t_s + 25, int(round(it.end_time * 1000.0)) + window_start_ms))
            if len(tok) <= 1:
                dur = max(25, t_e - t_s)
                conf = 0.94 if 35 <= dur <= 700 else 0.58
                raw_chars.append(CharTimestamp(char=tok, start_ms=t_s, end_ms=t_e, confidence=conf))
            else:
                word_dur = max(25, t_e - t_s)
                word_conf = 0.92 if 70 <= word_dur <= 1800 else 0.58
                step = max(12.0, word_dur / len(tok))
                for k, c in enumerate(tok):
                    cs = int(round(t_s + k * step))
                    ce = t_e if k == len(tok) - 1 else int(round(t_s + (k + 1) * step))
                    raw_chars.append(
                        CharTimestamp(
                            char=c,
                            start_ms=cs,
                            end_ms=max(cs + 12, ce),
                            confidence=word_conf,
                        )
                    )
        return raw_chars

    @staticmethod
    def _recommended_align_batch_size(used_dev: str) -> int:
        if not used_dev.startswith("cuda"):
            return 1
        free_mb = get_gpu_vram_info().get("free_mb", 0)
        if free_mb >= 2400:
            return 16
        if free_mb >= 1400:
            return 8
        return 4 if free_mb >= 900 else 1

    def align_utterance(
        self,
        audio_16k: np.ndarray,
        window_start_ms: int,
        window_end_ms: int,
        spoken_text: str,
        *,
        language: str = "zh",
        device: str = "auto",
        cache_dir: Path | None = None,
    ) -> AlignerOutput:
        segment, query_for_qwen, qwen_lang, prep_warns = self._prepare_query(
            audio_16k, window_start_ms, window_end_ms, spoken_text, language
        )
        if segment is None:
            return AlignerOutput(char_timings=[], backend_name=self.name, warnings=prep_warns)

        warnings_out: list[str] = []
        aligner, used_dev = self._get_model(device)
        try:
            results = aligner.align(
                audio=[(segment, 16000)],
                text=[query_for_qwen],
                language=[qwen_lang],
            )
        except Exception as exc:
            if is_oom_error(exc) and used_dev.startswith("cuda"):
                MODEL_POOL.unload_role("aligner")
                release_torch_memory()
                warnings_out.append("GPU 显存不足，Qwen3-ForcedAligner 已自动回退至 CPU 执行。")
                aligner, used_dev = self._get_model("cpu")
                results = aligner.align(
                    audio=[(segment, 16000)],
                    text=[query_for_qwen],
                    language=[qwen_lang],
                )
            else:
                raise

        if not results or not results[0].items:
            return AlignerOutput(
                char_timings=[],
                backend_name=f"{self.name} [{used_dev}]",
                warnings=warnings_out,
            )

        return AlignerOutput(
            char_timings=self._items_to_char_timestamps(results[0].items, window_start_ms, window_end_ms),
            backend_name=f"{self.name} [{used_dev}]",
            warnings=warnings_out,
        )

    def align_utterances_batch(
        self,
        audio_16k: np.ndarray,
        items: list[tuple[int, int, str]],
        *,
        language: str = "zh",
        device: str = "auto",
        cache_dir: Path | None = None,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> list[AlignerOutput]:
        """Align multiple utterance windows in length-sorted GPU batches with OOM protection."""
        if not items:
            return []

        outputs: list[AlignerOutput | None] = [None] * len(items)
        prepared: list[tuple[int, int, int, np.ndarray, str, str]] = []
        for idx, (ws, we, txt) in enumerate(items):
            seg, query, qwen_lang, prep_warns = self._prepare_query(audio_16k, ws, we, txt, language)
            if seg is None:
                outputs[idx] = AlignerOutput(char_timings=[], backend_name=self.name, warnings=prep_warns)
            else:
                prepared.append((idx, ws, we, seg, query, qwen_lang))

        if not prepared:
            return [o if o is not None else AlignerOutput(char_timings=[], backend_name=self.name) for o in outputs]

        aligner, used_dev = self._get_model(device)
        batch_sz = self._recommended_align_batch_size(used_dev)
        # Sort by audio length so batches pack similar-duration segments with minimal padding
        prepared.sort(key=lambda row: len(row[3]))

        total = len(prepared)
        pos = 0
        while pos < total:
            batch = prepared[pos : pos + batch_sz]
            if progress_cb:
                frac = 0.35 + 0.30 * (pos / max(1, total))
                progress_cb(frac, f"正在批量强制对齐 ({min(total, pos + len(batch))}/{total})...")

            try:
                results = aligner.align(
                    audio=[(row[3], 16000) for row in batch],
                    text=[row[4] for row in batch],
                    language=[row[5] for row in batch],
                )
            except Exception as exc:
                if is_oom_error(exc) and used_dev.startswith("cuda"):
                    release_torch_memory()
                    if batch_sz > 1:
                        batch_sz = max(1, batch_sz // 2)
                        continue
                    MODEL_POOL.unload_role("aligner")
                    release_torch_memory()
                    aligner, used_dev = self._get_model("cpu")
                    batch_sz = 1
                    continue
                raise

            be_label = f"{self.name} [{used_dev}]"
            for row, res in zip(batch, results):
                orig_idx, ws, we = row[0], row[1], row[2]
                if not res or not res.items:
                    outputs[orig_idx] = AlignerOutput(char_timings=[], backend_name=be_label)
                else:
                    outputs[orig_idx] = AlignerOutput(
                        char_timings=self._items_to_char_timestamps(res.items, ws, we),
                        backend_name=be_label,
                    )
            pos += len(batch)

        return [
            o if o is not None else AlignerOutput(char_timings=[], backend_name=f"{self.name} [{used_dev}]")
            for o in outputs
        ]
