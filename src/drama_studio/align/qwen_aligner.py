"""Qwen3-ForcedAligner-0.6B backend with Windows CJK-path safeguard and GPU/CPU fallback."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import types

import numpy as np

from drama_studio.align.base import AlignerOutput
from drama_studio.align.funasr_aligner import _resolve_device, _split_to_char_mapping
from drama_studio.align.model_manager import is_model_cached
from drama_studio.models import CharTimestamp

_LOCK = threading.Lock()
_QWEN_ALIGNERS: dict[str, object] = {}


def _ensure_nagisa_stub_if_needed() -> None:
    """Prevent dynet C++ runtime crash on Windows non-ASCII paths when Japanese tokenizer is unused."""
    if "nagisa" not in sys.modules:
        stub = types.ModuleType("nagisa")
        stub.tagging = lambda *args, **kwargs: types.SimpleNamespace(words=[])
        sys.modules["nagisa"] = stub


class Qwen3AlignerBackend:
    """Forced aligner backed by Alibaba Qwen3-ForcedAligner-0.6B (Apache-2.0)."""

    name = "Qwen3-ForcedAligner-0.6B"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def is_available(self) -> bool:
        return is_model_cached("qwen3_aligner", self.cache_dir)

    def _get_model(self, device_pref: str = "auto"):
        _ensure_nagisa_stub_if_needed()
        import torch
        from qwen_asr import Qwen3ForcedAligner

        dev = _resolve_device(device_pref, min_free_vram_mb=2200)
        key = f"{self.cache_dir}:{dev}"
        with _LOCK:
            if key in _QWEN_ALIGNERS:
                return _QWEN_ALIGNERS[key], dev

            model_dir = str(self.cache_dir / "qwen3_aligner")
            if dev.startswith("cuda"):
                try:
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    aligner = Qwen3ForcedAligner.from_pretrained(
                        model_dir,
                        dtype=dtype,
                        device_map=dev,
                    )
                    _QWEN_ALIGNERS[key] = aligner
                    return aligner, dev
                except Exception:
                    dev = "cpu"
                    key = f"{self.cache_dir}:cpu"
                    if key in _QWEN_ALIGNERS:
                        return _QWEN_ALIGNERS[key], dev

            aligner = Qwen3ForcedAligner.from_pretrained(
                model_dir,
                dtype=torch.float32,
                device_map="cpu",
            )
            _QWEN_ALIGNERS[key] = aligner
            return aligner, "cpu"

    def align_utterance(
        self,
        audio_16k: np.ndarray,
        window_start_ms: int,
        window_end_ms: int,
        spoken_text: str,
        *,
        device: str = "auto",
        cache_dir: Path | None = None,
    ) -> AlignerOutput:
        s_idx = max(0, int(round(window_start_ms * 16000 / 1000.0)))
        e_idx = min(len(audio_16k), int(round(window_end_ms * 16000 / 1000.0)))
        segment = audio_16k[s_idx:e_idx].astype(np.float32, copy=False)
        if len(segment) < 800:
            return AlignerOutput(char_timings=[], backend_name=self.name, warnings=["音频切片过短"])

        acoustic_query, mapping = _split_to_char_mapping(spoken_text)
        if not acoustic_query.strip() or not mapping:
            return AlignerOutput(char_timings=[], backend_name=self.name)

        # Qwen3ForcedAligner expects continuous natural tokens (without extra spaces between CJK)
        query_for_qwen = spoken_text.strip()
        aligner, used_dev = self._get_model(device)
        results = aligner.align(
            audio=[(segment, 16000)],
            text=[query_for_qwen],
            language=["Chinese"],
        )
        if not results or not results[0].items:
            return AlignerOutput(char_timings=[], backend_name=f"{self.name} [{used_dev}]")

        items = results[0].items
        # Expand any multi-char Latin/numeric token returned by Qwen3 into per-char CharTimestamps
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
                step = max(20.0, (t_e - t_s) / len(tok))
                for k, c in enumerate(tok):
                    cs = int(round(t_s + k * step))
                    ce = int(round(t_s + (k + 1) * step))
                    raw_chars.append(
                        CharTimestamp(char=c, start_ms=cs, end_ms=max(cs + 20, ce), confidence=0.90)
                    )

        return AlignerOutput(
            char_timings=raw_chars,
            backend_name=f"{self.name} [{used_dev}]",
        )
