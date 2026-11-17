"""FunASR (fa-zh + optional paraformer-zh) forced alignment backend with GPU/CPU fallback."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Literal

import numpy as np
import soundfile as sf

from drama_studio.align.base import AlignerOutput
from drama_studio.align.model_manager import is_model_cached
from drama_studio.models import CharTimestamp

_LOCK = threading.Lock()
_FA_MODELS: dict[str, object] = {}
_ASR_MODELS: dict[str, object] = {}

_DIGIT_MAP = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}


def _resolve_device(pref: str = "auto", min_free_vram_mb: int = 800) -> str:
    if pref == "cpu":
        return "cpu"
    import torch

    if torch.cuda.is_available():
        try:
            free_bytes, _ = torch.cuda.mem_get_info(0)
            if free_bytes / (1024 * 1024) >= min_free_vram_mb or pref == "cuda":
                return "cuda:0"
        except Exception:
            if pref == "cuda":
                return "cuda:0"
    return "cpu"


def _expand_numeric_for_acoustic(token: str) -> str:
    """Convert numeric tokens like '3.14' or '2026' into Mandarin spoken chars for acoustic aligner."""
    if not token:
        return token
    out: list[str] = []
    for i, ch in enumerate(token):
        if ch in _DIGIT_MAP:
            out.append(_DIGIT_MAP[ch])
        elif ch == "." and 0 < i < len(token) - 1 and token[i - 1].isdigit() and token[i + 1].isdigit():
            out.append("点")
        elif ch == "%":
            out.append("百分之")
        elif ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"):
            out.append(ch)
    return "".join(out)


def _split_to_char_mapping(spoken_text: str) -> tuple[str, list[tuple[str, int, int]]]:
    """Build acoustic query string and mapping from each original spoken char to slice in acoustic string."""
    # Preserve ASCII words/numbers mapped cleanly while mapping 1-to-1 for CJK chars
    acoustic_parts: list[str] = []
    mapping: list[tuple[str, int, int]] = []  # (orig_char, start_tok_idx, end_tok_idx)
    tok_cursor = 0

    i = 0
    n = len(spoken_text)
    while i < n:
        ch = spoken_text[i]
        if ch.isspace():
            i += 1
            continue
        if "\u4e00" <= ch <= "\u9fff":
            acoustic_parts.append(ch)
            mapping.append((ch, tok_cursor, tok_cursor + 1))
            tok_cursor += 1
            i += 1
        elif ch.isdigit():
            # Collect numeric run (with decimal point)
            j = i + 1
            while j < n and (spoken_text[j].isdigit() or (spoken_text[j] == "." and j + 1 < n and spoken_text[j + 1].isdigit())):
                j += 1
            num_str = spoken_text[i:j]
            spoken_num = _expand_numeric_for_acoustic(num_str)
            for c_sub in spoken_num:
                acoustic_parts.append(c_sub)
            end_tok = tok_cursor + max(1, len(spoken_num))
            # Distribute acoustic sub-tokens evenly across digits of original number
            sub_cnt = max(1, end_tok - tok_cursor)
            orig_len = j - i
            for k in range(orig_len):
                s_t = tok_cursor + int(k * sub_cnt / orig_len)
                e_t = tok_cursor + max(s_t - tok_cursor + 1, int((k + 1) * sub_cnt / orig_len))
                mapping.append((spoken_text[i + k], s_t, min(end_tok, e_t)))
            tok_cursor = end_tok
            i = j
        elif ch.isascii() and ch.isalpha():
            j = i + 1
            while j < n and spoken_text[j].isascii() and spoken_text[j].isalpha():
                j += 1
            word = spoken_text[i:j]
            acoustic_parts.append(word)
            for k in range(len(word)):
                mapping.append((word[k], tok_cursor, tok_cursor + 1))
            tok_cursor += 1
            i = j
        else:
            # Keep meaningful non-punct symbol if any without feeding invalid punct
            i += 1

    return " ".join(acoustic_parts), mapping


class FunASRAlignerBackend:
    """Forced aligner backed by FunASR fa-zh (TP-Aligner) with optional paraformer-zh anchor."""

    name = "FunASR (fa-zh)"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def is_available(self) -> bool:
        return is_model_cached("funasr_fa", self.cache_dir)

    def has_asr_anchor(self) -> bool:
        return is_model_cached("funasr_asr", self.cache_dir)

    def _get_fa_model(self, device_pref: str = "auto"):
        from funasr import AutoModel

        dev = _resolve_device(device_pref, min_free_vram_mb=700)
        key = f"{self.cache_dir}:{dev}"
        with _LOCK:
            if key in _FA_MODELS:
                return _FA_MODELS[key], dev
            model_dir = str(self.cache_dir / "funasr_fa")
            try:
                model = AutoModel(model=model_dir, device=dev, disable_update=True, disable_pbar=True)
            except Exception:
                dev = "cpu"
                key = f"{self.cache_dir}:cpu"
                if key in _FA_MODELS:
                    return _FA_MODELS[key], dev
                model = AutoModel(model=model_dir, device="cpu", disable_update=True, disable_pbar=True)
            _FA_MODELS[key] = model
            return model, dev

    def _get_asr_model(self, device_pref: str = "auto"):
        from funasr import AutoModel

        dev = _resolve_device(device_pref, min_free_vram_mb=1500)
        key = f"{self.cache_dir}:{dev}"
        with _LOCK:
            if key in _ASR_MODELS:
                return _ASR_MODELS[key], dev
            model_dir = str(self.cache_dir / "funasr_asr")
            try:
                model = AutoModel(model=model_dir, device=dev, disable_update=True, disable_pbar=True)
            except Exception:
                dev = "cpu"
                key = f"{self.cache_dir}:cpu"
                if key in _ASR_MODELS:
                    return _ASR_MODELS[key], dev
                model = AutoModel(model=model_dir, device="cpu", disable_update=True, disable_pbar=True)
            _ASR_MODELS[key] = model
            return model, dev

    def transcribe_anchors(self, audio_16k: np.ndarray, *, device: str = "auto") -> list[tuple[str, int, int]]:
        """Run full-timeline Paraformer ASR to locate character-level timeline anchors (handles edit shifts)."""
        if not self.has_asr_anchor() or len(audio_16k) < 1600:
            return []
        model, _ = self._get_asr_model(device)
        with tempfile.TemporaryDirectory() as td:
            wav_p = Path(td) / "asr.wav"
            sf.write(str(wav_p), audio_16k, 16000, subtype="PCM_16")
            res = model.generate(input=str(wav_p), batch_size_s=120)
        if not res:
            return []
        text = str(res[0].get("text", "")).replace(" ", "")
        ts = res[0].get("timestamp") or []
        anchors: list[tuple[str, int, int]] = []
        for ch, span in zip(text, ts):
            if len(span) >= 2:
                anchors.append((ch, int(span[0]), int(span[1])))
        return anchors

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
        segment = audio_16k[s_idx:e_idx]
        if len(segment) < 800:
            return AlignerOutput(char_timings=[], backend_name=self.name, warnings=["音频切片过短"])

        acoustic_query, mapping = _split_to_char_mapping(spoken_text)
        if not acoustic_query.strip() or not mapping:
            return AlignerOutput(char_timings=[], backend_name=self.name)

        model, used_dev = self._get_fa_model(device)
        with tempfile.TemporaryDirectory() as td:
            wav_p = Path(td) / "seg.wav"
            txt_p = Path(td) / "txt.txt"
            sf.write(str(wav_p), segment, 16000, subtype="PCM_16")
            txt_p.write_text(acoustic_query, encoding="utf-8")
            res = model.generate(input=(str(wav_p), str(txt_p)), data_type=("sound", "text"))

        if not res:
            return AlignerOutput(char_timings=[], backend_name=f"{self.name} ({used_dev})")

        raw_ts = res[0].get("timestamp") or []
        raw_tokens = str(res[0].get("text", "")).split()

        char_timings: list[CharTimestamp] = []
        for orig_ch, tok_s, tok_e in mapping:
            if tok_s < len(raw_ts):
                t_start = int(raw_ts[tok_s][0]) + window_start_ms
                last_idx = min(len(raw_ts) - 1, max(tok_s, tok_e - 1))
                t_end = int(raw_ts[last_idx][1]) + window_start_ms
                dur = max(25, t_end - t_start)
                # Score plausibility of per-char duration (ideal 60..600 ms)
                conf = 0.92 if 45 <= dur <= 700 else 0.55
                char_timings.append(
                    CharTimestamp(
                        char=orig_ch,
                        start_ms=max(window_start_ms, t_start),
                        end_ms=min(window_end_ms, max(t_start + 25, t_end)),
                        confidence=conf,
                    )
                )
            else:
                prev_end = char_timings[-1].end_ms if char_timings else window_start_ms
                char_timings.append(
                    CharTimestamp(
                        char=orig_ch,
                        start_ms=prev_end,
                        end_ms=min(window_end_ms, prev_end + 120),
                        confidence=0.30,
                        interpolated=True,
                    )
                )

        return AlignerOutput(
            char_timings=char_timings,
            backend_name=f"{self.name} [{used_dev}]",
            raw_asr_text=" ".join(raw_tokens),
        )
