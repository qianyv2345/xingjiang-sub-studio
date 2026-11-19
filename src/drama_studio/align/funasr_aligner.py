"""FunASR (fa-zh + optional paraformer-zh) backend with stage VRAM pool and GPU/CPU fallback."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from drama_studio.align.base import ASRSegment, AlignerOutput
from drama_studio.align.model_manager import is_model_cached
from drama_studio.align.runtime import MODEL_POOL, is_oom_error, release_torch_memory, resolve_torch_device
from drama_studio.models import CharTimestamp

_DIGIT_MAP = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}


def _resolve_device(pref: str = "auto", min_free_vram_mb: int = 800) -> str:
    return resolve_torch_device(pref, min_free_vram_mb=min_free_vram_mb)


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
    acoustic_parts: list[str] = []
    mapping: list[tuple[str, int, int]] = []
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
            j = i + 1
            while j < n and (spoken_text[j].isdigit() or (spoken_text[j] == "." and j + 1 < n and spoken_text[j + 1].isdigit())):
                j += 1
            num_str = spoken_text[i:j]
            spoken_num = _expand_numeric_for_acoustic(num_str)
            for c_sub in spoken_num:
                acoustic_parts.append(c_sub)
            end_tok = tok_cursor + max(1, len(spoken_num))
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
            i += 1

    return " ".join(acoustic_parts), mapping


class FunASRAlignerBackend:
    """Forced aligner backed by FunASR fa-zh (TP-Aligner) with optional paraformer-zh ASR."""

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
        key = f"funasr_fa:{self.cache_dir}:{dev}"
        cached = MODEL_POOL.get("aligner", key)
        if cached is not None:
            return cached

        model_dir = str(self.cache_dir / "funasr_fa")
        try:
            model = AutoModel(model=model_dir, device=dev, disable_update=True, disable_pbar=True)
            MODEL_POOL.put("aligner", key, model, dev, exclusive_gpu=True)
            return model, dev
        except Exception:
            release_torch_memory()
            dev = "cpu"
            key = f"funasr_fa:{self.cache_dir}:cpu"
            cached_cpu = MODEL_POOL.get("aligner", key)
            if cached_cpu is not None:
                return cached_cpu
            model = AutoModel(model=model_dir, device="cpu", disable_update=True, disable_pbar=True)
            MODEL_POOL.put("aligner", key, model, "cpu", exclusive_gpu=False)
            return model, "cpu"

    def _get_asr_model(self, device_pref: str = "auto"):
        from funasr import AutoModel

        dev = _resolve_device(device_pref, min_free_vram_mb=1500)
        key = f"funasr_asr:{self.cache_dir}:{dev}"
        cached = MODEL_POOL.get("asr", key)
        if cached is not None:
            return cached

        model_dir = str(self.cache_dir / "funasr_asr")
        try:
            model = AutoModel(model=model_dir, device=dev, disable_update=True, disable_pbar=True)
            MODEL_POOL.put("asr", key, model, dev, exclusive_gpu=True)
            return model, dev
        except Exception:
            release_torch_memory()
            dev = "cpu"
            key = f"funasr_asr:{self.cache_dir}:cpu"
            cached_cpu = MODEL_POOL.get("asr", key)
            if cached_cpu is not None:
                return cached_cpu
            model = AutoModel(model=model_dir, device="cpu", disable_update=True, disable_pbar=True)
            MODEL_POOL.put("asr", key, model, "cpu", exclusive_gpu=False)
            return model, "cpu"

    def transcribe_anchors(self, audio_16k: np.ndarray, *, device: str = "auto") -> list[tuple[str, int, int]]:
        """Run full-timeline Paraformer ASR to locate character-level timeline anchors (handles edit shifts)."""
        if not self.has_asr_anchor() or len(audio_16k) < 1600:
            return []
        model, used_dev = self._get_asr_model(device)
        try:
            with tempfile.TemporaryDirectory() as td:
                wav_p = Path(td) / "asr.wav"
                sf.write(str(wav_p), audio_16k, 16000, subtype="PCM_16")
                res = model.generate(input=str(wav_p), batch_size_s=120)
        except Exception as exc:
            if is_oom_error(exc) and used_dev.startswith("cuda"):
                MODEL_POOL.unload_role("asr")
                model, _ = self._get_asr_model("cpu")
                with tempfile.TemporaryDirectory() as td:
                    wav_p = Path(td) / "asr.wav"
                    sf.write(str(wav_p), audio_16k, 16000, subtype="PCM_16")
                    res = model.generate(input=str(wav_p), batch_size_s=120)
            else:
                raise
        if not res:
            return []
        text = str(res[0].get("text", "")).replace(" ", "")
        ts = res[0].get("timestamp") or []
        anchors: list[tuple[str, int, int]] = []
        for ch, span in zip(text, ts):
            if len(span) >= 2:
                anchors.append((ch, int(span[0]), int(span[1])))
        return anchors

    def transcribe_windows(
        self,
        audio_16k: np.ndarray,
        windows_ms: list[tuple[int, int]],
        *,
        device: str = "auto",
    ) -> list[ASRSegment]:
        """Run FunASR Paraformer-ZH over pre-segmented VAD speech windows (Chinese fast ASR)."""
        if not self.has_asr_anchor() or not windows_ms:
            return []
        model, used_dev = self._get_asr_model(device)
        segments: list[ASRSegment] = []
        with tempfile.TemporaryDirectory() as td:
            for idx, (w_s, w_e) in enumerate(windows_ms):
                s_i = max(0, int(round(w_s * 16.0)))
                e_i = min(len(audio_16k), int(round(w_e * 16.0)))
                chunk = audio_16k[s_i:e_i]
                if len(chunk) < 1600:
                    continue
                wav_p = Path(td) / f"seg_{idx}.wav"
                sf.write(str(wav_p), chunk, 16000, subtype="PCM_16")
                try:
                    res = model.generate(input=str(wav_p), batch_size_s=60)
                except Exception as exc:
                    if is_oom_error(exc) and used_dev.startswith("cuda"):
                        MODEL_POOL.unload_role("asr")
                        model, used_dev = self._get_asr_model("cpu")
                        res = model.generate(input=str(wav_p), batch_size_s=60)
                    else:
                        raise
                if not res:
                    continue
                txt = str(res[0].get("text", "")).strip()
                if not txt:
                    continue
                ts = res[0].get("timestamp") or []
                compact = txt.replace(" ", "")
                cts: list[CharTimestamp] = []
                for ch, span in zip(compact, ts):
                    if len(span) >= 2:
                        cts.append(
                            CharTimestamp(
                                char=ch,
                                start_ms=w_s + int(span[0]),
                                end_ms=w_s + int(span[1]),
                                confidence=0.88,
                            )
                        )
                seg_s = cts[0].start_ms if cts else w_s
                seg_e = cts[-1].end_ms if cts else w_e
                segments.append(
                    ASRSegment(
                        start_ms=max(w_s, seg_s),
                        end_ms=min(w_e, max(seg_s + 80, seg_e)),
                        text=txt,
                        language="zh",
                        confidence=0.88,
                        model_used=f"FunASR Paraformer-ZH [{used_dev}]",
                        raw_char_timings=cts,
                        src_win_start_ms=w_s,
                        src_win_end_ms=w_e,
                    )
                )
        return segments

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
        if (language or "").strip().lower() in {"en", "eng", "english"}:
            return AlignerOutput(
                char_timings=[],
                backend_name=self.name,
                warnings=["FunASR fa-zh 仅支持中文对齐，英文项目已跳过 fa-zh。"],
            )

        s_idx = max(0, int(round(window_start_ms * 16000 / 1000.0)))
        e_idx = min(len(audio_16k), int(round(window_end_ms * 16000 / 1000.0)))
        segment = audio_16k[s_idx:e_idx]
        if len(segment) < 800:
            return AlignerOutput(char_timings=[], backend_name=self.name, warnings=["音频切片过短"])

        acoustic_query, mapping = _split_to_char_mapping(spoken_text)
        if not acoustic_query.strip() or not mapping:
            return AlignerOutput(char_timings=[], backend_name=self.name)

        model, used_dev = self._get_fa_model(device)
        try:
            with tempfile.TemporaryDirectory() as td:
                wav_p = Path(td) / "seg.wav"
                txt_p = Path(td) / "txt.txt"
                sf.write(str(wav_p), segment, 16000, subtype="PCM_16")
                txt_p.write_text(acoustic_query, encoding="utf-8")
                res = model.generate(input=(str(wav_p), str(txt_p)), data_type=("sound", "text"))
        except Exception as exc:
            if is_oom_error(exc) and used_dev.startswith("cuda"):
                MODEL_POOL.unload_role("aligner")
                model, used_dev = self._get_fa_model("cpu")
                with tempfile.TemporaryDirectory() as td:
                    wav_p = Path(td) / "seg.wav"
                    txt_p = Path(td) / "txt.txt"
                    sf.write(str(wav_p), segment, 16000, subtype="PCM_16")
                    txt_p.write_text(acoustic_query, encoding="utf-8")
                    res = model.generate(input=(str(wav_p), str(txt_p)), data_type=("sound", "text"))
            else:
                raise

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
