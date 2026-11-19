"""VAD-windowed speech transcription orchestrator supporting Qwen3-ASR (0.6B/1.7B) and FunASR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Callable

import numpy as np

from drama_studio.align.base import ASRSegment
from drama_studio.align.funasr_aligner import FunASRAlignerBackend
from drama_studio.align.model_manager import is_model_cached
from drama_studio.align.qwen_asr_backend import Qwen3ASRBackend
from drama_studio.align.runtime import MODEL_POOL
from drama_studio.audio.vad import detect_speech_segments


class MissingASRModelError(RuntimeError):
    """Raised when automatic subtitle generation or script matching has no local ASR model available."""

    def __init__(self, recommended_model_id: str, message: str) -> None:
        super().__init__(message)
        self.recommended_model_id = recommended_model_id


@dataclass
class ASRRunResult:
    segments: list[ASRSegment]
    detected_language: str
    model_used: str
    warnings: list[str] = field(default_factory=list)


def prepare_enhanced_asr_audio(
    audio_16k: np.ndarray,
    vad_spans: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Create a temporary softly-normalized 16k copy for whisper/low-volume ASR without altering timeline length."""
    if len(audio_16k) == 0:
        return audio_16k.astype(np.float32, copy=False)

    arr = audio_16k.astype(np.float32, copy=True)
    # Measure active speech RMS (fallback to overall top-25% energy if VAD missed ultra-quiet lines)
    if vad_spans:
        chunks = [
            arr[max(0, int(s * 16)) : min(len(arr), int(e * 16))]
            for s, e in vad_spans
            if e > s
        ]
        sample_cat = np.concatenate(chunks) if chunks else arr
    else:
        sample_cat = arr

    if len(sample_cat) == 0:
        return arr

    rms = float(np.sqrt(np.mean(sample_cat.astype(np.float64) ** 2)))
    rms_db = 20.0 * np.log10(max(1e-8, rms))
    if rms_db >= -24.0:
        return arr

    # Gentle gain boost capped at +18dB with soft-knee limiting so whispers become intelligible for ASR
    target_rms = 10.0 ** (-18.0 / 20.0)
    gain = min(8.0, target_rms / max(1e-6, rms))
    boosted = arr * gain
    # Smooth soft-clip above 0.85 to avoid harsh digital clipping
    over = np.abs(boosted) > 0.85
    if np.any(over):
        boosted[over] = np.sign(boosted[over]) * (0.85 + 0.14 * np.tanh((np.abs(boosted[over]) - 0.85) / 0.35))
    assert len(boosted) == len(audio_16k)
    return boosted.astype(np.float32, copy=False)


def build_asr_speech_windows(
    audio_16k: np.ndarray,
    vad_spans: list[tuple[int, int]],
    duration_ms: int,
    *,
    merge_gap_ms: int = 240,
    max_window_ms: int = 8500,
    min_speech_ms: int = 120,
    pad_ms: int = 120,
) -> list[tuple[int, int]]:
    """Merge nearby Silero VAD speech islands into bounded ASR windows without crossing long silence/BGM."""
    if not vad_spans or len(audio_16k) == 0:
        return []

    # Noise floor estimation to drop faint non-speech clicks
    abs_p20 = float(np.percentile(np.abs(audio_16k), 20))
    noise_db = 20.0 * np.log10(max(1e-7, abs_p20))

    filtered: list[tuple[int, int]] = []
    for vs, ve in vad_spans:
        if ve - vs < min_speech_ms:
            continue
        s_i = max(0, int(round(vs * 16.0)))
        e_i = min(len(audio_16k), int(round(ve * 16.0)))
        if e_i <= s_i:
            continue
        seg = audio_16k[s_i:e_i]
        seg_rms_db = 20.0 * np.log10(max(1e-7, float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))))
        if seg_rms_db < noise_db + 2.5 and seg_rms_db < -52.0:
            continue
        filtered.append((max(0, vs), min(duration_ms, ve)))

    if not filtered:
        return []

    merged: list[tuple[int, int]] = [filtered[0]]
    for vs, ve in filtered[1:]:
        ps, pe = merged[-1]
        if (vs - pe) <= merge_gap_ms and (ve - ps) <= max_window_ms:
            merged[-1] = (ps, ve)
        else:
            merged.append((vs, ve))

    # Apply safe non-overlapping padding
    windows: list[tuple[int, int]] = []
    for idx, (ws, we) in enumerate(merged):
        prev_end = merged[idx - 1][1] if idx > 0 else 0
        next_start = merged[idx + 1][0] if idx + 1 < len(merged) else duration_ms
        s_pad = max(prev_end, ws - pad_ms)
        e_pad = min(next_start, we + pad_ms)
        if e_pad - s_pad >= min_speech_ms:
            windows.append((s_pad, e_pad))
    return windows


def infer_dominant_language(segments: list[ASRSegment], fallback: str = "zh") -> str:
    if not segments:
        return fallback
    en_score = 0
    zh_score = 0
    for seg in segments:
        cjk = sum(1 for c in seg.text if "\u4e00" <= c <= "\u9fff")
        latin = sum(1 for c in seg.text if c.isascii() and c.isalpha())
        zh_score += cjk * 2
        en_score += latin
        if seg.language == "en":
            en_score += 4
        elif seg.language == "zh":
            zh_score += 4
    return "en" if en_score > zh_score else "zh"


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。？！?!])\s*|(?<=\.)\s+(?=[A-Z\"'\u4e00-\u9fff])")


def _map_fraction_onto_active_ms(
    frac: float,
    active_intervals: list[tuple[int, int]],
    fallback_start: int,
    fallback_end: int,
) -> int:
    """Map [0..1] text proportion onto active speech intervals so internal silences are excluded."""
    if not active_intervals:
        return fallback_start + int(round((fallback_end - fallback_start) * frac))
    total_speech = sum(max(1, e - s) for s, e in active_intervals)
    target_speech = max(0.0, min(float(total_speech), frac * total_speech))
    acc = 0.0
    for s, e in active_intervals:
        dur = float(max(1, e - s))
        if acc + dur >= target_speech - 1e-6:
            offset = max(0.0, target_speech - acc)
            return int(round(s + offset))
        acc += dur
    return active_intervals[-1][1]


def split_asr_segment_sentences(
    seg: ASRSegment,
    vad_spans: list[tuple[int, int]] | None = None,
) -> list[ASRSegment]:
    """Split a long multi-sentence ASR segment onto VAD speech intervals while preserving parent window bounds."""
    src_w_s = seg.src_win_start_ms if seg.src_win_start_ms >= 0 else seg.start_ms
    src_w_e = seg.src_win_end_ms if seg.src_win_end_ms >= 0 else seg.end_ms
    txt = seg.text.strip()
    if not txt:
        return []
    # Protect common English title/abbrev dots from naive sentence split
    protected = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.\s+", r"\1<DOT> ", txt)
    parts = [p.replace("<DOT>", ".").strip() for p in _SENTENCE_SPLIT_RE.split(protected) if p.strip()]
    if len(parts) <= 1:
        if seg.src_win_start_ms < 0:
            return [
                ASRSegment(
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    text=seg.text,
                    language=seg.language,
                    confidence=seg.confidence,
                    model_used=seg.model_used,
                    raw_char_timings=seg.raw_char_timings,
                    src_win_start_ms=src_w_s,
                    src_win_end_ms=src_w_e,
                )
            ]
        return [seg]

    active_vad: list[tuple[int, int]] = []
    if vad_spans:
        active_vad = [
            (max(seg.start_ms, vs), min(seg.end_ms, ve))
            for vs, ve in vad_spans
            if ve > seg.start_ms + 30 and vs < seg.end_ms - 30
        ]

    weights = [max(1, len(re.sub(r"\s+", "", p))) for p in parts]
    total_w = max(1, sum(weights))
    out: list[ASRSegment] = []
    cur_ms = active_vad[0][0] if active_vad else seg.start_ms
    cum_w = 0
    for idx, (part, w) in enumerate(zip(parts, weights)):
        cum_w += w
        if idx == len(parts) - 1:
            end_ms = active_vad[-1][1] if active_vad else seg.end_ms
        else:
            mapped = _map_fraction_onto_active_ms(cum_w / total_w, active_vad, seg.start_ms, seg.end_ms)
            end_ms = min(seg.end_ms, max(cur_ms + 100, mapped))
        out.append(
            ASRSegment(
                start_ms=cur_ms,
                end_ms=max(cur_ms + 80, end_ms),
                text=part,
                language=seg.language,
                confidence=seg.confidence,
                model_used=seg.model_used,
                src_win_start_ms=src_w_s,
                src_win_end_ms=src_w_e,
            )
        )
        if idx + 1 < len(parts) and active_vad:
            next_start = end_ms
            for vs, ve in active_vad:
                if ve >= end_ms + 40:
                    next_start = max(end_ms, vs)
                    break
            cur_ms = min(seg.end_ms - 60, next_start)
        else:
            cur_ms = end_ms
    return out


def select_asr_model_id(
    asr_pref: str,
    language_pref: str,
    cache_dir: Path,
) -> tuple[str, list[str]]:
    """Resolve effective ASR model ID respecting user choice, language rules, and local availability."""
    warnings: list[str] = []
    lang = (language_pref or "auto").lower()

    if asr_pref == "qwen3_asr_1.7b":
        if is_model_cached("qwen3_asr_1.7b", cache_dir):
            return "qwen3_asr_1.7b", warnings
        if is_model_cached("qwen3_asr_0.6b", cache_dir):
            warnings.append("未安装 Qwen3-ASR-1.7B，已回退使用默认 Qwen3-ASR-0.6B。")
            return "qwen3_asr_0.6b", warnings
        raise MissingASRModelError(
            "qwen3_asr_1.7b",
            "未安装 Qwen3-ASR-1.7B（也未找到 0.6B 模型）。请在「模型下载与缓存」中下载后再执行识别。",
        )

    if asr_pref == "qwen3_asr_0.6b":
        if is_model_cached("qwen3_asr_0.6b", cache_dir):
            return "qwen3_asr_0.6b", warnings
        raise MissingASRModelError(
            "qwen3_asr_0.6b",
            "未安装默认识别模型 Qwen3-ASR-0.6B。请在「模型下载与缓存」中下载后继续。",
        )

    if asr_pref == "funasr_asr":
        if lang == "en":
            if is_model_cached("qwen3_asr_0.6b", cache_dir):
                warnings.append("FunASR Paraformer-ZH 仅支持中文，英文任务已自动切换至 Qwen3-ASR-0.6B。")
                return "qwen3_asr_0.6b", warnings
            raise MissingASRModelError(
                "qwen3_asr_0.6b",
                "英文识别不可使用中文 FunASR Paraformer-ZH，请先下载 Qwen3-ASR-0.6B。",
            )
        if is_model_cached("funasr_asr", cache_dir):
            return "funasr_asr", warnings
        if is_model_cached("qwen3_asr_0.6b", cache_dir):
            warnings.append("未安装 FunASR Paraformer-ZH，已改用 Qwen3-ASR-0.6B。")
            return "qwen3_asr_0.6b", warnings
        raise MissingASRModelError(
            "funasr_asr",
            "未安装 FunASR Paraformer-ZH，请先在模型管理中下载。",
        )

    # Auto mode: Qwen3-ASR-0.6B is the default general Chinese+English model
    if is_model_cached("qwen3_asr_0.6b", cache_dir):
        return "qwen3_asr_0.6b", warnings
    if lang != "en" and is_model_cached("funasr_asr", cache_dir):
        warnings.append("未安装 Qwen3-ASR-0.6B，中文识别临时使用已缓存的 FunASR Paraformer-ZH。")
        return "funasr_asr", warnings
    if is_model_cached("qwen3_asr_1.7b", cache_dir):
        warnings.append("未安装 Qwen3-ASR-0.6B，使用已安装的 Qwen3-ASR-1.7B。")
        return "qwen3_asr_1.7b", warnings

    raise MissingASRModelError(
        "qwen3_asr_0.6b",
        "未检测到本地语音识别模型（推荐下载默认中英模型 Qwen3-ASR-0.6B）。请在「模型下载与缓存」下载后运行。",
    )


def run_asr_transcription(
    audio_16k: np.ndarray,
    vad_spans: list[tuple[int, int]],
    duration_ms: int,
    *,
    cache_dir: Path,
    asr_pref: str = "auto",
    language_pref: str = "auto",
    device: str = "auto",
    enhance_quiet_speech: bool = True,
    allow_vram_downgrade: bool = True,
    unload_after: bool = True,
    progress_cb: Callable[[float, str], None] | None = None,
) -> ASRRunResult:
    """Execute Stage-2 VAD-windowed ASR transcription and optionally release model VRAM immediately."""
    model_id, warnings_out = select_asr_model_id(asr_pref, language_pref, cache_dir)

    work_audio = audio_16k
    eff_vad = list(vad_spans)
    if enhance_quiet_speech:
        work_audio = prepare_enhanced_asr_audio(audio_16k, vad_spans)
        # Re-run VAD on enhanced copy if raw audio yielded unusually sparse speech (< 5% of clip)
        speech_ms = sum(max(0, e - s) for s, e in eff_vad)
        if duration_ms > 2000 and speech_ms < duration_ms * 0.05:
            enh_vad, _, _ = detect_speech_segments(work_audio, 16000, threshold=0.28, neg_threshold=0.14)
            if len(enh_vad) > len(eff_vad):
                eff_vad = enh_vad

    windows = build_asr_speech_windows(work_audio, eff_vad, duration_ms)
    if not windows:
        return ASRRunResult(
            segments=[],
            detected_language="en" if language_pref == "en" else "zh",
            model_used=model_id,
            warnings=warnings_out + ["未检测到有效人声语音区间。"],
        )

    try:
        if model_id == "funasr_asr":
            if progress_cb:
                progress_cb(0.15, "使用 FunASR Paraformer-ZH 识别中文对白...")
            be = FunASRAlignerBackend(cache_dir)
            raw_segs = be.transcribe_windows(work_audio, windows, device=device)
            model_used = raw_segs[0].model_used if raw_segs else "FunASR Paraformer-ZH"
        else:
            be_q = Qwen3ASRBackend(model_id, cache_dir)
            qres = be_q.transcribe_windows(
                work_audio,
                windows,
                language_pref=language_pref,
                device=device,
                allow_vram_downgrade=allow_vram_downgrade,
                progress_cb=progress_cb,
            )
            raw_segs = qres.segments
            model_used = qres.model_used
            warnings_out.extend(qres.warnings)
    finally:
        if unload_after:
            MODEL_POOL.unload_role("asr")

    split_segs: list[ASRSegment] = []
    for seg in raw_segs:
        split_segs.extend(split_asr_segment_sentences(seg, vad_spans=eff_vad))

    dom_lang = (
        "en"
        if language_pref == "en"
        else ("zh" if language_pref == "zh" else infer_dominant_language(split_segs))
    )
    return ASRRunResult(
        segments=split_segs,
        detected_language=dom_lang,
        model_used=model_used,
        warnings=warnings_out,
    )
