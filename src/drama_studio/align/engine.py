"""End-to-end alignment orchestration, timeline drift compensation, DP layout, and QA checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np

from drama_studio.align.base import UtteranceWindow
from drama_studio.align.funasr_aligner import FunASRAlignerBackend
from drama_studio.align.qwen_aligner import Qwen3AlignerBackend
from drama_studio.align.runtime import MODEL_POOL
from drama_studio.models import (
    AlignmentConfig,
    AnomalyItem,
    AnomalySeverity,
    AnomalyType,
    CharTimestamp,
    SubtitleCue,
    SubtitleTemplate,
)
from drama_studio.nlp.punctuation import (
    STRONG_PUNCT,
    count_english_words,
    detect_text_language,
    extract_canonical_spoken_sequence,
    strip_display_punctuation,
)
from drama_studio.nlp.script_io import (
    ScriptAlignmentResult,
    ScriptDocument,
    align_script_turns_to_cues,
    parse_dialogue_script,
)
from drama_studio.nlp.segmenter import (
    UnitToken,
    build_unit_tokens,
    format_single_cue_without_split,
    measure_text_width,
    register_custom_words,
    segment_tokens_dp,
    verify_text_integrity,
    width_fits_limit,
)
from drama_studio.nlp.srt_io import normalize_cues_for_resolve


class MissingAlignerModelError(RuntimeError):
    """Raised when acoustic forced alignment has no local aligner model available and fallback is disabled."""

    def __init__(self, recommended_model_id: str, message: str) -> None:
        super().__init__(message)
        self.recommended_model_id = recommended_model_id


@dataclass
class AlignedUtteranceData:
    """Intermediate forced-aligned utterance window before template-driven DP segmentation."""

    clean_text: str
    window_start_ms: int
    window_end_ms: int
    char_timings: list[CharTimestamp]
    win_char_turns: list[int]
    win_char_hints: list[str]
    per_cue_turn: list[int]
    cues_snapshot: list[SubtitleCue]
    max_local_turn: int


@dataclass
class ForcedAlignmentStageResult:
    """Cacheable output of Stage 3A (forced alignment + acoustic refinement)."""

    aligned_windows: list[AlignedUtteranceData]
    global_offset_ms: int
    aligner_used: str
    effective_language: str
    extra_phrases: list[str]
    pre_warnings: list[AnomalyItem]
    script_mismatch_ratio: float | None = None


@dataclass
class AlignmentEngineResult:
    cues: list[SubtitleCue]
    anomalies: list[AnomalyItem]
    global_offset_ms: int
    aligner_used: str
    nlp_used: str
    text_integrity_ok: bool


def _vad_overlap_ratio(start_ms: int, end_ms: int, vad_spans: list[tuple[int, int]]) -> float:
    dur = max(1, end_ms - start_ms)
    overlap = 0
    for vs, ve in vad_spans:
        if ve <= start_ms:
            continue
        if vs >= end_ms:
            break
        overlap += min(end_ms, ve) - max(start_ms, vs)
    return min(1.0, overlap / dur)


def _segment_rms_db(audio_16k: np.ndarray, start_ms: int, end_ms: int) -> float:
    s = max(0, int(round(start_ms * 16.0)))
    e = min(len(audio_16k), int(round(end_ms * 16.0)))
    if e <= s:
        return -90.0
    seg = audio_16k[s:e]
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
    if rms <= 1e-8:
        return -90.0
    return float(20.0 * np.log10(rms))


def _estimate_global_vad_offset(
    cues: list[SubtitleCue],
    vad_spans: list[tuple[int, int]],
    duration_ms: int,
) -> int:
    """Find best global timeline shift in [-6000, +6000] ms maximizing cue overlap with VAD speech."""
    if not cues or not vad_spans:
        return 0

    def score_shift(delta: int) -> float:
        total = 0.0
        for c in cues:
            s = max(0, c.start_ms + delta)
            e = min(duration_ms, c.end_ms + delta)
            if e > s:
                total += _vad_overlap_ratio(s, e, vad_spans) * (e - s)
        return total

    base_score = score_shift(0)
    best_delta = 0
    best_score = base_score

    # Coarse search (200ms step)
    for d in range(-6000, 6200, 200):
        sc = score_shift(d)
        if sc > best_score:
            best_score = sc
            best_delta = d

    # Fine search around best_delta (50ms step)
    for d in range(best_delta - 200, best_delta + 250, 50):
        sc = score_shift(d)
        if sc > best_score:
            best_score = sc
            best_delta = d

    # Require meaningful improvement (>12%) and >= 200ms shift to avoid jitter
    if abs(best_delta) >= 200 and best_score > base_score * 1.12:
        return best_delta
    return 0


def _anchor_cues_with_asr(
    cues: list[SubtitleCue],
    asr_anchors: list[tuple[str, int, int]],
) -> dict[int, tuple[int, int]]:
    """Map cue index -> anchored (start_ms, end_ms) using full-timeline ASR sequence matching."""
    if not cues or not asr_anchors:
        return {}

    asr_str = "".join(ch for ch, _, _ in asr_anchors)
    cue_seqs = [extract_canonical_spoken_sequence(c.raw_text) for c in cues]
    full_srt = "".join(cue_seqs)
    if not full_srt or not asr_str:
        return {}

    sm = SequenceMatcher(None, full_srt, asr_str, autojunk=False)
    char_map: dict[int, int] = {}  # full_srt idx -> asr idx
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                char_map[i1 + k] = j1 + k

    anchored: dict[int, tuple[int, int]] = {}
    cursor = 0
    for idx, seq in enumerate(cue_seqs):
        n = len(seq)
        matched_asr = [char_map[p] for p in range(cursor, cursor + n) if p in char_map]
        cursor += n
        if matched_asr:
            s_ms = asr_anchors[min(matched_asr)][1]
            e_ms = asr_anchors[max(matched_asr)][2]
            anchored[idx] = (s_ms, max(s_ms + 100, e_ms))
    return anchored


def _find_nearest_vad_valley(
    approx_ms: int,
    vad_spans: list[tuple[int, int]],
    *,
    min_gap_ms: int = 160,
    max_dist_ms: int = 1200,
) -> int | None:
    """Locate the midpoint of the nearest VAD silence gap near `approx_ms`."""
    best_mid: int | None = None
    best_dist = max_dist_ms + 1
    for vi in range(len(vad_spans) - 1):
        ve = vad_spans[vi][1]
        vs_nxt = vad_spans[vi + 1][0]
        if vs_nxt - ve >= min_gap_ms:
            mid = (ve + vs_nxt) // 2
            dist = abs(mid - approx_ms)
            if dist <= max_dist_ms and dist < best_dist:
                best_dist = dist
                best_mid = mid
    return best_mid


def _group_cues_into_windows(
    cues: list[SubtitleCue],
    vad_spans: list[tuple[int, int]],
    anchored_bounds: dict[int, tuple[int, int]],
    global_offset_ms: int,
    duration_ms: int,
    max_window_ms: int = 24000,
) -> list[UtteranceWindow]:
    """Group nearby cues into continuous UtteranceWindows bounded safely by VAD speech & silence."""
    if not cues:
        return []

    # First compute preliminary adjusted bounds per cue
    prelim: list[tuple[int, int, SubtitleCue]] = []
    cue_pos_map: dict[int, int] = {}
    for idx, cue in enumerate(cues):
        cue_pos_map[id(cue)] = idx
        if idx in anchored_bounds:
            s, e = anchored_bounds[idx]
        else:
            s = max(0, cue.start_ms + global_offset_ms)
            e = min(duration_ms, max(s + 120, cue.end_ms + global_offset_ms))
        prelim.append((s, e, cue))

    groups: list[list[tuple[int, int, SubtitleCue]]] = []
    cur: list[tuple[int, int, SubtitleCue]] = [prelim[0]]
    for item in prelim[1:]:
        prev_s, prev_e, prev_cue = cur[-1]
        cur_s, cur_e, cur_cue = item
        gap = cur_s - prev_e
        span_dur = cur_e - cur[0][0]
        same_asr_win = (
            prev_cue.src_win_start_ms >= 0
            and prev_cue.src_win_start_ms == cur_cue.src_win_start_ms
            and prev_cue.src_win_end_ms == cur_cue.src_win_end_ms
        )
        if same_asr_win and (max(cur_e, cur_cue.src_win_end_ms) - min(cur[0][0], cur[0][2].src_win_start_ms)) <= max_window_ms:
            cur.append(item)
            continue

        prev_txt = prev_cue.raw_text.strip()
        prev_strong_end = bool(prev_txt) and prev_txt[-1] in STRONG_PUNCT
        # Check whether a real VAD silence gap separates prev_cue and cur_cue around boundary
        valley = _find_nearest_vad_valley((prev_e + cur_s) // 2, vad_spans, min_gap_ms=220, max_dist_ms=900)
        has_vad_sep = valley is not None and (prev_strong_end or gap >= 420)
        if (gap <= 1300 and not has_vad_sep) and span_dur <= max_window_ms:
            cur.append(item)
        else:
            groups.append(cur)
            cur = [item]
    groups.append(cur)

    windows: list[UtteranceWindow] = []
    for g_idx, grp in enumerate(groups):
        raw_s = grp[0][0]
        raw_e = grp[-1][1]
        src_starts = [it[2].src_win_start_ms for it in grp if it[2].src_win_start_ms >= 0]
        src_ends = [it[2].src_win_end_ms for it in grp if it[2].src_win_end_ms >= 0]
        has_src_win = bool(src_starts and src_ends)
        if has_src_win:
            cand_s = min(raw_s, min(src_starts))
            cand_e = max(raw_e, max(src_ends))
            if (cand_e - cand_s) <= max_window_ms:
                raw_s, raw_e = cand_s, cand_e

        # Prefer silent VAD valley midpoints over noisy raw SRT midpoints so speech onsets are never clipped
        if g_idx == 0:
            prev_limit = 0
        else:
            approx_prev = (groups[g_idx - 1][-1][1] + raw_s) // 2
            valley_prev = _find_nearest_vad_valley(approx_prev, vad_spans, min_gap_ms=140, max_dist_ms=1200)
            prev_limit = valley_prev if valley_prev is not None else approx_prev

        if g_idx == len(groups) - 1:
            next_limit = duration_ms
        else:
            approx_next = (raw_e + groups[g_idx + 1][0][0]) // 2
            valley_next = _find_nearest_vad_valley(approx_next, vad_spans, min_gap_ms=140, max_dist_ms=1200)
            next_limit = valley_next if valley_next is not None else approx_next

        # Keep true ASR window boundaries intact when groups are derived from Stage 2 VAD-bounded windows
        if has_src_win:
            prev_limit = min(prev_limit, raw_s)
            next_limit = max(next_limit, raw_e)

        # Detect per-shot local drift when ASR anchor is unavailable and current group misses speech VAD
        local_shift = 0
        if vad_spans and not has_src_win and not any(cue_pos_map.get(id(it[2]), -1) in anchored_bounds for it in grp):
            grp_dur = max(1, raw_e - raw_s)
            base_ov = _vad_overlap_ratio(raw_s, raw_e, vad_spans)
            if base_ov < 0.55 and grp_dur < 10000:
                best_ov = base_ov
                for cand_shift in range(-2500, 2600, 150):
                    cs = raw_s + cand_shift
                    ce = raw_e + cand_shift
                    if cs < prev_limit or ce > next_limit:
                        continue
                    ov = _vad_overlap_ratio(cs, ce, vad_spans)
                    if ov > best_ov:
                        best_ov = ov
                        local_shift = cand_shift
                if not (best_ov >= 0.70 and best_ov > max(0.25, base_ov) * 1.35):
                    local_shift = 0

        margin_ms = 260 if has_src_win else 1250
        win_s = max(prev_limit, raw_s + local_shift - margin_ms)
        win_e = min(next_limit, raw_e + local_shift + margin_ms)

        # Snap outward slightly if a VAD span starts/ends right inside search margin
        for vs, ve in vad_spans:
            if vs <= win_s <= ve and vs >= prev_limit - 100:
                win_s = max(prev_limit, vs - 80)
            if vs <= win_e <= ve and ve <= next_limit + 100:
                win_e = min(next_limit, ve + 80)

        combined_text = " ".join(it[2].raw_text.replace("\n", " ").strip() for it in grp)
        windows.append(
            UtteranceWindow(
                cues=[it[2] for it in grp],
                window_start_ms=max(0, win_s),
                window_end_ms=min(duration_ms, max(win_s + 200, win_e)),
                clean_text=combined_text,
            )
        )
    return windows


def _vad_fallback_align(
    audio_16k: np.ndarray,
    window: UtteranceWindow,
    vad_spans: list[tuple[int, int]],
) -> list[CharTimestamp]:
    """Conservative VAD-island fallback when no neural alignment model is cached locally."""
    spoken = extract_canonical_spoken_sequence(window.clean_text)
    if not spoken:
        return []

    # Collect active VAD intervals inside window
    active = [
        (max(window.window_start_ms, vs), min(window.window_end_ms, ve))
        for vs, ve in vad_spans
        if ve > window.window_start_ms and vs < window.window_end_ms
    ]
    if not active:
        s0 = window.cues[0].start_ms
        e0 = max(s0 + 200, window.cues[-1].end_ms)
        active = [(s0, e0)]

    total_speech_ms = sum(max(1, e - s) for s, e in active)
    n = len(spoken)
    out: list[CharTimestamp] = []
    # Map proportionally across actual VAD speech time (NOT across silence!)
    char_idx = 0
    cum = 0.0
    for s, e in active:
        dur = max(1, e - s)
        target_end_idx = n if (s, e) == active[-1] else min(n, int(round((cum + dur) / total_speech_ms * n)))
        cnt = max(0, target_end_idx - char_idx)
        if cnt > 0:
            step = dur / cnt
            for k in range(cnt):
                cs = int(round(s + k * step))
                ce = int(round(s + (k + 1) * step))
                out.append(
                    CharTimestamp(
                        char=spoken[char_idx + k],
                        start_ms=cs,
                        end_ms=max(cs + 25, ce),
                        confidence=0.40,
                        interpolated=True,
                    )
                )
            char_idx = target_end_idx
        cum += dur
    return out


def _apply_spacing_and_bridge(
    cues: list[SubtitleCue],
    template: SubtitleTemplate,
    duration_ms: int,
) -> list[SubtitleCue]:
    """Enforce Resolve monotonic non-overlap while bridging intra-sentence gaps to avoid flash frames."""
    ordered = normalize_cues_for_resolve(
        cues,
        min_gap_ms=1,
        min_duration_ms=min(80, template.min_duration_ms),
    )
    if len(ordered) <= 1:
        return ordered

    target_gap = max(1, template.min_gap_ms)
    for idx in range(len(ordered) - 1):
        prev = ordered[idx]
        nxt = ordered[idx + 1]
        prev_stripped = prev.raw_text.strip()
        prev_strong_end = bool(prev_stripped) and prev_stripped[-1] in STRONG_PUNCT
        same_turn = (prev.turn_id == nxt.turn_id) and not prev_strong_end
        cur_gap = nxt.start_ms - prev.end_ms

        if template.bridge_sentence_gaps and same_turn and 0 <= cur_gap <= template.bridge_max_gap_ms:
            # Extend right edge right up to nxt.start_ms - 1ms (1ms seam = zero blank frames in Resolve)
            bridged_end = max(prev.end_ms, nxt.start_ms - 1)
            prev.end_ms = min(duration_ms, bridged_end) if duration_ms > 0 else bridged_end
            continue

        if cur_gap < target_gap:
            min_len = min(80, template.min_duration_ms)
            needed = target_gap - cur_gap
            # Prefer shaving padded tail of prev before shifting nxt forward
            if prev.end_ms - needed >= prev.start_ms + min_len:
                prev.end_ms -= needed
            elif nxt.end_ms - (nxt.start_ms + needed) >= min_len:
                nxt.start_ms += needed
            else:
                prev.end_ms = max(prev.start_ms + 40, nxt.start_ms - 1)

    return normalize_cues_for_resolve(
        ordered,
        min_gap_ms=1,
        min_duration_ms=min(80, template.min_duration_ms),
    )


def _apply_non_invading_padding(
    cues: list[SubtitleCue],
    raw_bounds: list[tuple[int, int]],
    template: SubtitleTemplate,
    duration_ms: int,
    *,
    video_fps: float = 25.0,
    pad_mask: list[bool] | None = None,
) -> list[SubtitleCue]:
    """Expand head/tail breathing room only into real surrounding silence, never invading adjacent speech."""
    if not cues:
        return []
    n = len(cues)
    mask = pad_mask if (pad_mask is not None and len(pad_mask) == n) else [True] * n
    lead_ms, tail_ms = template.resolve_padding_ms(video_fps)
    end_cap = duration_ms if duration_ms > 0 else 10**9
    min_keep = min(80, template.min_duration_ms)

    starts = [max(0, int(rs)) for rs, _ in raw_bounds]
    ends = [max(max(0, int(rs)) + 1, int(re)) for rs, re in raw_bounds]

    if mask[0]:
        starts[0] = max(0, starts[0] - lead_ms)
    if mask[-1]:
        bounded_last = max(starts[-1] + 40, ends[-1] + tail_ms)
        ends[-1] = min(end_cap, bounded_last)

    want_tail = max(0, tail_ms)
    want_lead = max(0, lead_ms)
    shrink_tail = tail_ms if tail_ms < 0 else 0
    shrink_lead = lead_ms if lead_ms < 0 else 0

    for idx in range(n - 1):
        prev = cues[idx]
        nxt = cues[idx + 1]
        prev_re = raw_bounds[idx][1]
        nxt_rs = raw_bounds[idx + 1][0]

        prev_eff_shrink = shrink_tail if mask[idx] else 0
        nxt_eff_shrink = shrink_lead if mask[idx + 1] else 0
        eff_tail_base = prev_re + prev_eff_shrink
        eff_lead_base = nxt_rs - nxt_eff_shrink
        eff_gap = eff_lead_base - eff_tail_base

        prev_stripped = prev.raw_text.strip()
        prev_strong_end = bool(prev_stripped) and prev_stripped[-1] in STRONG_PUNCT
        same_turn = (prev.turn_id == nxt.turn_id) and not prev_strong_end
        will_bridge = (
            template.bridge_sentence_gaps
            and same_turn
            and 0 <= eff_gap <= template.bridge_max_gap_ms
        )
        req_gap = 1 if will_bridge else max(1, template.min_gap_ms)
        usable = max(0, eff_gap - req_gap)

        prev_want = want_tail if mask[idx] else 0
        nxt_want = want_lead if mask[idx + 1] else 0
        total_want = prev_want + nxt_want
        if total_want <= 0 or usable == 0:
            tail_add = 0
            lead_sub = 0
        elif usable >= total_want:
            tail_add = prev_want
            lead_sub = nxt_want
        else:
            # Share tight silence pauses proportionally without crossing the neighbor's spoken word edge
            tail_add = int(round(usable * (prev_want / total_want)))
            lead_sub = min(nxt_want, usable - tail_add)

        if mask[idx]:
            ends[idx] = min(end_cap, max(raw_bounds[idx][0] + 40, eff_tail_base + tail_add))
        if mask[idx + 1]:
            starts[idx + 1] = max(0, eff_lead_base - lead_sub)

    for cue, s_ms, e_ms in zip(cues, starts, ends):
        cue.start_ms = s_ms
        cue.end_ms = max(s_ms + min_keep, e_ms)

    return _apply_spacing_and_bridge(cues, template, duration_ms)


def _apply_padding_and_merge(
    cues: list[SubtitleCue],
    template: SubtitleTemplate,
    duration_ms: int,
    *,
    video_fps: float = 25.0,
) -> list[SubtitleCue]:
    """Apply lead/tail offsets, optional short-cue merging (within same speaker turn), and Resolve bridging."""
    if not cues:
        return []

    # Optional short-cue auto-merging before padding (never merge across turn boundaries)
    eff_width_mode = "strict" if template.language == "en" else template.width_mode
    merged: list[SubtitleCue] = []
    if template.auto_merge_short:
        for cue in cues:
            if merged:
                prev = merged[-1]
                prev_raw_stripped = prev.raw_text.strip()
                ends_with_strong_punct = bool(prev_raw_stripped) and prev_raw_stripped[-1] in STRONG_PUNCT
                gap = cue.start_ms - prev.end_ms
                prev_w = measure_text_width(prev.effective_text, eff_width_mode)
                cur_w = measure_text_width(cue.effective_text, eff_width_mode)
                joiner = " " if template.language == "en" else ""
                comb_flat = (prev.effective_text.replace("\n", " ").strip() + joiner + cue.effective_text.replace("\n", " ").strip()).strip()
                comb_w = measure_text_width(comb_flat, eff_width_mode)
                if (
                    prev.turn_id == cue.turn_id
                    and not ends_with_strong_punct
                    and min(prev_w, cur_w) <= template.short_merge_max_chars
                    and gap <= template.short_merge_max_gap_ms
                    and (cue.end_ms - prev.start_ms) <= template.max_duration_ms
                    and width_fits_limit(comb_w, template.max_chars_per_line * template.max_lines)
                ):
                    new_text, fits = format_single_cue_without_split(
                        prev.raw_text.strip() + " " + cue.raw_text.strip(),
                        template,
                    )
                    if fits:
                        prev.end_ms = cue.end_ms
                        prev.raw_text = (prev.raw_text.strip() + " " + cue.raw_text.strip()).strip()
                        prev.display_text = new_text
                        prev.confidence = min(prev.confidence, cue.confidence)
                        prev.char_timings.extend(cue.char_timings)
                        prev.source_cue_indices.extend(cue.source_cue_indices)
                        continue
            merged.append(cue)
    else:
        merged = cues

    # Expand pre/post breathing room strictly into genuine silence (never invade continuous neighbor words)
    raw_bounds = [(c.start_ms, c.end_ms) for c in merged]
    return _apply_non_invading_padding(merged, raw_bounds, template, duration_ms, video_fps=video_fps)


def run_quality_checks(
    cues: list[SubtitleCue],
    template: SubtitleTemplate,
    vad_spans: list[tuple[int, int]],
    audio_16k: np.ndarray,
    duration_ms: int,
    low_conf_thresh: float = 0.55,
) -> list[AnomalyItem]:
    """Inspect all output cues for timing, length, reading speed, silence, and confidence issues."""
    anomalies: list[AnomalyItem] = []

    # Estimate noise floor from 16k audio
    if len(audio_16k) > 0:
        abs_sorted = np.percentile(np.abs(audio_16k), 25)
        noise_db = 20.0 * np.log10(max(1e-7, float(abs_sorted)))
    else:
        noise_db = -70.0

    eff_width_mode = "strict" if template.language == "en" else template.width_mode
    for idx, cue in enumerate(cues):
        cue.anomaly_flags.clear()
        text = cue.effective_text
        lines = cue.lines()
        dur = cue.duration_ms
        max_w = measure_text_width(text, eff_width_mode)
        flat_chars = (
            len(text.replace("\n", " ").strip())
            if template.language == "en"
            else len(extract_canonical_spoken_sequence(text))
        )
        cps = flat_chars / max(0.1, dur / 1000.0)

        vad_ov = _vad_overlap_ratio(cue.start_ms, cue.end_ms, vad_spans)
        seg_rms = _segment_rms_db(audio_16k, cue.start_ms, cue.end_ms)

        is_vad_estimated = bool(cue.char_timings) and (
            sum(1 for ct in cue.char_timings if ct.interpolated) / len(cue.char_timings) >= 0.8
        )
        if is_vad_estimated:
            est_msg = "VAD估算时间轴(未声学对齐)"
            cue.anomaly_flags.append(est_msg)
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.LOW_CONFIDENCE,
                    severity=AnomalySeverity.WARNING,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    cue_index=cue.index,
                    message=f"{est_msg}：该字幕由语音区间按字数比例估算生成，非逐字声学强制对齐",
                    suggestion="建议安装并启用 Qwen3-ForcedAligner-0.6B 或 FunASR fa-zh 后重新执行对齐。",
                )
            )

        if vad_ov < 0.08 and seg_rms < noise_db + 4.5:
            cue.confidence = min(cue.confidence, 0.30)
            msg = "该字幕对应音频区段几乎无语音能量（处于静音或极轻声气口）"
            cue.anomaly_flags.append(msg)
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.SILENT_REGION,
                    severity=AnomalySeverity.WARNING,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    cue_index=cue.index,
                    message=msg,
                    suggestion="请在波形轴试听确认是否为轻声对白或时间轴错位。",
                )
            )
        elif cue.confidence < low_conf_thresh:
            msg = f"对齐置信度偏低 ({int(cue.confidence * 100)}%)"
            cue.anomaly_flags.append(msg)
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.LOW_CONFIDENCE,
                    severity=AnomalySeverity.WARNING,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    cue_index=cue.index,
                    message=msg,
                    suggestion="建议试听核对起止点，必要时在时间轴拖动边界微调。",
                )
            )

        if not width_fits_limit(max_w, template.max_chars_per_line) or len(lines) > template.max_lines:
            unit_lbl = "字符" if template.language == "en" else "字"
            msg = f"超出模板排版限制（{max_w:.1f}/{template.max_chars_per_line}{unit_lbl}，{len(lines)}/{template.max_lines}行）"
            cue.anomaly_flags.append(msg)
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.OVER_CHAR_LIMIT,
                    severity=AnomalySeverity.ERROR,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    cue_index=cue.index,
                    message=msg,
                    suggestion="可调大每行字数、切换2行模式或点击拆分该条字幕。",
                )
            )
        elif template.language == "en" and len(lines) == 2:
            wc1 = count_english_words(lines[0])
            wc2 = count_english_words(lines[1])
            if wc1 < template.min_words_line1 or wc2 < template.min_words_line2:
                msg = (
                    f"英文双行单词数不满足模板要求（第1行 {wc1}/{template.min_words_line1} 词，"
                    f"第2行 {wc2}/{template.min_words_line2} 词）"
                )
                cue.anomaly_flags.append(msg)
                anomalies.append(
                    AnomalyItem(
                        anomaly_type=AnomalyType.LAYOUT_WARNING,
                        severity=AnomalySeverity.WARNING,
                        start_ms=cue.start_ms,
                        end_ms=cue.end_ms,
                        cue_index=cue.index,
                        message=msg,
                        suggestion="可调整字符限制或手动合并/调整换行。",
                    )
                )

        if dur < template.min_duration_ms:
            msg = f"显示时长过短 ({dur} ms < {template.min_duration_ms} ms)"
            cue.anomaly_flags.append(msg)
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.TOO_SHORT,
                    severity=AnomalySeverity.INFO,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    cue_index=cue.index,
                    message=msg,
                    suggestion="可合并相邻短句或适度增加尾部延后偏移。",
                )
            )
        elif dur > template.max_duration_ms:
            msg = f"显示时长过长 ({dur} ms > {template.max_duration_ms} ms)"
            cue.anomaly_flags.append(msg)
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.TOO_LONG,
                    severity=AnomalySeverity.INFO,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    cue_index=cue.index,
                    message=msg,
                    suggestion="建议拆分长句或在时间轴收紧结束时间。",
                )
            )

        if cps > template.max_cps:
            msg = f"阅读速度偏快 ({cps:.1f} 字/秒 > {template.max_cps:.1f})"
            cue.anomaly_flags.append(msg)
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.OVER_CPS,
                    severity=AnomalySeverity.WARNING,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    cue_index=cue.index,
                    message=msg,
                    suggestion="台词较密，请确认语音起止点或放宽最长显示限制。",
                )
            )

        if idx > 0 and cue.start_ms < cues[idx - 1].end_ms:
            msg = "与上一条字幕时间重叠"
            cue.anomaly_flags.append(msg)
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.TIMELINE_OVERLAP,
                    severity=AnomalySeverity.ERROR,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    cue_index=cue.index,
                    message=msg,
                    suggestion="建议重新规范时间间隔。",
                )
            )

        if duration_ms > 0 and cue.end_ms > duration_ms + 50:
            msg = "字幕结束时间超出视频总时长"
            cue.anomaly_flags.append(msg)
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.OUT_OF_BOUNDS,
                    severity=AnomalySeverity.WARNING,
                    start_ms=cue.start_ms,
                    end_ms=cue.end_ms,
                    cue_index=cue.index,
                    message=msg,
                    suggestion="请截断至视频总时长以内。",
                )
            )

    return anomalies


def _compute_window_turn_ids(
    win_cues: list[SubtitleCue],
    cue_turn_map: dict[int, list[int]],
    cue_hint_map: dict[int, list[str]],
    script_matched: bool,
) -> tuple[list[int], list[str], list[int]]:
    """Construct monotonic per-spoken-char turn IDs combining dialogue script turns, strong punctuation, and pauses.

    Also returns per-cue resolved start turn ID for non-split branch.
    """
    window_turns: list[int] = []
    window_hints: list[str] = []
    per_cue_turn: list[int] = []

    composite_turn = 0
    prev_script_turn: int | None = None
    prev_strong_end = False

    for c_idx, c in enumerate(win_cues):
        c_spoken_len = len(extract_canonical_spoken_sequence(c.raw_text))
        c_turns = cue_turn_map.get(id(c), [])
        c_hints = cue_hint_map.get(id(c), [])

        if c_idx > 0:
            prev_c = win_cues[c_idx - 1]
            raw_cue_gap = c.start_ms - prev_c.end_ms
            if prev_strong_end:
                composite_turn += 1
                prev_strong_end = False
            elif not script_matched and raw_cue_gap >= 300:
                # Without a dialogue script, a clear >=300ms pause between distinct SRT cues signals a probable beat/speaker cut
                composite_turn += 1

        cue_first_turn = composite_turn
        # Build char-level turn IDs aligned with spoken characters of `c`
        for k in range(max(0, c_spoken_len)):
            s_turn = c_turns[k] if k < len(c_turns) else (prev_script_turn if prev_script_turn is not None else 0)
            s_hint = c_hints[k] if k < len(c_hints) else ""
            if prev_strong_end:
                composite_turn += 1
                prev_strong_end = False
            elif script_matched and prev_script_turn is not None and s_turn != prev_script_turn:
                composite_turn += 1

            if k == 0:
                cue_first_turn = composite_turn
            window_turns.append(composite_turn)
            window_hints.append(s_hint)
            prev_script_turn = s_turn
            if any(p in STRONG_PUNCT for p in s_hint):
                prev_strong_end = True

        c_stripped = c.raw_text.strip()
        if c_stripped and c_stripped[-1] in STRONG_PUNCT:
            prev_strong_end = True
        per_cue_turn.append(cue_first_turn)

    return window_turns, window_hints, per_cue_turn


def _tokens_to_char_timings(
    sub_toks: list[UnitToken],
    *,
    interpolated: bool = False,
) -> list[CharTimestamp]:
    """Expand atomic UnitTokens back to per-character CharTimestamps so subsequent relayouts align 1-to-1."""
    out: list[CharTimestamp] = []
    for t in sub_toks:
        txt = t.text
        if len(txt) <= 1:
            out.append(
                CharTimestamp(
                    char=txt,
                    start_ms=t.start_ms,
                    end_ms=t.end_ms,
                    confidence=t.confidence,
                    interpolated=interpolated,
                )
            )
        else:
            dur = max(len(txt) * 10, t.end_ms - t.start_ms)
            step = dur / len(txt)
            for k, ch in enumerate(txt):
                cs = int(round(t.start_ms + k * step))
                ce = t.end_ms if k == len(txt) - 1 else int(round(t.start_ms + (k + 1) * step))
                out.append(
                    CharTimestamp(
                        char=ch,
                        start_ms=cs,
                        end_ms=max(cs + 10, ce),
                        confidence=t.confidence,
                        interpolated=interpolated,
                    )
                )
    return out


def _join_sub_tokens_raw(sub_toks: list[UnitToken], language: str) -> str:
    if language == "en":
        return " ".join(t.raw_with_punct.strip() for t in sub_toks if t.raw_with_punct.strip()).strip()
    return "".join(t.raw_with_punct for t in sub_toks).strip()


def relayout_cues_with_template(
    cues: list[SubtitleCue],
    template: SubtitleTemplate,
    *,
    duration_ms: int,
    vad_spans: list[tuple[int, int]],
    audio_16k: np.ndarray,
    nlp_backend: str = "jieba",
    video_fps: float = 25.0,
) -> tuple[list[SubtitleCue], list[AnomalyItem], bool]:
    """Re-segment existing aligned cues dynamically when user changes slider or 1L/2L template."""
    out: list[SubtitleCue] = []
    raw_bounds: list[tuple[int, int]] = []
    pad_mask: list[bool] = []
    current_relayout_turn = -1
    prev_cue_turn: int | None = None
    prev_ended_strong = True

    def _next_turn_flag(cue_turn: int, ended_strong: bool) -> int:
        nonlocal current_relayout_turn, prev_cue_turn, prev_ended_strong
        if prev_cue_turn is None or cue_turn != prev_cue_turn or prev_ended_strong:
            current_relayout_turn += 1
        prev_cue_turn = cue_turn
        prev_ended_strong = ended_strong
        return current_relayout_turn

    for cue in cues:
        if cue.char_timings and template.smart_semantic_split and template.auto_split_long:
            tokens = build_unit_tokens(
                cue.raw_text,
                cue.char_timings,
                fallback_start_ms=cue.start_ms,
                fallback_end_ms=cue.end_ms,
                nlp_backend=nlp_backend,  # type: ignore[arg-type]
                language=template.language,
            )
            if tokens:
                segs = segment_tokens_dp(tokens, template)
                prev_local_turn: int | None = None
                for seg_idx, (s_i, e_i, disp, _) in enumerate(segs):
                    sub_toks = tokens[s_i:e_i]
                    local_turn = sub_toks[0].turn_id
                    raw_sub = _join_sub_tokens_raw(sub_toks, template.language)
                    seg_strong_end = bool(raw_sub) and raw_sub[-1] in STRONG_PUNCT
                    if seg_idx == 0:
                        assigned_turn = _next_turn_flag(cue.turn_id, seg_strong_end)
                    else:
                        if local_turn != prev_local_turn or prev_ended_strong:
                            current_relayout_turn += 1
                        prev_cue_turn = cue.turn_id
                        prev_ended_strong = seg_strong_end
                        assigned_turn = current_relayout_turn
                    prev_local_turn = local_turn

                    avg_conf = sum(t.confidence for t in sub_toks) / max(1, len(sub_toks))
                    sub_chars = _tokens_to_char_timings(sub_toks)
                    s_ms = sub_toks[0].start_ms
                    e_ms = max(s_ms + 40, sub_toks[-1].end_ms)
                    raw_bounds.append((s_ms, e_ms))
                    pad_mask.append(True)

                    out.append(
                        SubtitleCue(
                            index=len(out) + 1,
                            start_ms=s_ms,
                            end_ms=max(s_ms + 80, e_ms),
                            raw_text=raw_sub,
                            display_text=disp,
                            orig_start_ms=cue.orig_start_ms,
                            orig_end_ms=cue.orig_end_ms,
                            confidence=avg_conf,
                            char_timings=sub_chars,
                            source_cue_indices=list(cue.source_cue_indices),
                            turn_id=assigned_turn,
                        )
                    )
                continue

        # Fallback: format single cue without splitting
        disp, _ = format_single_cue_without_split(cue.raw_text, template, nlp_backend=nlp_backend)  # type: ignore[arg-type]
        c_copy = deepcopy(cue)
        c_copy.display_text = disp
        c_copy.index = len(out) + 1
        cue_strong_end = bool(cue.raw_text.strip()) and cue.raw_text.strip()[-1] in STRONG_PUNCT
        c_copy.turn_id = _next_turn_flag(cue.turn_id, cue_strong_end)
        if c_copy.char_timings:
            rs = max(0, int(c_copy.char_timings[0].start_ms))
            re = max(rs + 40, int(c_copy.char_timings[-1].end_ms))
            raw_bounds.append((rs, re))
            pad_mask.append(True)
        else:
            raw_bounds.append((c_copy.start_ms, c_copy.end_ms))
            pad_mask.append(False)
        out.append(c_copy)

    final_cues = _apply_non_invading_padding(
        out,
        raw_bounds,
        template,
        duration_ms,
        video_fps=video_fps,
        pad_mask=pad_mask,
    )
    ok, _ = verify_text_integrity(cues, final_cues)
    anomalies = run_quality_checks(final_cues, template, vad_spans, audio_16k, duration_ms)
    return final_cues, anomalies, ok


def _trim_char_trailing_silence(
    char_timings: list[CharTimestamp],
    vad_spans: list[tuple[int, int]],
) -> None:
    """Prevent CIF/CTC boundary overshoot from stretching a final character across a long pause."""
    if not char_timings or not vad_spans:
        return
    for ct in char_timings:
        for idx, (vs, ve) in enumerate(vad_spans):
            if vs <= ct.start_ms <= ve and ct.end_ms > ve + 140:
                next_vs = vad_spans[idx + 1][0] if idx + 1 < len(vad_spans) else 10**9
                # Only clamp if there is a true silence gap after `ve` AND ct does NOT extend into next VAD span
                if next_vs > ve + 180 and ct.end_ms <= next_vs + 20:
                    ct.end_ms = max(ct.start_ms + 60, ve + 60)
                    break


def _refine_char_boundaries_with_acoustics(
    char_timings: list[CharTimestamp],
    audio_16k: np.ndarray,
    duration_ms: int,
) -> None:
    """Refine forced-aligner timestamps at 10ms centered RMS resolution to fix 80ms grid lag & early tail cutoffs."""
    if not char_timings or len(audio_16k) < 640:
        return

    hop_samples = 160  # 10ms step
    half_win = 160     # 20ms centered analysis window [t - 10ms, t + 10ms]
    n_frames = max(1, len(audio_16k) // hop_samples)
    # Only compute RMS over active window slice for speed
    w_s = max(0, char_timings[0].start_ms - 360)
    w_e = min(duration_ms if duration_ms > 0 else len(audio_16k) // 16, char_timings[-1].end_ms + 360)
    f_start = max(0, w_s // 10)
    f_end = min(n_frames, max(f_start + 5, w_e // 10 + 1))
    if f_end - f_start < 6:
        return

    rms_db = np.full(f_end - f_start, -90.0, dtype=np.float64)
    for fi in range(f_start, f_end):
        c_smp = fi * hop_samples
        seg = audio_16k[max(0, c_smp - half_win) : min(len(audio_16k), c_smp + half_win)]
        if len(seg) > 0:
            val = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
            rms_db[fi - f_start] = 20.0 * np.log10(max(1e-6, val))

    noise_db = float(np.percentile(rms_db, 12))
    speech_db = float(np.percentile(rms_db, 85))
    # Skip acoustic snapping on heavy-BGM / low-SNR tracks to avoid cutting quiet syllables
    if speech_db - noise_db < 10.0:
        return
    speech_thresh = max(noise_db + 7.5, speech_db - 26.0)
    tail_thresh = max(speech_thresh, speech_db - 22.0)

    def _db_at(ms: int) -> float:
        idx = int(round(ms / 10.0)) - f_start
        if idx < 0:
            return float(rms_db[0])
        if idx >= len(rms_db):
            return float(rms_db[-1])
        return float(rms_db[idx])

    for i, ct in enumerate(char_timings):
        prev_end = char_timings[i - 1].end_ms if i > 0 else 0
        next_start = char_timings[i + 1].start_ms if i + 1 < len(char_timings) else duration_ms

        # 1. Advance ct.start_ms if aligner placed start in leading silence (including 80-160ms first chars)
        has_gap_before = (i == 0) or (ct.start_ms - prev_end >= 80)
        mid_ct = (ct.start_ms + ct.end_ms) // 2
        silent_start = (
            _db_at(ct.start_ms) < speech_thresh - 1.5
            or (
                has_gap_before
                and ct.end_ms - ct.start_ms >= 60
                and _db_at(ct.start_ms) < speech_thresh
                and _db_at(ct.start_ms + 10) < speech_thresh - 1.5
            )
        )
        trimmed_lead = False
        if (
            ((has_gap_before or ct.end_ms - ct.start_ms >= 180) and silent_start)
            or (len(char_timings) <= 3 and _db_at(mid_ct) < speech_thresh - 2.5)
        ):
            extra_fwd = 280 if len(char_timings) <= 3 else (80 if has_gap_before else -50)
            scan_end = min(next_start - 20, ct.end_ms + extra_fwd)
            scan = ct.start_ms
            found_onset = None
            while scan <= scan_end:
                if _db_at(scan) >= speech_thresh and (
                    scan + 10 > scan_end or _db_at(scan + 10) >= speech_thresh - 2.0
                ):
                    found_onset = scan
                    break
                scan += 10
            if found_onset is not None and found_onset - ct.start_ms >= 20:
                ct.start_ms = max(ct.start_ms, found_onset - 10)
                trimmed_lead = True
                if ct.end_ms <= ct.start_ms + 40:
                    ct.end_ms = min(next_start, ct.start_ms + 100)

        # 2. Snap quantized onset back by up to 130ms if clear speech already started right before ct.start_ms
        has_gap_for_back = (i == 0) or (ct.start_ms - prev_end >= 120)
        if not trimmed_lead and has_gap_for_back and _db_at(ct.start_ms) >= speech_thresh - 4.0:
            floor_ms = prev_end + 40 if i > 0 else 0
            limit_back = max(floor_ms, ct.start_ms - 130)
            scan = ct.start_ms - 10
            earliest = None
            quiet_run = 0
            while scan >= limit_back:
                if _db_at(scan) >= speech_thresh:
                    earliest = scan
                    quiet_run = 0
                else:
                    quiet_run += 1
                    if quiet_run >= 2:
                        break
                scan -= 10
            if earliest is not None and ct.start_ms - earliest >= 20:
                ct.start_ms = max(floor_ms, earliest - 10)

        # 3. Trim trailing silence overshoot at utterance end or pre-pause char
        has_gap_after = (i == len(char_timings) - 1) or (next_start - ct.end_ms >= 120)
        trimmed_tail = False
        if (has_gap_after or ct.end_ms - ct.start_ms >= 240) and _db_at(ct.end_ms) < speech_thresh - 2.0:
            scan = ct.end_ms
            found_offset = None
            min_end = ct.start_ms + 50
            while scan >= min_end:
                if _db_at(scan) >= speech_thresh:
                    found_offset = scan
                    break
                scan -= 10
            if found_offset is not None and ct.end_ms - found_offset >= 40:
                ct.end_ms = min(ct.end_ms, max(min_end, found_offset + 20))
                trimmed_tail = True

        # 4. Extend utterance-final / pre-pause boundary when aligner cuts off active vocal tail early
        is_tail_edge = (i == len(char_timings) - 1) or (next_start - ct.end_ms >= 260)
        if not trimmed_tail and is_tail_edge and _db_at(ct.end_ms) >= tail_thresh - 1.5:
            max_fwd = min(next_start - 40 if i + 1 < len(char_timings) else duration_ms, ct.end_ms + 320)
            scan = ct.end_ms + 10
            last_active = ct.end_ms
            quiet_run = 0
            while scan <= max_fwd:
                if _db_at(scan) >= tail_thresh:
                    last_active = scan
                    quiet_run = 0
                else:
                    quiet_run += 1
                    if quiet_run >= 2:
                        break
                scan += 10
            if last_active >= ct.end_ms + 20:
                ct.end_ms = min(max_fwd, last_active + 15)


def run_forced_alignment_only(
    raw_cues: list[SubtitleCue],
    audio_16k: np.ndarray,
    vad_spans: list[tuple[int, int]],
    duration_ms: int,
    alignment_cfg: AlignmentConfig,
    cache_dir: Path,
    *,
    script_path: str | Path | None = None,
    language: str = "auto",
    apply_script_turns: bool = True,
    skip_global_offset: bool = False,
    unload_after: bool = True,
    progress_cb: Callable[[float, str], None] | None = None,
) -> ForcedAlignmentStageResult:
    """Run Stage 3A: forced alignment & 10ms acoustic boundary refinement (independent of subtitle layout template)."""
    eff_lang = (
        language
        if language in {"zh", "en"}
        else detect_text_language(" ".join(c.raw_text for c in raw_cues))
    )

    if not raw_cues:
        return ForcedAlignmentStageResult(
            aligned_windows=[],
            global_offset_ms=0,
            aligner_used="none",
            effective_language=eff_lang,
            extra_phrases=[],
            pre_warnings=[],
            script_mismatch_ratio=None,
        )

    script_doc: ScriptDocument | None = None
    cue_turn_map: dict[int, list[int]] = {}
    cue_hint_map: dict[int, list[str]] = {}
    script_matched = False
    script_mismatch_ratio: float | None = None
    if apply_script_turns and script_path and Path(script_path).is_file():
        try:
            script_doc = parse_dialogue_script(script_path)
            if script_doc.characters:
                register_custom_words(script_doc.characters)
            script_align: ScriptAlignmentResult = align_script_turns_to_cues(
                [c.raw_text for c in raw_cues],
                script_doc,
            )
            script_matched = script_align.matched
            if script_matched:
                for idx_c, raw_c in enumerate(raw_cues):
                    cue_turn_map[id(raw_c)] = script_align.char_turn_ids_per_cue[idx_c]
                    cue_hint_map[id(raw_c)] = script_align.char_punct_after_per_cue[idx_c]
            else:
                script_mismatch_ratio = script_align.matched_ratio
        except Exception:
            script_doc = None

    funasr_be = FunASRAlignerBackend(cache_dir)
    qwen_be = Qwen3AlignerBackend(cache_dir)

    chosen_be = None
    pre_warnings: list[AnomalyItem] = []
    if eff_lang == "en":
        if alignment_cfg.backend == "funasr":
            pre_warnings.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.LOW_CONFIDENCE,
                    severity=AnomalySeverity.INFO,
                    start_ms=0,
                    end_ms=duration_ms,
                    cue_index=None,
                    message="英文项目不可使用中文专用 FunASR fa-zh 对齐模型，已自动切换至 Qwen3-ForcedAligner / VAD。",
                    suggestion="如需英文词级高精度对齐，请下载 Qwen3-ForcedAligner-0.6B。",
                )
            )
        if alignment_cfg.backend != "vad_fallback" and qwen_be.is_available():
            chosen_be = qwen_be
    else:
        if alignment_cfg.backend == "qwen3" and qwen_be.is_available():
            chosen_be = qwen_be
        elif alignment_cfg.backend == "funasr" and funasr_be.is_available():
            chosen_be = funasr_be
        elif alignment_cfg.backend == "auto":
            if funasr_be.is_available() and qwen_be.is_available():
                import torch
                free_mb = 0
                if torch.cuda.is_available() and alignment_cfg.device != "cpu":
                    try:
                        free_mb = torch.cuda.mem_get_info(0)[0] / (1024 * 1024)
                    except Exception:
                        free_mb = 0
                chosen_be = qwen_be if free_mb >= 2300 else funasr_be
            elif funasr_be.is_available():
                chosen_be = funasr_be
            elif qwen_be.is_available():
                chosen_be = qwen_be

    if chosen_be is None and alignment_cfg.backend != "vad_fallback" and not alignment_cfg.allow_vad_fallback:
        if eff_lang == "en":
            raise MissingAlignerModelError(
                "qwen3_aligner",
                "未安装英文可用强制对齐模型 Qwen3-ForcedAligner-0.6B（英文不可使用中文专用 FunASR fa-zh），"
                "为避免生成估算时间码已中止对齐。请在「模型下载与缓存」中下载 Qwen3-ForcedAligner-0.6B 后重试。",
            )
        rec_model = "qwen3_aligner" if alignment_cfg.backend == "qwen3" else "funasr_fa"
        raise MissingAlignerModelError(
            rec_model,
            "未检测到本地声学强制对齐模型（Qwen3-ForcedAligner-0.6B / FunASR fa-zh），"
            "为避免生成按字数比例估算的伪时间码已中止对齐。请在「模型下载与缓存」中下载对齐模型后重试。",
        )

    if chosen_be is None:
        pre_warnings.append(
            AnomalyItem(
                anomaly_type=AnomalyType.LOW_CONFIDENCE,
                severity=AnomalySeverity.WARNING,
                start_ms=0,
                end_ms=duration_ms,
                cue_index=None,
                message="当前未启用声学强制对齐模型，字幕时间码为 VAD 语音区间按字数比例估算结果（非逐字声学对齐）。",
                suggestion="建议在「模型下载与缓存」中下载 Qwen3-ForcedAligner-0.6B 或 FunASR fa-zh 后重新执行对齐。",
            )
        )

    aligner_label = chosen_be.name if chosen_be else (
        "VAD 回退对齐 (估算时间轴 · 建议下载 Qwen3-ForcedAligner-0.6B)"
        if eff_lang == "en"
        else "VAD 回退对齐 (估算时间轴 · 建议下载 fa-zh 或 Qwen3 模型)"
    )

    global_offset = 0
    anchored_bounds: dict[int, tuple[int, int]] = {}
    if alignment_cfg.detect_global_offset and not skip_global_offset:
        if progress_cb:
            progress_cb(0.30, "检测字幕时间轴与视频语音偏移...")
        if eff_lang != "en" and funasr_be.has_asr_anchor():
            try:
                asr_anchors = funasr_be.transcribe_anchors(audio_16k, device=alignment_cfg.device)
                anchored_bounds = _anchor_cues_with_asr(raw_cues, asr_anchors)
            except Exception:
                anchored_bounds = {}
            finally:
                MODEL_POOL.unload_role("asr")
        global_offset = _estimate_global_vad_offset(raw_cues, vad_spans, duration_ms)

    windows = _group_cues_into_windows(
        raw_cues,
        vad_spans,
        anchored_bounds,
        global_offset,
        duration_ms,
        max_window_ms=24000 if chosen_be is qwen_be else 22000,
    )

    aligned_windows: list[AlignedUtteranceData] = []
    used_backend_runtime = aligner_label
    extra_phrases = list(script_doc.characters) if script_doc is not None else []

    try:
        # Run batched alignment upfront when backend supports GPU batching (e.g. Qwen3-ForcedAligner)
        batched_outputs: list | None = None
        batch_fn = getattr(chosen_be, "align_utterances_batch", None) if chosen_be is not None else None
        if callable(batch_fn) and windows:
            win_spoken = [bool(extract_canonical_spoken_sequence(w.clean_text)) for w in windows]
            active_idx = [idx for idx, ok in enumerate(win_spoken) if ok]
            if active_idx:
                batch_items = [
                    (windows[idx].window_start_ms, windows[idx].window_end_ms, windows[idx].clean_text)
                    for idx in active_idx
                ]
                try:
                    batch_res = batch_fn(
                        audio_16k,
                        batch_items,
                        language=eff_lang,
                        device=alignment_cfg.device,
                        cache_dir=cache_dir,
                        progress_cb=progress_cb,
                    )
                    batched_outputs = [None] * len(windows)
                    for idx, out_item in zip(active_idx, batch_res):
                        batched_outputs[idx] = out_item
                except Exception as exc:
                    batched_outputs = None
                    used_backend_runtime = f"{aligner_label} (批处理回退单条: {exc})"

        for w_idx, win in enumerate(windows):
            if progress_cb and batched_outputs is None:
                frac = 0.35 + 0.30 * (w_idx / max(1, len(windows)))
                progress_cb(frac, f"正在语音强制对齐 ({w_idx + 1}/{len(windows)})...")

            win_char_turns, win_char_hints, per_cue_turn = _compute_window_turn_ids(
                win.cues,
                cue_turn_map,
                cue_hint_map,
                script_matched,
            )
            if not script_matched and any(c.turn_id != 0 for c in win.cues):
                win_char_turns = []
                per_cue_turn = []
                turn_remap: dict[int, int] = {}
                for c in win.cues:
                    mapped_t = turn_remap.setdefault(c.turn_id, len(turn_remap))
                    per_cue_turn.append(mapped_t)
                    c_len = len(extract_canonical_spoken_sequence(c.raw_text))
                    win_char_turns.extend([mapped_t] * max(0, c_len))

            max_local_turn = max(win_char_turns, default=0)
            spoken_clean = extract_canonical_spoken_sequence(win.clean_text)
            char_timings: list[CharTimestamp] = []

            if chosen_be is not None and spoken_clean:
                try:
                    if batched_outputs is not None and batched_outputs[w_idx] is not None:
                        out = batched_outputs[w_idx]
                    else:
                        out = chosen_be.align_utterance(
                            audio_16k,
                            win.window_start_ms,
                            win.window_end_ms,
                            win.clean_text,
                            language=eff_lang,
                            device=alignment_cfg.device,
                            cache_dir=cache_dir,
                        )
                    char_timings = out.char_timings
                    used_backend_runtime = out.backend_name

                    # Rescue isolated short utterance if ForcedAligner latched onto leading silence
                    if char_timings and vad_spans:
                        al_s, al_e = char_timings[0].start_ms, char_timings[-1].end_ms
                        ov_init = _vad_overlap_ratio(al_s, al_e, vad_spans)
                        if ov_init < 0.25 and len(win.cues) == 1:
                            cand_vad = [
                                (vs, ve)
                                for vs, ve in vad_spans
                                if ve >= win.window_start_ms - 800 and vs <= win.window_end_ms + 800 and (ve - vs) >= 80
                            ]
                            if cand_vad:
                                al_mid = (al_s + al_e) // 2
                                vs_near, ve_near = min(cand_vad, key=lambda sp: abs(((sp[0] + sp[1]) // 2) - al_mid))
                                retry_s = max(0, vs_near - 120)
                                retry_e = min(duration_ms, ve_near + 120)
                                if retry_e - retry_s >= 180 and (retry_s != win.window_start_ms or retry_e != win.window_end_ms):
                                    out_retry = chosen_be.align_utterance(
                                        audio_16k,
                                        retry_s,
                                        retry_e,
                                        win.clean_text,
                                        language=eff_lang,
                                        device=alignment_cfg.device,
                                        cache_dir=cache_dir,
                                    )
                                    if out_retry.char_timings:
                                        ov_retry = _vad_overlap_ratio(
                                            out_retry.char_timings[0].start_ms,
                                            out_retry.char_timings[-1].end_ms,
                                            vad_spans,
                                        )
                                        if ov_retry >= ov_init + 0.25:
                                            char_timings = out_retry.char_timings
                except Exception as exc:
                    char_timings = []
                    used_backend_runtime = f"{aligner_label} (异常回退: {exc})"

            if not char_timings:
                char_timings = _vad_fallback_align(audio_16k, win, vad_spans)

            _trim_char_trailing_silence(char_timings, vad_spans)
            _refine_char_boundaries_with_acoustics(char_timings, audio_16k, duration_ms)

            for ct in char_timings:
                ov = _vad_overlap_ratio(ct.start_ms, ct.end_ms, vad_spans)
                if ov == 0.0:
                    ct.confidence = min(ct.confidence, 0.42)

            aligned_windows.append(
                AlignedUtteranceData(
                    clean_text=win.clean_text,
                    window_start_ms=win.window_start_ms,
                    window_end_ms=win.window_end_ms,
                    char_timings=char_timings,
                    win_char_turns=win_char_turns,
                    win_char_hints=win_char_hints,
                    per_cue_turn=per_cue_turn,
                    cues_snapshot=deepcopy(win.cues),
                    max_local_turn=max_local_turn,
                )
            )
    finally:
        if unload_after:
            MODEL_POOL.unload_role("aligner")

    return ForcedAlignmentStageResult(
        aligned_windows=aligned_windows,
        global_offset_ms=global_offset,
        aligner_used=used_backend_runtime,
        effective_language=eff_lang,
        extra_phrases=extra_phrases,
        pre_warnings=pre_warnings,
        script_mismatch_ratio=script_mismatch_ratio,
    )


def segment_aligned_windows_to_cues(
    raw_cues: list[SubtitleCue],
    align_stage: ForcedAlignmentStageResult,
    audio_16k: np.ndarray,
    vad_spans: list[tuple[int, int]],
    duration_ms: int,
    template: SubtitleTemplate,
    alignment_cfg: AlignmentConfig,
    *,
    video_fps: float = 25.0,
) -> AlignmentEngineResult:
    """Run Stage 3B: DP semantic segmentation, padding/bridging, and QA checks from cached forced alignments."""
    if not raw_cues or not align_stage.aligned_windows:
        return AlignmentEngineResult(
            cues=[],
            anomalies=list(align_stage.pre_warnings),
            global_offset_ms=align_stage.global_offset_ms,
            aligner_used=align_stage.aligner_used,
            nlp_used=alignment_cfg.nlp_backend,
            text_integrity_ok=True,
        )

    eff_lang = template.language if template.language in {"zh", "en"} else align_stage.effective_language
    processed_cues: list[SubtitleCue] = []
    global_turn_base = 0
    extra_phrases = align_stage.extra_phrases or None
    if extra_phrases:
        register_custom_words(extra_phrases)

    for win_data in align_stage.aligned_windows:
        win_is_interpolated = bool(win_data.char_timings) and (
            sum(1 for ct in win_data.char_timings if ct.interpolated) / len(win_data.char_timings) >= 0.8
        )
        if template.smart_semantic_split and template.auto_split_long:
            tokens = build_unit_tokens(
                win_data.clean_text,
                win_data.char_timings,
                fallback_start_ms=win_data.window_start_ms,
                fallback_end_ms=win_data.window_end_ms,
                nlp_backend=alignment_cfg.nlp_backend,
                char_turn_ids=win_data.win_char_turns,
                char_punct_hints=win_data.win_char_hints,
                extra_phrases=extra_phrases,
                language=eff_lang,
            )
            segs = segment_tokens_dp(tokens, template)
            for s_i, e_i, disp_txt, _ in segs:
                sub_toks = tokens[s_i:e_i]
                raw_sub = _join_sub_tokens_raw(sub_toks, eff_lang)
                avg_conf = sum(t.confidence for t in sub_toks) / max(1, len(sub_toks))
                sub_chars = _tokens_to_char_timings(sub_toks, interpolated=win_is_interpolated)
                orig_s = win_data.cues_snapshot[0].orig_start_ms
                orig_e = win_data.cues_snapshot[-1].orig_end_ms
                processed_cues.append(
                    SubtitleCue(
                        index=len(processed_cues) + 1,
                        start_ms=sub_toks[0].start_ms,
                        end_ms=sub_toks[-1].end_ms,
                        raw_text=raw_sub,
                        display_text=disp_txt,
                        orig_start_ms=orig_s,
                        orig_end_ms=orig_e,
                        confidence=avg_conf,
                        char_timings=sub_chars,
                        source_cue_indices=[c.index for c in win_data.cues_snapshot],
                        turn_id=global_turn_base + sub_toks[0].turn_id,
                    )
                )
        else:
            cursor = 0
            for c_i, orig_cue in enumerate(win_data.cues_snapshot):
                c_spoken = extract_canonical_spoken_sequence(orig_cue.raw_text)
                cnt = len(c_spoken)
                slice_ct = win_data.char_timings[cursor : cursor + cnt]
                cursor += cnt
                if slice_ct:
                    s_ms = slice_ct[0].start_ms
                    e_ms = slice_ct[-1].end_ms
                    conf = sum(x.confidence for x in slice_ct) / len(slice_ct)
                else:
                    s_ms = max(0, orig_cue.start_ms + align_stage.global_offset_ms)
                    e_ms = min(duration_ms, max(s_ms + 200, orig_cue.end_ms + align_stage.global_offset_ms))
                    conf = 0.35

                if template.smart_semantic_split:
                    disp, _ = format_single_cue_without_split(
                        orig_cue.raw_text,
                        template,
                        nlp_backend=alignment_cfg.nlp_backend,
                    )
                else:
                    disp = (
                        strip_display_punctuation(orig_cue.raw_text)
                        if template.remove_punctuation
                        else orig_cue.raw_text
                    )
                cue_t = win_data.per_cue_turn[c_i] if c_i < len(win_data.per_cue_turn) else c_i
                processed_cues.append(
                    SubtitleCue(
                        index=len(processed_cues) + 1,
                        start_ms=s_ms,
                        end_ms=e_ms,
                        raw_text=orig_cue.raw_text,
                        display_text=disp,
                        orig_start_ms=orig_cue.orig_start_ms,
                        orig_end_ms=orig_cue.orig_end_ms,
                        confidence=conf,
                        char_timings=slice_ct,
                        source_cue_indices=[orig_cue.index],
                        turn_id=global_turn_base + cue_t,
                    )
                )
        global_turn_base += win_data.max_local_turn + 1

    final_cues = _apply_padding_and_merge(processed_cues, template, duration_ms, video_fps=video_fps)
    integrity_ok, integrity_msg = verify_text_integrity(raw_cues, final_cues)

    anomalies = list(align_stage.pre_warnings) + run_quality_checks(
        final_cues,
        template,
        vad_spans,
        audio_16k,
        duration_ms,
        low_conf_thresh=alignment_cfg.low_confidence_threshold,
    )
    if align_stage.script_mismatch_ratio is not None:
        anomalies.insert(
            0,
            AnomalyItem(
                anomaly_type=AnomalyType.TEXT_MISMATCH,
                severity=AnomalySeverity.INFO,
                start_ms=0,
                end_ms=duration_ms,
                cue_index=None,
                message=f"台词文稿与 SRT 文本匹配度偏低（{align_stage.script_mismatch_ratio:.0%}），已忽略说话轮次绑定。",
                suggestion="请确认是否为当前集对应的台词文稿。",
            ),
        )
    if not integrity_ok:
        anomalies.insert(
            0,
            AnomalyItem(
                anomaly_type=AnomalyType.TEXT_MISMATCH,
                severity=AnomalySeverity.ERROR,
                start_ms=0,
                end_ms=duration_ms,
                cue_index=None,
                message=integrity_msg,
                suggestion="已自动保留原文完整性检查告警，请核对输出文本。",
            ),
        )

    return AlignmentEngineResult(
        cues=final_cues,
        anomalies=anomalies,
        global_offset_ms=align_stage.global_offset_ms,
        aligner_used=align_stage.aligner_used,
        nlp_used=alignment_cfg.nlp_backend,
        text_integrity_ok=integrity_ok,
    )


def run_alignment_and_segmentation(
    raw_cues: list[SubtitleCue],
    audio_16k: np.ndarray,
    vad_spans: list[tuple[int, int]],
    duration_ms: int,
    template: SubtitleTemplate,
    alignment_cfg: AlignmentConfig,
    cache_dir: Path,
    *,
    video_fps: float = 25.0,
    script_path: str | Path | None = None,
    language: str = "auto",
    apply_script_turns: bool = True,
    skip_global_offset: bool = False,
    unload_after: bool = True,
    progress_cb: Callable[[float, str], None] | None = None,
) -> AlignmentEngineResult:
    """Execute the full forced alignment and semantic segmentation workflow (Chinese & English)."""
    align_stage = run_forced_alignment_only(
        raw_cues=raw_cues,
        audio_16k=audio_16k,
        vad_spans=vad_spans,
        duration_ms=duration_ms,
        alignment_cfg=alignment_cfg,
        cache_dir=cache_dir,
        script_path=script_path,
        language=language,
        apply_script_turns=apply_script_turns,
        skip_global_offset=skip_global_offset,
        unload_after=unload_after,
        progress_cb=progress_cb,
    )
    return segment_aligned_windows_to_cues(
        raw_cues=raw_cues,
        align_stage=align_stage,
        audio_16k=audio_16k,
        vad_spans=vad_spans,
        duration_ms=duration_ms,
        template=template,
        alignment_cfg=alignment_cfg,
        video_fps=video_fps,
    )
