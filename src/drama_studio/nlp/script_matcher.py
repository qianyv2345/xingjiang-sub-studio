"""Local-sequence ASR-to-script matching engine (supports deleted lines, out-of-order edits, duplicates, and EN/ZH)."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
from typing import Any

from drama_studio.align.base import ASRSegment
from drama_studio.models import AnomalyItem, AnomalySeverity, AnomalyType, SubtitleCue
from drama_studio.nlp.punctuation import detect_text_language
from drama_studio.nlp.script_io import ScriptDocument

_DIGIT_ZH = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}

_EN_CONTRACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"\bgonna\b"), "going to"),
    (re.compile(r"\bwanna\b"), "want to"),
    (re.compile(r"\bgotta\b"), "got to"),
    (re.compile(r"\b'cause\b"), "because"),
    (re.compile(r"\bdr\.\b"), "doctor"),
    (re.compile(r"\bmr\.\b"), "mister"),
    (re.compile(r"\bmrs\.\b"), "misses"),
    (re.compile(r"\bms\.\b"), "miss"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), " is"),
]

_SENT_SPLIT_ZH_EN = re.compile(r"(?<=[。？！?!])\s*|(?<=\.)\s+(?=[A-Z\"'\u4e00-\u9fff])")


@dataclass
class ScriptUnit:
    unit_idx: int
    turn_id: int
    speaker: str
    raw_text: str
    norm_str: str
    norm_tokens: list[str]
    language: str


@dataclass
class ScriptMatchResult:
    cues: list[SubtitleCue]
    anomalies: list[AnomalyItem]
    overall_confidence: float
    matched_count: int
    total_asr_segments: int
    matched_script_unit_indices: list[int]
    skipped_script_turns: list[dict[str, Any]]
    extra_audio_segments: list[dict[str, Any]]
    duplicate_lines: list[dict[str, Any]]
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "overall_confidence": round(self.overall_confidence, 4),
            "matched_segments": self.matched_count,
            "total_asr_segments": self.total_asr_segments,
            "matched_script_units": len(set(self.matched_script_unit_indices)),
            "skipped_lines_count": len(self.skipped_script_turns),
            "extra_audio_count": len(self.extra_audio_segments),
            "duplicate_count": len(self.duplicate_lines),
            "skipped_lines": self.skipped_script_turns[:50],
            "extra_audio": self.extra_audio_segments[:50],
            "duplicate_lines": self.duplicate_lines[:50],
        }


def _normalize_zh_digits(text: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        tok = m.group(0)
        if "." in tok:
            parts = tok.split(".", 1)
            return "".join(_DIGIT_ZH.get(c, c) for c in parts[0]) + "点" + "".join(_DIGIT_ZH.get(c, c) for c in parts[1])
        return "".join(_DIGIT_ZH.get(c, c) for c in tok)

    text = re.sub(r"\d+(?:\.\d+)?", _repl, text)
    return text.replace("〇", "零")


def normalize_for_matching(text: str, language: str = "auto") -> tuple[str, list[str]]:
    """Normalize text for resilient ASR <-> script similarity scoring."""
    lang = detect_text_language(text) if language == "auto" else language
    low = text.lower().strip()
    low = low.replace("’", "'").replace("‘", "'")
    if lang == "en":
        for pat, rep in _EN_CONTRACTIONS:
            low = pat.sub(rep, low)
        low = low.replace("u.s.", "us").replace("u.k.", "uk")
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", low)
        return " ".join(tokens), tokens

    zh_norm = _normalize_zh_digits(low)
    chars = [c for c in zh_norm if c.isalnum() or ("\u4e00" <= c <= "\u9fff")]
    # Tokens mix CJK single chars + ASCII words
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", zh_norm)
    return "".join(chars), tokens


def compute_text_similarity(
    asr_norm: str,
    asr_tokens: list[str],
    scr_norm: str,
    scr_tokens: list[str],
    language: str,
) -> float:
    """Compute bounded [0, 1] similarity combining sequence order, token overlap, and length plausibility."""
    if not asr_norm or not scr_norm:
        return 0.0

    len_a = len(asr_norm)
    len_b = len(scr_norm)
    len_ratio = min(len_a, len_b) / max(1, max(len_a, len_b))
    if len_ratio < 0.35:
        return 0.0

    if language == "en":
        word_ratio = SequenceMatcher(None, asr_tokens, scr_tokens, autojunk=False).ratio()
        char_ratio = SequenceMatcher(None, asr_norm, scr_norm, autojunk=False).ratio()
        set_a = set(asr_tokens)
        set_b = set(scr_tokens)
        jaccard = len(set_a & set_b) / max(1, len(set_a | set_b))
        sim = 0.55 * word_ratio + 0.25 * char_ratio + 0.20 * jaccard
    else:
        seq_ratio = SequenceMatcher(None, asr_norm, scr_norm, autojunk=False).ratio()
        set_a = set(asr_tokens)
        set_b = set(scr_tokens)
        tok_f1 = (2.0 * len(set_a & set_b)) / max(1, len(set_a) + len(set_b))
        # Bigram overlap catches slight insertions/homophones cleanly
        bg_a = {asr_norm[k : k + 2] for k in range(max(0, len_a - 1))} if len_a >= 2 else {asr_norm}
        bg_b = {scr_norm[k : k + 2] for k in range(max(0, len_b - 1))} if len_b >= 2 else {scr_norm}
        bg_sim = len(bg_a & bg_b) / max(1, len(bg_a | bg_b))
        sim = 0.58 * seq_ratio + 0.22 * tok_f1 + 0.20 * bg_sim

    return float(min(1.0, sim * (0.65 + 0.35 * len_ratio)))


def _build_script_units(script_doc: ScriptDocument) -> list[ScriptUnit]:
    units: list[ScriptUnit] = []
    for turn in script_doc.turns:
        # Protect title dots like Dr. / Mr. before splitting multi-sentence turns
        raw = turn.raw_line.strip()
        prot = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc)\.\s+", r"\1<DOT> ", raw)
        parts = [p.replace("<DOT>", ".").strip() for p in _SENT_SPLIT_ZH_EN.split(prot) if p.strip()]
        if not parts:
            parts = [raw]
        for p in parts:
            norm_s, norm_toks = normalize_for_matching(p, turn.language)
            if not norm_s:
                continue
            units.append(
                ScriptUnit(
                    unit_idx=len(units),
                    turn_id=turn.turn_id,
                    speaker=turn.speaker,
                    raw_text=p,
                    norm_str=norm_s,
                    norm_tokens=norm_toks,
                    language=turn.language,
                )
            )
    return units


def _all_span_units_covered(
    asr_norm: str,
    asr_tokens: list[str],
    span_units: list[ScriptUnit],
    language: str,
    min_unit_cov: float = 0.45,
) -> bool:
    """Ensure every unit inside a multi-unit span has genuine evidence in ASR (prevents swallowing deleted lines)."""
    if len(span_units) <= 1:
        return True
    if language == "en":
        comb_toks: list[str] = []
        unit_ranges: list[tuple[int, int]] = []
        for u in span_units:
            s = len(comb_toks)
            comb_toks.extend(u.norm_tokens)
            unit_ranges.append((s, len(comb_toks)))
        sm = SequenceMatcher(None, asr_tokens, comb_toks, autojunk=False)
        matched_pos: set[int] = set()
        for b in sm.get_matching_blocks():
            for pos in range(b.b, b.b + b.size):
                matched_pos.add(pos)
        for s, e in unit_ranges:
            u_len = max(1, e - s)
            hit = sum(1 for p in range(s, e) if p in matched_pos)
            if (hit / u_len) < min_unit_cov:
                return False
        return True

    comb_chars: list[str] = []
    unit_ranges_c: list[tuple[int, int]] = []
    for u in span_units:
        s = len(comb_chars)
        comb_chars.extend(list(u.norm_str))
        unit_ranges_c.append((s, len(comb_chars)))
    sm = SequenceMatcher(None, list(asr_norm), comb_chars, autojunk=False)
    matched_pos_c: set[int] = set()
    for b in sm.get_matching_blocks():
        for pos in range(b.b, b.b + b.size):
            matched_pos_c.add(pos)
    for s, e in unit_ranges_c:
        u_len = max(1, e - s)
        hit = sum(1 for p in range(s, e) if p in matched_pos_c)
        if (hit / u_len) < min_unit_cov:
            return False
    return True


def _all_asr_segments_covered(
    asr_slices: list[tuple[str, list[str]]],
    comb_u_norm: str,
    comb_u_tokens: list[str],
    language: str,
    min_seg_cov: float = 0.45,
) -> bool:
    """Ensure every ASR segment inside a multi-ASR merge contributes genuine aligned text."""
    if len(asr_slices) <= 1:
        return True
    if language == "en":
        comb_toks: list[str] = []
        seg_ranges: list[tuple[int, int]] = []
        for _, toks in asr_slices:
            s = len(comb_toks)
            comb_toks.extend(toks)
            seg_ranges.append((s, len(comb_toks)))
        sm = SequenceMatcher(None, comb_toks, comb_u_tokens, autojunk=False)
        matched_pos: set[int] = set()
        for b in sm.get_matching_blocks():
            for pos in range(b.a, b.a + b.size):
                matched_pos.add(pos)
        for s, e in seg_ranges:
            seg_len = max(1, e - s)
            hit = sum(1 for p in range(s, e) if p in matched_pos)
            if hit == 0 or (hit / seg_len) < min_seg_cov:
                return False
        return True

    comb_chars: list[str] = []
    seg_ranges_c: list[tuple[int, int]] = []
    for norm_s, _ in asr_slices:
        s = len(comb_chars)
        comb_chars.extend(list(norm_s))
        seg_ranges_c.append((s, len(comb_chars)))
    sm = SequenceMatcher(None, comb_chars, list(comb_u_norm), autojunk=False)
    matched_pos_c: set[int] = set()
    for b in sm.get_matching_blocks():
        for pos in range(b.a, b.a + b.size):
            matched_pos_c.add(pos)
    for s, e in seg_ranges_c:
        seg_len = max(1, e - s)
        hit = sum(1 for p in range(s, e) if p in matched_pos_c)
        if hit == 0 or (hit / seg_len) < min_seg_cov:
            return False
    return True


@dataclass
class _CandidateSpan:
    u_start: int
    u_end: int  # exclusive
    sim: float


@dataclass
class _MatchOp:
    asr_start: int
    asr_end: int  # exclusive
    cand: _CandidateSpan | None


def match_asr_to_script(
    asr_segments: list[ASRSegment],
    script_doc: ScriptDocument,
    *,
    min_match_confidence: float = 0.54,
) -> ScriptMatchResult:
    """Match VAD-windowed ASR segments onto script lines using local anchored sequence DP.

    Guarantees:
    - Unmatched script lines (deleted from rough cut) are NEVER invented in output cues.
    - Handles partial rough-cut clips, deleted lines, duplicate lines, out-of-order scenes, and minor ASR errors.
    - Establishes explicit anomalies for extra spoken audio, missing script lines, and duplicates.
    """
    units = _build_script_units(script_doc)
    if not asr_segments or not units:
        anomalies: list[AnomalyItem] = []
        skipped = [
            {"turn_id": t.turn_id, "speaker": t.speaker, "text": t.raw_line}
            for t in script_doc.turns
        ]
        if not asr_segments and units:
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.SCRIPT_MISSING_LINE,
                    severity=AnomalySeverity.WARNING,
                    start_ms=0,
                    end_ms=1000,
                    cue_index=None,
                    message=f"音频中未识别到有效语音，剧本共 {len(script_doc.turns)} 句台词均未匹配。",
                    suggestion="请确认音轨选择是否正确或音频是否静音。",
                )
            )
        return ScriptMatchResult(
            cues=[],
            anomalies=anomalies,
            overall_confidence=0.0,
            matched_count=0,
            total_asr_segments=len(asr_segments),
            matched_script_unit_indices=[],
            skipped_script_turns=skipped,
            extra_audio_segments=[],
            duplicate_lines=[],
        )

    n_asr = len(asr_segments)
    n_units = len(units)
    asr_norms = [normalize_for_matching(seg.text, seg.language) for seg in asr_segments]

    # Step 1: Retrieve top candidate script spans [u_s : u_e] for each ASR segment
    candidates_per_asr: list[list[_CandidateSpan]] = []
    for i in range(n_asr):
        a_norm, a_toks = asr_norms[i]
        lang_i = asr_segments[i].language
        cands: list[_CandidateSpan] = []
        if not a_norm:
            candidates_per_asr.append(cands)
            continue

        target_len = len(a_norm)
        for u_s in range(n_units):
            comb_norm_parts: list[str] = []
            comb_toks: list[str] = []
            for span_len in range(1, 9):
                u_e = u_s + span_len
                if u_e > n_units:
                    break
                unit = units[u_e - 1]
                comb_norm_parts.append(unit.norm_str)
                comb_toks.extend(unit.norm_tokens)
                comb_norm = (" " if lang_i == "en" else "").join(comb_norm_parts)
                # Fast length prune
                if len(comb_norm) > target_len * 2.5 + 8 and span_len > 1:
                    break
                if len(comb_norm) < target_len * 0.35:
                    continue
                sim = compute_text_similarity(a_norm, a_toks, comb_norm, comb_toks, lang_i)
                if sim >= 0.42:
                    if span_len > 1 and not _all_span_units_covered(a_norm, a_toks, units[u_s:u_e], lang_i):
                        continue
                    cands.append(_CandidateSpan(u_start=u_s, u_end=u_e, sim=sim))

        cands.sort(key=lambda c: c.sim, reverse=True)
        candidates_per_asr.append(cands[:12])

    # Step 1.5: Precompute multi-ASR merged candidates (2..4 consecutive ASR chunks -> 1..2 ScriptUnits)
    # Handles a single long script sentence split into 2 ASR chunks by a short breath pause.
    multi_cands_by_asr: list[list[tuple[int, _CandidateSpan]]] = [[] for _ in range(n_asr)]
    for i in range(n_asr):
        lang_i = asr_segments[i].language
        if not asr_norms[i][0]:
            continue
        for asr_k in range(2, 5):
            asr_end = i + asr_k
            if asr_end > n_asr:
                break
            prev_seg = asr_segments[asr_end - 2]
            curr_seg = asr_segments[asr_end - 1]
            if (curr_seg.start_ms - prev_seg.end_ms) > 2200:
                break
            if (curr_seg.end_ms - asr_segments[i].start_ms) > 16000:
                break
            if curr_seg.language != lang_i or not asr_norms[asr_end - 1][0]:
                break

            comb_a_norm = (" " if lang_i == "en" else "").join(
                asr_norms[m][0] for m in range(i, asr_end)
            )
            comb_a_toks = [t for m in range(i, asr_end) for t in asr_norms[m][1]]
            target_comb_len = len(comb_a_norm)
            if target_comb_len < 3:
                continue

            slice_norms = [asr_norms[m] for m in range(i, asr_end)]
            any_strong_single = any(
                bool(candidates_per_asr[m] and candidates_per_asr[m][0].sim >= 0.72)
                for m in range(i, asr_end)
            )

            merged_hits: list[tuple[int, _CandidateSpan]] = []
            for u_s in range(n_units):
                for u_span in (1, 2):
                    u_e = u_s + u_span
                    if u_e > n_units:
                        break
                    if u_span == 2 and any_strong_single:
                        continue
                    span_units = units[u_s:u_e]
                    comb_u_norm = (" " if lang_i == "en" else "").join(u.norm_str for u in span_units)
                    comb_u_toks = [t for u in span_units for t in u.norm_tokens]
                    if len(comb_u_norm) < target_comb_len * 0.50:
                        continue
                    if len(comb_u_norm) > target_comb_len * 1.85 + 4:
                        continue
                    comb_sim = compute_text_similarity(
                        comb_a_norm, comb_a_toks, comb_u_norm, comb_u_toks, lang_i
                    )
                    min_comb = 0.72 if u_span == 2 else 0.62
                    if comb_sim < min_comb:
                        continue
                    if u_span > 1 and not _all_span_units_covered(
                        comb_a_norm, comb_a_toks, span_units, lang_i, min_unit_cov=0.50
                    ):
                        continue

                    max_sub_sim = max(
                        compute_text_similarity(
                            asr_norms[m][0], asr_norms[m][1], comb_u_norm, comb_u_toks, lang_i
                        )
                        for m in range(i, asr_end)
                    )
                    if u_span > 1:
                        for u_one in span_units:
                            for m in range(i, asr_end):
                                s_one = compute_text_similarity(
                                    asr_norms[m][0],
                                    asr_norms[m][1],
                                    u_one.norm_str,
                                    u_one.norm_tokens,
                                    lang_i,
                                )
                                if s_one > max_sub_sim:
                                    max_sub_sim = s_one
                    if comb_sim < max_sub_sim + 0.08:
                        continue
                    if not _all_asr_segments_covered(
                        slice_norms, comb_u_norm, comb_u_toks, lang_i, min_seg_cov=0.45
                    ):
                        continue
                    merged_hits.append((asr_end, _CandidateSpan(u_start=u_s, u_end=u_e, sim=comb_sim)))

            if merged_hits:
                merged_hits.sort(key=lambda item: item[1].sim, reverse=True)
                multi_cands_by_asr[i].extend(merged_hits[:6])
        if multi_cands_by_asr[i]:
            multi_cands_by_asr[i].sort(key=lambda item: item[1].sim, reverse=True)
            multi_cands_by_asr[i] = multi_cands_by_asr[i][:8]

    def _transition_bonus(last_end: int | None, u_start: int) -> float:
        if last_end is None:
            return 0.0
        delta = u_start - last_end
        if delta == 0:
            return 0.26
        if 1 <= delta <= 4:
            return 0.16 - 0.025 * delta
        if 4 < delta <= 20:
            return 0.04
        return -0.09

    # Step 2: Beam-search / Viterbi sequence disambiguation over ASR positions (supports 1-to-N and N-to-1)
    beam_width = 16
    beams_by_pos: list[list[tuple[float, int | None, frozenset[int], list[_MatchOp]]]] = [
        [] for _ in range(n_asr + 1)
    ]
    beams_by_pos[0] = [(0.0, None, frozenset(), [])]

    def _prune_beams(
        raw_beams: list[tuple[float, int | None, frozenset[int], list[_MatchOp]]],
    ) -> list[tuple[float, int | None, frozenset[int], list[_MatchOp]]]:
        raw_beams.sort(key=lambda b: b[0], reverse=True)
        pruned: list[tuple[float, int | None, frozenset[int], list[_MatchOp]]] = []
        seen_end: dict[int | None, int] = {}
        for b in raw_beams:
            cnt = seen_end.get(b[1], 0)
            if cnt < 2:
                pruned.append(b)
                seen_end[b[1]] = cnt + 1
            if len(pruned) >= beam_width:
                break
        return pruned

    for i in range(n_asr):
        cur_beams = _prune_beams(beams_by_pos[i])
        a_norm, a_toks = asr_norms[i]
        is_ultra_short = len(a_norm) <= 2 or (asr_segments[i].language == "en" and len(a_toks) <= 1)
        eff_thresh = max(min_match_confidence, 0.72) if is_ultra_short else min_match_confidence

        for score, last_end, used_set, ops in cur_beams:
            # Option A: Leave segment unmatched (baseline score = eff_thresh)
            beams_by_pos[i + 1].append(
                (score + eff_thresh, last_end, used_set, ops + [_MatchOp(i, i + 1, None)])
            )

            # Option B: Match one single-ASR candidate script span
            for cand in candidates_per_asr[i]:
                is_contiguous = last_end is not None and cand.u_start == last_end
                if is_ultra_short and cand.sim < 0.75 and not is_contiguous:
                    continue

                trans_bonus = _transition_bonus(last_end, cand.u_start)
                span_ids = set(range(cand.u_start, cand.u_end))
                overlap_cnt = len(span_ids & used_set)
                dup_penalty = 0.32 * overlap_cnt

                step_score = cand.sim + trans_bonus - dup_penalty
                if cand.sim + max(0.0, trans_bonus) < eff_thresh or cand.sim < 0.45:
                    continue
                if step_score <= eff_thresh - 0.02:
                    continue

                beams_by_pos[i + 1].append(
                    (
                        score + step_score,
                        cand.u_end,
                        used_set | frozenset(span_ids),
                        ops + [_MatchOp(i, i + 1, cand)],
                    )
                )

            # Option C: Merge 2..4 close consecutive ASR chunks onto a script span
            for asr_end, m_cand in multi_cands_by_asr[i]:
                asr_k = asr_end - i
                trans_bonus = _transition_bonus(last_end, m_cand.u_start)
                span_ids = set(range(m_cand.u_start, m_cand.u_end))
                overlap_cnt = len(span_ids & used_set)
                dup_penalty = 0.32 * overlap_cnt * asr_k

                step_score = asr_k * m_cand.sim + trans_bonus - dup_penalty
                if m_cand.sim + max(0.0, trans_bonus / asr_k) < min_match_confidence:
                    continue
                if step_score <= asr_k * min_match_confidence - 0.02:
                    continue

                beams_by_pos[asr_end].append(
                    (
                        score + step_score,
                        m_cand.u_end,
                        used_set | frozenset(span_ids),
                        ops + [_MatchOp(i, asr_end, m_cand)],
                    )
                )

    best_score, _, _, best_ops_raw = _prune_beams(beams_by_pos[n_asr])[0]
    best_ops: list[_MatchOp] = list(best_ops_raw)

    # Step 2.5: Context-sandwiched anchor bridging for homophone names & tight constrained slots
    # Example: ASR="嗯。" + ASR="宋彦周。" sandwiched between matched anchors around Script="宋砚舟。"
    used_bridge_units: set[int] = {
        uid for op in best_ops if op.cand is not None for uid in range(op.cand.u_start, op.cand.u_end)
    }
    bridged_op_indices: set[int] = set()

    def _is_substantive_asr(asr_idx: int) -> bool:
        norm_s, toks = asr_norms[asr_idx]
        if asr_segments[asr_idx].language == "en":
            return len(norm_s) >= 3 and len(toks) >= 1
        return len(norm_s) >= 2

    def _homophone_bridge_sim(asr_idx: int, u_idx: int, *, allow_zero_overlap: bool) -> float | None:
        a_norm_s, a_tok_s = asr_norms[asr_idx]
        u_obj = units[u_idx]
        u_norm_s = u_obj.norm_str
        la, lu = len(a_norm_s), len(u_norm_s)
        if not la or not lu:
            return None
        len_r = min(la, lu) / max(1, max(la, lu))
        if len_r < 0.65:
            return None
        shared = len(set(a_norm_s) & set(u_norm_s))
        sim_val = compute_text_similarity(a_norm_s, a_tok_s, u_norm_s, u_obj.norm_tokens, u_obj.language)
        if shared >= 1:
            if not allow_zero_overlap and shared < max(1, (min(la, lu) + 1) // 3):
                return None
            return max(0.68, sim_val)
        if (
            allow_zero_overlap
            and u_obj.language == "zh"
            and 2 <= la <= 5
            and 2 <= lu <= 5
            and abs(la - lu) <= 1
        ):
            return max(0.68, sim_val)
        return None

    def _any_mid_matches_skipped(mid_ops: list[int], skipped_units: list[int]) -> bool:
        if not mid_ops or not skipped_units:
            return False
        for u_idx in skipped_units:
            u_obj = units[u_idx]
            for mid in mid_ops:
                asr_idx = best_ops[mid].asr_start
                a_n, a_t = asr_norms[asr_idx]
                if compute_text_similarity(a_n, a_t, u_obj.norm_str, u_obj.norm_tokens, u_obj.language) >= 0.38:
                    return True
                if len(skipped_units) == 1 and len(mid_ops) == 1:
                    if _homophone_bridge_sim(asr_idx, u_idx, allow_zero_overlap=True) is not None:
                        return True
        return False

    # Step 2.4: Shift misplaced ultra-short anchors (e.g. duplicate "嗯。"/"啊。") when doing so
    # opens an in-order sandwich for a previously skipped script unit (e.g. "宋彦周" -> "宋砚舟").
    for m_idx in range(len(best_ops)):
        op_m = best_ops[m_idx]
        if op_m.cand is None or (op_m.asr_end - op_m.asr_start) != 1:
            continue
        m_asr = op_m.asr_start
        m_norm, m_toks = asr_norms[m_asr]
        if not (len(m_norm) <= 2 or (asr_segments[m_asr].language == "en" and len(m_toks) <= 1)):
            continue
        cand_m = op_m.cand
        lang_m = asr_segments[m_asr].language
        comb_u_n = (" " if lang_m == "en" else "").join(
            units[ux].norm_str for ux in range(cand_m.u_start, cand_m.u_end)
        )
        comb_u_t = [t for ux in range(cand_m.u_start, cand_m.u_end) for t in units[ux].norm_tokens]

        prev_u_end = 0
        for l_idx in range(m_idx - 1, -1, -1):
            lc = best_ops[l_idx].cand
            if lc is not None:
                prev_u_end = lc.u_end
                break
        next_u_start = n_units
        for r_idx in range(m_idx + 1, len(best_ops)):
            rc = best_ops[r_idx].cand
            if rc is not None:
                next_u_start = rc.u_start
                break

        for direction, skipped_units, step_range in (
            (
                1,
                [u for u in range(prev_u_end, cand_m.u_start) if u not in used_bridge_units],
                range(m_idx + 1, min(len(best_ops), m_idx + 5)),
            ),
            (
                -1,
                [u for u in range(cand_m.u_end, next_u_start) if u not in used_bridge_units],
                range(m_idx - 1, max(-1, m_idx - 5), -1),
            ),
        ):
            if not skipped_units or len(skipped_units) > 3:
                continue
            shifted = False
            for k_swap in step_range:
                if best_ops[k_swap].cand is not None:
                    break
                if (best_ops[k_swap].asr_end - best_ops[k_swap].asr_start) != 1:
                    continue
                k_asr = best_ops[k_swap].asr_start
                k_norm, k_toks = asr_norms[k_asr]
                if not k_norm:
                    continue
                k_sim = compute_text_similarity(
                    k_norm, k_toks, comb_u_n, comb_u_t, asr_segments[k_asr].language
                )
                if k_sim < max(0.85, cand_m.sim - 0.05):
                    continue
                lo, hi = (m_idx + 1, k_swap) if direction > 0 else (k_swap + 1, m_idx)
                mid_sub = [
                    mid for mid in range(lo, hi)
                    if best_ops[mid].cand is None
                    and (best_ops[mid].asr_end - best_ops[mid].asr_start) == 1
                    and _is_substantive_asr(best_ops[mid].asr_start)
                ]
                if _any_mid_matches_skipped(mid_sub, skipped_units):
                    best_ops[m_idx] = _MatchOp(m_asr, m_asr + 1, None)
                    best_ops[k_swap] = _MatchOp(
                        k_asr, k_asr + 1, _CandidateSpan(cand_m.u_start, cand_m.u_end, k_sim)
                    )
                    shifted = True
                    break
            if shifted:
                break

    op_i = 0
    while op_i < len(best_ops):
        if best_ops[op_i].cand is not None:
            op_i += 1
            continue

        left_u_end = 0
        for l_idx in range(op_i - 1, -1, -1):
            lc = best_ops[l_idx].cand
            if lc is not None:
                left_u_end = lc.u_end
                break

        right_u_start = n_units
        right_op_idx = len(best_ops)
        for r_idx in range(op_i + 1, len(best_ops)):
            rc = best_ops[r_idx].cand
            if rc is not None:
                right_u_start = rc.u_start
                right_op_idx = r_idx
                break

        op_run_end = op_i + 1
        while op_run_end < right_op_idx and best_ops[op_run_end].cand is None:
            op_run_end += 1

        gap_units = [
            u_idx for u_idx in range(left_u_end, right_u_start) if u_idx not in used_bridge_units
        ]
        if left_u_end < right_u_start and gap_units and (right_u_start - left_u_end) <= 6:
            # Pass 1: Local relaxed similarity match inside constrained gap (greedy by score descending)
            local_scored: list[tuple[float, int, int, int]] = []
            for k in range(op_i, op_run_end):
                asr_idx = best_ops[k].asr_start
                a_norm_k, a_toks_k = asr_norms[asr_idx]
                if not a_norm_k:
                    continue
                lang_k = asr_segments[asr_idx].language
                for u_s in range(left_u_end, right_u_start):
                    if u_s in used_bridge_units:
                        continue
                    for u_e in range(u_s + 1, min(right_u_start, u_s + 4) + 1):
                        if any(ux in used_bridge_units for ux in range(u_s, u_e)):
                            break
                        comb_n = (" " if lang_k == "en" else "").join(u.norm_str for u in units[u_s:u_e])
                        comb_t = [t for u in units[u_s:u_e] for t in u.norm_tokens]
                        sim_val = compute_text_similarity(a_norm_k, a_toks_k, comb_n, comb_t, lang_k)
                        if sim_val >= 0.38 and _all_span_units_covered(
                            a_norm_k, a_toks_k, units[u_s:u_e], lang_k, min_unit_cov=0.35
                        ):
                            local_scored.append((sim_val, u_e - u_s, k, u_s, u_e))
            local_scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            for sim_val, _, k, u_s, u_e in local_scored:
                if best_ops[k].cand is not None:
                    continue
                if any(ux in used_bridge_units for ux in range(u_s, u_e)):
                    continue
                best_ops[k] = _MatchOp(
                    best_ops[k].asr_start,
                    best_ops[k].asr_end,
                    _CandidateSpan(u_start=u_s, u_end=u_e, sim=sim_val),
                )
                used_bridge_units.update(range(u_s, u_e))
                bridged_op_indices.add(k)

            # Pass 2: Homophone / CJK-name bridging (exact 1-to-1 or 1-to-1 ignoring 1-char fillers like 嗯/哼)
            rem_ops = [k for k in range(op_i, op_run_end) if best_ops[k].cand is None]
            rem_units = [u_idx for u_idx in gap_units if u_idx not in used_bridge_units]
            if rem_ops and rem_units:
                if len(rem_ops) == 1 and len(rem_units) == 1:
                    b_sim = _homophone_bridge_sim(
                        best_ops[rem_ops[0]].asr_start, rem_units[0], allow_zero_overlap=True
                    )
                    if b_sim is not None:
                        k0, u0 = rem_ops[0], rem_units[0]
                        best_ops[k0] = _MatchOp(
                            best_ops[k0].asr_start,
                            best_ops[k0].asr_end,
                            _CandidateSpan(u_start=u0, u_end=u0 + 1, sim=b_sim),
                        )
                        used_bridge_units.add(u0)
                        bridged_op_indices.add(k0)
                elif len(rem_units) == 1 and len(units[rem_units[0]].norm_str) >= 2:
                    sub_ops = [k for k in rem_ops if _is_substantive_asr(best_ops[k].asr_start)]
                    if len(sub_ops) == 1:
                        k0, u0 = sub_ops[0], rem_units[0]
                        b_sim = _homophone_bridge_sim(
                            best_ops[k0].asr_start, u0, allow_zero_overlap=True
                        )
                        if b_sim is not None:
                            best_ops[k0] = _MatchOp(
                                best_ops[k0].asr_start,
                                best_ops[k0].asr_end,
                                _CandidateSpan(u_start=u0, u_end=u0 + 1, sim=b_sim),
                            )
                            used_bridge_units.add(u0)
                            bridged_op_indices.add(k0)
                elif 1 < len(rem_units) <= 3:
                    sub_ops = [k for k in rem_ops if _is_substantive_asr(best_ops[k].asr_start)]
                    if len(sub_ops) == len(rem_units):
                        pair_sims = [
                            _homophone_bridge_sim(
                                best_ops[k].asr_start, u_idx, allow_zero_overlap=False
                            )
                            for k, u_idx in zip(sub_ops, rem_units)
                        ]
                        if all(s is not None for s in pair_sims):
                            for k, u_idx, s_val in zip(sub_ops, rem_units, pair_sims):
                                assert s_val is not None
                                best_ops[k] = _MatchOp(
                                    best_ops[k].asr_start,
                                    best_ops[k].asr_end,
                                    _CandidateSpan(u_start=u_idx, u_end=u_idx + 1, sim=s_val),
                                )
                                used_bridge_units.add(u_idx)
                                bridged_op_indices.add(k)

        op_i = op_run_end

    # Step 3: Build coarse SubtitleCues (ONLY for observed audio windows!) & track anomalies
    cues: list[SubtitleCue] = []
    anomalies: list[AnomalyItem] = []
    extra_audio: list[dict[str, Any]] = []
    duplicate_records: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    matched_unit_counts: dict[int, int] = {}
    matched_asr_count = 0
    sim_sum = 0.0

    for op_idx, op in enumerate(best_ops):
        segs_slice = asr_segments[op.asr_start : op.asr_end]
        seg_start_ms = segs_slice[0].start_ms
        seg_end_ms = segs_slice[-1].end_ms
        valid_ws = [s.src_win_start_ms for s in segs_slice if s.src_win_start_ms >= 0]
        valid_we = [s.src_win_end_ms for s in segs_slice if s.src_win_end_ms >= 0]
        src_w_s = min(valid_ws) if valid_ws else seg_start_ms
        src_w_e = max(valid_we) if valid_we else seg_end_ms
        asr_joined_text = (" " if segs_slice[0].language == "en" else "").join(s.text for s in segs_slice)
        min_seg_conf = min(s.confidence for s in segs_slice)

        cand = op.cand
        if cand is None:
            # Extra spoken dialogue not found reliably in script -> keep ASR text marked low-confidence, NEVER invent script
            cue_idx = len(cues) + 1
            cues.append(
                SubtitleCue(
                    index=cue_idx,
                    start_ms=seg_start_ms,
                    end_ms=seg_end_ms,
                    raw_text=asr_joined_text,
                    display_text=asr_joined_text,
                    confidence=min(0.45, min_seg_conf),
                    anomaly_flags=["剧本外对白（文稿未匹配）"],
                    turn_id=cue_idx,
                    src_win_start_ms=src_w_s,
                    src_win_end_ms=src_w_e,
                )
            )
            extra_audio.append(
                {
                    "start_ms": seg_start_ms,
                    "end_ms": seg_end_ms,
                    "asr_text": asr_joined_text,
                }
            )
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.SCRIPT_EXTRA_AUDIO,
                    severity=AnomalySeverity.WARNING,
                    start_ms=seg_start_ms,
                    end_ms=seg_end_ms,
                    cue_index=cue_idx,
                    message=f"剧本外对白：视频中出现「{asr_joined_text[:28]}」，剧本文稿未可靠匹配",
                    suggestion="已保留识别结果供复核，请确认是否为临场加词或删减版本差异。",
                )
            )
            details.append({"asr_text": asr_joined_text, "matched": False, "sim": 0.0})
            continue

        asr_k = op.asr_end - op.asr_start
        matched_asr_count += asr_k
        sim_sum += cand.sim * asr_k
        span_units = units[cand.u_start : cand.u_end]

        is_dup = any(matched_unit_counts.get(u.unit_idx, 0) > 0 for u in span_units)
        for u in span_units:
            matched_unit_counts[u.unit_idx] = matched_unit_counts.get(u.unit_idx, 0) + 1

        # Distribute coarse [seg_start_ms, seg_end_ms] proportionally across matched ScriptUnits
        # (Stage 3 forced alignment will align sibling cues from the same src_win jointly on real audio)
        weights = [max(1, len(u.norm_str)) for u in span_units]
        total_w = max(1, sum(weights))
        total_dur = max(120, seg_end_ms - seg_start_ms)
        cur_ms = seg_start_ms

        first_cue_idx_for_seg = len(cues) + 1
        flags_extra = ["同音/上下文桥接匹配"] if op_idx in bridged_op_indices else []
        for u_pos, (u, w) in enumerate(zip(span_units, weights)):
            if u_pos == len(span_units) - 1:
                u_end_ms = seg_end_ms
            else:
                u_end_ms = min(seg_end_ms, cur_ms + max(100, int(round(total_dur * w / total_w))))
            cue_idx = len(cues) + 1
            cues.append(
                SubtitleCue(
                    index=cue_idx,
                    start_ms=cur_ms,
                    end_ms=max(cur_ms + 80, u_end_ms),
                    raw_text=u.raw_text,
                    display_text=u.raw_text,
                    confidence=round(min(0.98, 0.45 + 0.55 * cand.sim), 3),
                    anomaly_flags=list(flags_extra),
                    turn_id=u.turn_id,
                    src_win_start_ms=src_w_s,
                    src_win_end_ms=src_w_e,
                )
            )
            cur_ms = u_end_ms

        if is_dup:
            joined_txt = " ".join(u.raw_text for u in span_units)
            duplicate_records.append(
                {
                    "start_ms": seg_start_ms,
                    "end_ms": seg_end_ms,
                    "text": joined_txt,
                }
            )
            anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.SCRIPT_DUPLICATE_LINE,
                    severity=AnomalySeverity.INFO,
                    start_ms=seg_start_ms,
                    end_ms=seg_end_ms,
                    cue_index=first_cue_idx_for_seg,
                    message=f"重复台词/NG 重录片段：「{joined_txt[:28]}」在音频中再次出现",
                    suggestion="请核对粗剪是否保留了重拍片段。",
                )
            )

        details.append(
            {
                "asr_text": asr_joined_text,
                "matched": True,
                "script_text": " ".join(u.raw_text for u in span_units),
                "turn_ids": [u.turn_id for u in span_units],
                "sim": round(cand.sim, 3),
            }
        )

    # Identify skipped/deleted script lines (grouped cleanly by turn)
    matched_unit_set = set(matched_unit_counts.keys())
    units_by_turn: dict[int, list[ScriptUnit]] = {}
    for u in units:
        units_by_turn.setdefault(u.turn_id, []).append(u)

    skipped_turns: list[dict[str, Any]] = []
    for turn in script_doc.turns:
        t_units = units_by_turn.get(turn.turn_id, [])
        if t_units and not any(u.unit_idx in matched_unit_set for u in t_units):
            skipped_turns.append(
                {
                    "turn_id": turn.turn_id,
                    "speaker": turn.speaker,
                    "text": turn.raw_line,
                }
            )

    if skipped_turns:
        preview = "；".join(s["text"][:16] for s in skipped_turns[:3])
        anomalies.append(
            AnomalyItem(
                anomaly_type=AnomalyType.SCRIPT_MISSING_LINE,
                severity=AnomalySeverity.INFO,
                start_ms=0,
                end_ms=asr_segments[-1].end_ms if asr_segments else 1000,
                cue_index=None,
                message=f"剧本中有 {len(skipped_turns)} 句台词未在视频音频中出现，已自动跳过（例如：{preview}）",
                suggestion="粗剪已删掉的台词不会强行生成字幕，可在报告中查看完整缺失清单。",
            )
        )

    overall_conf = sim_sum / max(1, n_asr)
    return ScriptMatchResult(
        cues=cues,
        anomalies=anomalies,
        overall_confidence=overall_conf,
        matched_count=matched_asr_count,
        total_asr_segments=n_asr,
        matched_script_unit_indices=sorted(matched_unit_set),
        skipped_script_turns=skipped_turns,
        extra_audio_segments=extra_audio,
        duplicate_lines=duplicate_records,
        details=details,
    )
