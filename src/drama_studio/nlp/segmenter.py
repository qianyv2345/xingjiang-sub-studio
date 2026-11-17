"""Smart Chinese semantic segmentation and DP-based single/dual-line subtitle layout engine."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
from typing import Literal

from drama_studio.models import CharTimestamp, SubtitleCue, SubtitleTemplate
from drama_studio.nlp.punctuation import (
    ALL_PUNCT_CHARS,
    DECORATIVE_PUNCT,
    SECONDARY_PUNCT,
    STRONG_PUNCT,
    classify_trailing_punctuation_strength,
    extract_canonical_spoken_sequence,
    is_protected_punct,
    normalize_ellipsis_and_dashes,
    strip_display_punctuation,
)

# Common Chinese short-drama idioms & fixed phrases that must never be split internally
_PROTECTED_PHRASES = [
    "莫名其妙", "破镜重圆", "不可思议", "情不自禁", "不知所措", "理所当然",
    "毫不在意", "千方百计", "一举两得", "小心翼翼", "堂堂正正", "迫不得已",
    "之所以", "不仅如此", "也就是说", "换句话说", "无论如何", "不管怎样",
    "与此同时", "除此之外", "总而言之", "顺便说一下", "实话实说", "说实话",
    "对不起", "没关系", "怎么可能", "不可能", "为什么", "听我解释",
    "不要这样", "求求你", "滚出去", "放开我", "别过来", "怎么回事",
    "凭什么", "算了吧", "好不好", "是不是", "知不知道", "能不能",
    "有没有", "干什么", "说什么", "一模一样", "无缘无故", "大惊小怪",
    "自以为是", "口是心非", "背信弃义", "忍无可忍", "死心塌地", "一见钟情",
]

_CONJUNCTIONS = {
    "因为", "所以", "但是", "不过", "而且", "虽然", "可是", "如果", "只要",
    "除非", "既然", "然而", "于是", "因此", "否则", "哪怕", "即使", "虽然",
    "然后", "接着", "另外", "其实", "结果", "毕竟", "况且", "何况",
}

_DISCOURSE_ADVERBS = {
    "难道", "居然", "竟然", "根本", "实在", "终于", "幸好", "果然",
    "本来", "原本", "几乎", "差点", "简直", "到底", "究竟", "万万",
}

_PREPOSITIONS = {"在", "把", "被", "给", "对", "向", "从", "往", "跟", "和", "替", "为", "向", "朝", "沿", "比"}
_NEG_ADVERBS = {"不", "没", "没有", "别", "莫", "勿", "很", "太", "真", "最", "更", "都", "也", "就", "才", "还", "再", "又"}
_AUX_VERBS = {"想", "要", "能", "会", "可以", "应该", "必须", "敢", "愿意", "打算", "准备", "开始"}
_PARTICLES_ATTACH_LEFT = {"的", "地", "得", "了", "着", "过", "吗", "呢", "吧", "啊", "呀", "嘛", "啦", "呗", "罢了"}
_PRONOUNS_SHORT = {"我", "你", "他", "她", "它", "这", "那", "谁"}

_NLP_INIT_DONE = False
_HANLP_PIPELINE = None
_REGISTERED_CUSTOM_WORDS: set[str] = set()


def _ensure_jieba_initialized() -> None:
    global _NLP_INIT_DONE
    if _NLP_INIT_DONE:
        return
    import jieba

    for phrase in _PROTECTED_PHRASES:
        jieba.add_word(phrase, freq=50000)
        _REGISTERED_CUSTOM_WORDS.add(phrase)
    for pair in [("别", "以为"), ("就", "让"), ("能", "让"), ("只", "想"), ("不", "求"), ("绝", "不")]:
        jieba.suggest_freq(pair, tune=True)
    _NLP_INIT_DONE = True


def register_custom_words(words: list[str] | tuple[str, ...]) -> None:
    """Register drama-specific character names/proper nouns into jieba dictionary."""
    if not words:
        return
    _ensure_jieba_initialized()
    import jieba

    for raw in words:
        w = raw.strip()
        if 2 <= len(w) <= 8 and w not in _REGISTERED_CUSTOM_WORDS:
            jieba.add_word(w, freq=50000)
            _REGISTERED_CUSTOM_WORDS.add(w)


def _get_hanlp_pipeline():
    """Optional HanLP pipeline (only loaded if user explicitly selects hanlp backend and it is available)."""
    global _HANLP_PIPELINE
    if _HANLP_PIPELINE is not None:
        return _HANLP_PIPELINE
    try:
        import hanlp

        _HANLP_PIPELINE = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
        return _HANLP_PIPELINE
    except Exception:
        return None


def char_visual_width(ch: str, width_mode: Literal["visual", "strict"] = "visual") -> float:
    """Measure display width of a single character for subtitle fitting."""
    if ch in {"\n", "\r"}:
        return 0.0
    if width_mode == "strict":
        return 1.0
    if ch.isspace():
        return 0.35
    # Half-width ASCII letters, digits, narrow punctuation
    if ord(ch) < 128:
        return 0.55
    return 1.0


def measure_text_width(text: str, width_mode: Literal["visual", "strict"] = "visual") -> float:
    """Measure the maximum single-line display width of `text`."""
    if not text:
        return 0.0
    lines = text.splitlines() or [text]
    return max(sum(char_visual_width(c, width_mode) for c in line) for line in lines)


def measure_flat_width(text: str, width_mode: Literal["visual", "strict"] = "visual") -> float:
    """Measure total width ignoring newlines."""
    return sum(char_visual_width(c, width_mode) for c in text if c not in {"\n", "\r"})


def width_fits_limit(width: float, max_chars: int) -> bool:
    return width <= float(max_chars) + 0.08


@dataclass
class UnitToken:
    """An atomic display/alignment unit (usually 1 CJK char or 1 ASCII word/number, plus trailing punct)."""

    text: str
    raw_with_punct: str
    start_ms: int
    end_ms: int
    confidence: float
    trailing_punct: str = ""
    pause_after_ms: int = 0
    word_id: int = 0
    pos: str = "x"
    is_word_end: bool = True
    protect_next: bool = False
    turn_id: int = 0
    script_punct_hint: str = ""


def tokenize_with_pos(text: str, nlp_backend: Literal["auto", "jieba", "hanlp"] = "jieba") -> list[tuple[str, str]]:
    """Segment Chinese/mixed text into (word, pos_tag) pairs using jieba.posseg (or optional HanLP)."""
    _ensure_jieba_initialized()
    if nlp_backend == "hanlp":
        pipe = _get_hanlp_pipeline()
        if pipe is not None:
            try:
                out = pipe(text, tasks=["tok/fine", "pos/ctb"])
                toks = out.get("tok/fine", [])
                poss = out.get("pos/ctb", [])
                if toks and len(toks) == len(poss):
                    return list(zip(toks, poss))
            except Exception:
                pass

    import jieba.posseg as pseg

    pairs: list[tuple[str, str]] = []
    for w, flag in pseg.cut(text):
        if w:
            pairs.append((w, flag))
    return pairs


def _split_word_into_atomic_units(word: str) -> list[str]:
    """Keep ASCII alphanumeric runs (words/decimals like 'AI-2026' or '3.14') unbreakable; split CJK char-by-char."""
    units: list[str] = []
    buf: list[str] = []
    for ch in word:
        if ch.isascii() and (ch.isalnum() or ch in {".", "-", "_", "'", "%"}):
            buf.append(ch)
        else:
            if buf:
                units.append("".join(buf))
                buf.clear()
            units.append(ch)
    if buf:
        units.append("".join(buf))
    return units


def build_unit_tokens(
    raw_text: str,
    char_timings: list[CharTimestamp],
    *,
    fallback_start_ms: int,
    fallback_end_ms: int,
    nlp_backend: Literal["auto", "jieba", "hanlp"] = "jieba",
    char_turn_ids: list[int] | None = None,
    char_punct_hints: list[str] | None = None,
    extra_phrases: list[str] | None = None,
) -> list[UnitToken]:
    """Map raw punctuated text + character timestamps into semantic UnitTokens with boundary metadata."""
    if extra_phrases:
        register_custom_words(extra_phrases)
    normalized = normalize_ellipsis_and_dashes(raw_text.replace("\n", " ").strip())
    if not normalized:
        return []

    # Step 1: Parse raw text into spoken characters + attached trailing punctuation
    # Match each spoken char against char_timings sequentially
    spoken_items: list[tuple[str, str]] = []  # (char, trailing_punct)
    i = 0
    n = len(normalized)
    while i < n:
        ch = normalized[i]
        if ch.isspace():
            if spoken_items and not spoken_items[-1][1].endswith(" "):
                prev_ch, prev_p = spoken_items[-1]
                # Preserve space after ASCII alphanumeric for word spacing
                if prev_ch.isascii() and prev_ch.isalnum():
                    spoken_items[-1] = (prev_ch, prev_p + " ")
            i += 1
            continue

        is_punct = (ch in ALL_PUNCT_CHARS) and not is_protected_punct(normalized, i)
        if is_punct:
            if spoken_items:
                prev_ch, prev_p = spoken_items[-1]
                spoken_items[-1] = (prev_ch, prev_p + ch)
            i += 1
            continue

        # Handle protected multi-char ellipsis or dash only if not spoken
        spoken_items.append((ch, ""))
        i += 1

    if not spoken_items:
        return []

    # Step 2: Assign timestamps from char_timings using global SequenceMatcher (immune to dropped '.'/'-' or minor ASR/SRT diffs)
    valid_cts = [ct for ct in char_timings if ct.char.strip() != ""]
    n_sp = len(spoken_items)
    total_dur = max(1, fallback_end_ms - fallback_start_ms)
    step_ms = total_dur / max(1, n_sp)

    matched_slots: list[tuple[int, int, float] | None] = [None] * n_sp
    if valid_cts:
        srt_seq = [ch.lower() for ch, _ in spoken_items]
        align_seq = [ct.char.lower() for ct in valid_cts]
        sm = SequenceMatcher(None, srt_seq, align_seq, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for off in range(i2 - i1):
                    ct = valid_cts[j1 + off]
                    matched_slots[i1 + off] = (ct.start_ms, ct.end_ms, ct.confidence)
            elif tag == "replace" and j2 > j1 and i2 > i1:
                span_s = valid_cts[j1].start_ms
                span_e = max(span_s + 20 * (i2 - i1), valid_cts[j2 - 1].end_ms)
                span_step = (span_e - span_s) / max(1, i2 - i1)
                span_conf = sum(valid_cts[j].confidence for j in range(j1, j2)) / max(1, j2 - j1)
                if i2 - i1 != j2 - j1:
                    span_conf = min(span_conf, 0.72)
                for off in range(i2 - i1):
                    cs = int(round(span_s + off * span_step))
                    ce = int(round(span_s + (off + 1) * span_step))
                    matched_slots[i1 + off] = (cs, max(cs + 20, ce), span_conf)

    # Interpolate any unmatched positions (e.g. '.' in 3.14 or '-' in AI-2026) strictly between neighboring anchors
    item_timings: list[tuple[int, int, float]] = [(0, 0, 0.35)] * n_sp
    last_anchor = -1
    next_anchors: list[int] = [-1] * n_sp
    nxt_a = -1
    for k in range(n_sp - 1, -1, -1):
        if matched_slots[k] is not None:
            nxt_a = k
        next_anchors[k] = nxt_a

    k = 0
    while k < n_sp:
        slot = matched_slots[k]
        if slot is not None:
            item_timings[k] = slot
            last_anchor = k
            k += 1
            continue
        run_start = k
        nxt_a = next_anchors[k]
        run_end = nxt_a if nxt_a >= 0 else n_sp
        run_len = run_end - run_start
        left_ms = item_timings[last_anchor][1] if last_anchor >= 0 else fallback_start_ms
        right_ms = matched_slots[nxt_a][0] if nxt_a >= 0 and matched_slots[nxt_a] is not None else fallback_end_ms
        if right_ms <= left_ms:
            right_ms = left_ms + max(25 * run_len, int(round(run_len * step_ms)))
        local_step = (right_ms - left_ms) / max(1, run_len)
        for off in range(run_len):
            cs = int(round(left_ms + off * local_step))
            ce = int(round(left_ms + (off + 1) * local_step))
            item_timings[run_start + off] = (cs, max(cs + 15, ce), 0.75 if run_len <= 2 and last_anchor >= 0 and nxt_a >= 0 else 0.35)
        k = run_end

    # Assign per-spoken-item turn_id and script punctuation hints
    resolved_turns: list[int] = []
    resolved_hints: list[str] = []
    running_turn = 0
    for idx_sp, (_ch, tp) in enumerate(spoken_items):
        if char_turn_ids is not None and idx_sp < len(char_turn_ids):
            t_id = char_turn_ids[idx_sp]
        else:
            t_id = running_turn
        hint = (
            char_punct_hints[idx_sp]
            if (char_punct_hints is not None and idx_sp < len(char_punct_hints))
            else ""
        )
        resolved_turns.append(t_id)
        resolved_hints.append(hint)
        if any(p in STRONG_PUNCT for p in tp) or any(p in STRONG_PUNCT for p in hint):
            running_turn += 1

    # Step 3: Run Chinese NLP segmentation PER CLAUSE & PER TURN (split at punctuation/turn change so
    # adjacent turns or sentences like "...求你。好！" never merge into a fake word "你好").
    clauses: list[tuple[int, int, str]] = []
    c_start = 0
    for idx_sp, (ch, tp) in enumerate(spoken_items):
        turn_changed = idx_sp + 1 < len(spoken_items) and resolved_turns[idx_sp] != resolved_turns[idx_sp + 1]
        hint_break = bool(resolved_hints[idx_sp].strip())
        if tp.strip() or hint_break or turn_changed or idx_sp == len(spoken_items) - 1:
            end_sp = idx_sp + 1
            clause_str = "".join(
                spoken_items[k][0] + (" " if (" " in spoken_items[k][1] and spoken_items[k][0].isascii()) else "")
                for k in range(c_start, end_sp)
            ).strip()
            clauses.append((c_start, end_sp, clause_str))
            c_start = end_sp

    char_to_word: list[tuple[int, str, bool]] = []
    word_texts: list[str] = []
    word_poses: list[str] = []

    for _cs, _ce, c_str in clauses:
        if not c_str:
            continue
        word_pos_pairs = tokenize_with_pos(c_str, nlp_backend=nlp_backend)
        for w_str, pos in word_pos_pairs:
            w_clean = "".join(c for c in w_str if not c.isspace())
            if not w_clean:
                continue
            actual_wid = len(word_texts)
            word_texts.append(w_clean)
            word_poses.append(pos)
            for ci in range(len(w_clean)):
                char_to_word.append((actual_wid, pos, ci == len(w_clean) - 1))

    # Align spoken_items to char_to_word (robust fallback if counts differ slightly)
    units_per_char: list[tuple[int, str, bool]] = []
    if len(char_to_word) == len(spoken_items):
        units_per_char = char_to_word
    else:
        for idx in range(len(spoken_items)):
            if idx < len(char_to_word):
                units_per_char.append(char_to_word[idx])
            else:
                units_per_char.append((idx, "x", True))

    # Merge ASCII alphanumeric runs into atomic UnitTokens so English words/numbers never split
    tokens: list[UnitToken] = []
    idx = 0
    m = len(spoken_items)
    while idx < m:
        ch, tp = spoken_items[idx]
        wid, pos, is_wend = units_per_char[idx]
        s_ms, e_ms, conf = item_timings[idx]
        cur_turn = resolved_turns[idx]
        cur_hint = resolved_hints[idx]

        # Group contiguous ASCII word/number characters that have no punctuation between them
        if ch.isascii() and (ch.isalnum() or ch in {".", "-", "_"}) and not any(
            p in ALL_PUNCT_CHARS for p in tp.strip()
        ):
            j = idx + 1
            text_buf = [ch]
            conf_sum = conf
            last_end = e_ms
            last_tp = tp
            last_hint = cur_hint
            last_wid = wid
            last_pos = pos
            last_wend = is_wend
            while j < m:
                nch, ntp = spoken_items[j]
                if " " in last_tp or resolved_turns[j] != cur_turn:
                    break
                if nch.isascii() and (nch.isalnum() or nch in {".", "-", "_"}):
                    text_buf.append(nch)
                    ns, ne, nc = item_timings[j]
                    last_end = max(last_end, ne)
                    conf_sum += nc
                    last_tp = ntp
                    last_hint = resolved_hints[j]
                    last_wid, last_pos, last_wend = units_per_char[j]
                    j += 1
                    if any(p in ALL_PUNCT_CHARS for p in ntp.strip()):
                        break
                else:
                    break
            cnt = j - idx
            word_str = "".join(text_buf)
            tokens.append(
                UnitToken(
                    text=word_str,
                    raw_with_punct=word_str + last_tp,
                    start_ms=s_ms,
                    end_ms=last_end,
                    confidence=conf_sum / cnt,
                    trailing_punct=last_tp,
                    word_id=last_wid,
                    pos=last_pos,
                    is_word_end=last_wend,
                    turn_id=cur_turn,
                    script_punct_hint=last_hint,
                )
            )
            idx = j
        else:
            tokens.append(
                UnitToken(
                    text=ch,
                    raw_with_punct=ch + tp,
                    start_ms=s_ms,
                    end_ms=e_ms,
                    confidence=conf,
                    trailing_punct=tp,
                    word_id=wid,
                    pos=pos,
                    is_word_end=is_wend,
                    turn_id=cur_turn,
                    script_punct_hint=cur_hint,
                )
            )
            idx += 1

    # Compute inter-token pause durations and syntactic protect_next flags
    for k in range(len(tokens) - 1):
        curr = tokens[k]
        nxt = tokens[k + 1]
        curr.pause_after_ms = max(0, nxt.start_ms - curr.end_ms)

        if curr.turn_id != nxt.turn_id:
            curr.protect_next = False
            continue

        # Protect word interior
        if curr.word_id == nxt.word_id:
            curr.protect_next = True
            continue

        # Protect numeral + classifier/noun (e.g., 三十万 / 两个人)
        if curr.pos == "m" and nxt.pos in {"q", "m", "n"}:
            curr.protect_next = True

        # Protect negation/adverb before single-syllable verb/adj if not punctuated
        curr_word = word_texts[curr.word_id] if curr.word_id < len(word_texts) else curr.text
        nxt_word = word_texts[nxt.word_id] if nxt.word_id < len(word_texts) else nxt.text
        if not curr.trailing_punct.strip():
            if curr_word in _PROTECTED_PHRASES or nxt_word in _PROTECTED_PHRASES:
                pass
            if curr.pos in {"nr", "ns", "nt"} and nxt.pos in {"nr", "ns", "nt"} and curr.pause_after_ms < 80:
                curr.protect_next = True
            # Never split right before particles like 的/地/得/了/着/过
            if nxt.text in _PARTICLES_ATTACH_LEFT and curr.pause_after_ms < 120:
                curr.protect_next = True

    return tokens


def compute_boundary_cost(
    tokens: list[UnitToken],
    split_idx: int,
) -> float:
    """Compute semantic + acoustic naturalness cost for splitting between tokens[split_idx-1] and tokens[split_idx].

    Lower cost = more natural break point.
    """
    left = tokens[split_idx - 1]
    right = tokens[split_idx]
    if left.turn_id != right.turn_id:
        return -1000.0
    cost = 0.0

    # 1. Word interior protection (strongest non-punct penalty)
    if left.word_id == right.word_id:
        cost += 4000.0
    elif left.protect_next:
        cost += 750.0

    # 2. Punctuation strength bonus (considering both SRT punctuation and script hint punctuation)
    p_strength = max(
        classify_trailing_punctuation_strength(left.trailing_punct),
        classify_trailing_punctuation_strength(left.script_punct_hint),
    )
    if p_strength >= 0.99:
        cost -= 360.0
    elif p_strength >= 0.6:
        cost -= 230.0
    elif p_strength > 0:
        cost -= 70.0

    # 3. Real acoustic pause duration bonus
    pause = left.pause_after_ms
    if pause >= 450:
        cost -= 280.0
    elif pause >= 220:
        cost -= 180.0
    elif pause >= 100:
        cost -= 95.0
    elif pause >= 45:
        cost -= 35.0
    elif pause <= 10 and p_strength == 0.0:
        cost += 25.0

    # 4. Syntactic POS boundary heuristics (only when not separated by punctuation)
    if p_strength == 0.0:
        if right.text in _PARTICLES_ATTACH_LEFT:
            cost += 260.0
        if left.text in _NEG_ADVERBS and left.pos in {"d", "v"}:
            cost += 170.0
        if left.text in _PREPOSITIONS and left.pos == "p":
            cost += 160.0
        if left.text in _AUX_VERBS:
            cost += 110.0
        if left.pos == "v" and right.text in _PRONOUNS_SHORT and left.pause_after_ms < 100:
            cost += 180.0
        if left.text in {"让", "叫", "使", "把", "被", "给"} and right.text in _PRONOUNS_SHORT:
            cost += 220.0
        if left.text in {"的", "地", "得"}:
            cost += 180.0
        if right.text in _CONJUNCTIONS or right.text in _DISCOURSE_ADVERBS:
            cost -= 110.0
        if right.text in {"就", "才", "也", "却", "便"} and left.pos in {"r", "n", "d"} and left.pause_after_ms >= 30:
            cost -= 45.0
        if left.text in {"吗", "呢", "吧", "啊", "呀", "嘛", "啦"}:
            cost -= 90.0

    return cost


def _render_slice_text(
    tokens: list[UnitToken],
    start: int,
    end: int,
    *,
    remove_punctuation: bool,
) -> str:
    """Reconstruct text from token slice [start:end], preserving ASCII spacing and handling punctuation."""
    parts: list[str] = []
    for k in range(start, end):
        tok = tokens[k]
        parts.append(tok.raw_with_punct)
    raw = "".join(parts).strip()
    if remove_punctuation:
        cleaned = strip_display_punctuation(raw)
        return cleaned if cleaned else raw
    # Strip trailing comma-only at end of display line? Keep raw when remove_punctuation=False
    return raw


def _slice_width(
    tokens: list[UnitToken],
    start: int,
    end: int,
    *,
    width_mode: Literal["visual", "strict"],
    remove_punctuation: bool,
) -> float:
    txt = _render_slice_text(tokens, start, end, remove_punctuation=remove_punctuation)
    return measure_flat_width(txt, width_mode=width_mode)


def optimize_two_line_wrap(
    tokens: list[UnitToken],
    start: int,
    end: int,
    *,
    max_chars_per_line: int,
    width_mode: Literal["visual", "strict"],
    remove_punctuation: bool,
) -> tuple[str, bool]:
    """Find the best semantic two-line wrap inside [start:end], or keep single line if it fits cleanly.

    Returns (formatted_text, fits_constraints).
    """
    full_text = _render_slice_text(tokens, start, end, remove_punctuation=remove_punctuation)
    full_width = measure_flat_width(full_text, width_mode=width_mode)

    if width_fits_limit(full_width, max_chars_per_line):
        return full_text, True

    best_mid: int | None = None
    best_score = float("inf")

    for mid in range(start + 1, end):
        # Never split inside a protected word unless impossible
        w1 = _slice_width(tokens, start, mid, width_mode=width_mode, remove_punctuation=remove_punctuation)
        w2 = _slice_width(tokens, mid, end, width_mode=width_mode, remove_punctuation=remove_punctuation)
        if not (width_fits_limit(w1, max_chars_per_line) and width_fits_limit(w2, max_chars_per_line)):
            continue

        b_cost = compute_boundary_cost(tokens, mid)
        # Mild aesthetic balance preference (avoid 9-char line 1 + 1-char line 2 unless strong punct)
        balance_penalty = abs(w1 - w2) * 4.5
        if w2 < 2.0 and classify_trailing_punctuation_strength(tokens[mid - 1].trailing_punct) < 0.9:
            balance_penalty += 120.0
        if w1 < 2.0 and classify_trailing_punctuation_strength(tokens[mid - 1].trailing_punct) < 0.9:
            balance_penalty += 120.0

        score = b_cost + balance_penalty
        if score < best_score:
            best_score = score
            best_mid = mid

    if best_mid is not None:
        line1 = _render_slice_text(tokens, start, best_mid, remove_punctuation=remove_punctuation)
        line2 = _render_slice_text(tokens, best_mid, end, remove_punctuation=remove_punctuation)
        return f"{line1}\n{line2}", True

    # Emergency fallback if a single unbreakable token exceeded limit
    mid = (start + end) // 2
    if mid <= start:
        return full_text, False
    line1 = _render_slice_text(tokens, start, mid, remove_punctuation=remove_punctuation)
    line2 = _render_slice_text(tokens, mid, end, remove_punctuation=remove_punctuation)
    fits = width_fits_limit(measure_flat_width(line1, width_mode), max_chars_per_line) and width_fits_limit(
        measure_flat_width(line2, width_mode), max_chars_per_line
    )
    return f"{line1}\n{line2}", fits


def segment_tokens_dp(
    tokens: list[UnitToken],
    template: SubtitleTemplate,
) -> list[tuple[int, int, str, bool]]:
    """Globally optimal Dynamic Programming subtitle segmentation over UnitTokens.

    Single-line and Dual-line modes run distinct, dedicated layout checks.
    Returns list of (start_tok_idx, end_tok_idx, formatted_display_text, fits_all_constraints).
    """
    n = len(tokens)
    if n == 0:
        return []

    max_chars = max(1, int(template.max_chars_per_line))
    max_lines = 2 if template.max_lines >= 2 else 1
    width_mode = template.width_mode
    remove_punct = template.remove_punctuation

    # Prefix widths (approximate quick bound) + exact cached slice measurements
    dp = [float("inf")] * (n + 1)
    prev = [-1] * (n + 1)
    chosen_text: dict[tuple[int, int], tuple[str, bool]] = {}
    dp[0] = 0.0

    for i in range(n):
        if math.isinf(dp[i]):
            continue

        # Boundary cost for starting a new cue at i (0 for start of utterance)
        split_cost = compute_boundary_cost(tokens, i) if i > 0 else 0.0

        found_valid_from_i = False
        for j in range(i + 1, n + 1):
            if tokens[j - 1].turn_id != tokens[i].turn_id:
                break
            flat_w = _slice_width(tokens, i, j, width_mode=width_mode, remove_punctuation=remove_punct)

            # Upper bound pruning: max capacity is max_chars * max_lines
            if flat_w > max_chars * max_lines + 0.15 and j > i + 1:
                break

            if max_lines == 1:
                txt = _render_slice_text(tokens, i, j, remove_punctuation=remove_punct)
                fits = width_fits_limit(flat_w, max_chars)
                if not fits and j > i + 1:
                    break
            else:
                txt, fits = optimize_two_line_wrap(
                    tokens,
                    i,
                    j,
                    max_chars_per_line=max_chars,
                    width_mode=width_mode,
                    remove_punctuation=remove_punct,
                )
                if not fits and j > i + 1:
                    continue

            found_valid_from_i = True
            chosen_text[(i, j)] = (txt, fits)

            # Duration & CPS quality terms
            dur_ms = max(100, tokens[j - 1].end_ms - tokens[i].start_ms)
            dur_sec = dur_ms / 1000.0
            cps = flat_w / max(0.15, dur_sec)

            seg_cost = 40.0  # Base per-cue cost to avoid over-fragmenting into tiny cues
            if flat_w < 3.0:
                has_strong_break = (
                    (j == n)
                    or (j < n and tokens[j].turn_id != tokens[j - 1].turn_id)
                    or max(
                        classify_trailing_punctuation_strength(tokens[j - 1].trailing_punct),
                        classify_trailing_punctuation_strength(tokens[j - 1].script_punct_hint),
                    )
                    >= 0.9
                    or tokens[j - 1].pause_after_ms >= 350
                )
                if not has_strong_break:
                    seg_cost += 180.0
            if dur_ms < template.min_duration_ms:
                seg_cost += (template.min_duration_ms - dur_ms) * 0.15
            if dur_ms > template.max_duration_ms:
                seg_cost += (dur_ms - template.max_duration_ms) * 0.20
            if cps > template.max_cps:
                seg_cost += (cps - template.max_cps) * 45.0
            if not fits:
                seg_cost += 5000.0

            # Slight preference for balanced natural chunk length without forcing full fill
            if max_lines == 1 and flat_w >= max(3, int(max_chars * 0.45)):
                seg_cost -= 15.0
            elif max_lines == 2 and flat_w >= max(4, int(max_chars * 0.9)):
                seg_cost -= 18.0

            total_cost = dp[i] + split_cost + seg_cost
            if total_cost < dp[j]:
                dp[j] = total_cost
                prev[j] = i

        # Safety: if even a single atomic token exceeded max_chars, force advance by 1 token
        if not found_valid_from_i:
            j = i + 1
            txt = _render_slice_text(tokens, i, j, remove_punctuation=remove_punct)
            chosen_text[(i, j)] = (txt, False)
            total_cost = dp[i] + split_cost + 6000.0
            if total_cost < dp[j]:
                dp[j] = total_cost
                prev[j] = i

    # Backtrack optimal path
    segments: list[tuple[int, int, str, bool]] = []
    cur = n
    while cur > 0:
        p = prev[cur]
        if p < 0:
            p = cur - 1
        txt, fits = chosen_text.get(
            (p, cur),
            (_render_slice_text(tokens, p, cur, remove_punctuation=remove_punct), False),
        )
        segments.append((p, cur, txt, fits))
        cur = p
    segments.reverse()
    return segments


def format_single_cue_without_split(
    raw_text: str,
    template: SubtitleTemplate,
    nlp_backend: Literal["auto", "jieba", "hanlp"] = "jieba",
) -> tuple[str, bool]:
    """Layout an existing cue's text (without splitting across multiple cues) according to 1L/2L rules."""
    clean_raw = raw_text.replace("\n", " ").strip()
    if template.max_lines == 1:
        out = strip_display_punctuation(clean_raw) if template.remove_punctuation else clean_raw
        fits = width_fits_limit(measure_flat_width(out, template.width_mode), template.max_chars_per_line)
        return out, fits

    # Dual-line mode: use semantic token wrap
    tokens = build_unit_tokens(
        clean_raw,
        [],
        fallback_start_ms=0,
        fallback_end_ms=1000,
        nlp_backend=nlp_backend,
    )
    if not tokens:
        out = strip_display_punctuation(clean_raw) if template.remove_punctuation else clean_raw
        return out, True
    return optimize_two_line_wrap(
        tokens,
        0,
        len(tokens),
        max_chars_per_line=template.max_chars_per_line,
        width_mode=template.width_mode,
        remove_punctuation=template.remove_punctuation,
    )


def verify_text_integrity(original_cues: list[SubtitleCue], processed_cues: list[SubtitleCue]) -> tuple[bool, str]:
    """Verify that semantic segmentation and punctuation removal did not drop, duplicate, or alter any spoken text."""
    orig_seq = extract_canonical_spoken_sequence(" ".join(c.raw_text for c in original_cues))
    proc_seq = extract_canonical_spoken_sequence(" ".join(c.effective_text for c in processed_cues))
    if orig_seq == proc_seq:
        return True, ""
    return (
        False,
        f"台词文字序列一致性校验失败：原文 {len(orig_seq)} 字，处理后 {len(proc_seq)} 字。",
    )
