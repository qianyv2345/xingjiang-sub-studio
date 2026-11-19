"""Smart bilingual (Chinese & English) semantic segmentation and DP single/dual-line subtitle layout engine."""

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
    count_english_words,
    detect_text_language,
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
    "除非", "既然", "然而", "于是", "因此", "否则", "哪怕", "即使",
    "然后", "接着", "另外", "其实", "结果", "毕竟", "况且", "何况",
}

_DISCOURSE_ADVERBS = {
    "难道", "居然", "竟然", "根本", "实在", "终于", "幸好", "果然",
    "本来", "原本", "几乎", "差点", "简直", "到底", "究竟", "万万",
}

_PREPOSITIONS = {"在", "把", "被", "给", "对", "向", "从", "往", "跟", "和", "替", "为", "朝", "沿", "比"}
_NEG_ADVERBS = {"不", "没", "没有", "别", "莫", "勿", "很", "太", "真", "最", "更", "都", "也", "就", "才", "还", "再", "又"}
_AUX_VERBS = {"想", "要", "能", "会", "可以", "应该", "必须", "敢", "愿意", "打算", "准备", "开始"}
_PARTICLES_ATTACH_LEFT = {"的", "地", "得", "了", "着", "过", "吗", "呢", "吧", "啊", "呀", "嘛", "啦", "呗", "罢了"}
_PRONOUNS_SHORT = {"我", "你", "他", "她", "它", "这", "那", "谁"}

# English syntactic & lexical grouping constants
_EN_TITLE_ABBREVS = {"dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "st.", "rev.", "capt.", "gen."}
_EN_MULTIWORD_COMPOUNDS = {
    ("new", "york"), ("los", "angeles"), ("san", "francisco"), ("las", "vegas"),
    ("united", "states"), ("white", "house"), ("high", "school"), ("hong", "kong"),
    ("prime", "minister"), ("ice", "cream"), ("cell", "phone"), ("next", "door"),
    ("right", "now"), ("each", "other"), ("of", "course"), ("at", "least"),
}
_EN_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "no", "every", "each", "either", "neither",
}
_EN_AUXILIARIES = {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
}
_EN_PREPOSITIONS = {
    "in", "on", "at", "to", "for", "with", "from", "by", "of", "about",
    "into", "onto", "over", "under", "after", "before", "between", "through",
    "during", "without", "within", "against", "toward", "towards", "around",
}
_EN_CONJUNCTIONS = {
    "and", "but", "or", "so", "yet", "nor", "because", "although", "though",
    "since", "while", "whereas", "when", "whenever", "if", "unless", "until",
    "whether", "even",
}
_EN_CLAUSE_STARTERS = {"that", "which", "who", "whom", "whose", "where", "why", "how"}
_EN_NEGATIONS = {"not", "never", "n't"}
_OPENING_WRAP_CHARS = set("“‘([（【「『《〈")

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
        if 2 <= len(w) <= 8 and any("\u4e00" <= c <= "\u9fff" for c in w) and w not in _REGISTERED_CUSTOM_WORDS:
            jieba.add_word(w, freq=50000)
            _REGISTERED_CUSTOM_WORDS.add(w)


def _get_hanlp_pipeline():
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
    """An atomic display/alignment unit (1 CJK char or 1 English word/number, plus optional leading/trailing punct)."""

    text: str
    raw_with_punct: str
    start_ms: int
    end_ms: int
    confidence: float
    trailing_punct: str = ""
    leading_punct: str = ""
    pause_after_ms: int = 0
    word_id: int = 0
    pos: str = "x"
    is_word_end: bool = True
    protect_next: bool = False
    turn_id: int = 0
    script_punct_hint: str = ""
    is_english_word: bool = False


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


def _is_ascii_word_char(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in {".", "-", "_", "'", "’"})


def _is_opening_punct(normalized: str, idx: int) -> bool:
    ch = normalized[idx]
    if ch in _OPENING_WRAP_CHARS:
        return True
    if ch in {'"', "'"}:
        prev_ch = normalized[idx - 1] if idx > 0 else ""
        next_ch = normalized[idx + 1] if idx + 1 < len(normalized) else ""
        if (idx == 0 or prev_ch.isspace() or prev_ch in _OPENING_WRAP_CHARS) and bool(next_ch and not next_ch.isspace()):
            return True
    return False


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
    language: str = "auto",
) -> list[UnitToken]:
    """Map raw punctuated text + character/word timestamps into semantic UnitTokens with boundary metadata."""
    if extra_phrases:
        register_custom_words(extra_phrases)
    normalized = normalize_ellipsis_and_dashes(raw_text.replace("\n", " ").strip())
    if not normalized:
        return []

    eff_lang = detect_text_language(normalized) if language == "auto" else language

    # Step 1: Parse raw text into spoken characters + leading/trailing punctuation
    spoken_items: list[tuple[str, str, str]] = []  # (char, trailing_punct, leading_punct)
    pending_leading = ""
    i = 0
    n = len(normalized)
    while i < n:
        ch = normalized[i]
        if ch.isspace():
            if spoken_items and not spoken_items[-1][1].endswith(" "):
                prev_ch, prev_p, prev_l = spoken_items[-1]
                if prev_ch.isascii() and (prev_ch.isalnum() or prev_ch in {".", "'", "’", "-"}):
                    spoken_items[-1] = (prev_ch, prev_p + " ", prev_l)
            i += 1
            continue

        is_punct = (ch in ALL_PUNCT_CHARS) and not is_protected_punct(normalized, i)
        if is_punct:
            if _is_opening_punct(normalized, i):
                pending_leading += ch
            elif spoken_items:
                prev_ch, prev_p, prev_l = spoken_items[-1]
                spoken_items[-1] = (prev_ch, prev_p + ch, prev_l)
            else:
                pending_leading += ch
            i += 1
            continue

        spoken_items.append((ch, "", pending_leading))
        pending_leading = ""
        i += 1

    if not spoken_items:
        return []

    # Step 2: Assign timestamps from char_timings using global SequenceMatcher
    valid_cts = [ct for ct in char_timings if ct.char.strip() != ""]
    n_sp = len(spoken_items)
    total_dur = max(1, fallback_end_ms - fallback_start_ms)
    step_ms = total_dur / max(1, n_sp)

    matched_slots: list[tuple[int, int, float] | None] = [None] * n_sp
    if valid_cts:
        srt_seq = [ch.lower() for ch, _, _ in spoken_items]
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
            item_timings[run_start + off] = (
                cs,
                max(cs + 15, ce),
                0.75 if run_len <= 2 and last_anchor >= 0 and nxt_a >= 0 else 0.35,
            )
        k = run_end

    # Assign per-spoken-item turn_id and script punctuation hints
    resolved_turns: list[int] = []
    resolved_hints: list[str] = []
    running_turn = 0
    for idx_sp, (_ch, tp, _lp) in enumerate(spoken_items):
        t_id = char_turn_ids[idx_sp] if (char_turn_ids is not None and idx_sp < len(char_turn_ids)) else running_turn
        hint = char_punct_hints[idx_sp] if (char_punct_hints is not None and idx_sp < len(char_punct_hints)) else ""
        resolved_turns.append(t_id)
        resolved_hints.append(hint)
        if any(p in STRONG_PUNCT for p in tp) or any(p in STRONG_PUNCT for p in hint):
            running_turn += 1

    # Step 3: Run Chinese NLP segmentation per clause (skip jieba when pure English)
    char_to_word: list[tuple[int, str, bool]] = []
    word_texts: list[str] = []
    if eff_lang != "en":
        clauses: list[tuple[int, int, str]] = []
        c_start = 0
        for idx_sp, (ch, tp, _lp) in enumerate(spoken_items):
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
                for ci in range(len(w_clean)):
                    char_to_word.append((actual_wid, pos, ci == len(w_clean) - 1))

    units_per_char: list[tuple[int, str, bool]] = []
    if eff_lang != "en" and len(char_to_word) == len(spoken_items):
        units_per_char = char_to_word
    else:
        for idx in range(len(spoken_items)):
            if eff_lang != "en" and idx < len(char_to_word):
                units_per_char.append(char_to_word[idx])
            else:
                units_per_char.append((idx, "x", True))

    # Merge contiguous ASCII words/contractions/abbreviations/numbers into atomic UnitTokens
    tokens: list[UnitToken] = []
    idx = 0
    m = len(spoken_items)
    while idx < m:
        ch, tp, lp = spoken_items[idx]
        wid, pos, is_wend = units_per_char[idx]
        s_ms, e_ms, conf = item_timings[idx]
        cur_turn = resolved_turns[idx]
        cur_hint = resolved_hints[idx]

        # Contiguous ASCII word/number characters without separating punctuation/spaces
        has_separator_punct = any(p in (STRONG_PUNCT | SECONDARY_PUNCT | DECORATIVE_PUNCT) for p in tp.strip())
        if _is_ascii_word_char(ch) and not has_separator_punct and " " not in tp:
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
                nch, ntp, nlp_tok = spoken_items[j]
                if " " in last_tp or resolved_turns[j] != cur_turn or nlp_tok:
                    break
                if _is_ascii_word_char(nch):
                    text_buf.append(nch)
                    _ns, ne, nc = item_timings[j]
                    last_end = max(last_end, ne)
                    conf_sum += nc
                    last_tp = ntp
                    last_hint = resolved_hints[j]
                    last_wid, last_pos, last_wend = units_per_char[j]
                    j += 1
                    if " " in ntp or any(p in (STRONG_PUNCT | SECONDARY_PUNCT | DECORATIVE_PUNCT) for p in ntp.strip()):
                        break
                else:
                    break
            cnt = j - idx
            word_str = "".join(text_buf)
            tokens.append(
                UnitToken(
                    text=word_str,
                    raw_with_punct=lp + word_str + last_tp,
                    start_ms=s_ms,
                    end_ms=last_end,
                    confidence=conf_sum / cnt,
                    trailing_punct=last_tp,
                    leading_punct=lp,
                    word_id=last_wid if eff_lang != "en" else len(tokens),
                    pos=last_pos,
                    is_word_end=last_wend,
                    turn_id=cur_turn,
                    script_punct_hint=last_hint,
                    is_english_word=any(c.isalpha() and c.isascii() for c in word_str),
                )
            )
            idx = j
        else:
            tokens.append(
                UnitToken(
                    text=ch,
                    raw_with_punct=lp + ch + tp,
                    start_ms=s_ms,
                    end_ms=e_ms,
                    confidence=conf,
                    trailing_punct=tp,
                    leading_punct=lp,
                    word_id=wid if eff_lang != "en" else len(tokens),
                    pos=pos,
                    is_word_end=is_wend,
                    turn_id=cur_turn,
                    script_punct_hint=cur_hint,
                    is_english_word=ch.isascii() and ch.isalpha(),
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

        if eff_lang == "en" or (curr.is_english_word and nxt.is_english_word):
            c_low = curr.text.lower()
            n_low = nxt.text.lower()
            # Protect title abbreviations ("Dr. Smith", "Mr. Lee") & known multi-word names ("New York", "U.S.")
            if c_low in _EN_TITLE_ABBREVS:
                curr.protect_next = True
                continue
            if (c_low, n_low) in _EN_MULTIWORD_COMPOUNDS:
                curr.protect_next = True
                continue
            # Consecutive capitalized proper nouns with no punctuation between them ("John Smith")
            if (
                not curr.trailing_punct.strip()
                and curr.text[:1].isupper()
                and nxt.text[:1].isupper()
                and len(curr.text) >= 2
                and len(nxt.text) >= 2
                and c_low not in {"i"}
                and curr.pause_after_ms < 160
            ):
                curr.protect_next = True
            continue

        if curr.word_id == nxt.word_id:
            curr.protect_next = True
            continue

        if curr.pos == "m" and nxt.pos in {"q", "m", "n"}:
            curr.protect_next = True

        if not curr.trailing_punct.strip():
            if curr.pos in {"nr", "ns", "nt"} and nxt.pos in {"nr", "ns", "nt"} and curr.pause_after_ms < 80:
                curr.protect_next = True
            if nxt.text in _PARTICLES_ATTACH_LEFT and curr.pause_after_ms < 120:
                curr.protect_next = True

    return tokens


def compute_boundary_cost(
    tokens: list[UnitToken],
    split_idx: int,
    *,
    language: str = "auto",
) -> float:
    """Compute semantic + acoustic naturalness cost for splitting between tokens[split_idx-1] and tokens[split_idx]."""
    left = tokens[split_idx - 1]
    right = tokens[split_idx]
    if left.turn_id != right.turn_id:
        return -1000.0
    cost = 0.0

    is_en = (language == "en") or (language == "auto" and left.is_english_word and right.is_english_word)

    # 1. Word/collocation interior protection
    if not is_en and left.word_id == right.word_id:
        cost += 4000.0
    elif left.protect_next:
        cost += 950.0 if is_en else 750.0

    # 2. Punctuation strength bonus
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

    # 4. Language-specific syntactic boundary heuristics (when not punctuated)
    if p_strength == 0.0:
        if is_en:
            l_low = left.text.lower()
            r_low = right.text.lower()
            if l_low in _EN_DETERMINERS:
                cost += 240.0
            if l_low in _EN_AUXILIARIES and r_low not in _EN_CONJUNCTIONS:
                cost += 165.0
            if l_low in _EN_PREPOSITIONS:
                cost += 170.0
            if l_low in _EN_NEGATIONS:
                cost += 170.0
            if l_low == "to" and right.is_english_word:
                cost += 180.0
            if r_low in _EN_CONJUNCTIONS:
                cost -= 145.0
            elif r_low in _EN_CLAUSE_STARTERS:
                cost -= 95.0
            elif r_low in _EN_PREPOSITIONS and l_low not in _EN_AUXILIARIES:
                cost -= 55.0
        else:
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
    """Reconstruct text from token slice [start:end], preserving English spacing and punctuation rules."""
    parts: list[str] = []
    for k in range(start, end):
        tok = tokens[k]
        piece = tok.raw_with_punct
        # Ensure space between consecutive English words even if intermediate punctuation stripped space
        if k > start and (tokens[k - 1].is_english_word or tok.is_english_word):
            prev_piece = parts[-1] if parts else ""
            if prev_piece and not prev_piece[-1].isspace() and not piece[:1].isspace():
                parts.append(" ")
        parts.append(piece)
    raw = "".join(parts).strip()
    if remove_punctuation:
        cleaned = strip_display_punctuation(raw)
        return cleaned if cleaned else raw
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


def _line_word_constraints_met(
    line1: str,
    line2: str | None,
    template: SubtitleTemplate,
) -> bool:
    """Enforce English min-words constraints (line2 minimum applies strictly when 2 lines are rendered)."""
    if template.language != "en" and not any(c.isascii() and c.isalpha() for c in (line1 + (line2 or ""))):
        return True
    if template.language == "en":
        w1 = count_english_words(line1)
        if w1 < max(1, int(template.min_words_line1)):
            return False
        if line2 is not None:
            w2 = count_english_words(line2)
            if w2 < max(1, int(template.min_words_line2)):
                return False
    return True


def optimize_two_line_wrap(
    tokens: list[UnitToken],
    start: int,
    end: int,
    *,
    max_chars_per_line: int,
    width_mode: Literal["visual", "strict"],
    remove_punctuation: bool,
    language: str = "zh",
    min_words_line1: int = 1,
    min_words_line2: int = 2,
    balance_two_lines: bool = True,
) -> tuple[str, bool]:
    """Find the best semantic two-line wrap inside [start:end], or keep single line if it fits cleanly."""
    full_text = _render_slice_text(tokens, start, end, remove_punctuation=remove_punctuation)
    full_width = measure_flat_width(full_text, width_mode=width_mode)

    if width_fits_limit(full_width, max_chars_per_line):
        if language != "en" or count_english_words(full_text) >= min_words_line1 or (end - start) <= 1:
            return full_text, True

    best_mid: int | None = None
    best_score = float("inf")

    for mid in range(start + 1, end):
        line1 = _render_slice_text(tokens, start, mid, remove_punctuation=remove_punctuation)
        line2 = _render_slice_text(tokens, mid, end, remove_punctuation=remove_punctuation)
        w1 = measure_flat_width(line1, width_mode=width_mode)
        w2 = measure_flat_width(line2, width_mode=width_mode)
        if not (width_fits_limit(w1, max_chars_per_line) and width_fits_limit(w2, max_chars_per_line)):
            continue

        if language == "en":
            wc1 = count_english_words(line1)
            wc2 = count_english_words(line2)
            if wc1 < max(1, min_words_line1) or wc2 < max(1, min_words_line2):
                continue

        b_cost = compute_boundary_cost(tokens, mid, language=language)
        if balance_two_lines:
            balance_penalty = abs(w1 - w2) * (6.0 if language == "en" else 4.5)
        else:
            balance_penalty = abs(w1 - w2) * 1.0

        if language != "en":
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

    # For English: do NOT force an invalid 2-line wrap violating min_words_line2; let DP split into separate cues
    if language == "en":
        return full_text, False

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
    """Globally optimal Dynamic Programming subtitle segmentation over UnitTokens."""
    n = len(tokens)
    if n == 0:
        return []

    max_chars = max(1, int(template.max_chars_per_line))
    max_lines = 2 if template.max_lines >= 2 else 1
    lang = template.language
    width_mode: Literal["visual", "strict"] = "strict" if lang == "en" else template.width_mode
    remove_punct = template.remove_punctuation

    dp = [float("inf")] * (n + 1)
    prev = [-1] * (n + 1)
    chosen_text: dict[tuple[int, int], tuple[str, bool]] = {}
    dp[0] = 0.0

    for i in range(n):
        if math.isinf(dp[i]):
            continue

        split_cost = compute_boundary_cost(tokens, i, language=lang) if i > 0 else 0.0

        found_valid_from_i = False
        for j in range(i + 1, n + 1):
            if tokens[j - 1].turn_id != tokens[i].turn_id:
                break
            flat_w = _slice_width(tokens, i, j, width_mode=width_mode, remove_punctuation=remove_punct)

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
                    language=lang,
                    min_words_line1=template.min_words_line1,
                    min_words_line2=template.min_words_line2,
                    balance_two_lines=template.balance_two_lines,
                )
                if not fits and j > i + 1:
                    continue

            found_valid_from_i = True
            chosen_text[(i, j)] = (txt, fits)

            dur_ms = max(100, tokens[j - 1].end_ms - tokens[i].start_ms)
            dur_sec = dur_ms / 1000.0
            cps = flat_w / max(0.15, dur_sec)

            seg_cost = 40.0
            if lang == "en":
                wc = count_english_words(txt)
                if wc < 2:
                    has_strong_break = (
                        (j == n)
                        or (j < n and tokens[j].turn_id != tokens[j - 1].turn_id)
                        or max(
                            classify_trailing_punctuation_strength(tokens[j - 1].trailing_punct),
                            classify_trailing_punctuation_strength(tokens[j - 1].script_punct_hint),
                        ) >= 0.9
                        or tokens[j - 1].pause_after_ms >= 350
                    )
                    if not has_strong_break:
                        seg_cost += 160.0
            elif flat_w < 3.0:
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

            if max_lines == 1 and flat_w >= max(3, int(max_chars * 0.45)):
                seg_cost -= 15.0
            elif max_lines == 2 and flat_w >= max(4, int(max_chars * 0.9)):
                seg_cost -= 18.0

            total_cost = dp[i] + split_cost + seg_cost
            if total_cost < dp[j]:
                dp[j] = total_cost
                prev[j] = i

        if not found_valid_from_i:
            j = i + 1
            txt = _render_slice_text(tokens, i, j, remove_punctuation=remove_punct)
            chosen_text[(i, j)] = (txt, False)
            total_cost = dp[i] + split_cost + 6000.0
            if total_cost < dp[j]:
                dp[j] = total_cost
                prev[j] = i

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
    eff_width_mode: Literal["visual", "strict"] = "strict" if template.language == "en" else template.width_mode
    if template.max_lines == 1:
        out = strip_display_punctuation(clean_raw) if template.remove_punctuation else clean_raw
        fits = width_fits_limit(measure_flat_width(out, eff_width_mode), template.max_chars_per_line)
        return out, fits

    tokens = build_unit_tokens(
        clean_raw,
        [],
        fallback_start_ms=0,
        fallback_end_ms=1000,
        nlp_backend=nlp_backend,
        language=template.language,
    )
    if not tokens:
        out = strip_display_punctuation(clean_raw) if template.remove_punctuation else clean_raw
        return out, True
    return optimize_two_line_wrap(
        tokens,
        0,
        len(tokens),
        max_chars_per_line=template.max_chars_per_line,
        width_mode=eff_width_mode,
        remove_punctuation=template.remove_punctuation,
        language=template.language,
        min_words_line1=template.min_words_line1,
        min_words_line2=template.min_words_line2,
        balance_two_lines=template.balance_two_lines,
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
