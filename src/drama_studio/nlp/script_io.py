"""台词/剧本文稿解析与说话轮次提取工具（支持中文短剧、英文剧本及双语混合格式）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
import re

from drama_studio.nlp.punctuation import (
    ALL_PUNCT_CHARS,
    SECONDARY_PUNCT,
    STRONG_PUNCT,
    detect_text_language,
    is_protected_punct,
    normalize_ellipsis_and_dashes,
)

_MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_SCENE_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"第[0-9一二三四五六七八九十百千万]+(?:集|场|幕|章)"
    r"|(?:场景|时间|地点|人物|演员表|出场人物|角色表|备注|说明|镜头|画面|景别|转场|音效|配乐|动作|提示)[:：\s]"
    r"|(?:INT\.|EXT\.|INT\./EXT\.|I/E\.)\s*"
    r"|(?:SCENE|ACT|SHOT|CAMERA|SFX|BGM|ACTION|NOTE|DIRECTION)\b[:：\s\-]*"
    r"|(?:FADE\s+IN|FADE\s+OUT|CUT\s+TO|DISSOLVE\s+TO|SMASH\s+CUT|MATCH\s+CUT)[:：.\s]*$"
    r")",
    re.IGNORECASE,
)
_EXPLICIT_DIRECTION_LINE_RE = re.compile(r"^\s*(?:△|▲|※|//|-->|★|•\s*(?:镜头|动作|场景|音效))")
_BRACKET_NOTE_RE = re.compile(r"^\s*[（(\[【][^（）\[\]【】]{1,120}[）)\]】]\s*$")
_INLINE_STAGE_BRACKET_RE = re.compile(r"[（(\[【](?:动作|镜头|特写|画外音|旁白|停顿|笑|哭|sfx|beat|v\.o\.|o\.s\.|cont'd|whispering|shouting|pause|laughs|sighs)[^）)\]】]{0,40}[）)\]】]", re.IGNORECASE)
_META_BRACKET_SPEAKER = {"旁白提示", "场景", "动作", "镜头", "音效", "配乐", "备注", "说明", "sfx", "action", "camera", "note"}

_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"【([^】]{1,16})】"
    r"|([^\s：:，。？！、；,?!;]{1,16}(?:\s+[A-Za-z0-9.'\-]{1,12}){0,2}(?:\s*\([^)]{1,16}\))?)\s*[：:]"
    r")\s*(.+)$"
)
_BLOCK_SPEAKER_EN_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9 .'\-#]{1,28})(?:\s*\((?:V\.O\.|O\.S\.|CONT'D|O\.C\.|PRE-LAP|FILTERED)\))?\s*$"
)
_SPLIT_DELIM_RE = re.compile(r"[、,，\s]+")
_TRAILING_QUANTIFIER_RE = re.compile(r"(?:若干|等人|等众|等)$")
_SENTENCE_PUNCT_SET = STRONG_PUNCT | SECONDARY_PUNCT | {"…", "—"}
_STOP_SPEAKER_WORDS = {"I", "NO", "YES", "OH", "OK", "OKAY", "WAIT", "STOP", "HELP", "WHAT", "WHY", "HEY", "HI", "BYE"}


@dataclass
class ScriptTurn:
    """台词稿中的一条独立发言轮次/段落。"""

    turn_id: int
    raw_line: str
    spoken_text: str
    speaker: str = ""
    char_punct_after: list[str] = field(default_factory=list)
    language: str = "zh"


@dataclass
class ScriptDocument:
    """解析后的单集台词文稿。"""

    path: str
    characters: list[str] = field(default_factory=list)
    turns: list[ScriptTurn] = field(default_factory=list)
    language: str = "zh"


@dataclass
class ScriptAlignmentResult:
    """台词稿与原始 SRT 字符的对齐映射结果。"""

    matched: bool
    matched_ratio: float
    char_turn_ids_per_cue: list[list[int]] = field(default_factory=list)
    char_punct_after_per_cue: list[list[str]] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)


def _extract_spoken_and_punct(text: str) -> tuple[str, list[str]]:
    """提取台词连续正文字符以及每个字符后面紧跟的脚本标点（与 segmenter 的字符提取一致）。"""
    normalized = normalize_ellipsis_and_dashes(text.replace("\n", " ").strip())
    spoken_chars: list[str] = []
    punct_after: list[str] = []
    for idx, ch in enumerate(normalized):
        if ch.isspace():
            continue
        is_punct = (ch in ALL_PUNCT_CHARS) and not is_protected_punct(normalized, idx)
        if is_punct:
            if punct_after and (ch in _SENTENCE_PUNCT_SET or ch in ALL_PUNCT_CHARS):
                punct_after[-1] += ch
            continue
        spoken_chars.append(ch)
        punct_after.append("")
    return "".join(spoken_chars), punct_after


def _normalize_match_char(ch: str) -> str:
    return ch.lower()


def _looks_like_cast_header(line: str) -> list[str] | None:
    """识别场次开头/切换时的出场人物清单（如 `顾玉冗、温云霄、崇义`）。"""
    stripped = line.strip()
    if not stripped or "、" not in stripped:
        return None
    if any(p in stripped for p in ("。", "？", "！", "?", "!", "，", ",", "：", ":", "；", ";")):
        return None
    parts = [p.strip() for p in _SPLIT_DELIM_RE.split(stripped) if p.strip()]
    if len(parts) < 2:
        return None
    if any(len(p) > 6 for p in parts):
        return None
    names: list[str] = []
    for part in parts:
        cleaned = _TRAILING_QUANTIFIER_RE.sub("", part).strip()
        if 2 <= len(cleaned) <= 4 and all("\u4e00" <= c <= "\u9fff" for c in cleaned):
            if cleaned not in names:
                names.append(cleaned)
    return names if names else None


def read_script_text(path: str | Path) -> str:
    """多编码容错读取台词稿文本。"""
    p = Path(path)
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "utf-16", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _clean_dialogue_line(text: str) -> str:
    cleaned = _INLINE_STAGE_BRACKET_RE.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _is_non_dialogue_line(line: str) -> bool:
    if not line or re.fullmatch(r"[-=*_~]{3,}", line):
        return True
    if _SCENE_HEADER_RE.match(line):
        return True
    if _EXPLICIT_DIRECTION_LINE_RE.match(line):
        return True
    if _BRACKET_NOTE_RE.match(line):
        return True
    return False


def parse_dialogue_script(path: str | Path) -> ScriptDocument:
    """解析 `.md` / `.txt` 台词文稿（中/英文），提取出场人名与纯净对白轮次。"""
    text = read_script_text(path)
    characters: list[str] = []
    turns: list[ScriptTurn] = []

    def _add_char_name(name: str) -> None:
        clean = re.sub(r"\s*\([^)]*\)$", "", name).strip()
        if 2 <= len(clean) <= 24 and clean not in characters:
            characters.append(clean)

    raw_lines = text.splitlines()
    n_lines = len(raw_lines)
    pending_speaker = ""

    idx = 0
    while idx < n_lines:
        raw_line = raw_lines[idx]
        line = _MARKDOWN_HEADING_RE.sub("", raw_line).strip()
        idx += 1

        if not line:
            pending_speaker = ""
            continue
        if _is_non_dialogue_line(line):
            continue

        cast_names = _looks_like_cast_header(line)
        if cast_names is not None:
            for name in cast_names:
                _add_char_name(name)
            pending_speaker = ""
            continue

        # Check standalone screenplay speaker block (e.g. "DR. SMITH" / "ALEX (V.O.)" on its own line)
        blk_m = _BLOCK_SPEAKER_EN_RE.match(line)
        if blk_m and blk_m.group(1).strip() not in _STOP_SPEAKER_WORDS:
            # Confirm next non-empty, non-bracket line looks like dialogue
            look = idx
            has_following_dialogue = False
            while look < n_lines:
                nxt = raw_lines[look].strip()
                if not nxt:
                    break
                if _BRACKET_NOTE_RE.match(nxt):
                    look += 1
                    continue
                if not _is_non_dialogue_line(nxt):
                    has_following_dialogue = True
                break
            if has_following_dialogue:
                cand = blk_m.group(1).strip().title() if blk_m.group(1).isupper() and len(blk_m.group(1)) > 3 else blk_m.group(1).strip()
                pending_speaker = cand
                _add_char_name(cand)
                continue

        speaker = pending_speaker
        dialogue = line
        speaker_match = _SPEAKER_PREFIX_RE.match(line)
        if speaker_match:
            candidate_speaker = (speaker_match.group(1) or speaker_match.group(2) or "").strip()
            candidate_clean = re.sub(r"\s*\([^)]*\)$", "", candidate_speaker).strip()
            rest = (speaker_match.group(3) or "").strip()
            if candidate_clean.lower() in _META_BRACKET_SPEAKER:
                continue
            if (
                candidate_clean
                and rest
                and not any(pron in candidate_clean for pron in ("我", "你", "他", "她", "它", "谁"))
                and not re.match(r"^\d{1,2}$", candidate_clean)  # avoid timestamp-like "12:30"
            ):
                speaker = candidate_clean
                dialogue = rest
                _add_char_name(candidate_clean)
                pending_speaker = ""

        dialogue = _clean_dialogue_line(dialogue)
        if not dialogue:
            continue

        spoken_text, char_punct_after = _extract_spoken_and_punct(dialogue)
        if not spoken_text:
            continue

        turn_lang = detect_text_language(dialogue)
        turns.append(
            ScriptTurn(
                turn_id=len(turns),
                raw_line=dialogue,
                spoken_text=spoken_text,
                speaker=speaker,
                char_punct_after=char_punct_after,
                language=turn_lang,
            )
        )

    doc_lang = detect_text_language(" ".join(t.raw_line for t in turns))
    return ScriptDocument(
        path=str(path),
        characters=characters,
        turns=turns,
        language=doc_lang,
    )


def align_script_turns_to_cues(
    cue_texts: list[str],
    script_doc: ScriptDocument,
    min_match_ratio: float = 0.55,
) -> ScriptAlignmentResult:
    """将台词文稿中的说话轮次（`turn_id`）对齐回原始 SRT 每条字幕的每个 spoken 字符。"""
    cue_spoken_lengths: list[int] = []
    srt_spoken: list[str] = []
    srt_cue_index: list[int] = []

    for c_idx, text in enumerate(cue_texts):
        spoken, _ = _extract_spoken_and_punct(text)
        cue_spoken_lengths.append(len(spoken))
        for ch in spoken:
            srt_spoken.append(_normalize_match_char(ch))
            srt_cue_index.append(c_idx)

    script_spoken: list[str] = []
    script_char_turn: list[int] = []
    script_char_punct: list[str] = []
    for turn in script_doc.turns:
        for idx, ch in enumerate(turn.spoken_text):
            script_spoken.append(_normalize_match_char(ch))
            script_char_turn.append(turn.turn_id)
            p_after = turn.char_punct_after[idx] if idx < len(turn.char_punct_after) else ""
            script_char_punct.append(p_after)

    if not srt_spoken or not script_spoken:
        return ScriptAlignmentResult(
            matched=False,
            matched_ratio=0.0,
            char_turn_ids_per_cue=[[-1] * n for n in cue_spoken_lengths],
            char_punct_after_per_cue=[[""] * n for n in cue_spoken_lengths],
            characters=list(script_doc.characters),
        )

    matcher = SequenceMatcher(None, srt_spoken, script_spoken, autojunk=False)
    blocks = matcher.get_matching_blocks()
    equal_chars = sum(b.size for b in blocks)
    longest_block = max((b.size for b in blocks), default=0)
    srt_coverage = equal_chars / max(1, len(srt_spoken))
    min_contiguous = min(3, len(srt_spoken))
    if srt_coverage < min_match_ratio or longest_block < min_contiguous:
        return ScriptAlignmentResult(
            matched=False,
            matched_ratio=srt_coverage,
            char_turn_ids_per_cue=[[-1] * n for n in cue_spoken_lengths],
            char_punct_after_per_cue=[[""] * n for n in cue_spoken_lengths],
            characters=list(script_doc.characters),
        )
    ratio = float(srt_coverage)

    srt_turn_ids = [-1] * len(srt_spoken)
    srt_punct_after = [""] * len(srt_spoken)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                srt_turn_ids[i1 + offset] = script_char_turn[j1 + offset]
                srt_punct_after[i1 + offset] = script_char_punct[j1 + offset]
        elif tag == "replace":
            srt_span = max(1, i2 - i1)
            scr_span = max(1, j2 - j1)
            for offset in range(i2 - i1):
                mapped_j = j1 + min(scr_span - 1, int(offset * scr_span / srt_span))
                srt_turn_ids[i1 + offset] = script_char_turn[mapped_j]
                if mapped_j == j2 - 1 or offset == i2 - i1 - 1:
                    srt_punct_after[i1 + offset] = script_char_punct[mapped_j]

    last_turn = 0
    for i in range(len(srt_turn_ids)):
        if srt_turn_ids[i] >= 0:
            last_turn = srt_turn_ids[i]
            continue
        assigned = last_turn
        for lookahead in range(i + 1, min(len(srt_turn_ids), i + 8)):
            if srt_cue_index[lookahead] != srt_cue_index[i]:
                break
            if srt_turn_ids[lookahead] >= 0:
                assigned = srt_turn_ids[lookahead]
                break
        srt_turn_ids[i] = assigned
        last_turn = assigned

    char_turn_ids_per_cue: list[list[int]] = []
    char_punct_after_per_cue: list[list[str]] = []
    cursor = 0
    for length in cue_spoken_lengths:
        char_turn_ids_per_cue.append(srt_turn_ids[cursor : cursor + length])
        char_punct_after_per_cue.append(srt_punct_after[cursor : cursor + length])
        cursor += length

    return ScriptAlignmentResult(
        matched=True,
        matched_ratio=ratio,
        char_turn_ids_per_cue=char_turn_ids_per_cue,
        char_punct_after_per_cue=char_punct_after_per_cue,
        characters=list(script_doc.characters),
    )
