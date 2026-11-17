"""台词文稿解析与 SRT 说话轮次 / 专有名词提取工具。

支持用户提供的 `.md` / `.txt` 台词提取文稿，例如每场开头的出场人物行（`宋砚舟、顾恒、温云霄、顾玉冗、侍卫若干`）
以及按说话轮次分段的台词正文。解析结果用于：
1. 向中文分词器注入角色人名，避免剧名/人名被拆断；
2. 标记不同说话人/段落轮次边界（`turn_id`），防止两人对话被合并进同一条字幕；
3. 提供脚本内的停顿标点位置参考（绝不改写原 SRT spoken 台词）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
import re

from drama_studio.nlp.punctuation import (
    ALL_PUNCT_CHARS,
    SECONDARY_PUNCT,
    STRONG_PUNCT,
    is_protected_punct,
    normalize_ellipsis_and_dashes,
)


_MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_SCENE_HEADER_RE = re.compile(
    r"^\s*(?:第[0-9一二三四五六七八九十百千万]+(?:集|场|幕)|场景|人物|演员表|出场人物|角色表|备注|说明)[:：\s].*$"
)
_BRACKET_NOTE_RE = re.compile(r"^\s*[（(\[【][^（）\[\]【】]{1,24}[）)\]】]\s*$")
_SPEAKER_PREFIX_RE = re.compile(r"^\s*(?:【([^】]{1,8})】|([^\s：:，。？！、；]{1,6})[：:])\s*(.+)$")
_SPLIT_DELIM_RE = re.compile(r"[、,，\s]+")
_TRAILING_QUANTIFIER_RE = re.compile(r"(?:若干|等人|等众|等)$")
_SENTENCE_PUNCT_SET = STRONG_PUNCT | SECONDARY_PUNCT | {"…", "—"}


@dataclass
class ScriptTurn:
    """台词稿中的一条独立发言轮次/段落。"""

    turn_id: int
    raw_line: str
    spoken_text: str
    speaker: str = ""
    char_punct_after: list[str] = field(default_factory=list)


@dataclass
class ScriptDocument:
    """解析后的单集台词文稿。"""

    path: str
    characters: list[str] = field(default_factory=list)
    turns: list[ScriptTurn] = field(default_factory=list)


@dataclass
class ScriptAlignmentResult:
    """台词稿与原始 SRT 字符的对齐映射结果。"""

    matched: bool
    matched_ratio: float
    char_turn_ids_per_cue: list[list[int]] = field(default_factory=list)
    char_punct_after_per_cue: list[list[str]] = field(default_factory=list)
    characters: list[str] = field(default_factory=list)


def _extract_spoken_and_punct(text: str) -> tuple[str, list[str]]:
    """提取台词连续正文字符以及每个字符后面紧跟的脚本标点（与 segmenter 的字符提取完全一致）。"""
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
    """识别场次开头/切换时的出场人物清单（如 `顾玉冗、温云霄、崇义`）。

    这类行通常由 `、`（或逗号/空格）分隔多个 2~4 字人名，不含句末/句中常见标点，
    也不包含动词长句。命中时返回提取的人名列表，否则返回 None。
    """
    stripped = line.strip()
    if not stripped or "、" not in stripped:
        return None
    if any(p in stripped for p in ("。", "？", "！", "?", "!", "，", ",", "：", ":", "；", ";")):
        return None
    parts = [p.strip() for p in _SPLIT_DELIM_RE.split(stripped) if p.strip()]
    if len(parts) < 2:
        return None
    # 每段应为短称呼（人名或角色标签），不能出现长句
    if any(len(p) > 6 for p in parts):
        return None
    names: list[str] = []
    for part in parts:
        cleaned = _TRAILING_QUANTIFIER_RE.sub("", part).strip()
        # 仅保留看起来像专有名词/称谓的 2~4 字片段
        if 2 <= len(cleaned) <= 4 and all("\u4e00" <= c <= "\u9fff" for c in cleaned):
            if cleaned not in names:
                names.append(cleaned)
    return names if names else None


def read_script_text(path: str | Path) -> str:
    """多编码容错读取台词稿文本。"""
    p = Path(path)
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_dialogue_script(path: str | Path) -> ScriptDocument:
    """解析 `.md` / `.txt` 台词文稿，提取出场人名列表和发言轮次。"""
    text = read_script_text(path)
    characters: list[str] = []
    turns: list[ScriptTurn] = []

    def _add_char_name(name: str) -> None:
        if 2 <= len(name) <= 5 and name not in characters:
            characters.append(name)

    for raw_line in text.splitlines():
        line = _MARKDOWN_HEADING_RE.sub("", raw_line).strip()
        if not line:
            continue
        if _SCENE_HEADER_RE.match(line):
            continue
        if _BRACKET_NOTE_RE.match(line):
            continue

        cast_names = _looks_like_cast_header(line)
        if cast_names is not None:
            for name in cast_names:
                _add_char_name(name)
            continue

        speaker = ""
        dialogue = line
        speaker_match = _SPEAKER_PREFIX_RE.match(line)
        if speaker_match:
            candidate_speaker = (speaker_match.group(1) or speaker_match.group(2) or "").strip()
            rest = (speaker_match.group(3) or "").strip()
            # 确认不是剧内正常对话（比如“你说：……”通常不会当作角色前缀）
            if (
                candidate_speaker
                and rest
                and not any(pron in candidate_speaker for pron in ("我", "你", "他", "她", "它", "谁"))
            ):
                speaker = candidate_speaker
                dialogue = rest
                _add_char_name(candidate_speaker)

        spoken_text, char_punct_after = _extract_spoken_and_punct(dialogue)
        if not spoken_text:
            continue

        turns.append(
            ScriptTurn(
                turn_id=len(turns),
                raw_line=dialogue,
                spoken_text=spoken_text,
                speaker=speaker,
                char_punct_after=char_punct_after,
            )
        )

    return ScriptDocument(
        path=str(path),
        characters=characters,
        turns=turns,
    )


def align_script_turns_to_cues(
    cue_texts: list[str],
    script_doc: ScriptDocument,
    min_match_ratio: float = 0.55,
) -> ScriptAlignmentResult:
    """将台词文稿中的说话轮次（`turn_id`）对齐回原始 SRT 每条字幕的每个 spoken 字符。

    采用 SRT 字符覆盖率而非整篇文稿比值，兼容单场分镜切片（例如 125-03-01）
    匹配整集台词稿（第125集.md）；绝不修改 SRT 原文字。
    """
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

    # 对 'delete'（SRT 中多余的语气词或未收录短句）按同一 cue 内相邻匹配填补，确保单调不减
    last_turn = 0
    for i in range(len(srt_turn_ids)):
        if srt_turn_ids[i] >= 0:
            last_turn = srt_turn_ids[i]
            continue
        # 若后续同 cue 内就有明确 turn_id，则优先就近对齐
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
