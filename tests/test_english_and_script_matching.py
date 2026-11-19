"""Tests for English punctuation/layout rules, bilingual screenplay parsing, and ASR-to-script matching."""

from __future__ import annotations

from pathlib import Path

from drama_studio.align.base import ASRSegment
from drama_studio.align.engine import relayout_cues_with_template
from drama_studio.models import AnomalyType, CharTimestamp, SubtitleCue, SubtitleTemplate
from drama_studio.nlp.punctuation import (
    count_english_words,
    extract_canonical_spoken_sequence,
    strip_display_punctuation,
)
from drama_studio.nlp.script_io import parse_dialogue_script
from drama_studio.nlp.script_matcher import match_asr_to_script
from drama_studio.nlp.segmenter import (
    build_unit_tokens,
    measure_text_width,
    segment_tokens_dp,
    verify_text_integrity,
    width_fits_limit,
)
import numpy as np


def _mock_en_char_timings(text: str, start_ms: int = 500, step_ms: int = 55) -> list[CharTimestamp]:
    spoken = extract_canonical_spoken_sequence(text)
    cts: list[CharTimestamp] = []
    cur = start_ms
    for ch in spoken:
        cts.append(CharTimestamp(char=ch, start_ms=cur, end_ms=cur + step_ms - 5, confidence=0.94))
        cur += step_ms
    return cts


def test_english_punctuation_apostrophes_abbreviations_and_word_count() -> None:
    assert count_english_words("I don't know what") == 4
    assert count_english_words("you're talking about.") == 3
    assert count_english_words("Dr. Smith visited the U.S. yesterday.") == 6
    assert count_english_words("— ...") == 0

    raw = "Dr. Smith said, \"I don't know what you're talking about in the U.S.!\""
    cleaned = strip_display_punctuation(raw)
    assert "don't" in cleaned
    assert "you're" in cleaned
    assert "Dr." in cleaned
    assert "U.S." in cleaned


def test_english_two_line_26_chars_min_words_and_protection() -> None:
    tpl = SubtitleTemplate.preset_english_default()
    assert tpl.language == "en"
    assert tpl.max_lines == 2
    assert tpl.max_chars_per_line == 26
    assert tpl.min_words_line1 == 1
    assert tpl.min_words_line2 == 2
    assert tpl.remove_punctuation is False
    assert tpl.width_mode == "strict"

    text = "I don't know what you're talking about."
    cts = _mock_en_char_timings(text)
    tokens = build_unit_tokens(text, cts, fallback_start_ms=500, fallback_end_ms=3200, language="en")
    segs = segment_tokens_dp(tokens, tpl)
    assert len(segs) == 1
    disp = segs[0][2]
    lines = disp.splitlines()
    assert len(lines) == 2
    for ln in lines:
        assert len(ln) <= 26
        assert width_fits_limit(measure_text_width(ln, "strict"), 26)
    assert count_english_words(lines[0]) >= 1
    assert count_english_words(lines[1]) >= 2
    assert "don't" in disp
    assert "you're" in disp

    # Abbreviations & multi-word proper nouns must stay glued cleanly
    text2 = "Dr. Smith went to New York to meet the U.S. senator."
    tokens2 = build_unit_tokens(
        text2,
        _mock_en_char_timings(text2),
        fallback_start_ms=500,
        fallback_end_ms=4200,
        language="en",
    )
    segs2 = segment_tokens_dp(tokens2, tpl)
    all_lines2 = "\n".join(s[2] for s in segs2)
    for s in segs2:
        for ln in s[2].splitlines():
            assert len(ln) <= 26
        if len(s[2].splitlines()) == 2:
            l1, l2 = s[2].splitlines()
            assert count_english_words(l1) >= 1
            assert count_english_words(l2) >= 2
    assert "Dr. Smith" in all_lines2
    assert "New York" in all_lines2
    assert "U.S." in all_lines2

    # Short single-line cue should NOT fail min_words_line2
    short_txt = "Wait a second!"
    tok_short = build_unit_tokens(short_txt, _mock_en_char_timings(short_txt), fallback_start_ms=500, fallback_end_ms=1500, language="en")
    segs_short = segment_tokens_dp(tok_short, tpl)
    assert len(segs_short) == 1
    assert segs_short[0][2] == "Wait a second!"

    # 2 long words > 26 chars total cannot satisfy 2 lines with >=2 words on line2 -> splits into 2 cues!
    two_long = "Extraordinary circumstances"
    tok_two = build_unit_tokens(two_long, _mock_en_char_timings(two_long), fallback_start_ms=500, fallback_end_ms=2500, language="en")
    segs_two = segment_tokens_dp(tok_two, tpl)
    assert len(segs_two) == 2
    assert [s[2] for s in segs_two] == ["Extraordinary", "circumstances"]

    # Relayout preserves English words & punctuation across repeated template edits
    cue_in = [
        SubtitleCue(
            index=1,
            start_ms=500,
            end_ms=3200,
            raw_text=text,
            display_text=disp,
            char_timings=cts,
        )
    ]
    tpl_1l = SubtitleTemplate.preset_english_default()
    tpl_1l.max_lines = 1
    tpl_1l.max_chars_per_line = 22
    relaid, anomalies, ok = relayout_cues_with_template(
        cue_in,
        tpl_1l,
        duration_ms=4000,
        vad_spans=[(450, 3300)],
        audio_16k=np.zeros(4000 * 16, dtype=np.float32),
    )
    assert ok is True
    assert len(relaid) >= 2
    for c in relaid:
        assert len(c.lines()) == 1
        assert len(c.effective_text) <= 22

    # Unbreakable single token > max_chars stays whole and triggers layout/char warning without dropping words
    huge_word = "Supercalifragilisticexpialidocious"
    cue_huge = [
        SubtitleCue(
            index=1,
            start_ms=500,
            end_ms=2500,
            raw_text=huge_word,
            display_text=huge_word,
            char_timings=_mock_en_char_timings(huge_word),
        )
    ]
    relaid_h, anom_h, ok_h = relayout_cues_with_template(
        cue_huge,
        tpl,
        duration_ms=3000,
        vad_spans=[(450, 2600)],
        audio_16k=np.zeros(3000 * 16, dtype=np.float32),
    )
    assert ok_h is True
    assert huge_word in relaid_h[0].effective_text
    assert any(a.anomaly_type == AnomalyType.OVER_CHAR_LIMIT for a in anom_h)


def test_bilingual_script_parser_ignores_directions_and_camera_notes(tmp_path: Path) -> None:
    en_script = tmp_path / "ep01_en.md"
    en_script.write_text(
        "# Scene 1\n"
        "INT. HOSPITAL CORRIDOR - NIGHT\n\n"
        "[Camera pans across the hallway]\n"
        "DR. SMITH\n"
        "(checking clipboard)\n"
        "I don't know what you're talking about.\n\n"
        "SARAH: Wait! You promised we'd save him!\n"
        "(Sarah turns away in tears.)\n"
        "CUT TO:\n",
        encoding="utf-8",
    )
    doc = parse_dialogue_script(en_script)
    assert doc.language == "en"
    assert len(doc.turns) == 2
    assert doc.turns[0].raw_line == "I don't know what you're talking about."
    assert doc.turns[1].raw_line == "Wait! You promised we'd save him!"
    assert "Dr. Smith" in doc.characters or "DR. SMITH" in [c.upper() for c in doc.characters]

    zh_script = tmp_path / "ep01_zh.txt"
    zh_script.write_text(
        "第1场\n"
        "场景：会议室 日\n"
        "人物：顾寒川、苏晚晴\n"
        "△ 镜头缓缓推进\n"
        "（众人沉默）\n"
        "顾寒川：你别以为这样就能让我低头！\n"
        "【动作】苏晚晴拍桌而起\n"
        "苏晚晴：这三点一四亿你拿什么还？\n",
        encoding="utf-8",
    )
    doc_zh = parse_dialogue_script(zh_script)
    assert doc_zh.language == "zh"
    assert [t.raw_line for t in doc_zh.turns] == [
        "你别以为这样就能让我低头！",
        "这三点一四亿你拿什么还？",
    ]


def test_asr_to_script_matcher_handles_deletions_out_of_order_duplicates_and_homophones(tmp_path: Path) -> None:
    # Chinese: deleted lines, homophone typo in ASR ("顾汉川"), duplicate line, extra ad-lib
    zh_script = tmp_path / "drama_zh.md"
    zh_script.write_text(
        "顾寒川：你别以为这样就能让我低头！\n"
        "旁白：这句被粗剪删掉了，绝对不能出现在输出里。\n"
        "苏晚晴：这三点一四亿你拿什么还？\n"
        "顾寒川：放手！\n"
        "苏晚晴：绝不可能！\n"
        "顾寒川：放手！\n",
        encoding="utf-8",
    )
    doc_zh = parse_dialogue_script(zh_script)
    asr_segs_zh = [
        ASRSegment(start_ms=600, end_ms=2600, text="苏晚晴这3.14亿你拿什么还", language="zh"),
        # Out-of-order: earlier turn appears after later turn! Plus homophone "顾汉川" vs "顾寒川"
        ASRSegment(start_ms=3000, end_ms=5200, text="你别以为这样就能让我低头", language="zh"),
        ASRSegment(start_ms=5600, end_ms=6400, text="放手", language="zh"),
        # Extra ad-lib not in script
        ASRSegment(start_ms=6800, end_ms=7900, text="保安把门关上", language="zh"),
        # Second occurrence of duplicate line
        ASRSegment(start_ms=8200, end_ms=9000, text="放手", language="zh"),
    ]
    res_zh = match_asr_to_script(asr_segs_zh, doc_zh)
    texts = [c.raw_text for c in res_zh.cues]
    # Deleted line must NEVER be synthesized into output cues
    assert not any("粗剪删掉了" in t for t in texts)
    assert any("这三点一四亿你拿什么还" in t for t in texts)
    assert any("你别以为这样就能让我低头" in t for t in texts)
    assert texts.count("放手！") == 2
    assert any("保安把门关上" in t for t in texts)
    assert len(res_zh.skipped_script_turns) == 2  # deleted line + "绝不可能！"
    assert any(a.anomaly_type == AnomalyType.SCRIPT_MISSING_LINE for a in res_zh.anomalies)
    assert any(a.anomaly_type == AnomalyType.SCRIPT_EXTRA_AUDIO for a in res_zh.anomalies)

    # English: contraction differences + deleted line
    en_script = tmp_path / "drama_en.md"
    en_script.write_text(
        "ALEX: I don't know what you're talking about.\n"
        "ALEX: This scene was cut from the final edit completely.\n"
        "SARAH: Dr. Smith is already waiting in New York.\n",
        encoding="utf-8",
    )
    doc_en = parse_dialogue_script(en_script)
    asr_segs_en = [
        ASRSegment(start_ms=500, end_ms=2900, text="I do not know what you are talking about", language="en"),
        ASRSegment(start_ms=3400, end_ms=5900, text="Doctor Smith is already waiting in New York", language="en"),
    ]
    res_en = match_asr_to_script(asr_segs_en, doc_en)
    assert res_en.matched_count == 2
    assert res_en.cues[0].raw_text == "I don't know what you're talking about."
    assert res_en.cues[1].raw_text == "Dr. Smith is already waiting in New York."
    assert len(res_en.skipped_script_turns) == 1
    assert "cut from the final edit" in res_en.skipped_script_turns[0]["text"]


def test_script_matcher_many_to_one_split_pause_and_filler_homophone_sandwich(tmp_path: Path) -> None:
    script = tmp_path / "ep_homophone_split.md"
    script.write_text(
        "因为是你戴。\n"
        "宋砚舟。\n"
        "嗯？\n"
        "我真的很想你。\n"
        "早知道见面能亲上，我就该早点来盛京。\n"
        "那当然。\n"
        "宋砚舟。\n"
        "嗯。\n"
        "我还想亲。\n",
        encoding="utf-8",
    )
    doc = parse_dialogue_script(script)
    asr_segs = [
        ASRSegment(start_ms=1000, end_ms=2200, text="因为是你带。", language="zh", src_win_start_ms=1000, src_win_end_ms=2200),
        # Duplicated ultra-short filler "嗯。" before homophone name "宋彦周。" + real "嗯。" after it
        ASRSegment(start_ms=2500, end_ms=3100, text="嗯。", language="zh", src_win_start_ms=2500, src_win_end_ms=3100),
        ASRSegment(start_ms=3400, end_ms=4400, text="宋彦周。", language="zh", src_win_start_ms=3400, src_win_end_ms=4400),
        ASRSegment(start_ms=4800, end_ms=5400, text="嗯。", language="zh", src_win_start_ms=4800, src_win_end_ms=5400),
        ASRSegment(start_ms=5900, end_ms=7200, text="我真的很想你。", language="zh", src_win_start_ms=5900, src_win_end_ms=7200),
        # Single long script turn split into 2 close ASR segments across an 80ms breath pause
        ASRSegment(start_ms=8000, end_ms=9800, text="早知道见面能亲上。", language="zh", src_win_start_ms=8000, src_win_end_ms=9800),
        ASRSegment(start_ms=9880, end_ms=11500, text="我就该早点来圣境。", language="zh", src_win_start_ms=9880, src_win_end_ms=11500),
        ASRSegment(start_ms=12200, end_ms=13200, text="那当然。", language="zh", src_win_start_ms=12200, src_win_end_ms=13200),
        # Filler "哼。" paired with homophone "宋彦周。" between matched anchors
        ASRSegment(start_ms=13500, end_ms=14100, text="哼。", language="zh", src_win_start_ms=13500, src_win_end_ms=14100),
        ASRSegment(start_ms=14600, end_ms=15500, text="宋彦周。", language="zh", src_win_start_ms=14600, src_win_end_ms=15500),
        ASRSegment(start_ms=16000, end_ms=16600, text="嗯。", language="zh", src_win_start_ms=16000, src_win_end_ms=16600),
        ASRSegment(start_ms=17200, end_ms=18300, text="我还想亲。", language="zh", src_win_start_ms=17200, src_win_end_ms=18300),
    ]
    res = match_asr_to_script(asr_segs, doc)
    assert res.skipped_script_turns == []
    texts = [c.raw_text for c in res.cues]
    assert sum(1 for t in texts if t == "宋砚舟。") == 2
    # Merged long sentence must emit once across combined [8000..11500] window, with no duplicate anomaly
    merged_cues = [c for c in res.cues if "盛京" in c.raw_text]
    assert len(merged_cues) == 1
    assert merged_cues[0].start_ms == 8000 and merged_cues[0].end_ms == 11500
    assert merged_cues[0].src_win_start_ms == 8000 and merged_cues[0].src_win_end_ms == 11500
    assert res.duplicate_lines == []
    extra_texts = {e["asr_text"] for e in res.extra_audio_segments}
    assert {"嗯。", "哼。"} <= extra_texts
