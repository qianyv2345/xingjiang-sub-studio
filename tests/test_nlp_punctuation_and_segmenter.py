"""Tests for punctuation safety, Chinese DP semantic segmentation (1L & 2L), and Resolve SRT output."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from drama_studio.align.engine import relayout_cues_with_template
from drama_studio.models import CharTimestamp, SubtitleCue, SubtitleTemplate
from drama_studio.nlp.punctuation import (
    extract_canonical_spoken_sequence,
    strip_display_punctuation,
)
from drama_studio.nlp.script_io import align_script_turns_to_cues, parse_dialogue_script
from drama_studio.nlp.segmenter import (
    build_unit_tokens,
    measure_text_width,
    segment_tokens_dp,
    verify_text_integrity,
    width_fits_limit,
)
from drama_studio.nlp.srt_io import dumps_srt, load_srt, save_srt


def test_punctuation_preserves_decimals_hyphens_and_all_spoken_text() -> None:
    raw = "苏晚晴，“这3.14亿和1,000万，AI-2026项目到底行不行……？”"
    cleaned = strip_display_punctuation(raw)
    assert "3.14" in cleaned
    assert "1,000" in cleaned
    assert "AI-2026" in cleaned
    assert "，" not in cleaned
    assert "？" not in cleaned
    assert "“" not in cleaned
    assert "……" not in cleaned
    assert extract_canonical_spoken_sequence(raw) == extract_canonical_spoken_sequence(cleaned)


def _make_mock_tokens(text: str, pause_indices: dict[int, int] | None = None):
    spoken = extract_canonical_spoken_sequence(text)
    cts: list[CharTimestamp] = []
    cur = 500
    pause_indices = pause_indices or {}
    for idx, ch in enumerate(spoken):
        cts.append(CharTimestamp(char=ch, start_ms=cur, end_ms=cur + 160, confidence=0.95))
        cur += 160 + pause_indices.get(idx, 10)
    return build_unit_tokens(text, cts, fallback_start_ms=500, fallback_end_ms=cur)


def test_single_line_dp_respects_limit_and_semantic_words() -> None:
    text = "顾寒川，你别以为这样就能让我低头，这三千万我会还给你！"
    tokens = _make_mock_tokens(text, pause_indices={2: 350, 14: 400})
    tpl = SubtitleTemplate.preset_portrait_9_16()
    tpl.max_chars_per_line = 9
    tpl.max_lines = 1

    segs = segment_tokens_dp(tokens, tpl)
    assert len(segs) >= 3
    for _, _, disp, fits in segs:
        assert "\n" not in disp
        assert fits
        assert width_fits_limit(measure_text_width(disp, tpl.width_mode), tpl.max_chars_per_line)

    # Protected person name "顾寒川" should not be split apart
    assert "顾寒川" in segs[0][2]

    # Verify strict text integrity
    orig = [SubtitleCue(index=1, start_ms=500, end_ms=6000, raw_text=text)]
    out = [
        SubtitleCue(index=i + 1, start_ms=tokens[s].start_ms, end_ms=tokens[e - 1].end_ms, raw_text=disp, display_text=disp)
        for i, (s, e, disp, _) in enumerate(segs)
    ]
    ok, err = verify_text_integrity(orig, out)
    assert ok, err


def test_two_line_dp_independent_layout_and_natural_line_break() -> None:
    text = "那就让我们拭目以待，看看AI-2026项目到底鹿死谁手。"
    tokens = _make_mock_tokens(text, pause_indices={8: 420})
    tpl = SubtitleTemplate.preset_landscape_16_9()
    tpl.max_chars_per_line = 12
    tpl.max_lines = 2

    segs = segment_tokens_dp(tokens, tpl)
    assert len(segs) >= 1
    for _, _, disp, fits in segs:
        lines = disp.splitlines()
        assert 1 <= len(lines) <= 2
        assert fits
        for ln in lines:
            assert width_fits_limit(measure_text_width(ln, tpl.width_mode), tpl.max_chars_per_line)

    # "AI-2026" and "拭目以待" and "鹿死谁手" must never be split across line break
    joined = "\n".join(s[2] for s in segs)
    assert "AI-2026" in joined
    assert "拭目以待" in joined
    assert "鹿死谁手" in joined


def test_srt_resolve_compatibility_no_bom_lf_no_overlap(tmp_path: Path) -> None:
    cues = [
        SubtitleCue(index=99, start_ms=1000, end_ms=2500, raw_text="第一句", display_text="第一句"),
        SubtitleCue(index=2, start_ms=2400, end_ms=4000, raw_text="重叠第二句", display_text="重叠第二句"),
    ]
    out_srt = tmp_path / "resolve_check.srt"
    save_srt(cues, out_srt, enforce_resolve_safe=True)

    raw_bytes = out_srt.read_bytes()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw_bytes

    reloaded = load_srt(out_srt)
    assert len(reloaded) == 2
    assert reloaded[0].index == 1
    assert reloaded[1].index == 2
    assert reloaded[0].end_ms < reloaded[1].start_ms


def test_clause_punctuation_prevents_cross_sentence_word_merge() -> None:
    """Ensure '...求你。好！' never merges '你'+'好' into fake word '你好'."""
    text = "我哪怕拼尽全力，也绝不求你。 好！ 那就让我们拭目以待。"
    tokens = _make_mock_tokens(text, pause_indices={12: 450, 13: 600})
    tpl = SubtitleTemplate.preset_portrait_9_16()
    tpl.max_lines = 2
    tpl.max_chars_per_line = 14

    segs = segment_tokens_dp(tokens, tpl)
    texts = [s[2] for s in segs]
    assert not any("你好" in t for t in texts), f"Unexpected cross-sentence merge in {texts}"
    assert any("也绝不求你" in t for t in texts), f"Expected '也绝不求你' intact in {texts}"
    assert "好" in texts, f"Expected standalone cue '好' in {texts}"


def _make_char_timings_for_text(text: str, start_ms: int, step_ms: int = 130) -> list[CharTimestamp]:
    spoken = extract_canonical_spoken_sequence(text)
    out: list[CharTimestamp] = []
    cur = start_ms
    for ch in spoken:
        out.append(CharTimestamp(char=ch, start_ms=cur, end_ms=cur + step_ms - 10, confidence=0.95))
        cur += step_ms
    return out


def test_script_parsing_and_turn_boundary_prevents_cross_speaker_merge(tmp_path: Path) -> None:
    """Verify .md cast header extraction, character-name protection, and turn separation even without SRT punct."""
    md = tmp_path / "第125集.md"
    md.write_text(
        "宋砚舟、顾恒、温云霄、顾玉冗、侍卫若干\n\n"
        "顾少爷且慢。\n\n"
        "你是？\n\n"
        "我乃顾玉冗，有要事禀报温云霄。\n\n"
        "顾玉冗、温云霄、崇义\n\n"
        "祁王世子，思念罗姑娘过度。\n",
        encoding="utf-8",
    )
    doc = parse_dialogue_script(md)
    for name in ("宋砚舟", "顾恒", "温云霄", "顾玉冗", "崇义", "侍卫"):
        assert name in doc.characters
    assert all("若干" not in ch for ch in doc.characters)
    assert len(doc.turns) == 4

    # Jianying SRT mistakenly concatenated two speakers with NO punctuation in one 7-char cue
    merged_raw_text = "顾少爷且慢你是"
    align_map = align_script_turns_to_cues([merged_raw_text], doc)
    assert align_map.matched is True
    turn_ids = align_map.char_turn_ids_per_cue[0]
    hints = align_map.char_punct_after_per_cue[0]

    # Even in generous 16:9 20-char mode, the two speakers must NOT be merged into one cue
    tpl_169 = SubtitleTemplate.preset_landscape_16_9()
    cts = _make_char_timings_for_text(merged_raw_text, start_ms=1000)
    tokens = build_unit_tokens(
        merged_raw_text,
        cts,
        fallback_start_ms=1000,
        fallback_end_ms=2000,
        char_turn_ids=turn_ids,
        char_punct_hints=hints,
        extra_phrases=doc.characters,
    )
    segs = segment_tokens_dp(tokens, tpl_169)
    seg_texts = [s[2] for s in segs]
    assert seg_texts == ["顾少爷且慢", "你是"], f"Cross-speaker lines merged: {seg_texts}"

    # Verify fictional character names (顾玉冗, 温云霄) are protected from mid-name splitting
    name_sentence = "我乃顾玉冗，有要事禀报温云霄。"
    tpl_916 = SubtitleTemplate.preset_portrait_9_16()
    tpl_916.max_chars_per_line = 7
    tokens_n = build_unit_tokens(
        name_sentence,
        _make_char_timings_for_text(name_sentence, start_ms=500),
        fallback_start_ms=500,
        fallback_end_ms=2500,
        extra_phrases=doc.characters,
    )
    segs_n = segment_tokens_dp(tokens_n, tpl_916)
    texts_n = [s[2] for s in segs_n]
    assert any("顾玉冗" in t for t in texts_n), f"Name '顾玉冗' split awkwardly in {texts_n}"
    assert any("温云霄" in t for t in texts_n), f"Name '温云霄' split awkwardly in {texts_n}"


def test_intra_sentence_bridging_eliminates_flash_gaps_while_preserving_turn_gap() -> None:
    """Same-sentence chunks bridge to 1ms seam (no flash frames in Resolve), while speaker changes keep min_gap_ms."""
    c1_text = "祁王世子"
    c2_text = "思念罗姑娘过度"
    c3_text = "我知道了。"
    cues = [
        SubtitleCue(
            index=1,
            start_ms=1000,
            end_ms=1650,
            raw_text=c1_text,
            display_text=c1_text,
            char_timings=_make_char_timings_for_text(c1_text, start_ms=1080),
            turn_id=0,
        ),
        SubtitleCue(
            index=2,
            start_ms=1920,
            end_ms=2950,
            raw_text=c2_text,
            display_text=c2_text,
            char_timings=_make_char_timings_for_text(c2_text, start_ms=2000),
            turn_id=0,
        ),
        SubtitleCue(
            index=3,
            start_ms=3050,
            end_ms=3600,
            raw_text=c3_text,
            display_text="我知道了",
            char_timings=_make_char_timings_for_text(c3_text, start_ms=3130),
            turn_id=1,
        ),
    ]
    dummy_audio = np.zeros(5000 * 16, dtype=np.float32)

    tpl = SubtitleTemplate.preset_portrait_9_16()
    assert tpl.padding_unit == "frames"
    assert tpl.lead_frames == 3
    assert tpl.tail_frames == 3
    assert tpl.resolve_padding_ms(25.0) == (120, 120)
    assert tpl.resolve_padding_ms(30.0) == (100, 100)
    assert tpl.bridge_sentence_gaps is True

    # Test customizable frame count & ms mode conversion
    tpl_custom = SubtitleTemplate.preset_portrait_9_16()
    tpl_custom.lead_frames = 5
    tpl_custom.tail_frames = 2
    assert tpl_custom.resolve_padding_ms(25.0) == (200, 80)
    tpl_custom.padding_unit = "ms"
    tpl_custom.lead_offset_ms = 90
    tpl_custom.tail_offset_ms = 160
    assert tpl_custom.resolve_padding_ms(30.0) == (90, 160)

    bridged, _, ok = relayout_cues_with_template(
        cues,
        tpl,
        duration_ms=5000,
        vad_spans=[(1000, 3700)],
        audio_16k=dummy_audio,
    )
    assert ok is True
    assert len(bridged) == 3
    assert bridged[1].start_ms - bridged[0].end_ms == 1
    assert bridged[0].end_ms < bridged[1].start_ms
    assert bridged[2].start_ms - bridged[1].end_ms >= tpl.min_gap_ms

    # Disabling bridge mode preserves standard min_gap_ms spacing
    tpl_no_bridge = SubtitleTemplate.preset_portrait_9_16()
    tpl_no_bridge.bridge_sentence_gaps = False
    spaced, _, ok2 = relayout_cues_with_template(
        cues,
        tpl_no_bridge,
        duration_ms=5000,
        vad_spans=[(1000, 3700)],
        audio_16k=dummy_audio,
    )
    assert ok2 is True
    assert spaced[1].start_ms - spaced[0].end_ms >= tpl_no_bridge.min_gap_ms


def test_build_unit_tokens_keeps_sync_when_aligner_drops_decimal_or_hyphen() -> None:
    """Ensure omitted '.' in '3.14' or '-' in 'AI-2026' never desyncs following character timestamps."""
    text = "这3.14亿和AI-2026到底怎么算"
    # Simulate Qwen3 omitting '.' and '-' while emitting all spoken characters accurately
    emitted = "这314亿和AI2026到底怎么算"
    char_timings = [
        CharTimestamp(char=ch, start_ms=1000 + i * 160, end_ms=1000 + (i + 1) * 160, confidence=0.94)
        for i, ch in enumerate(emitted)
    ]
    tokens = build_unit_tokens(text, char_timings, fallback_start_ms=0, fallback_end_ms=20000)
    assert [t.text for t in tokens][-5:] == ["到", "底", "怎", "么", "算"]
    # Last character '算' must keep its true aligned timestamp (~3720), NOT 20000ms fallback drift!
    assert abs(tokens[-1].end_ms - char_timings[-1].end_ms) <= 20
    assert tokens[-1].confidence >= 0.90


def test_non_invading_padding_preserves_continuous_word_boundaries() -> None:
    """Pre/post padding must expand only into true silence pauses, never eating continuous neighbor chars."""
    from drama_studio.align.engine import _apply_padding_and_merge

    tpl = SubtitleTemplate.preset_portrait_9_16()
    tpl.lead_frames = 3
    tpl.tail_frames = 3
    dummy_audio = np.zeros(8000 * 16, dtype=np.float32)

    # 1. Continuous mid-sentence split (0ms gap: '是不是把自己' ends 2160, '攒的银子都花光了' starts 2160)
    # Followed by tight 80ms pause before '还有呢' (2240..2840) and ample 600ms pause before '好了' (3440..3940).
    raw_cues = [
        SubtitleCue(
            index=1,
            start_ms=1460,
            end_ms=2160,
            raw_text="是不是把自己",
            display_text="是不是把自己",
            char_timings=_make_char_timings_for_text("是不是把自己", start_ms=1460, step_ms=116),
            turn_id=0,
        ),
        SubtitleCue(
            index=2,
            start_ms=2160,
            end_ms=3305,
            raw_text="攒的银子都花光了",
            display_text="攒的银子都花光了",
            char_timings=_make_char_timings_for_text("攒的银子都花光了", start_ms=2160, step_ms=143),
            turn_id=0,
        ),
        SubtitleCue(
            index=3,
            start_ms=3385,  # 80ms genuine pause after 3305
            end_ms=4000,
            raw_text="还有吗？",
            display_text="还有吗",
            char_timings=_make_char_timings_for_text("还有吗？", start_ms=3385, step_ms=205),
            turn_id=0,
        ),
        SubtitleCue(
            index=4,
            start_ms=4600,  # 600ms genuine pause across turn change
            end_ms=5100,
            raw_text="没有。",
            display_text="没有",
            char_timings=_make_char_timings_for_text("没有。", start_ms=4600, step_ms=250),
            turn_id=1,
        ),
    ]
    # Exact char ends match fixture boundaries
    raw_cues[0].char_timings[-1].end_ms = 2160
    raw_cues[1].char_timings[-1].end_ms = 3305
    raw_cues[2].char_timings[-1].end_ms = 4000
    raw_cues[3].char_timings[-1].end_ms = 5100

    # At 24fps, 3 frames = 125ms
    padded = _apply_padding_and_merge([SubtitleCue(**{**c.__dict__, "char_timings": list(c.char_timings)}) for c in raw_cues], tpl, 8000, video_fps=24.0)
    # Outer edges get full 125ms breathing room into silence
    assert padded[0].start_ms == 1460 - 125
    assert padded[3].end_ms == 5100 + 125
    # Continuous 0ms split does NOT steal 125ms from '是不是把自己'
    assert padded[0].end_ms == 2159
    assert padded[1].start_ms == 2160
    # 80ms pause (raw 3305 -> 3385) shares silence proportionally without crossing 3305 or 3385
    assert 3305 <= padded[1].end_ms < padded[2].start_ms <= 3385
    assert padded[2].start_ms >= 3305
    # 600ms cross-turn pause gives full 125ms on both sides while keeping gap >= min_gap_ms
    assert padded[2].end_ms == 4000 + 125
    assert padded[3].start_ms == 4600 - 125

    # 2. Relayout from previously polluted stale start/end timings recovers exact non-invading cut
    stale_cues = [
        SubtitleCue(
            index=1,
            start_ms=1335,
            end_ms=2034,  # Stale buggy -125ms truncated end
            raw_text="是不是把自己",
            display_text="是不是把自己",
            char_timings=list(raw_cues[0].char_timings),
            turn_id=0,
        ),
        SubtitleCue(
            index=2,
            start_ms=2035,  # Stale buggy early start
            end_ms=3430,
            raw_text="攒的银子都花光了",
            display_text="攒的银子都花光了",
            char_timings=list(raw_cues[1].char_timings),
            turn_id=0,
        ),
    ]
    relaid, _, ok = relayout_cues_with_template(stale_cues, tpl, duration_ms=8000, vad_spans=[], audio_16k=dummy_audio, video_fps=24.0)
    assert ok is True
    assert relaid[0].start_ms == 1335
    assert relaid[0].end_ms == 2159
    assert relaid[1].start_ms == 2160
    assert relaid[1].end_ms == 3430

    # Zero-padding relayout removes breathing room cleanly
    tpl_zero = SubtitleTemplate.preset_portrait_9_16()
    tpl_zero.lead_frames = 0
    tpl_zero.tail_frames = 0
    relaid_zero, _, _ = relayout_cues_with_template(relaid, tpl_zero, duration_ms=8000, vad_spans=[], audio_16k=dummy_audio, video_fps=24.0)
    assert relaid_zero[0].start_ms == 1460
    assert relaid_zero[0].end_ms == 2159
    assert relaid_zero[1].start_ms == 2160
    assert relaid_zero[1].end_ms == 3305

    # 3. Unaligned manual cues without char_timings remain idempotent across repeated relayouts
    manual = [SubtitleCue(index=1, start_ms=1000, end_ms=2000, raw_text="你好", display_text="你好", char_timings=[])]
    m1, _, _ = relayout_cues_with_template(manual, tpl, duration_ms=8000, vad_spans=[], audio_16k=dummy_audio, video_fps=24.0)
    m2, _, _ = relayout_cues_with_template(m1, tpl, duration_ms=8000, vad_spans=[], audio_16k=dummy_audio, video_fps=24.0)
    assert (m1[0].start_ms, m1[0].end_ms) == (1000, 2000)
    assert (m2[0].start_ms, m2[0].end_ms) == (1000, 2000)

