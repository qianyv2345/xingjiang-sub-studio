"""Comprehensive tests for 3-mode workflow, stage caching, OOM recovery, non-overwriting export, and real E2E ASR."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from drama_studio.align.asr_engine import ASRRunResult
from drama_studio.align.base import ASRSegment
from drama_studio.align.engine import (
    MissingAlignerModelError,
    run_forced_alignment_only,
    segment_aligned_windows_to_cues,
)
from drama_studio.align.model_manager import (
    MODEL_REGISTRY,
    delete_cached_model,
    is_model_cached,
)
from drama_studio.align.qwen_asr_backend import Qwen3ASRBackend
from drama_studio.app import configure_runtime_environment
from drama_studio.cache import (
    clear_stage_cache,
    get_stage_cache_size_mb,
    prune_stage_cache,
)
from drama_studio.config import ConfigStore, default_model_cache_dir
from drama_studio.exporters.exporter import export_episode_outputs
from drama_studio.media.ffmpeg_wrapper import extract_audio_to_wav, find_ffmpeg_binaries, probe_media
from drama_studio.media.pairing import disambiguate_episode_labels, pair_media_files
from drama_studio.models import (
    AlignmentConfig,
    AnomalyType,
    AudioLevelingConfig,
    EpisodeTask,
    ExportConfig,
    ManualGainPoint,
    SubtitleCue,
    SubtitleTemplate,
    TaskMode,
)
from drama_studio.nlp.punctuation import count_english_words
from drama_studio.nlp.srt_io import load_srt
from drama_studio.pipeline import process_single_episode

FX = Path(__file__).resolve().parent / "fixtures"


def _assert_cues_valid(cues: list[SubtitleCue], duration_ms: int, tpl: SubtitleTemplate) -> None:
    assert len(cues) >= 1
    for i, c in enumerate(cues):
        assert 0 <= c.start_ms < c.end_ms
        if duration_ms > 0:
            assert c.end_ms <= duration_ms + 80
        if i > 0:
            assert c.start_ms >= cues[i - 1].end_ms, f"Overlap or regression between cue {i} and {i + 1}"
        lines = c.lines()
        assert 1 <= len(lines) <= tpl.max_lines
        if tpl.language == "en":
            for ln in lines:
                assert len(ln) <= tpl.max_chars_per_line + 1
            if len(lines) == 2:
                assert count_english_words(lines[0]) >= tpl.min_words_line1
                assert count_english_words(lines[1]) >= tpl.min_words_line2


def test_task_mode_auto_detection_and_manual_preservation() -> None:
    t_only_vid = EpisodeTask(task_id="1", episode_label="ep1", video_path="v.mp4")
    assert t_only_vid.resolve_effective_mode() == TaskMode.ASR_GENERATE

    t_vid_script = EpisodeTask(task_id="2", episode_label="ep2", video_path="v.mp4", script_path="s.md")
    assert t_vid_script.resolve_effective_mode() == TaskMode.SCRIPT_MATCH

    t_vid_srt = EpisodeTask(task_id="3", episode_label="ep3", video_path="v.mp4", srt_path="a.srt")
    assert t_vid_srt.resolve_effective_mode() == TaskMode.SRT_FIX

    # Both script and SRT exist: default SRT_FIX, manual switch to SCRIPT_MATCH keeps both intact
    t_both = EpisodeTask(task_id="4", episode_label="ep4", video_path="v.mp4", srt_path="a.srt", script_path="s.md")
    assert t_both.resolve_effective_mode() == TaskMode.SRT_FIX
    t_both.task_mode = "script_match"
    assert t_both.resolve_effective_mode() == TaskMode.SCRIPT_MATCH
    assert t_both.srt_path == "a.srt" and t_both.script_path == "s.md"


def test_model_registry_integrity_incomplete_guard_and_missing_1_7b(tmp_path: Path) -> None:
    assert "qwen3_asr_0.6b" in MODEL_REGISTRY
    assert "qwen3_asr_1.7b" in MODEL_REGISTRY

    # Incomplete download must NOT report cached
    m06 = tmp_path / "qwen3_asr_0.6b"
    m06.mkdir()
    (m06 / "config.json").write_text("{}", encoding="utf-8")
    (m06 / "model.safetensors").write_bytes(b"0" * 2_000_000)
    (m06 / "part.incomplete").write_bytes(b"123")
    assert is_model_cached("qwen3_asr_0.6b", tmp_path) is False

    (m06 / "part.incomplete").unlink()
    assert is_model_cached("qwen3_asr_0.6b", tmp_path) is True

    # Sharded safetensors index with missing shard must fail integrity check
    m17 = tmp_path / "qwen3_asr_1.7b"
    m17.mkdir()
    (m17 / "config.json").write_text("{}", encoding="utf-8")
    (m17 / "model-00001-of-00002.safetensors").write_bytes(b"0" * 2_000_000)
    (m17 / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    assert is_model_cached("qwen3_asr_1.7b", tmp_path) is False
    # Missing 1.7B does NOT break 0.6B availability
    assert is_model_cached("qwen3_asr_0.6b", tmp_path) is True

    assert delete_cached_model("qwen3_asr_0.6b", tmp_path) is True
    assert is_model_cached("qwen3_asr_0.6b", tmp_path) is False


def test_english_alignment_never_uses_chinese_funasr_fa(tmp_path: Path) -> None:
    configure_runtime_environment()
    # Fake a cache dir that ONLY has funasr_fa (no qwen3_aligner)
    fa_dir = tmp_path / "funasr_fa"
    fa_dir.mkdir()
    (fa_dir / "config.yaml").write_text("stub: true", encoding="utf-8")
    (fa_dir / "model.pt").write_bytes(b"0" * 1_500_000)

    cues = [SubtitleCue(index=1, start_ms=200, end_ms=1600, raw_text="I don't know what you mean.")]
    audio_16k = np.zeros(2000 * 16, dtype=np.float32)

    # Default strict behavior: missing English aligner must NOT silently emit interpolated timestamps
    with pytest.raises(MissingAlignerModelError) as exc_info:
        run_forced_alignment_only(
            raw_cues=cues,
            audio_16k=audio_16k,
            vad_spans=[(200, 1600)],
            duration_ms=2000,
            alignment_cfg=AlignmentConfig(backend="funasr", language="en"),
            cache_dir=tmp_path,
            language="en",
        )
    assert exc_info.value.recommended_model_id == "qwen3_aligner"

    # Chinese with empty cache dir must also refuse silent estimation by default
    empty_cache = tmp_path / "empty_models"
    empty_cache.mkdir()
    with pytest.raises(MissingAlignerModelError) as exc_zh:
        run_forced_alignment_only(
            raw_cues=[SubtitleCue(index=1, start_ms=200, end_ms=1600, raw_text="你好世界")],
            audio_16k=audio_16k,
            vad_spans=[(200, 1600)],
            duration_ms=2000,
            alignment_cfg=AlignmentConfig(backend="auto", language="zh"),
            cache_dir=empty_cache,
            language="zh",
        )
    assert exc_zh.value.recommended_model_id == "funasr_fa"

    # Explicit opt-in allow_vad_fallback=True: still forbids fa-zh, falls back to VAD with explicit warnings & cue flags
    cfg_fb = AlignmentConfig(backend="funasr", language="en", allow_vad_fallback=True)
    res = run_forced_alignment_only(
        raw_cues=cues,
        audio_16k=audio_16k,
        vad_spans=[(200, 1600)],
        duration_ms=2000,
        alignment_cfg=cfg_fb,
        cache_dir=tmp_path,
        language="en",
    )
    assert "fa-zh" not in res.aligner_used
    assert "VAD" in res.aligner_used
    assert any("FunASR fa-zh" in w.message for w in res.pre_warnings)
    assert any("估算" in w.message for w in res.pre_warnings)

    seg_res = segment_aligned_windows_to_cues(
        cues,
        res,
        audio_16k,
        [(200, 1600)],
        2000,
        SubtitleTemplate.preset_english_default(),
        cfg_fb,
    )
    assert seg_res.cues
    assert any("VAD估算时间轴" in f for c in seg_res.cues for f in c.anomaly_flags)


def test_qwen_1_7b_oom_recovery_downgrades_cleanly_without_crash(tmp_path: Path) -> None:
    # Prepare stub directories for both 1.7B and 0.6B
    for mid in ("qwen3_asr_1.7b", "qwen3_asr_0.6b"):
        d = tmp_path / mid
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "model.safetensors").write_bytes(b"0" * 1_500_000)

    class _OOMModel:
        def transcribe(self, **_kwargs):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    class _Good06Model:
        def transcribe(self, audio, **_kwargs):
            from types import SimpleNamespace

            return [SimpleNamespace(language="English", text="I don't know what you're talking about.") for _ in audio]

    be17 = Qwen3ASRBackend("qwen3_asr_1.7b", tmp_path)
    audio_16k = np.ones(16000 * 2, dtype=np.float32) * 0.05

    with patch.object(Qwen3ASRBackend, "_load_model", side_effect=[(_OOMModel(), "cuda:0"), (_Good06Model(), "cuda:0")]):
        # First call OOMs on batch_size (already 1) -> falls back to 0.6B cleanly without raising!
        with patch.object(Qwen3ASRBackend, "_recommended_batch_size", return_value=1):
            out = be17.transcribe_windows(
                audio_16k,
                [(0, 1800)],
                language_pref="en",
                device="auto",
                allow_vram_downgrade=True,
            )
    assert len(out.segments) == 1
    assert "don't know" in out.segments[0].text
    assert any("降级" in w for w in out.warnings)


def test_stage_cache_reuse_on_template_and_audio_edits_and_non_overwrite_export(tmp_path: Path) -> None:
    configure_runtime_environment()
    cache_dir = default_model_cache_dir()
    work_dir = tmp_path / "stage_cache"
    out_dir = tmp_path / "exports"

    asr_calls = {"count": 0}

    def _fake_asr(*_args, **_kwargs) -> ASRRunResult:
        asr_calls["count"] += 1
        return ASRRunResult(
            segments=[
                ASRSegment(
                    start_ms=700,
                    end_ms=4600,
                    text="顾寒川，你别以为这样就能让我低头！",
                    language="zh",
                    confidence=0.92,
                    model_used="Mock-Qwen3-ASR-0.6B",
                ),
                ASRSegment(
                    start_ms=5100,
                    end_ms=8800,
                    text="苏晚晴，这三点一四亿你拿什么还？",
                    language="zh",
                    confidence=0.92,
                    model_used="Mock-Qwen3-ASR-0.6B",
                ),
            ],
            detected_language="zh",
            model_used="Mock-Qwen3-ASR-0.6B",
        )

    task = EpisodeTask(
        task_id="cache_t1",
        episode_label="第01集",
        video_path=str(FX / "第01集_粗剪.mp4"),
    )

    with (
        patch("drama_studio.pipeline.select_asr_model_id", return_value=("qwen3_asr_0.6b", [])),
        patch("drama_studio.pipeline.run_asr_transcription", side_effect=_fake_asr),
    ):
        # Run 1: full auto-subtitle pipeline (9-char 1-line)
        tpl1 = SubtitleTemplate.preset_portrait_9_16()
        res1 = process_single_episode(
            task=task,
            template=tpl1,
            audio_cfg=AudioLevelingConfig.from_mode("standard"),
            alignment_cfg=AlignmentConfig(backend="vad_fallback", language="zh"),
            export_cfg=ExportConfig(output_dir=str(out_dir)),
            cache_dir=cache_dir,
            work_dir=work_dir,
        )
        assert asr_calls["count"] == 1
        assert res1.sync_verified is True
        assert Path(res1.exported_files["srt"]).name == "第01集_修正字幕.srt"
        assert Path(res1.exported_files["wav"]).name == "第01集_修正音频.wav"

        # Run 2: change template ONLY (20 chars, 2 lines) -> ASR & ForcedAlign MUST be cache hits, NO re-ASR!
        tpl2 = SubtitleTemplate.preset_landscape_16_9()
        tpl2.max_lines = 2
        res2 = process_single_episode(
            task=task,
            template=tpl2,
            audio_cfg=AudioLevelingConfig.from_mode("standard"),
            alignment_cfg=AlignmentConfig(backend="vad_fallback", language="zh"),
            export_cfg=ExportConfig(output_dir=str(out_dir)),
            cache_dir=cache_dir,
            work_dir=work_dir,
            stage_action="relayout_only",
        )
        assert asr_calls["count"] == 1  # Zero extra ASR calls!
        assert "asr" in res2.cache_hits
        assert "forced_align" in res2.cache_hits
        # Non-overwriting export creates _v2 files!
        assert Path(res2.exported_files["srt"]).name == "第01集_v2_修正字幕.srt"
        assert Path(res1.exported_files["srt"]).is_file()

        # Run 3: change audio gain params ONLY -> ALL subtitle caches hit (asr, forced_align, layout)!
        audio_strong = AudioLevelingConfig.from_mode("strong")
        audio_strong.manual_points = [ManualGainPoint(time_ms=1500, gain_db=3.5)]
        res3 = process_single_episode(
            task=task,
            template=tpl2,
            audio_cfg=audio_strong,
            alignment_cfg=AlignmentConfig(backend="vad_fallback", language="zh"),
            export_cfg=ExportConfig(output_dir=str(out_dir), export_wav=True, export_srt=False),
            cache_dir=cache_dir,
            work_dir=work_dir,
            stage_action="relevel_only",
        )
        assert asr_calls["count"] == 1
        assert "asr" in res3.cache_hits
        assert "forced_align" in res3.cache_hits
        assert "layout" in res3.cache_hits
        assert "wav" in res3.exported_files and "srt" not in res3.exported_files
        assert Path(res3.exported_files["wav"]).name == "第01集_v3_修正音频.wav"

    # Cache quota & pruning verification
    assert get_stage_cache_size_mb(work_dir) > 0
    prune_stage_cache(work_dir, max_size_mb=1)
    freed = clear_stage_cache(work_dir)
    assert freed >= 0.0


def test_stage_actions_and_script_replacement_reuse_asr_cache(tmp_path: Path) -> None:
    """Verify asr_only / script_match_only / align_only and ASR cache reuse when changing scripts."""
    from unittest.mock import patch

    work_dir = tmp_path / "stage_work"
    cache_dir = default_model_cache_dir()
    out_dir = tmp_path / "out"
    asr_calls = {"count": 0}

    def _fake_asr(*_args, **_kwargs) -> ASRRunResult:
        asr_calls["count"] += 1
        return ASRRunResult(
            segments=[
                ASRSegment(
                    start_ms=700,
                    end_ms=4600,
                    text="顾寒川，你别以为这样就能让我低头！",
                    language="zh",
                    confidence=0.95,
                    model_used="Mock-Qwen3-ASR-0.6B",
                ),
                ASRSegment(
                    start_ms=5100,
                    end_ms=8800,
                    text="我哪怕拼尽全力，也绝不求你。",
                    language="zh",
                    confidence=0.94,
                    model_used="Mock-Qwen3-ASR-0.6B",
                ),
            ],
            detected_language="zh",
            model_used="Mock-Qwen3-ASR-0.6B",
        )

    tpl = SubtitleTemplate.preset_portrait_9_16()
    audio_cfg = AudioLevelingConfig.from_mode("standard")
    align_cfg = AlignmentConfig(backend="vad_fallback", language="zh")
    exp_cfg = ExportConfig(output_dir=str(out_dir))

    with (
        patch("drama_studio.pipeline.select_asr_model_id", return_value=("qwen3_asr_0.6b", [])),
        patch("drama_studio.pipeline.run_asr_transcription", side_effect=_fake_asr),
    ):
        # 1. ASR-only stage stops at Stage 2 without exporting files
        t_auto = EpisodeTask(task_id="ep_sa", episode_label="第01集", video_path=str(FX / "第01集_粗剪.mp4"))
        r1 = process_single_episode(
            task=t_auto,
            template=tpl,
            audio_cfg=audio_cfg,
            alignment_cfg=align_cfg,
            export_cfg=exp_cfg,
            cache_dir=cache_dir,
            work_dir=work_dir,
            stage_action="asr_only",
        )
        assert asr_calls["count"] == 1
        assert r1.exported_files == {}
        assert r1.aligner_used == "待执行强制对齐"
        assert len(r1.processed_cues) == 2

        # 2. Add script v1 and run script_match_only -> reuses ASR cache!
        scr1 = tmp_path / "script_v1.md"
        scr1.write_text("顾寒川，你别以为这样就能让我低头！\n我哪怕拼尽全力，也绝不求你。\n", encoding="utf-8")
        t_sm1 = EpisodeTask(task_id="ep_sa", episode_label="第01集", video_path=str(FX / "第01集_粗剪.mp4"), script_path=str(scr1))
        r2 = process_single_episode(
            task=t_sm1,
            template=tpl,
            audio_cfg=audio_cfg,
            alignment_cfg=align_cfg,
            export_cfg=exp_cfg,
            cache_dir=cache_dir,
            work_dir=work_dir,
            stage_action="script_match_only",
        )
        assert asr_calls["count"] == 1
        assert "asr" in r2.cache_hits
        assert r2.task_mode_used == "script_match"
        assert r2.exported_files == {}

        # 3. Replace script -> ASR STILL reused, only script_match recomputed
        scr2 = tmp_path / "script_v2.md"
        scr2.write_text("顾寒川，你别以为这样就能让我低头！\n我哪怕拼尽全力也绝对不求你。\n", encoding="utf-8")
        t_sm2 = EpisodeTask(task_id="ep_sa", episode_label="第01集", video_path=str(FX / "第01集_粗剪.mp4"), script_path=str(scr2))
        r3 = process_single_episode(
            task=t_sm2,
            template=tpl,
            audio_cfg=audio_cfg,
            alignment_cfg=align_cfg,
            export_cfg=exp_cfg,
            cache_dir=cache_dir,
            work_dir=work_dir,
            stage_action="script_match_only",
        )
        assert asr_calls["count"] == 1
        assert "asr" in r3.cache_hits
        assert "script_match" not in r3.cache_hits
        assert any("绝对不求你" in c.raw_text for c in r3.processed_cues)

        # 4. Run align_only with force_recompute=True -> reuses asr + script_match, recomputes forced_align & exports
        r4 = process_single_episode(
            task=t_sm2,
            template=tpl,
            audio_cfg=audio_cfg,
            alignment_cfg=align_cfg,
            export_cfg=exp_cfg,
            cache_dir=cache_dir,
            work_dir=work_dir,
            stage_action="align_only",
            force_recompute=True,
        )
        assert asr_calls["count"] == 1
        assert "asr" in r4.cache_hits
        assert "script_match" in r4.cache_hits
        assert "forced_align" not in r4.cache_hits
        assert Path(r4.exported_files["srt"]).is_file()
        assert Path(r4.exported_files["wav"]).is_file()


def test_batch_duplicate_episode_label_isolation() -> None:
    t1 = EpisodeTask(task_id="a", episode_label="第01集", video_path="dirA/01.mp4")
    t2 = EpisodeTask(task_id="b", episode_label="第01集", video_path="dirB/01.mp4")
    disambiguate_episode_labels([t1, t2])
    assert t1.episode_label != t2.episode_label


def test_ui_bilingual_templates_mode_switching_and_stage_strip(tmp_path: Path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from drama_studio.gui.main_window import MainWindow
    from drama_studio.gui.widgets.model_manager_dialog import ModelManagerDialog

    configure_runtime_environment()
    app = QApplication.instance() or QApplication([])
    store = ConfigStore(base_dir=tmp_path / "ui_cfg")
    win = MainWindow(store=store)

    # Switch language -> English loads 26-char 2-line template, Chinese keeps 9-char 1-line separately
    en_idx = win.combo_language.findData("en")
    win.combo_language.setCurrentIndex(en_idx)
    assert win.template_panel.current_template().language == "en"
    assert win.template_panel.current_template().max_chars_per_line == 26
    assert win.template_panel.current_template().max_lines == 2

    zh_idx = win.combo_language.findData("zh")
    win.combo_language.setCurrentIndex(zh_idx)
    assert win.template_panel.current_template().language == "zh"
    assert win.template_panel.current_template().max_chars_per_line == 9

    # Import video + bind both script and SRT -> manual mode switching preserves both paths
    vid = tmp_path / "ep01.mp4"
    srt = tmp_path / "ep01.srt"
    scr = tmp_path / "ep01.md"
    vid.write_bytes(b"stub")
    srt.write_text("1\n00:00:00,500 --> 00:00:02,000\nHello world\n", encoding="utf-8")
    scr.write_text("ALEX: Hello world!\n", encoding="utf-8")
    win.import_paths([str(vid), str(srt), str(scr)])
    assert len(win.tasks) == 1
    assert win.tasks[0].srt_path == str(srt)
    assert win.tasks[0].script_path == str(scr)

    sm_idx = win.import_panel.combo_task_mode.findData("script_match")
    win.import_panel.combo_task_mode.setCurrentIndex(sm_idx)
    assert win.tasks[0].resolve_effective_mode() == TaskMode.SCRIPT_MATCH
    assert win.tasks[0].srt_path == str(srt)
    assert win.tasks[0].script_path == str(scr)
    assert "文稿匹配" in win.lbl_stage1.text()

    from drama_studio.gui.theme import THEME_ORDER

    for theme_id in THEME_ORDER:
        win.apply_theme(theme_id, save=False)
        assert "QToolButton" in app.styleSheet()
    assert win.btn_stage_actions.menu() is not None

    # Verify panel visibility toggles, compact sub-tabs, and secondary engine settings dialog
    assert win.template_panel.tabs.count() == 2
    assert win.center_bottom_tabs.count() == 3
    win.btn_toggle_import.setChecked(False)
    app.processEvents()
    assert win.import_panel.isHidden()
    win.btn_toggle_import.setChecked(True)
    app.processEvents()
    assert not win.import_panel.isHidden()

    win.btn_toggle_template.setChecked(False)
    app.processEvents()
    assert win.template_panel.isHidden()
    win.btn_toggle_template.setChecked(True)
    app.processEvents()
    assert not win.template_panel.isHidden()

    win.audio_panel.btn_kpi_toggle.setChecked(False)
    app.processEvents()
    assert win.audio_panel.kpi_container.isHidden()
    win.audio_panel.btn_kpi_toggle.setChecked(True)

    assert win._engine_settings_dlg.combo_aligner is win.combo_aligner
    assert win._engine_settings_dlg.combo_device is win.combo_device
    win.chk_mux_mp4.setChecked(True)
    assert "合成MP4" in win.btn_engine_settings.toolTip()

    dlg = ModelManagerDialog(store, win)
    assert dlg.table.rowCount() >= 6
    dlg.close()
    dlg.deleteLater()

    app.removeEventFilter(win)
    win.audio_panel.release_sources()
    win.close()
    win.deleteLater()
    app.processEvents()


def _create_english_video_fixture(out_mp4: Path) -> bool:
    """Create a real spoken English MP4 video offline via Windows SAPI (or edge-tts fallback)."""
    import asyncio
    import subprocess
    import sys

    audio_in = out_mp4.with_suffix(".tts.wav")
    spoken_script = (
        "I don't know what you're talking about, Doctor Smith! "
        "We have to leave New York before midnight."
    )

    if sys.platform == "win32":
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like 'en-*' } | Select-Object -First 1; "
            "if (-not $v) { exit 2 }; "
            "$s.SelectVoice($v.VoiceInfo.Name); "
            "$s.Rate = -1; "
            f"$s.SetOutputToWaveFile('{str(audio_in)}'); "
            f"$s.Speak(\"{spoken_script}\"); "
            "$s.Dispose()"
        )
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not audio_in.is_file() or audio_in.stat().st_size < 4000:
            audio_in.unlink(missing_ok=True)

    if not audio_in.is_file():
        try:
            import edge_tts

            mp3_p = out_mp4.with_suffix(".mp3")

            async def _gen() -> None:
                comm = edge_tts.Communicate(spoken_script, "en-US-GuyNeural")
                await comm.save(str(mp3_p))

            asyncio.run(_gen())
            if mp3_p.is_file() and mp3_p.stat().st_size >= 1000:
                audio_in = mp3_p
        except Exception:
            return False

    if not audio_in.is_file() or audio_in.stat().st_size < 1000:
        return False

    ffmpeg_bin, _ = find_ffmpeg_binaries()
    enc_proc = subprocess.run([str(ffmpeg_bin), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    vcodec = next((c for c in ("libopenh264", "libx264", "mpeg4") if c in (enc_proc.stdout or "")), "mpeg4")
    subprocess.run(
        [
            str(ffmpeg_bin), "-y",
            "-f", "lavfi", "-i", "color=c=navy:s=1280x720:r=24:d=20",
            "-i", str(audio_in),
            "-c:v", vcodec, "-c:a", "aac", "-ar", "48000", "-shortest", str(out_mp4),
        ],
        check=True,
        capture_output=True,
    )
    return out_mp4.is_file()


def test_e2e_real_qwen3_asr_chinese_and_english_modes(tmp_path: Path) -> None:
    """Real E2E validation of Chinese + English Script Match and Auto-Sub modes with Qwen3-ASR-0.6B."""
    configure_runtime_environment()
    cdir = default_model_cache_dir()
    if not is_model_cached("qwen3_asr_0.6b", cdir):
        pytest.skip("qwen3_asr_0.6b model not cached locally")

    work_dir = tmp_path / "real_cache"

    # 1. Chinese Video + Script Match (with deleted line that must NOT appear in output)
    zh_script = tmp_path / "ep01_zh_script.md"
    zh_script.write_text(
        "场景：办公室\n"
        "顾寒川：顾寒川，你别以为这样就能让我低头！\n"
        "旁白：这一整句未在视频中出现，绝不能虚构生成字幕！\n"
        "苏晚晴：苏晚晴，这三点一四亿你拿什么还？\n"
        "顾寒川：我哪怕拼尽全力，也绝不求你。\n"
        "苏晚晴：好！\n"
        "顾寒川：那就让我们拭目以待，看看AI-2026项目到底鹿死谁手。\n",
        encoding="utf-8",
    )
    t_zh_sm = EpisodeTask(
        task_id="zh_sm",
        episode_label="中文剧本匹配",
        video_path=str(FX / "第01集_粗剪.mp4"),
        script_path=str(zh_script),
    )
    res_zh_sm = process_single_episode(
        task=t_zh_sm,
        template=SubtitleTemplate.preset_portrait_9_16(),
        audio_cfg=AudioLevelingConfig.from_mode("standard"),
        alignment_cfg=AlignmentConfig(backend="auto", asr_model="qwen3_asr_0.6b", language="zh"),
        export_cfg=ExportConfig(output_dir=str(tmp_path / "out_zh_sm")),
        cache_dir=cdir,
        work_dir=work_dir,
    )
    assert res_zh_sm.task_mode_used == "script_match"
    assert res_zh_sm.sync_verified is True
    assert not any("绝不能虚构" in c.effective_text for c in res_zh_sm.processed_cues)
    _assert_cues_valid(res_zh_sm.processed_cues, res_zh_sm.media_info.duration_ms, SubtitleTemplate.preset_portrait_9_16())
    assert Path(res_zh_sm.exported_files["srt"]).is_file()
    assert Path(res_zh_sm.exported_files["wav"]).is_file()

    # 2. Chinese Video without Script -> Auto-Generate Subtitles (reuses extracted audio/VAD/ASR cache!)
    t_zh_auto = EpisodeTask(
        task_id="zh_auto",
        episode_label="中文自动字幕",
        video_path=str(FX / "第01集_粗剪.mp4"),
    )
    res_zh_auto = process_single_episode(
        task=t_zh_auto,
        template=SubtitleTemplate.preset_portrait_9_16(),
        audio_cfg=AudioLevelingConfig.from_mode("standard"),
        alignment_cfg=AlignmentConfig(backend="auto", asr_model="qwen3_asr_0.6b", language="zh"),
        export_cfg=ExportConfig(output_dir=str(tmp_path / "out_zh_auto")),
        cache_dir=cdir,
        work_dir=work_dir,
    )
    assert res_zh_auto.task_mode_used == "asr_generate"
    assert "asr" in res_zh_auto.cache_hits
    assert len(res_zh_auto.processed_cues) >= 3
    _assert_cues_valid(res_zh_auto.processed_cues, res_zh_auto.media_info.duration_ms, SubtitleTemplate.preset_portrait_9_16())

    # 3 & 4. English Video Script Match + Auto-Generate Subtitles
    en_mp4 = tmp_path / "en_sample.mp4"
    if not _create_english_video_fixture(en_mp4):
        pytest.skip("Could not synthesize English audio fixture offline")

    en_script = tmp_path / "en_script.md"
    en_script.write_text(
        "INT. OFFICE - NIGHT\n"
        "ALEX: I don't know what you're talking about, Dr. Smith!\n"
        "ALEX: This deleted sentence was cut out and should never appear.\n"
        "ALEX: We have to leave New York before midnight.\n",
        encoding="utf-8",
    )
    en_tpl = SubtitleTemplate.preset_english_default()

    t_en_sm = EpisodeTask(
        task_id="en_sm",
        episode_label="English_Match",
        video_path=str(en_mp4),
        script_path=str(en_script),
    )
    res_en_sm = process_single_episode(
        task=t_en_sm,
        template=en_tpl,
        audio_cfg=AudioLevelingConfig.from_mode("standard"),
        alignment_cfg=AlignmentConfig(backend="auto", asr_model="qwen3_asr_0.6b", language="en"),
        export_cfg=ExportConfig(output_dir=str(tmp_path / "out_en_sm")),
        cache_dir=cdir,
        work_dir=work_dir,
    )
    assert res_en_sm.detected_language == "en"
    assert "fa-zh" not in res_en_sm.aligner_used
    assert res_en_sm.sync_verified is True
    assert not any("deleted sentence" in c.effective_text.lower() for c in res_en_sm.processed_cues)
    all_en_sm_txt = " ".join(c.effective_text for c in res_en_sm.processed_cues)
    assert "don't" in all_en_sm_txt or "talking" in all_en_sm_txt
    _assert_cues_valid(res_en_sm.processed_cues, res_en_sm.media_info.duration_ms, en_tpl)

    # English Auto-Generate (without script) + Template Relayout Cache Reuse
    t_en_auto = EpisodeTask(
        task_id="en_auto",
        episode_label="English_Auto",
        video_path=str(en_mp4),
    )
    res_en_auto = process_single_episode(
        task=t_en_auto,
        template=en_tpl,
        audio_cfg=AudioLevelingConfig.from_mode("standard"),
        alignment_cfg=AlignmentConfig(backend="auto", asr_model="qwen3_asr_0.6b", language="en"),
        export_cfg=ExportConfig(output_dir=str(tmp_path / "out_en_auto")),
        cache_dir=cdir,
        work_dir=work_dir,
    )
    assert res_en_auto.task_mode_used == "asr_generate"
    assert "asr" in res_en_auto.cache_hits
    _assert_cues_valid(res_en_auto.processed_cues, res_en_auto.media_info.duration_ms, en_tpl)

    # Change English template (26 -> 22 chars) -> relayout hits ASR & ForcedAlign caches!
    en_tpl_narrow = SubtitleTemplate.preset_english_portrait()
    res_en_relayout = process_single_episode(
        task=t_en_auto,
        template=en_tpl_narrow,
        audio_cfg=AudioLevelingConfig.from_mode("standard"),
        alignment_cfg=AlignmentConfig(backend="auto", asr_model="qwen3_asr_0.6b", language="en"),
        export_cfg=ExportConfig(output_dir=str(tmp_path / "out_en_auto")),
        cache_dir=cdir,
        work_dir=work_dir,
        stage_action="relayout_only",
    )
    assert "asr" in res_en_relayout.cache_hits
    assert "forced_align" in res_en_relayout.cache_hits
    _assert_cues_valid(res_en_relayout.processed_cues, res_en_relayout.media_info.duration_ms, en_tpl_narrow)


def test_extract_audio_av_offset_no_double_delay(tmp_path: Path) -> None:
    """Verify A/V positive audio & video start offsets align onto video PTS 0 without double delay."""
    import subprocess

    configure_runtime_environment()
    ffmpeg_bin, _ = find_ffmpeg_binaries()
    enc_proc = subprocess.run([str(ffmpeg_bin), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    vcodec = next((c for c in ("libopenh264", "libx264", "mpeg4") if c in (enc_proc.stdout or "")), "mpeg4")

    sr = 48000
    wav_beep = tmp_path / "beep_1000ms.wav"
    sig = np.zeros(sr * 3, dtype=np.float32)
    t = np.arange(int(sr * 0.5)) / sr
    sig[sr : sr + int(sr * 0.5)] = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(wav_beep), sig, sr, subtype="PCM_16")

    def _beep_bounds_ms(wav_path: Path) -> tuple[float, float]:
        y, r = sf.read(str(wav_path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        idx = np.where(np.abs(y) > 0.08)[0]
        assert len(idx) > 0, f"Expected beep signal in {wav_path}"
        return idx[0] * 1000.0 / r, idx[-1] * 1000.0 / r

    # Case 1: audio starts +0.60s after video PTS 0 -> beep at ~1600ms (NEVER double-shifted to ~2200ms)
    mp4_aud_late = tmp_path / "aud_late.mp4"
    subprocess.run(
        [
            str(ffmpeg_bin), "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=3.6",
            "-itsoffset", "0.60", "-i", str(wav_beep),
            "-c:v", vcodec, "-c:a", "aac", "-ar", "48000", str(mp4_aud_late),
        ],
        check=True,
        capture_output=True,
    )
    info_a = probe_media(mp4_aud_late)
    assert info_a.av_offset_ms > 500
    wav_out_a = extract_audio_to_wav(
        mp4_aud_late,
        tmp_path / "out_aud_late.wav",
        video_start_time_ms=info_a.video_start_time_ms,
        target_duration_ms=info_a.duration_ms,
    )
    s_a, e_a = _beep_bounds_ms(wav_out_a)
    expected_a = 1000.0 + info_a.av_offset_ms
    assert abs(s_a - expected_a) < 80.0, f"Unexpected beep onset {s_a:.1f}ms (expected ~{expected_a:.1f}ms)"
    assert abs(e_a - (expected_a + 500.0)) < 80.0

    # Case 2: video starts +0.50s after audio PTS 0 -> audio trimmed so beep shifts to ~500ms on video timeline
    mp4_vid_late = tmp_path / "vid_late.mp4"
    subprocess.run(
        [
            str(ffmpeg_bin), "-y",
            "-itsoffset", "0.50", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=3.0",
            "-i", str(wav_beep),
            "-c:v", vcodec, "-c:a", "aac", "-ar", "48000", str(mp4_vid_late),
        ],
        check=True,
        capture_output=True,
    )
    info_v = probe_media(mp4_vid_late)
    assert info_v.video_start_time_ms > 400
    wav_out_v = extract_audio_to_wav(
        mp4_vid_late,
        tmp_path / "out_vid_late.wav",
        video_start_time_ms=info_v.video_start_time_ms,
        target_duration_ms=max(100, int(round(info_v.duration_ms - info_v.video_start_time_ms))),
    )
    s_v, e_v = _beep_bounds_ms(wav_out_v)
    expected_v = 1000.0 - info_v.video_start_time_ms
    assert abs(s_v - expected_v) < 80.0, f"Unexpected beep onset {s_v:.1f}ms (expected ~{expected_v:.1f}ms)"
    assert abs(e_v - (expected_v + 500.0)) < 80.0


def test_build_exe_guards_optional_qwen_dependencies(tmp_path: Path) -> None:
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_exe.py"
    spec = importlib.util.spec_from_file_location("build_exe_mod", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 1. Minimal environment WITHOUT qwen-asr / transformers / librosa -> succeeds cleanly without those flags
    cmd_min, warns_min = mod.build_pyinstaller_cmd(
        tmp_path,
        require_qwen=False,
        has_module=lambda m: m not in {"qwen_asr", "transformers", "librosa", "huggingface_hub"},
        has_dist=lambda d: d not in {"qwen-asr", "transformers", "accelerate", "librosa", "lazy_loader", "soxr", "tokenizers", "safetensors", "huggingface-hub"},
    )
    assert "qwen_asr" not in cmd_min
    assert "qwen-asr" not in cmd_min
    assert "transformers" not in cmd_min
    assert any("qwen_asr" in w or "qwen-asr" in w for w in warns_min)

    # 2. --require-qwen fails fast with clear error when optional Qwen deps are missing
    with pytest.raises(RuntimeError, match="qwen-asr"):
        mod.build_pyinstaller_cmd(
            tmp_path,
            require_qwen=True,
            has_module=lambda _m: False,
            has_dist=lambda d: d != "qwen-asr",
        )

    # 3. Full environment WITH qwen deps -> collects qwen_asr and metadata
    cmd_full, _ = mod.build_pyinstaller_cmd(
        tmp_path,
        require_qwen=True,
        has_module=lambda _m: True,
        has_dist=lambda _d: True,
    )
    assert "qwen_asr" in cmd_full
    assert "qwen-asr" in cmd_full
    assert "--clean" in cmd_full

    # 4. Incremental build omits --clean and declares __main__ upfront for PyInstaller cache stability
    cmd_inc, _ = mod.build_pyinstaller_cmd(
        tmp_path,
        require_qwen=True,
        clean=False,
        has_module=lambda _m: True,
        has_dist=lambda _d: True,
    )
    assert "--clean" not in cmd_inc
    assert "__main__" in cmd_inc


def test_build_exe_incremental_relink_safety_guards(tmp_path: Path) -> None:
    import importlib.util
    import os

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_exe.py"
    spec = importlib.util.spec_from_file_location("build_exe_mod_relink", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pkg_dir = tmp_path / "src" / "drama_studio"
    pkg_dir.mkdir(parents=True)
    app_py = pkg_dir / "app.py"
    helper_py = pkg_dir / "helper.py"
    app_py.write_text("import sys\n", encoding="utf-8")
    helper_py.write_text("import numpy\n", encoding="utf-8")

    asset_file = tmp_path / "assets" / "app_icon.png"
    asset_file.parent.mkdir(parents=True)
    asset_file.write_bytes(b"png-v1")

    bin_file = tmp_path / "vendor" / "ffmpeg" / "bin" / "ffmpeg.exe"
    bin_file.parent.mkdir(parents=True)
    bin_file.write_bytes(b"bin-v1")

    work_dir = tmp_path / "build" / "XingJiangSubStudio"
    work_dir.mkdir(parents=True)
    base_zip = work_dir / "base_library.zip"
    base_zip.write_bytes(b"zip")

    dist_app = tmp_path / "dist" / "XingJiangSubStudio"
    internal_dir = dist_app / "_internal"
    internal_dir.mkdir(parents=True)
    (dist_app / "XingJiangSubStudio.exe").write_bytes(b"exe-v1")
    built_exe = work_dir / "XingJiangSubStudio.exe"
    built_exe.write_bytes(b"exe-v2")

    # No Analysis-00.toc yet -> must fall back to full Analysis
    ok, _ = mod._can_use_cached_relink(tmp_path, require_qwen=True)
    assert not ok

    toc_data = [[] for _ in range(20)]
    toc_data[13] = [("app", str(app_py), "PYSOURCE")]
    toc_data[14] = [
        ("drama_studio.helper", str(helper_py), "PYMODULE"),
        ("numpy", "site-packages/numpy/__init__.py", "PYMODULE"),
        ("qwen_asr", "site-packages/qwen_asr/__init__.py", "PYMODULE"),
    ]
    toc_data[15] = [
        ("ffmpeg\\bin\\ffmpeg.exe", str(bin_file), "BINARY"),
        ("PySide6\\QtCore.cp312-win_amd64.pyd", "site-packages/PySide6/QtCore.pyd", "EXTENSION"),
    ]
    toc_data[18] = [
        ("assets\\app_icon.png", str(asset_file), "DATA"),
        ("base_library.zip", str(base_zip), "DATA"),
    ]
    toc_path = work_dir / "Analysis-00.toc"
    toc_path.write_text(repr(tuple(toc_data)), encoding="utf-8")

    # Align timestamps so Analysis-00.toc is newer than initial files
    base_ts = 1_700_000_000.0
    for f in (app_py, helper_py, asset_file, bin_file, base_zip):
        os.utime(f, (base_ts, base_ts))
    os.utime(toc_path, (base_ts + 10.0, base_ts + 10.0))

    # Edit existing drama_studio module using stdlib/already-analyzed .py & .pyd imports -> fast relink allowed
    helper_py.write_text("import numpy\nimport json\nfrom PySide6.QtCore import Qt\n", encoding="utf-8")
    os.utime(helper_py, (base_ts + 20.0, base_ts + 20.0))
    ok, reason = mod._can_use_cached_relink(tmp_path, require_qwen=True)
    assert ok, reason

    # Selective sync copies rebuilt EXE and changed asset while skipping unchanged binary
    dst_bin = internal_dir / "ffmpeg" / "bin" / "ffmpeg.exe"
    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    dst_bin.write_bytes(b"bin-v1")
    os.utime(dst_bin, (base_ts + 5.0, base_ts + 5.0))
    asset_file.write_bytes(b"png-v2-changed")
    copied = mod._sync_incremental_dist_files(tmp_path, toc_path, dist_app)
    assert (dist_app / "XingJiangSubStudio.exe").read_bytes() == b"exe-v2"
    assert (internal_dir / "assets" / "app_icon.png").read_bytes() == b"png-v2-changed"
    assert copied == 2  # app_icon.png + base_library.zip (not ffmpeg.exe)

    # Adding a brand-new external dependency forces full Analysis
    helper_py.write_text("import brand_new_dep\n", encoding="utf-8")
    os.utime(helper_py, (base_ts + 30.0, base_ts + 30.0))
    ok_dep, reason_dep = mod._can_use_cached_relink(tmp_path, require_qwen=True)
    assert not ok_dep and "brand_new_dep" in reason_dep

    # Restoring known import re-enables relink; adding a new .py module forces full Analysis
    helper_py.write_text("import numpy\n", encoding="utf-8")
    os.utime(helper_py, (base_ts + 35.0, base_ts + 35.0))
    (pkg_dir / "new_module.py").write_text("x = 1\n", encoding="utf-8")
    ok_new, reason_new = mod._can_use_cached_relink(tmp_path, require_qwen=True)
    assert not ok_new and "added or removed" in reason_new


def test_batched_qwen_asr_and_aligner_preserve_order_and_recover_oom(tmp_path: Path) -> None:
    """Verify duration-sorted ASR & Aligner batching restore chronological order and halve batch on OOM."""
    from types import SimpleNamespace
    from drama_studio.align.qwen_aligner import Qwen3AlignerBackend

    # 1. Duration-sorted ASR batching restores original window timeline order
    d06 = tmp_path / "qwen3_asr_0.6b"
    d06.mkdir()
    (d06 / "config.json").write_text("{}", encoding="utf-8")
    (d06 / "model.safetensors").write_bytes(b"0" * 1_500_000)

    seen_lens: list[int] = []

    class _FakeASR:
        max_inference_batch_size = 8
        max_new_tokens = 192

        def transcribe(self, audio, **_kwargs):
            seen_lens.extend(len(a[0]) for a in audio)
            return [SimpleNamespace(language="Chinese", text=f"字{len(a[0])}") for a in audio]

    be_asr = Qwen3ASRBackend("qwen3_asr_0.6b", tmp_path)
    audio_16k = np.zeros(16000 * 5, dtype=np.float32)
    wins = [(0, 2000), (2100, 2600), (2800, 4200)]  # lengths: 2.0s, 0.5s, 1.4s
    with (
        patch.object(Qwen3ASRBackend, "_load_model", return_value=(_FakeASR(), "cuda:0")),
        patch.object(Qwen3ASRBackend, "_recommended_batch_size", return_value=8),
    ):
        res = be_asr.transcribe_windows(audio_16k, wins, language_pref="zh", device="cuda")
    # Sorted shortest -> longest during inference, but returned in original window order!
    assert seen_lens == [500 * 16, 1400 * 16, 2000 * 16]
    assert [s.start_ms for s in res.segments] == [0, 2100, 2800]
    assert [s.text for s in res.segments] == [f"字{2000*16}", f"字{500*16}", f"字{1400*16}"]

    # 2. Batched ForcedAligner sorts by length, halves batch on OOM, and restores window order
    da = tmp_path / "qwen3_aligner"
    da.mkdir()
    (da / "config.json").write_text("{}", encoding="utf-8")
    (da / "model.safetensors").write_bytes(b"0" * 1_500_000)

    batch_sizes_tried: list[int] = []

    class _FakeAligner:
        def align(self, audio, text, language):
            batch_sizes_tried.append(len(audio))
            if len(audio) > 2:
                raise RuntimeError("CUDA out of memory")
            out = []
            for t in text:
                items = [SimpleNamespace(text=ch, start_time=0.05 + k * 0.08, end_time=0.13 + k * 0.08) for k, ch in enumerate(t)]
                out.append(SimpleNamespace(items=items))
            return out

    be_al = Qwen3AlignerBackend(tmp_path)
    al_items = [(0, 2000, "你好吗"), (2100, 2600, "嗯"), (2800, 4200, "很好"), (4300, 4900, "行")]
    with (
        patch.object(Qwen3AlignerBackend, "_get_model", return_value=(_FakeAligner(), "cuda:0")),
        patch.object(Qwen3AlignerBackend, "_recommended_align_batch_size", return_value=4),
    ):
        outs = be_al.align_utterances_batch(audio_16k, al_items, language="zh", device="cuda")
    assert batch_sizes_tried[0] == 4 and 2 in batch_sizes_tried
    assert ["".join(c.char for c in o.char_timings) for o in outs] == ["你好吗", "嗯", "很好", "行"]
    assert outs[0].char_timings[0].start_ms == 50
    assert outs[1].char_timings[0].start_ms == 2150


def test_cross_project_same_episode_label_never_overwrites_even_in_overwrite_mode(tmp_path: Path) -> None:
    from drama_studio.models import AudioAnalysisReport, MediaInfo, Orientation, PipelineResult

    out_dir = tmp_path / "shared_exports"
    proj_a_vid = tmp_path / "项目A" / "第01集.mp4"
    proj_b_vid = tmp_path / "项目B" / "第01集.mp4"
    proj_a_vid.parent.mkdir(parents=True)
    proj_b_vid.parent.mkdir(parents=True)
    proj_a_vid.write_bytes(b"a")
    proj_b_vid.write_bytes(b"b")

    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    sf.write(str(wav_a), np.zeros(4800, dtype=np.float32), 48000, subtype="PCM_16")
    sf.write(str(wav_b), np.ones(4800, dtype=np.float32) * 0.1, 48000, subtype="PCM_16")

    def _make_result(vid: Path, wav: Path, text: str) -> PipelineResult:
        return PipelineResult(
            episode_label="第01集",
            media_info=MediaInfo(
                path=str(vid),
                file_name=vid.name,
                format_name="mp4",
                duration_ms=2000,
                width=1080,
                height=1920,
                fps=30.0,
                fps_str="30/1",
                aspect_ratio="9:16",
                orientation=Orientation.PORTRAIT,
            ),
            raw_cues=[SubtitleCue(index=1, start_ms=100, end_ms=1000, raw_text=text, display_text=text)],
            processed_cues=[SubtitleCue(index=1, start_ms=100, end_ms=1000, raw_text=text, display_text=text)],
            anomalies=[],
            audio_report=AudioAnalysisReport(
                sample_rate=48000, channels=1, duration_ms=2000, speech_ratio=0.5,
                before_lufs=-20.0, after_lufs=-16.0,
                before_short_term_min_lufs=-22.0, before_short_term_max_lufs=-18.0,
                after_short_term_min_lufs=-17.0, after_short_term_max_lufs=-15.0,
                before_lra_lu=4.0, after_lra_lu=2.0,
                before_peak_dbfs=-3.0, after_peak_dbfs=-2.0,
                before_true_peak_dbtp=-3.0, after_true_peak_dbtp=-2.0,
                noise_floor_dbfs=-55.0, estimated_snr_db=25.0,
                max_applied_gain_db=4.0, min_applied_gain_db=0.0,
            ),
            extracted_wav_path=str(wav),
            processed_wav_path=str(wav),
        )

    cfg_overwrite = ExportConfig(output_dir=str(out_dir), overwrite_policy="overwrite")
    exp_a = export_episode_outputs(_make_result(proj_a_vid, wav_a, "项目A第一句"), cfg_overwrite)
    exp_b = export_episode_outputs(_make_result(proj_b_vid, wav_b, "项目B第一句"), cfg_overwrite)

    srt_a = Path(exp_a["srt"])
    srt_b = Path(exp_b["srt"])
    assert srt_a != srt_b
    assert srt_a.is_file() and srt_b.is_file()
    assert "项目A第一句" in srt_a.read_text(encoding="utf-8")
    assert "项目B第一句" in srt_b.read_text(encoding="utf-8")
    assert "项目B" in srt_b.name

    # Re-exporting the SAME source video A in overwrite mode overwrites A cleanly without creating extra files
    exp_a2 = export_episode_outputs(_make_result(proj_a_vid, wav_a, "项目A更新版"), cfg_overwrite)
    assert Path(exp_a2["srt"]) == srt_a
    assert "项目A更新版" in srt_a.read_text(encoding="utf-8")
    assert "项目B第一句" in srt_b.read_text(encoding="utf-8")
