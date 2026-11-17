"""End-to-end pipeline tests with FunASR fa-zh, Qwen3-ForcedAligner-0.6B, Resolve export, and batch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf

from drama_studio.align.model_manager import is_model_cached
from drama_studio.app import configure_runtime_environment
from drama_studio.config import default_model_cache_dir
from drama_studio.media.ffmpeg_wrapper import probe_media
from drama_studio.models import (
    AlignmentConfig,
    AnomalyType,
    AudioLevelingConfig,
    EpisodeTask,
    ExportConfig,
    SubtitleTemplate,
)
from drama_studio.nlp.punctuation import extract_canonical_spoken_sequence
from drama_studio.nlp.segmenter import measure_text_width, width_fits_limit
from drama_studio.nlp.srt_io import load_srt
from drama_studio.pipeline import process_single_episode

FX = Path(r"d:\自动化\tests\fixtures")


def _run_and_verify_backend(backend_name: str, tmp_path: Path, max_lines: int, max_chars: int) -> None:
    configure_runtime_environment()
    cache_dir = default_model_cache_dir()

    tpl = (
        SubtitleTemplate.preset_portrait_9_16()
        if max_chars <= 12
        else SubtitleTemplate.preset_landscape_16_9()
    )
    tpl.max_chars_per_line = max_chars
    tpl.max_lines = max_lines

    out_dir = tmp_path / f"out_{backend_name}_{max_lines}L"
    task = EpisodeTask(
        task_id=f"ep01_{backend_name}",
        episode_label="第01集",
        video_path=str(FX / "第01集_粗剪.mp4"),
        srt_path=str(FX / "第01集_粗剪.srt"),
    )
    res = process_single_episode(
        task=task,
        template=tpl,
        audio_cfg=AudioLevelingConfig.from_mode("standard"),
        alignment_cfg=AlignmentConfig(backend=backend_name, device="auto"),  # type: ignore[arg-type]
        export_cfg=ExportConfig(
            output_dir=str(out_dir),
            export_wav=True,
            export_srt=True,
            export_report_json=True,
            export_muxed_mp4=(max_lines == 1 and backend_name == "funasr"),
        ),
        cache_dir=cache_dir,
        work_dir=tmp_path / f"work_{backend_name}_{max_lines}L",
    )

    assert res.text_integrity_ok is True
    assert len(res.processed_cues) >= 5

    # Verify raw spoken sequence is preserved 100%
    raw_seq = extract_canonical_spoken_sequence(" ".join(c.raw_text for c in res.raw_cues))
    out_seq = extract_canonical_spoken_sequence(" ".join(c.effective_text for c in res.processed_cues))
    assert raw_seq == out_seq

    # Verify line count and char width rules
    for cue in res.processed_cues:
        lines = cue.lines()
        assert 1 <= len(lines) <= max_lines
        for ln in lines:
            assert width_fits_limit(measure_text_width(ln, tpl.width_mode), max_chars)

    # Verify aligned first cue snaps close to true speech start (~750-950ms, NOT raw shifted 300ms!)
    gt = json.loads((FX / "ground_truth_ep01.json").read_text(encoding="utf-8"))
    assert abs(res.processed_cues[0].start_ms - gt[0]["start_ms"]) < 350

    # Verify exported Resolve files exist and are valid
    wav_out = Path(res.exported_files["wav"])
    srt_out = Path(res.exported_files["srt"])
    rep_out = Path(res.exported_files["report"])
    assert wav_out.is_file() and wav_out.name == "第01集_修正音频.wav"
    assert srt_out.is_file() and srt_out.name == "第01集_修正字幕.srt"
    assert rep_out.is_file() and rep_out.name == "第01集_修正报告.json"

    # WAV exact duration sync check
    sig, sr = sf.read(str(wav_out), dtype="float32")
    assert sr == 48000
    wav_dur_ms = int(round(len(sig) * 1000.0 / sr))
    assert abs(wav_dur_ms - res.media_info.duration_ms) <= 60

    # SRT DaVinci Resolve check (no BOM, LF only, non-overlapping)
    srt_bytes = srt_out.read_bytes()
    assert not srt_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in srt_bytes
    reloaded = load_srt(srt_out)
    assert len(reloaded) == len(res.processed_cues)

    if "mp4" in res.exported_files:
        mp4_p = Path(res.exported_files["mp4"])
        assert mp4_p.is_file()
        mp4_info = probe_media(mp4_p)
        assert mp4_info.width == 1080 and mp4_info.height == 1920


def test_e2e_funasr_1line_and_2line(tmp_path: Path) -> None:
    if not is_model_cached("funasr_fa", default_model_cache_dir()):
        pytest.skip("funasr_fa model not cached locally")
    _run_and_verify_backend("funasr", tmp_path, max_lines=1, max_chars=9)
    _run_and_verify_backend("funasr", tmp_path, max_lines=2, max_chars=14)


def test_e2e_qwen3_1line_and_2line(tmp_path: Path) -> None:
    if not is_model_cached("qwen3_aligner", default_model_cache_dir()):
        pytest.skip("qwen3_aligner model not cached locally")
    _run_and_verify_backend("qwen3", tmp_path, max_lines=1, max_chars=10)
    _run_and_verify_backend("qwen3", tmp_path, max_lines=2, max_chars=16)


def test_silent_region_anomaly_detection(tmp_path: Path) -> None:
    configure_runtime_environment()
    # Create SRT that claims dialogue exists inside pure silence at the very beginning (0..400ms) or muted audio
    silent_wav = tmp_path / "silent.wav"
    import numpy as np
    import subprocess

    sf.write(str(silent_wav), np.zeros((48000 * 4, 2), dtype=np.float32), 48000, subtype="PCM_16")
    silent_mp4 = tmp_path / "silent.mp4"
    subprocess.run(
        [
            r"d:\自动化\vendor\ffmpeg\bin\ffmpeg.exe", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=720x1280:r=25:d=4",
            "-i", str(silent_wav),
            "-c:v", "libopenh264", "-c:a", "aac", "-shortest", str(silent_mp4),
        ],
        check=True,
        capture_output=True,
    )
    bogus_srt = tmp_path / "silent.srt"
    bogus_srt.write_text("1\n00:00:00,500 --> 00:00:02,000\n这段台词在静音区\n", encoding="utf-8")

    task = EpisodeTask(task_id="sil", episode_label="静音测试", video_path=str(silent_mp4), srt_path=str(bogus_srt))
    res = process_single_episode(
        task=task,
        template=SubtitleTemplate.preset_portrait_9_16(),
        audio_cfg=AudioLevelingConfig.from_mode("standard"),
        alignment_cfg=AlignmentConfig(backend="vad_fallback"),
        export_cfg=ExportConfig(output_dir=str(tmp_path / "out")),
        cache_dir=default_model_cache_dir(),
        work_dir=tmp_path / "work",
    )
    assert any(a.anomaly_type in {AnomalyType.SILENT_REGION, AnomalyType.LOW_CONFIDENCE} for a in res.anomalies)


def test_batch_three_episodes_worker(tmp_path: Path) -> None:
    """Verify multi-episode batch ProcessingWorker across MP4, MOV, and multi-track MKV."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from drama_studio.gui.workers import ProcessingWorker
    from drama_studio.media.pairing import pair_media_files

    configure_runtime_environment()
    app = QApplication.instance() or QApplication([])

    tasks, _ = pair_media_files([
        FX / "第01集_粗剪.mp4", FX / "第01集_粗剪.srt",
        FX / "第02集_粗剪.mov", FX / "第02集_粗剪.srt",
        FX / "第03集_多音轨.mkv", FX / "第03集_多音轨.srt",
    ])
    assert len(tasks) == 3
    # Ep03 track 1 holds Mandarin speech; track 0 holds 440Hz tone
    tasks[2].selected_audio_index = 1

    tpl_916 = SubtitleTemplate.preset_portrait_9_16()
    tpl_169 = SubtitleTemplate.preset_landscape_16_9()
    jobs = [
        (tasks[0], tpl_916, AudioLevelingConfig.from_mode("standard")),
        (tasks[1], tpl_169, AudioLevelingConfig.from_mode("light")),
        (tasks[2], tpl_169, AudioLevelingConfig.from_mode("strong")),
    ]

    out_dir = tmp_path / "batch_out"
    worker = ProcessingWorker(
        jobs=jobs,
        alignment_cfg=AlignmentConfig(backend="funasr" if is_model_cached("funasr_fa", default_model_cache_dir()) else "vad_fallback"),
        export_cfg=ExportConfig(output_dir=str(out_dir), export_wav=True, export_srt=True, export_report_json=True),
        cache_dir=default_model_cache_dir(),
        work_root=tmp_path / "batch_work",
    )
    completed: dict[str, object] = {}
    failed: list[str] = []
    worker.taskCompleted.connect(lambda tid, r: completed.__setitem__(tid, r))
    worker.taskFailed.connect(lambda tid, e: failed.append(f"{tid}: {e}"))
    worker.run()  # Run synchronously in test thread for deterministic assertion

    assert not failed, failed
    assert len(completed) == 3
    for label in ("第01集", "第02集", "第03集"):
        assert (out_dir / f"{label}_修正音频.wav").is_file()
        assert (out_dir / f"{label}_修正字幕.srt").is_file()
        assert (out_dir / f"{label}_修正报告.json").is_file()


def test_acoustic_boundary_refinement_and_local_drift_windowing() -> None:
    """Verify 10ms RMS refinement fixes smeared silence onsets and VAD valleys recover local 1.5s drift."""
    import numpy as np
    from drama_studio.align.engine import _group_cues_into_windows, _refine_char_boundaries_with_acoustics
    from drama_studio.models import CharTimestamp, SubtitleCue

    # Build 16kHz audio with two bursts separated by silence: [800..1500ms] and [1950..2700ms]
    sr = 16000
    audio = np.zeros(4 * sr, dtype=np.float32)
    t1 = np.arange(int(0.70 * sr)) / sr
    audio[int(0.80 * sr) : int(1.50 * sr)] = 0.25 * np.sin(2 * np.pi * 220.0 * t1)
    t2 = np.arange(int(0.75 * sr)) / sr
    audio[int(1.95 * sr) : int(2.70 * sr)] = 0.25 * np.sin(2 * np.pi * 240.0 * t2)

    # Simulate Qwen3 80ms late start (880 vs 800) + smeared silence onto next char (1520..2240 instead of 1950)
    cts = [
        CharTimestamp(char="顾", start_ms=880, end_ms=1120, confidence=0.94),
        CharTimestamp(char="川", start_ms=1120, end_ms=1520, confidence=0.94),
        CharTimestamp(char="你", start_ms=1520, end_ms=2240, confidence=0.94),
        CharTimestamp(char="好", start_ms=2240, end_ms=2950, confidence=0.94),
    ]
    _refine_char_boundaries_with_acoustics(cts, audio, 4000)
    assert cts[0].start_ms <= 820  # Pulled back onto ~800ms onset
    assert cts[2].start_ms >= 1900  # Trimmed smeared silence before 1950ms speech onset!
    assert cts[2].start_ms - cts[1].end_ms >= 350  # Recovers real pause for segmentation!
    assert cts[3].end_ms <= 2780  # Trims trailing silence overshoot after 2700ms

    # Local VAD windowing test: cue 2 shifted +1500ms relative to actual speech VAD (4200..5200)
    cues = [
        SubtitleCue(index=1, start_ms=500, end_ms=1500, raw_text="第一句"),
        SubtitleCue(index=2, start_ms=5800, end_ms=6800, raw_text="局部漂移句"),
    ]
    wins = _group_cues_into_windows(cues, vad_spans=[(500, 1500), (4200, 5200)], anchored_bounds={}, global_offset_ms=0, duration_ms=8000)
    assert len(wins) == 2
    assert wins[1].window_start_ms <= 4250  # Recovers real speech at 4200 despite 1600ms local SRT drift

