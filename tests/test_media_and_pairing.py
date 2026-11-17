"""Tests for FFmpeg probing, multi-track extraction, and batch episode pairing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from drama_studio.app import configure_runtime_environment
from drama_studio.media.ffmpeg_wrapper import extract_audio_to_wav, probe_media
from drama_studio.media.pairing import assign_script_files_to_tasks, pair_media_files
from drama_studio.models import Orientation

FX = Path(__file__).resolve().parent / "fixtures"


def test_batch_pairing_mp4_mov_mkv_srt() -> None:
    configure_runtime_environment()
    tasks, unmatched = pair_media_files([
        FX / "第03集_多音轨.mkv",
        FX / "第01集_粗剪.mp4",
        FX / "第02集_粗剪.srt",
        FX / "第02集_粗剪.mov",
        FX / "第01集_粗剪.srt",
        FX / "第03集_多音轨.srt",
    ])
    assert unmatched == []
    assert [t.episode_label for t in tasks] == ["第01集", "第02集", "第03集"]
    for t in tasks:
        assert Path(t.video_path).is_file()
        assert Path(t.srt_path).is_file()


def test_probe_portrait_landscape_and_multitrack() -> None:
    configure_runtime_environment()
    info1 = probe_media(FX / "第01集_粗剪.mp4")
    assert info1.width == 1080 and info1.height == 1920
    assert info1.aspect_ratio == "9:16"
    assert info1.orientation == Orientation.PORTRAIT
    assert abs(info1.fps - 30.0) < 0.1
    assert info1.duration_ms > 19000

    info2 = probe_media(FX / "第02集_粗剪.mov")
    assert info2.width == 1920 and info2.height == 1080
    assert info2.aspect_ratio == "16:9"
    assert info2.orientation == Orientation.LANDSCAPE
    assert abs(info2.fps - 24.0) < 0.1

    info3 = probe_media(FX / "第03集_多音轨.mkv")
    assert len(info3.audio_tracks) == 2
    assert any("检测到 2 条音轨" in w for w in info3.warnings)


def test_extract_specific_audio_track(tmp_path: Path) -> None:
    configure_runtime_environment()
    # Track 0 in MKV is 440Hz sine tone; Track 1 is Mandarin speech
    wav_t0 = extract_audio_to_wav(FX / "第03集_多音轨.mkv", tmp_path / "t0.wav", audio_stream_index=0)
    wav_t1 = extract_audio_to_wav(FX / "第03集_多音轨.mkv", tmp_path / "t1.wav", audio_stream_index=1)

    s0, sr0 = sf.read(str(wav_t0), dtype="float32")
    s1, sr1 = sf.read(str(wav_t1), dtype="float32")
    assert sr0 == 48000 and sr1 == 48000
    rms0 = float(np.sqrt(np.mean(s0 ** 2)))
    rms1 = float(np.sqrt(np.mean(s1 ** 2)))
    # Track 1 speech has noticeably higher/varied RMS than quiet 0.02 tone on track 0
    assert rms1 > rms0


def test_script_md_auto_pairing_and_late_drop(tmp_path: Path) -> None:
    configure_runtime_environment()
    vid = tmp_path / "125-03-01-59b96be0.mp4"
    srt = tmp_path / "125-03-01-59b96be0.srt"
    vid.write_bytes(b"mp4stub")
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n祁王世子\n", encoding="utf-8")

    scr_dir = tmp_path / "台词提取_第124-132集_20261116"
    scr_dir.mkdir()
    (scr_dir / "README.md").write_text("# 忽略说明文档", encoding="utf-8")
    ep124 = scr_dir / "第124集.md"
    ep125 = scr_dir / "第125集.md"
    ep124.write_text("宋砚舟\n\n上一集。\n", encoding="utf-8")
    ep125.write_text("宋砚舟、顾玉冗\n\n祁王世子，思念罗姑娘过度。\n", encoding="utf-8")

    tasks, unmatched = pair_media_files([vid, srt, scr_dir])
    assert unmatched == []
    assert len(tasks) == 1
    assert tasks[0].srt_path == str(srt)
    assert tasks[0].script_path == str(ep125)

    # Late drop of script folder onto already-imported tasks
    tasks[0].script_path = ""
    matched = assign_script_files_to_tasks(tasks, [scr_dir])
    assert matched == 1
    assert tasks[0].script_path == str(ep125)
