"""Tests for intelligent voice loudness leveling across anomalies, modes, and manual gain points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from drama_studio.audio.leveler import _lookahead_true_peak_limiter, process_voice_leveling
from drama_studio.audio.loudness import measure_true_peak_dbtp
from drama_studio.models import AudioLevelingConfig, ManualGainPoint

FX = Path(__file__).resolve().parent / "fixtures"


def _clip_rms_db(sig: np.ndarray, sr: int, start_ms: int, end_ms: int) -> float:
    s = int(start_ms * sr / 1000)
    e = int(end_ms * sr / 1000)
    seg = sig[s:e]
    if seg.ndim > 1:
        seg = np.mean(seg, axis=1)
    return float(20.0 * np.log10(max(1e-8, float(np.sqrt(np.mean(seg ** 2))))))


def test_leveling_fixes_wild_volume_anomalies_without_speed_change(tmp_path: Path) -> None:
    gt = json.loads((FX / "ground_truth_ep01.json").read_text(encoding="utf-8"))
    orig, sr = sf.read(str(FX / "ep01_audio.wav"), dtype="float64")
    orig_spread = max(_clip_rms_db(orig, sr, g["start_ms"], g["end_ms"]) for g in gt) - min(
        _clip_rms_db(orig, sr, g["start_ms"], g["end_ms"]) for g in gt
    )
    assert orig_spread > 15.0  # u3 whisper vs u4 shout has ~19.8 dB spread

    # Standard mode
    out_std = process_voice_leveling(
        FX / "ep01_audio.wav",
        tmp_path / "std.wav",
        AudioLevelingConfig.from_mode("standard"),
        bit_depth=24,
    )
    assert len(out_std.processed_signal) == len(orig)  # Zero speed change / sample-exact sync
    assert out_std.report.after_true_peak_dbtp <= -1.3
    assert -17.5 <= out_std.report.after_lufs <= -14.5

    std_levels = [_clip_rms_db(out_std.processed_signal, sr, g["start_ms"], g["end_ms"]) for g in gt]
    std_spread = max(std_levels) - min(std_levels)
    assert std_spread < orig_spread * 0.5
    # Emotion preservation: shout u4 (idx 3) stays gently louder than whisper u3 (idx 2)
    assert std_levels[3] > std_levels[2]

    # Strong mode tightens spread even further
    out_strong = process_voice_leveling(
        FX / "ep01_audio.wav",
        tmp_path / "strong.wav",
        AudioLevelingConfig.from_mode("strong"),
        bit_depth=24,
    )
    strong_levels = [_clip_rms_db(out_strong.processed_signal, sr, g["start_ms"], g["end_ms"]) for g in gt]
    assert (max(strong_levels) - min(strong_levels)) <= std_spread + 0.5


def test_manual_gain_point_and_bgm_risk_detection(tmp_path: Path) -> None:
    gt = json.loads((FX / "ground_truth_ep01.json").read_text(encoding="utf-8"))
    u2 = gt[1]
    mid_ms = (u2["start_ms"] + u2["end_ms"]) // 2

    cfg_base = AudioLevelingConfig.from_mode("standard")
    out_base = process_voice_leveling(FX / "ep01_audio.wav", tmp_path / "base.wav", cfg_base)

    cfg_boost = AudioLevelingConfig.from_mode("standard")
    cfg_boost.manual_points = [ManualGainPoint(time_ms=mid_ms, gain_db=6.0, radius_ms=400)]
    out_boost = process_voice_leveling(FX / "ep01_audio.wav", tmp_path / "boost.wav", cfg_boost)

    sr = out_base.report.sample_rate
    rms_base = _clip_rms_db(out_base.processed_signal, sr, mid_ms - 180, mid_ms + 180)
    rms_boost = _clip_rms_db(out_boost.processed_signal, sr, mid_ms - 180, mid_ms + 180)
    assert rms_boost > rms_base + 3.0

    # Mix heavy BGM noise floor and verify risk detection fires
    orig, sr = sf.read(str(FX / "ep01_audio.wav"), dtype="float64")
    rng = np.random.default_rng(7)
    noisy = np.clip(orig + rng.normal(0.0, 0.04, orig.shape), -0.99, 0.99)
    noisy_wav = tmp_path / "bgm_noisy.wav"
    sf.write(str(noisy_wav), noisy, sr, subtype="PCM_16")
    out_noisy = process_voice_leveling(noisy_wav, tmp_path / "bgm_out.wav", AudioLevelingConfig.from_mode("standard"))
    assert out_noisy.report.bgm_or_sfx_risk is True
    assert len(out_noisy.report.risk_messages) >= 1


def test_true_peak_limiter_guarantees_intersample_ceiling_strictly(tmp_path: Path) -> None:
    sr = 48000
    n = sr  # 1 second
    t = np.arange(n, dtype=np.float64) / sr
    # Construct a classic high-ISP signal: 12 kHz cosine sampled at pi/4 phase has sample peak = 0.707 * amp
    # while true inter-sample peak = 1.0 * amp (+3.01 dB above sample peak!)
    amp = 10.0 ** (-1.5 / 20.0) * 1.38  # sample peak ~ -1.7 dBFS, true peak ~ +1.3 dBTP
    isp_sig = amp * np.cos(2.0 * np.pi * 12000.0 * t + np.pi / 4.0)
    sample_peak_db = 20.0 * np.log10(float(np.max(np.abs(isp_sig))))
    raw_tp_db = measure_true_peak_dbtp(isp_sig, oversample=4)
    assert sample_peak_db < -1.55
    assert raw_tp_db > 0.5  # Severe inter-sample overshoot despite sample peak below ceiling!

    limited = _lookahead_true_peak_limiter(isp_sig, sr, ceiling_dbtp=-1.5)
    assert len(limited) == n
    assert measure_true_peak_dbtp(limited, oversample=4) <= -1.5

    # Also verify full stereo process_voice_leveling respects ceiling strictly
    stereo = np.stack([isp_sig, -isp_sig], axis=1)
    in_wav = tmp_path / "isp_stereo.wav"
    out_wav = tmp_path / "isp_stereo_out.wav"
    sf.write(str(in_wav), stereo, sr, subtype="PCM_24")
    cfg = AudioLevelingConfig.from_mode("standard")
    cfg.true_peak_ceiling_dbtp = -1.5
    res = process_voice_leveling(in_wav, out_wav, cfg, bit_depth=24)
    assert len(res.processed_signal) == n
    assert res.report.after_true_peak_dbtp <= -1.5
    reloaded, _ = sf.read(str(out_wav), dtype="float64")
    assert measure_true_peak_dbtp(reloaded, oversample=4) <= -1.5
