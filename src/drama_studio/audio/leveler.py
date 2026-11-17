"""Intelligent 48kHz broadcast vocal loudness leveling engine without cutting or time-stretching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt
import soundfile as sf

from drama_studio.audio.loudness import measure_loudness, measure_speech_gated_lufs
from drama_studio.audio.vad import detect_speech_segments, spans_to_sample_mask
from drama_studio.models import (
    AudioAnalysisReport,
    AudioLevelingConfig,
    ManualGainPoint,
    SubtitleCue,
)


@dataclass
class LevelingOutput:
    processed_signal: np.ndarray
    sample_rate: int
    report: AudioAnalysisReport
    vad_spans_ms: list[tuple[int, int]]
    gain_curve_times_ms: list[int]
    gain_curve_db: list[float]
    waveform_times_ms: list[int]
    waveform_before_peaks: list[float]
    waveform_after_peaks: list[float]


def _highpass_zero_phase(signal: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 5.0 or len(signal) < 32 or cutoff_hz >= sample_rate * 0.45:
        return signal.copy()
    sos = butter(2, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    if signal.ndim == 1:
        return sosfiltfilt(sos, signal)
    out = np.empty_like(signal)
    for ch in range(signal.shape[1]):
        out[:, ch] = sosfiltfilt(sos, signal[:, ch])
    return out


def _sliding_rms(mono: np.ndarray, sample_rate: int, window_ms: float) -> np.ndarray:
    win = max(1, int(round(sample_rate * window_ms / 1000.0)))
    sq = mono.astype(np.float64) ** 2
    mean_sq = uniform_filter1d(sq, size=win, mode="nearest")
    return np.sqrt(np.maximum(mean_sq, 1e-16))


def _interpolate_silence_gain(gain_db: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    """Smoothly interpolate gain across silence gaps so room noise / breaths don't pump."""
    result = gain_db.copy()
    speech_idx = np.where(active_mask)[0]
    if len(speech_idx) == 0:
        return np.zeros_like(gain_db)

    silence_idx = np.where(~active_mask)[0]
    if len(silence_idx) == 0:
        return result

    first = int(speech_idx[0])
    last = int(speech_idx[-1])
    # Gently taper leading/trailing non-speech towards safe neutral / first speech hold
    result[:first] = np.clip(gain_db[first], -6.0, 6.0)
    result[last + 1 :] = np.clip(gain_db[last], -6.0, 6.0)

    diff = np.diff(active_mask.astype(np.int8))
    gap_starts = np.where(diff == -1)[0] + 1
    gap_ends = np.where(diff == 1)[0]

    for gs in gap_starts:
        ge_candidates = gap_ends[gap_ends >= gs]
        if len(ge_candidates) == 0:
            continue
        ge = int(ge_candidates[0])
        if gs > 0 and ge + 1 < len(result):
            g_left = float(result[gs - 1])
            g_right = float(result[ge + 1])
            gap_len = ge - gs + 1
            # For long pauses (>1.2s), dip slightly in the middle to prevent background noise swelling
            interp = np.linspace(g_left, g_right, gap_len, dtype=np.float64)
            if gap_len > 24000:  # >0.5s at 48kHz
                mid_dip = np.sin(np.linspace(0.0, np.pi, gap_len)) * min(4.0, max(0.0, max(g_left, g_right) * 0.35))
                interp = interp - mid_dip
            result[gs : ge + 1] = interp

    return result


def _apply_manual_points(
    gain_db: np.ndarray,
    sample_rate: int,
    points: list[ManualGainPoint],
) -> np.ndarray:
    if not points:
        return gain_db
    out = gain_db.copy()
    n = len(out)
    for pt in points:
        center = int(round(pt.time_ms * sample_rate / 1000.0))
        radius = max(int(sample_rate * 0.05), int(round(pt.radius_ms * sample_rate / 1000.0)))
        s = max(0, center - radius)
        e = min(n, center + radius + 1)
        if e <= s:
            continue
        idx = np.arange(s, e)
        dist = np.abs(idx - center) / float(radius)
        weight = 0.5 * (1.0 + np.cos(np.pi * np.clip(dist, 0.0, 1.0)))
        out[s:e] += pt.gain_db * weight
    return out


def _soft_knee_compressor(
    signal: np.ndarray,
    mono: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    ratio: float,
    knee_db: float = 4.0,
) -> np.ndarray:
    """Linked multi-channel soft-knee downward compressor to tame abrupt vocal spikes."""
    if ratio <= 1.05 or len(mono) == 0:
        return signal

    # Fast RMS detector (~15ms attack / 60ms release approximation via fast sliding RMS)
    env = _sliding_rms(mono, sample_rate, 25.0)
    env_db = 20.0 * np.log10(np.maximum(env, 1e-8))

    half_knee = knee_db * 0.5
    over = env_db - threshold_db
    reduction_db = np.zeros_like(env_db)

    in_knee = (over > -half_knee) & (over < half_knee)
    above_knee = over >= half_knee

    slope = 1.0 - 1.0 / ratio
    reduction_db[in_knee] = slope * ((over[in_knee] + half_knee) ** 2) / (2.0 * knee_db)
    reduction_db[above_knee] = slope * over[above_knee]

    # Smooth gain reduction slightly (attack ~8ms, release ~80ms emulated via asymmetric smoothing)
    smooth_samples = max(1, int(sample_rate * 0.03))
    reduction_db = uniform_filter1d(reduction_db, size=smooth_samples, mode="nearest")
    gain_lin = 10.0 ** (-reduction_db / 20.0)

    if signal.ndim == 1:
        return signal * gain_lin
    return signal * gain_lin[:, np.newaxis]


def _lookahead_true_peak_limiter(
    signal: np.ndarray,
    sample_rate: int,
    ceiling_dbtp: float = -1.5,
    lookahead_ms: float = 5.0,
    release_ms: float = 60.0,
) -> np.ndarray:
    """Linked multi-channel lookahead limiter preventing clipping without distortion."""
    if len(signal) == 0:
        return signal.copy()

    # Leave 0.15 dB margin for inter-sample true peaks
    ceiling_lin = 10.0 ** ((ceiling_dbtp - 0.15) / 20.0)
    abs_peak = np.abs(signal) if signal.ndim == 1 else np.max(np.abs(signal), axis=1)

    gain_red = np.where(
        abs_peak > ceiling_lin,
        ceiling_lin / np.maximum(abs_peak, 1e-10),
        1.0,
    )

    lookahead = max(1, int(round(sample_rate * lookahead_ms / 1000.0)))
    if 1 < lookahead < len(gain_red):
        from numpy.lib.stride_tricks import sliding_window_view

        padded = np.pad(gain_red, (0, lookahead), mode="edge")
        windows = sliding_window_view(padded, lookahead + 1)
        gain_red = np.min(windows[: len(gain_red)], axis=1)

    if np.all(gain_red >= 1.0 - 1e-8):
        return signal

    release_coeff = float(np.exp(-1.0 / max(1.0, sample_rate * release_ms / 1000.0)))
    from scipy.signal import lfilter, lfilter_zi

    b = np.array([1.0 - release_coeff])
    a = np.array([1.0, -release_coeff])
    zi = lfilter_zi(b, a) * gain_red[0]
    env = gain_red.copy()
    for _ in range(3):
        filtered, _ = lfilter(b, a, env, zi=zi)
        nxt = np.minimum(gain_red, filtered)
        if np.allclose(nxt, env, atol=1e-7):
            env = nxt
            break
        env = nxt

    if signal.ndim == 1:
        return signal * env
    return signal * env[:, np.newaxis]


def _downsample_peak_envelope(signal: np.ndarray, sample_rate: int, bin_ms: int = 20) -> tuple[list[int], list[float]]:
    mono = signal if signal.ndim == 1 else np.max(np.abs(signal), axis=1)
    abs_m = np.abs(mono)
    step = max(1, int(round(sample_rate * bin_ms / 1000.0)))
    n = len(abs_m)
    times: list[int] = []
    peaks: list[float] = []
    for s in range(0, n, step):
        e = min(n, s + step)
        times.append(int(round(s * 1000.0 / sample_rate)))
        peaks.append(round(float(np.max(abs_m[s:e])), 5))
    return times, peaks


def _downsample_curve(values: np.ndarray, sample_rate: int, bin_ms: int = 20) -> tuple[list[int], list[float]]:
    step = max(1, int(round(sample_rate * bin_ms / 1000.0)))
    n = len(values)
    times: list[int] = []
    vals: list[float] = []
    for s in range(0, n, step):
        e = min(n, s + step)
        times.append(int(round(s * 1000.0 / sample_rate)))
        vals.append(round(float(np.mean(values[s:e])), 3))
    return times, vals


def process_voice_leveling(
    wav_path: str | Path,
    output_wav_path: str | Path,
    config: AudioLevelingConfig,
    *,
    cues: list[SubtitleCue] | None = None,
    bit_depth: int = 24,
    precomputed_vad_spans: list[tuple[int, int]] | None = None,
) -> LevelingOutput:
    """Run intelligent, timeline-exact voice loudness leveling on an extracted WAV file."""
    data, sample_rate = sf.read(str(wav_path), dtype="float64", always_2d=False)
    if data.size == 0:
        raise ValueError(f"音频文件为空: {wav_path}")

    n_samples = len(data)
    channels = 1 if data.ndim == 1 else data.shape[1]
    duration_ms = int(round(n_samples * 1000.0 / sample_rate))

    before_stats = measure_loudness(data, sample_rate)

    # 1. DC removal & zero-phase subsonic high-pass
    cleaned = data - np.mean(data, axis=0, keepdims=True)
    cleaned = _highpass_zero_phase(cleaned, sample_rate, config.highpass_hz)
    mono = cleaned if cleaned.ndim == 1 else np.mean(cleaned, axis=1)

    # 2. Silero VAD + optional aligned subtitle cue windows
    if precomputed_vad_spans is not None:
        vad_spans = precomputed_vad_spans
    else:
        vad_spans, _, _ = detect_speech_segments(mono, sample_rate)

    vad_mask = spans_to_sample_mask(vad_spans, n_samples, sample_rate)
    active_mask = vad_mask.copy()

    if config.use_subtitle_windows and cues:
        sub_mask = np.zeros(n_samples, dtype=bool)
        for cue in cues:
            s_idx = max(0, int(round(cue.start_ms * sample_rate / 1000.0)))
            e_idx = min(n_samples, int(round(cue.end_ms * sample_rate / 1000.0)))
            if e_idx > s_idx:
                sub_mask[s_idx:e_idx] = True
        # Include subtitle windows that have vocal energy above noise floor
        active_mask = vad_mask | sub_mask

    speech_ratio = float(np.mean(vad_mask)) if n_samples > 0 else 0.0

    # 3. Noise floor & speech SNR diagnostics
    frame_rms = _sliding_rms(mono, sample_rate, 120.0)
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-8))

    if np.any(~vad_mask):
        silence_db = frame_db[~vad_mask]
        noise_floor_db = float(np.percentile(silence_db, 30))
        silence_median_db = float(np.median(silence_db))
    else:
        noise_floor_db = float(np.percentile(frame_db, 10))
        silence_median_db = noise_floor_db

    if np.any(vad_mask):
        speech_median_db = float(np.median(frame_db[vad_mask]))
    else:
        speech_median_db = float(np.median(frame_db))

    estimated_snr = max(0.0, speech_median_db - silence_median_db)

    # Check risks (BGM/SFX mixed in, or heavily over-compressed source)
    risk_messages: list[str] = []
    bgm_risk = False
    over_comp_risk = False

    if np.any(~vad_mask) and silence_median_db > -34.0 and estimated_snr < 12.0:
        bgm_risk = True
        risk_messages.append(
            f"检测到视频可能包含后期背景音乐或较高底噪（非语音区电平约 {silence_median_db:.1f} dBFS，信噪比约 {estimated_snr:.1f} dB）。"
            "程序将避免过度提升静音段，但无法无损分离已混音 BGM/音效，建议使用轻度或标准模式并做 A/B 试听。"
        )

    crest_db = before_stats.sample_peak_dbfs - speech_median_db
    if before_stats.lra_lu < 2.2 and crest_db < 7.0 and speech_ratio > 0.2:
        over_comp_risk = True
        risk_messages.append(
            f"原音频动态范围较小（LRA {before_stats.lra_lu:.1f} LU，波峰因数 {crest_db:.1f} dB），"
            "人声可能已被前期重度压缩/限幅，继续大幅增益易引入失真或放大底噪。"
        )

    # 4. Multi-scale vocal loudness envelope (phrase + syllable micro-envelope)
    phrase_rms = _sliding_rms(mono, sample_rate, config.rms_window_ms)
    micro_rms = _sliding_rms(mono, sample_rate, max(65.0, config.rms_window_ms * 0.25))
    local_rms = 0.65 * phrase_rms + 0.35 * micro_rms
    local_db = 20.0 * np.log10(np.maximum(local_rms, 1e-8))

    # Calibrate initial speech target RMS from target LUFS (typical speech K-weight offset ~ -1.5 dB)
    speech_lufs_before = measure_speech_gated_lufs(cleaned, sample_rate, vad_mask if np.any(vad_mask) else active_mask)
    if np.any(vad_mask):
        lufs_to_rms_offset = speech_median_db - speech_lufs_before if speech_lufs_before > -65 else -1.5
        lufs_to_rms_offset = float(np.clip(lufs_to_rms_offset, -4.0, 4.0))
    else:
        lufs_to_rms_offset = -1.5
    target_rms_db = config.target_lufs + lufs_to_rms_offset

    # Desired raw gain at every speech sample
    raw_delta_db = target_rms_db - local_db

    # Emotion preservation: blend full correction with relative deviation preservation
    # Protect extreme shouts / whispers from flattening completely into robot voice
    dev_from_median = local_db - speech_median_db
    preserve_factor = float(np.clip(config.emotion_preserve, 0.0, 0.9))
    global_shift_db = target_rms_db - speech_median_db
    desired_gain_db = global_shift_db + (1.0 - preserve_factor) * (-dev_from_median)
    # Blend slightly with direct local tracking when deviations are large (>5 dB)
    large_dev = np.maximum(0.0, np.abs(dev_from_median) - 5.0)
    extra_catch = np.clip(large_dev * 0.08, 0.0, 0.45)
    desired_gain_db = (1.0 - extra_catch) * desired_gain_db + extra_catch * raw_delta_db

    # Noise gate protection: smoothly taper boost when local level sits near noise floor
    gate_floor = max(config.noise_gate_db, silence_median_db + 2.0 if bgm_risk else config.noise_gate_db)
    snr_above_gate = np.clip((local_db - gate_floor) / 8.0, 0.0, 1.0)
    desired_gain_db = np.where(desired_gain_db > 0.0, desired_gain_db * snr_above_gate, desired_gain_db)

    # Clamp gain envelope bounds
    desired_gain_db = np.clip(desired_gain_db, -config.max_cut_db, config.max_boost_db)

    # Smoothly interpolate across non-speech segments & apply bidirectional smoothing
    gain_db = _interpolate_silence_gain(desired_gain_db, active_mask)
    smooth_samples = max(1, int(round(sample_rate * config.smooth_window_ms / 1000.0)))
    gain_db = uniform_filter1d(gain_db, size=smooth_samples, mode="nearest")
    gain_db = uniform_filter1d(gain_db, size=max(1, smooth_samples // 2), mode="nearest")

    # Apply manual user local gain adjustments
    if config.manual_points:
        gain_db = _apply_manual_points(gain_db, sample_rate, config.manual_points)

    gain_linear = 10.0 ** (gain_db / 20.0)
    leveled = cleaned * (gain_linear if channels == 1 else gain_linear[:, np.newaxis])

    # 5. Soft-knee compressor for transient 1-2 character hot spikes
    leveled_mono = leveled if channels == 1 else np.mean(leveled, axis=1)
    comp_thresh = min(config.true_peak_ceiling_dbtp - 3.0, config.target_lufs + 5.5)
    leveled = _soft_knee_compressor(
        leveled,
        leveled_mono,
        sample_rate,
        threshold_db=comp_thresh,
        ratio=config.compression_ratio,
    )

    # 6. Second-pass speech-gated EBU R128 precision trim
    mask_for_lufs = vad_mask if np.any(vad_mask) else active_mask
    post_speech_lufs = measure_speech_gated_lufs(leveled, sample_rate, mask_for_lufs)
    trim_db = 0.0
    if post_speech_lufs > -60.0 and np.any(mask_for_lufs):
        trim_db = float(np.clip(config.target_lufs - post_speech_lufs, -5.0, 5.0))
        leveled = leveled * (10.0 ** (trim_db / 20.0))
        gain_db = gain_db + trim_db

    # 7. Lookahead true-peak limiter to guarantee zero clipping
    final_audio = _lookahead_true_peak_limiter(
        leveled,
        sample_rate,
        ceiling_dbtp=config.true_peak_ceiling_dbtp,
        lookahead_ms=5.0,
        release_ms=60.0,
    )
    # Hard safety clip guard
    final_audio = np.clip(final_audio, -0.9999, 0.9999)

    # Ensure exact sample count preservation (timeline sync invariant)
    assert len(final_audio) == n_samples

    after_stats = measure_loudness(final_audio, sample_rate)

    # Write output WAV (24-bit or 16-bit PCM)
    out_path = Path(output_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subtype = "PCM_24" if bit_depth == 24 else "PCM_16"
    sf.write(str(out_path), final_audio, sample_rate, subtype=subtype)

    # Build UI visualization curves (20ms bins)
    wf_times, wf_before = _downsample_peak_envelope(data, sample_rate, bin_ms=20)
    _, wf_after = _downsample_peak_envelope(final_audio, sample_rate, bin_ms=20)

    # Compute actual effective gain curve (including compressor/limiter) on 20ms bins
    eps = 1e-6
    before_abs = np.abs(data if channels == 1 else np.mean(data, axis=1))
    after_abs = np.abs(final_audio if channels == 1 else np.mean(final_audio, axis=1))
    # Use smoothed planned gain_db + limiter reduction where signal exceeded ceiling
    g_times, g_vals = _downsample_curve(gain_db, sample_rate, bin_ms=20)

    report = AudioAnalysisReport(
        sample_rate=sample_rate,
        channels=channels,
        duration_ms=duration_ms,
        speech_ratio=round(speech_ratio, 4),
        before_lufs=before_stats.integrated_lufs,
        after_lufs=after_stats.integrated_lufs,
        before_short_term_min_lufs=before_stats.short_term_min_lufs,
        before_short_term_max_lufs=before_stats.short_term_max_lufs,
        after_short_term_min_lufs=after_stats.short_term_min_lufs,
        after_short_term_max_lufs=after_stats.short_term_max_lufs,
        before_lra_lu=before_stats.lra_lu,
        after_lra_lu=after_stats.lra_lu,
        before_peak_dbfs=before_stats.sample_peak_dbfs,
        after_peak_dbfs=after_stats.sample_peak_dbfs,
        before_true_peak_dbtp=before_stats.true_peak_dbtp,
        after_true_peak_dbtp=after_stats.true_peak_dbtp,
        noise_floor_dbfs=round(noise_floor_db, 2),
        estimated_snr_db=round(estimated_snr, 2),
        max_applied_gain_db=round(float(np.max(gain_db)), 2),
        min_applied_gain_db=round(float(np.min(gain_db)), 2),
        bgm_or_sfx_risk=bgm_risk,
        over_compressed_risk=over_comp_risk,
        risk_messages=risk_messages,
    )

    return LevelingOutput(
        processed_signal=final_audio,
        sample_rate=sample_rate,
        report=report,
        vad_spans_ms=vad_spans,
        gain_curve_times_ms=g_times,
        gain_curve_db=g_vals,
        waveform_times_ms=wf_times,
        waveform_before_peaks=wf_before,
        waveform_after_peaks=wf_after,
    )
