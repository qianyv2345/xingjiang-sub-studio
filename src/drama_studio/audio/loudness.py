"""ITU-R BS.1770-4 / EBU R128 loudness, true-peak, LRA, and noise floor diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
from scipy.signal import lfilter, resample_poly


@dataclass
class LoudnessStats:
    integrated_lufs: float
    short_term_min_lufs: float
    short_term_max_lufs: float
    lra_lu: float
    sample_peak_dbfs: float
    true_peak_dbtp: float


def _linear_to_db(val: float, floor_db: float = -90.0) -> float:
    if val <= 1e-10:
        return floor_db
    return float(max(floor_db, 20.0 * np.log10(val)))


def find_true_peak_candidate_runs(
    abs_max_per_sample: np.ndarray,
    threshold: float,
    block: int = 4096,
) -> list[tuple[int, int]]:
    """Return merged `(start_sample, end_sample)` spans whose block peak reaches `threshold`."""
    n = len(abs_max_per_sample)
    if n == 0:
        return []
    n_blocks = (n + block - 1) // block
    pad_len = n_blocks * block - n
    padded = np.pad(abs_max_per_sample, (0, pad_len)) if pad_len > 0 else abs_max_per_sample
    blk_max = padded.reshape(n_blocks, block).max(axis=1)
    cand = np.where(blk_max >= threshold)[0]
    if len(cand) == 0:
        return []
    runs: list[tuple[int, int]] = []
    rs = re_blk = int(cand[0])
    for blk_idx in cand[1:]:
        b = int(blk_idx)
        if b <= re_blk + 1:
            re_blk = b
        else:
            runs.append((rs * block, min(n, (re_blk + 1) * block)))
            rs = re_blk = b
    runs.append((rs * block, min(n, (re_blk + 1) * block)))
    return runs


def measure_true_peak_dbtp(signal: np.ndarray, oversample: int = 4) -> float:
    """Estimate inter-sample true peak in dBTP using 4x polyphase oversampling."""
    if signal.size == 0:
        return -90.0
    sample_peak = float(np.max(np.abs(signal)))
    if sample_peak <= 1e-10:
        return -90.0

    n = len(signal)
    if n <= 16384:
        if signal.ndim == 1:
            up = resample_poly(signal.astype(np.float64, copy=False), oversample, 1)
            peak = float(np.max(np.abs(up)))
        else:
            peak = sample_peak
            for ch in range(signal.shape[1]):
                up = resample_poly(signal[:, ch].astype(np.float64, copy=False), oversample, 1)
                peak = max(peak, float(np.max(np.abs(up))))
        return _linear_to_db(max(sample_peak, peak))

    # Only upsample blocks >= 40% of sample_peak (polyphase sinc FIR cannot exceed ~2.5x sample peak)
    pad = 64
    thresh = sample_peak * 0.40
    peak = sample_peak
    channels = (signal,) if signal.ndim == 1 else tuple(signal[:, ch] for ch in range(signal.shape[1]))
    for ch_arr in channels:
        ch64 = ch_arr.astype(np.float64, copy=False)
        for core_s, core_e in find_true_peak_candidate_runs(np.abs(ch64), thresh):
            s_i = max(0, core_s - pad)
            e_i = min(n, core_e + pad)
            up = resample_poly(ch64[s_i:e_i], oversample, 1)
            trim_l = (core_s - s_i) * oversample
            trim_r = len(up) - (e_i - core_e) * oversample
            core_up = up[trim_l:trim_r] if trim_r > trim_l else up
            peak = max(peak, float(np.max(np.abs(core_up))))
    return _linear_to_db(peak)


def _k_weighted_cumsum(arr: np.ndarray, meter: pyln.Meter) -> np.ndarray:
    """Apply BS.1770 K-weighting once across all channels and return cumulative weighted energy."""
    data = arr[:, np.newaxis] if arr.ndim == 1 else arr
    n_ch = data.shape[1]
    filtered = data.astype(np.float64, copy=True)
    for _cls, f_stage in meter._filters.items():
        filtered = f_stage.passband_gain * lfilter(f_stage.b, f_stage.a, filtered, axis=0)
    gains = np.array([1.0, 1.0, 1.0, 1.41, 1.41][:n_ch], dtype=np.float64)
    weighted_sq = np.sum((filtered ** 2) * gains[np.newaxis, :], axis=1)
    csum = np.empty(len(weighted_sq) + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(weighted_sq, out=csum[1:])
    return csum


def _bs1770_gated_lufs(z_blocks: np.ndarray) -> np.ndarray:
    """Compute ITU-R BS.1770-4 absolute+relative gated LUFS along axis=-1 for 1D or 2D block energies."""
    with np.errstate(divide="ignore", invalid="ignore"):
        l_blk = -0.691 + 10.0 * np.log10(np.maximum(z_blocks, 1e-30))
        abs_mask = l_blk >= -70.0
        abs_cnt = np.sum(abs_mask, axis=-1)
        z_abs = np.sum(np.where(abs_mask, z_blocks, 0.0), axis=-1) / np.maximum(abs_cnt, 1)
        gamma_r = -0.691 + 10.0 * np.log10(np.maximum(z_abs, 1e-30)) - 10.0
        rel_mask = abs_mask & (l_blk > np.expand_dims(gamma_r, axis=-1))
        rel_cnt = np.sum(rel_mask, axis=-1)
        z_rel = np.sum(np.where(rel_mask, z_blocks, 0.0), axis=-1) / np.maximum(rel_cnt, 1)
        vals = -0.691 + 10.0 * np.log10(np.maximum(z_rel, 1e-30))
        return np.where(rel_cnt > 0, vals, -70.0)


def _vectorized_integrated_lufs(
    arr: np.ndarray,
    sample_rate: int,
    meter: pyln.Meter,
    csum: np.ndarray | None = None,
) -> float:
    """Compute ITU-R BS.1770-4 integrated LUFS using vectorized cumulative-sum block energies."""
    t_g = meter.block_size
    step = 1.0 - meter.overlap
    eff_csum = csum if csum is not None else (_k_weighted_cumsum(arr, meter) if hasattr(meter, "_filters") else None)
    if eff_csum is not None:
        total_blocks = max(1, int(np.round(((len(arr) / sample_rate) - t_g) / (t_g * step))) + 1)
        j_idx = np.arange(total_blocks, dtype=np.int64)
        l_full = (t_g * (j_idx * step) * sample_rate).astype(np.int64)
        u_full = np.minimum(len(arr), (t_g * (j_idx * step + 1.0) * sample_rate).astype(np.int64))
        z_full = (eff_csum[u_full] - eff_csum[l_full]) / max(1.0, t_g * sample_rate)
        val = float(_bs1770_gated_lufs(z_full))
        return val if np.isfinite(val) else -70.0
    try:
        val = float(meter.integrated_loudness(arr))
        return val if np.isfinite(val) else -70.0
    except Exception:
        return -70.0


def measure_loudness(signal: np.ndarray, sample_rate: int) -> LoudnessStats:
    """Measure integrated LUFS, short-term min/max LUFS, LRA, sample peak, and true peak."""
    if signal.size == 0 or len(signal) < max(64, int(sample_rate * 0.4)):
        peak_db = _linear_to_db(float(np.max(np.abs(signal))) if signal.size else 0.0)
        return LoudnessStats(
            integrated_lufs=-70.0,
            short_term_min_lufs=-70.0,
            short_term_max_lufs=-70.0,
            lra_lu=0.0,
            sample_peak_dbfs=peak_db,
            true_peak_dbtp=peak_db,
        )

    arr = signal.astype(np.float64, copy=False)
    sample_peak = float(np.max(np.abs(arr)))
    sample_peak_db = _linear_to_db(sample_peak)
    true_peak_db = measure_true_peak_dbtp(arr)

    meter = pyln.Meter(sample_rate)
    t_g = meter.block_size
    step = 1.0 - meter.overlap
    win_samples = max(int(sample_rate * 1.5), int(sample_rate * 0.5))
    hop_samples = max(1, int(sample_rate * 0.25))

    csum = _k_weighted_cumsum(arr, meter) if hasattr(meter, "_filters") else None
    integrated = _vectorized_integrated_lufs(arr, sample_rate, meter, csum=csum)

    # Vectorized 1.5s hopping (0.25s step) short-term LUFS for min/max & LRA estimation
    short_arr = np.zeros(0, dtype=np.float64)
    if len(arr) >= win_samples:
        if csum is not None:
            starts = np.arange(0, len(arr) - win_samples + 1, hop_samples, dtype=np.int64)
            t_win = win_samples / sample_rate
            win_blocks = max(1, int(np.round((t_win - t_g) / (t_g * step))) + 1)
            j_win = np.arange(win_blocks, dtype=np.int64)
            blk_l = (t_g * (j_win * step) * sample_rate).astype(np.int64)
            blk_u = np.minimum(win_samples, (t_g * (j_win * step + 1.0) * sample_rate).astype(np.int64))
            z_win = (csum[starts[:, np.newaxis] + blk_u[np.newaxis, :]] - csum[starts[:, np.newaxis] + blk_l[np.newaxis, :]]) / max(1.0, t_g * sample_rate)
            vals = _bs1770_gated_lufs(z_win)
            short_arr = vals[np.isfinite(vals) & (vals > -65.0)]
        else:
            short_lufs: list[float] = []
            for start in range(0, len(arr) - win_samples + 1, hop_samples):
                try:
                    val = float(meter.integrated_loudness(arr[start : start + win_samples]))
                    if np.isfinite(val) and val > -65.0:
                        short_lufs.append(val)
                except Exception:
                    pass
            short_arr = np.asarray(short_lufs, dtype=np.float64)

    if len(short_arr) > 0:
        # Relative gate at integrated - 20 LU per EBU Tech 3342
        gated = short_arr[short_arr >= max(-65.0, integrated - 20.0)]
        if len(gated) >= 2:
            st_min = float(np.min(gated))
            st_max = float(np.max(gated))
            lra = float(np.percentile(gated, 95) - np.percentile(gated, 10))
        else:
            st_min = float(np.min(short_arr))
            st_max = float(np.max(short_arr))
            lra = float(np.max(short_arr) - np.min(short_arr))
    else:
        st_min = integrated
        st_max = integrated
        lra = 0.0

    return LoudnessStats(
        integrated_lufs=round(integrated, 2),
        short_term_min_lufs=round(st_min, 2),
        short_term_max_lufs=round(st_max, 2),
        lra_lu=round(max(0.0, lra), 2),
        sample_peak_dbfs=round(sample_peak_db, 2),
        true_peak_dbtp=round(true_peak_db, 2),
    )


def measure_speech_gated_lufs(
    signal: np.ndarray,
    sample_rate: int,
    speech_mask: np.ndarray,
) -> float:
    """Measure integrated LUFS over active speech samples only."""
    if signal.size == 0 or not np.any(speech_mask):
        return -70.0
    speech_only = signal[speech_mask]
    if len(speech_only) < int(sample_rate * 0.45):
        rms = float(np.sqrt(np.mean(speech_only.astype(np.float64) ** 2)))
        return _linear_to_db(rms) - 0.5
    meter = pyln.Meter(sample_rate)
    try:
        val = _vectorized_integrated_lufs(speech_only.astype(np.float64, copy=False), sample_rate, meter)
        return round(val, 2) if np.isfinite(val) else -70.0
    except Exception:
        rms = float(np.sqrt(np.mean(speech_only.astype(np.float64) ** 2)))
        return round(_linear_to_db(rms) - 0.5, 2)
