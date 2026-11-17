"""ITU-R BS.1770-4 / EBU R128 loudness, true-peak, LRA, and noise floor diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly


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


def measure_true_peak_dbtp(signal: np.ndarray, oversample: int = 4) -> float:
    """Estimate inter-sample true peak in dBTP using 4x polyphase oversampling."""
    if signal.size == 0:
        return -90.0
    if signal.ndim == 1:
        up = resample_poly(signal.astype(np.float64), oversample, 1)
        peak = float(np.max(np.abs(up)))
    else:
        peak = 0.0
        for ch in range(signal.shape[1]):
            up = resample_poly(signal[:, ch].astype(np.float64), oversample, 1)
            peak = max(peak, float(np.max(np.abs(up))))
    return _linear_to_db(peak)


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
    try:
        integrated = float(meter.integrated_loudness(arr))
        if not np.isfinite(integrated):
            integrated = -70.0
    except Exception:
        integrated = -70.0

    # Measure 1-second hopping (0.2s step) short-term LUFS for min/max & LRA estimation
    # Use 1.5s windows for short dramas (1-2 min) so local quiet/loud phrases are captured accurately
    win_samples = max(int(sample_rate * 1.5), int(sample_rate * 0.5))
    hop_samples = max(1, int(sample_rate * 0.25))
    short_lufs: list[float] = []

    if len(arr) >= win_samples:
        for start in range(0, len(arr) - win_samples + 1, hop_samples):
            chunk = arr[start : start + win_samples]
            try:
                val = float(meter.integrated_loudness(chunk))
                if np.isfinite(val) and val > -65.0:
                    short_lufs.append(val)
            except Exception:
                pass

    if short_lufs:
        short_arr = np.asarray(short_lufs, dtype=np.float64)
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
        val = float(meter.integrated_loudness(speech_only.astype(np.float64)))
        return round(val, 2) if np.isfinite(val) else -70.0
    except Exception:
        rms = float(np.sqrt(np.mean(speech_only.astype(np.float64) ** 2)))
        return round(_linear_to_db(rms) - 0.5, 2)
