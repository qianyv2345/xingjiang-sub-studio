"""Silero VAD ONNX inference for fast, torch-free voice activity detection."""

from __future__ import annotations

from importlib import resources
import threading

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly

_LOCK = threading.Lock()
_SESSION: ort.InferenceSession | None = None


def get_vad_session() -> ort.InferenceSession:
    """Lazily load Silero VAD ONNX model using CPUExecutionProvider."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _LOCK:
        if _SESSION is not None:
            return _SESSION
        model_path = str(resources.files("silero_vad.data").joinpath("silero_vad.onnx"))
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        _SESSION = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        return _SESSION


def to_16k_mono(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert mono or multi-channel float audio at `sample_rate` to 16kHz mono float32."""
    if signal.ndim == 2:
        mono = np.mean(signal, axis=1, dtype=np.float64)
    else:
        mono = signal.astype(np.float64, copy=False)

    if len(mono) == 0:
        return np.zeros(0, dtype=np.float32)

    if sample_rate == 16000:
        return mono.astype(np.float32)

    from math import gcd

    g = gcd(16000, int(sample_rate))
    up = 16000 // g
    down = int(sample_rate) // g
    resampled = resample_poly(mono, up, down)
    return resampled.astype(np.float32)


def detect_speech_segments(
    signal: np.ndarray,
    sample_rate: int,
    *,
    threshold: float = 0.35,
    neg_threshold: float = 0.18,
    min_speech_ms: int = 120,
    min_silence_ms: int = 140,
    pad_ms: int = 80,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Run Silero VAD ONNX and return (spans_ms, frame_times_ms, frame_probs).

    Each span is `(start_ms, end_ms)` on the original timeline.
    """
    audio_16k = to_16k_mono(signal, sample_rate)
    if len(audio_16k) == 0:
        return [], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    session = get_vad_session()
    window_size = 512  # 32ms at 16kHz
    context_size = 64
    state = np.zeros((2, 1, 128), dtype=np.float32)

    num_samples = len(audio_16k)
    n_chunks = (num_samples + window_size - 1) // window_size
    padded_audio = np.zeros(n_chunks * window_size, dtype=np.float32)
    padded_audio[:num_samples] = audio_16k.astype(np.float32, copy=False)

    x_buf = np.zeros((1, context_size + window_size), dtype=np.float32)
    sr_arr = np.array(16000, dtype=np.int64)
    inp_dict = {"input": x_buf, "state": state, "sr": sr_arr}

    probs_arr = np.empty(n_chunks, dtype=np.float32)
    times_arr = np.rint(np.arange(n_chunks, dtype=np.float64) * (window_size * 1000.0 / 16000.0)).astype(np.int64)

    for idx in range(n_chunks):
        start = idx * window_size
        x_buf[0, context_size:] = padded_audio[start : start + window_size]
        inp_dict["state"] = state
        out, state = session.run(None, inp_dict)
        probs_arr[idx] = float(np.asarray(out).squeeze())
        x_buf[0, :context_size] = x_buf[0, -context_size:]

    # Hysteresis state machine to form clean speech segments
    frame_dur_ms = 32
    triggered = False
    temp_end = 0
    speeches: list[tuple[int, int]] = []
    cur_start = 0

    for idx, prob in enumerate(probs_arr):
        t_ms = idx * frame_dur_ms
        if prob >= threshold and temp_end:
            temp_end = 0
        if prob >= threshold and not triggered:
            triggered = True
            cur_start = t_ms
            continue
        if triggered and prob < neg_threshold:
            if temp_end == 0:
                temp_end = t_ms
            if (t_ms - temp_end) >= min_silence_ms:
                if (temp_end - cur_start) >= min_speech_ms:
                    speeches.append((cur_start, temp_end))
                cur_start = 0
                temp_end = 0
                triggered = False

    total_ms = int(round(num_samples * 1000.0 / 16000.0))
    if triggered and (total_ms - cur_start) >= min_speech_ms:
        speeches.append((cur_start, total_ms))

    # Apply padding & merge overlapping padded spans
    if not speeches:
        return [], times_arr, probs_arr

    padded: list[tuple[int, int]] = []
    for s, e in speeches:
        ps = max(0, s - pad_ms)
        pe = min(total_ms, e + pad_ms)
        if padded and ps <= padded[-1][1]:
            padded[-1] = (padded[-1][0], max(padded[-1][1], pe))
        else:
            padded.append((ps, pe))

    return padded, times_arr, probs_arr


def spans_to_sample_mask(
    spans_ms: list[tuple[int, int]],
    total_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """Convert millisecond `(start_ms, end_ms)` spans to a boolean mask of length `total_samples`."""
    mask = np.zeros(total_samples, dtype=bool)
    for s_ms, e_ms in spans_ms:
        s_idx = max(0, int(round(s_ms * sample_rate / 1000.0)))
        e_idx = min(total_samples, int(round(e_ms * sample_rate / 1000.0)))
        if e_idx > s_idx:
            mask[s_idx:e_idx] = True
    return mask
