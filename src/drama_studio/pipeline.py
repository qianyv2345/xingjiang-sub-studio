"""End-to-end single-episode pipeline orchestrating FFmpeg, VAD, alignment, DP layout, and voice leveling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading
from typing import Callable

import numpy as np
import soundfile as sf

from drama_studio.align.engine import AlignmentEngineResult, run_alignment_and_segmentation
from drama_studio.audio.leveler import process_voice_leveling
from drama_studio.audio.vad import detect_speech_segments, to_16k_mono
from drama_studio.exporters.exporter import export_episode_outputs
from drama_studio.media.ffmpeg_wrapper import extract_audio_to_wav, probe_media
from drama_studio.models import (
    AlignmentConfig,
    AnomalyItem,
    AnomalySeverity,
    AnomalyType,
    AudioLevelingConfig,
    EpisodeTask,
    ExportConfig,
    PipelineResult,
    SubtitleCue,
    SubtitleTemplate,
)
from drama_studio.nlp.srt_io import load_srt


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise InterruptedError("任务已被用户取消")


def process_single_episode(
    task: EpisodeTask,
    template: SubtitleTemplate,
    audio_cfg: AudioLevelingConfig,
    alignment_cfg: AlignmentConfig,
    export_cfg: ExportConfig,
    cache_dir: Path,
    work_dir: Path,
    *,
    cancel_token: CancellationToken | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> PipelineResult:
    """Run complete extraction, alignment, semantic segmentation, voice leveling, QA, and export."""
    token = cancel_token or CancellationToken()
    work_dir.mkdir(parents=True, exist_ok=True)

    def _p(frac: float, msg: str) -> None:
        token.check()
        if progress_cb:
            progress_cb(frac, msg)

    # 1. Media probe
    _p(0.05, "检测视频分辨率、帧率、音轨与时间基准...")
    media_info = task.media_info or probe_media(task.video_path)
    media_info.selected_audio_index = task.selected_audio_index
    if not media_info.audio_tracks:
        raise RuntimeError(f"视频文件 {media_info.file_name} 不包含音轨，无法处理。")

    track_idx = max(0, min(task.selected_audio_index, len(media_info.audio_tracks) - 1))
    sel_track = media_info.audio_tracks[track_idx]
    track_offset_ms = round(sel_track.start_time_ms - media_info.video_start_time_ms, 2)

    # 2. Extract broadcast-quality WAV (48kHz) synchronized to video start PTS=0
    _p(0.14, f"使用 FFmpeg 自动提取音轨 #{track_idx + 1} (48kHz)...")
    raw_wav_path = work_dir / f"{task.episode_label}_extracted_48k.wav"
    extract_audio_to_wav(
        task.video_path,
        raw_wav_path,
        audio_stream_index=track_idx,
        sample_rate=export_cfg.wav_sample_rate,
        bit_depth=export_cfg.wav_bit_depth,
        sync_offset_ms=track_offset_ms,
        target_duration_ms=media_info.duration_ms,
    )

    # 3. Load audio & compute Silero VAD + 16kHz downmix
    _p(0.22, "执行 Silero VAD 语音活动检测...")
    sig_full, sr_full = sf.read(str(raw_wav_path), dtype="float64", always_2d=False)
    audio_16k = to_16k_mono(sig_full, sr_full)
    duration_ms = media_info.duration_ms or int(round(len(sig_full) * 1000.0 / sr_full))
    vad_spans, _, _ = detect_speech_segments(audio_16k, 16000)

    # 4. Load SRT & execute Forced Alignment + Smart Semantic Segmentation
    raw_cues: list[SubtitleCue] = []
    if task.srt_path and Path(task.srt_path).is_file():
        _p(0.28, "解析原始剪映 SRT 字幕...")
        raw_cues = load_srt(task.srt_path)
        align_res = run_alignment_and_segmentation(
            raw_cues=raw_cues,
            audio_16k=audio_16k,
            vad_spans=vad_spans,
            duration_ms=duration_ms,
            template=template,
            alignment_cfg=alignment_cfg,
            cache_dir=cache_dir,
            video_fps=media_info.fps or 25.0,
            script_path=task.script_path or None,
            progress_cb=lambda f, m: _p(f, m),
        )
    else:
        _p(0.50, "未提供 SRT 字幕，仅执行音频智能音量均衡与质检...")
        align_res = AlignmentEngineResult(
            cues=[],
            anomalies=[],
            global_offset_ms=0,
            aligner_used="无字幕（仅音频处理）",
            nlp_used=alignment_cfg.nlp_backend,
            text_integrity_ok=True,
        )

    # 5. Intelligent Voice Loudness Leveling (48kHz, timeline-synced)
    _p(0.72, "执行智能人声音量均衡 (VAD + 字幕窗 + 动态增益 + 真峰值限制)...")
    processed_wav_path = work_dir / f"{task.episode_label}_processed_48k.wav"
    level_out = process_voice_leveling(
        raw_wav_path,
        processed_wav_path,
        audio_cfg,
        cues=align_res.cues,
        bit_depth=export_cfg.wav_bit_depth,
        precomputed_vad_spans=vad_spans,
    )

    # Merge audio risk warnings into anomaly list for visibility
    all_anomalies: list[AnomalyItem] = list(align_res.anomalies)
    if abs(track_offset_ms) >= 15.0:
        all_anomalies.insert(
            0,
            AnomalyItem(
                anomaly_type=AnomalyType.AV_OFFSET_WARNING,
                severity=AnomalySeverity.INFO,
                start_ms=0,
                end_ms=min(duration_ms, 2000),
                cue_index=None,
                message=f"原视频音画起始时间存在 {track_offset_ms:+.1f} ms 偏移，导出 WAV 已自动对齐视频时间零点。",
                suggestion="导回达芬奇时间线起点 00:00:00:00 即可音画同步。",
            ),
        )
    if level_out.report.bgm_or_sfx_risk:
        all_anomalies.insert(
            0,
            AnomalyItem(
                anomaly_type=AnomalyType.HIGH_BGM_RISK,
                severity=AnomalySeverity.WARNING,
                start_ms=0,
                end_ms=duration_ms,
                cue_index=None,
                message=level_out.report.risk_messages[0] if level_out.report.risk_messages else "检测到较高背景音/BGM 混音风险",
                suggestion="建议对比 A/B 试听，如背景起伏偏大可切换「轻度」模式。",
            ),
        )
    if level_out.report.over_compressed_risk:
        all_anomalies.insert(
            0,
            AnomalyItem(
                anomaly_type=AnomalyType.OVER_COMPRESSED_RISK,
                severity=AnomalySeverity.WARNING,
                start_ms=0,
                end_ms=duration_ms,
                cue_index=None,
                message=next(
                    (m for m in level_out.report.risk_messages if "压缩" in m),
                    "检测到原音频动态范围偏小（可能已经过重度压缩）",
                ),
                suggestion="建议适当降低「最大提升」或使用「轻度」模式。",
            ),
        )

    result = PipelineResult(
        episode_label=task.episode_label,
        media_info=media_info,
        raw_cues=raw_cues,
        processed_cues=align_res.cues,
        anomalies=all_anomalies,
        audio_report=level_out.report,
        extracted_wav_path=str(raw_wav_path),
        processed_wav_path=str(processed_wav_path),
        waveform_times_ms=level_out.waveform_times_ms,
        waveform_before_peaks=level_out.waveform_before_peaks,
        waveform_after_peaks=level_out.waveform_after_peaks,
        gain_curve_times_ms=level_out.gain_curve_times_ms,
        gain_curve_db=level_out.gain_curve_db,
        vad_spans_ms=level_out.vad_spans_ms,
        global_offset_ms=align_res.global_offset_ms,
        aligner_used=align_res.aligner_used,
        nlp_used=align_res.nlp_used,
        text_integrity_ok=align_res.text_integrity_ok,
    )

    # 6. Export DaVinci Resolve-ready deliverables
    _p(0.88, "正在导出达芬奇可用 WAV / SRT / 分析报告...")
    eff_export = deepcopy(export_cfg)
    if not eff_export.output_dir:
        eff_export.output_dir = str(Path(task.video_path).parent / "智能修正输出")
    export_episode_outputs(result, eff_export, progress_cb=lambda m: _p(0.94, m))

    _p(1.0, f"处理完成：输出 {len(result.processed_cues)} 条字幕，异常提示 {len(result.anomalies)} 项。")
    return result
