"""Unified three-stage episode pipeline with deterministic stage caching and step-level re-execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import shutil
import threading
import time
from typing import Callable, Literal

import numpy as np
import soundfile as sf

from drama_studio.align.asr_engine import run_asr_transcription, select_asr_model_id
from drama_studio.align.base import ASRSegment
from drama_studio.align.engine import (
    AlignmentEngineResult,
    ForcedAlignmentStageResult,
    run_forced_alignment_only,
    segment_aligned_windows_to_cues,
)
from drama_studio.align.runtime import MODEL_POOL, snapshot_resource_usage
from drama_studio.audio.leveler import LevelingOutput, process_voice_leveling
from drama_studio.audio.vad import detect_speech_segments, to_16k_mono
from drama_studio.cache import (
    atomic_replace_file,
    atomic_write_json,
    compute_file_hash,
    compute_media_fingerprint,
    get_episode_cache_dir,
    load_asr_cache,
    load_forced_align_cache,
    load_layout_cache,
    load_leveling_meta_cache,
    load_script_match_cache,
    prune_stage_cache,
    read_json_safe,
    save_asr_cache,
    save_forced_align_cache,
    save_layout_cache,
    save_leveling_meta_cache,
    save_script_match_cache,
    stable_param_hash,
)
from drama_studio.exporters.exporter import export_episode_outputs
from drama_studio.media.ffmpeg_wrapper import extract_audio_to_wav, probe_media
from drama_studio.models import (
    AlignmentConfig,
    AnomalyItem,
    AnomalySeverity,
    AnomalyType,
    AudioAnalysisReport,
    AudioLevelingConfig,
    EpisodeTask,
    ExportConfig,
    Orientation,
    PipelineResult,
    SubtitleCue,
    SubtitleTemplate,
    TaskMode,
)
from drama_studio.nlp.punctuation import detect_text_language
from drama_studio.nlp.script_io import parse_dialogue_script
from drama_studio.nlp.script_matcher import ScriptMatchResult, match_asr_to_script
from drama_studio.nlp.srt_io import load_srt

StageAction = Literal[
    "full",
    "asr_only",
    "script_match_only",
    "align_only",
    "relayout_only",
    "relevel_only",
]


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


def _select_template_for_language(
    base_template: SubtitleTemplate,
    detected_lang: str,
    task: EpisodeTask,
    media_info,
    templates_by_lang: dict[str, SubtitleTemplate] | None,
) -> SubtitleTemplate:
    if task.template_override is not None:
        return deepcopy(task.template_override)
    if base_template.language == detected_lang:
        return deepcopy(base_template)
    if templates_by_lang and detected_lang in templates_by_lang:
        return deepcopy(templates_by_lang[detected_lang])
    if detected_lang == "en":
        if media_info is not None and media_info.orientation == Orientation.PORTRAIT:
            return SubtitleTemplate.preset_english_portrait()
        return SubtitleTemplate.preset_english_default()
    if media_info is not None and media_info.orientation == Orientation.LANDSCAPE:
        return SubtitleTemplate.preset_landscape_16_9()
    return SubtitleTemplate.preset_portrait_9_16()


def _asr_segments_to_coarse_cues(segments: list[ASRSegment]) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for idx, seg in enumerate(segments, start=1):
        cues.append(
            SubtitleCue(
                index=idx,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                raw_text=seg.text,
                display_text=seg.text,
                orig_start_ms=seg.start_ms,
                orig_end_ms=seg.end_ms,
                confidence=seg.confidence,
                turn_id=idx,
                src_win_start_ms=seg.src_win_start_ms if seg.src_win_start_ms >= 0 else seg.start_ms,
                src_win_end_ms=seg.src_win_end_ms if seg.src_win_end_ms >= 0 else seg.end_ms,
            )
        )
    return cues


def _build_empty_audio_report(sr: int, channels: int, duration_ms: int) -> AudioAnalysisReport:
    return AudioAnalysisReport(
        sample_rate=sr,
        channels=channels,
        duration_ms=duration_ms,
        speech_ratio=0.0,
        before_lufs=-24.0,
        after_lufs=-24.0,
        before_short_term_min_lufs=-30.0,
        before_short_term_max_lufs=-18.0,
        after_short_term_min_lufs=-30.0,
        after_short_term_max_lufs=-18.0,
        before_lra_lu=0.0,
        after_lra_lu=0.0,
        before_peak_dbfs=-6.0,
        after_peak_dbfs=-6.0,
        before_true_peak_dbtp=-6.0,
        after_true_peak_dbtp=-6.0,
        noise_floor_dbfs=-60.0,
        estimated_snr_db=20.0,
        max_applied_gain_db=0.0,
        min_applied_gain_db=0.0,
    )


def process_single_episode(
    task: EpisodeTask,
    template: SubtitleTemplate,
    audio_cfg: AudioLevelingConfig,
    alignment_cfg: AlignmentConfig,
    export_cfg: ExportConfig,
    cache_dir: Path,
    work_dir: Path,
    *,
    stage_action: StageAction = "full",
    force_recompute: bool = False,
    templates_by_lang: dict[str, SubtitleTemplate] | None = None,
    cache_max_size_mb: int = 4096,
    cancel_token: CancellationToken | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> PipelineResult:
    """Run complete or stage-scoped episode processing with verifiable stage caching."""
    token = cancel_token or CancellationToken()
    work_dir.mkdir(parents=True, exist_ok=True)
    t_start_all = time.perf_counter()
    stage_timings: dict[str, int] = {}
    cache_hits: list[str] = []

    def _p(frac: float, msg: str) -> None:
        token.check()
        if progress_cb:
            progress_cb(frac, msg)

    # ---- Stage 1: Probe & Audio Extraction ----
    t0 = time.perf_counter()
    _p(0.04, "阶段 1/3 · 检测视频编码、帧率、音轨与文件指纹...")
    media_info = task.media_info or probe_media(task.video_path)
    media_info.selected_audio_index = task.selected_audio_index
    if not media_info.audio_tracks:
        raise RuntimeError(f"视频文件 {media_info.file_name} 不包含音轨，无法处理。")

    track_idx = max(0, min(task.selected_audio_index, len(media_info.audio_tracks) - 1))
    sel_track = media_info.audio_tracks[track_idx]
    track_offset_ms = round(sel_track.start_time_ms - media_info.video_start_time_ms, 2)

    media_fp = compute_media_fingerprint(task.video_path)
    ep_cache = get_episode_cache_dir(work_dir, task.video_path, media_fp)

    audio_key = stable_param_hash(
        {
            "fp": media_fp,
            "track": track_idx,
            "sr": export_cfg.wav_sample_rate,
            "bits": export_cfg.wav_bit_depth,
            "offset": track_offset_ms,
            "v_start": media_info.video_start_time_ms,
            "dur": media_info.duration_ms,
            "algo_v": 2,
        }
    )
    raw_wav_path = ep_cache / f"extracted_{audio_key}.wav"
    audio_meta_path = ep_cache / f"extracted_{audio_key}.meta.json"

    if raw_wav_path.is_file() and audio_meta_path.is_file() and not force_recompute:
        cache_hits.append("audio_extract")
        _p(0.12, "命中阶段缓存：复用已提取的 48kHz 原始音轨")
    else:
        _p(0.12, f"使用 FFmpeg 提取音轨 #{track_idx + 1} ({export_cfg.wav_sample_rate}Hz)...")
        tmp_wav = ep_cache / f".tmp_extracted_{audio_key}.wav"
        target_dur = (
            max(100, media_info.duration_ms - int(round(media_info.video_start_time_ms)))
            if media_info.duration_ms > 0 and media_info.video_start_time_ms > 5.0
            else media_info.duration_ms
        )
        try:
            extract_audio_to_wav(
                task.video_path,
                tmp_wav,
                audio_stream_index=track_idx,
                sample_rate=export_cfg.wav_sample_rate,
                bit_depth=export_cfg.wav_bit_depth,
                sync_offset_ms=track_offset_ms,
                target_duration_ms=target_dur,
                video_start_time_ms=media_info.video_start_time_ms,
            )
            atomic_replace_file(tmp_wav, raw_wav_path)
        except Exception:
            tmp_wav.unlink(missing_ok=True)
            raise
        atomic_write_json(audio_meta_path, {"audio_key": audio_key, "duration_ms": media_info.duration_ms})
    stage_timings["stage1_extract_ms"] = int((time.perf_counter() - t0) * 1000)

    # Load 48k WAV & build 16k mono + VAD (with cache)
    t_vad = time.perf_counter()
    sig_full, sr_full = sf.read(str(raw_wav_path), dtype="float64", always_2d=False)
    channels_full = 1 if sig_full.ndim == 1 else sig_full.shape[1]
    audio_16k = to_16k_mono(sig_full, sr_full)
    duration_ms = media_info.duration_ms or int(round(len(sig_full) * 1000.0 / sr_full))

    vad_cache_path = ep_cache / f"vad_{audio_key}.json"
    vad_cached = None if force_recompute else read_json_safe(vad_cache_path)
    if isinstance(vad_cached, dict) and "vad_spans" in vad_cached:
        vad_spans = [(int(s), int(e)) for s, e in vad_cached["vad_spans"]]
        cache_hits.append("vad")
    else:
        _p(0.18, "执行 Silero VAD 语音活动检测...")
        vad_spans, _, _ = detect_speech_segments(audio_16k, 16000)
        atomic_write_json(vad_cache_path, {"vad_spans": [list(sp) for sp in vad_spans]})
    stage_timings["stage1_vad_ms"] = int((time.perf_counter() - t_vad) * 1000)

    # ---- Resolve Mode & Language ----
    eff_mode = task.resolve_effective_mode()
    req_lang = (
        task.language_override
        if task.language_override in {"zh", "en"}
        else alignment_cfg.language
    )

    if eff_mode == TaskMode.SRT_FIX and not (task.srt_path and Path(task.srt_path).is_file()):
        raise RuntimeError("当前选择为「修正已有 SRT」模式，但未提供有效的 SRT 字幕文件。")
    if eff_mode == TaskMode.SCRIPT_MATCH and not (task.script_path and Path(task.script_path).is_file()):
        raise RuntimeError("当前选择为「文稿匹配」模式，但未提供有效的剧本文稿 (.txt / .md) 文件。")

    # ---- Stage 2: ASR Recognition / Script Matching / Existing SRT ----
    t2 = time.perf_counter()
    raw_cues: list[SubtitleCue] = []
    asr_model_used = ""
    asr_segments: list[ASRSegment] = []
    script_match_res: ScriptMatchResult | None = None
    stage2_anomalies: list[AnomalyItem] = []
    detected_lang = req_lang if req_lang in {"zh", "en"} else "zh"
    source_key = ""

    if eff_mode in {TaskMode.ASR_GENERATE, TaskMode.SCRIPT_MATCH}:
        resolved_asr_id, sel_warns = select_asr_model_id(alignment_cfg.asr_model, req_lang, cache_dir)
        asr_key = stable_param_hash(
            {
                "audio_key": audio_key,
                "asr_model": resolved_asr_id,
                "lang": req_lang,
                "enhance": alignment_cfg.enhance_quiet_speech_for_asr,
                "algo_v": 2,
            }
        )
        asr_cache_path = ep_cache / f"asr_{asr_key}.json"
        force_asr = force_recompute and stage_action in {"full", "asr_only"}
        cached_asr = None if force_asr else load_asr_cache(asr_cache_path)

        if cached_asr is not None:
            asr_res = cached_asr
            cache_hits.append("asr")
            _p(0.32, f"命中阶段缓存：复用语音识别结果 ({asr_res.model_used})")
        else:
            _p(0.24, f"阶段 2/3 · 运行语音识别 ({resolved_asr_id})...")
            asr_res = run_asr_transcription(
                audio_16k,
                vad_spans,
                duration_ms,
                cache_dir=cache_dir,
                asr_pref=alignment_cfg.asr_model,
                language_pref=req_lang,
                device=alignment_cfg.device,
                enhance_quiet_speech=alignment_cfg.enhance_quiet_speech_for_asr,
                allow_vram_downgrade=alignment_cfg.allow_vram_auto_downgrade,
                unload_after=True,
                progress_cb=lambda f, m: _p(0.24 + 0.16 * f, m),
            )
            asr_res.warnings = sel_warns + asr_res.warnings
            save_asr_cache(asr_cache_path, asr_res)

        asr_segments = asr_res.segments
        asr_model_used = asr_res.model_used
        detected_lang = req_lang if req_lang in {"zh", "en"} else asr_res.detected_language
        for w_msg in asr_res.warnings:
            stage2_anomalies.append(
                AnomalyItem(
                    anomaly_type=AnomalyType.LOW_CONFIDENCE,
                    severity=AnomalySeverity.INFO,
                    start_ms=0,
                    end_ms=duration_ms,
                    cue_index=None,
                    message=w_msg,
                    suggestion="",
                )
            )

        if eff_mode == TaskMode.SCRIPT_MATCH:
            script_hash = compute_file_hash(task.script_path)
            sm_key = stable_param_hash({"asr_key": asr_key, "script_hash": script_hash, "algo_v": 2})
            sm_cache_path = ep_cache / f"script_match_{sm_key}.json"
            force_sm = force_recompute and stage_action in {"full", "script_match_only"}
            cached_sm = None if force_sm else load_script_match_cache(sm_cache_path)

            if cached_sm is not None:
                script_match_res = cached_sm
                cache_hits.append("script_match")
                _p(0.42, "命中阶段缓存：复用 ASR-文稿匹配结果")
            else:
                _p(0.41, "阶段 2/3 · 解析剧本文稿并执行局部序列匹配...")
                script_doc = parse_dialogue_script(task.script_path)
                if req_lang == "auto" and script_doc.turns:
                    detected_lang = script_doc.language
                script_match_res = match_asr_to_script(asr_segments, script_doc)
                save_script_match_cache(sm_cache_path, script_match_res)

            raw_cues = deepcopy(script_match_res.cues)
            stage2_anomalies.extend(script_match_res.anomalies)
            source_key = f"sm_{sm_key}"
        else:
            raw_cues = _asr_segments_to_coarse_cues(asr_segments)
            source_key = f"asr_{asr_key}"
    else:
        _p(0.26, "阶段 2/3 · 解析已有 SRT 字幕...")
        raw_cues = load_srt(task.srt_path)
        if req_lang == "auto" and raw_cues:
            detected_lang = detect_text_language(" ".join(c.raw_text for c in raw_cues))
        srt_hash = compute_file_hash(task.srt_path)
        scr_hash = compute_file_hash(task.script_path) if task.script_path else "none"
        source_key = stable_param_hash(
            {"srt": srt_hash, "scr": scr_hash, "offset_flag": alignment_cfg.detect_global_offset}
        )

    stage_timings["stage2_recognize_match_ms"] = int((time.perf_counter() - t2) * 1000)

    # Resolve language-appropriate template automatically when language="auto"
    eff_template = _select_template_for_language(
        template,
        detected_lang,
        task,
        media_info,
        templates_by_lang,
    )

    # Early exit if user requested ASR-only or Script-Match-only inspection
    if stage_action in {"asr_only", "script_match_only"}:
        preview_cues = deepcopy(raw_cues)
        for idx_c, c in enumerate(preview_cues, start=1):
            c.index = idx_c
        empty_report = _build_empty_audio_report(sr_full, channels_full, duration_ms)
        _p(1.0, f"已完成阶段 2（{eff_mode.label_zh}）：共 {len(preview_cues)} 条候选对白，可继续微调或进入精修导出。")
        return PipelineResult(
            episode_label=task.episode_label,
            media_info=media_info,
            raw_cues=raw_cues,
            processed_cues=preview_cues,
            anomalies=stage2_anomalies,
            audio_report=empty_report,
            extracted_wav_path=str(raw_wav_path),
            processed_wav_path="",
            vad_spans_ms=vad_spans,
            aligner_used="待执行强制对齐",
            nlp_used=alignment_cfg.nlp_backend,
            text_integrity_ok=True,
            task_mode_used=eff_mode.value,
            detected_language=detected_lang,
            asr_model_used=asr_model_used,
            asr_segments=[asdict(s) for s in asr_segments],
            script_match_report=script_match_res.to_report_dict() if script_match_res else {},
            cache_hits=cache_hits,
            stage_timings_ms=stage_timings,
            resource_usage=snapshot_resource_usage(cache_dir, work_dir),
        )

    # ---- Stage 3A & 3B: Forced Alignment + DP Semantic Layout ----
    t3_align = time.perf_counter()
    align_key = stable_param_hash(
        {
            "src": source_key,
            "backend": alignment_cfg.backend,
            "lang": detected_lang,
            "algo_v": 3,
        }
    )
    align_cache_path = ep_cache / f"align_{align_key}.json"
    force_align = force_recompute and stage_action in {"full", "align_only"}
    cached_align = None if force_align else load_forced_align_cache(align_cache_path)

    if cached_align is not None:
        align_stage: ForcedAlignmentStageResult = cached_align
        cache_hits.append("forced_align")
        _p(0.55, f"命中阶段缓存：复用强制对齐结果 ({align_stage.aligner_used})")
    else:
        _p(0.46, f"阶段 3/3 · 执行强制对齐 ({detected_lang.upper()})...")
        align_stage = run_forced_alignment_only(
            raw_cues=raw_cues,
            audio_16k=audio_16k,
            vad_spans=vad_spans,
            duration_ms=duration_ms,
            alignment_cfg=alignment_cfg,
            cache_dir=cache_dir,
            script_path=task.script_path or None,
            language=detected_lang,
            apply_script_turns=(eff_mode == TaskMode.SRT_FIX),
            skip_global_offset=(eff_mode != TaskMode.SRT_FIX),
            unload_after=True,
            progress_cb=lambda f, m: _p(f, m),
        )
        save_forced_align_cache(align_cache_path, align_stage)
    stage_timings["stage3_align_ms"] = int((time.perf_counter() - t3_align) * 1000)

    t3_layout = time.perf_counter()
    tpl_hash = stable_param_hash(eff_template.to_dict())
    layout_key = stable_param_hash(
        {
            "align_key": align_key,
            "tpl": tpl_hash,
            "nlp": alignment_cfg.nlp_backend,
            "fps": round(media_info.fps or 25.0, 3),
            "layout_algo_v": 2,
        }
    )
    layout_cache_path = ep_cache / f"layout_{layout_key}.json"
    force_layout = force_recompute and stage_action in {"full", "align_only", "relayout_only"}
    cached_layout = None if force_layout else load_layout_cache(layout_cache_path)

    if cached_layout is not None:
        align_res: AlignmentEngineResult = cached_layout
        cache_hits.append("layout")
        _p(0.68, "命中阶段缓存：复用智能断句与排版结果")
    else:
        _p(0.64, "阶段 3/3 · 执行智能断句、单双行排版与质量检查...")
        align_res = segment_aligned_windows_to_cues(
            raw_cues=raw_cues,
            align_stage=align_stage,
            audio_16k=audio_16k,
            vad_spans=vad_spans,
            duration_ms=duration_ms,
            template=eff_template,
            alignment_cfg=alignment_cfg,
            video_fps=media_info.fps or 25.0,
        )
        save_layout_cache(layout_cache_path, align_res)
    stage_timings["stage3_layout_ms"] = int((time.perf_counter() - t3_layout) * 1000)

    # ---- Stage 3C: Voice Loudness Leveling (48kHz) ----
    t3_lvl = time.perf_counter()
    cues_win_hash = (
        stable_param_hash([(c.start_ms, c.end_ms) for c in align_res.cues])
        if audio_cfg.use_subtitle_windows
        else "no_cues"
    )
    audio_cfg_hash = stable_param_hash(audio_cfg.to_dict())
    level_key = stable_param_hash(
        {
            "audio_key": audio_key,
            "cues": cues_win_hash,
            "cfg": audio_cfg_hash,
            "bits": export_cfg.wav_bit_depth,
        }
    )
    processed_wav_path = ep_cache / f"processed_{level_key}.wav"
    level_meta_path = ep_cache / f"processed_{level_key}.meta.json"
    force_level = force_recompute and stage_action in {"full", "relevel_only"}
    cached_level = (
        None
        if force_level or not processed_wav_path.is_file()
        else load_leveling_meta_cache(level_meta_path)
    )

    if cached_level is not None:
        cache_hits.append("audio_level")
        _p(0.80, "命中阶段缓存：复用 48kHz 智能音量均衡音频")
        level_report: AudioAnalysisReport = cached_level["report_obj"]
        level_vad_spans = cached_level["vad_spans_ms"]
        wf_times = cached_level["waveform_times_ms"]
        wf_before = cached_level["waveform_before_peaks"]
        wf_after = cached_level["waveform_after_peaks"]
        gain_times = cached_level["gain_curve_times_ms"]
        gain_db = cached_level["gain_curve_db"]
    else:
        _p(0.74, "阶段 3/3 · 执行智能人声音量均衡 (VAD + 字幕窗 + 动态增益 + 真峰值限制)...")
        tmp_proc_wav = ep_cache / f".tmp_processed_{level_key}.wav"
        try:
            level_out: LevelingOutput = process_voice_leveling(
                raw_wav_path,
                tmp_proc_wav,
                audio_cfg,
                cues=align_res.cues,
                bit_depth=export_cfg.wav_bit_depth,
                precomputed_vad_spans=vad_spans,
            )
            atomic_replace_file(tmp_proc_wav, processed_wav_path)
        except Exception:
            tmp_proc_wav.unlink(missing_ok=True)
            raise
        save_leveling_meta_cache(level_meta_path, level_out)
        level_report = level_out.report
        level_vad_spans = level_out.vad_spans_ms
        wf_times = level_out.waveform_times_ms
        wf_before = level_out.waveform_before_peaks
        wf_after = level_out.waveform_after_peaks
        gain_times = level_out.gain_curve_times_ms
        gain_db = level_out.gain_curve_db
    stage_timings["stage3_level_ms"] = int((time.perf_counter() - t3_lvl) * 1000)

    # ---- Stage 3D: Audio-Video Duration & Sync Verification ----
    proc_info = sf.info(str(processed_wav_path))
    proc_frames = int(proc_info.frames)
    proc_sr = int(proc_info.samplerate)
    proc_dur_ms = (proc_frames * 1000.0) / max(1, proc_sr)
    sync_diff_ms = round(proc_dur_ms - duration_ms, 2) if duration_ms > 0 else 0.0
    sample_exact = (proc_frames == len(sig_full))
    sync_verified = sample_exact and (duration_ms <= 0 or abs(sync_diff_ms) <= 80.0)

    # Merge anomalies cleanly
    all_anomalies: list[AnomalyItem] = list(stage2_anomalies) + list(align_res.anomalies)
    if not sync_verified:
        all_anomalies.insert(
            0,
            AnomalyItem(
                anomaly_type=AnomalyType.SYNC_CHECK_WARNING,
                severity=AnomalySeverity.WARNING,
                start_ms=0,
                end_ms=duration_ms,
                cue_index=None,
                message=f"音频同步检查提示：导出 WAV 时长与原视频存在 {sync_diff_ms:+.1f} ms 偏差（样本一致={sample_exact}）。",
                suggestion="请在剪辑软件中核对音轨头尾帧对齐位置。",
            ),
        )
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
    if level_report.bgm_or_sfx_risk:
        all_anomalies.insert(
            0,
            AnomalyItem(
                anomaly_type=AnomalyType.HIGH_BGM_RISK,
                severity=AnomalySeverity.WARNING,
                start_ms=0,
                end_ms=duration_ms,
                cue_index=None,
                message=level_report.risk_messages[0] if level_report.risk_messages else "检测到较高背景音/BGM 混音风险",
                suggestion="建议对比 A/B 试听，如背景起伏偏大可切换「轻度」模式。",
            ),
        )
    if level_report.over_compressed_risk:
        all_anomalies.insert(
            0,
            AnomalyItem(
                anomaly_type=AnomalyType.OVER_COMPRESSED_RISK,
                severity=AnomalySeverity.WARNING,
                start_ms=0,
                end_ms=duration_ms,
                cue_index=None,
                message=next(
                    (m for m in level_report.risk_messages if "压缩" in m),
                    "检测到原音频动态范围偏小（可能已经过重度压缩）",
                ),
                suggestion="建议适当降低「最大提升」或使用「轻度」模式。",
            ),
        )

    stage_timings["total_ms"] = int((time.perf_counter() - t_start_all) * 1000)
    res_snap = snapshot_resource_usage(cache_dir, work_dir)

    result = PipelineResult(
        episode_label=task.episode_label,
        media_info=media_info,
        raw_cues=raw_cues,
        processed_cues=align_res.cues,
        anomalies=all_anomalies,
        audio_report=level_report,
        extracted_wav_path=str(raw_wav_path),
        processed_wav_path=str(processed_wav_path),
        waveform_times_ms=wf_times,
        waveform_before_peaks=wf_before,
        waveform_after_peaks=wf_after,
        gain_curve_times_ms=gain_times,
        gain_curve_db=gain_db,
        vad_spans_ms=level_vad_spans,
        global_offset_ms=align_res.global_offset_ms,
        aligner_used=align_res.aligner_used,
        nlp_used=align_res.nlp_used,
        text_integrity_ok=align_res.text_integrity_ok,
        task_mode_used=eff_mode.value,
        detected_language=detected_lang,
        asr_model_used=asr_model_used,
        asr_segments=[asdict(s) for s in asr_segments],
        script_match_report=script_match_res.to_report_dict() if script_match_res else {},
        sync_verified=sync_verified,
        sync_diff_ms=sync_diff_ms,
        cache_hits=cache_hits,
        stage_timings_ms=stage_timings,
        resource_usage=res_snap,
    )

    # ---- Stage 3E: Export Deliverables ----
    _p(0.88, "阶段 3/3 · 正在导出达芬奇可用文件...")
    eff_export = deepcopy(export_cfg)
    if not eff_export.output_dir:
        eff_export.output_dir = str(Path(task.video_path).parent / "智能修正输出")
    export_episode_outputs(result, eff_export, progress_cb=lambda m: _p(0.94, m))

    # Enforce LRU cache quota protecting current episode cache dir
    prune_stage_cache(work_dir, max_size_mb=cache_max_size_mb, protect_dir=ep_cache)
    MODEL_POOL.unload_all()

    _p(1.0, f"处理完成：输出 {len(result.processed_cues)} 条字幕，异常提示 {len(result.anomalies)} 项。")
    return result
