"""Export deliverables compatible with DaVinci Resolve Studio 20."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Callable

from drama_studio.media.ffmpeg_wrapper import mux_audio_with_video
from drama_studio.models import ExportConfig, PipelineResult
from drama_studio.nlp.srt_io import save_srt


def build_report_payload(result: PipelineResult) -> dict:
    """Serialize full episode diagnostics, timing corrections, and anomalies to JSON-friendly dict."""
    return {
        "episode_label": result.episode_label,
        "source_video": result.media_info.path,
        "video_info": {
            "width": result.media_info.width,
            "height": result.media_info.height,
            "fps": result.media_info.fps,
            "aspect_ratio": result.media_info.aspect_ratio,
            "orientation": result.media_info.orientation.value,
            "duration_ms": result.media_info.duration_ms,
            "av_offset_ms": result.media_info.av_offset_ms,
            "selected_audio_index": result.media_info.selected_audio_index,
        },
        "alignment": {
            "aligner_used": result.aligner_used,
            "nlp_used": result.nlp_used,
            "global_offset_ms": result.global_offset_ms,
            "text_integrity_ok": result.text_integrity_ok,
            "raw_cue_count": len(result.raw_cues),
            "processed_cue_count": len(result.processed_cues),
        },
        "audio_report": asdict(result.audio_report),
        "anomalies": [a.to_dict() for a in result.anomalies],
        "cues": [
            {
                "index": c.index,
                "start_ms": c.start_ms,
                "end_ms": c.end_ms,
                "orig_start_ms": c.orig_start_ms,
                "orig_end_ms": c.orig_end_ms,
                "text": c.effective_text,
                "raw_text": c.raw_text,
                "confidence": round(c.confidence, 3),
                "flags": c.anomaly_flags,
            }
            for c in result.processed_cues
        ],
    }


def export_episode_outputs(
    result: PipelineResult,
    export_cfg: ExportConfig,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Export DaVinci Resolve-ready WAV, SRT, JSON report, and optional remuxed MP4 without overwriting inputs."""
    src_video = Path(result.media_info.path)
    out_dir = Path(export_cfg.output_dir) if export_cfg.output_dir else (src_video.parent / "智能修正输出")
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = result.episode_label or src_video.stem
    exported: dict[str, str] = {}

    def _safe_target(target: Path) -> Path:
        # Never overwrite original source files
        if target.resolve() == src_video.resolve():
            return out_dir / f"{prefix}_输出{target.suffix}"
        return target

    if export_cfg.export_wav:
        wav_target = _safe_target(out_dir / f"{prefix}_修正音频.wav")
        src_processed = Path(result.processed_wav_path)
        if src_processed.resolve() != wav_target.resolve():
            shutil.copy2(src_processed, wav_target)
        exported["wav"] = str(wav_target)
        if progress_cb:
            progress_cb(f"已导出修正音频: {wav_target.name}")

    if export_cfg.export_srt:
        srt_target = _safe_target(out_dir / f"{prefix}_修正字幕.srt")
        save_srt(result.processed_cues, srt_target, enforce_resolve_safe=True)
        exported["srt"] = str(srt_target)
        if progress_cb:
            progress_cb(f"已导出修正字幕 (DaVinci Resolve UTF-8无BOM/LF): {srt_target.name}")

    if export_cfg.export_report_json:
        rep_target = _safe_target(out_dir / f"{prefix}_修正报告.json")
        payload = build_report_payload(result)
        rep_target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        exported["report"] = str(rep_target)
        if progress_cb:
            progress_cb(f"已导出修正报告: {rep_target.name}")

    if export_cfg.export_muxed_mp4:
        mp4_target = _safe_target(out_dir / f"{prefix}_音画合成.mp4")
        wav_for_mux = exported.get("wav") or result.processed_wav_path
        mux_audio_with_video(
            src_video,
            wav_for_mux,
            mp4_target,
            progress_cb=progress_cb,
        )
        exported["mp4"] = str(mp4_target)

    result.exported_files = exported
    return exported
