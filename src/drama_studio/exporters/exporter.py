"""Export deliverables compatible with DaVinci Resolve Studio 20 with non-overwriting safety."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Callable

from drama_studio.media.ffmpeg_wrapper import mux_audio_with_video
from drama_studio.models import ExportConfig, PipelineResult
from drama_studio.nlp.srt_io import save_srt


def build_report_payload(result: PipelineResult) -> dict:
    """Serialize full episode diagnostics, timing corrections, script matching, and resource telemetry."""
    return {
        "episode_label": result.episode_label,
        "source_video": result.media_info.path,
        "workflow": {
            "task_mode_used": result.task_mode_used,
            "detected_language": result.detected_language,
            "asr_model_used": result.asr_model_used,
            "cache_hits": result.cache_hits,
            "stage_timings_ms": result.stage_timings_ms,
        },
        "sync_check": {
            "sync_verified": result.sync_verified,
            "sync_diff_ms": round(result.sync_diff_ms, 2),
        },
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
        "script_match_report": result.script_match_report,
        "audio_report": asdict(result.audio_report),
        "resource_usage": result.resource_usage,
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


_SAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _sanitize_filename_token(text: str, max_len: int = 40) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", text.strip()).strip(" ._-")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip(" ._-")
    return cleaned


def _existing_prefix_belongs_to_other_source(out_dir: Path, prefix: str, src_video: Path) -> bool:
    """Return True if an existing export report under `prefix` came from a different source video."""
    rep_path = out_dir / f"{prefix}_修正报告.json"
    if not rep_path.is_file():
        return False
    try:
        data = json.loads(rep_path.read_text(encoding="utf-8"))
        prev_src = str(data.get("source_video") or "").strip()
        if not prev_src:
            return False
        return Path(prev_src).resolve() != src_video.resolve()
    except Exception:
        return False


def _build_source_qualified_prefix(base_prefix: str, src_video: Path, *, include_hash: bool = False) -> str:
    """Append project/parent folder or source stem so different projects sharing '第01集' never collide."""
    parts = [base_prefix]
    parent_tok = _sanitize_filename_token(src_video.parent.name, max_len=28)
    stem_tok = _sanitize_filename_token(src_video.stem, max_len=32)
    if parent_tok and parent_tok not in base_prefix:
        parts.append(parent_tok)
    if stem_tok and stem_tok != base_prefix and stem_tok not in base_prefix and stem_tok != parent_tok:
        parts.append(stem_tok)
    if include_hash or len(parts) == 1:
        short_hash = hashlib.sha1(str(src_video.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:6]
        parts.append(short_hash)
    return "_".join(parts)


def _resolve_non_overwriting_prefix(
    out_dir: Path,
    base_prefix: str,
    src_video: Path,
    export_cfg: ExportConfig,
    has_cues: bool,
    has_audio: bool,
) -> str:
    """Choose a collision-free prefix: isolate cross-project conflicts even in overwrite mode, version same-source retries."""

    def _planned_names(pfx: str) -> list[Path]:
        names: list[Path] = []
        if export_cfg.export_wav and has_audio:
            names.append(out_dir / f"{pfx}_修正音频.wav")
        if export_cfg.export_srt and has_cues:
            names.append(out_dir / f"{pfx}_修正字幕.srt")
        if export_cfg.export_report_json:
            names.append(out_dir / f"{pfx}_修正报告.json")
        if export_cfg.export_muxed_mp4 and has_audio:
            names.append(out_dir / f"{pfx}_音画合成.mp4")
        return names

    eff_base = base_prefix
    if _existing_prefix_belongs_to_other_source(out_dir, eff_base, src_video):
        cand_qual = _build_source_qualified_prefix(base_prefix, src_video, include_hash=False)
        if _existing_prefix_belongs_to_other_source(out_dir, cand_qual, src_video):
            cand_qual = _build_source_qualified_prefix(base_prefix, src_video, include_hash=True)
        eff_base = cand_qual

    if export_cfg.overwrite_policy == "overwrite":
        return eff_base

    cand = eff_base
    ver = 2
    while any(p.exists() for p in _planned_names(cand)):
        cand = f"{eff_base}_v{ver}"
        ver += 1
    return cand


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

    base_prefix = _sanitize_filename_token(result.episode_label or src_video.stem, max_len=80) or "episode"
    has_audio = bool(result.processed_wav_path and Path(result.processed_wav_path).is_file())
    has_cues = bool(result.processed_cues)
    prefix = _resolve_non_overwriting_prefix(out_dir, base_prefix, src_video, export_cfg, has_cues, has_audio)
    exported: dict[str, str] = {}

    def _safe_target(target: Path) -> Path:
        if target.resolve() == src_video.resolve():
            return out_dir / f"{prefix}_输出{target.suffix}"
        return target

    if export_cfg.export_wav and has_audio:
        wav_target = _safe_target(out_dir / f"{prefix}_修正音频.wav")
        src_processed = Path(result.processed_wav_path)
        if src_processed.resolve() != wav_target.resolve():
            shutil.copy2(src_processed, wav_target)
        exported["wav"] = str(wav_target)
        if progress_cb:
            progress_cb(f"已导出修正音频: {wav_target.name}")

    if export_cfg.export_srt:
        if has_cues:
            srt_target = _safe_target(out_dir / f"{prefix}_修正字幕.srt")
            save_srt(result.processed_cues, srt_target, enforce_resolve_safe=True)
            exported["srt"] = str(srt_target)
            if progress_cb:
                progress_cb(f"已导出修正字幕 (DaVinci Resolve UTF-8无BOM/LF): {srt_target.name}")
        elif progress_cb:
            progress_cb("当前无字幕条目，跳过 SRT 导出。")

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

    if export_cfg.export_muxed_mp4 and has_audio:
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
