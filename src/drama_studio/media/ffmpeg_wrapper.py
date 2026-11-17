"""FFmpeg and ffprobe integration for media inspection, audio extraction, and muxing."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

from drama_studio.models import AudioTrackInfo, MediaInfo, Orientation

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
SUPPORTED_SUBTITLE_EXTS = {".srt"}


def _subprocess_creation_flags() -> int:
    if sys.platform == "win32":
        return 0x08000000  # CREATE_NO_WINDOW
    return 0


def find_ffmpeg_binaries() -> tuple[Path, Path]:
    """Locate ffmpeg and ffprobe executables (bundled vendor first, then PATH)."""
    candidates: list[Path] = []

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        meipass = Path(getattr(sys, "_MEIPASS"))
        candidates.extend([meipass / "ffmpeg" / "bin", meipass / "vendor" / "ffmpeg" / "bin", meipass])

    exe_dir = Path(sys.executable).resolve().parent
    candidates.extend([exe_dir / "ffmpeg" / "bin", exe_dir / "vendor" / "ffmpeg" / "bin", exe_dir])

    project_root = Path(__file__).resolve().parents[3]
    candidates.extend([
        project_root / "vendor" / "ffmpeg" / "bin",
        project_root / "resources" / "ffmpeg" / "bin",
    ])

    env_dir = os.environ.get("DRAMA_STUDIO_FFMPEG_DIR")
    if env_dir:
        candidates.insert(0, Path(env_dir))

    for base in candidates:
        ff = base / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
        fp = base / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if ff.is_file() and fp.is_file():
            return ff, fp

    ff_sys = shutil.which("ffmpeg")
    fp_sys = shutil.which("ffprobe")
    if ff_sys and fp_sys:
        return Path(ff_sys), Path(fp_sys)

    raise FileNotFoundError(
        "未找到完整的 ffmpeg 与 ffprobe 可执行文件。请确认 vendor/ffmpeg/bin 目录或系统 PATH 配置。"
    )


def _run_tool(cmd: list[str], timeout_sec: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        creationflags=_subprocess_creation_flags(),
    )


def _parse_fps(rate_str: str) -> float:
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        num_s, den_s = rate_str.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
            return round(num / den, 3) if den > 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return round(float(rate_str), 3)
    except ValueError:
        return 0.0


def _compute_aspect_label(width: int, height: int) -> tuple[str, Orientation]:
    if width <= 0 or height <= 0:
        return "未知", Orientation.LANDSCAPE

    ratio = width / height
    if abs(ratio - (9 / 16)) < 0.04:
        aspect = "9:16"
    elif abs(ratio - (16 / 9)) < 0.04:
        aspect = "16:9"
    elif abs(ratio - 1.0) < 0.04:
        aspect = "1:1"
    elif abs(ratio - (3 / 4)) < 0.04:
        aspect = "3:4"
    elif abs(ratio - (4 / 3)) < 0.04:
        aspect = "4:3"
    else:
        g = math.gcd(width, height)
        aspect = f"{width // g}:{height // g}"

    if height > width * 1.03:
        orient = Orientation.PORTRAIT
    elif width > height * 1.03:
        orient = Orientation.LANDSCAPE
    else:
        orient = Orientation.SQUARE
    return aspect, orient


def _extract_rotation_degrees(stream: dict) -> int:
    tags = stream.get("tags") or {}
    if "rotate" in tags:
        try:
            return int(round(float(tags["rotate"])))
        except ValueError:
            pass
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"])))
            except ValueError:
                pass
    return 0


def probe_media(video_path: str | Path) -> MediaInfo:
    """Inspect video and audio tracks with ffprobe, including A/V start offset checks."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {path}")

    _, ffprobe_bin = find_ffmpeg_binaries()
    cmd = [
        str(ffprobe_bin),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = _run_tool(cmd, timeout_sec=30.0)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 检测失败 ({path.name}): {proc.stderr.strip()}")

    data = json.loads(proc.stdout)
    fmt = data.get("format") or {}
    streams = data.get("streams") or []

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    width = 0
    height = 0
    fps = 0.0
    fps_str = ""
    video_codec = ""
    video_start_ms = 0.0

    if video_streams:
        vs = video_streams[0]
        width = int(vs.get("width") or 0)
        height = int(vs.get("height") or 0)
        rotation = abs(_extract_rotation_degrees(vs)) % 180
        if rotation == 90:
            width, height = height, width
        fps_str = vs.get("avg_frame_rate") or vs.get("r_frame_rate") or ""
        fps = _parse_fps(fps_str)
        if fps == 0.0 and vs.get("r_frame_rate"):
            fps_str = vs["r_frame_rate"]
            fps = _parse_fps(fps_str)
        video_codec = str(vs.get("codec_name") or "")
        if vs.get("start_time") is not None:
            try:
                video_start_ms = float(vs["start_time"]) * 1000.0
            except ValueError:
                video_start_ms = 0.0

    duration_sec = 0.0
    if fmt.get("duration") is not None:
        try:
            duration_sec = float(fmt["duration"])
        except ValueError:
            duration_sec = 0.0
    if duration_sec <= 0 and video_streams and video_streams[0].get("duration"):
        try:
            duration_sec = float(video_streams[0]["duration"])
        except ValueError:
            pass
    duration_ms = max(0, int(round(duration_sec * 1000.0)))

    aspect_ratio, orientation = _compute_aspect_label(width, height)

    tracks: list[AudioTrackInfo] = []
    for idx, aus in enumerate(audio_streams):
        tags = aus.get("tags") or {}
        start_ms = 0.0
        if aus.get("start_time") is not None:
            try:
                start_ms = float(aus["start_time"]) * 1000.0
            except ValueError:
                start_ms = 0.0
        track_dur_ms = duration_ms
        if aus.get("duration") is not None:
            try:
                track_dur_ms = max(0, int(round(float(aus["duration"]) * 1000.0)))
            except ValueError:
                pass
        tracks.append(
            AudioTrackInfo(
                stream_index=int(aus.get("index", idx)),
                audio_index=idx,
                codec=str(aus.get("codec_name") or "unknown"),
                sample_rate=int(aus.get("sample_rate") or 48000),
                channels=int(aus.get("channels") or 2),
                channel_layout=str(aus.get("channel_layout") or ""),
                language=str(tags.get("language") or "und"),
                title=str(tags.get("title") or tags.get("handler_name") or ""),
                bit_rate=int(aus.get("bit_rate") or 0),
                start_time_ms=round(start_ms, 2),
                duration_ms=track_dur_ms,
            )
        )

    warnings: list[str] = []
    av_offset_ms = 0.0
    if not tracks:
        warnings.append("该视频文件不包含音轨，无法执行音频均衡与语音对齐。")
    else:
        t0 = tracks[0]
        av_offset_ms = round(t0.start_time_ms - video_start_ms, 2)
        if abs(av_offset_ms) >= 15.0:
            warnings.append(
                f"检测到音视频流起始时间基准偏移 ({av_offset_ms:+.1f} ms)，导出音轨将自动按视频时间零点对齐补偿。"
            )
        if duration_ms > 0 and t0.duration_ms > 0 and abs(t0.duration_ms - duration_ms) > 150:
            warnings.append(
                f"音轨时长 ({t0.duration_ms / 1000:.2f}s) 与视频时长 ({duration_ms / 1000:.2f}s) 存在差异，建议核对头尾同步。"
            )
        if len(tracks) > 1:
            warnings.append(f"检测到 {len(tracks)} 条音轨，请在左侧确认用于人声提取的目标音轨。")

    return MediaInfo(
        path=str(path),
        file_name=path.name,
        format_name=str(fmt.get("format_name") or path.suffix.lstrip(".")),
        duration_ms=duration_ms,
        width=width,
        height=height,
        fps=fps,
        fps_str=fps_str,
        aspect_ratio=aspect_ratio,
        orientation=orientation,
        video_codec=video_codec,
        video_start_time_ms=round(video_start_ms, 2),
        audio_tracks=tracks,
        selected_audio_index=0,
        av_offset_ms=av_offset_ms,
        warnings=warnings,
    )


def extract_audio_to_wav(
    video_path: str | Path,
    output_wav_path: str | Path,
    *,
    audio_stream_index: int = 0,
    sample_rate: int = 48000,
    channels: int | None = None,
    bit_depth: int = 24,
    sync_offset_ms: float = 0.0,
    target_duration_ms: int | None = None,
) -> Path:
    """Extract an audio track from video to WAV, preserving timeline sync with video PTS=0."""
    ffmpeg_bin, _ = find_ffmpeg_binaries()
    out = Path(output_wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    codec = "pcm_s24le" if bit_depth == 24 else "pcm_s16le"

    filter_parts: list[str] = ["aresample=async=1:first_pts=0"]
    # Compensate if audio starts later/earlier relative to video start time
    if sync_offset_ms > 5.0:
        delay_ms = int(round(sync_offset_ms))
        filter_parts.append(f"adelay=delays={delay_ms}:all=1")
    elif sync_offset_ms < -5.0:
        trim_sec = abs(sync_offset_ms) / 1000.0
        filter_parts.append(f"atrim=start={trim_sec:.6f},asetpts=PTS-STARTPTS")

    if target_duration_ms and target_duration_ms > 0:
        dur_sec = target_duration_ms / 1000.0
        filter_parts.append(f"apad=whole_dur={dur_sec:.6f},atrim=0:{dur_sec:.6f}")

    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i", str(video_path),
        "-map", f"0:a:{audio_stream_index}",
        "-vn",
        "-af", ",".join(filter_parts),
        "-ar", str(sample_rate),
    ]
    if channels is not None:
        cmd.extend(["-ac", str(channels)])
    cmd.extend(["-c:a", codec, str(out)])

    proc = _run_tool(cmd, timeout_sec=300.0)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 提取音频失败: {proc.stderr.strip()}")
    return out


def mux_audio_with_video(
    video_path: str | Path,
    wav_path: str | Path,
    output_mp4_path: str | Path,
    *,
    audio_bitrate: str = "256k",
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """Replace video's audio track with processed WAV using lossless video stream copy."""
    ffmpeg_bin, _ = find_ffmpeg_binaries()
    out = Path(output_mp4_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb("正在封装替换音轨的 MP4 (无损复制视频流)...")

    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-i", str(video_path),
        "-i", str(wav_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", "48000",
        "-movflags", "+faststart",
        "-shortest",
        str(out),
    ]
    proc = _run_tool(cmd, timeout_sec=600.0)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 封装 MP4 失败: {proc.stderr.strip()}")
    return out
