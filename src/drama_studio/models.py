"""Core data models for drama_studio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal
import uuid


class Orientation(str, Enum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    SQUARE = "square"

    @property
    def label_zh(self) -> str:
        return {
            Orientation.PORTRAIT: "竖屏",
            Orientation.LANDSCAPE: "横屏",
            Orientation.SQUARE: "方屏",
        }[self]


class AnomalySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AnomalyType(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    OVER_CHAR_LIMIT = "over_char_limit"
    OVER_CPS = "over_cps"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    TIMELINE_OVERLAP = "timeline_overlap"
    OUT_OF_BOUNDS = "out_of_bounds"
    SILENT_REGION = "silent_region"
    TEXT_MISMATCH = "text_mismatch"
    AUDIO_CLIPPING = "audio_clipping"
    HIGH_BGM_RISK = "high_bgm_risk"
    OVER_COMPRESSED_RISK = "over_compressed_risk"
    AV_OFFSET_WARNING = "av_offset_warning"


@dataclass
class AudioTrackInfo:
    """Metadata for a single audio stream inside a video file."""

    stream_index: int
    audio_index: int
    codec: str
    sample_rate: int
    channels: int
    channel_layout: str = ""
    language: str = "und"
    title: str = ""
    bit_rate: int = 0
    start_time_ms: float = 0.0
    duration_ms: int = 0

    def display_label(self) -> str:
        parts = [f"音轨 #{self.audio_index + 1} (流 {self.stream_index})"]
        parts.append(f"{self.codec.upper()}")
        parts.append(f"{self.sample_rate}Hz/{self.channels}ch")
        if self.language and self.language != "und":
            parts.append(self.language)
        if self.title:
            parts.append(self.title)
        return " · ".join(parts)


@dataclass
class MediaInfo:
    """Video and audio stream metadata detected via ffprobe."""

    path: str
    file_name: str
    format_name: str
    duration_ms: int
    width: int
    height: int
    fps: float
    fps_str: str
    aspect_ratio: str
    orientation: Orientation
    video_codec: str = ""
    video_start_time_ms: float = 0.0
    audio_tracks: list[AudioTrackInfo] = field(default_factory=list)
    selected_audio_index: int = 0
    av_offset_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def selected_track(self) -> AudioTrackInfo | None:
        if not self.audio_tracks:
            return None
        idx = max(0, min(self.selected_audio_index, len(self.audio_tracks) - 1))
        return self.audio_tracks[idx]


@dataclass
class CharTimestamp:
    """Character-level timestamp and alignment confidence."""

    char: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    interpolated: bool = False


@dataclass
class SubtitleCue:
    """Represents a single subtitle block before/after processing."""

    index: int
    start_ms: int
    end_ms: int
    raw_text: str
    display_text: str = ""
    orig_start_ms: int | None = None
    orig_end_ms: int | None = None
    confidence: float = 1.0
    char_timings: list[CharTimestamp] = field(default_factory=list)
    anomaly_flags: list[str] = field(default_factory=list)
    source_cue_indices: list[int] = field(default_factory=list)
    locked: bool = False
    turn_id: int = 0

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def effective_text(self) -> str:
        return self.display_text if self.display_text else self.raw_text

    def lines(self) -> list[str]:
        return [ln for ln in self.effective_text.splitlines() if ln.strip() != ""]


@dataclass
class SubtitleTemplate:
    """Customizable subtitle layout and segmentation template."""

    template_id: str
    name: str
    aspect_preset: Literal["9:16", "16:9", "custom"] = "9:16"
    max_chars_per_line: int = 9
    max_lines: int = 1
    smart_semantic_split: bool = True
    remove_punctuation: bool = True
    width_mode: Literal["visual", "strict"] = "visual"
    min_duration_ms: int = 500
    max_duration_ms: int = 5000
    max_cps: float = 14.0
    min_gap_ms: int = 20
    padding_unit: Literal["frames", "ms"] = "frames"
    lead_frames: int = 3
    tail_frames: int = 3
    lead_offset_ms: int = 120
    tail_offset_ms: int = 120
    bridge_sentence_gaps: bool = True
    bridge_max_gap_ms: int = 550
    auto_merge_short: bool = True
    short_merge_max_chars: int = 3
    short_merge_max_gap_ms: int = 350
    auto_split_long: bool = True
    is_builtin: bool = False

    def resolve_padding_ms(self, fps: float | None = None) -> tuple[int, int]:
        """Resolve head/tail breathing room into milliseconds (frames mode converts via actual video FPS)."""
        if self.padding_unit == "frames":
            eff_fps = float(fps) if (fps is not None and fps >= 10.0) else 25.0
            frame_ms = 1000.0 / eff_fps
            return int(round(self.lead_frames * frame_ms)), int(round(self.tail_frames * frame_ms))
        return int(self.lead_offset_ms), int(self.tail_offset_ms)

    @classmethod
    def preset_portrait_9_16(cls) -> SubtitleTemplate:
        return cls(
            template_id="preset_9_16",
            name="竖屏短剧默认 (9:16)",
            aspect_preset="9:16",
            max_chars_per_line=9,
            max_lines=1,
            smart_semantic_split=True,
            remove_punctuation=True,
            width_mode="visual",
            min_duration_ms=450,
            max_duration_ms=4500,
            max_cps=14.0,
            min_gap_ms=20,
            padding_unit="frames",
            lead_frames=3,
            tail_frames=3,
            lead_offset_ms=120,
            tail_offset_ms=120,
            bridge_sentence_gaps=True,
            bridge_max_gap_ms=550,
            auto_merge_short=True,
            short_merge_max_chars=3,
            short_merge_max_gap_ms=350,
            auto_split_long=True,
            is_builtin=True,
        )

    @classmethod
    def preset_landscape_16_9(cls) -> SubtitleTemplate:
        return cls(
            template_id="preset_16_9",
            name="横屏短剧默认 (16:9)",
            aspect_preset="16:9",
            max_chars_per_line=20,
            max_lines=1,
            smart_semantic_split=True,
            remove_punctuation=True,
            width_mode="visual",
            min_duration_ms=500,
            max_duration_ms=5500,
            max_cps=14.0,
            min_gap_ms=20,
            padding_unit="frames",
            lead_frames=3,
            tail_frames=3,
            lead_offset_ms=120,
            tail_offset_ms=120,
            bridge_sentence_gaps=True,
            bridge_max_gap_ms=550,
            auto_merge_short=True,
            short_merge_max_chars=3,
            short_merge_max_gap_ms=350,
            auto_split_long=True,
            is_builtin=True,
        )

    def clone(self, new_name: str | None = None, template_id: str | None = None) -> SubtitleTemplate:
        data = asdict(self)
        data["is_builtin"] = False
        data["template_id"] = template_id or f"tpl_{uuid.uuid4().hex[:8]}"
        if new_name:
            data["name"] = new_name
        return SubtitleTemplate(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubtitleTemplate:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


@dataclass
class ManualGainPoint:
    """User-defined local volume adjustment point (in dB)."""

    time_ms: int
    gain_db: float
    radius_ms: int = 350


@dataclass
class AudioLevelingConfig:
    """Parameters controlling intelligent vocal loudness leveling."""

    mode: Literal["light", "standard", "strong", "custom"] = "standard"
    target_lufs: float = -16.0
    true_peak_ceiling_dbtp: float = -1.5
    max_boost_db: float = 18.0
    max_cut_db: float = 10.0
    rms_window_ms: float = 320.0
    smooth_window_ms: float = 180.0
    emotion_preserve: float = 0.35
    compression_ratio: float = 2.4
    noise_gate_db: float = -48.0
    highpass_hz: float = 75.0
    use_subtitle_windows: bool = True
    manual_points: list[ManualGainPoint] = field(default_factory=list)

    @classmethod
    def from_mode(cls, mode: Literal["light", "standard", "strong"]) -> AudioLevelingConfig:
        if mode == "light":
            return cls(
                mode="light",
                target_lufs=-16.0,
                true_peak_ceiling_dbtp=-1.5,
                max_boost_db=10.0,
                max_cut_db=6.0,
                rms_window_ms=450.0,
                smooth_window_ms=260.0,
                emotion_preserve=0.55,
                compression_ratio=1.6,
                noise_gate_db=-46.0,
                highpass_hz=70.0,
            )
        if mode == "strong":
            return cls(
                mode="strong",
                target_lufs=-15.0,
                true_peak_ceiling_dbtp=-1.2,
                max_boost_db=24.0,
                max_cut_db=14.0,
                rms_window_ms=240.0,
                smooth_window_ms=130.0,
                emotion_preserve=0.18,
                compression_ratio=3.5,
                noise_gate_db=-52.0,
                highpass_hz=80.0,
            )
        return cls(mode="standard")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioLevelingConfig:
        pts_raw = data.get("manual_points", [])
        pts = [ManualGainPoint(**p) if isinstance(p, dict) else p for p in pts_raw]
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid and k != "manual_points"}
        filtered["manual_points"] = pts
        return cls(**filtered)


@dataclass
class AudioAnalysisReport:
    """Pre/post loudness and risk diagnostics for an audio stream."""

    sample_rate: int
    channels: int
    duration_ms: int
    speech_ratio: float
    before_lufs: float
    after_lufs: float
    before_short_term_min_lufs: float
    before_short_term_max_lufs: float
    after_short_term_min_lufs: float
    after_short_term_max_lufs: float
    before_lra_lu: float
    after_lra_lu: float
    before_peak_dbfs: float
    after_peak_dbfs: float
    before_true_peak_dbtp: float
    after_true_peak_dbtp: float
    noise_floor_dbfs: float
    estimated_snr_db: float
    max_applied_gain_db: float
    min_applied_gain_db: float
    bgm_or_sfx_risk: bool = False
    over_compressed_risk: bool = False
    risk_messages: list[str] = field(default_factory=list)


@dataclass
class AnomalyItem:
    """Quality-check issue flagged for human review."""

    anomaly_type: AnomalyType
    severity: AnomalySeverity
    start_ms: int
    end_ms: int
    cue_index: int | None
    message: str
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["anomaly_type"] = self.anomaly_type.value
        d["severity"] = self.severity.value
        return d


@dataclass
class AlignmentConfig:
    """Forced alignment engine settings."""

    backend: Literal["auto", "funasr", "qwen3", "vad_fallback"] = "auto"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    low_confidence_threshold: float = 0.55
    detect_global_offset: bool = True
    nlp_backend: Literal["auto", "jieba", "hanlp"] = "jieba"


@dataclass
class ExportConfig:
    """Output file options."""

    output_dir: str = ""
    export_wav: bool = True
    export_srt: bool = True
    export_report_json: bool = True
    export_muxed_mp4: bool = False
    wav_bit_depth: Literal[16, 24] = 24
    wav_sample_rate: int = 48000


@dataclass
class EpisodeTask:
    """Represents one episode pair (video + SRT) in batch or single view."""

    task_id: str
    episode_label: str
    video_path: str
    srt_path: str = ""
    script_path: str = ""
    media_info: MediaInfo | None = None
    selected_audio_index: int = 0
    template_override: SubtitleTemplate | None = None
    audio_config_override: AudioLevelingConfig | None = None
    status: Literal["pending", "running", "done", "error", "cancelled"] = "pending"
    progress: float = 0.0
    status_message: str = "等待处理"
    error_message: str = ""


@dataclass
class ProjectConfig:
    """Project-level persistent settings."""

    project_id: str
    project_name: str
    active_template: SubtitleTemplate = field(default_factory=SubtitleTemplate.preset_portrait_9_16)
    audio_config: AudioLevelingConfig = field(default_factory=lambda: AudioLevelingConfig.from_mode("standard"))
    alignment_config: AlignmentConfig = field(default_factory=AlignmentConfig)
    export_config: ExportConfig = field(default_factory=ExportConfig)
    batch_use_unified_template: bool = True
    last_output_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "active_template": self.active_template.to_dict(),
            "audio_config": self.audio_config.to_dict(),
            "alignment_config": asdict(self.alignment_config),
            "export_config": asdict(self.export_config),
            "batch_use_unified_template": self.batch_use_unified_template,
            "last_output_dir": self.last_output_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConfig:
        return cls(
            project_id=data.get("project_id", "default"),
            project_name=data.get("project_name", "默认短剧项目"),
            active_template=SubtitleTemplate.from_dict(data["active_template"])
            if "active_template" in data
            else SubtitleTemplate.preset_portrait_9_16(),
            audio_config=AudioLevelingConfig.from_dict(data["audio_config"])
            if "audio_config" in data
            else AudioLevelingConfig.from_mode("standard"),
            alignment_config=AlignmentConfig(**data["alignment_config"])
            if "alignment_config" in data
            else AlignmentConfig(),
            export_config=ExportConfig(**data["export_config"])
            if "export_config" in data
            else ExportConfig(),
            batch_use_unified_template=data.get("batch_use_unified_template", True),
            last_output_dir=data.get("last_output_dir", ""),
        )


@dataclass
class PipelineResult:
    """Complete output of processing a single episode."""

    episode_label: str
    media_info: MediaInfo
    raw_cues: list[SubtitleCue]
    processed_cues: list[SubtitleCue]
    anomalies: list[AnomalyItem]
    audio_report: AudioAnalysisReport
    extracted_wav_path: str
    processed_wav_path: str
    waveform_times_ms: list[int] = field(default_factory=list)
    waveform_before_peaks: list[float] = field(default_factory=list)
    waveform_after_peaks: list[float] = field(default_factory=list)
    gain_curve_times_ms: list[int] = field(default_factory=list)
    gain_curve_db: list[float] = field(default_factory=list)
    vad_spans_ms: list[tuple[int, int]] = field(default_factory=list)
    global_offset_ms: int = 0
    aligner_used: str = ""
    nlp_used: str = ""
    text_integrity_ok: bool = True
    exported_files: dict[str, str] = field(default_factory=dict)
