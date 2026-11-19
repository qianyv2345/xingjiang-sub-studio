"""Background worker threads for non-blocking GUI media probing, AI processing, and audio re-leveling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import traceback

from PySide6.QtCore import QThread, Signal

from drama_studio.align.asr_engine import MissingASRModelError
from drama_studio.align.engine import MissingAlignerModelError, relayout_cues_with_template
from drama_studio.audio.leveler import process_voice_leveling
from drama_studio.audio.vad import to_16k_mono
from drama_studio.media.ffmpeg_wrapper import probe_media
from drama_studio.models import (
    AlignmentConfig,
    AudioLevelingConfig,
    EpisodeTask,
    ExportConfig,
    SubtitleCue,
    SubtitleTemplate,
)
from drama_studio.pipeline import CancellationToken, StageAction, process_single_episode


class MediaProbeWorker(QThread):
    """Probes imported video files in the background using ffprobe."""

    probed = Signal(str, object)       # task_id, MediaInfo
    probeFailed = Signal(str, str)     # task_id, error

    def __init__(self, tasks: list[EpisodeTask]) -> None:
        super().__init__()
        self.tasks = list(tasks)

    def run(self) -> None:
        for task in self.tasks:
            if self.isInterruptionRequested():
                return
            try:
                info = probe_media(task.video_path)
                self.probed.emit(task.task_id, info)
            except Exception as exc:
                self.probeFailed.emit(task.task_id, str(exc))


class ProcessingWorker(QThread):
    """Executes single-episode or batch intelligent subtitle + audio correction off the UI thread."""

    taskProgress = Signal(str, float, str)  # task_id, progress (0..1), message
    taskCompleted = Signal(str, object)     # task_id, PipelineResult
    taskFailed = Signal(str, str)           # task_id, error_message
    missingModel = Signal(str, str, str)    # task_id, recommended_model_id, error_message
    allFinished = Signal(bool)              # cancelled_flag
    logMessage = Signal(str)

    def __init__(
        self,
        jobs: list[tuple[EpisodeTask, SubtitleTemplate, AudioLevelingConfig]],
        alignment_cfg: AlignmentConfig,
        export_cfg: ExportConfig,
        cache_dir: Path,
        work_root: Path,
        *,
        stage_action: StageAction = "full",
        force_recompute: bool = False,
        templates_by_lang: dict[str, SubtitleTemplate] | None = None,
        cache_max_size_mb: int = 4096,
    ) -> None:
        super().__init__()
        self.jobs = jobs
        self.alignment_cfg = deepcopy(alignment_cfg)
        self.export_cfg = deepcopy(export_cfg)
        self.cache_dir = cache_dir
        self.work_root = work_root
        self.stage_action: StageAction = stage_action
        self.force_recompute = force_recompute
        self.templates_by_lang = deepcopy(templates_by_lang) if templates_by_lang else None
        self.cache_max_size_mb = cache_max_size_mb
        self.cancel_token = CancellationToken()

    def request_cancel(self) -> None:
        self.cancel_token.cancel()
        self.requestInterruption()

    def run(self) -> None:
        for task, tpl, audio_cfg in self.jobs:
            if self.cancel_token.is_cancelled:
                self.allFinished.emit(True)
                return

            tid = task.task_id
            eff_mode = task.resolve_effective_mode().label_zh
            self.logMessage.emit(
                f"=== 开始处理：{task.episode_label} ({Path(task.video_path).name}) · 模式={eff_mode} · 动作={self.stage_action} ==="
            )
            self.taskProgress.emit(tid, 0.02, "准备任务...")

            try:
                def _prog(p: float, msg: str, _tid: str = tid) -> None:
                    self.taskProgress.emit(_tid, p, msg)
                    self.logMessage.emit(f"[{task.episode_label}] {msg}")

                result = process_single_episode(
                    task=task,
                    template=tpl,
                    audio_cfg=audio_cfg,
                    alignment_cfg=self.alignment_cfg,
                    export_cfg=self.export_cfg,
                    cache_dir=self.cache_dir,
                    work_dir=self.work_root,
                    stage_action=self.stage_action,
                    force_recompute=self.force_recompute,
                    templates_by_lang=self.templates_by_lang,
                    cache_max_size_mb=self.cache_max_size_mb,
                    cancel_token=self.cancel_token,
                    progress_cb=_prog,
                )
                self.taskCompleted.emit(tid, result)
                hits_note = f" · 缓存命中={','.join(result.cache_hits)}" if result.cache_hits else ""
                self.logMessage.emit(
                    f"=== 完成处理：{task.episode_label} ({result.detected_language.upper()}{hits_note}) ==="
                )
            except InterruptedError:
                self.logMessage.emit(f"[{task.episode_label}] 用户取消任务。")
                self.allFinished.emit(True)
                return
            except (MissingASRModelError, MissingAlignerModelError) as exc:
                self.logMessage.emit(f"[{task.episode_label}] 缺少所需 AI 模型：{exc}")
                self.missingModel.emit(tid, exc.recommended_model_id, str(exc))
                self.taskFailed.emit(tid, str(exc))
            except Exception as exc:
                tb = traceback.format_exc()
                self.logMessage.emit(f"[{task.episode_label}] 错误：{exc}\n{tb}")
                self.taskFailed.emit(tid, str(exc))

        self.allFinished.emit(self.cancel_token.is_cancelled)


class AudioRelevelWorker(QThread):
    """Fast re-application of voice leveling when user tunes DSP params or manual gain points."""

    finishedOk = Signal(str, object)   # task_id, LevelingOutput
    failed = Signal(str, str)

    def __init__(
        self,
        task_id: str,
        extracted_wav_path: str,
        output_wav_path: str,
        audio_cfg: AudioLevelingConfig,
        cues: list[SubtitleCue],
        vad_spans_ms: list[tuple[int, int]],
        bit_depth: int = 24,
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.extracted_wav_path = extracted_wav_path
        self.output_wav_path = output_wav_path
        self.audio_cfg = deepcopy(audio_cfg)
        self.cues = deepcopy(cues)
        self.vad_spans_ms = list(vad_spans_ms)
        self.bit_depth = bit_depth

    def run(self) -> None:
        try:
            out = process_voice_leveling(
                self.extracted_wav_path,
                self.output_wav_path,
                self.audio_cfg,
                cues=self.cues,
                bit_depth=self.bit_depth,
                precomputed_vad_spans=self.vad_spans_ms,
            )
            self.finishedOk.emit(self.task_id, out)
        except Exception as exc:
            self.failed.emit(self.task_id, str(exc))


class RelayoutWorker(QThread):
    """Non-blocking background relayout & WAV 16k conversion so GUI stays responsive."""

    finishedOk = Signal(str, object, object, object)  # task_id, before_cues, new_cues, audio_16k
    failed = Signal(str, str)

    def __init__(
        self,
        task_id: str,
        raw_cues: list[SubtitleCue],
        template: SubtitleTemplate,
        *,
        duration_ms: int,
        vad_spans_ms: list[tuple[int, int]],
        wav_path: str,
        cached_16k: object = None,
        video_fps: float = 25.0,
    ) -> None:
        super().__init__()
        self.task_id = task_id
        self.raw_cues = deepcopy(raw_cues)
        self.template = deepcopy(template)
        self.duration_ms = duration_ms
        self.vad_spans_ms = list(vad_spans_ms)
        self.wav_path = wav_path
        self.cached_16k = cached_16k
        self.video_fps = video_fps

    def run(self) -> None:
        try:
            import numpy as np
            import soundfile as sf

            a16 = self.cached_16k
            if a16 is None:
                if self.wav_path and Path(self.wav_path).is_file():
                    sig, sr = sf.read(self.wav_path, dtype="float32")
                    a16 = to_16k_mono(sig, sr)
                else:
                    a16 = np.zeros(0, dtype=np.float32)

            new_cues, _, _ = relayout_cues_with_template(
                self.raw_cues,
                self.template,
                duration_ms=self.duration_ms,
                vad_spans=self.vad_spans_ms,
                audio_16k=a16,
                video_fps=self.video_fps,
            )
            self.finishedOk.emit(self.task_id, self.raw_cues, new_cues, a16)
        except Exception as exc:
            self.failed.emit(self.task_id, str(exc))
