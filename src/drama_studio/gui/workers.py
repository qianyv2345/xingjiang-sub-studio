"""Background worker threads for non-blocking GUI media probing, AI processing, and audio re-leveling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading
import traceback

from PySide6.QtCore import QThread, Signal

from drama_studio.audio.leveler import LevelingOutput, process_voice_leveling
from drama_studio.media.ffmpeg_wrapper import probe_media
from drama_studio.models import (
    AlignmentConfig,
    AudioLevelingConfig,
    EpisodeTask,
    ExportConfig,
    MediaInfo,
    PipelineResult,
    SubtitleCue,
    SubtitleTemplate,
)
from drama_studio.pipeline import CancellationToken, process_single_episode


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
    allFinished = Signal(bool)              # cancelled_flag
    logMessage = Signal(str)

    def __init__(
        self,
        jobs: list[tuple[EpisodeTask, SubtitleTemplate, AudioLevelingConfig]],
        alignment_cfg: AlignmentConfig,
        export_cfg: ExportConfig,
        cache_dir: Path,
        work_root: Path,
    ) -> None:
        super().__init__()
        self.jobs = jobs
        self.alignment_cfg = deepcopy(alignment_cfg)
        self.export_cfg = deepcopy(export_cfg)
        self.cache_dir = cache_dir
        self.work_root = work_root
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
            self.logMessage.emit(f"=== 开始处理：{task.episode_label} ({Path(task.video_path).name}) ===")
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
                    work_dir=self.work_root / tid,
                    cancel_token=self.cancel_token,
                    progress_cb=_prog,
                )
                self.taskCompleted.emit(tid, result)
                self.logMessage.emit(f"=== 完成处理：{task.episode_label} ===")
            except InterruptedError:
                self.logMessage.emit(f"[{task.episode_label}] 用户取消任务。")
                self.allFinished.emit(True)
                return
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
