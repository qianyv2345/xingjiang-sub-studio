"""Main application window with Light Neomorphism & Aurora Dark Glassmorphism themes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QUrl
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QKeyEvent,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPixmap,
    QRadialGradient,
    QUndoStack,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from drama_studio import __app_name__, __brand_name__
from drama_studio.align.engine import relayout_cues_with_template, run_quality_checks
from drama_studio.audio.vad import to_16k_mono
from drama_studio.config import ConfigStore, default_app_data_dir
from drama_studio.exporters.exporter import export_episode_outputs
from drama_studio.gui.theme import (
    THEME_ORDER,
    THEMES,
    build_studio_qss,
    get_palette,
    normalize_theme_mode,
    resolve_asset_path,
)
from drama_studio.gui.undo_commands import ReplaceCuesCommand, ReplaceManualGainCommand
from drama_studio.gui.widgets.anomaly_log_panel import AnomalyAndLogPanel
from drama_studio.gui.widgets.audio_panel import AudioPanel
from drama_studio.gui.widgets.glass_card import FrostedHeaderBar
from drama_studio.gui.widgets.import_panel import ImportBatchPanel
from drama_studio.gui.widgets.model_manager_dialog import ModelManagerDialog
from drama_studio.gui.widgets.subtitle_editor import SubtitleEditorPanel
from drama_studio.gui.widgets.template_panel import TemplateSettingsPanel
from drama_studio.gui.widgets.themed_controls import ThemedComboBox, apply_win11_dwm_chrome
from drama_studio.gui.widgets.waveform_timeline import WaveformTimelineWidget
from drama_studio.gui.workers import AudioRelevelWorker, MediaProbeWorker, ProcessingWorker
from drama_studio.media.pairing import assign_script_files_to_tasks, pair_media_files
from drama_studio.models import (
    AudioLevelingConfig,
    EpisodeTask,
    ManualGainPoint,
    MediaInfo,
    Orientation,
    PipelineResult,
    SubtitleCue,
    SubtitleTemplate,
)
from drama_studio.nlp.srt_io import normalize_cues_for_resolve


class AuroraCanvasWidget(QWidget):
    """Paints a luminous pastel mesh canvas behind translucent frosted glass panels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.theme_mode: str = "light"

    def set_theme(self, mode: str) -> None:
        self.theme_mode = normalize_theme_mode(mode)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = float(self.width()), float(self.height())
        pal = get_palette(self.theme_mode)

        base = QLinearGradient(0, 0, w, h)
        if pal.is_dark:
            base.setColorAt(0.0, QColor("#12151C"))
            base.setColorAt(0.55, QColor("#181C25"))
            base.setColorAt(1.0, QColor("#13161E"))
            c1 = QColor(135, 157, 245, 42)
            c2 = QColor(56, 189, 248, 22)
            c3 = QColor(168, 85, 247, 24)
        elif pal.mode == "pink":
            base.setColorAt(0.0, QColor("#F8EDF3"))
            base.setColorAt(0.5, QColor("#FDF4F8"))
            base.setColorAt(1.0, QColor("#F2E4EC"))
            c1 = QColor(252, 108, 133, 44)
            c2 = QColor(251, 146, 175, 32)
            c3 = QColor(192, 132, 252, 26)
        else:
            base.setColorAt(0.0, QColor("#EBF0FA"))
            base.setColorAt(0.5, QColor("#F5F8FD"))
            base.setColorAt(1.0, QColor("#E9EFF9"))
            c1 = QColor(94, 118, 214, 42)
            c2 = QColor(56, 189, 248, 28)
            c3 = QColor(139, 92, 246, 26)

        p.fillRect(0, 0, int(w), int(h), base)

        span = max(w, h)
        g1 = QRadialGradient(w * 0.14, h * 0.12, span * 0.50)
        g1.setColorAt(0.0, c1)
        g1.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, int(w), int(h), g1)

        g2 = QRadialGradient(w * 0.86, h * 0.18, span * 0.46)
        g2.setColorAt(0.0, c3)
        g2.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, int(w), int(h), g2)

        g3 = QRadialGradient(w * 0.48, h * 0.88, span * 0.55)
        g3.setColorAt(0.0, c2)
        g3.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(0, 0, int(w), int(h), g3)


class MainWindow(QMainWindow):
    """Complete Windows desktop studio for AI short-drama audio leveling and subtitle alignment."""

    def __init__(self, store: ConfigStore | None = None) -> None:
        super().__init__()
        self.store = store or ConfigStore()
        self.setWindowTitle(__app_name__)
        icon_p = resolve_asset_path("app_icon.ico") or resolve_asset_path("app_icon.png")
        if icon_p is not None:
            self.setWindowIcon(QIcon(str(icon_p)))
        self.resize(1580, 960)
        self.setMinimumSize(1260, 760)
        self.setAcceptDrops(True)

        self.undo_stack = QUndoStack(self)
        self.tasks: list[EpisodeTask] = []
        self.results: dict[str, PipelineResult] = {}
        self.current_task_id: str = ""

        self._probe_worker: MediaProbeWorker | None = None
        self._proc_worker: ProcessingWorker | None = None
        self._relevel_worker: AudioRelevelWorker | None = None
        self._audio_16k_cache: dict[str, object] = {}
        self._last_scrub_cue_idx: int = -1
        self._manual_aspect_tasks: set[str] = set()
        self._auto_switched_aspect: bool = False
        self._suppress_project_template_save: bool = False

        self._build_central()
        self._build_statusbar()
        self._load_active_project_to_ui()
        self._connect_signals()
        self.apply_theme(self.store.preferences.get("theme_mode", "light"), save=False)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def _build_header_bar(self) -> QFrame:
        header = FrostedHeaderBar()
        h_lay = QVBoxLayout(header)
        h_lay.setContentsMargins(14, 10, 14, 10)
        h_lay.setSpacing(8)

        # --- Row 1: Brand + Project + AI Engine + Undo/Redo + Models ---
        row1 = QHBoxLayout()
        row1.setSpacing(8)

        avatar_path = resolve_asset_path("app_icon.png")
        if avatar_path is not None:
            avatar_lbl = QLabel()
            avatar_lbl.setObjectName("BrandAvatar")
            avatar_lbl.setFixedSize(30, 30)
            pm = QPixmap(str(avatar_path))
            if not pm.isNull():
                pm = pm.scaled(
                    60,
                    60,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                pm.setDevicePixelRatio(2.0)
                avatar_lbl.setPixmap(pm)
            row1.addWidget(avatar_lbl)

        brand = QLabel(__brand_name__)
        brand.setObjectName("BrandTitle")
        badge = QLabel("AI 短剧工作站")
        badge.setObjectName("BrandBadge")
        row1.addWidget(brand)
        row1.addWidget(badge)
        row1.addSpacing(10)

        row1.addWidget(QLabel("项目:"))
        self.combo_project = ThemedComboBox()
        self.combo_project.setMinimumWidth(145)
        self.btn_new_proj = QPushButton("新建")
        self.btn_del_proj = QPushButton("删除")
        row1.addWidget(self.combo_project)
        row1.addWidget(self.btn_new_proj)
        row1.addWidget(self.btn_del_proj)
        row1.addSpacing(8)

        row1.addWidget(QLabel("对齐引擎:"))
        self.combo_aligner = ThemedComboBox()
        self.combo_aligner.addItem("自动优选 (推荐)", "auto")
        self.combo_aligner.addItem("FunASR fa-zh (轻量高速)", "funasr")
        self.combo_aligner.addItem("Qwen3-ForcedAligner-0.6B (高精度)", "qwen3")
        self.combo_aligner.addItem("仅 VAD 辅助 (无对齐模型)", "vad_fallback")
        self.combo_aligner.setMinimumWidth(195)
        row1.addWidget(self.combo_aligner)

        row1.addWidget(QLabel("设备:"))
        self.combo_device = ThemedComboBox()
        self.combo_device.addItem("自动 (GPU优先)", "auto")
        self.combo_device.addItem("CUDA GPU", "cuda")
        self.combo_device.addItem("CPU 模式", "cpu")
        row1.addWidget(self.combo_device)

        row1.addStretch()

        self.btn_undo = QPushButton("撤销")
        self.btn_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.btn_undo.setEnabled(False)
        self.btn_redo = QPushButton("重做")
        self.btn_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.btn_redo.setEnabled(False)
        self.undo_stack.canUndoChanged.connect(self.btn_undo.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.btn_redo.setEnabled)
        self.btn_undo.clicked.connect(self.undo_stack.undo)
        self.btn_redo.clicked.connect(self.undo_stack.redo)

        self.btn_models = QPushButton("模型下载与缓存")
        self.combo_theme = ThemedComboBox()
        for t_mode in THEME_ORDER:
            self.combo_theme.addItem(THEMES[t_mode].display_name, t_mode)
        self.combo_theme.setMinimumWidth(134)
        self.combo_theme.setToolTip("切换 纯净白 (浅色) / 曜石黑 (深色) / 浅樱粉 (粉色) 主题")

        row1.addWidget(self.btn_undo)
        row1.addWidget(self.btn_redo)
        row1.addWidget(self.btn_models)
        row1.addSpacing(4)
        row1.addWidget(QLabel("主题:"))
        row1.addWidget(self.combo_theme)
        h_lay.addLayout(row1)

        # --- Row 2: Output Options + Primary Action CTAs ---
        row2 = QHBoxLayout()
        row2.setSpacing(8)

        row2.addWidget(QLabel("导出目录:"))
        self.edit_out_dir = QLineEdit()
        self.edit_out_dir.setPlaceholderText("默认输出到原视频同级「智能修正输出」文件夹（绝不覆盖源文件）")
        self.btn_browse_out = QPushButton("选择目录")

        self.combo_bit_depth = ThemedComboBox()
        self.combo_bit_depth.addItem("WAV 24-bit / 48kHz", 24)
        self.combo_bit_depth.addItem("WAV 16-bit / 48kHz", 16)
        self.chk_mux_mp4 = QCheckBox("同步导出替换音轨MP4")

        self.btn_run_one = QPushButton("✦ 一键智能修正")
        self.btn_run_one.setObjectName("PrimaryButton")
        self.btn_run_batch = QPushButton("批量处理全部集")
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setObjectName("DangerButton")
        self.btn_cancel.setEnabled(False)
        self.btn_export = QPushButton("重新导出当前集")
        self.btn_export.setObjectName("SuccessButton")
        self.btn_open_out = QPushButton("打开输出目录")

        row2.addWidget(self.edit_out_dir, 1)
        row2.addWidget(self.btn_browse_out)
        row2.addWidget(self.combo_bit_depth)
        row2.addWidget(self.chk_mux_mp4)
        row2.addSpacing(10)
        row2.addWidget(self.btn_run_one)
        row2.addWidget(self.btn_run_batch)
        row2.addWidget(self.btn_cancel)
        row2.addWidget(self.btn_export)
        row2.addWidget(self.btn_open_out)
        h_lay.addLayout(row2)

        return header

    def _build_central(self) -> None:
        self.central_canvas = AuroraCanvasWidget()
        root_lay = QVBoxLayout(self.central_canvas)
        root_lay.setContentsMargins(12, 10, 12, 8)
        root_lay.setSpacing(10)

        root_lay.addWidget(self._build_header_bar())

        # Main 3-column horizontal splitter
        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.setChildrenCollapsible(False)

        self.import_panel = ImportBatchPanel()
        self.import_panel.setMinimumWidth(315)

        center_container = QWidget()
        center_container.setObjectName("TransparentContainer")
        center_lay = QVBoxLayout(center_container)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(8)

        self.audio_panel = AudioPanel()
        self.center_split = QSplitter(Qt.Orientation.Vertical)
        self.center_split.setChildrenCollapsible(False)
        self.waveform_widget = WaveformTimelineWidget()
        self.subtitle_editor = SubtitleEditorPanel()
        self.anomaly_log_panel = AnomalyAndLogPanel()

        self.center_split.addWidget(self.waveform_widget)
        self.center_split.addWidget(self.subtitle_editor)
        self.center_split.addWidget(self.anomaly_log_panel)
        self.center_split.setStretchFactor(0, 2)
        self.center_split.setStretchFactor(1, 4)
        self.center_split.setStretchFactor(2, 1)
        self.center_split.setSizes([195, 295, 115])

        center_lay.addWidget(self.audio_panel, 0)
        center_lay.addWidget(self.center_split, 1)

        self.template_panel = TemplateSettingsPanel(self.store)
        self.template_panel.setMinimumWidth(375)

        main_split.addWidget(self.import_panel)
        main_split.addWidget(center_container)
        main_split.addWidget(self.template_panel)
        main_split.setSizes([330, 840, 390])

        root_lay.addWidget(main_split, 1)
        self.setCentralWidget(self.central_canvas)

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.lbl_status = QLabel("就绪：拖入粗剪视频与剪映 SRT 字幕即可开始智能修正")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(280)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        sb.addWidget(self.lbl_status, 1)
        sb.addPermanentWidget(self.progress_bar)

    def _connect_signals(self) -> None:
        self.combo_project.currentIndexChanged.connect(self._on_project_switched)
        self.btn_new_proj.clicked.connect(self._create_project)
        self.btn_del_proj.clicked.connect(self._delete_project)
        self.combo_aligner.currentIndexChanged.connect(self._save_project_settings)
        self.combo_device.currentIndexChanged.connect(self._save_project_settings)
        self.btn_models.clicked.connect(self._open_model_manager)
        self.combo_theme.currentIndexChanged.connect(self._on_theme_combo_changed)

        self.btn_run_one.clicked.connect(lambda: self._start_processing(batch=False))
        self.btn_run_batch.clicked.connect(lambda: self._start_processing(batch=True))
        self.btn_cancel.clicked.connect(self._cancel_processing)
        self.btn_export.clicked.connect(self._reexport_current)
        self.btn_open_out.clicked.connect(self._open_output_folder)

        self.btn_browse_out.clicked.connect(self._browse_output_dir)
        self.edit_out_dir.textChanged.connect(self._save_project_settings)
        self.chk_mux_mp4.toggled.connect(self._save_project_settings)
        self.combo_bit_depth.currentIndexChanged.connect(self._save_project_settings)

        self.import_panel.filesImported.connect(self.import_paths)
        self.import_panel.episodeSelected.connect(self._on_episode_selected)
        self.import_panel.audioTrackChanged.connect(self._on_audio_track_changed)
        self.import_panel.srtBound.connect(self._on_srt_bound)
        self.import_panel.scriptBound.connect(self._on_script_bound)
        self.import_panel.removeEpisodeRequested.connect(self._on_remove_episode)

        self.template_panel.templateChanged.connect(self._on_template_changed)
        self.template_panel.batchModeChanged.connect(self._on_batch_mode_changed)
        self.template_panel.btn_preset_916.clicked.connect(self._on_user_preset_clicked)
        self.template_panel.btn_preset_169.clicked.connect(self._on_user_preset_clicked)

        self.audio_panel.configChanged.connect(self._on_audio_cfg_changed)
        self.audio_panel.relevelRequested.connect(self._relevel_current_audio)
        self.audio_panel.playheadUpdated.connect(self._on_playhead_moved)
        self.audio_panel.playheadScrubbed.connect(self._on_playhead_moved)

        self.waveform_widget.canvas.playheadSeeked.connect(self._on_canvas_playhead_seeked)
        self.waveform_widget.canvas.playheadScrubbed.connect(self._on_canvas_playhead_scrubbed)
        self.waveform_widget.canvas.cueSelected.connect(self._on_canvas_cue_selected)
        self.waveform_widget.canvas.cuesEdited.connect(self._on_cues_edited)
        self.waveform_widget.canvas.manualPointsEdited.connect(self._on_manual_points_edited)

        self.subtitle_editor.cueSelected.connect(self._on_table_cue_selected)
        self.subtitle_editor.cuesEdited.connect(self._on_cues_edited)
        self.subtitle_editor.relayoutRequested.connect(self._relayout_current_subtitles)

        self.anomaly_log_panel.anomalyClicked.connect(self._on_anomaly_clicked)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.import_paths(paths)
            event.acceptProposedAction()

    def _load_active_project_to_ui(self) -> None:
        self.combo_project.blockSignals(True)
        self.combo_project.clear()
        for pid, proj in self.store.projects.items():
            self.combo_project.addItem(proj.project_name, pid)
        idx = self.combo_project.findData(self.store.active_project_id)
        if idx >= 0:
            self.combo_project.setCurrentIndex(idx)
        self.combo_project.blockSignals(False)

        proj = self.store.get_active_project()
        self.template_panel.set_template(proj.active_template, emit_signal=False)
        self.template_panel.chk_unified_batch.setChecked(proj.batch_use_unified_template)
        self.audio_panel.set_config(proj.audio_config, emit_signal=False)

        a_idx = self.combo_aligner.findData(proj.alignment_config.backend)
        if a_idx >= 0:
            self.combo_aligner.setCurrentIndex(a_idx)
        d_idx = self.combo_device.findData(proj.alignment_config.device)
        if d_idx >= 0:
            self.combo_device.setCurrentIndex(d_idx)

        self.edit_out_dir.blockSignals(True)
        self.edit_out_dir.setText(proj.export_config.output_dir)
        self.edit_out_dir.blockSignals(False)
        self.chk_mux_mp4.blockSignals(True)
        self.chk_mux_mp4.setChecked(proj.export_config.export_muxed_mp4)
        self.chk_mux_mp4.blockSignals(False)
        b_idx = self.combo_bit_depth.findData(proj.export_config.wav_bit_depth)
        if b_idx >= 0:
            self.combo_bit_depth.setCurrentIndex(b_idx)

    def _save_project_settings(self) -> None:
        proj = self.store.get_active_project()
        if not self._suppress_project_template_save:
            proj.active_template = self.template_panel.current_template()
        proj.audio_config = self.audio_panel.current_config()
        proj.alignment_config.backend = self.combo_aligner.currentData() or "auto"
        proj.alignment_config.device = self.combo_device.currentData() or "auto"
        proj.export_config.output_dir = self.edit_out_dir.text().strip()
        proj.export_config.export_muxed_mp4 = self.chk_mux_mp4.isChecked()
        proj.export_config.wav_bit_depth = self.combo_bit_depth.currentData() or 24
        proj.batch_use_unified_template = self.template_panel.chk_unified_batch.isChecked()
        self.store.save_projects()

    def _on_project_switched(self, idx: int) -> None:
        if idx < 0:
            return
        pid = self.combo_project.itemData(idx)
        if pid:
            self._save_project_settings()
            self.store.switch_project(pid)
            self._load_active_project_to_ui()

    def _create_project(self) -> None:
        name, ok = QInputDialog.getText(self, "新建短剧项目", "请输入项目名称（每个项目将独立记住模板与处理设置）:")
        if ok and name.strip():
            self._save_project_settings()
            self.store.create_project(name.strip())
            self._load_active_project_to_ui()

    def _delete_project(self) -> None:
        pid = self.store.active_project_id
        if pid == "default":
            QMessageBox.information(self, "提示", "默认项目不可删除。")
            return
        ans = QMessageBox.question(self, "删除项目", "确认删除当前项目配置？")
        if ans == QMessageBox.StandardButton.Yes:
            self.store.delete_project(pid)
            self._load_active_project_to_ui()

    def import_paths(self, paths: list[str]) -> None:
        new_tasks, unmatched_srts = pair_media_files(paths)
        matched_scripts = assign_script_files_to_tasks(self.tasks, paths) if (not new_tasks and self.tasks) else 0

        if not new_tasks and (unmatched_srts or matched_scripts) and self.tasks:
            for srt_p in unmatched_srts:
                target = next((t for t in self.tasks if t.task_id == self.current_task_id), None)
                if target and not target.srt_path:
                    target.srt_path = srt_p
                    self.anomaly_log_panel.append_log(f"已将字幕 {Path(srt_p).name} 绑定到 {target.episode_label}")
                else:
                    unmatched = next((t for t in self.tasks if not t.srt_path), None)
                    if unmatched:
                        unmatched.srt_path = srt_p
            if matched_scripts:
                self.anomaly_log_panel.append_log(f"已自动匹配 {matched_scripts} 份台词文稿到剧集任务队列。")
            self.import_panel.set_tasks(self.tasks, self.current_task_id)
            return

        if not new_tasks:
            QMessageBox.warning(
                self,
                "未识别到支持文件",
                "请拖入 MP4 / MOV / MKV 视频、SRT 字幕，或先导入视频后再拖入台词文稿 (.md/.txt)。",
            )
            return

        existing_videos = {Path(t.video_path).resolve() for t in self.tasks}
        added: list[EpisodeTask] = []
        for nt in new_tasks:
            rv = Path(nt.video_path).resolve()
            if rv not in existing_videos:
                self.tasks.append(nt)
                added.append(nt)
                existing_videos.add(rv)
            else:
                for ex in self.tasks:
                    if Path(ex.video_path).resolve() == rv:
                        if nt.srt_path and not ex.srt_path:
                            ex.srt_path = nt.srt_path
                        if nt.script_path and not ex.script_path:
                            ex.script_path = nt.script_path

        if added and not self.current_task_id:
            self.current_task_id = added[0].task_id

        self.import_panel.set_tasks(self.tasks, self.current_task_id)
        if added:
            script_cnt = sum(1 for t in added if t.script_path)
            extra_note = f"（已自动配对 {script_cnt} 份台词稿）" if script_cnt else ""
            self.anomaly_log_panel.append_log(
                f"导入 {len(added)} 个剧集任务{extra_note}，正在后台检测视频编码与音轨..."
            )
            self._probe_worker = MediaProbeWorker(added)
            self._probe_worker.probed.connect(self._on_media_probed)
            self._probe_worker.probeFailed.connect(self._on_probe_failed)
            self._probe_worker.start()

    def _on_user_preset_clicked(self) -> None:
        if self.current_task_id:
            self._manual_aspect_tasks.add(self.current_task_id)
        self._auto_switched_aspect = False

    def _sync_template_to_video_orientation(self, task: EpisodeTask) -> None:
        info = task.media_info
        if info is None:
            return
        if task.task_id in self._manual_aspect_tasks or task.template_override is not None:
            return
        cur_tpl = self.template_panel.current_template()
        if not cur_tpl.is_builtin:
            return
        target_aspect = "16:9" if info.orientation == Orientation.LANDSCAPE else "9:16"
        if cur_tpl.aspect_preset != target_aspect:
            self._suppress_project_template_save = True
            try:
                self.template_panel.apply_aspect_preset(target_aspect)
            finally:
                self._suppress_project_template_save = False
            self._auto_switched_aspect = True
            label = "横屏 16:9 (20字)" if target_aspect == "16:9" else "竖屏 9:16 (9字)"
            self.lbl_status.setText(
                f"已根据导入视频 ({info.width}×{info.height} {info.aspect_ratio}) 自动切换为「{label}」字幕模板"
            )
            self.anomaly_log_panel.append_log(
                f"画面比例自适应：{task.episode_label} ({info.width}×{info.height}) → {label}"
            )

    def _on_media_probed(self, task_id: str, info: MediaInfo) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task is None:
            return
        task.media_info = info
        if task_id == self.current_task_id:
            res = self.results.get(task_id)
            risks = res.audio_report.risk_messages if res else None
            self.import_panel.show_media_info(info, risks)
            self.template_panel.set_video_fps(info.fps)
            self._sync_template_to_video_orientation(task)

    def _on_probe_failed(self, task_id: str, err: str) -> None:
        self.anomaly_log_panel.append_log(f"[检测失败] {err}")

    def _on_episode_selected(self, task_id: str) -> None:
        self.current_task_id = task_id
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task is None:
            return

        if not self.template_panel.chk_unified_batch.isChecked() and task.template_override:
            self.template_panel.set_template(task.template_override, emit_signal=False)
        else:
            self._sync_template_to_video_orientation(task)

        res = self.results.get(task_id)
        if res is not None:
            self._display_pipeline_result(res)
            self.template_panel.set_video_fps(res.media_info.fps if res.media_info else 25.0)
        else:
            self.import_panel.show_media_info(task.media_info)
            self.template_panel.set_video_fps(task.media_info.fps if task.media_info else 25.0)
            self.audio_panel.set_audio_sources("", "", None)
            self.waveform_widget.canvas.set_data(
                duration_ms=0,
                wf_times_ms=[],
                wf_before=[],
                wf_after=[],
                gain_times_ms=[],
                gain_db=[],
                vad_spans_ms=[],
                cues=[],
                anomalies=[],
                manual_points=[],
            )
            self.subtitle_editor.set_data([], self.template_panel.current_template())
            self.anomaly_log_panel.set_anomalies([])

    def _on_audio_track_changed(self, task_id: str, audio_idx: int) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task:
            task.selected_audio_index = audio_idx
            if task.media_info:
                task.media_info.selected_audio_index = audio_idx

    def _on_srt_bound(self, task_id: str, srt_path: str) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task:
            task.srt_path = srt_path
            self.import_panel.set_tasks(self.tasks, task_id)
            self.anomaly_log_panel.append_log(f"已手动绑定 SRT：{Path(srt_path).name} -> {task.episode_label}")

    def _on_script_bound(self, task_id: str, script_path: str) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task:
            task.script_path = script_path
            self.import_panel.set_tasks(self.tasks, task_id)
            self.anomaly_log_panel.append_log(f"已手动绑定台词文稿：{Path(script_path).name} -> {task.episode_label}")

    def _on_remove_episode(self, task_id: str) -> None:
        self.tasks = [t for t in self.tasks if t.task_id != task_id]
        self.results.pop(task_id, None)
        self._manual_aspect_tasks.discard(task_id)
        if self.current_task_id == task_id:
            self.current_task_id = self.tasks[0].task_id if self.tasks else ""
        self.import_panel.set_tasks(self.tasks, self.current_task_id)
        if self.current_task_id:
            self._on_episode_selected(self.current_task_id)
        else:
            self.import_panel.show_media_info(None)
            self.template_panel.set_video_fps(25.0)
            self.audio_panel.set_audio_sources("", "", None)
            if self._auto_switched_aspect:
                self._auto_switched_aspect = False
                self._suppress_project_template_save = True
                try:
                    self.template_panel.apply_aspect_preset("9:16")
                finally:
                    self._suppress_project_template_save = False

    def _on_template_changed(self, tpl: SubtitleTemplate) -> None:
        if not self.template_panel.chk_unified_batch.isChecked() and self.current_task_id:
            task = next((t for t in self.tasks if t.task_id == self.current_task_id), None)
            if task:
                task.template_override = deepcopy(tpl)
        self._save_project_settings()

    def _on_batch_mode_changed(self, unified: bool) -> None:
        self._save_project_settings()

    def _on_audio_cfg_changed(self, cfg: AudioLevelingConfig) -> None:
        self._save_project_settings()

    def _start_processing(self, batch: bool) -> None:
        if self._proc_worker is not None and self._proc_worker.isRunning():
            QMessageBox.information(self, "处理中", "后台任务正在运行，请等待完成或点击取消。")
            return
        if not self.tasks:
            QMessageBox.information(self, "请先导入文件", "请先拖入或导入视频（MP4/MOV/MKV）与 SRT 字幕文件。")
            return

        self._save_project_settings()
        proj = self.store.get_active_project()

        if batch:
            target_tasks = list(self.tasks)
        else:
            t = next((x for x in self.tasks if x.task_id == self.current_task_id), self.tasks[0])
            target_tasks = [t]

        unified = self.template_panel.chk_unified_batch.isChecked()
        global_tpl = self.template_panel.current_template()
        global_audio = self.audio_panel.current_config()

        jobs: list[tuple[EpisodeTask, SubtitleTemplate, AudioLevelingConfig]] = []
        for t in target_tasks:
            if unified:
                tpl = global_tpl
            elif t.template_override:
                tpl = t.template_override
            elif global_tpl.is_builtin and t.media_info is not None:
                tpl = (
                    SubtitleTemplate.preset_landscape_16_9()
                    if t.media_info.orientation == Orientation.LANDSCAPE
                    else SubtitleTemplate.preset_portrait_9_16()
                )
            else:
                tpl = global_tpl
            acfg = t.audio_config_override or global_audio
            t.status = "pending"
            t.progress = 0.0
            jobs.append((t, deepcopy(tpl), deepcopy(acfg)))

        self.import_panel.set_tasks(self.tasks, self.current_task_id)
        self.btn_run_one.setEnabled(False)
        self.btn_run_batch.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)

        cache_dir = Path(self.store.preferences["model_cache_dir"])
        work_root = default_app_data_dir() / "workspace_cache"

        self._proc_worker = ProcessingWorker(
            jobs=jobs,
            alignment_cfg=proj.alignment_config,
            export_cfg=proj.export_config,
            cache_dir=cache_dir,
            work_root=work_root,
        )
        self._proc_worker.taskProgress.connect(self._on_task_progress)
        self._proc_worker.taskCompleted.connect(self._on_task_completed)
        self._proc_worker.taskFailed.connect(self._on_task_failed)
        self._proc_worker.allFinished.connect(self._on_all_finished)
        self._proc_worker.logMessage.connect(self.anomaly_log_panel.append_log)
        self._proc_worker.start()

    def _cancel_processing(self) -> None:
        if self._proc_worker and self._proc_worker.isRunning():
            self._proc_worker.request_cancel()
            self.lbl_status.setText("正在取消后台任务...")

    def _on_task_progress(self, task_id: str, progress: float, msg: str) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task:
            task.status = "running"
            task.progress = progress
            task.status_message = msg
        self.lbl_status.setText(f"[{task.episode_label if task else ''}] {msg}")
        self.progress_bar.setValue(int(progress * 100))
        self.import_panel.set_tasks(self.tasks, self.current_task_id)

    def _on_task_completed(self, task_id: str, result: PipelineResult) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task:
            task.status = "done"
            task.progress = 1.0
            task.media_info = result.media_info
        self.results[task_id] = result
        self.import_panel.set_tasks(self.tasks, self.current_task_id)
        if task_id == self.current_task_id:
            self._display_pipeline_result(result)

    def _on_task_failed(self, task_id: str, err: str) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task:
            task.status = "error"
            task.error_message = err
        self.import_panel.set_tasks(self.tasks, self.current_task_id)
        QMessageBox.warning(self, "处理异常", f"{task.episode_label if task else ''} 处理失败：\n{err}")

    def _on_all_finished(self, cancelled: bool) -> None:
        self.btn_run_one.setEnabled(True)
        self.btn_run_batch.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if cancelled:
            for t in self.tasks:
                if t.status in {"pending", "running"}:
                    t.status = "cancelled"
            self.import_panel.set_tasks(self.tasks, self.current_task_id)
            self.lbl_status.setText("任务已取消")
        else:
            self.progress_bar.setValue(100)
            self.lbl_status.setText("全部处理与导出完成，结果已就绪（可直接导回达芬奇 Studio 20）")

    def _display_pipeline_result(self, res: PipelineResult) -> None:
        self.import_panel.show_media_info(res.media_info, res.audio_report.risk_messages)
        self.audio_panel.set_audio_sources(res.extracted_wav_path, res.processed_wav_path, res.audio_report)

        audio_cfg = self.audio_panel.current_config()
        self.waveform_widget.canvas.set_data(
            duration_ms=res.audio_report.duration_ms,
            wf_times_ms=res.waveform_times_ms,
            wf_before=res.waveform_before_peaks,
            wf_after=res.waveform_after_peaks,
            gain_times_ms=res.gain_curve_times_ms,
            gain_db=res.gain_curve_db,
            vad_spans_ms=res.vad_spans_ms,
            cues=res.processed_cues,
            anomalies=res.anomalies,
            manual_points=audio_cfg.manual_points,
        )
        self.waveform_widget.sync_scrollbar()

        tpl = self.template_panel.current_template()
        self.subtitle_editor.set_data(res.processed_cues, tpl)
        self.anomaly_log_panel.set_anomalies(res.anomalies)
        if res.processed_cues:
            sample = " ".join(c.raw_text for c in res.processed_cues[:3]).strip()
            self.template_panel.set_preview_sample_text(sample)

    def _on_cues_edited(self, description: str, before: list[SubtitleCue], after: list[SubtitleCue]) -> None:
        if self.current_task_id not in self.results:
            return
        tid = self.current_task_id

        def _apply(cues_snapshot: list[SubtitleCue]) -> None:
            res = self.results.get(tid)
            if res is None:
                return
            tpl = self.template_panel.current_template()
            cleaned = normalize_cues_for_resolve(deepcopy(cues_snapshot), min_gap_ms=1)
            res.processed_cues = cleaned
            audio_16k = self._get_cached_16k(res.extracted_wav_path)
            sub_anom = run_quality_checks(
                cleaned,
                tpl,
                res.vad_spans_ms,
                audio_16k,
                res.audio_report.duration_ms,
            )
            audio_anom = [a for a in res.anomalies if a.cue_index is None]
            res.anomalies = audio_anom + sub_anom
            if tid == self.current_task_id:
                self.subtitle_editor.set_data(cleaned, tpl)
                self.waveform_widget.canvas.set_cues(cleaned)
                self.waveform_widget.canvas.anomalies = res.anomalies
                self.waveform_widget.canvas.update()
                self.anomaly_log_panel.set_anomalies(res.anomalies)

        cmd = ReplaceCuesCommand(description, before, after, _apply)
        self.undo_stack.push(cmd)

    def _on_manual_points_edited(
        self,
        description: str,
        before: list[ManualGainPoint],
        after: list[ManualGainPoint],
    ) -> None:
        def _apply(pts: list[ManualGainPoint]) -> None:
            cfg = self.audio_panel.current_config()
            cfg.manual_points = deepcopy(pts)
            self.audio_panel.set_config(cfg, emit_signal=True)
            self.waveform_widget.canvas.set_manual_points(pts)

        cmd = ReplaceManualGainCommand(description, before, after, _apply)
        self.undo_stack.push(cmd)

    def _get_cached_16k(self, wav_path: str):
        import numpy as np
        import soundfile as sf

        if wav_path in self._audio_16k_cache:
            return self._audio_16k_cache[wav_path]
        if not wav_path or not Path(wav_path).is_file():
            return np.zeros(0, dtype=np.float32)
        sig, sr = sf.read(wav_path, dtype="float32")
        a16 = to_16k_mono(sig, sr)
        self._audio_16k_cache[wav_path] = a16
        return a16

    def _relayout_current_subtitles(self) -> None:
        res = self.results.get(self.current_task_id)
        if res is None or not res.processed_cues:
            return
        tpl = self.template_panel.current_template()
        before = deepcopy(res.processed_cues)
        audio_16k = self._get_cached_16k(res.extracted_wav_path)
        new_cues, _, _ = relayout_cues_with_template(
            res.processed_cues,
            tpl,
            duration_ms=res.audio_report.duration_ms,
            vad_spans=res.vad_spans_ms,
            audio_16k=audio_16k,
            video_fps=res.media_info.fps if res.media_info else 25.0,
        )
        self._on_cues_edited("按当前模板重新语义断句", before, new_cues)
        self.lbl_status.setText(f"已按当前模板 ({tpl.max_chars_per_line}字/{tpl.max_lines}行) 重新断句，共 {len(new_cues)} 条。")

    def _relevel_current_audio(self) -> None:
        res = self.results.get(self.current_task_id)
        if res is None or not res.extracted_wav_path:
            QMessageBox.information(self, "提示", "请先运行「一键智能修正」提取音轨后再重新应用音量均衡。")
            return
        if self._relevel_worker and self._relevel_worker.isRunning():
            return

        self.audio_panel.release_sources()
        self.lbl_status.setText("正在按新参数与人工增益节点重新计算 48kHz 音频...")
        cfg = self.audio_panel.current_config()
        bit_depth = self.combo_bit_depth.currentData() or 24

        self._relevel_worker = AudioRelevelWorker(
            task_id=self.current_task_id,
            extracted_wav_path=res.extracted_wav_path,
            output_wav_path=res.processed_wav_path,
            audio_cfg=cfg,
            cues=res.processed_cues,
            vad_spans_ms=res.vad_spans_ms,
            bit_depth=bit_depth,
        )
        self._relevel_worker.finishedOk.connect(self._on_relevel_done)
        self._relevel_worker.failed.connect(lambda _tid, e: QMessageBox.warning(self, "重算失败", e))
        self._relevel_worker.start()

    def _on_relevel_done(self, task_id: str, out) -> None:
        res = self.results.get(task_id)
        if res is None:
            return
        res.audio_report = out.report
        res.waveform_before_peaks = out.waveform_before_peaks
        res.waveform_after_peaks = out.waveform_after_peaks
        res.gain_curve_times_ms = out.gain_curve_times_ms
        res.gain_curve_db = out.gain_curve_db
        if task_id == self.current_task_id:
            self._display_pipeline_result(res)
        self.lbl_status.setText(f"音频均衡已更新：{out.report.before_lufs:.1f} → {out.report.after_lufs:.1f} LUFS")

    def _on_canvas_cue_selected(self, idx: int) -> None:
        self.subtitle_editor.select_cue_index(idx)
        res = self.results.get(self.current_task_id)
        if res and 0 <= idx < len(res.processed_cues):
            self.template_panel.set_preview_sample_text(res.processed_cues[idx].raw_text)

    def _on_table_cue_selected(self, idx: int) -> None:
        self.waveform_widget.canvas.set_selected_cue(idx)
        res = self.results.get(self.current_task_id)
        if res and 0 <= idx < len(res.processed_cues):
            cue = res.processed_cues[idx]
            self.waveform_widget.canvas.focus_time_ms(cue.start_ms)
            self.audio_panel.seek_to_ms(cue.start_ms)
            self.template_panel.set_preview_sample_text(cue.raw_text)

    def _on_anomaly_clicked(self, start_ms: int, cue_index: object) -> None:
        self.waveform_widget.canvas.focus_time_ms(start_ms)
        self.audio_panel.seek_to_ms(start_ms)
        if isinstance(cue_index, int) and cue_index >= 1:
            self.subtitle_editor.select_cue_index(cue_index - 1)
            self.waveform_widget.canvas.set_selected_cue(cue_index - 1)

    def _reexport_current(self) -> None:
        res = self.results.get(self.current_task_id)
        if res is None:
            QMessageBox.information(self, "提示", "当前集尚未处理完成。")
            return
        self._save_project_settings()
        cfg = deepcopy(self.store.get_active_project().export_config)
        if not cfg.output_dir:
            cfg.output_dir = str(Path(res.media_info.path).parent / "智能修正输出")
        try:
            exported = export_episode_outputs(
                res,
                cfg,
                progress_cb=lambda m: self.anomaly_log_panel.append_log(m),
            )
            files_list = "\n".join(f"• {k.upper()}: {v}" for k, v in exported.items())
            QMessageBox.information(self, "导出成功", f"已导出达芬奇可用文件：\n\n{files_list}")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def _browse_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择导出输出文件夹", self.edit_out_dir.text())
        if folder:
            self.edit_out_dir.setText(folder)

    def _open_output_folder(self) -> None:
        res = self.results.get(self.current_task_id)
        target = self.edit_out_dir.text().strip()
        if not target and res and res.exported_files:
            any_file = next(iter(res.exported_files.values()))
            target = str(Path(any_file).parent)
        elif not target and self.tasks:
            target = str(Path(self.tasks[0].video_path).parent / "智能修正输出")
        if target and Path(target).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(target))
        else:
            QMessageBox.information(self, "提示", "输出目录尚未生成，请先执行一次处理导出。")

    def _on_canvas_playhead_scrubbed(self, ms: int) -> None:
        self.audio_panel.scrub_to_ms(ms)
        self._sync_live_preview_at_ms(ms)

    def _on_canvas_playhead_seeked(self, ms: int) -> None:
        self.audio_panel.seek_to_ms(ms)
        self._sync_live_preview_at_ms(ms)

    def _on_playhead_moved(self, ms: int) -> None:
        self.waveform_widget.canvas.set_playhead(ms)
        self._sync_live_preview_at_ms(ms)

    def _sync_live_preview_at_ms(self, ms: int) -> None:
        cues = self.waveform_widget.canvas.cues
        if not cues:
            return
        hit_idx = -1
        for idx, cue in enumerate(cues):
            if cue.start_ms <= ms <= cue.end_ms:
                hit_idx = idx
                break
        if hit_idx >= 0 and hit_idx != self._last_scrub_cue_idx:
            self._last_scrub_cue_idx = hit_idx
            self.waveform_widget.canvas.set_selected_cue(hit_idx)
            self.subtitle_editor.select_cue_index(hit_idx)
            self.template_panel.set_preview_sample_text(cues[hit_idx].raw_text)

    def eventFilter(self, obj, event) -> bool:
        etype = event.type()
        if etype in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and isinstance(event, QKeyEvent):
            key = event.key()
            if key in (Qt.Key.Key_Space, Qt.Key.Key_Left, Qt.Key.Key_Right):
                fw = QApplication.focusWidget()
                if fw is not None and fw.window() is not self:
                    return False
                if isinstance(fw, (QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox)):
                    return False
                if self.subtitle_editor.table.state() == QAbstractItemView.State.EditingState:
                    return False
                if isinstance(fw, QComboBox) and fw.view() is not None and fw.view().isVisible():
                    return False

                if key == Qt.Key.Key_Space:
                    if etype == QEvent.Type.KeyPress and not event.isAutoRepeat():
                        self.audio_panel.toggle_play()
                    return True

                if key in (Qt.Key.Key_Left, Qt.Key.Key_Right) and not isinstance(fw, QComboBox):
                    if etype == QEvent.Type.KeyPress:
                        step = 2000 if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 500
                        delta = -step if key == Qt.Key.Key_Left else step
                        cur_ms = self.waveform_widget.canvas.playhead_ms
                        new_ms = max(0, cur_ms + delta)
                        self.audio_panel.scrub_to_ms(new_ms)
                        self.waveform_widget.canvas.focus_time_ms(new_ms)
                    return True
        return super().eventFilter(obj, event)

    def _on_theme_combo_changed(self, _idx: int) -> None:
        mode = self.combo_theme.currentData()
        if isinstance(mode, str):
            self.apply_theme(mode, save=True)

    def toggle_theme(self) -> None:
        cur = normalize_theme_mode(self.store.preferences.get("theme_mode", "light"))
        nxt = THEME_ORDER[(THEME_ORDER.index(cur) + 1) % len(THEME_ORDER)]
        self.apply_theme(nxt, save=True)

    def apply_theme(self, mode: str, *, save: bool = True) -> None:
        normalized = normalize_theme_mode(mode)
        pal = get_palette(normalized)
        app = QApplication.instance()
        if app is not None:
            app.setProperty("theme_mode", normalized)
            app.setStyleSheet(build_studio_qss(normalized))
        apply_win11_dwm_chrome(
            self,
            border_hex=pal.line,
            is_dark=pal.is_dark,
            round_corners=False,
            caption_hex=pal.header_bg,
            text_hex=pal.text,
        )
        if hasattr(self, "central_canvas"):
            self.central_canvas.set_theme(normalized)
        if hasattr(self, "import_panel"):
            self.import_panel.set_theme(normalized)
        if hasattr(self, "template_panel"):
            self.template_panel.set_theme(normalized)
        if hasattr(self, "waveform_widget"):
            self.waveform_widget.canvas.set_theme(normalized)
        if hasattr(self, "subtitle_editor"):
            self.subtitle_editor.set_theme(normalized)
        if hasattr(self, "anomaly_log_panel"):
            self.anomaly_log_panel.set_theme(normalized)
        if hasattr(self, "combo_theme"):
            idx = self.combo_theme.findData(normalized)
            if idx >= 0 and self.combo_theme.currentIndex() != idx:
                self.combo_theme.blockSignals(True)
                self.combo_theme.setCurrentIndex(idx)
                self.combo_theme.blockSignals(False)
        if save:
            self.store.preferences["theme_mode"] = normalized
            self.store.save_prefs()

    def _open_model_manager(self) -> None:
        dlg = ModelManagerDialog(self.store, self)
        dlg.exec()

    def closeEvent(self, event) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self.audio_panel.release_sources()
        super().closeEvent(event)
