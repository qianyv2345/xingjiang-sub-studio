"""Main application window with Light Neomorphism & Aurora Dark Glassmorphism themes."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

from PySide6.QtCore import QEvent, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QAction,
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
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from drama_studio import __app_name__, __brand_name__, __version__
from drama_studio.align.engine import relayout_cues_with_template, run_quality_checks
from drama_studio.align.model_manager import check_model_vram_risk
from drama_studio.audio.vad import to_16k_mono
from drama_studio.config import ConfigStore, default_work_cache_dir
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
from drama_studio.gui.workers import AudioRelevelWorker, MediaProbeWorker, ProcessingWorker, RelayoutWorker
from drama_studio.media.pairing import (
    assign_script_files_to_tasks,
    disambiguate_episode_labels,
    pair_media_files,
)
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


class EngineExportSettingsDialog(QDialog):
    """Secondary settings dialog hosting low-frequency AI backend, device, and export format controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("处理引擎与导出参数设置")
        self.setModal(True)
        self.resize(470, 320)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        ai_box = QGroupBox("AI 强制对齐与硬件推理")
        ai_form = QFormLayout(ai_box)
        ai_form.setSpacing(8)
        self.combo_aligner = ThemedComboBox()
        self.combo_aligner.addItem("自动优选 (中文可选 FunASR / 双语推荐 Qwen3)", "auto")
        self.combo_aligner.addItem("Qwen3-ForcedAligner (中英通用精修)", "qwen3")
        self.combo_aligner.addItem("FunASR fa-zh (仅限中文快速模式)", "funasr")
        self.combo_aligner.addItem("仅 VAD 辅助 (无强制对齐模型时兜底)", "vad_fallback")
        self.combo_device = ThemedComboBox()
        self.combo_device.addItem("自动 (GPU 优先，显存不足自动降级)", "auto")
        self.combo_device.addItem("CUDA GPU 加速", "cuda")
        self.combo_device.addItem("CPU 安全模式", "cpu")
        ai_form.addRow("强制对齐引擎:", self.combo_aligner)
        ai_form.addRow("推理计算设备:", self.combo_device)
        lay.addWidget(ai_box)

        exp_box = QGroupBox("导出音频规格与防覆盖策略")
        exp_form = QFormLayout(exp_box)
        exp_form.setSpacing(8)
        self.combo_bit_depth = ThemedComboBox()
        self.combo_bit_depth.addItem("24-bit / 48 kHz PCM (广播与达芬奇推荐)", 24)
        self.combo_bit_depth.addItem("16-bit / 48 kHz PCM (兼容体积较小)", 16)
        self.chk_mux_mp4 = QCheckBox("同步封装导出参考 MP4 视频（自动嵌入新字幕与新音轨）")
        hint = QLabel("安全导出：同一目录下自动使用 _v2 / _v3 后缀，绝不覆盖原始粗剪或已有成品。")
        hint.setObjectName("MutedLabel")
        hint.setWordWrap(True)
        exp_form.addRow("WAV 音频位深:", self.combo_bit_depth)
        exp_form.addRow("视频封装选项:", self.chk_mux_mp4)
        exp_form.addRow("", hint)
        lay.addWidget(exp_box)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        ok_btn = btns.button(QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setText("完成")
        btns.accepted.connect(self.accept)
        lay.addWidget(btns)


class MainWindow(QMainWindow):
    """Complete Windows desktop studio for AI short-drama audio leveling and subtitle alignment."""

    def __init__(self, store: ConfigStore | None = None) -> None:
        super().__init__()
        self.store = store or ConfigStore()
        self.setWindowTitle(f"{__app_name__} v{__version__}")
        icon_p = resolve_asset_path("app_icon.ico") or resolve_asset_path("app_icon.png")
        if icon_p is not None:
            self.setWindowIcon(QIcon(str(icon_p)))
        self.resize(1540, 920)
        self.setMinimumSize(1160, 720)
        self.setAcceptDrops(True)

        self.undo_stack = QUndoStack(self)
        self.tasks: list[EpisodeTask] = []
        self.results: dict[str, PipelineResult] = {}
        self.current_task_id: str = ""

        self._probe_worker: MediaProbeWorker | None = None
        self._probe_workers: list[MediaProbeWorker] = []
        self._proc_worker: ProcessingWorker | None = None
        self._relevel_worker: AudioRelevelWorker | None = None
        self._relayout_worker: RelayoutWorker | None = None
        self._audio_16k_cache: dict[str, object] = {}
        self._last_scrub_cue_idx: int = -1
        self._last_prog_ui_ts: float = 0.0
        self._last_prog_pct: int = -1
        self._manual_aspect_tasks: set[str] = set()
        self._auto_switched_aspect: bool = False
        self._suppress_project_template_save: bool = False
        self._loading_project_ui: bool = False
        self._handled_missing_model_tasks: set[str] = set()
        self._last_known_lang: str = "zh"

        self._save_prefs_timer = QTimer(self)
        self._save_prefs_timer.setSingleShot(True)
        self._save_prefs_timer.setInterval(250)
        self._save_prefs_timer.timeout.connect(self._save_project_settings)

        self._pending_preview_text: str = ""
        self._preview_sync_timer = QTimer(self)
        self._preview_sync_timer.setSingleShot(True)
        self._preview_sync_timer.setInterval(70)
        self._preview_sync_timer.timeout.connect(self._flush_preview_sample_text)

        self._engine_settings_dlg = EngineExportSettingsDialog(self)
        self.combo_aligner = self._engine_settings_dlg.combo_aligner
        self.combo_device = self._engine_settings_dlg.combo_device
        self.combo_bit_depth = self._engine_settings_dlg.combo_bit_depth
        self.chk_mux_mp4 = self._engine_settings_dlg.chk_mux_mp4

        self._build_central()
        self._build_statusbar()
        self._load_active_project_to_ui()
        self._connect_signals()
        self._refresh_engine_settings_summary()
        self.apply_theme(self.store.preferences.get("theme_mode", "light"), save=False)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def _build_header_bar(self) -> QFrame:
        header = FrostedHeaderBar()
        h_lay = QVBoxLayout(header)
        h_lay.setContentsMargins(14, 9, 14, 9)
        h_lay.setSpacing(7)

        # --- Row 1: Brand + Project + Language + ASR Model + Secondary Dialogs + Theme ---
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        avatar_path = resolve_asset_path("app_icon.png")
        if avatar_path is not None:
            avatar_lbl = QLabel()
            avatar_lbl.setObjectName("BrandAvatar")
            avatar_lbl.setFixedSize(28, 28)
            pm = QPixmap(str(avatar_path))
            if not pm.isNull():
                pm = pm.scaled(
                    56,
                    56,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                pm.setDevicePixelRatio(2.0)
                avatar_lbl.setPixmap(pm)
            row1.addWidget(avatar_lbl)

        brand = QLabel(__brand_name__)
        brand.setObjectName("BrandTitle")
        row1.addWidget(brand)
        row1.addSpacing(8)

        row1.addWidget(QLabel("项目:"))
        self.combo_project = ThemedComboBox()
        self.combo_project.setMinimumWidth(100)
        self.btn_new_proj = QPushButton("＋")
        self.btn_new_proj.setFixedWidth(32)
        self.btn_new_proj.setToolTip("新建项目")
        self.btn_del_proj = QPushButton("－")
        self.btn_del_proj.setFixedWidth(32)
        self.btn_del_proj.setToolTip("删除当前项目")
        row1.addWidget(self.combo_project)
        row1.addWidget(self.btn_new_proj)
        row1.addWidget(self.btn_del_proj)
        row1.addSpacing(4)

        row1.addWidget(QLabel("语言:"))
        self.combo_language = ThemedComboBox()
        self.combo_language.addItem("自动检测", "auto")
        self.combo_language.addItem("中文", "zh")
        self.combo_language.addItem("English", "en")
        self.combo_language.setMinimumWidth(86)
        row1.addWidget(self.combo_language)

        row1.addWidget(QLabel("识别:"))
        self.combo_asr_model = ThemedComboBox()
        self.combo_asr_model.addItem("自动 (Qwen3-0.6B)", "auto")
        self.combo_asr_model.addItem("Qwen3-ASR-0.6B", "qwen3_asr_0.6b")
        self.combo_asr_model.addItem("Qwen3-ASR-1.7B", "qwen3_asr_1.7b")
        self.combo_asr_model.addItem("FunASR (仅中文)", "funasr_asr")
        self.combo_asr_model.setMinimumWidth(142)
        row1.addWidget(self.combo_asr_model)

        row1.addStretch()

        self.btn_undo = QPushButton("↶")
        self.btn_undo.setFixedWidth(36)
        self.btn_undo.setToolTip("撤销上一步修改 (Ctrl+Z)")
        self.btn_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.btn_undo.setEnabled(False)
        self.btn_redo = QPushButton("↷")
        self.btn_redo.setFixedWidth(36)
        self.btn_redo.setToolTip("重做修改 (Ctrl+Y)")
        self.btn_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.btn_redo.setEnabled(False)
        self.undo_stack.canUndoChanged.connect(self.btn_undo.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.btn_redo.setEnabled)
        self.btn_undo.clicked.connect(self.undo_stack.undo)
        self.btn_redo.clicked.connect(self.undo_stack.redo)

        self.btn_engine_settings = QPushButton("⚙ 引擎与导出")
        self.btn_engine_settings.setToolTip("打开二级设置：配置强制对齐引擎、GPU/CPU 设备、WAV 位深与 MP4 封装")
        self.btn_models = QPushButton("模型与缓存")
        self.combo_theme = ThemedComboBox()
        for t_mode in THEME_ORDER:
            self.combo_theme.addItem(THEMES[t_mode].display_name, t_mode)
        self.combo_theme.setMinimumWidth(112)
        self.combo_theme.setToolTip("切换 纯净白 (浅色) / 曜石黑 (深色) / 浅樱粉 (粉色) 主题")

        row1.addWidget(self.btn_undo)
        row1.addWidget(self.btn_redo)
        row1.addWidget(self.btn_engine_settings)
        row1.addWidget(self.btn_models)
        row1.addWidget(self.combo_theme)
        h_lay.addLayout(row1)

        # --- Row 2: Output Dir + Export Scope + Panel Toggles + Single Primary Start CTA ---
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        row2.addWidget(QLabel("导出:"))
        self.edit_out_dir = QLineEdit()
        self.edit_out_dir.setMinimumWidth(140)
        self.edit_out_dir.setPlaceholderText("默认输出到原视频同级「智能修正输出」文件夹（绝不覆盖已有文件）")
        self.btn_browse_out = QPushButton("浏览")

        self.combo_export_scope = ThemedComboBox()
        self.combo_export_scope.addItem("字幕 + 音频", "both")
        self.combo_export_scope.addItem("仅字幕 (SRT)", "srt_only")
        self.combo_export_scope.addItem("仅音频 (WAV)", "wav_only")

        self.btn_toggle_import = QPushButton("◧ 素材栏")
        self.btn_toggle_import.setObjectName("TogglePill")
        self.btn_toggle_import.setCheckable(True)
        self.btn_toggle_import.setChecked(bool(self.store.preferences.get("show_import_panel", True)))
        self.btn_toggle_import.setToolTip("快速展开或收起左侧素材导入与队列栏，给波形和字幕留出更多空间")

        self.btn_toggle_template = QPushButton("◨ 排版栏")
        self.btn_toggle_template.setObjectName("TogglePill")
        self.btn_toggle_template.setCheckable(True)
        self.btn_toggle_template.setChecked(bool(self.store.preferences.get("show_template_panel", True)))
        self.btn_toggle_template.setToolTip("快速展开或收起右侧字幕模板与规则栏，切换专注精修视野")

        self.combo_run_scope = ThemedComboBox()
        self.combo_run_scope.addItem("当前集", "current")
        self.combo_run_scope.addItem("批量全部", "batch")

        self.btn_start = QPushButton("✦ 开始处理")
        self.btn_start.setObjectName("PrimaryButton")

        self.btn_stage_actions = QToolButton()
        self.btn_stage_actions.setText("高级单步 ▾")
        self.btn_stage_actions.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        stage_menu = QMenu(self)
        self.act_asr_only = QAction("1. 仅运行字幕识别 (Stage 2)", self)
        self.act_script_match = QAction("2. 仅重新文稿匹配 (复用 ASR)", self)
        self.act_align_only = QAction("3. 仅重新强制对齐与排版 (Stage 3)", self)
        self.act_relayout_only = QAction("4. 仅重新字幕排版 (复用对齐)", self)
        self.act_relevel_only = QAction("5. 仅重新音量均衡 (复用字幕)", self)
        self.act_reexport = QAction("6. 重新导出当前集结果", self)
        self.act_force_full = QAction("7. 忽略缓存强制完整重跑", self)
        for act in (
            self.act_asr_only,
            self.act_script_match,
            self.act_align_only,
            self.act_relayout_only,
            self.act_relevel_only,
            self.act_reexport,
            self.act_force_full,
        ):
            stage_menu.addAction(act)
        self.btn_stage_actions.setMenu(stage_menu)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setObjectName("DangerButton")
        self.btn_cancel.setEnabled(False)
        self.btn_open_out = QPushButton("打开目录")

        row2.addWidget(self.edit_out_dir, 1)
        row2.addWidget(self.btn_browse_out)
        row2.addWidget(self.combo_export_scope)
        row2.addWidget(self.btn_toggle_import)
        row2.addWidget(self.btn_toggle_template)
        row2.addWidget(self.combo_run_scope)
        row2.addWidget(self.btn_start)
        row2.addWidget(self.btn_stage_actions)
        row2.addWidget(self.btn_cancel)
        row2.addWidget(self.btn_open_out)
        h_lay.addLayout(row2)

        # --- Row 3: Compact 3-Stage Workflow Status Strip ---
        row3 = QHBoxLayout()
        row3.setSpacing(8)
        self.lbl_stage1 = QLabel("① 导入素材：等待导入粗剪视频")
        self.lbl_stage2 = QLabel("② 字幕识别：语言 - · 模型 - · 对白 0 条 · 异常 0 项")
        self.lbl_stage3 = QLabel("③ 精修导出：等待对齐、排版与音量均衡")
        for lbl in (self.lbl_stage1, self.lbl_stage2, self.lbl_stage3):
            lbl.setObjectName("PillBadge")
            lbl.setMinimumWidth(90)
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            row3.addWidget(lbl, 1)
        h_lay.addLayout(row3)

        return header

    def _build_central(self) -> None:
        self.central_canvas = AuroraCanvasWidget()
        root_lay = QVBoxLayout(self.central_canvas)
        root_lay.setContentsMargins(12, 10, 12, 8)
        root_lay.setSpacing(8)

        root_lay.addWidget(self._build_header_bar())

        # Main 3-column horizontal splitter (Left & Right can be toggled cleanly via header pills)
        self.main_split = QSplitter(Qt.Orientation.Horizontal)
        self.main_split.setChildrenCollapsible(False)

        self.import_panel = ImportBatchPanel()
        self.import_panel.setMinimumWidth(290)

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
        self.anomaly_log_panel = AnomalyAndLogPanel(embedded=True)

        self.center_bottom_tabs = QTabWidget()
        self.center_bottom_tabs.addTab(self.subtitle_editor, "字幕校对编辑 (0)")
        self.center_bottom_tabs.addTab(self.anomaly_log_panel.anom_page, "异常检查 (0)")
        self.center_bottom_tabs.addTab(self.anomaly_log_panel.log_page, "运行日志")

        self.center_split.addWidget(self.waveform_widget)
        self.center_split.addWidget(self.center_bottom_tabs)
        self.center_split.setStretchFactor(0, 3)
        self.center_split.setStretchFactor(1, 5)
        self.center_split.setSizes([235, 395])

        center_lay.addWidget(self.audio_panel, 0)
        center_lay.addWidget(self.center_split, 1)

        self.template_panel = TemplateSettingsPanel(self.store)
        self.template_panel.setMinimumWidth(345)

        self.main_split.addWidget(self.import_panel)
        self.main_split.addWidget(center_container)
        self.main_split.addWidget(self.template_panel)
        self.main_split.setSizes([315, 860, 365])

        self.import_panel.setVisible(self.btn_toggle_import.isChecked())
        self.template_panel.setVisible(self.btn_toggle_template.isChecked())

        root_lay.addWidget(self.main_split, 1)
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
        self.combo_language.currentIndexChanged.connect(self._on_language_changed)
        self.combo_asr_model.currentIndexChanged.connect(self._on_asr_model_changed)
        self.combo_aligner.currentIndexChanged.connect(self._save_project_settings)
        self.combo_device.currentIndexChanged.connect(self._save_project_settings)
        self.btn_engine_settings.clicked.connect(self._open_engine_settings)
        self.btn_models.clicked.connect(self._open_model_manager)
        self.combo_theme.currentIndexChanged.connect(self._on_theme_combo_changed)
        self.btn_toggle_import.toggled.connect(lambda checked: self._toggle_side_panel("import", checked))
        self.btn_toggle_template.toggled.connect(lambda checked: self._toggle_side_panel("template", checked))
        self.anomaly_log_panel.anomaliesCountChanged.connect(
            lambda count: self.center_bottom_tabs.setTabText(1, f"异常检查 ({count})")
        )

        self.btn_start.clicked.connect(lambda: self._start_processing(stage_action="full"))
        self.act_asr_only.triggered.connect(lambda: self._start_processing(stage_action="asr_only"))
        self.act_script_match.triggered.connect(lambda: self._start_processing(stage_action="script_match_only"))
        self.act_align_only.triggered.connect(lambda: self._start_processing(stage_action="align_only"))
        self.act_relayout_only.triggered.connect(lambda: self._start_processing(stage_action="relayout_only"))
        self.act_relevel_only.triggered.connect(lambda: self._start_processing(stage_action="relevel_only"))
        self.act_reexport.triggered.connect(self._reexport_current)
        self.act_force_full.triggered.connect(lambda: self._start_processing(stage_action="full", force_recompute=True))

        self.btn_cancel.clicked.connect(self._cancel_processing)
        self.btn_open_out.clicked.connect(self._open_output_folder)

        self.btn_browse_out.clicked.connect(self._browse_output_dir)
        self.edit_out_dir.textChanged.connect(self._save_project_settings)
        self.combo_export_scope.currentIndexChanged.connect(self._save_project_settings)
        self.chk_mux_mp4.toggled.connect(self._save_project_settings)
        self.combo_bit_depth.currentIndexChanged.connect(self._save_project_settings)

        self.import_panel.filesImported.connect(self.import_paths)
        self.import_panel.episodeSelected.connect(self._on_episode_selected)
        self.import_panel.audioTrackChanged.connect(self._on_audio_track_changed)
        self.import_panel.srtBound.connect(self._on_srt_bound)
        self.import_panel.scriptBound.connect(self._on_script_bound)
        self.import_panel.taskModeChanged.connect(self._on_task_mode_changed)
        self.import_panel.removeEpisodeRequested.connect(self._on_remove_episode)

        self.template_panel.templateChanged.connect(self._on_template_changed)
        self.template_panel.batchModeChanged.connect(self._on_batch_mode_changed)
        self.template_panel.btn_preset_916.clicked.connect(self._on_user_preset_clicked)
        self.template_panel.btn_preset_169.clicked.connect(self._on_user_preset_clicked)
        self.template_panel.btn_preset_en.clicked.connect(self._on_user_preset_clicked)

        self.audio_panel.configChanged.connect(self._on_audio_cfg_changed)
        self.audio_panel.relevelRequested.connect(self._relevel_current_audio)
        self.audio_panel.playheadUpdated.connect(self._on_playhead_moved)
        self.audio_panel.playheadScrubbed.connect(self._on_playhead_moved)

        self.waveform_widget.canvas.playheadSeeked.connect(self._on_canvas_playhead_seeked)
        self.waveform_widget.canvas.playheadScrubbed.connect(self._on_canvas_playhead_scrubbed)
        self.waveform_widget.canvas.playheadScrubEnded.connect(self._on_canvas_playhead_scrub_ended)
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
        self._loading_project_ui = True
        try:
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

            self.combo_language.blockSignals(True)
            l_idx = self.combo_language.findData(proj.alignment_config.language)
            self.combo_language.setCurrentIndex(max(0, l_idx))
            self.combo_language.blockSignals(False)
            self._last_known_lang = (
                proj.alignment_config.language
                if proj.alignment_config.language in {"zh", "en"}
                else proj.active_template.language
            )

            self.combo_asr_model.blockSignals(True)
            m_idx = self.combo_asr_model.findData(proj.alignment_config.asr_model)
            self.combo_asr_model.setCurrentIndex(max(0, m_idx))
            self.combo_asr_model.blockSignals(False)

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

            self.combo_export_scope.blockSignals(True)
            if proj.export_config.export_srt and not proj.export_config.export_wav:
                scope_val = "srt_only"
            elif proj.export_config.export_wav and not proj.export_config.export_srt:
                scope_val = "wav_only"
            else:
                scope_val = "both"
            s_idx = self.combo_export_scope.findData(scope_val)
            self.combo_export_scope.setCurrentIndex(max(0, s_idx))
            self.combo_export_scope.blockSignals(False)
        finally:
            self._loading_project_ui = False
        if hasattr(self, "btn_engine_settings"):
            self._refresh_engine_settings_summary()

    def _sync_project_settings_to_memory(self) -> None:
        if self._loading_project_ui:
            return
        proj = self.store.get_active_project()
        if not self._suppress_project_template_save:
            cur_tpl = self.template_panel.current_template()
            proj.set_template_for_language(cur_tpl.language, cur_tpl)
        proj.audio_config = self.audio_panel.current_config()
        proj.alignment_config.language = self.combo_language.currentData() or "auto"
        proj.alignment_config.asr_model = self.combo_asr_model.currentData() or "auto"
        proj.alignment_config.backend = self.combo_aligner.currentData() or "auto"
        proj.alignment_config.device = self.combo_device.currentData() or "auto"
        proj.export_config.output_dir = self.edit_out_dir.text().strip()
        exp_scope = self.combo_export_scope.currentData() or "both"
        proj.export_config.export_srt = exp_scope in {"both", "srt_only"}
        proj.export_config.export_wav = exp_scope in {"both", "wav_only"}
        proj.export_config.export_muxed_mp4 = self.chk_mux_mp4.isChecked()
        proj.export_config.wav_bit_depth = self.combo_bit_depth.currentData() or 24
        proj.batch_use_unified_template = self.template_panel.chk_unified_batch.isChecked()
        self._refresh_engine_settings_summary()

    def _save_project_settings(self) -> None:
        if self._loading_project_ui:
            return
        self._sync_project_settings_to_memory()
        self.store.save_projects()

    def _on_language_changed(self, _idx: int) -> None:
        if self._loading_project_ui:
            return
        lang_sel = self.combo_language.currentData() or "auto"
        proj = self.store.get_active_project()
        if lang_sel in {"zh", "en"} and lang_sel != self._last_known_lang:
            cur_tpl = self.template_panel.current_template()
            proj.set_template_for_language(cur_tpl.language, cur_tpl)
            target_tpl = proj.get_template_for_language(lang_sel)
            self._last_known_lang = lang_sel
            self._suppress_project_template_save = True
            try:
                self.template_panel.set_template(target_tpl, emit_signal=False)
            finally:
                self._suppress_project_template_save = False
        self._save_project_settings()
        self._update_stage_strip()

    def _on_asr_model_changed(self, _idx: int) -> None:
        if self._loading_project_ui:
            return
        model_id = self.combo_asr_model.currentData() or "auto"
        if model_id == "qwen3_asr_1.7b":
            warn = check_model_vram_risk("qwen3_asr_1.7b")
            if warn:
                self.lbl_status.setText(f"⚠ {warn}")
                self.anomaly_log_panel.append_log(f"[显存提醒] {warn}")
        self._save_project_settings()
        self._update_stage_strip()

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
                "请拖入粗剪视频（MP4 / MOV / MKV 等）、剧本文稿 (.md/.txt) 或 SRT 字幕。",
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

        if added:
            disambiguate_episode_labels(self.tasks)
            if not self.current_task_id:
                self.current_task_id = added[0].task_id

        self.import_panel.set_tasks(self.tasks, self.current_task_id)
        if len(self.tasks) > 1:
            idx_b = self.combo_run_scope.findData("batch")
            if idx_b >= 0 and len(added) > 1:
                self.combo_run_scope.setCurrentIndex(idx_b)
        self._update_stage_strip()
        if added:
            script_cnt = sum(1 for t in added if t.script_path)
            extra_note = f"（已自动配对 {script_cnt} 份台词稿）" if script_cnt else ""
            self.anomaly_log_panel.append_log(
                f"导入 {len(added)} 个剧集任务{extra_note}，正在后台检测视频编码与音轨..."
            )
            worker = MediaProbeWorker(added)
            self._probe_worker = worker
            self._probe_workers.append(worker)
            worker.probed.connect(self._on_media_probed)
            worker.probeFailed.connect(self._on_probe_failed)
            worker.finished.connect(lambda w=worker: self._cleanup_probe_worker(w))
            worker.start()

    def _cleanup_probe_worker(self, worker: MediaProbeWorker) -> None:
        if worker in self._probe_workers:
            self._probe_workers.remove(worker)
        if self._probe_worker is worker:
            self._probe_worker = None
        worker.deleteLater()

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
        if not cur_tpl.is_builtin or cur_tpl.language == "en":
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
            self.center_bottom_tabs.setTabText(0, "字幕校对编辑 (0)")
            self.anomaly_log_panel.set_anomalies([])
        self._update_stage_strip()

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
            self._update_stage_strip()
            self.anomaly_log_panel.append_log(f"已手动绑定 SRT：{Path(srt_path).name} -> {task.episode_label}（保留已有文稿）")

    def _on_script_bound(self, task_id: str, script_path: str) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task:
            task.script_path = script_path
            self.import_panel.set_tasks(self.tasks, task_id)
            self._update_stage_strip()
            self.anomaly_log_panel.append_log(f"已手动绑定台词文稿：{Path(script_path).name} -> {task.episode_label}（保留已有 SRT）")

    def _on_task_mode_changed(self, task_id: str, mode_val: str) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task:
            task.task_mode = mode_val  # type: ignore[assignment]
            self.import_panel.set_tasks(self.tasks, task_id)
            self._update_stage_strip()
            self.anomaly_log_panel.append_log(
                f"已切换 {task.episode_label} 处理模式为：{task.resolve_effective_mode().label_zh}"
            )

    def _on_remove_episode(self, task_id: str) -> None:
        if self.current_task_id == task_id:
            self.audio_panel.release_sources()
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
            self._update_stage_strip()

    def _on_template_changed(self, tpl: SubtitleTemplate) -> None:
        if not self.template_panel.chk_unified_batch.isChecked() and self.current_task_id:
            task = next((t for t in self.tasks if t.task_id == self.current_task_id), None)
            if task:
                task.template_override = deepcopy(tpl)
        if not self._loading_project_ui:
            self._sync_project_settings_to_memory()
            self._save_prefs_timer.start()

    def _on_batch_mode_changed(self, unified: bool) -> None:
        self._save_project_settings()

    def _on_audio_cfg_changed(self, cfg: AudioLevelingConfig) -> None:
        self._save_project_settings()

    def _start_processing(
        self,
        *,
        batch: bool | None = None,
        stage_action: str = "full",
        force_recompute: bool = False,
    ) -> None:
        if self._proc_worker is not None and self._proc_worker.isRunning():
            QMessageBox.information(self, "处理中", "后台任务正在运行，请等待完成或点击取消。")
            return
        if not self.tasks:
            QMessageBox.information(self, "请先导入文件", "请先拖入或导入粗剪视频（MP4/MOV/MKV 等）、可选文稿或 SRT。")
            return

        self._save_project_settings()
        proj = self.store.get_active_project()

        if batch is None:
            use_batch = (self.combo_run_scope.currentData() == "batch") and (stage_action == "full")
        else:
            use_batch = batch

        if use_batch:
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
            elif global_tpl.is_builtin and global_tpl.language != "en" and t.media_info is not None:
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

        self._handled_missing_model_tasks.clear()
        self._last_prog_ui_ts = 0.0
        self._last_prog_pct = -1
        self._save_prefs_timer.stop()
        self.audio_panel.release_sources()
        self.import_panel.set_tasks(self.tasks, self.current_task_id)
        self.btn_start.setEnabled(False)
        self.btn_stage_actions.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)

        cache_dir = Path(self.store.preferences["model_cache_dir"])
        work_root = Path(self.store.preferences.get("work_cache_dir") or default_work_cache_dir(self.store.base_dir))
        cache_max_mb = int(self.store.preferences.get("cache_max_size_mb", 4096))

        self._proc_worker = ProcessingWorker(
            jobs=jobs,
            alignment_cfg=proj.alignment_config,
            export_cfg=proj.export_config,
            cache_dir=cache_dir,
            work_root=work_root,
            stage_action=stage_action,  # type: ignore[arg-type]
            force_recompute=force_recompute,
            templates_by_lang=proj.active_templates_by_lang,
            cache_max_size_mb=cache_max_mb,
        )
        self._proc_worker.taskProgress.connect(self._on_task_progress)
        self._proc_worker.taskCompleted.connect(self._on_task_completed)
        self._proc_worker.missingModel.connect(self._on_missing_model)
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
        pct = int(progress * 100)
        now = time.monotonic()
        if pct != self._last_prog_pct or (now - self._last_prog_ui_ts) >= 0.08:
            self._last_prog_pct = pct
            self._last_prog_ui_ts = now
            self.lbl_status.setText(f"[{task.episode_label if task else ''}] {msg}")
            self.progress_bar.setValue(pct)
            if task is not None:
                self.import_panel.update_task_progress(task_id, task)

    def _on_task_completed(self, task_id: str, result: PipelineResult) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task:
            task.status = "done"
            task.progress = 1.0
            task.media_info = result.media_info
            self.import_panel.update_task_progress(task_id, task)
        self.results[task_id] = result
        if task_id == self.current_task_id:
            self._display_pipeline_result(result)
        self._update_stage_strip()

    def _on_missing_model(self, task_id: str, model_id: str, err: str) -> None:
        self._handled_missing_model_tasks.add(task_id)
        ans = QMessageBox.question(
            self,
            "需要下载 AI 语音/对齐模型",
            f"{err}\n\n是否立即打开「模型下载与缓存管理」下载 {model_id}？",
        )
        if ans == QMessageBox.StandardButton.Yes:
            self._open_model_manager(highlight_model_id=model_id)

    def _on_task_failed(self, task_id: str, err: str) -> None:
        task = next((t for t in self.tasks if t.task_id == task_id), None)
        if task:
            task.status = "error"
            task.error_message = err
            self.import_panel.update_task_progress(task_id, task)
        if task_id not in self._handled_missing_model_tasks:
            QMessageBox.warning(self, "处理异常", f"{task.episode_label if task else ''} 处理失败：\n{err}")

    def _on_all_finished(self, cancelled: bool) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stage_actions.setEnabled(True)
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

    def _update_stage_strip(self) -> None:
        cur_task = next((t for t in self.tasks if t.task_id == self.current_task_id), None)
        if cur_task is None:
            self.lbl_stage1.setText("① 导入素材：等待导入粗剪视频")
            self.lbl_stage2.setText("② 字幕识别：语言 - · 模型 - · 对白 0 条 · 异常 0 项")
            self.lbl_stage3.setText("③ 精修导出：等待对齐、排版与音量均衡")
            return

        eff_mode = cur_task.resolve_effective_mode().label_zh
        assets = []
        if cur_task.script_path:
            assets.append("文稿")
        if cur_task.srt_path:
            assets.append("SRT")
        asset_note = "+".join(assets) if assets else "仅视频"
        self.lbl_stage1.setText(f"① 导入素材：{cur_task.episode_label} ({asset_note}) · 模式: {eff_mode}")

        res = self.results.get(cur_task.task_id)
        if res is None:
            lang_sel = self.combo_language.currentText()
            asr_sel = self.combo_asr_model.currentText()
            self.lbl_stage2.setText(f"② 字幕识别：预设语言 {lang_sel} · ASR {asr_sel} · 待执行")
            self.lbl_stage3.setText("③ 精修导出：自动对齐 · 排版 · 音量均衡 · 质检导出")
            return

        lang_disp = "English" if res.detected_language == "en" else "中文"
        asr_disp = res.asr_model_used or ("已有SRT" if res.task_mode_used == "srt_fix" else "-")
        match_part = ""
        if res.script_match_report:
            conf_pct = int(round(res.script_match_report.get("overall_confidence", 0.0) * 100))
            match_part = f" · 匹配 {conf_pct}%"
        self.lbl_stage2.setText(
            f"② 字幕识别：{lang_disp} · {asr_disp} · 对白 {len(res.processed_cues)} 条{match_part} · 异常 {len(res.anomalies)}"
        )

        sync_txt = "同步✓" if res.sync_verified else f"偏移{res.sync_diff_ms:+.0f}ms"
        cache_txt = f" · 缓存({len(res.cache_hits)})" if res.cache_hits else ""
        self.lbl_stage3.setText(
            f"③ 精修导出：{res.aligner_used} · {res.audio_report.before_lufs:.1f}→{res.audio_report.after_lufs:.1f} LUFS · {sync_txt}{cache_txt}"
        )

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

        # Sync template language to processed result when auto-detected English/Chinese
        cur_tpl = self.template_panel.current_template()
        if res.detected_language != cur_tpl.language:
            proj = self.store.get_active_project()
            target_tpl = proj.get_template_for_language(res.detected_language)
            self._suppress_project_template_save = True
            try:
                self.template_panel.set_template(target_tpl, emit_signal=False)
            finally:
                self._suppress_project_template_save = False
            cur_tpl = target_tpl

        self.subtitle_editor.set_data(res.processed_cues, cur_tpl)
        self.center_bottom_tabs.setTabText(0, f"字幕校对编辑 ({len(res.processed_cues)})")
        self.anomaly_log_panel.set_anomalies(res.anomalies)
        if res.processed_cues:
            sample = " ".join(c.raw_text for c in res.processed_cues[:3]).strip()
            self.template_panel.set_preview_sample_text(sample)
        self._update_stage_strip()

    def _on_cues_edited(self, description: str, before: list[SubtitleCue], after: list[SubtitleCue]) -> None:
        if self.current_task_id not in self.results:
            return
        tid = self.current_task_id

        def _apply(cues_snapshot: list[SubtitleCue]) -> None:
            import numpy as np

            res = self.results.get(tid)
            if res is None:
                return
            tpl = self.template_panel.current_template()
            cleaned = normalize_cues_for_resolve(deepcopy(cues_snapshot), min_gap_ms=1)
            res.processed_cues = cleaned
            cached_a16 = self._audio_16k_cache.get(res.extracted_wav_path)
            audio_16k = cached_a16 if cached_a16 is not None else np.zeros(0, dtype=np.float32)
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
                self.center_bottom_tabs.setTabText(0, f"字幕校对编辑 ({len(cleaned)})")
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
        if self._relayout_worker is not None and self._relayout_worker.isRunning():
            return
        tpl = self.template_panel.current_template()
        self.lbl_status.setText("正在按当前模板后台重新语义断句...")
        cached_16k = self._audio_16k_cache.get(res.extracted_wav_path)
        self._relayout_worker = RelayoutWorker(
            task_id=self.current_task_id,
            raw_cues=res.processed_cues,
            template=tpl,
            duration_ms=res.audio_report.duration_ms,
            vad_spans_ms=res.vad_spans_ms,
            wav_path=res.extracted_wav_path,
            cached_16k=cached_16k,
            video_fps=res.media_info.fps if res.media_info else 25.0,
        )
        self._relayout_worker.finishedOk.connect(self._on_relayout_done)
        self._relayout_worker.failed.connect(lambda _tid, e: QMessageBox.warning(self, "重排失败", e))
        self._relayout_worker.start()

    def _on_relayout_done(self, task_id: str, before_cues: object, new_cues: object, a16: object) -> None:
        res = self.results.get(task_id)
        if res is None:
            return
        if res.extracted_wav_path and a16 is not None:
            self._audio_16k_cache[res.extracted_wav_path] = a16
        if task_id != self.current_task_id:
            return
        tpl = self.template_panel.current_template()
        self._on_cues_edited("按当前模板重新语义断句", before_cues, new_cues)
        self.lbl_status.setText(
            f"已按当前模板 ({tpl.max_chars_per_line}字/{tpl.max_lines}行) 重新断句，共 {len(new_cues)} 条。"
        )

    def _relevel_current_audio(self) -> None:
        res = self.results.get(self.current_task_id)
        if res is None or not res.extracted_wav_path:
            QMessageBox.information(self, "提示", "请先点击「开始处理」提取音轨后再重新应用音量均衡。")
            return
        if self._relevel_worker and self._relevel_worker.isRunning():
            return
        if not res.processed_wav_path:
            res.processed_wav_path = str(Path(res.extracted_wav_path).with_name(f"{res.episode_label}_relevel.wav"))

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

    def _on_canvas_playhead_scrub_ended(self) -> None:
        self.audio_panel._scrubber.stop()
        self._preview_sync_timer.stop()
        self._flush_preview_sample_text()

    def _on_canvas_playhead_seeked(self, ms: int) -> None:
        self.audio_panel.seek_to_ms(ms)
        self._sync_live_preview_at_ms(ms, immediate=True)

    def _on_playhead_moved(self, ms: int) -> None:
        self.waveform_widget.canvas.set_playhead(ms)
        self._sync_live_preview_at_ms(ms)

    def _sync_live_preview_at_ms(self, ms: int, *, immediate: bool = False) -> None:
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
            self._pending_preview_text = cues[hit_idx].raw_text
            if immediate:
                self._preview_sync_timer.stop()
                self._flush_preview_sample_text()
            else:
                self._preview_sync_timer.start()

    def _flush_preview_sample_text(self) -> None:
        if self._pending_preview_text:
            self.template_panel.set_preview_sample_text(self._pending_preview_text)

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

    def _open_model_manager(self, *, highlight_model_id: str = "") -> None:
        dlg = ModelManagerDialog(self.store, self, highlight_model_id=highlight_model_id)
        dlg.exec()

    def _open_engine_settings(self) -> None:
        self._engine_settings_dlg.exec()
        self._refresh_engine_settings_summary()

    def _refresh_engine_settings_summary(self) -> None:
        align_txt = self.combo_aligner.currentText().split("(")[0].strip()
        dev_txt = self.combo_device.currentText().split("(")[0].strip()
        bit_txt = f"{self.combo_bit_depth.currentData() or 24}-bit"
        mux_txt = " · 合成MP4" if self.chk_mux_mp4.isChecked() else ""
        self.btn_engine_settings.setToolTip(
            f"当前参数：对齐={align_txt} · 设备={dev_txt} · 音频={bit_txt}/48k{mux_txt}\n"
            "点击打开二级设置修改强制对齐引擎、计算设备与导出格式"
        )

    def _toggle_side_panel(self, which: str, visible: bool) -> None:
        if which == "import":
            self.import_panel.setVisible(visible)
            self.store.preferences["show_import_panel"] = visible
        elif which == "template":
            self.template_panel.setVisible(visible)
            self.store.preferences["show_template_panel"] = visible
        self.store.save_prefs()

    def closeEvent(self, event) -> None:
        from drama_studio.align.runtime import MODEL_POOL

        if self._save_prefs_timer.isActive():
            self._save_prefs_timer.stop()
            self._save_project_settings()
        self._preview_sync_timer.stop()

        if self._proc_worker is not None and self._proc_worker.isRunning():
            self._proc_worker.request_cancel()
            self._proc_worker.wait(3000)
        for pw in list(self._probe_workers):
            if pw.isRunning():
                pw.requestInterruption()
                pw.wait(1000)
        for rw in (self._relevel_worker, self._relayout_worker):
            if rw is not None and rw.isRunning():
                rw.requestInterruption()
                rw.wait(1500)

        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self.audio_panel.release_sources()
        MODEL_POOL.unload_all()
        super().closeEvent(event)
