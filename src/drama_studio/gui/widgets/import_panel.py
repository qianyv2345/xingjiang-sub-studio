"""Video/SRT drag-and-drop import, batch episode list, and media info inspector."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from drama_studio.gui.theme import get_palette
from drama_studio.gui.widgets.glass_card import GlassCard
from drama_studio.gui.widgets.themed_controls import ThemedComboBox
from drama_studio.models import EpisodeTask, MediaInfo


class DropZoneFrame(QFrame):
    """Glassmorphism drag-and-drop target accepting MP4, MOV, MKV, SRT, and folders."""

    filesDropped = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(96)
        self._theme_mode: str = "light"
        self._hovered: bool = False

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        self.icon_lbl = QLabel("✦  拖拽粗剪视频、SRT 与台词文稿到此处  ✦")
        self.icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.sub_lbl = QLabel("支持 MP4 / MOV / MKV / SRT / 台词文稿 (.md/.txt) · 多集自动配对")
        self.sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub_lbl.setWordWrap(True)

        lay.addWidget(self.icon_lbl)
        lay.addWidget(self.sub_lbl)
        self.set_theme("light")

    def set_theme(self, mode: str) -> None:
        self._theme_mode = mode
        pal = get_palette(mode)
        self.icon_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {pal.accent}; background: transparent;"
        )
        self.sub_lbl.setStyleSheet(
            f"font-size: 11px; color: {pal.muted}; background: transparent;"
        )
        self._set_hover(self._hovered)

    def _set_hover(self, hovered: bool) -> None:
        self._hovered = hovered
        pal = get_palette(self._theme_mode)
        if hovered:
            self.setStyleSheet(
                f"DropZoneFrame {{ background-color: {pal.selection}; "
                f"border: 1.5px dashed {pal.accent}; border-radius: 10px; }}"
            )
        else:
            self.setStyleSheet(
                f"DropZoneFrame {{ background-color: {pal.glass_surface_alt}; "
                f"border: 1.5px dashed {pal.glass_border}; border-radius: 10px; }}"
            )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._set_hover(False)

    def dropEvent(self, event: QDropEvent) -> None:
        self._set_hover(False)
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls if u.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()


class ImportBatchPanel(QWidget):
    """Left glass panel managing multi-episode queue, SRT pairing, and FFprobe details."""

    filesImported = Signal(list)
    episodeSelected = Signal(str)
    audioTrackChanged = Signal(str, int)
    srtBound = Signal(str, str)
    scriptBound = Signal(str, str)
    removeEpisodeRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TransparentContainer")
        self._tasks: list[EpisodeTask] = []
        self._current_task_id: str = ""
        self._loading_tracks = False
        self._theme_mode: str = "light"
        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        # Card 1: Dropzone & Import Buttons (2x2 grid prevents button clipping)
        imp_card = GlassCard("素材导入", badge="无需手动导WAV")
        imp_card.body().addWidget(self._make_dropzone())

        grid = QGridLayout()
        grid.setSpacing(6)
        self.btn_add_files = QPushButton("导入视频 / SRT / 台词稿")
        self.btn_add_folder = QPushButton("导入剧集文件夹")
        self.btn_bind_srt = QPushButton("绑定 / 替换 SRT")
        self.btn_bind_script = QPushButton("绑定台词文稿")
        self.btn_remove = QPushButton("移除选中剧集")
        grid.addWidget(self.btn_add_files, 0, 0)
        grid.addWidget(self.btn_add_folder, 0, 1)
        grid.addWidget(self.btn_bind_srt, 1, 0)
        grid.addWidget(self.btn_bind_script, 1, 1)
        grid.addWidget(self.btn_remove, 2, 0, 1, 2)
        imp_card.body().addLayout(grid)
        lay.addWidget(imp_card)

        self.btn_add_files.clicked.connect(self._browse_files)
        self.btn_add_folder.clicked.connect(self._browse_folder)
        self.btn_bind_srt.clicked.connect(self._browse_srt_for_selected)
        self.btn_bind_script.clicked.connect(self._browse_script_for_selected)
        self.btn_remove.clicked.connect(self._remove_selected)

        # Card 2: Batch Queue Table
        queue_card = GlassCard("剧集任务队列", badge="自动配对")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["集数", "视频", "SRT", "状态"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        queue_card.body().addWidget(self.table, 1)
        lay.addWidget(queue_card, 1)

        # Card 3: Video Info & Multi-track Inspector
        info_card = GlassCard("视频信息与音轨基准")

        badges_row = QHBoxLayout()
        badges_row.setSpacing(6)
        self.pill_res = QLabel("分辨率: -")
        self.pill_fps = QLabel("帧率: -")
        self.pill_orient = QLabel("方向: -")
        for p in (self.pill_res, self.pill_fps, self.pill_orient):
            p.setObjectName("PillBadge")
            badges_row.addWidget(p)
        badges_row.addStretch()
        info_card.body().addLayout(badges_row)

        self.lbl_res_fps = QLabel("等待导入视频...")
        self.lbl_res_fps.setObjectName("MutedLabel")
        self.lbl_res_fps.setWordWrap(True)
        info_card.body().addWidget(self.lbl_res_fps)

        self.lbl_script_status = QLabel("台词文稿: 未绑定（可选，防两人台词切在一起）")
        self.lbl_script_status.setObjectName("MutedLabel")
        self.lbl_script_status.setWordWrap(True)
        info_card.body().addWidget(self.lbl_script_status)

        track_row = QHBoxLayout()
        track_row.addWidget(QLabel("目标音轨:"))
        self.combo_track = ThemedComboBox()
        self.combo_track.currentIndexChanged.connect(self._on_track_changed)
        track_row.addWidget(self.combo_track, 1)
        info_card.body().addLayout(track_row)

        self.lbl_warnings = QLabel("")
        self.lbl_warnings.setObjectName("WarningBanner")
        self.lbl_warnings.setWordWrap(True)
        self.lbl_warnings.setVisible(False)
        info_card.body().addWidget(self.lbl_warnings)
        lay.addWidget(info_card)

    def _make_dropzone(self) -> DropZoneFrame:
        self.dropzone = DropZoneFrame()
        self.dropzone.filesDropped.connect(self.filesImported.emit)
        return self.dropzone

    def _browse_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择视频、SRT 字幕与台词文稿",
            "",
            "媒体·字幕·台词 (*.mp4 *.mov *.mkv *.srt *.md *.txt);;视频 (*.mp4 *.mov *.mkv);;SRT 字幕 (*.srt);;台词文稿 (*.md *.txt)",
        )
        if files:
            self.filesImported.emit(files)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择包含剧集视频、SRT 与台词文稿的文件夹")
        if folder:
            self.filesImported.emit([folder])

    def _browse_srt_for_selected(self) -> None:
        if not self._current_task_id:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "为当前剧集绑定 SRT 字幕",
            "",
            "SRT 字幕 (*.srt)",
        )
        if file_path:
            self.srtBound.emit(self._current_task_id, file_path)

    def _browse_script_for_selected(self) -> None:
        if not self._current_task_id:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "为当前剧集绑定台词文稿（防止两人对白合并）",
            "",
            "台词文稿 (*.md *.txt);;所有文件 (*.*)",
        )
        if file_path:
            self.scriptBound.emit(self._current_task_id, file_path)

    def _remove_selected(self) -> None:
        if self._current_task_id:
            self.removeEpisodeRequested.emit(self._current_task_id)

    def set_theme(self, mode: str) -> None:
        self._theme_mode = mode
        if hasattr(self, "dropzone"):
            self.dropzone.set_theme(mode)
        if self._tasks:
            self.set_tasks(self._tasks, select_task_id=self._current_task_id)

    def set_tasks(self, tasks: list[EpisodeTask], select_task_id: str | None = None) -> None:
        self._tasks = tasks
        self.table.blockSignals(True)
        self.table.setRowCount(len(tasks))
        target_row = -1
        pal = get_palette(self._theme_mode)
        c_done = QColor(pal.ok_text)
        c_err = QColor(pal.err_text)
        c_run = QColor(pal.accent)
        c_warn = QColor("#FBBF24" if pal.is_dark else "#D97706")
        for row, task in enumerate(tasks):
            ep_item = QTableWidgetItem(task.episode_label)
            ep_item.setData(Qt.ItemDataRole.UserRole, task.task_id)
            vid_name = Path(task.video_path).name
            srt_base = Path(task.srt_path).name if task.srt_path else "未配对"
            srt_name = f"{srt_base} · 稿" if task.script_path else srt_base
            status_map = {
                "pending": "待处理",
                "running": f"{int(task.progress * 100)}%",
                "done": "已完成",
                "error": "失败",
                "cancelled": "已取消",
            }
            st_item = QTableWidgetItem(status_map.get(task.status, task.status))
            if task.status == "done":
                st_item.setForeground(c_done)
            elif task.status == "error":
                st_item.setForeground(c_err)
            elif task.status == "running":
                st_item.setForeground(c_run)

            srt_item = QTableWidgetItem(srt_name)
            if task.script_path:
                srt_item.setToolTip(f"字幕: {task.srt_path or '未配对'}\n台词稿: {task.script_path}")
            if not task.srt_path:
                srt_item.setForeground(c_warn)

            self.table.setItem(row, 0, ep_item)
            self.table.setItem(row, 1, QTableWidgetItem(vid_name))
            self.table.setItem(row, 2, srt_item)
            self.table.setItem(row, 3, st_item)
            self.table.setRowHeight(row, 30)

            if task.task_id == (select_task_id or self._current_task_id):
                target_row = row

        self.table.blockSignals(False)
        if target_row < 0 and tasks:
            target_row = 0
        if target_row >= 0:
            self.table.selectRow(target_row)
            self._current_task_id = tasks[target_row].task_id
            self.show_media_info(tasks[target_row].media_info)
        else:
            self._current_task_id = ""
            self.show_media_info(None)

    def _on_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        item = self.table.item(row, 0)
        if item is None:
            return
        tid = item.data(Qt.ItemDataRole.UserRole)
        self._current_task_id = tid
        task = next((t for t in self._tasks if t.task_id == tid), None)
        self.show_media_info(task.media_info if task else None)
        self.episodeSelected.emit(tid)

    def show_media_info(self, info: MediaInfo | None, extra_risks: list[str] | None = None) -> None:
        self._loading_tracks = True
        self.combo_track.clear()
        cur_task = next((t for t in self._tasks if t.task_id == self._current_task_id), None)
        if cur_task and cur_task.script_path:
            self.lbl_script_status.setText(f"台词文稿: 已绑定 {Path(cur_task.script_path).name}")
        else:
            self.lbl_script_status.setText("台词文稿: 未绑定（可选，防两人台词切在一起）")

        if info is None:
            self.pill_res.setText("分辨率: -")
            self.pill_fps.setText("帧率: -")
            self.pill_orient.setText("方向: -")
            self.lbl_res_fps.setText("等待导入视频或正在探测...")
            self.lbl_warnings.setVisible(False)
            self._loading_tracks = False
            return

        dur_s = info.duration_ms / 1000.0
        mm = int(dur_s // 60)
        ss = dur_s % 60
        self.pill_res.setText(f"{info.width}×{info.height}")
        self.pill_fps.setText(f"{info.fps:.3g} fps")
        self.pill_orient.setText(f"{info.aspect_ratio} {info.orientation.label_zh}")
        self.lbl_res_fps.setText(
            f"时长 {mm:02d}:{ss:04.1f} · {info.video_codec.upper() or '-'} · 音画偏移 {info.av_offset_ms:+.1f}ms"
        )
        for tr in info.audio_tracks:
            self.combo_track.addItem(tr.display_label(), tr.audio_index)
        if info.audio_tracks:
            idx = max(0, min(info.selected_audio_index, len(info.audio_tracks) - 1))
            self.combo_track.setCurrentIndex(idx)

        all_warn = list(info.warnings)
        if extra_risks:
            for r in extra_risks:
                if r not in all_warn:
                    all_warn.append(r)

        if all_warn:
            self.lbl_warnings.setText("⚠ 风险与同步提示：\n• " + "\n• ".join(all_warn))
            self.lbl_warnings.setVisible(True)
        else:
            self.lbl_warnings.setVisible(False)

        self._loading_tracks = False

    def _on_track_changed(self, idx: int) -> None:
        if self._loading_tracks or idx < 0 or not self._current_task_id:
            return
        audio_idx = self.combo_track.itemData(idx)
        if audio_idx is not None:
            self.audioTrackChanged.emit(self._current_task_id, int(audio_idx))
