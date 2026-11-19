"""Editable subtitle preview table inside glass card with semantic break editing and integrity checks."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from drama_studio.gui.theme import get_palette
from drama_studio.gui.widgets.glass_card import GlassCard
from drama_studio.models import SubtitleCue, SubtitleTemplate
from drama_studio.nlp.punctuation import extract_canonical_spoken_sequence
from drama_studio.nlp.segmenter import measure_text_width, width_fits_limit
from drama_studio.nlp.srt_io import format_srt_time


class SubtitleEditorPanel(GlassCard):
    """Interactive table for inspecting and fine-tuning aligned & semantically segmented subtitles."""

    cueSelected = Signal(int)
    cuesEdited = Signal(str, list, list)
    relayoutRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("智能断句预览与编辑", badge="共 0 条", parent=parent)
        self._cues: list[SubtitleCue] = []
        self._template: SubtitleTemplate = SubtitleTemplate.preset_portrait_9_16()
        self._updating = False
        self._visible_indices: list[int] = []
        self._theme_mode: str = "light"
        self._build_ui()

    def set_theme(self, mode: str) -> None:
        self._theme_mode = mode
        if self._cues:
            self._refresh_table()

    def _build_ui(self) -> None:
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.chk_only_anomaly = QCheckBox("仅显示异常")
        self.chk_only_anomaly.toggled.connect(self._refresh_table)

        self.btn_merge = QPushButton("合并相邻")
        self.btn_split = QPushButton("拆分选中")
        self.btn_shift_back = QPushButton("-50ms")
        self.btn_shift_fwd = QPushButton("+50ms")
        self.btn_relayout = QPushButton("按当前模板重排断句")
        self.btn_relayout.setObjectName("PrimaryButton")

        self.btn_merge.clicked.connect(self._merge_selected)
        self.btn_split.clicked.connect(self._split_selected)
        self.btn_shift_back.clicked.connect(lambda: self._nudge_selected(-50))
        self.btn_shift_fwd.clicked.connect(lambda: self._nudge_selected(50))
        self.btn_relayout.clicked.connect(self.relayoutRequested.emit)

        bar.addWidget(self.chk_only_anomaly)
        bar.addStretch()
        bar.addWidget(self.btn_merge)
        bar.addWidget(self.btn_split)
        bar.addWidget(self.btn_shift_back)
        bar.addWidget(self.btn_shift_fwd)
        bar.addWidget(self.btn_relayout)
        self.body().addLayout(bar)

        self.table = QTableWidget(0, 7)
        self.table.setMinimumHeight(96)
        self.table.setHorizontalHeaderLabels(
            ["#", "修正时间码", "时长", "字宽/行数", "语速", "置信度", "显示字幕文本 (双击可手动调整换行)"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(True)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)

        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemChanged.connect(self._on_item_changed)
        self.body().addWidget(self.table, 1)

    def set_data(self, cues: list[SubtitleCue], template: SubtitleTemplate) -> None:
        self._cues = deepcopy(cues)
        self._template = deepcopy(template)
        self._refresh_table()

    def select_cue_index(self, cue_idx: int) -> None:
        if cue_idx < 0 or cue_idx >= len(self._cues):
            return
        if cue_idx in self._visible_indices:
            row = self._visible_indices.index(cue_idx)
            self.table.blockSignals(True)
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0))
            self.table.blockSignals(False)

    def _refresh_table(self) -> None:
        self._updating = True
        self.table.setUpdatesEnabled(False)
        try:
            only_anom = self.chk_only_anomaly.isChecked()
            self._visible_indices = [
                i for i, c in enumerate(self._cues) if (not only_anom or bool(c.anomaly_flags))
            ]
            self.table.setRowCount(len(self._visible_indices))

            anom_count = sum(1 for c in self._cues if c.anomaly_flags)
            if self.badge_label is not None:
                self.badge_label.setText(f"共 {len(self._cues)} 条 · 待核查 {anom_count}")

            pal = get_palette(self._theme_mode)
            c_warn = QColor("#FBBF24" if pal.is_dark else "#D97706")
            c_err = QColor(pal.err_text)
            c_ok = QColor(pal.ok_text)

            for row, c_idx in enumerate(self._visible_indices):
                cue = self._cues[c_idx]
                dur_ms = cue.duration_ms
                text = cue.effective_text
                lines = cue.lines()
                max_w = measure_text_width(text, self._template.width_mode)
                flat_chars = len(extract_canonical_spoken_sequence(text))
                cps = flat_chars / max(0.1, dur_ms / 1000.0)

                it_idx = QTableWidgetItem(str(c_idx + 1))
                it_idx.setFlags(it_idx.flags() & ~Qt.ItemFlag.ItemIsEditable)
                it_idx.setData(Qt.ItemDataRole.UserRole, c_idx)

                time_str = f"{format_srt_time(cue.start_ms)} --> {format_srt_time(cue.end_ms)}"
                it_time = QTableWidgetItem(time_str)
                it_time.setFlags(it_time.flags() & ~Qt.ItemFlag.ItemIsEditable)

                it_dur = QTableWidgetItem(f"{dur_ms}ms")
                it_dur.setFlags(it_dur.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if dur_ms < self._template.min_duration_ms or dur_ms > self._template.max_duration_ms:
                    it_dur.setForeground(c_warn)

                it_dim = QTableWidgetItem(f"{max_w:.1f}/{self._template.max_chars_per_line} · {len(lines)}行")
                it_dim.setFlags(it_dim.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if not width_fits_limit(max_w, self._template.max_chars_per_line) or len(lines) > self._template.max_lines:
                    it_dim.setForeground(c_err)

                it_cps = QTableWidgetItem(f"{cps:.1f}/s")
                it_cps.setFlags(it_cps.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if cps > self._template.max_cps:
                    it_cps.setForeground(c_warn)

                it_conf = QTableWidgetItem(f"{int(round(cue.confidence * 100))}%")
                it_conf.setFlags(it_conf.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if cue.confidence < 0.55:
                    it_conf.setForeground(c_err)
                elif cue.confidence < 0.78:
                    it_conf.setForeground(c_warn)
                else:
                    it_conf.setForeground(c_ok)

                it_txt = QTableWidgetItem(text)
                if cue.anomaly_flags:
                    it_txt.setToolTip("异常标记：" + "；".join(cue.anomaly_flags))

                self.table.setItem(row, 0, it_idx)
                self.table.setItem(row, 1, it_time)
                self.table.setItem(row, 2, it_dur)
                self.table.setItem(row, 3, it_dim)
                self.table.setItem(row, 4, it_cps)
                self.table.setItem(row, 5, it_conf)
                self.table.setItem(row, 6, it_txt)
                self.table.setRowHeight(row, 46 if len(lines) >= 2 else 30)
        finally:
            self.table.setUpdatesEnabled(True)
            self._updating = False

    def _selected_cue_indices(self) -> list[int]:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        return [self._visible_indices[r] for r in rows if 0 <= r < len(self._visible_indices)]

    def _on_selection_changed(self) -> None:
        if self._updating:
            return
        sel = self._selected_cue_indices()
        if sel:
            self.cueSelected.emit(sel[0])

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != 6:
            return
        row = item.row()
        if row < 0 or row >= len(self._visible_indices):
            return
        c_idx = self._visible_indices[row]
        new_text = item.text().strip()
        old_cue = self._cues[c_idx]
        if new_text == old_cue.effective_text:
            return

        old_seq = extract_canonical_spoken_sequence(old_cue.effective_text)
        new_seq = extract_canonical_spoken_sequence(new_text)
        if old_seq != new_seq:
            ans = QMessageBox.warning(
                self,
                "台词文字变动提醒",
                "检测到您修改了台词文字本身（而不仅是调整换行或标点）。\n"
                "为保证不漏字不改词，建议仅修改换行。是否保留本次修改？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                self._refresh_table()
                return

        before = deepcopy(self._cues)
        after = deepcopy(self._cues)
        after[c_idx].display_text = new_text
        self._cues = after
        self.cuesEdited.emit(f"编辑第 {c_idx + 1} 条字幕文本", before, after)
        self._refresh_table()

    def _merge_selected(self) -> None:
        sel = self._selected_cue_indices()
        if len(sel) < 2:
            QMessageBox.information(self, "提示", "请按住 Ctrl 或 Shift 选中两条及以上连续字幕再合并。")
            return
        if sel != list(range(sel[0], sel[-1] + 1)):
            QMessageBox.warning(self, "提示", "只能合并连续相邻的字幕条目。")
            return

        before = deepcopy(self._cues)
        after = deepcopy(self._cues)
        group = after[sel[0] : sel[-1] + 1]
        merged_raw = " ".join(c.raw_text.replace("\n", " ").strip() for c in group)
        merged_disp = "".join(c.effective_text.replace("\n", "") for c in group)
        avg_conf = sum(c.confidence for c in group) / len(group)

        merged_cue = SubtitleCue(
            index=group[0].index,
            start_ms=group[0].start_ms,
            end_ms=group[-1].end_ms,
            raw_text=merged_raw,
            display_text=merged_disp,
            orig_start_ms=group[0].orig_start_ms,
            orig_end_ms=group[-1].orig_end_ms,
            confidence=avg_conf,
            char_timings=[ct for c in group for ct in c.char_timings],
            source_cue_indices=[idx for c in group for idx in c.source_cue_indices],
        )
        new_list = after[: sel[0]] + [merged_cue] + after[sel[-1] + 1 :]
        for i, c in enumerate(new_list, start=1):
            c.index = i
        self._cues = new_list
        self.cuesEdited.emit(f"合并第 {sel[0] + 1}~{sel[-1] + 1} 条字幕", before, new_list)
        self._refresh_table()

    def _split_selected(self) -> None:
        sel = self._selected_cue_indices()
        if len(sel) != 1:
            QMessageBox.information(self, "提示", "请选中 1 条字幕进行拆分。")
            return
        idx = sel[0]
        cue = self._cues[idx]
        flat = cue.effective_text.replace("\n", "")
        if len(flat) < 2:
            QMessageBox.information(self, "提示", "该字幕字数太少，无法继续拆分。")
            return

        before = deepcopy(self._cues)
        after = deepcopy(self._cues)
        lines = cue.lines()
        if len(lines) >= 2:
            t1 = lines[0]
            t2 = "".join(lines[1:])
        else:
            mid = len(flat) // 2
            t1 = flat[:mid]
            t2 = flat[mid:]

        split_ms = cue.start_ms + int(round(cue.duration_ms * (len(t1) / max(1, len(flat)))))
        if cue.char_timings and len(cue.char_timings) >= len(t1) and len(t1) > 0:
            ct_idx = min(len(cue.char_timings) - 1, len(t1) - 1)
            cand_ms = cue.char_timings[ct_idx].end_ms
            if cue.start_ms + 80 <= cand_ms <= cue.end_ms - 80:
                split_ms = cand_ms

        gap = min(self._template.min_gap_ms, max(10, cue.duration_ms // 8))
        c1 = SubtitleCue(
            index=cue.index,
            start_ms=cue.start_ms,
            end_ms=max(cue.start_ms + 80, split_ms - gap // 2),
            raw_text=t1,
            display_text=t1,
            confidence=cue.confidence,
        )
        c2 = SubtitleCue(
            index=cue.index + 1,
            start_ms=min(cue.end_ms - 80, c1.end_ms + gap),
            end_ms=cue.end_ms,
            raw_text=t2,
            display_text=t2,
            confidence=cue.confidence,
        )
        new_list = after[:idx] + [c1, c2] + after[idx + 1 :]
        for i, c in enumerate(new_list, start=1):
            c.index = i
        self._cues = new_list
        self.cuesEdited.emit(f"拆分第 {idx + 1} 条字幕", before, new_list)
        self._refresh_table()

    def _nudge_selected(self, delta_ms: int) -> None:
        sel = self._selected_cue_indices()
        if not sel:
            return
        before = deepcopy(self._cues)
        after = deepcopy(self._cues)
        for idx in sel:
            c = after[idx]
            c.start_ms = max(0, c.start_ms + delta_ms)
            c.end_ms = max(c.start_ms + 80, c.end_ms + delta_ms)
        self._cues = after
        self.cuesEdited.emit(f"微调选中字幕时间 ({delta_ms:+d} ms)", before, after)
        self._refresh_table()
