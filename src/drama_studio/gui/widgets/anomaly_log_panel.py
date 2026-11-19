"""Anomaly review table (click to jump) and real-time processing log console."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from drama_studio.gui.theme import get_palette
from drama_studio.models import AnomalyItem, AnomalySeverity
from drama_studio.nlp.srt_io import format_srt_time


class AnomalyAndLogPanel(QWidget):
    """Bottom tabbed glass drawer for anomaly inspection and execution logs."""

    anomalyClicked = Signal(int, object)
    anomaliesCountChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None, *, embedded: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("TransparentContainer")
        self._embedded = embedded
        self._anomalies: list[AnomalyItem] = []
        self._theme_mode: str = "light"
        self._build_ui()

    def set_theme(self, mode: str) -> None:
        self._theme_mode = mode
        if self._anomalies:
            self.set_anomalies(self._anomalies)

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tabs: QTabWidget | None = None

        self.anom_page = QWidget()
        self.anom_page.setObjectName("TransparentContainer")
        a_lay = QVBoxLayout(self.anom_page)
        a_lay.setContentsMargins(6, 6, 6, 6)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["级别", "时间区间", "对应字幕", "异常说明", "处理建议"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._on_cell_clicked)
        a_lay.addWidget(self.table)

        self.log_page = QWidget()
        self.log_page.setObjectName("TransparentContainer")
        l_lay = QVBoxLayout(self.log_page)
        l_lay.setContentsMargins(6, 6, 6, 6)
        l_bar = QHBoxLayout()
        btn_clear = QPushButton("清空日志")
        btn_clear.clicked.connect(lambda: self.log_edit.clear())
        l_bar.addStretch()
        l_bar.addWidget(btn_clear)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(3000)
        l_lay.addLayout(l_bar)
        l_lay.addWidget(self.log_edit)

        if not self._embedded:
            self.tabs = QTabWidget()
            self.tabs.addTab(self.anom_page, "异常区间检查 (0)")
            self.tabs.addTab(self.log_page, "运行日志")
            lay.addWidget(self.tabs)

    def set_anomalies(self, anomalies: list[AnomalyItem]) -> None:
        self._anomalies = list(anomalies)
        if self.tabs is not None:
            self.tabs.setTabText(0, f"异常区间检查 ({len(anomalies)})")
        self.anomaliesCountChanged.emit(len(anomalies))
        self.table.setRowCount(len(anomalies))

        pal = get_palette(self._theme_mode)
        sev_map = {
            AnomalySeverity.ERROR: ("严重", QColor(pal.err_text)),
            AnomalySeverity.WARNING: ("警告", QColor("#FBBF24" if pal.is_dark else "#D97706")),
            AnomalySeverity.INFO: ("提示", QColor(pal.accent)),
        }
        default_col = QColor(pal.muted)

        for row, item in enumerate(anomalies):
            sev_txt, sev_col = sev_map.get(item.severity, ("提示", default_col))
            it_sev = QTableWidgetItem(sev_txt)
            it_sev.setForeground(sev_col)

            span_txt = f"{format_srt_time(item.start_ms)} - {format_srt_time(item.end_ms)}"
            cue_txt = f"第 {item.cue_index} 条" if item.cue_index is not None else "整集音频"

            self.table.setItem(row, 0, it_sev)
            self.table.setItem(row, 1, QTableWidgetItem(span_txt))
            self.table.setItem(row, 2, QTableWidgetItem(cue_txt))
            self.table.setItem(row, 3, QTableWidgetItem(item.message))
            self.table.setItem(row, 4, QTableWidgetItem(item.suggestion))
            self.table.setRowHeight(row, 28)

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._anomalies):
            item = self._anomalies[row]
            self.anomalyClicked.emit(item.start_ms, item.cue_index)

    def append_log(self, msg: str) -> None:
        self.log_edit.appendPlainText(msg)
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())
