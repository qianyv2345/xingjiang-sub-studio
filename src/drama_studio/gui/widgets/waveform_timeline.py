"""Interactive dual-waveform, AGC gain curve, manual gain nodes, and subtitle timeline editor."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)

from drama_studio.gui.theme import get_palette
from drama_studio.gui.widgets.glass_card import GlassCard
from drama_studio.models import AnomalyItem, ManualGainPoint, SubtitleCue


class TimelineCanvas(QWidget):
    """High-DPI QPainter canvas rendering audio waveform, gain curve, manual points, and subtitle blocks."""

    playheadSeeked = Signal(int)
    playheadScrubbed = Signal(int)
    cueSelected = Signal(int)
    cuesEdited = Signal(str, list, list)
    manualPointsEdited = Signal(str, list, list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(124)
        self.setMouseTracking(True)

        self.duration_ms: int = 0
        self.view_start_ms: int = 0
        self.px_per_sec: float = 95.0
        self.playhead_ms: int = 0

        self.wf_times_ms: list[int] = []
        self.wf_before: list[float] = []
        self.wf_after: list[float] = []
        self.gain_times_ms: list[int] = []
        self.gain_db: list[float] = []
        self.vad_spans_ms: list[tuple[int, int]] = []
        self.anomalies: list[AnomalyItem] = []

        self.cues: list[SubtitleCue] = []
        self.selected_cue_idx: int = -1
        self.manual_points: list[ManualGainPoint] = []
        self.theme_mode: str = "light"

        self._drag_mode: str = ""
        self._drag_cue_idx: int = -1
        self._drag_pt_idx: int = -1
        self._drag_before_cues: list[SubtitleCue] = []
        self._drag_before_pts: list[ManualGainPoint] = []

    def set_theme(self, mode: str) -> None:
        self.theme_mode = mode
        self.update()

    def set_data(
        self,
        *,
        duration_ms: int,
        wf_times_ms: list[int],
        wf_before: list[float],
        wf_after: list[float],
        gain_times_ms: list[int],
        gain_db: list[float],
        vad_spans_ms: list[tuple[int, int]],
        cues: list[SubtitleCue],
        anomalies: list[AnomalyItem],
        manual_points: list[ManualGainPoint],
    ) -> None:
        self.duration_ms = max(1000, duration_ms)
        self.wf_times_ms = wf_times_ms
        self.wf_before = wf_before
        self.wf_after = wf_after
        self.gain_times_ms = gain_times_ms
        self.gain_db = gain_db
        self.vad_spans_ms = vad_spans_ms
        self.cues = deepcopy(cues)
        self.anomalies = list(anomalies)
        self.manual_points = deepcopy(manual_points)
        self.update()

    def set_cues(self, cues: list[SubtitleCue]) -> None:
        self.cues = deepcopy(cues)
        self.update()

    def set_manual_points(self, pts: list[ManualGainPoint]) -> None:
        self.manual_points = deepcopy(pts)
        self.update()

    def set_playhead(self, ms: int) -> None:
        self.playhead_ms = max(0, min(self.duration_ms, ms))
        vis_ms = self._visible_duration_ms()
        if self.playhead_ms < self.view_start_ms or self.playhead_ms > self.view_start_ms + vis_ms:
            self.view_start_ms = max(0, min(max(0, self.duration_ms - vis_ms), self.playhead_ms - vis_ms // 4))
        self.update()

    def set_selected_cue(self, idx: int) -> None:
        self.selected_cue_idx = idx
        self.update()

    def focus_time_ms(self, ms: int) -> None:
        vis_ms = self._visible_duration_ms()
        self.view_start_ms = max(0, min(max(0, self.duration_ms - vis_ms), ms - vis_ms // 3))
        self.playhead_ms = max(0, min(self.duration_ms, ms))
        self.update()

    def _visible_duration_ms(self) -> int:
        if self.px_per_sec <= 0:
            return 10000
        return max(500, int(round(self.width() * 1000.0 / self.px_per_sec)))

    def _ms_to_x(self, ms: int) -> float:
        return (ms - self.view_start_ms) * self.px_per_sec / 1000.0

    def _x_to_ms(self, x: float) -> int:
        ms = self.view_start_ms + int(round(x * 1000.0 / max(1.0, self.px_per_sec)))
        return max(0, min(self.duration_ms, ms))

    def _ruler_h(self) -> int:
        return 22

    def _wave_rect(self) -> QRectF:
        rh = self._ruler_h()
        avail = max(60.0, float(self.height() - rh - 4))
        sub_h = min(48.0, max(34.0, avail * 0.34))
        wh = max(36.0, avail - sub_h)
        return QRectF(0, rh, self.width(), wh)

    def _sub_rect(self) -> QRectF:
        wr = self._wave_rect()
        return QRectF(0, wr.bottom() + 2, self.width(), max(30.0, self.height() - wr.bottom() - 3))

    def _gain_db_to_y(self, db: float, wr: QRectF) -> float:
        norm = (db - (-18.0)) / (24.0 - (-18.0))
        norm = max(0.0, min(1.0, norm))
        return wr.bottom() - norm * (wr.height() - 12) - 6

    def _y_to_gain_db(self, y: float, wr: QRectF) -> float:
        norm = (wr.bottom() - 6 - y) / max(10.0, wr.height() - 12)
        db = -18.0 + max(0.0, min(1.0, norm)) * (24.0 - (-18.0))
        return round(max(-12.0, min(12.0, db)), 1)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()

        pal = get_palette(self.theme_mode)
        bg_col = QColor(pal.surface)
        bg_col.setAlpha(196 if not pal.is_dark else 206)
        placeholder_col = QColor(pal.muted)
        outer_border_col = QColor(pal.line)
        outer_border_col.setAlpha(205 if not pal.is_dark else 160)
        ruler_bg = QColor(pal.header_bg)
        ruler_bg.setAlpha(212)
        ruler_line = QColor(pal.line)
        tick_col = QColor(pal.line)
        tick_txt = QColor(pal.muted)
        vad_col = QColor(16, 185, 129, 26)
        before_wf_col = QColor(pal.muted)
        before_wf_col.setAlpha(120)
        after_wf_col = QColor(pal.accent)
        after_wf_col.setAlpha(225)
        zero_guide_col = QColor(pal.line)
        gain_curve_col = QColor("#FBBF24" if pal.is_dark else "#D97706")
        pt_ring_col = QColor(pal.surface)
        pt_text_col = QColor(pal.text)
        sub_bg_col = QColor(pal.background)
        sub_bg_col.setAlpha(175 if not pal.is_dark else 190)
        playhead_col = QColor("#F43F5E" if pal.is_dark else "#E11D48")

        # Rounded canvas
        clip_path = QPainterPath()
        clip_path.addRoundedRect(QRectF(0, 0, w, h), 10.0, 10.0)
        p.setClipPath(clip_path)
        p.fillRect(0, 0, w, h, bg_col)

        if self.duration_ms <= 0:
            p.setPen(placeholder_col)
            p.drawText(
                QRectF(16, 0, w - 32, h),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                "拖入视频与 SRT 后点击顶部「一键智能修正」，在此查看双层人声波形、自动增益曲线与可拖拽字幕时间轴",
            )
            p.setPen(QPen(outer_border_col, 1))
            p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 10.0, 10.0)
            return

        wr = self._wave_rect()
        sr = self._sub_rect()
        vis_end_ms = self.view_start_ms + self._visible_duration_ms()

        # Ruler
        p.fillRect(QRectF(0, 0, w, self._ruler_h()), ruler_bg)
        p.setPen(QPen(ruler_line, 1))
        p.drawLine(0, self._ruler_h(), w, self._ruler_h())

        step_ms = 1000
        if self.px_per_sec < 35:
            step_ms = 5000
        elif self.px_per_sec < 70:
            step_ms = 2000
        elif self.px_per_sec > 200:
            step_ms = 500

        first_tick = (self.view_start_ms // step_ms) * step_ms
        p.setFont(QFont("Consolas", 8))
        for t in range(first_tick, min(self.duration_ms, vis_end_ms) + step_ms, step_ms):
            x = self._ms_to_x(t)
            p.setPen(tick_col)
            p.drawLine(QPointF(x, 12), QPointF(x, self._ruler_h()))
            p.setPen(tick_txt)
            sec = t / 1000.0
            p.drawText(QPointF(x + 4, 15), f"{int(sec // 60)}:{sec % 60:04.1f}")

        # VAD bands
        p.setPen(Qt.PenStyle.NoPen)
        for s_ms, e_ms in self.vad_spans_ms:
            if e_ms < self.view_start_ms or s_ms > vis_end_ms:
                continue
            x1 = max(0.0, self._ms_to_x(s_ms))
            x2 = min(float(w), self._ms_to_x(e_ms))
            p.fillRect(QRectF(x1, wr.top(), max(1.0, x2 - x1), wr.height()), vad_col)

        # Anomaly zones
        for anom in self.anomalies:
            if anom.end_ms < self.view_start_ms or anom.start_ms > vis_end_ms:
                continue
            x1 = max(0.0, self._ms_to_x(anom.start_ms))
            x2 = min(float(w), self._ms_to_x(anom.end_ms))
            col = QColor(239, 68, 68, 42) if anom.severity.value == "error" else QColor(245, 158, 11, 32)
            p.fillRect(QRectF(x1, wr.top(), max(3.0, x2 - x1), wr.height()), col)

        # Dual Waveforms
        mid_y = wr.center().y()
        half_h = wr.height() * 0.43
        if self.wf_times_ms:
            for idx, t in enumerate(self.wf_times_ms):
                if t < self.view_start_ms - 40 or t > vis_end_ms + 40:
                    continue
                x = self._ms_to_x(t)
                b_amp = min(1.0, self.wf_before[idx]) * half_h
                a_amp = min(1.0, self.wf_after[idx]) * half_h if idx < len(self.wf_after) else 0.0

                p.setPen(QPen(before_wf_col, 1.4))
                p.drawLine(QPointF(x, mid_y - b_amp), QPointF(x, mid_y + b_amp))

                p.setPen(QPen(after_wf_col, 1.6))
                p.drawLine(QPointF(x, mid_y - a_amp), QPointF(x, mid_y + a_amp))

        # 0 dB guide line
        y_zero = self._gain_db_to_y(0.0, wr)
        p.setPen(QPen(zero_guide_col, 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(0, y_zero), QPointF(w, y_zero))

        # Auto Gain Curve
        if self.gain_times_ms:
            path = QPainterPath()
            started = False
            for idx, t in enumerate(self.gain_times_ms):
                if t < self.view_start_ms - 80 or t > vis_end_ms + 80:
                    continue
                x = self._ms_to_x(t)
                y = self._gain_db_to_y(self.gain_db[idx], wr)
                if not started:
                    path.moveTo(x, y)
                    started = True
                else:
                    path.lineTo(x, y)
            p.setPen(QPen(gain_curve_col, 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

        # Manual Gain Points
        for pt in self.manual_points:
            if pt.time_ms < self.view_start_ms - 50 or pt.time_ms > vis_end_ms + 50:
                continue
            x = self._ms_to_x(pt.time_ms)
            y = self._gain_db_to_y(pt.gain_db, wr)
            p.setPen(QPen(pt_ring_col, 1.8))
            p.setBrush(QColor(pal.accent if pal.mode == "pink" else pal.ok_start))
            p.drawEllipse(QPointF(x, y), 6.0, 6.0)
            tag_txt = f"{pt.gain_db:+.1f}dB"
            p.setFont(QFont("Microsoft YaHei UI", 8, QFont.Weight.DemiBold))
            tag_rect = QRectF(x + 8, y - 9, 44, 17)
            bg_pill = QColor(pal.surface)
            bg_pill.setAlpha(225)
            p.setPen(QPen(outer_border_col, 1.0))
            p.setBrush(bg_pill)
            p.drawRoundedRect(tag_rect, 4.0, 4.0)
            p.setPen(pt_text_col)
            p.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, tag_txt)

        # Subtitle Timeline Track
        p.fillRect(sr, sub_bg_col)
        p.setPen(QPen(outer_border_col, 1))
        p.drawLine(QPointF(0, sr.top()), QPointF(w, sr.top()))

        p.setFont(QFont("Microsoft YaHei UI", 9))
        for idx, cue in enumerate(self.cues):
            if cue.end_ms < self.view_start_ms or cue.start_ms > vis_end_ms:
                continue
            x1 = self._ms_to_x(cue.start_ms)
            x2 = self._ms_to_x(cue.end_ms)
            rect = QRectF(x1, sr.top() + 4, max(6.0, x2 - x1), max(16.0, sr.height() - 8))

            is_sel = idx == self.selected_cue_idx
            has_err = bool(cue.anomaly_flags)
            if is_sel:
                fill = QColor(pal.selection)
                border = QColor(pal.accent)
            elif has_err:
                fill = QColor("#452708" if pal.is_dark else pal.warn_bg)
                border = QColor("#FBBF24" if pal.is_dark else "#D97706")
            else:
                fill = QColor(pal.surface)
                border = QColor(pal.line)
            cue_txt_col = QColor(pal.text)

            p.setPen(QPen(border, 1.6 if is_sel else 1.0))
            p.setBrush(fill)
            p.drawRoundedRect(rect, 6.0, 6.0)

            txt = cue.effective_text.replace("\n", " / ")
            if rect.width() > 26:
                p.setPen(cue_txt_col)
                p.drawText(rect.adjusted(6, 2, -6, -2), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, txt)

        # Playhead with draggable NLE cap
        px = self._ms_to_x(self.playhead_ms)
        if -8 <= px <= w + 8:
            p.setPen(QPen(playhead_col, 2.2))
            p.drawLine(QPointF(px, 6), QPointF(px, h))
            cap = QPolygonF(
                [
                    QPointF(px - 6.0, 1.0),
                    QPointF(px + 6.0, 1.0),
                    QPointF(px + 6.0, 9.0),
                    QPointF(px, 16.0),
                    QPointF(px - 6.0, 9.0),
                ]
            )
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(playhead_col)
            p.drawPolygon(cap)

        # Outer border
        p.setPen(QPen(outer_border_col, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1, h - 1), 10.0, 10.0)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.duration_ms <= 0:
            return
        # On Windows drivers, Alt+Wheel often populates angleDelta().x() instead of y()
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta == 0:
            return
        mods = event.modifiers()
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            anchor_x = event.position().x()
            anchor_ms = self._x_to_ms(anchor_x)
            factor = 1.20 if delta > 0 else (1.0 / 1.20)
            self.px_per_sec = max(18.0, min(600.0, self.px_per_sec * factor))
            vis = self._visible_duration_ms()
            new_start = anchor_ms - int((anchor_x / max(1e-6, self.px_per_sec)) * 1000.0)
            self.view_start_ms = max(0, min(max(0, self.duration_ms - vis), new_start))
        else:
            vis = self._visible_duration_ms()
            shift = int(vis * 0.18) * (-1 if delta > 0 else 1)
            self.view_start_ms = max(0, min(max(0, self.duration_ms - vis), self.view_start_ms + shift))
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.duration_ms <= 0:
            return
        pos = event.position()
        wr = self._wave_rect()
        sr = self._sub_rect()
        px_playhead = self._ms_to_x(self.playhead_ms)

        if pos.y() <= self._ruler_h() or (
            event.button() == Qt.MouseButton.LeftButton and abs(pos.x() - px_playhead) <= 9.0
        ):
            self._drag_mode = "playhead"
            ms = self._x_to_ms(pos.x())
            self.playhead_ms = ms
            self.playheadScrubbed.emit(ms)
            self.update()
            return

        if wr.contains(pos):
            for idx, pt in enumerate(self.manual_points):
                px = self._ms_to_x(pt.time_ms)
                py = self._gain_db_to_y(pt.gain_db, wr)
                if (pos.x() - px) ** 2 + (pos.y() - py) ** 2 <= 100.0:
                    if event.button() == Qt.MouseButton.RightButton:
                        before = deepcopy(self.manual_points)
                        after = deepcopy(self.manual_points)
                        after.pop(idx)
                        self.manual_points = after
                        self.manualPointsEdited.emit("删除局部音量点", before, after)
                        self.update()
                        return
                    self._drag_mode = "gain_pt"
                    self._drag_pt_idx = idx
                    self._drag_before_pts = deepcopy(self.manual_points)
                    return

            if event.button() == Qt.MouseButton.LeftButton:
                before = deepcopy(self.manual_points)
                t_ms = self._x_to_ms(pos.x())
                g_db = self._y_to_gain_db(pos.y(), wr)
                new_pt = ManualGainPoint(time_ms=t_ms, gain_db=g_db, radius_ms=350)
                after = sorted(deepcopy(self.manual_points) + [new_pt], key=lambda p: p.time_ms)
                self.manual_points = after
                self.manualPointsEdited.emit(f"添加局部音量点 ({g_db:+.1f} dB)", before, after)
                self.update()
            return

        if sr.contains(pos):
            for idx, cue in enumerate(self.cues):
                x1 = self._ms_to_x(cue.start_ms)
                x2 = self._ms_to_x(cue.end_ms)
                if x1 - 6 <= pos.x() <= x2 + 6:
                    self.selected_cue_idx = idx
                    self.cueSelected.emit(idx)
                    if abs(pos.x() - x1) <= 7:
                        self._drag_mode = "cue_left"
                        self._drag_cue_idx = idx
                        self._drag_before_cues = deepcopy(self.cues)
                    elif abs(pos.x() - x2) <= 7:
                        self._drag_mode = "cue_right"
                        self._drag_cue_idx = idx
                        self._drag_before_cues = deepcopy(self.cues)
                    else:
                        self.playhead_ms = cue.start_ms
                        self.playheadSeeked.emit(cue.start_ms)
                    self.update()
                    return

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        wr = self._wave_rect()
        sr = self._sub_rect()

        if self._drag_mode == "playhead":
            # Auto-pan smoothly if dragging near canvas edges
            vis = self._visible_duration_ms()
            if pos.x() > self.width() - 16 and self.view_start_ms + vis < self.duration_ms:
                self.view_start_ms = min(self.duration_ms - vis, self.view_start_ms + max(30, vis // 30))
            elif pos.x() < 16 and self.view_start_ms > 0:
                self.view_start_ms = max(0, self.view_start_ms - max(30, vis // 30))
            ms = self._x_to_ms(pos.x())
            self.playhead_ms = ms
            self.playheadScrubbed.emit(ms)
            self.update()
            return
        if self._drag_mode == "gain_pt" and 0 <= self._drag_pt_idx < len(self.manual_points):
            self.manual_points[self._drag_pt_idx].time_ms = self._x_to_ms(pos.x())
            self.manual_points[self._drag_pt_idx].gain_db = self._y_to_gain_db(pos.y(), wr)
            self.update()
            return
        if self._drag_mode in {"cue_left", "cue_right"} and 0 <= self._drag_cue_idx < len(self.cues):
            ms = self._x_to_ms(pos.x())
            cue = self.cues[self._drag_cue_idx]
            if self._drag_mode == "cue_left":
                min_s = self.cues[self._drag_cue_idx - 1].end_ms + 10 if self._drag_cue_idx > 0 else 0
                cue.start_ms = max(min_s, min(cue.end_ms - 80, ms))
            else:
                max_e = self.cues[self._drag_cue_idx + 1].start_ms - 10 if self._drag_cue_idx + 1 < len(self.cues) else self.duration_ms
                cue.end_ms = min(max_e, max(cue.start_ms + 80, ms))
            self.update()
            return

        if self.duration_ms > 0 and (
            pos.y() <= self._ruler_h() or abs(pos.x() - self._ms_to_x(self.playhead_ms)) <= 9.0
        ):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            return

        if sr.contains(pos):
            for cue in self.cues:
                x1 = self._ms_to_x(cue.start_ms)
                x2 = self._ms_to_x(cue.end_ms)
                if abs(pos.x() - x1) <= 7 or abs(pos.x() - x2) <= 7:
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                    return
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode == "playhead":
            self.playheadSeeked.emit(self.playhead_ms)
        elif self._drag_mode in {"cue_left", "cue_right"}:
            self.cuesEdited.emit("调整字幕时间轴", self._drag_before_cues, deepcopy(self.cues))
        elif self._drag_mode == "gain_pt":
            self.manualPointsEdited.emit("调整局部音量点", self._drag_before_pts, deepcopy(self.manual_points))
        self._drag_mode = ""
        self._drag_cue_idx = -1
        self._drag_pt_idx = -1


class WaveformTimelineWidget(GlassCard):
    """Frosted glass card wrapping the interactive waveform & subtitle timeline."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("人声波形 · 增益曲线 · 字幕时间轴", parent=parent)

        bar = QHBoxLayout()
        legend = QLabel("主色:均衡后  浅灰:原声  黄线:自动增益  圆点:人工节点 · Alt+滚轮缩放 · 拖红线实时预览")
        legend.setObjectName("MutedLabel")
        legend.setStyleSheet("font-size: 11px;")
        bar.addWidget(legend)
        bar.addStretch()
        self.btn_zoom_out = QPushButton("缩小 -")
        self.btn_zoom_out.setToolTip("缩小时间轴 (Alt+滚轮向下 / Ctrl+滚轮向下)")
        self.btn_zoom_in = QPushButton("放大 +")
        self.btn_zoom_in.setToolTip("放大时间轴 (Alt+滚轮向上 / Ctrl+滚轮向上)")
        self.btn_clear_pts = QPushButton("清空节点")
        bar.addWidget(self.btn_zoom_out)
        bar.addWidget(self.btn_zoom_in)
        bar.addWidget(self.btn_clear_pts)
        self.body().addLayout(bar)

        self.canvas = TimelineCanvas()
        self.body().addWidget(self.canvas, 1)

        self.hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self.hscroll.valueChanged.connect(self._on_scroll)
        self.body().addWidget(self.hscroll)

        self.btn_zoom_in.clicked.connect(lambda: self._zoom(1.35))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom(1.0 / 1.35))
        self.btn_clear_pts.clicked.connect(self._clear_points)

    def _zoom(self, factor: float) -> None:
        self.canvas.px_per_sec = max(18.0, min(600.0, self.canvas.px_per_sec * factor))
        self.sync_scrollbar()
        self.canvas.update()

    def _on_scroll(self, val: int) -> None:
        self.canvas.view_start_ms = val
        self.canvas.update()

    def _clear_points(self) -> None:
        if not self.canvas.manual_points:
            return
        before = deepcopy(self.canvas.manual_points)
        self.canvas.manual_points = []
        self.canvas.manualPointsEdited.emit("清空局部人工音量点", before, [])
        self.canvas.update()

    def sync_scrollbar(self) -> None:
        vis = self.canvas._visible_duration_ms()
        max_start = max(0, self.canvas.duration_ms - vis)
        self.hscroll.blockSignals(True)
        self.hscroll.setRange(0, max_start)
        self.hscroll.setPageStep(vis)
        self.hscroll.setValue(min(max_start, self.canvas.view_start_ms))
        self.hscroll.blockSignals(False)
