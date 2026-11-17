"""Shared themed & animated UI controls (DWM-rounded ComboBox & frosted sliding SegmentTrack)."""

from __future__ import annotations

import ctypes

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPropertyAnimation,
    QRectF,
    Qt,
    QVariantAnimation,
)
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QListView,
    QPushButton,
    QWidget,
)

from drama_studio.gui.theme import get_palette


def _hex_to_colorref(hex_color: str) -> ctypes.c_uint32:
    c = QColor(hex_color)
    return ctypes.c_uint32((c.blue() << 16) | (c.green() << 8) | c.red())


def apply_win11_dwm_chrome(
    widget: QWidget,
    *,
    border_hex: str,
    is_dark: bool,
    round_corners: bool = True,
    caption_hex: str | None = None,
    text_hex: str | None = None,
) -> bool:
    """Apply Windows 11 DWM native titlebar colors, rounded corners, and border tint."""
    app = QApplication.instance()
    if app is not None and app.platformName() == "offscreen":
        return False
    try:
        hwnd = int(widget.winId())
        if hwnd <= 0:
            return False
        dwmapi = ctypes.windll.dwmapi
        dark_val = ctypes.c_int(1 if is_dark else 0)
        dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark_val), ctypes.sizeof(dark_val))
        if caption_hex:
            cap_ref = _hex_to_colorref(caption_hex)
            dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(cap_ref), ctypes.sizeof(cap_ref))
        if text_hex:
            txt_ref = _hex_to_colorref(text_hex)
            dwmapi.DwmSetWindowAttribute(hwnd, 36, ctypes.byref(txt_ref), ctypes.sizeof(txt_ref))
        if round_corners:
            pref = ctypes.c_int(2)  # DWMWCP_ROUND
            r_corner = dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(pref), ctypes.sizeof(pref))
            colorref = _hex_to_colorref(border_hex)
            dwmapi.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(colorref), ctypes.sizeof(colorref))
            return r_corner == 0
        return True
    except Exception:
        return False


class ThemedComboBox(QComboBox):
    """QComboBox with Windows 11 DWM native rounded popup (zero black corners) and smooth animation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._popup_anim: QPropertyAnimation | None = None
        self._pos_anim: QPropertyAnimation | None = None

        list_view = QListView(self)
        list_view.setUniformItemSizes(True)
        list_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        list_view.setMouseTracking(True)
        list_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        list_view.setSpacing(2)
        self.setView(list_view)

    def _current_theme_mode(self) -> str:
        app = QApplication.instance()
        if app is not None:
            prop = app.property("theme_mode")
            if isinstance(prop, str) and prop:
                return prop
        return "light"

    def showPopup(self) -> None:
        super().showPopup()
        view = self.view()
        if view is None:
            return
        popup = view.window()
        if popup is None or popup is self.window():
            return

        pal = get_palette(self._current_theme_mode())

        flags = popup.windowFlags()
        if not (flags & Qt.WindowType.FramelessWindowHint):
            popup.setWindowFlags(flags | Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
            popup.show()

        # Paint opaque 100% solid surface edge-to-edge so Win11 DWM rounds corners with zero black pixels
        dwm_ok = apply_win11_dwm_chrome(
            popup,
            border_hex=pal.line,
            is_dark=pal.is_dark,
            round_corners=True,
        )
        border_css = "border: none;" if dwm_ok else f"border: 1px solid {pal.line};"

        popup.setMinimumWidth(max(self.width(), popup.minimumWidth()))
        popup.setStyleSheet(
            f"QFrame {{ background-color: {pal.surface}; {border_css} border-radius: 0px; }}"
        )
        view.setStyleSheet(
            f"QListView {{ background-color: {pal.surface}; color: {pal.text}; "
            f"border: none; border-radius: 0px; padding: 5px; outline: none; }}"
            f"QListView::item {{ min-height: 28px; padding: 4px 10px; border-radius: 6px; "
            f"color: {pal.text}; background: transparent; }}"
            f"QListView::item:hover {{ background-color: {pal.selection}; color: {pal.text}; }}"
            f"QListView::item:selected {{ background-color: {pal.selection}; "
            f"color: {pal.accent}; font-weight: 700; }}"
        )

        # Smooth fade + gentle 6px slide-in animation (parented to self for safe teardown)
        if self._popup_anim is not None:
            self._popup_anim.stop()
        if self._pos_anim is not None:
            self._pos_anim.stop()

        popup.setWindowOpacity(0.0)
        anim = QPropertyAnimation(popup, b"windowOpacity", self)
        anim.setDuration(140)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._popup_anim = anim

        target_pos = popup.pos()
        start_pos = QPoint(target_pos.x(), target_pos.y() - 5)
        popup.move(start_pos)
        pos_anim = QPropertyAnimation(popup, b"pos", self)
        pos_anim.setDuration(145)
        pos_anim.setStartValue(start_pos)
        pos_anim.setEndValue(target_pos)
        pos_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        pos_anim.start()
        self._pos_anim = pos_anim


class AnimatedSegmentTrack(QFrame):
    """Frosted segmented pill track with a smooth sliding accent capsule."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("SegmentTrack")
        self._buttons: list[QPushButton] = []
        self._pill_rect = QRectF()
        self._anim: QVariantAnimation | None = None
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(3, 3, 3, 3)
        self._lay.setSpacing(4)

    def add_segment(self, btn: QPushButton, stretch: int = 0) -> None:
        btn.setCheckable(True)
        btn.setProperty("segment", "true")
        self._buttons.append(btn)
        self._lay.addWidget(btn, stretch)
        btn.toggled.connect(self._on_btn_toggled)
        if btn.isChecked():
            self._sync_pill(animate=False)

    def _active_button(self) -> QPushButton | None:
        for b in self._buttons:
            if b.isChecked():
                return b
        return None

    def _on_btn_toggled(self, checked: bool) -> None:
        if checked:
            self._sync_pill(animate=True)

    def _sync_pill(self, *, animate: bool) -> None:
        active = self._active_button()
        if active is None or active.width() <= 0:
            return
        target = QRectF(active.geometry())
        if not animate or self._pill_rect.isNull() or not self.isVisible():
            self._pill_rect = target
            self.update()
            return

        if self._anim is not None:
            self._anim.stop()
        anim = QVariantAnimation(self)
        anim.setDuration(170)
        anim.setStartValue(QRectF(self._pill_rect))
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(self._on_anim_val)
        anim.start()
        self._anim = anim

    def _on_anim_val(self, val: object) -> None:
        if isinstance(val, QRectF):
            self._pill_rect = val
            self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_pill(animate=False)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_pill(animate=False)

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange):
            self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._pill_rect.isNull():
            active = self._active_button()
            if active is not None and active.width() > 0:
                self._pill_rect = QRectF(active.geometry())
        if self._pill_rect.width() <= 2 or self._pill_rect.height() <= 2:
            return

        app = QApplication.instance()
        mode = app.property("theme_mode") if app is not None else "light"
        pal = get_palette(mode if isinstance(mode, str) else "light")

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = QPainterPath()
        path.addRoundedRect(self._pill_rect, 7.0, 7.0)

        grad = QLinearGradient(self._pill_rect.topLeft(), self._pill_rect.bottomRight())
        grad.setColorAt(0.0, QColor(pal.primary_start))
        grad.setColorAt(1.0, QColor(pal.primary_end))
        p.fillPath(path, grad)

        # Subtle frosted inner top gloss on active capsule
        p.setPen(QPen(QColor(255, 255, 255, 72 if not pal.is_dark else 45), 1.0))
        p.drawRoundedRect(self._pill_rect.adjusted(0.5, 0.5, -0.5, -0.5), 6.5, 6.5)
