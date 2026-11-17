"""Reusable frosted Glassmorphism UI card, header bar, and KPI primitives."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from drama_studio.gui.theme import get_palette


def paint_frosted_panel(widget: QWidget, *, radius: float = 12.0, subtle: bool = False) -> None:
    """Paint a multi-layer frosted glass surface with top specular rim highlight."""
    app = QApplication.instance()
    mode = app.property("theme_mode") if app is not None else "light"
    pal = get_palette(mode if isinstance(mode, str) else "light")

    p = QPainter(widget)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    w, h = float(widget.width()), float(widget.height())
    if w <= 2 or h <= 2:
        return

    rect = QRectF(0.75, 0.75, w - 1.5, h - 1.5)
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)

    # Translucent frosted acrylic gradient fill
    bg = QLinearGradient(0, 0, 0, h)
    if pal.is_dark:
        if subtle:
            bg.setColorAt(0.0, QColor(48, 53, 66, 178))
            bg.setColorAt(1.0, QColor(33, 36, 46, 162))
        else:
            bg.setColorAt(0.0, QColor(42, 46, 56, 206))
            bg.setColorAt(1.0, QColor(29, 32, 40, 194))
    elif pal.mode == "pink":
        if subtle:
            bg.setColorAt(0.0, QColor(255, 252, 254, 186))
            bg.setColorAt(1.0, QColor(253, 240, 247, 160))
        else:
            bg.setColorAt(0.0, QColor(255, 253, 254, 214))
            bg.setColorAt(1.0, QColor(253, 245, 250, 196))
    else:
        if subtle:
            bg.setColorAt(0.0, QColor(255, 255, 255, 184))
            bg.setColorAt(1.0, QColor(244, 247, 254, 158))
        else:
            bg.setColorAt(0.0, QColor(255, 255, 255, 214))
            bg.setColorAt(1.0, QColor(247, 250, 255, 194))

    p.fillPath(path, bg)

    # Frosted specular top-to-bottom glass edge rim
    rim = QLinearGradient(0, 0, 0, h)
    if pal.is_dark:
        rim.setColorAt(0.0, QColor(255, 255, 255, 42))
        rim.setColorAt(0.45, QColor(135, 157, 245, 28))
        rim.setColorAt(1.0, QColor(255, 255, 255, 14))
    else:
        line_c = QColor(pal.line)
        line_c.setAlpha(205)
        rim.setColorAt(0.0, QColor(255, 255, 255, 245))
        rim.setColorAt(0.35, line_c)
        rim.setColorAt(1.0, line_c)

    p.setPen(QPen(rim, 1.0))
    p.drawPath(path)

    # Crisp inner top catchlight line for elevated acrylic feel
    top_gloss = QColor(255, 255, 255, 28 if pal.is_dark else 165)
    p.setPen(QPen(top_gloss, 1.0))
    p.drawLine(
        int(radius + 2),
        2,
        max(int(radius + 2), int(w - radius - 2)),
        2,
    )


class FrostedHeaderBar(QFrame):
    """Top frosted glass toolbar with subtle specular highlight."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("HeaderBar")

    def paintEvent(self, event) -> None:
        paint_frosted_panel(self, radius=13.0, subtle=False)


class GlassCard(QFrame):
    """Frosted translucent studio card replacing flat opaque panels."""

    def __init__(
        self,
        title: str = "",
        badge: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("GlassCard")
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(11, 8, 11, 9)
        self.root_layout.setSpacing(6)

        if title:
            hdr = QHBoxLayout()
            hdr.setContentsMargins(0, 0, 0, 2)
            dot = QLabel("●")
            dot.setObjectName("CardAccentDot")
            lbl = QLabel(title)
            lbl.setObjectName("SectionTitle")
            hdr.addWidget(dot)
            hdr.addWidget(lbl)
            hdr.addStretch()
            if badge:
                b_lbl = QLabel(badge)
                b_lbl.setObjectName("PillBadge")
                hdr.addWidget(b_lbl)
                self.badge_label: QLabel | None = b_lbl
            else:
                self.badge_label = None
            self.root_layout.addLayout(hdr)
        else:
            self.badge_label = None

    def body(self) -> QVBoxLayout:
        return self.root_layout

    def paintEvent(self, event) -> None:
        paint_frosted_panel(self, radius=12.0, subtle=False)


class KpiCard(QFrame):
    """Compact translucent frosted metric card for pre/post LUFS, LRA, and True Peak."""

    def __init__(
        self,
        title: str,
        initial_val: str = "-",
        sub_text: str = "",
        accent: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("KpiCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 5, 10, 5)
        lay.setSpacing(1)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("KpiTitle")
        self.val_lbl = QLabel(initial_val)
        self.val_lbl.setObjectName("KpiAccent" if accent else "KpiValue")
        self.sub_lbl = QLabel(sub_text)
        self.sub_lbl.setObjectName("MutedLabel")
        self.sub_lbl.setStyleSheet("font-size: 11px;")

        lay.addWidget(self.title_lbl)
        lay.addWidget(self.val_lbl)
        if sub_text or True:
            lay.addWidget(self.sub_lbl)

    def set_metric(self, value: str, sub: str = "") -> None:
        self.val_lbl.setText(value)
        self.sub_lbl.setText(sub)

    def paintEvent(self, event) -> None:
        paint_frosted_panel(self, radius=10.0, subtle=True)
