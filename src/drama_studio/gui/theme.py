"""Unified multi-theme palette and QSS generator (Light / Dark / Sakura Pink).

Inspired by the palette tokens and visual hierarchy of `UI_SPEC.md`:
- ``light``: 纯净白 (#F4F5F8 canvas, #FFFFFF cards, #5E76D6 accent) [default]
- ``dark``: 曜石黑 (#17191E canvas, #22252C cards, #879DF5 accent)
- ``pink``: 浅樱粉 (#F6EDF2 canvas, #FFFFFF cards, #FC6C85 accent)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class ThemePalette:
    """Semantic palette tokens driving QSS and custom QPainter widgets."""

    mode: str
    display_name: str
    is_dark: bool
    background: str
    surface: str
    surface_alt: str
    header_bg: str
    text: str
    muted: str
    line: str
    accent: str
    primary_start: str
    primary_end: str
    accent_hover: str
    accent_pressed: str
    accent_border: str
    selection: str
    selection_text: str
    track_bg: str
    ok_text: str
    ok_fg: str
    ok_start: str
    ok_end: str
    ok_hover: str
    ok_border: str
    warn_bg: str
    warn_text: str
    warn_border: str
    err_text: str
    danger_bg: str
    danger_border: str
    banner_bg: str
    banner_text: str
    banner_border: str
    badge_bg: str
    badge_text: str
    badge_border: str
    scroll_idle: str
    scroll_hover: str
    splitter: str
    splitter_hover: str
    glass_surface: str
    glass_surface_alt: str
    glass_track: str
    glass_border: str


THEMES: dict[str, ThemePalette] = {
    "light": ThemePalette(
        mode="light",
        display_name="纯净白 (浅色)",
        is_dark=False,
        background="#F4F5F8",
        surface="#FFFFFF",
        surface_alt="#F8F9FC",
        header_bg="#EFF1F6",
        text="#252A35",
        muted="#778091",
        line="#DDE1E9",
        accent="#5E76D6",
        primary_start="#5E76D6",
        primary_end="#6D8CF0",
        accent_hover="#4F67C7",
        accent_pressed="#3E54B2",
        accent_border="#4D64C5",
        selection="#E5EBFF",
        selection_text="#252A35",
        track_bg="#EAECEF",
        ok_text="#217294",
        ok_fg="#FFFFFF",
        ok_start="#3E8DB8",
        ok_end="#2D759D",
        ok_hover="#28698E",
        ok_border="#26688C",
        warn_bg="#FEF3C7",
        warn_text="#92400E",
        warn_border="#FCD34D",
        err_text="#D73741",
        danger_bg="#FEF2F2",
        danger_border="#FECACA",
        banner_bg="#E5EBFF",
        banner_text="#252A35",
        banner_border="#C7D4FA",
        badge_bg="#EDF1FE",
        badge_text="#4A62C4",
        badge_border="#CED7F8",
        scroll_idle="rgba(119, 128, 145, 0.30)",
        scroll_hover="rgba(94, 118, 214, 0.65)",
        splitter="rgba(221, 225, 233, 0.90)",
        splitter_hover="rgba(94, 118, 214, 0.60)",
        glass_surface="rgba(255, 255, 255, 0.72)",
        glass_surface_alt="rgba(247, 250, 255, 0.62)",
        glass_track="rgba(232, 236, 245, 0.72)",
        glass_border="rgba(209, 218, 235, 0.85)",
    ),
    "dark": ThemePalette(
        mode="dark",
        display_name="曜石黑 (深色)",
        is_dark=True,
        background="#17191E",
        surface="#22252C",
        surface_alt="#1C1F26",
        header_bg="#1B1E25",
        text="#EEEEF3",
        muted="#A3A9B6",
        line="#383D48",
        accent="#879DF5",
        primary_start="#5C74DB",
        primary_end="#7B92F2",
        accent_hover="#7087EA",
        accent_pressed="#4E65CA",
        accent_border="#6F86EB",
        selection="#394362",
        selection_text="#EEEEF3",
        track_bg="#1C1F26",
        ok_text="#5EEAD4",
        ok_fg="#FFFFFF",
        ok_start="#2D84A8",
        ok_end="#216A89",
        ok_hover="#26779A",
        ok_border="#3A96BC",
        warn_bg="rgba(217, 119, 6, 0.16)",
        warn_text="#FDE68A",
        warn_border="rgba(245, 158, 11, 0.38)",
        err_text="#F87171",
        danger_bg="rgba(239, 68, 68, 0.16)",
        danger_border="rgba(248, 113, 113, 0.38)",
        banner_bg="#292F3F",
        banner_text="#EEEEF3",
        banner_border="#48536F",
        badge_bg="rgba(135, 157, 245, 0.16)",
        badge_text="#C4D1FC",
        badge_border="rgba(135, 157, 245, 0.38)",
        scroll_idle="rgba(163, 169, 182, 0.32)",
        scroll_hover="rgba(135, 157, 245, 0.65)",
        splitter="rgba(56, 61, 72, 0.90)",
        splitter_hover="rgba(135, 157, 245, 0.65)",
        glass_surface="rgba(34, 37, 45, 0.72)",
        glass_surface_alt="rgba(26, 29, 36, 0.62)",
        glass_track="rgba(24, 27, 34, 0.76)",
        glass_border="rgba(255, 255, 255, 0.12)",
    ),
    "pink": ThemePalette(
        mode="pink",
        display_name="浅樱粉 (粉色)",
        is_dark=False,
        background="#F6EDF2",
        surface="#FFFFFF",
        surface_alt="#FCF7FA",
        header_bg="#F8E9F1",
        text="#3E222D",
        muted="#916575",
        line="#E5D3DB",
        accent="#FC6C85",
        primary_start="#FC6C85",
        primary_end="#EB516E",
        accent_hover="#EE5672",
        accent_pressed="#D83E5B",
        accent_border="#E5506C",
        selection="#FFE6EE",
        selection_text="#3E222D",
        track_bg="#F0E2E9",
        ok_text="#C4385D",
        ok_fg="#B63056",
        ok_start="#FFE9F1",
        ok_end="#FFD5E5",
        ok_hover="#FFC7DC",
        ok_border="#EE9EB8",
        warn_bg="#FEF3C7",
        warn_text="#92400E",
        warn_border="#FCD34D",
        err_text="#D73741",
        danger_bg="#FEF2F2",
        danger_border="#FECACA",
        banner_bg="#FFE6EE",
        banner_text="#3E222D",
        banner_border="#F4C2D2",
        badge_bg="#FFE6EE",
        badge_text="#C73857",
        badge_border="#F4C2D2",
        scroll_idle="rgba(145, 101, 117, 0.28)",
        scroll_hover="rgba(252, 108, 133, 0.65)",
        splitter="rgba(229, 211, 219, 0.95)",
        splitter_hover="rgba(252, 108, 133, 0.60)",
        glass_surface="rgba(255, 252, 254, 0.74)",
        glass_surface_alt="rgba(253, 245, 249, 0.64)",
        glass_track="rgba(241, 224, 233, 0.74)",
        glass_border="rgba(229, 206, 217, 0.85)",
    ),
}

THEME_ORDER: tuple[str, ...] = ("light", "dark", "pink")


def normalize_theme_mode(mode: str | None) -> str:
    if mode in THEMES:
        return str(mode)
    return "light"


def get_palette(mode: str | None = "light") -> ThemePalette:
    return THEMES[normalize_theme_mode(mode)]


def resolve_asset_path(rel_path: str) -> Path | None:
    """Resolve a bundled asset path across PyInstaller _MEIPASS and source checkouts."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        cand = Path(getattr(sys, "_MEIPASS")) / "assets" / rel_path
        if cand.exists():
            return cand
    cand = Path(__file__).resolve().parents[3] / "assets" / rel_path
    return cand if cand.exists() else None


def _resolve_icons_dir() -> Path:
    resolved = resolve_asset_path("icons")
    if resolved is not None:
        return resolved
    return Path(__file__).resolve().parents[3] / "assets" / "icons"


def build_studio_qss(theme_mode: str = "light") -> str:
    """Render unified studio QSS stylesheet for 'light', 'dark', or 'pink'."""
    pal = get_palette(theme_mode)
    icons = _resolve_icons_dir()
    tokens = {
        **asdict(pal),
        "check_icon": (icons / "check.svg").as_posix(),
        "down_icon": (icons / "arrow_down.svg").as_posix(),
        "up_icon": (icons / "arrow_up.svg").as_posix(),
    }
    return STUDIO_QSS_TEMPLATE % tokens


STUDIO_QSS_TEMPLATE = """
QMainWindow, QDialog {
    background-color: %(background)s;
    color: %(text)s;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}

QWidget {
    color: %(text)s;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
}

QWidget#TransparentContainer {
    background: transparent;
}

QFrame#HeaderBar, QFrame#GlassCard, QFrame#KpiCard {
    background: transparent;
    border: none;
}

QFrame#Card {
    background-color: %(glass_surface)s;
    border: 1px solid %(glass_border)s;
    border-radius: 12px;
}

QLabel#BrandTitle {
    font-size: 15px;
    font-weight: 800;
    color: %(text)s;
    letter-spacing: 0.4px;
}

QLabel#BrandBadge {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 %(primary_start)s, stop:1 %(primary_end)s);
    color: #FFFFFF;
    border-radius: 7px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 700;
}

QLabel#CardAccentDot {
    color: %(accent)s;
    font-size: 9px;
}

QLabel#KpiTitle {
    color: %(muted)s;
    font-size: 11px;
    font-weight: 600;
}

QLabel#KpiValue {
    color: %(text)s;
    font-size: 16px;
    font-weight: 700;
}

QLabel#KpiAccent {
    color: %(accent)s;
    font-size: 16px;
    font-weight: 800;
}

QLabel#SectionTitle {
    font-size: 13px;
    font-weight: 700;
    color: %(text)s;
}

QLabel#MutedLabel {
    color: %(muted)s;
    font-size: 12px;
}

QLabel#MonoTimeLabel {
    font-family: Consolas, monospace;
    color: %(text)s;
    font-size: 12px;
    font-weight: 600;
}

QLabel#MetricsLabel {
    color: %(accent)s;
    font-size: 11px;
    font-weight: 600;
}

QLabel#BigValueLabel {
    font-size: 22px;
    font-weight: 800;
    color: %(accent)s;
}

QLabel#PillBadge {
    background-color: %(badge_bg)s;
    color: %(badge_text)s;
    border: 1px solid %(badge_border)s;
    border-radius: 10px;
    padding: 2px 9px;
    font-size: 11px;
    font-weight: 600;
}

QLabel#PreviewCueList {
    background-color: %(surface_alt)s;
    border: 1px solid %(line)s;
    border-radius: 8px;
    padding: 7px 9px;
    color: %(text)s;
    font-size: 12px;
}

QLabel#WarningBanner {
    color: %(warn_text)s;
    background-color: %(warn_bg)s;
    border: 1px solid %(warn_border)s;
    border-radius: 8px;
    padding: 7px 9px;
    font-size: 12px;
}

QLabel#HardwareBanner {
    background-color: %(banner_bg)s;
    border: 1px solid %(banner_border)s;
    border-radius: 10px;
    padding: 10px 12px;
    color: %(banner_text)s;
}

QPushButton {
    background-color: %(glass_surface)s;
    color: %(text)s;
    border: 1px solid %(glass_border)s;
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: %(selection)s;
    border-color: %(accent)s;
    color: %(text)s;
}

QPushButton:pressed {
    background-color: %(track_bg)s;
}

QPushButton:disabled {
    background-color: %(glass_surface_alt)s;
    color: %(muted)s;
    border-color: %(line)s;
}

QPushButton#PrimaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 %(primary_start)s, stop:1 %(primary_end)s);
    color: #FFFFFF;
    border: 1px solid %(accent_border)s;
    border-radius: 8px;
    font-weight: 700;
    padding: 7px 16px;
}

QPushButton#PrimaryButton:hover {
    background: %(accent_hover)s;
    color: #FFFFFF;
}

QPushButton#PrimaryButton:pressed {
    background: %(accent_pressed)s;
    color: #FFFFFF;
}

QPushButton#PrimaryButton:disabled {
    background: %(selection)s;
    color: %(muted)s;
    border-color: %(line)s;
}

QPushButton#SuccessButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 %(ok_start)s, stop:1 %(ok_end)s);
    color: %(ok_fg)s;
    border: 1px solid %(ok_border)s;
    border-radius: 8px;
    font-weight: 700;
    padding: 6px 14px;
}

QPushButton#SuccessButton:hover {
    background: %(ok_hover)s;
    color: %(ok_fg)s;
}

QPushButton#DangerButton {
    background-color: %(danger_bg)s;
    color: %(err_text)s;
    border: 1px solid %(danger_border)s;
    border-radius: 8px;
    font-weight: 600;
}

QPushButton#DangerButton:hover {
    background-color: %(err_text)s;
    color: #FFFFFF;
    border-color: %(err_text)s;
}

QPushButton#DangerButton:disabled {
    background-color: %(surface_alt)s;
    color: %(muted)s;
    border-color: %(line)s;
}

QFrame#SegmentTrack {
    background-color: %(glass_track)s;
    border: 1px solid %(glass_border)s;
    border-radius: 9px;
}

QPushButton[segment="true"] {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 5px 12px;
    font-weight: 600;
    color: %(muted)s;
}

QPushButton[segment="true"]:!checked:hover {
    background-color: %(surface)s;
    color: %(text)s;
}

QPushButton[segment="true"]:checked {
    background-color: transparent;
    border: 1px solid transparent;
    color: #FFFFFF;
}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: %(glass_surface)s;
    color: %(text)s;
    border: 1px solid %(glass_border)s;
    border-radius: 8px;
    padding: 2px 8px;
    min-height: 22px;
}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: %(accent)s;
}

QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border: 1.5px solid %(accent)s;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox::down-arrow {
    image: url("%(down_icon)s");
    width: 10px;
    height: 10px;
}

QComboBox QAbstractItemView {
    background-color: %(surface)s;
    color: %(text)s;
    border: 1px solid %(line)s;
    border-radius: 10px;
    padding: 4px;
    selection-background-color: %(selection)s;
    selection-color: %(accent)s;
    outline: none;
}

QComboBox QAbstractItemView::item {
    min-height: 28px;
    padding: 4px 10px;
    border-radius: 6px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: %(selection)s;
    color: %(text)s;
}

QComboBox QAbstractItemView::item:selected {
    background-color: %(selection)s;
    color: %(accent)s;
    font-weight: 700;
}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: %(surface_alt)s;
    border: none;
    width: 18px;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    border-top-right-radius: 7px;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    border-bottom-right-radius: 7px;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url("%(up_icon)s");
    width: 9px;
    height: 9px;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url("%(down_icon)s");
    width: 9px;
    height: 9px;
}

QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: %(selection)s;
}

QCheckBox {
    spacing: 8px;
    color: %(text)s;
    font-weight: 500;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1.5px solid %(line)s;
    background-color: %(surface)s;
}

QCheckBox::indicator:hover {
    border-color: %(accent)s;
}

QCheckBox::indicator:checked {
    background-color: %(accent)s;
    border-color: %(accent_border)s;
    image: url("%(check_icon)s");
}

QSlider:horizontal {
    min-height: 20px;
}

QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: %(track_bg)s;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 %(primary_start)s, stop:1 %(primary_end)s);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: %(surface)s;
    border: 2px solid %(accent)s;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    border: 2px solid %(accent_pressed)s;
    background: %(selection)s;
}

QTableWidget, QListWidget, QPlainTextEdit {
    background-color: %(glass_surface)s;
    alternate-background-color: %(glass_surface_alt)s;
    color: %(text)s;
    border: 1px solid %(glass_border)s;
    border-radius: 9px;
    gridline-color: transparent;
    selection-background-color: %(selection)s;
    selection-color: %(selection_text)s;
    outline: none;
}

QTableWidget::item {
    padding: 4px 8px;
}

QHeaderView::section {
    background-color: %(glass_surface_alt)s;
    color: %(muted)s;
    border: none;
    border-bottom: 1px solid %(line)s;
    padding: 6px 8px;
    font-weight: 700;
    font-size: 12px;
}

QTabWidget::pane {
    border: 1px solid %(glass_border)s;
    background-color: %(glass_surface)s;
    border-radius: 10px;
}

QTabBar::tab {
    background-color: %(glass_track)s;
    color: %(muted)s;
    border: 1px solid %(glass_border)s;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 7px 16px;
    margin-right: 4px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background-color: %(glass_surface)s;
    border-color: %(glass_border)s;
    color: %(accent)s;
    font-weight: 700;
}

QStatusBar {
    background-color: %(glass_surface)s;
    border-top: 1px solid %(glass_border)s;
    color: %(muted)s;
}

QProgressBar {
    background-color: %(track_bg)s;
    border: 1px solid %(line)s;
    border-radius: 6px;
    text-align: center;
    color: %(text)s;
    font-size: 11px;
    font-weight: 700;
    min-height: 18px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 %(primary_start)s, stop:1 %(primary_end)s);
    border-radius: 5px;
}

QScrollArea {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    background: transparent;
    width: 6px;
    margin: 2px 1px;
}

QScrollBar::handle:vertical {
    background: %(scroll_idle)s;
    border-radius: 3px;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover {
    background: %(scroll_hover)s;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}

QScrollBar:horizontal {
    background: transparent;
    height: 6px;
    margin: 1px 2px;
}

QScrollBar::handle:horizontal {
    background: %(scroll_idle)s;
    border-radius: 3px;
    min-width: 28px;
}

QScrollBar::handle:horizontal:hover {
    background: %(scroll_hover)s;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
    width: 0px;
}

QSplitter::handle {
    background-color: %(splitter)s;
    border-radius: 2px;
}

QSplitter::handle:hover {
    background-color: %(splitter_hover)s;
}

QToolTip {
    background-color: %(surface)s;
    color: %(text)s;
    border: 1px solid %(accent)s;
    border-radius: 6px;
    padding: 6px;
}
"""

STUDIO_QSS = build_studio_qss("light")
