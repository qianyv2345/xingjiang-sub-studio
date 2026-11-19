"""Customizable subtitle template panel with aurora glassmorphism, slider, 1/2-line toggle, and live preview."""

from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
        QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from drama_studio.config import ConfigStore
from drama_studio.gui.theme import normalize_theme_mode
from drama_studio.gui.widgets.glass_card import GlassCard
from drama_studio.gui.widgets.themed_controls import AnimatedSegmentTrack, ThemedComboBox
from drama_studio.models import SubtitleTemplate
from drama_studio.nlp.punctuation import count_english_words
from drama_studio.nlp.segmenter import (
    build_unit_tokens,
    measure_text_width,
    segment_tokens_dp,
)

_DEFAULT_ZH_SAMPLE = "顾寒川，你别以为这样就能让我低头，这三千万我会一分不少还给你！"
_DEFAULT_EN_SAMPLE = "I don't know what you're talking about, Dr. Smith!"


class LiveSubtitlePreviewBox(GlassCard):
    """Simulates 9:16 or 16:9 safe frame and renders real-time DP segmentation output."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("实时字幕排版预览", parent=parent)
        self.setMinimumWidth(120)
        self.body().setSpacing(4)
        self.layout().setContentsMargins(12, 9, 12, 9)
        self.metrics_lbl = QLabel("")
        self.metrics_lbl.setObjectName("MetricsLabel")
        self.metrics_lbl.setWordWrap(True)
        self.metrics_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.body().addWidget(self.metrics_lbl)

        self.sample_input = QLineEdit(
            "顾寒川，你别以为这样就能让我低头，这三千万我会一分不少还给你！"
        )
        self.sample_input.setCursorPosition(0)
        self.sample_input.setMinimumWidth(100)
        self.sample_input.setPlaceholderText("输入或联动当前台词预览断句...")
        self.body().addWidget(self.sample_input)

        screen_row = QHBoxLayout()
        screen_row.addStretch()
        self.stage_frame = QFrame()
        stage_lay = QVBoxLayout(self.stage_frame)
        stage_lay.setContentsMargins(8, 6, 8, 8)
        stage_lay.addStretch()

        self.sub_label = QLabel("")
        self.sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub_label.setWordWrap(False)
        stage_lay.addWidget(self.sub_label, 0, Qt.AlignmentFlag.AlignHCenter)
        screen_row.addWidget(self.stage_frame)
        screen_row.addStretch()
        self.body().addLayout(screen_row)

        self.all_cues_lbl = QLabel("")
        self.all_cues_lbl.setObjectName("PreviewCueList")
        self.all_cues_lbl.setWordWrap(True)
        self.all_cues_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.body().addWidget(self.all_cues_lbl)
        self.set_theme("light")

    def set_theme(self, mode: str) -> None:
        m = normalize_theme_mode(mode)
        if m == "pink":
            self.stage_frame.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
                "stop:0 #F9E3EE, stop:0.55 #F3D7E8, stop:1 #ECCCE4);"
                "border: 1px solid rgba(226, 158, 192, 0.78); border-radius: 10px;"
            )
            self.sub_label.setStyleSheet(
                "color: #FFFFFF; background-color: rgba(74, 38, 56, 188);"
                "border: 1px solid rgba(255, 255, 255, 0.28); border-radius: 6px; "
                "padding: 5px 12px; font-weight: 700; font-size: 14px;"
            )
        elif m == "dark":
            self.stage_frame.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
                "stop:0 #141926, stop:0.55 #1B2133, stop:1 #22263C);"
                "border: 1px solid rgba(135, 157, 245, 0.30); border-radius: 10px;"
            )
            self.sub_label.setStyleSheet(
                "color: #FFFFFF; background-color: rgba(11, 14, 22, 192);"
                "border: 1px solid rgba(255, 255, 255, 0.16); border-radius: 6px; "
                "padding: 5px 12px; font-weight: 700; font-size: 14px;"
            )
        else:
            self.stage_frame.setStyleSheet(
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
                "stop:0 #E3ECFB, stop:0.55 #D7E4F8, stop:1 #CAD9F4);"
                "border: 1px solid rgba(154, 178, 224, 0.72); border-radius: 10px;"
            )
            self.sub_label.setStyleSheet(
                "color: #FFFFFF; background-color: rgba(38, 46, 70, 188);"
                "border: 1px solid rgba(255, 255, 255, 0.24); border-radius: 6px; "
                "padding: 5px 12px; font-weight: 700; font-size: 14px;"
            )

    def update_preview(self, template: SubtitleTemplate, custom_text: str | None = None) -> None:
        if custom_text is not None and custom_text.strip():
            self.sample_input.blockSignals(True)
            self.sample_input.setText(custom_text.replace("\n", " "))
            self.sample_input.setCursorPosition(0)
            self.sample_input.blockSignals(False)
        else:
            cur_sample = self.sample_input.text().strip()
            if template.language == "en" and cur_sample == _DEFAULT_ZH_SAMPLE:
                self.sample_input.blockSignals(True)
                self.sample_input.setText(_DEFAULT_EN_SAMPLE)
                self.sample_input.setCursorPosition(0)
                self.sample_input.blockSignals(False)
            elif template.language == "zh" and cur_sample == _DEFAULT_EN_SAMPLE:
                self.sample_input.blockSignals(True)
                self.sample_input.setText(_DEFAULT_ZH_SAMPLE)
                self.sample_input.setCursorPosition(0)
                self.sample_input.blockSignals(False)

        text = self.sample_input.text().strip() or (
            _DEFAULT_EN_SAMPLE if template.language == "en" else "这件事情没有你想的那么简单"
        )
        if template.aspect_preset == "9:16":
            self.stage_frame.setFixedSize(108 if template.language == "en" else 102, 46)
        else:
            self.stage_frame.setFixedSize(236, 46)

        tokens = build_unit_tokens(
            text,
            [],
            fallback_start_ms=0,
            fallback_end_ms=max(1200, len(text) * 140),
            nlp_backend="jieba",
            language=template.language,
        )
        if template.smart_semantic_split and tokens:
            segs = segment_tokens_dp(tokens, template)
            preview_lines = [s[2] for s in segs]
        else:
            preview_lines = [text]

        first_cue = preview_lines[0] if preview_lines else ""
        self.sub_label.setText(first_cue)

        eff_wm = "strict" if template.language == "en" else template.width_mode
        max_w = max((measure_text_width(p, eff_wm) for p in preview_lines), default=0.0)
        unit_lbl = "字符" if template.language == "en" else "字"
        extra_en = ""
        if template.language == "en" and "\n" in first_cue:
            l1, l2 = first_cue.splitlines()[:2]
            extra_en = f" · L1:{count_english_words(l1)}词/L2:{count_english_words(l2)}词"
        self.metrics_lbl.setText(
            f"智能切分 {len(preview_lines)} 条 · 首条宽 {max_w:.1f}/{template.max_chars_per_line} {unit_lbl} · "
            f"{'2行模式' if template.max_lines >= 2 else '1行模式'}{extra_en}"
        )
        numbered = "  |  ".join(f"[{i+1}] {p.replace(chr(10), ' ↵ ')}" for i, p in enumerate(preview_lines))
        self.all_cues_lbl.setText(f"完整切分：{numbered}")


class TemplateSettingsPanel(QWidget):
    """Right-side glassmorphism subtitle template configurator with zero clipping."""

    templateChanged = Signal(object)
    batchModeChanged = Signal(bool)

    def __init__(self, store: ConfigStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TransparentContainer")
        self.store = store
        self._template: SubtitleTemplate = deepcopy(store.get_active_project().active_template)
        self._video_fps: float = 25.0
        self._loading = False
        self._build_ui()
        self.set_theme(store.preferences.get("theme_mode", "light"))
        self.reload_templates_combo()
        self.set_template(self._template, emit_signal=False)

    def set_theme(self, mode: str) -> None:
        if hasattr(self, "preview_box"):
            self.preview_box.set_theme(mode)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        # ---- Tab 1: Basic Presets & Core Layout Controls ----
        basic_scroll = QScrollArea()
        basic_scroll.setWidgetResizable(True)
        basic_scroll.setFrameShape(QFrame.Shape.NoFrame)
        basic_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        basic_container = QWidget()
        basic_container.setObjectName("TransparentContainer")
        layout = QVBoxLayout(basic_container)
        layout.setContentsMargins(4, 6, 6, 4)
        layout.setSpacing(6)
        basic_scroll.setWidget(basic_container)
        self.tabs.addTab(basic_scroll, "核心排版与预设")

        # --- Card 1: Template Management ---
        tpl_card = GlassCard("字幕模板预设管理")
        tpl_card.layout().setContentsMargins(12, 9, 12, 9)
        tpl_card.body().setSpacing(4)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("模板:"))
        self.tpl_combo = ThemedComboBox()
        self.tpl_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.tpl_combo.setMinimumContentsLength(10)
        self.tpl_combo.currentIndexChanged.connect(self._on_template_selected)
        row1.addWidget(self.tpl_combo, 1)
        tpl_card.body().addLayout(row1)

        btn_grid = QGridLayout()
        btn_grid.setSpacing(4)
        self.btn_save_tpl = QPushButton("保存")
        self.btn_save_as_tpl = QPushButton("另存为")
        self.btn_rename_tpl = QPushButton("重命名")
        self.btn_copy_tpl = QPushButton("复制")
        self.btn_del_tpl = QPushButton("删除")
        self.btn_reset_default = QPushButton("恢复默认")
        self.btn_reset_default.setToolTip("恢复系统内置模板默认参数")
        for b in (
            self.btn_save_tpl,
            self.btn_save_as_tpl,
            self.btn_rename_tpl,
            self.btn_copy_tpl,
            self.btn_del_tpl,
            self.btn_reset_default,
        ):
            b.setMinimumWidth(0)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_grid.addWidget(self.btn_save_tpl, 0, 0)
        btn_grid.addWidget(self.btn_save_as_tpl, 0, 1)
        btn_grid.addWidget(self.btn_rename_tpl, 0, 2)
        btn_grid.addWidget(self.btn_copy_tpl, 1, 0)
        btn_grid.addWidget(self.btn_del_tpl, 1, 1)
        btn_grid.addWidget(self.btn_reset_default, 1, 2)
        tpl_card.body().addLayout(btn_grid)

        self.btn_save_tpl.clicked.connect(self._on_save_template)
        self.btn_save_as_tpl.clicked.connect(self._on_save_as_template)
        self.btn_rename_tpl.clicked.connect(self._on_rename_template)
        self.btn_copy_tpl.clicked.connect(self._on_copy_template)
        self.btn_del_tpl.clicked.connect(self._on_delete_template)
        self.btn_reset_default.clicked.connect(self._on_reset_preset_defaults)

        self.chk_unified_batch = QCheckBox("批量任务统一使用当前模板")
        self.chk_unified_batch.setChecked(self.store.get_active_project().batch_use_unified_template)
        self.chk_unified_batch.toggled.connect(self.batchModeChanged.emit)
        tpl_card.body().addWidget(self.chk_unified_batch)
        layout.addWidget(tpl_card)

        # --- Card 2: Core Layout Controls ---
        core_card = GlassCard("核心排版参数", badge="实时预览")
        core_card.layout().setContentsMargins(12, 9, 12, 9)
        core_card.body().setSpacing(4)

        # Aspect / Language Preset Segmented Track
        preset_track = AnimatedSegmentTrack()
        self.btn_preset_916 = QPushButton("竖屏 9:16")
        self.btn_preset_169 = QPushButton("横屏 16:9")
        self.btn_preset_en = QPushButton("EN 26字")
        for b in (self.btn_preset_916, self.btn_preset_169, self.btn_preset_en):
            b.setMinimumWidth(0)
            b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        self.preset_group.addButton(self.btn_preset_916, 0)
        self.preset_group.addButton(self.btn_preset_169, 1)
        self.preset_group.addButton(self.btn_preset_en, 2)
        preset_track.add_segment(self.btn_preset_916, 1)
        preset_track.add_segment(self.btn_preset_169, 1)
        preset_track.add_segment(self.btn_preset_en, 1)
        core_card.body().addWidget(preset_track)
        self.btn_preset_916.clicked.connect(lambda: self.apply_aspect_preset("9:16"))
        self.btn_preset_169.clicked.connect(lambda: self.apply_aspect_preset("16:9"))
        self.btn_preset_en.clicked.connect(lambda: self.apply_aspect_preset("en"))

        # Language selector & Max chars per line
        chars_hdr = QHBoxLayout()
        chars_hdr.addWidget(QLabel("语言:"))
        self.combo_tpl_lang = ThemedComboBox()
        self.combo_tpl_lang.addItem("中文 (按字/视觉宽)", "zh")
        self.combo_tpl_lang.addItem("English (严格字符)", "en")
        self.combo_tpl_lang.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_tpl_lang.setMinimumContentsLength(8)
        self.combo_tpl_lang.currentIndexChanged.connect(self._on_tpl_lang_changed)
        chars_hdr.addWidget(self.combo_tpl_lang, 1)
        self.chars_live_label = QLabel("9 字")
        self.chars_live_label.setObjectName("BigValueLabel")
        chars_hdr.addWidget(self.chars_live_label)
        core_card.body().addLayout(chars_hdr)

        chars_ctrl = QHBoxLayout()
        chars_ctrl.setSpacing(8)
        self.chars_slider = QSlider(Qt.Orientation.Horizontal)
        self.chars_slider.setRange(5, 45)
        self.chars_spin = QSpinBox()
        self.chars_spin.setRange(5, 45)
        self.chars_spin.setSuffix(" 字")
        self.chars_spin.setMinimumWidth(88)
        self.chars_spin.setMinimumHeight(29)
        chars_ctrl.addWidget(self.chars_slider, 1)
        chars_ctrl.addWidget(self.chars_spin)
        core_card.body().addLayout(chars_ctrl)

        self.chars_slider.valueChanged.connect(self._on_chars_slider_changed)
        self.chars_spin.valueChanged.connect(self._on_chars_spin_changed)

        # Max lines per cue (1-line / 2-line Segmented Track)
        lines_row = QHBoxLayout()
        lines_row.addWidget(QLabel("最大行数:"))
        lines_track = AnimatedSegmentTrack()
        self.btn_1line = QPushButton("1 行")
        self.btn_2line = QPushButton("2 行")
        self.lines_group = QButtonGroup(self)
        self.lines_group.setExclusive(True)
        self.lines_group.addButton(self.btn_1line, 1)
        self.lines_group.addButton(self.btn_2line, 2)
        lines_track.add_segment(self.btn_1line, 1)
        lines_track.add_segment(self.btn_2line, 1)
        lines_row.addWidget(lines_track, 1)
        core_card.body().addLayout(lines_row)
        self.btn_1line.clicked.connect(lambda: self._on_lines_changed(1))
        self.btn_2line.clicked.connect(lambda: self._on_lines_changed(2))

        layout.addWidget(core_card)
        layout.addStretch()

        # ---- Tab 2: Advanced NLP, Bilingual Rules & Timing Constraints ----
        adv_scroll = QScrollArea()
        adv_scroll.setWidgetResizable(True)
        adv_scroll.setFrameShape(QFrame.Shape.NoFrame)
        adv_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        adv_container = QWidget()
        adv_container.setObjectName("TransparentContainer")
        adv_lay = QVBoxLayout(adv_container)
        adv_lay.setContentsMargins(4, 6, 6, 4)
        adv_lay.setSpacing(6)
        adv_scroll.setWidget(adv_container)
        self.tabs.addTab(adv_scroll, "高级断句与时长规则")

        nlp_card = GlassCard("智能断句 · 标点 · 双语词数规则")
        nlp_card.layout().setContentsMargins(12, 9, 12, 9)
        nlp_card.body().setSpacing(6)
        self.chk_smart_split = QCheckBox("智能语义断句")
        self.chk_remove_punct = QCheckBox("自动去除标点")
        self.chk_auto_merge = QCheckBox("合并过短碎句")
        self.chk_auto_split = QCheckBox("拆分超限长句")
        self.chk_balance_lines = QCheckBox("平衡双行长度")
        self.chk_smart_split.setToolTip("结合分词/语义边界、词性保护、真实停顿与动态规划全局断句")
        self.chk_remove_punct.setToolTip("去除常规显示标点，智能保留小数点(3.14)、英文撇号(don't/you're)与缩写(Dr./U.S.)")

        chk_grid = QGridLayout()
        chk_grid.setSpacing(4)
        chk_grid.addWidget(self.chk_smart_split, 0, 0)
        chk_grid.addWidget(self.chk_remove_punct, 0, 1)
        chk_grid.addWidget(self.chk_auto_merge, 1, 0)
        chk_grid.addWidget(self.chk_auto_split, 1, 1)
        chk_grid.addWidget(self.chk_balance_lines, 2, 0, 1, 2)
        for cb in (
            self.chk_smart_split,
            self.chk_remove_punct,
            self.chk_auto_merge,
            self.chk_auto_split,
            self.chk_balance_lines,
        ):
            cb.toggled.connect(self._emit_template_update)
        nlp_card.body().addLayout(chk_grid)

        en_row = QHBoxLayout()
        en_row.setSpacing(6)
        en_row.addWidget(QLabel("英文1行词数≥"))
        self.spin_min_words_l1 = QSpinBox()
        self.spin_min_words_l1.setRange(1, 5)
        self.spin_min_words_l1.setMinimumWidth(52)
        self.spin_min_words_l1.setMinimumHeight(26)
        self.spin_min_words_l1.valueChanged.connect(self._emit_template_update)
        en_row.addWidget(self.spin_min_words_l1)
        en_row.addWidget(QLabel("2行词数≥"))
        self.spin_min_words_l2 = QSpinBox()
        self.spin_min_words_l2.setRange(1, 6)
        self.spin_min_words_l2.setMinimumWidth(52)
        self.spin_min_words_l2.setMinimumHeight(26)
        self.spin_min_words_l2.valueChanged.connect(self._emit_template_update)
        en_row.addWidget(self.spin_min_words_l2)
        en_row.addStretch()
        nlp_card.body().addLayout(en_row)

        w_row = QHBoxLayout()
        w_row.addWidget(QLabel("字宽:"))
        self.combo_width_mode = ThemedComboBox()
        self.combo_width_mode.addItem("视觉等效 (英文数字 0.55字)", "visual")
        self.combo_width_mode.addItem("严格计数 (每字符 1字)", "strict")
        self.combo_width_mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_width_mode.setMinimumContentsLength(10)
        self.combo_width_mode.currentIndexChanged.connect(self._emit_template_update)
        w_row.addWidget(self.combo_width_mode, 1)
        nlp_card.body().addLayout(w_row)
        adv_lay.addWidget(nlp_card)

        timing_card = GlassCard("显示时长 · 阅读速度 · 安全留白")
        timing_card.layout().setContentsMargins(12, 9, 12, 9)
        timing_card.body().setSpacing(6)

        p_grid = QGridLayout()
        p_grid.setSpacing(5)
        p_grid.setColumnStretch(1, 1)

        self.spin_min_dur = QSpinBox()
        self.spin_min_dur.setRange(150, 3000)
        self.spin_min_dur.setSingleStep(50)
        self.spin_min_dur.setSuffix(" ms")
        self.spin_min_dur.setMinimumWidth(0)

        self.spin_max_dur = QSpinBox()
        self.spin_max_dur.setRange(1500, 15000)
        self.spin_max_dur.setSingleStep(250)
        self.spin_max_dur.setSuffix(" ms")
        self.spin_max_dur.setMinimumWidth(0)

        self.spin_max_cps = QDoubleSpinBox()
        self.spin_max_cps.setRange(4.0, 30.0)
        self.spin_max_cps.setSingleStep(0.5)
        self.spin_max_cps.setSuffix(" 字/s")
        self.spin_max_cps.setMinimumWidth(0)

        self.spin_min_gap = QSpinBox()
        self.spin_min_gap.setRange(0, 500)
        self.spin_min_gap.setSingleStep(10)
        self.spin_min_gap.setSuffix(" ms")
        self.spin_min_gap.setMinimumWidth(0)

        self.combo_padding_unit = ThemedComboBox()
        self.combo_padding_unit.addItem("按帧数 (随视频帧率)", "frames")
        self.combo_padding_unit.addItem("按毫秒 (固定 ms)", "ms")
        self.combo_padding_unit.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_padding_unit.setMinimumContentsLength(8)
        self.combo_padding_unit.setToolTip("预留空间可按视频帧数（推荐，默认前后各 3 帧）或固定毫秒自定义设置")
        self.combo_padding_unit.currentIndexChanged.connect(self._on_padding_unit_changed)

        self.lbl_padding_hint = QLabel("前120/后120ms@25fps")
        self.lbl_padding_hint.setObjectName("MutedLabel")
        self.lbl_padding_hint.setWordWrap(True)

        self.spin_lead_offset = QSpinBox()
        self.spin_lead_offset.setRange(-10, 30)
        self.spin_lead_offset.setSingleStep(1)
        self.spin_lead_offset.setSuffix(" 帧")
        self.spin_lead_offset.setMinimumWidth(0)
        self.spin_lead_offset.setToolTip(
            "字幕提前入场缓冲（仅向句前真实静音/停顿区间扩展；若与上一条为连续不断句对话，会自动保持原字边界不抢跑）。"
        )

        self.spin_tail_offset = QSpinBox()
        self.spin_tail_offset.setRange(-10, 30)
        self.spin_tail_offset.setSingleStep(1)
        self.spin_tail_offset.setSuffix(" 帧")
        self.spin_tail_offset.setMinimumWidth(0)
        self.spin_tail_offset.setToolTip(
            "字幕延后退场缓冲（仅向句末真实静音/停顿区间扩展，不挤压下一条已开口发音的时间边界）。"
        )

        self.chk_bridge_gaps = QCheckBox("同句衔接防闪烁")
        self.chk_bridge_gaps.setToolTip(
            "同一句连续字幕若间隔不超过上限，将延伸衔接至下一条起始（留 1ms 防重叠间隙），"
            "避免达芬奇画面闪烁；不同说话人或句末保持最小间隔。"
        )
        self.spin_bridge_gap = QSpinBox()
        self.spin_bridge_gap.setRange(50, 1500)
        self.spin_bridge_gap.setSingleStep(50)
        self.spin_bridge_gap.setSuffix(" ms")
        self.spin_bridge_gap.setMinimumWidth(0)

        for sp in (
            self.spin_min_dur,
            self.spin_max_dur,
            self.spin_max_cps,
            self.spin_min_gap,
            self.spin_lead_offset,
            self.spin_tail_offset,
            self.spin_bridge_gap,
        ):
            sp.setMinimumHeight(26)
            sp.valueChanged.connect(self._emit_template_update)
        self.chk_bridge_gaps.toggled.connect(self._emit_template_update)

        p_grid.addWidget(QLabel("最短显示:"), 0, 0)
        p_grid.addWidget(self.spin_min_dur, 0, 1)
        p_grid.addWidget(QLabel("最长显示:"), 1, 0)
        p_grid.addWidget(self.spin_max_dur, 1, 1)
        p_grid.addWidget(QLabel("阅读速度:"), 2, 0)
        p_grid.addWidget(self.spin_max_cps, 2, 1)
        p_grid.addWidget(QLabel("最小间隔:"), 3, 0)
        p_grid.addWidget(self.spin_min_gap, 3, 1)
        p_grid.addWidget(QLabel("预留单位:"), 4, 0)
        p_grid.addWidget(self.combo_padding_unit, 4, 1)
        p_grid.addWidget(QLabel("前预留:"), 5, 0)
        p_grid.addWidget(self.spin_lead_offset, 5, 1)
        p_grid.addWidget(QLabel("后预留:"), 6, 0)
        p_grid.addWidget(self.spin_tail_offset, 6, 1)
        p_grid.addWidget(self.chk_bridge_gaps, 7, 0)
        p_grid.addWidget(self.spin_bridge_gap, 7, 1)

        timing_card.body().addLayout(p_grid)
        timing_card.body().addWidget(self.lbl_padding_hint)
        adv_lay.addWidget(timing_card)
        adv_lay.addStretch()

        # --- Card 4: Pinned Live Subtitle Preview ---
        self.preview_box = LiveSubtitlePreviewBox()
        self.preview_box.sample_input.textChanged.connect(
            lambda: self.preview_box.update_preview(self._template)
        )
        preview_wrap = QHBoxLayout()
        preview_wrap.setContentsMargins(0, 0, 4, 0)
        preview_wrap.addWidget(self.preview_box)
        outer.addLayout(preview_wrap)

    def reload_templates_combo(self, select_id: str | None = None) -> None:
        self._loading = True
        self.tpl_combo.clear()
        for tpl in self.store.templates:
            tag = "[内置]" if tpl.is_builtin else "[自定]"
            self.tpl_combo.addItem(f"{tag} {tpl.name}", tpl.template_id)
        target_id = select_id or self._template.template_id
        idx = self.tpl_combo.findData(target_id)
        if idx >= 0:
            self.tpl_combo.setCurrentIndex(idx)
        self._loading = False

    def current_template(self) -> SubtitleTemplate:
        return deepcopy(self._template)

    def set_template(self, tpl: SubtitleTemplate, *, emit_signal: bool = False) -> None:
        self._loading = True
        self._template = deepcopy(tpl)

        self.btn_preset_en.setChecked(tpl.language == "en")
        self.btn_preset_916.setChecked(tpl.language != "en" and tpl.aspect_preset == "9:16")
        self.btn_preset_169.setChecked(tpl.language != "en" and tpl.aspect_preset == "16:9")

        l_idx = self.combo_tpl_lang.findData(tpl.language)
        self.combo_tpl_lang.setCurrentIndex(max(0, l_idx))
        unit_suffix = " 字符" if tpl.language == "en" else " 字"
        self.chars_spin.setSuffix(unit_suffix)

        self.chars_slider.setValue(max(5, min(45, tpl.max_chars_per_line)))
        self.chars_spin.setValue(max(5, min(45, tpl.max_chars_per_line)))
        self.chars_live_label.setText(f"{tpl.max_chars_per_line}{unit_suffix}")

        self.btn_1line.setChecked(tpl.max_lines == 1)
        self.btn_2line.setChecked(tpl.max_lines >= 2)

        self.chk_smart_split.setChecked(tpl.smart_semantic_split)
        self.chk_remove_punct.setChecked(tpl.remove_punctuation)
        self.chk_auto_merge.setChecked(tpl.auto_merge_short)
        self.chk_auto_split.setChecked(tpl.auto_split_long)
        self.chk_balance_lines.setChecked(tpl.balance_two_lines)
        self.spin_min_words_l1.setValue(max(1, tpl.min_words_line1))
        self.spin_min_words_l2.setValue(max(1, tpl.min_words_line2))
        self.spin_min_words_l1.setEnabled(tpl.language == "en")
        self.spin_min_words_l2.setEnabled(tpl.language == "en")

        widx = self.combo_width_mode.findData(tpl.width_mode)
        self.combo_width_mode.setCurrentIndex(max(0, widx))

        self.spin_min_dur.setValue(tpl.min_duration_ms)
        self.spin_max_dur.setValue(tpl.max_duration_ms)
        self.spin_max_cps.setValue(tpl.max_cps)
        self.spin_min_gap.setValue(tpl.min_gap_ms)
        p_idx = self.combo_padding_unit.findData(tpl.padding_unit)
        self.combo_padding_unit.setCurrentIndex(max(0, p_idx))
        self._configure_padding_spinboxes(tpl.padding_unit, tpl)
        self.chk_bridge_gaps.setChecked(tpl.bridge_sentence_gaps)
        self.spin_bridge_gap.setValue(tpl.bridge_max_gap_ms)
        self.spin_bridge_gap.setEnabled(tpl.bridge_sentence_gaps)
        self._update_padding_hint()

        idx = self.tpl_combo.findData(tpl.template_id)
        if idx >= 0:
            self.tpl_combo.setCurrentIndex(idx)
        self._update_template_buttons_state()
        self._loading = False

        self.preview_box.update_preview(self._template)
        if emit_signal:
            self.templateChanged.emit(self.current_template())

    def apply_aspect_preset(self, aspect: str, *, keep_custom_if_modified: bool = False) -> None:
        if aspect == "en":
            preset = SubtitleTemplate.preset_english_default()
        elif aspect == "9:16":
            preset = SubtitleTemplate.preset_portrait_9_16()
        else:
            preset = SubtitleTemplate.preset_landscape_16_9()
        if keep_custom_if_modified and not self._template.is_builtin:
            return
        self.set_template(preset, emit_signal=True)

    def _on_tpl_lang_changed(self, _idx: int) -> None:
        if self._loading:
            return
        lang = self.combo_tpl_lang.currentData() or "zh"
        if lang == "en":
            self._template.language = "en"
            self._template.width_mode = "strict"
            if self._template.is_builtin and self._template.template_id in {"preset_9_16", "preset_16_9"}:
                self.set_template(SubtitleTemplate.preset_english_default(), emit_signal=True)
                return
        else:
            self._template.language = "zh"
            if self._template.is_builtin and self._template.template_id.startswith("preset_en"):
                self.set_template(SubtitleTemplate.preset_portrait_9_16(), emit_signal=True)
                return
        self.set_template(self._template, emit_signal=True)

    def _update_template_buttons_state(self) -> None:
        is_builtin = self._template.is_builtin
        self.btn_rename_tpl.setEnabled(not is_builtin)
        self.btn_del_tpl.setEnabled(not is_builtin)

    def _on_chars_slider_changed(self, val: int) -> None:
        self.chars_spin.blockSignals(True)
        self.chars_spin.setValue(val)
        self.chars_spin.blockSignals(False)
        unit_suffix = " 字符" if self._template.language == "en" else " 字"
        self.chars_live_label.setText(f"{val}{unit_suffix}")
        self._emit_template_update()

    def _on_chars_spin_changed(self, val: int) -> None:
        self.chars_slider.blockSignals(True)
        self.chars_slider.setValue(val)
        self.chars_slider.blockSignals(False)
        unit_suffix = " 字符" if self._template.language == "en" else " 字"
        self.chars_live_label.setText(f"{val}{unit_suffix}")
        self._emit_template_update()

    def _on_lines_changed(self, lines: int) -> None:
        self._template.max_lines = lines
        self._emit_template_update()

    def set_video_fps(self, fps: float) -> None:
        """Update active video frame rate so frame-based padding readout reflects actual ms."""
        self._video_fps = float(fps) if fps and fps >= 10.0 else 25.0
        self._update_padding_hint()

    def _configure_padding_spinboxes(self, unit: str, tpl: SubtitleTemplate) -> None:
        for sp in (self.spin_lead_offset, self.spin_tail_offset):
            sp.blockSignals(True)
        if unit == "frames":
            self.spin_lead_offset.setRange(-10, 30)
            self.spin_lead_offset.setSingleStep(1)
            self.spin_lead_offset.setSuffix(" 帧")
            self.spin_lead_offset.setValue(tpl.lead_frames)

            self.spin_tail_offset.setRange(-10, 30)
            self.spin_tail_offset.setSingleStep(1)
            self.spin_tail_offset.setSuffix(" 帧")
            self.spin_tail_offset.setValue(tpl.tail_frames)
        else:
            self.spin_lead_offset.setRange(-300, 1000)
            self.spin_lead_offset.setSingleStep(10)
            self.spin_lead_offset.setSuffix(" ms")
            self.spin_lead_offset.setValue(tpl.lead_offset_ms)

            self.spin_tail_offset.setRange(-300, 1000)
            self.spin_tail_offset.setSingleStep(10)
            self.spin_tail_offset.setSuffix(" ms")
            self.spin_tail_offset.setValue(tpl.tail_offset_ms)
        for sp in (self.spin_lead_offset, self.spin_tail_offset):
            sp.blockSignals(False)

    def _update_padding_hint(self) -> None:
        t = self._template
        lead_ms, tail_ms = t.resolve_padding_ms(self._video_fps)
        self.lbl_padding_hint.setText(f"前{lead_ms}/后{tail_ms}ms@{self._video_fps:.3g}fps")

    def _on_padding_unit_changed(self, _idx: int) -> None:
        if self._loading:
            return
        unit = self.combo_padding_unit.currentData() or "frames"
        self._template.padding_unit = unit
        self._loading = True
        self._configure_padding_spinboxes(unit, self._template)
        self._loading = False
        self._emit_template_update()

    def _emit_template_update(self) -> None:
        if self._loading:
            return
        t = self._template
        t.language = self.combo_tpl_lang.currentData() or "zh"
        t.max_chars_per_line = self.chars_spin.value()
        t.max_lines = 2 if self.btn_2line.isChecked() else 1
        t.min_words_line1 = self.spin_min_words_l1.value()
        t.min_words_line2 = self.spin_min_words_l2.value()
        t.balance_two_lines = self.chk_balance_lines.isChecked()
        t.smart_semantic_split = self.chk_smart_split.isChecked()
        t.remove_punctuation = self.chk_remove_punct.isChecked()
        t.auto_merge_short = self.chk_auto_merge.isChecked()
        t.auto_split_long = self.chk_auto_split.isChecked()
        t.width_mode = "strict" if t.language == "en" else (self.combo_width_mode.currentData() or "visual")
        t.min_duration_ms = self.spin_min_dur.value()
        t.max_duration_ms = max(t.min_duration_ms + 200, self.spin_max_dur.value())
        t.max_cps = self.spin_max_cps.value()
        t.min_gap_ms = self.spin_min_gap.value()
        unit = self.combo_padding_unit.currentData() or "frames"
        t.padding_unit = unit
        if unit == "frames":
            t.lead_frames = self.spin_lead_offset.value()
            t.tail_frames = self.spin_tail_offset.value()
            t.lead_offset_ms, t.tail_offset_ms = t.resolve_padding_ms(self._video_fps)
        else:
            t.lead_offset_ms = self.spin_lead_offset.value()
            t.tail_offset_ms = self.spin_tail_offset.value()
        t.bridge_sentence_gaps = self.chk_bridge_gaps.isChecked()
        t.bridge_max_gap_ms = self.spin_bridge_gap.value()
        self.spin_bridge_gap.setEnabled(t.bridge_sentence_gaps)
        self._update_padding_hint()

        self.preview_box.update_preview(t)
        self.templateChanged.emit(self.current_template())

    def _on_template_selected(self, idx: int) -> None:
        if self._loading or idx < 0:
            return
        tid = self.tpl_combo.itemData(idx)
        tpl = next((t for t in self.store.templates if t.template_id == tid), None)
        if tpl is not None:
            self.set_template(tpl, emit_signal=True)

    def _on_save_template(self) -> None:
        if self._template.is_builtin:
            self._on_save_as_template()
            return
        saved = self.store.add_or_update_template(self._template)
        self._template = deepcopy(saved)
        self.reload_templates_combo(saved.template_id)
        QMessageBox.information(self, "模板已保存", f"自定义模板「{saved.name}」已保存。")

    def _on_save_as_template(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "保存新模板",
            "请输入模板名称:",
            text=f"{self._template.name} (自定义)",
        )
        if not ok or not name.strip():
            return
        cloned = self._template.clone(new_name=name.strip())
        saved = self.store.add_or_update_template(cloned)
        self.reload_templates_combo(saved.template_id)
        self.set_template(saved, emit_signal=True)

    def _on_rename_template(self) -> None:
        if self._template.is_builtin:
            return
        name, ok = QInputDialog.getText(
            self,
            "重命名模板",
            "新模板名称:",
            text=self._template.name,
        )
        if ok and name.strip():
            self.store.rename_template(self._template.template_id, name.strip())
            self._template.name = name.strip()
            self.reload_templates_combo(self._template.template_id)

    def _on_copy_template(self) -> None:
        copied = self.store.duplicate_template(self._template.template_id)
        if copied is not None:
            self.reload_templates_combo(copied.template_id)
            self.set_template(copied, emit_signal=True)

    def _on_delete_template(self) -> None:
        if self._template.is_builtin:
            return
        ans = QMessageBox.question(
            self,
            "删除模板",
            f"确认删除自定义模板「{self._template.name}」？",
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.store.delete_template(self._template.template_id)
            fallback = SubtitleTemplate.preset_portrait_9_16()
            self.reload_templates_combo(fallback.template_id)
            self.set_template(fallback, emit_signal=True)

    def _on_reset_preset_defaults(self) -> None:
        if self.btn_preset_169.isChecked() or self._template.aspect_preset == "16:9":
            self.apply_aspect_preset("16:9")
        else:
            self.apply_aspect_preset("9:16")

    def set_preview_sample_text(self, text: str) -> None:
        self.preview_box.update_preview(self._template, custom_text=text)
