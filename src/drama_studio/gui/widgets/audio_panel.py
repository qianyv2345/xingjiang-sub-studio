"""Glassmorphism audio leveling controls, KPI cards, collapsible advanced params, and A/B player."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import QEasingCurve, QEventLoop, QPropertyAnimation, Qt, QTimer, QUrl, Signal
from PySide6.QtMultimedia import (
    QAudioFormat,
    QAudioOutput,
    QAudioSink,
    QMediaDevices,
    QMediaPlayer,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
import soundfile as sf

from drama_studio.gui.widgets.glass_card import GlassCard, KpiCard
from drama_studio.gui.widgets.themed_controls import AnimatedSegmentTrack
from drama_studio.models import AudioAnalysisReport, AudioLevelingConfig


def _fmt_ms(ms: int) -> str:
    ms = max(0, int(ms))
    total_s = ms / 1000.0
    m = int(total_s // 60)
    s = total_s % 60
    return f"{m:02d}:{s:04.1f}"


class _PcmScrubber:
    """Low-latency audio grain scrubber via QtMultimedia push QAudioSink (header-only cache, no WASAPI lock)."""

    def __init__(self, parent: QWidget) -> None:
        self._parent = parent
        self._pcm_cache: dict[str, tuple[int, int, int]] = {}  # path -> (sr, channels, frames)
        self._active_path: str = ""
        self._sink: QAudioSink | None = None
        self._io = None
        self._sink_fmt: QAudioFormat | None = None
        self._last_scrub_ts: float = 0.0
        self._last_ms: int = -1000
        self._idle_timer = QTimer(parent)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(180)
        self._idle_timer.timeout.connect(self.stop)

    def load_file(self, path: str) -> None:
        self._active_path = path
        if not path or path in self._pcm_cache or not Path(path).is_file():
            return
        try:
            info = sf.info(path)
            self._pcm_cache[path] = (int(info.samplerate), int(info.channels), int(info.frames))
        except Exception:
            pass

    def invalidate_cache(self) -> None:
        self._idle_timer.stop()
        self.stop()
        self._pcm_cache.clear()
        self._active_path = ""

    def stop(self) -> None:
        self._idle_timer.stop()
        self._io = None
        if self._sink is not None:
            try:
                self._sink.stop()
            except Exception:
                pass

    def _ensure_sink(self, src_sr: int, src_ch: int) -> bool:
        app = QApplication.instance()
        if app is None or app.platformName() == "offscreen":
            return False
        try:
            dev = QMediaDevices.defaultAudioOutput()
            if dev.isNull():
                return False
            desired = QAudioFormat()
            desired.setSampleRate(src_sr)
            desired.setChannelCount(min(2, max(1, src_ch)))
            desired.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            fmt = desired if dev.isFormatSupported(desired) else dev.preferredFormat()

            if self._sink is None or self._sink_fmt != fmt:
                self.stop()
                if self._sink is not None:
                    self._sink.deleteLater()
                self._sink = QAudioSink(dev, fmt, self._parent)
                self._sink.setVolume(0.92)
                buf_bytes = max(4096, fmt.bytesPerFrame() * fmt.sampleRate() * 110 // 1000)
                self._sink.setBufferSize(buf_bytes)
                self._sink_fmt = fmt
                self._io = None
            if self._io is None:
                self._io = self._sink.start()
            return self._io is not None
        except Exception:
            return False

    def play_grain(self, ms: int, active_path: str) -> None:
        if not active_path:
            return
        now = time.monotonic()
        if now - self._last_scrub_ts < 0.055 and abs(ms - self._last_ms) < 40:
            return
        self._last_scrub_ts = now
        self._last_ms = ms

        if active_path not in self._pcm_cache:
            self.load_file(active_path)
        entry = self._pcm_cache.get(active_path)
        if entry is None:
            return
        sr, channels, total_frames = entry
        if total_frames <= 0 or sr <= 0:
            return

        if not self._ensure_sink(sr, channels) or self._sink is None or self._io is None or self._sink_fmt is None:
            return

        grain_len = max(1, int(sr * 0.072))
        start = max(0, min(total_frames - 1, int(round(ms * sr / 1000.0))))
        to_read = min(grain_len, total_frames - start)
        if to_read < 8:
            return
        try:
            with sf.SoundFile(active_path) as sf_fh:
                sf_fh.seek(start)
                seg = sf_fh.read(to_read, dtype="float32", always_2d=True)
        except Exception:
            return
        if seg.shape[0] < 8:
            return

        # Smooth raised-cosine fade-in (6ms) and fade-out (14ms) eliminates scrub clicks
        fade_in = min(seg.shape[0] // 3, max(2, int(sr * 0.006)))
        fade_out = min(seg.shape[0] // 2, max(2, int(sr * 0.014)))
        win = np.ones(seg.shape[0], dtype=np.float32)
        win[:fade_in] = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
        win[-fade_out:] = np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
        seg *= win[:, None]

        # Adapt sample rate / channels / sample format to hardware format
        out_sr = self._sink_fmt.sampleRate()
        out_ch = self._sink_fmt.channelCount()
        if out_sr != sr and seg.shape[0] > 1:
            n_out = max(4, int(round(seg.shape[0] * out_sr / sr)))
            xp = np.linspace(0.0, 1.0, seg.shape[0])
            x_new = np.linspace(0.0, 1.0, n_out)
            seg = np.column_stack([np.interp(x_new, xp, seg[:, c]) for c in range(seg.shape[1])]).astype(np.float32)

        if seg.shape[1] != out_ch:
            if out_ch == 1:
                seg = np.mean(seg, axis=1, keepdims=True)
            else:
                mono = seg[:, :1]
                seg = np.repeat(mono, out_ch, axis=1)

        if self._sink_fmt.sampleFormat() == QAudioFormat.SampleFormat.Float:
            raw_bytes = np.clip(seg, -1.0, 1.0).astype(np.float32).tobytes()
        else:
            raw_bytes = (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

        # Never reset/restart WASAPI sink mid-drag; skip frame if audio device buffer is still busy
        free = self._sink.bytesFree()
        if free >= max(64, len(raw_bytes) // 3) and self._io is not None:
            self._io.write(raw_bytes[:free])
        self._idle_timer.start()


class AudioPanel(QWidget):
    """Sleek studio bar with A/B comparison player, 4 KPI cards, and leveling mode controls."""

    configChanged = Signal(object)
    relevelRequested = Signal()
    playheadUpdated = Signal(int)
    playheadScrubbed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("TransparentContainer")
        self._cfg = AudioLevelingConfig.from_mode("standard")
        self._loading = False
        self._raw_wav: str = ""
        self._proc_wav: str = ""
        self._active_channel: str = "B"
        self._duration_ms: int = 0
        self._pending_seek_ms: int | None = None
        self._resume_after_switch: bool = False
        self._scrubber = _PcmScrubber(self)
        self._adv_anim: QPropertyAnimation | None = None

        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.audio_out.setVolume(0.95)
        self.player.setAudioOutput(self.audio_out)
        self.player.positionChanged.connect(self._on_player_pos)
        self.player.durationChanged.connect(self._on_player_dur)
        self.player.mediaStatusChanged.connect(self._on_media_status)

        self._build_ui()
        self.set_config(self._cfg, emit_signal=False)

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # --- Card 1: A/B Player Bar + Mode Presets ---
        top_card = GlassCard()
        top_card.body().setSpacing(8)

        # Row 1: Player Transport & A/B Switch
        p_lay = QHBoxLayout()
        p_lay.setSpacing(8)
        self.btn_play = QPushButton("▶ 试听")
        self.btn_play.setMinimumWidth(80)
        self.btn_play.setToolTip("播放 / 暂停试听 (快捷键: 空格 Space)")
        self.btn_stop = QPushButton("■")
        self.btn_stop.setMaximumWidth(36)
        self.btn_stop.setToolTip("停止试听")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop.clicked.connect(self.stop_audio)

        ab_track = AnimatedSegmentTrack()
        self.btn_mode_a = QPushButton("A · 原片")
        self.btn_mode_b = QPushButton("B · 均衡后")
        self.ab_group = QButtonGroup(self)
        self.ab_group.setExclusive(True)
        self.ab_group.addButton(self.btn_mode_a, 0)
        self.ab_group.addButton(self.btn_mode_b, 1)
        self.btn_mode_b.setChecked(True)
        ab_track.add_segment(self.btn_mode_a)
        ab_track.add_segment(self.btn_mode_b)
        self.btn_mode_a.clicked.connect(lambda: self.switch_ab("A"))
        self.btn_mode_b.clicked.connect(lambda: self.switch_ab("B"))

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setToolTip("拖动可实时有声预览音频与字幕")
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        self.seek_slider.sliderReleased.connect(self._scrubber.stop)
        self.lbl_time = QLabel("00:00.0 / 00:00.0")
        self.lbl_time.setObjectName("MonoTimeLabel")

        p_lay.addWidget(self.btn_play)
        p_lay.addWidget(self.btn_stop)
        p_lay.addWidget(ab_track)
        p_lay.addWidget(self.seek_slider, 1)
        p_lay.addWidget(self.lbl_time)
        top_card.body().addLayout(p_lay)

        # Row 2: Mode Segmented Track + Relevel + Collapsible Advanced Toggle
        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(QLabel("人声均衡:"))

        mode_track = AnimatedSegmentTrack()
        self.btn_light = QPushButton("轻度")
        self.btn_std = QPushButton("标准")
        self.btn_std.setToolTip("标准人声均衡（推荐）")
        self.btn_strong = QPushButton("强力")
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_light, 0)
        self.mode_group.addButton(self.btn_std, 1)
        self.mode_group.addButton(self.btn_strong, 2)
        self.btn_std.setChecked(True)
        mode_track.add_segment(self.btn_light)
        mode_track.add_segment(self.btn_std)
        mode_track.add_segment(self.btn_strong)
        self.btn_light.clicked.connect(lambda: self._apply_preset("light"))
        self.btn_std.clicked.connect(lambda: self._apply_preset("standard"))
        self.btn_strong.clicked.connect(lambda: self._apply_preset("strong"))

        self.btn_adv_toggle = QPushButton("高级参数 ▾")
        self.btn_adv_toggle.setObjectName("TogglePill")
        self.btn_adv_toggle.setCheckable(True)
        self.btn_adv_toggle.toggled.connect(self._toggle_advanced)

        self.btn_kpi_toggle = QPushButton("响度仪表 ▴")
        self.btn_kpi_toggle.setObjectName("TogglePill")
        self.btn_kpi_toggle.setCheckable(True)
        self.btn_kpi_toggle.setChecked(True)
        self.btn_kpi_toggle.setToolTip("折叠或展开下方的 4 项人声响度统计卡片")
        self.btn_kpi_toggle.toggled.connect(self._toggle_kpi_cards)

        self.btn_relevel = QPushButton("重新应用均衡曲线")
        self.btn_relevel.setObjectName("SuccessButton")
        self.btn_relevel.clicked.connect(self.relevelRequested.emit)

        mode_row.addWidget(mode_track)
        mode_row.addWidget(self.btn_adv_toggle)
        mode_row.addWidget(self.btn_kpi_toggle)
        mode_row.addStretch()
        mode_row.addWidget(self.btn_relevel)
        top_card.body().addLayout(mode_row)

        # Collapsible Advanced Panel with smooth height reveal & generous spinbox height
        self.adv_widget = QWidget()
        self.adv_widget.setMaximumHeight(0)
        self.adv_widget.setVisible(False)
        adv_grid = QGridLayout(self.adv_widget)
        adv_grid.setContentsMargins(0, 4, 0, 0)
        adv_grid.setHorizontalSpacing(10)
        adv_grid.setVerticalSpacing(5)

        self.spin_target_lufs = QDoubleSpinBox()
        self.spin_target_lufs.setRange(-26.0, -10.0)
        self.spin_target_lufs.setSingleStep(0.5)
        self.spin_target_lufs.setSuffix(" LUFS")

        self.spin_peak_ceil = QDoubleSpinBox()
        self.spin_peak_ceil.setRange(-6.0, -0.3)
        self.spin_peak_ceil.setSingleStep(0.2)
        self.spin_peak_ceil.setSuffix(" dBTP")

        self.spin_max_boost = QDoubleSpinBox()
        self.spin_max_boost.setRange(3.0, 30.0)
        self.spin_max_boost.setSingleStep(1.0)
        self.spin_max_boost.setSuffix(" dB")

        self.spin_emotion = QDoubleSpinBox()
        self.spin_emotion.setRange(0.0, 0.9)
        self.spin_emotion.setSingleStep(0.05)

        self.spin_gate = QDoubleSpinBox()
        self.spin_gate.setRange(-65.0, -25.0)
        self.spin_gate.setSingleStep(1.0)
        self.spin_gate.setSuffix(" dB")

        self.spin_hp = QSpinBox()
        self.spin_hp.setRange(20, 150)
        self.spin_hp.setSuffix(" Hz")

        self.chk_sub_win = QCheckBox("结合字幕时间窗保护轻声气口")

        for w in (
            self.spin_target_lufs,
            self.spin_peak_ceil,
            self.spin_max_boost,
            self.spin_emotion,
            self.spin_gate,
            self.spin_hp,
        ):
            w.setFixedHeight(28)
            w.setMinimumWidth(98)
            w.valueChanged.connect(self._on_param_edited)
        self.chk_sub_win.toggled.connect(self._on_param_edited)

        adv_grid.addWidget(QLabel("目标响度:"), 0, 0)
        adv_grid.addWidget(self.spin_target_lufs, 0, 1)
        adv_grid.addWidget(QLabel("峰值上限:"), 0, 2)
        adv_grid.addWidget(self.spin_peak_ceil, 0, 3)
        adv_grid.addWidget(QLabel("最大提升:"), 0, 4)
        adv_grid.addWidget(self.spin_max_boost, 0, 5)
        adv_grid.setColumnStretch(6, 1)

        adv_grid.addWidget(QLabel("情绪保留:"), 1, 0)
        adv_grid.addWidget(self.spin_emotion, 1, 1)
        adv_grid.addWidget(QLabel("底噪门限:"), 1, 2)
        adv_grid.addWidget(self.spin_gate, 1, 3)
        adv_grid.addWidget(QLabel("高通滤波:"), 1, 4)
        adv_grid.addWidget(self.spin_hp, 1, 5)
        adv_grid.addWidget(self.chk_sub_win, 1, 6, 1, 2)
        top_card.body().addWidget(self.adv_widget)

        lay.addWidget(top_card)

        # --- Row 2: Collapsible 4 KPI Glass Cards (Aurora Dashboard Style) ---
        self.kpi_container = QWidget()
        self.kpi_container.setObjectName("TransparentContainer")
        kpi_row = QHBoxLayout(self.kpi_container)
        kpi_row.setContentsMargins(0, 0, 0, 0)
        kpi_row.setSpacing(6)
        self.kpi_before = KpiCard("原片综合响度", "-", "短时区间 -")
        self.kpi_after = KpiCard("均衡后响度", "-", "目标 -16 LUFS", accent=True)
        self.kpi_lra = KpiCard("动态范围 LRA", "-", "语音占比 -")
        self.kpi_peak = KpiCard("真峰值 True Peak", "-", "增益范围 -")
        for k in (self.kpi_before, self.kpi_after, self.kpi_lra, self.kpi_peak):
            kpi_row.addWidget(k, 1)
        lay.addWidget(self.kpi_container)

    def _toggle_kpi_cards(self, checked: bool) -> None:
        self.btn_kpi_toggle.setText("响度仪表 ▴" if checked else "响度仪表 ▾")
        self.kpi_container.setVisible(checked)

    def _toggle_advanced(self, checked: bool) -> None:
        self.btn_adv_toggle.setText("高级参数 ▴" if checked else "高级参数 ▾")
        if self._adv_anim is not None:
            self._adv_anim.stop()

        if checked:
            self.adv_widget.setVisible(True)
            self.adv_widget.setMaximumHeight(16777215)
            end_h = max(72, self.adv_widget.sizeHint().height())
            self.adv_widget.setMaximumHeight(0)
            start_h = 0
        else:
            start_h = max(0, self.adv_widget.height())
            end_h = 0

        anim = QPropertyAnimation(self.adv_widget, b"maximumHeight", self)
        anim.setDuration(165)
        anim.setStartValue(start_h)
        anim.setEndValue(end_h)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _done() -> None:
            if self.btn_adv_toggle.isChecked():
                self.adv_widget.setMaximumHeight(16777215)
            else:
                self.adv_widget.setVisible(False)
                self.adv_widget.setMaximumHeight(0)

        anim.finished.connect(_done)
        anim.start()
        self._adv_anim = anim

    def current_config(self) -> AudioLevelingConfig:
        return deepcopy(self._cfg)

    def set_config(self, cfg: AudioLevelingConfig, *, emit_signal: bool = False) -> None:
        self._loading = True
        self._cfg = deepcopy(cfg)
        self.btn_light.setChecked(cfg.mode == "light")
        self.btn_std.setChecked(cfg.mode == "standard")
        self.btn_strong.setChecked(cfg.mode == "strong")

        self.spin_target_lufs.setValue(cfg.target_lufs)
        self.spin_peak_ceil.setValue(cfg.true_peak_ceiling_dbtp)
        self.spin_max_boost.setValue(cfg.max_boost_db)
        self.spin_emotion.setValue(cfg.emotion_preserve)
        self.spin_gate.setValue(cfg.noise_gate_db)
        self.spin_hp.setValue(int(cfg.highpass_hz))
        self.chk_sub_win.setChecked(cfg.use_subtitle_windows)
        self._loading = False
        if emit_signal:
            self.configChanged.emit(self.current_config())

    def _apply_preset(self, mode: str) -> None:
        pts = deepcopy(self._cfg.manual_points)
        new_cfg = AudioLevelingConfig.from_mode(mode)  # type: ignore[arg-type]
        new_cfg.manual_points = pts
        self.set_config(new_cfg, emit_signal=True)

    def _on_param_edited(self) -> None:
        if self._loading:
            return
        self._cfg.target_lufs = self.spin_target_lufs.value()
        self._cfg.true_peak_ceiling_dbtp = self.spin_peak_ceil.value()
        self._cfg.max_boost_db = self.spin_max_boost.value()
        self._cfg.emotion_preserve = self.spin_emotion.value()
        self._cfg.noise_gate_db = self.spin_gate.value()
        self._cfg.highpass_hz = float(self.spin_hp.value())
        self._cfg.use_subtitle_windows = self.chk_sub_win.isChecked()
        self.configChanged.emit(self.current_config())

    def _active_wav_path(self) -> str:
        target = self._proc_wav if (self._active_channel == "B" and self._proc_wav) else self._raw_wav
        return target if (target and Path(target).is_file()) else ""

    def set_audio_sources(self, raw_wav: str, processed_wav: str, report: AudioAnalysisReport | None) -> None:
        self._scrubber.invalidate_cache()
        self.player.stop()
        self._raw_wav = raw_wav
        self._proc_wav = processed_wav
        if report:
            self.kpi_before.set_metric(
                f"{report.before_lufs:.1f} LUFS",
                f"短时 [{report.before_short_term_min_lufs:.1f}, {report.before_short_term_max_lufs:.1f}]",
            )
            self.kpi_after.set_metric(
                f"{report.after_lufs:.1f} LUFS",
                f"短时 [{report.after_short_term_min_lufs:.1f}, {report.after_short_term_max_lufs:.1f}]",
            )
            self.kpi_lra.set_metric(
                f"{report.before_lra_lu:.1f} → {report.after_lra_lu:.1f} LU",
                f"语音占比 {report.speech_ratio * 100:.1f}%",
            )
            self.kpi_peak.set_metric(
                f"{report.after_true_peak_dbtp:.1f} dBTP",
                f"增益 {report.min_applied_gain_db:+.1f} ~ {report.max_applied_gain_db:+.1f} dB",
            )
        else:
            self.kpi_before.set_metric("-", "短时区间 -")
            self.kpi_after.set_metric("-", f"目标 {self._cfg.target_lufs:.0f} LUFS")
            self.kpi_lra.set_metric("-", "语音占比 -")
            self.kpi_peak.set_metric("-", "增益范围 -")

        target = self._active_wav_path()
        if target:
            self.player.setSource(QUrl.fromLocalFile(target))
            self._scrubber.load_file(target)
        else:
            self.player.setSource(QUrl())

    def switch_ab(self, channel: str) -> None:
        if channel == self._active_channel:
            return
        self._scrubber.stop()
        self._active_channel = channel
        target = self._proc_wav if channel == "B" else self._raw_wav
        if not target or not Path(target).is_file():
            return
        cur_pos = self.player.position()
        was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self._pending_seek_ms = cur_pos
        self._resume_after_switch = was_playing
        self.player.setSource(QUrl.fromLocalFile(target))
        self._scrubber.load_file(target)

    def _on_media_status(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.LoadedMedia and self._pending_seek_ms is not None:
            pos = self._pending_seek_ms
            resume = self._resume_after_switch
            self._pending_seek_ms = None
            self._resume_after_switch = False
            self.player.setPosition(pos)
            if resume:
                self.player.play()
                self.btn_play.setText("⏸ 暂停")

    def toggle_play(self) -> None:
        self._scrubber.stop()
        if not self.player.source().isValid():
            target = self._active_wav_path()
            if target:
                self.player.setSource(QUrl.fromLocalFile(target))
            else:
                return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play.setText("▶ 试听")
        else:
            self.player.play()
            self.btn_play.setText("⏸ 暂停")

    def _toggle_play(self) -> None:
        self.toggle_play()

    def stop_audio(self) -> None:
        self._scrubber.stop()
        self.player.stop()
        self.btn_play.setText("▶ 试听")

    def release_sources(self) -> None:
        """Stop playback and release file handles held by QMediaPlayer on Windows."""
        self._scrubber.invalidate_cache()
        self.player.stop()
        self.player.setSource(QUrl())
        self.btn_play.setText("▶ 试听")
        app = QApplication.instance()
        if app is not None:
            app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    def seek_to_ms(self, ms: int) -> None:
        clamped = max(0, min(self._duration_ms, ms) if self._duration_ms > 0 else ms)
        self.player.setPosition(clamped)
        if self._duration_ms > 0 and not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(int(round(clamped * 1000.0 / self._duration_ms)))
        self.lbl_time.setText(f"{_fmt_ms(clamped)} / {_fmt_ms(self._duration_ms)}")

    def scrub_to_ms(self, ms: int) -> None:
        """Move playhead continuously with real-time PCM audio preview while paused."""
        clamped = max(0, min(self._duration_ms, ms) if self._duration_ms > 0 else ms)
        self.seek_to_ms(clamped)
        self.playheadScrubbed.emit(clamped)
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            target = self._active_wav_path()
            if target:
                self._scrubber.play_grain(clamped, target)

    def _on_slider_moved(self, val: int) -> None:
        if self._duration_ms <= 0:
            return
        ms = int(round(val * self._duration_ms / 1000.0))
        self.scrub_to_ms(ms)
        self.playheadUpdated.emit(ms)

    def _on_player_pos(self, pos_ms: int) -> None:
        if self._duration_ms > 0 and not self.seek_slider.isSliderDown():
            self.seek_slider.setValue(int(round(pos_ms * 1000.0 / self._duration_ms)))
        self.lbl_time.setText(f"{_fmt_ms(pos_ms)} / {_fmt_ms(self._duration_ms)}")
        self.playheadUpdated.emit(pos_ms)

    def _on_player_dur(self, dur_ms: int) -> None:
        self._duration_ms = max(0, dur_ms)
        self.lbl_time.setText(f"{_fmt_ms(self.player.position())} / {_fmt_ms(self._duration_ms)}")
