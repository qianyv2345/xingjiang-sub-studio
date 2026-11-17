"""Offscreen PySide6 GUI tests for template slider, 1L/2L toggle, import, undo/redo, and persistence."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from drama_studio.app import configure_runtime_environment
from drama_studio.config import ConfigStore
from drama_studio.gui.main_window import MainWindow
from drama_studio.models import SubtitleCue


def _app() -> QApplication:
    configure_runtime_environment()
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _cleanup_window(win: MainWindow, app: QApplication) -> None:
    app.removeEventFilter(win)
    win.audio_panel.release_sources()
    win.close()
    win.deleteLater()
    app.processEvents()
    app.processEvents()


def test_template_slider_1l_2l_and_persistence(tmp_path: Path) -> None:
    app = _app()
    store = ConfigStore(base_dir=tmp_path / "cfg")
    win = MainWindow(store=store)

    tp = win.template_panel
    # Default 9:16 preset (default 3 frames head/tail breathing room & anti-flash bridging)
    assert tp.chars_slider.value() == 9
    assert tp.chars_spin.value() == 9
    assert tp.btn_1line.isChecked()
    assert tp.combo_padding_unit.currentData() == "frames"
    assert tp.spin_lead_offset.value() == 3
    assert tp.spin_tail_offset.value() == 3
    assert tp.spin_min_gap.value() == 20
    assert tp.chk_bridge_gaps.isChecked() is True
    assert tp.spin_bridge_gap.value() == 550

    # Customize padding frames (e.g. 5 front, 2 back) and toggle to ms mode
    tp.set_video_fps(30.0)
    tp.spin_lead_offset.setValue(6)
    assert tp.current_template().lead_frames == 6
    assert tp.current_template().resolve_padding_ms(30.0) == (200, 100)
    assert "30fps" in tp.lbl_padding_hint.text()

    ms_idx = tp.combo_padding_unit.findData("ms")
    tp.combo_padding_unit.setCurrentIndex(ms_idx)
    assert tp.current_template().padding_unit == "ms"
    tp.spin_tail_offset.setValue(180)
    assert tp.current_template().tail_offset_ms == 180

    # Switch back to frames mode with default 3 frames before testing project persistence
    fr_idx = tp.combo_padding_unit.findData("frames")
    tp.combo_padding_unit.setCurrentIndex(fr_idx)
    tp.spin_lead_offset.setValue(3)
    tp.spin_tail_offset.setValue(3)

    tp.chk_bridge_gaps.setChecked(False)
    assert tp.spin_bridge_gap.isEnabled() is False
    assert tp.current_template().bridge_sentence_gaps is False
    tp.chk_bridge_gaps.setChecked(True)
    assert tp.spin_bridge_gap.isEnabled() is True

    # Drag slider to 16 chars
    tp.chars_slider.setValue(16)
    assert tp.chars_spin.value() == 16
    assert "16" in tp.chars_live_label.text()

    # Switch to 2-line segmented button
    tp.btn_2line.click()
    assert tp.current_template().max_lines == 2
    assert tp.current_template().max_chars_per_line == 16

    # Verify live preview renders without error
    assert tp.preview_box.sub_label.text().strip() != ""

    # Create new project & switch back -> settings must persist independently
    store.create_project("古装竖屏项目")
    win._load_active_project_to_ui()
    tp.apply_aspect_preset("16:9")
    win._save_project_settings()
    assert tp.chars_slider.value() == 20

    store.switch_project("default")
    win._load_active_project_to_ui()
    assert tp.chars_slider.value() == 16
    assert tp.current_template().max_lines == 2

    # Save custom template & duplicate
    custom = tp.current_template().clone("我的短剧模板")
    saved = store.add_or_update_template(custom)
    assert any(t.name == "我的短剧模板" for t in store.templates)
    copied = store.duplicate_template(saved.template_id)
    assert copied is not None

    _cleanup_window(win, app)
    del win
    app.processEvents()


def test_import_and_undo_redo_in_gui(tmp_path: Path) -> None:
    app = _app()
    store = ConfigStore(base_dir=tmp_path / "cfg2")
    win = MainWindow(store=store)

    fx = Path(r"d:\自动化\tests\fixtures")
    win.import_paths([str(fx / "第01集_粗剪.mp4"), str(fx / "第01集_粗剪.srt")])
    assert len(win.tasks) == 1
    assert win.tasks[0].episode_label == "第01集"

    # Wait briefly for probe worker to finish if running
    if win._probe_worker is not None:
        win._probe_worker.wait(5000)

    # Late drop of .md dialogue script matching 第01集 binds cleanly without error dialog
    ep1_md = tmp_path / "第01集.md"
    ep1_md.write_text("顾寒川、苏晚晴\n\n第一句对白。\n", encoding="utf-8")
    win.import_paths([str(ep1_md)])
    assert win.tasks[0].script_path == str(ep1_md)
    assert "已绑定" in win.import_panel.lbl_script_status.text()

    # Test Undo/Redo on manual gain points
    from drama_studio.models import ManualGainPoint

    pts_after = [ManualGainPoint(time_ms=1500, gain_db=4.5)]
    win._on_manual_points_edited("添加测试增益点", [], pts_after)
    assert len(win.audio_panel.current_config().manual_points) == 1
    win.undo_stack.undo()
    assert len(win.audio_panel.current_config().manual_points) == 0
    win.undo_stack.redo()
    assert len(win.audio_panel.current_config().manual_points) == 1

    # Test Light / Dark / Sakura Pink theme switching & persistence
    assert store.preferences.get("theme_mode", "light") == "light"
    win.apply_theme("pink", save=True)
    assert store.preferences["theme_mode"] == "pink"
    assert win.combo_theme.currentData() == "pink"
    assert win.waveform_widget.canvas.theme_mode == "pink"

    win.apply_theme("dark", save=True)
    assert store.preferences["theme_mode"] == "dark"
    assert win.combo_theme.currentData() == "dark"

    win.toggle_theme()
    assert store.preferences["theme_mode"] == "pink"

    win.apply_theme("light", save=True)
    reloaded = ConfigStore(base_dir=tmp_path / "cfg2")
    assert reloaded.preferences["theme_mode"] == "light"

    # Test Alt+Wheel zoom, red playhead drag inside waveform, live scrub, & Space shortcut
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent

    canvas = win.waveform_widget.canvas
    cues = [
        SubtitleCue(index=1, start_ms=400, end_ms=2400, raw_text="第一句对白"),
        SubtitleCue(index=2, start_ms=2600, end_ms=5200, raw_text="第二句对白"),
    ]
    canvas.set_data(
        duration_ms=8000,
        wf_times_ms=[i * 80 for i in range(100)],
        wf_before=[0.4] * 100,
        wf_after=[0.6] * 100,
        gain_times_ms=[i * 80 for i in range(100)],
        gain_db=[1.0] * 100,
        vad_spans_ms=[(400, 5200)],
        anomalies=[],
        cues=cues,
        manual_points=[],
    )
    canvas.resize(700, 180)
    win.show()
    app.processEvents()

    old_pps = canvas.px_per_sec
    we = QWheelEvent(
        QPointF(200.0, 90.0),
        QPointF(200.0, 90.0),
        QPointF(0, 0).toPoint(),
        QPointF(0, 120).toPoint(),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.AltModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(we)
    assert canvas.px_per_sec > old_pps

    # Grab red playhead right in the middle of the waveform (y=80) and drag to 3000ms
    scrubbed_positions: list[int] = []
    canvas.playheadScrubbed.connect(scrubbed_positions.append)
    px_start = canvas._ms_to_x(0)
    press_ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(px_start, 80.0),
        QPointF(px_start, 80.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(press_ev)
    drag_x = canvas._ms_to_x(3000)
    move_ev = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(drag_x, 80.0),
        QPointF(drag_x, 80.0),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseMoveEvent(move_ev)
    rel_ev = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(drag_x, 80.0),
        QPointF(drag_x, 80.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseReleaseEvent(rel_ev)
    assert len(scrubbed_positions) >= 2
    assert abs(canvas.playhead_ms - 3000) < 150
    assert len(canvas.manual_points) == 0  # dragging playhead must NOT drop gain points
    assert "第二句对白" in win.template_panel.preview_box.sample_input.text()

    # Advanced params drawer unclipped check
    win.audio_panel.btn_adv_toggle.click()
    app.processEvents()
    assert win.audio_panel.spin_target_lufs.minimumHeight() >= 28

    # Spacebar shortcut check (must invoke play/pause toggle without typing into inputs)
    toggled = []
    orig_toggle = win.audio_panel.toggle_play
    win.audio_panel.toggle_play = lambda: toggled.append(True)
    try:
        ke = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        assert win.eventFilter(canvas, ke) is True
        assert toggled == [True]
    finally:
        win.audio_panel.toggle_play = orig_toggle

    # Real-time PCM scrubber loads WAV bytes cleanly into memory and clears on release
    import numpy as np
    import soundfile as sf

    wav_p = tmp_path / "scrub_test.wav"
    sf.write(str(wav_p), np.zeros((48000, 1), dtype=np.float32), 48000, subtype="PCM_24")
    win.audio_panel.set_audio_sources(str(wav_p), str(wav_p), None)
    win.audio_panel.scrub_to_ms(400)
    assert str(wav_p) in win.audio_panel._scrubber._pcm_cache
    win.audio_panel.release_sources()
    assert len(win.audio_panel._scrubber._pcm_cache) == 0

    # Video orientation auto-detection: landscape -> 16:9 (20 chars), portrait -> 9:16 (9 chars), no video -> keep default
    from drama_studio.models import EpisodeTask, MediaInfo, Orientation

    land_task = EpisodeTask(task_id="t_land", episode_label="第99集", video_path="land.mp4")
    win.tasks.append(land_task)
    win.current_task_id = "t_land"
    win._on_media_probed(
        "t_land",
        MediaInfo(
            path="land.mp4",
            file_name="land.mp4",
            format_name="mov,mp4",
            duration_ms=10000,
            width=1920,
            height=1080,
            fps=30.0,
            fps_str="30/1",
            aspect_ratio="16:9",
            orientation=Orientation.LANDSCAPE,
        ),
    )
    assert win.template_panel.current_template().aspect_preset == "16:9"
    assert win.template_panel.chars_slider.value() == 20

    # Remove all tasks -> restores default 9:16
    win.tasks = [land_task]
    win._on_remove_episode("t_land")
    assert win.template_panel.current_template().aspect_preset == "9:16"
    assert win.template_panel.chars_slider.value() == 9

    # Brand title, avatar icon & ModelManagerDialog action buttons unclipped check
    assert "星酱声幕" in win.windowTitle()
    from PySide6.QtWidgets import QLabel

    avatar = win.findChild(QLabel, "BrandAvatar")
    assert avatar is not None and not avatar.pixmap().isNull()

    from drama_studio.gui.widgets.model_manager_dialog import ModelManagerDialog

    dlg = ModelManagerDialog(store, win)
    for row in range(dlg.table.rowCount()):
        assert dlg.table.rowHeight(row) >= 46
    dlg.close()
    dlg.deleteLater()
    del dlg
    app.processEvents()

    _cleanup_window(win, app)
    del win
    app.processEvents()
