"""Application entrypoint with Windows DPI scaling, bundled FFmpeg PATH injection, and theme support."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from drama_studio import __brand_name__, __version__
from drama_studio.config import ConfigStore
from drama_studio.gui.main_window import MainWindow
from drama_studio.gui.theme import build_studio_qss, resolve_asset_path
from drama_studio.media.ffmpeg_wrapper import find_ffmpeg_binaries


def configure_runtime_environment() -> None:
    """Inject bundled FFmpeg into PATH, apply Windows Qwen safeguard, and prevent CPU thread starvation."""
    from drama_studio.align.runtime import ensure_qwen_windows_safe

    # Cap BLAS/OpenMP worker threads so heavy DSP/ML workers cannot starve the Windows Qt UI thread
    safe_threads = str(min(4, max(2, (os.cpu_count() or 4) // 2)))
    os.environ.setdefault("OMP_NUM_THREADS", safe_threads)
    os.environ.setdefault("MKL_NUM_THREADS", safe_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", safe_threads)

    ensure_qwen_windows_safe()
    warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"jieba.*")
    warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"pydub.*")

    try:
        ff_bin, _ = find_ffmpeg_binaries()
        ff_dir = str(ff_bin.parent)
        cur_path = os.environ.get("PATH", "")
        if ff_dir not in cur_path:
            os.environ["PATH"] = ff_dir + os.pathsep + cur_path
    except FileNotFoundError:
        pass


def main() -> int:
    configure_runtime_environment()

    smoke_test = "--smoke-test" in sys.argv
    smoke_e2e = "--smoke-e2e" in sys.argv
    smoke_tmp_dir = None
    if smoke_test or smoke_e2e:
        import tempfile

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        smoke_tmp_dir = tempfile.TemporaryDirectory(prefix="drama_smoke_")

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("XingJiangSubStudio")
    app.setApplicationDisplayName(__brand_name__)
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    store = ConfigStore(base_dir=Path(smoke_tmp_dir.name)) if smoke_tmp_dir else ConfigStore()
    init_theme = store.preferences.get("theme_mode", "light")
    app.setProperty("theme_mode", init_theme)
    app.setStyleSheet(build_studio_qss(init_theme))

    for icon_name in ("app_icon.ico", "app_icon.png"):
        ic = resolve_asset_path(icon_name)
        if ic is not None and ic.is_file():
            app.setWindowIcon(QIcon(str(ic)))
            break

    win = MainWindow(store)
    if smoke_test or smoke_e2e:
        # Verify frozen binary dependencies (FFmpeg, ffprobe, Silero VAD, jieba, Qt widgets)
        ff, fp = find_ffmpeg_binaries()
        assert ff.is_file() and fp.is_file(), "FFmpeg/ffprobe missing in bundle"
        from drama_studio.audio.vad import get_vad_session
        get_vad_session()
        from drama_studio.nlp.segmenter import tokenize_with_pos
        toks = tokenize_with_pos("测试中文分词")
        assert len(toks) > 0
        from importlib import util as importlib_util
        if importlib_util.find_spec("librosa") is not None:
            import librosa
            assert hasattr(librosa, "resample")
        if importlib_util.find_spec("qwen_asr") is not None:
            from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
            assert Qwen3ASRModel is not None and Qwen3ForcedAligner is not None
        from drama_studio.cache import stable_param_hash
        from drama_studio.nlp.script_matcher import match_asr_to_script
        assert callable(stable_param_hash) and callable(match_asr_to_script)
        win.template_panel.chars_slider.setValue(14)
        win.template_panel.btn_2line.click()
        assert win.template_panel.current_template().max_lines == 2

        if smoke_e2e:
            import tempfile
            from drama_studio.config import default_model_cache_dir
            from drama_studio.align.model_manager import is_model_cached
            from drama_studio.models import AlignmentConfig, AudioLevelingConfig, EpisodeTask, ExportConfig, SubtitleTemplate
            from drama_studio.pipeline import process_single_episode

            idx = sys.argv.index("--smoke-e2e")
            vid_p = sys.argv[idx + 1]
            srt_p = sys.argv[idx + 2]
            out_p = sys.argv[idx + 3]
            cdir = default_model_cache_dir()
            with tempfile.TemporaryDirectory() as td:
                for be in ("funasr", "qwen3"):
                    if (be == "funasr" and not is_model_cached("funasr_fa", cdir)) or (
                        be == "qwen3" and not is_model_cached("qwen3_aligner", cdir)
                    ):
                        continue
                    t = EpisodeTask(task_id=f"e2e_{be}", episode_label=f"第01集_{be}", video_path=vid_p, srt_path=srt_p)
                    res = process_single_episode(
                        t,
                        SubtitleTemplate.preset_portrait_9_16(),
                        AudioLevelingConfig.from_mode("standard"),
                        AlignmentConfig(backend=be),
                        ExportConfig(output_dir=out_p),
                        cdir,
                        Path(td) / be,
                    )
                    assert res.text_integrity_ok and len(res.processed_cues) >= 5
                    print("SMOKE_E2E_OK", be, res.aligner_used, len(res.processed_cues))

        print("SMOKE_TEST_OK", ff, fp)
        win.audio_panel.release_sources()
        win.close()
        del win
        app.processEvents()
        if smoke_tmp_dir is not None:
            smoke_tmp_dir.cleanup()
        return 0

    win.show()
    rc = app.exec()
    win.audio_panel.release_sources()
    del win
    app.processEvents()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
