"""PyInstaller build script for 星酱声幕 / XingJiangSubStudio (Windows 11 standalone package)."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ffmpeg_bin = root / "vendor" / "ffmpeg" / "bin"
    if not (ffmpeg_bin / "ffmpeg.exe").is_file() or not (ffmpeg_bin / "ffprobe.exe").is_file():
        print("ERROR: vendor/ffmpeg/bin/ffmpeg.exe or ffprobe.exe missing!", file=sys.stderr)
        return 1

    entry = root / "src" / "drama_studio" / "app.py"
    dist_dir = root / "dist"
    build_dir = root / "build"

    # Clean previous build artifacts (including legacy DramaSubStudio bundle if present)
    for d in (dist_dir / "XingJiangSubStudio", dist_dir / "DramaSubStudio", build_dir):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    pyinstaller_exe = root / ".venv" / "Scripts" / "pyinstaller.exe"
    icon_ico = root / "assets" / "app_icon.ico"
    cmd = [
        str(pyinstaller_exe),
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name", "XingJiangSubStudio",
        "--icon", str(icon_ico),
        "--paths", str(root / "src"),
        # Bundle LGPL FFmpeg + ffprobe & app icon
        "--add-binary", f"{ffmpeg_bin / 'ffmpeg.exe'};ffmpeg/bin",
        "--add-binary", f"{ffmpeg_bin / 'ffprobe.exe'};ffmpeg/bin",
        "--add-data", f"{root / 'assets'};assets",
        # Bundle required data files
        "--collect-data", "silero_vad",
        "--collect-data", "jieba",
        "--collect-data", "jieba_pyfast",
        "--collect-data", "qwen_asr",
        "--collect-data", "transformers",
        "--collect-all", "funasr",
        "--collect-submodules", "qwen_asr",
        "--collect-submodules", "modelscope",
        "--collect-submodules", "huggingface_hub",
        "--collect-submodules", "pyloudnorm",
        "--collect-submodules", "pysubs2",
        "--copy-metadata", "torch",
        "--copy-metadata", "transformers",
        "--copy-metadata", "tokenizers",
        "--copy-metadata", "safetensors",
        "--copy-metadata", "accelerate",
        "--copy-metadata", "huggingface-hub",
        "--copy-metadata", "funasr",
        "--copy-metadata", "modelscope",
        "--copy-metadata", "qwen-asr",
        "--copy-metadata", "silero-vad",
        "--copy-metadata", "pyloudnorm",
        "--copy-metadata", "pysubs2",
        # Exclude unused heavyweight modules
        "--exclude-module", "nagisa",
        "--exclude-module", "dynet38",
        "--exclude-module", "gradio",
        "--exclude-module", "fastapi",
        "--exclude-module", "uvicorn",
        "--exclude-module", "PySide6.QtWebEngineCore",
        "--exclude-module", "PySide6.QtWebEngineWidgets",
        "--exclude-module", "PySide6.QtWebEngineQuick",
        "--exclude-module", "matplotlib",
        "--exclude-module", "tkinter",
        str(entry),
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(root))
    if proc.returncode != 0:
        return proc.returncode

    # Copy documentation and license notices into dist/XingJiangSubStudio
    out_app_dir = dist_dir / "XingJiangSubStudio"
    for doc in ("README.md", "LICENSE", "THIRD_PARTY_LICENSES.md"):
        doc_p = root / doc
        if doc_p.is_file():
            shutil.copy2(doc_p, out_app_dir / doc)

    print("Build succeeded ->", out_app_dir / "XingJiangSubStudio.exe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
