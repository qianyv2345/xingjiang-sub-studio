# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

ROOT = Path(SPECPATH)
datas = [(str(ROOT / 'assets'), 'assets')]
binaries = [
    (str(ROOT / 'vendor/ffmpeg/bin/ffmpeg.exe'), 'ffmpeg/bin'),
    (str(ROOT / 'vendor/ffmpeg/bin/ffprobe.exe'), 'ffmpeg/bin'),
]
hiddenimports = []
datas += collect_data_files('silero_vad')
datas += collect_data_files('jieba')
datas += collect_data_files('jieba_pyfast')
datas += collect_data_files('qwen_asr')
datas += collect_data_files('transformers')
datas += copy_metadata('torch')
datas += copy_metadata('transformers')
datas += copy_metadata('tokenizers')
datas += copy_metadata('safetensors')
datas += copy_metadata('accelerate')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('funasr')
datas += copy_metadata('modelscope')
datas += copy_metadata('qwen-asr')
datas += copy_metadata('silero-vad')
datas += copy_metadata('pyloudnorm')
datas += copy_metadata('pysubs2')
hiddenimports += collect_submodules('qwen_asr')
hiddenimports += collect_submodules('modelscope')
hiddenimports += collect_submodules('huggingface_hub')
hiddenimports += collect_submodules('pyloudnorm')
hiddenimports += collect_submodules('pysubs2')
tmp_ret = collect_all('funasr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    [str(ROOT / 'src/drama_studio/app.py')],
    pathex=[str(ROOT / 'src')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['nagisa', 'dynet38', 'gradio', 'fastapi', 'uvicorn', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick', 'matplotlib', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='XingJiangSubStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(ROOT / 'assets/app_icon.ico')],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='XingJiangSubStudio',
)
