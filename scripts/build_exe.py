"""PyInstaller build script for 星酱声幕 / XingJiangSubStudio (Windows 11 standalone package)."""

from __future__ import annotations

import argparse
import ast
from importlib import metadata as importlib_metadata
from importlib import util as importlib_util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable


_REQUIRED_DATA_MODULES = (
    "silero_vad",
    "jieba",
    "jieba_pyfast",
)
_OPTIONAL_DATA_MODULES = (
    "librosa",
    "qwen_asr",
    "transformers",
)
_REQUIRED_SUBMODULES = (
    "drama_studio",
    "modelscope",
    "pyloudnorm",
    "pysubs2",
)
_OPTIONAL_SUBMODULES = (
    "librosa",
    "qwen_asr",
    "huggingface_hub",
)
_REQUIRED_METADATA_DISTS = (
    "torch",
    "torchaudio",
    "numpy",
    "scipy",
    "soundfile",
    "onnxruntime",
    "funasr",
    "modelscope",
    "silero-vad",
    "pyloudnorm",
    "pysubs2",
)
_OPTIONAL_METADATA_DISTS = (
    "librosa",
    "lazy_loader",
    "soxr",
    "transformers",
    "tokenizers",
    "safetensors",
    "accelerate",
    "huggingface-hub",
    "qwen-asr",
)
_QWEN_REQUIRED_DISTS = ("qwen-asr", "transformers", "accelerate")


def _default_has_module(module_name: str) -> bool:
    try:
        return importlib_util.find_spec(module_name) is not None
    except Exception:
        return False


def _default_has_dist(dist_name: str) -> bool:
    try:
        importlib_metadata.distribution(dist_name)
        return True
    except importlib_metadata.PackageNotFoundError:
        return False


def build_pyinstaller_cmd(
    root: Path,
    *,
    require_qwen: bool = False,
    clean: bool = True,
    has_module: Callable[[str], bool] = _default_has_module,
    has_dist: Callable[[str], bool] = _default_has_dist,
) -> tuple[list[str], list[str]]:
    """Build PyInstaller command line while gracefully guarding optional dependencies like qwen-asr."""
    ffmpeg_bin = root / "vendor" / "ffmpeg" / "bin"
    entry = root / "src" / "drama_studio" / "app.py"
    pyinstaller_exe = root / ".venv" / "Scripts" / "pyinstaller.exe"
    icon_ico = root / "assets" / "app_icon.ico"

    warnings: list[str] = []
    if require_qwen:
        missing_qwen = [d for d in _QWEN_REQUIRED_DISTS if not has_dist(d)]
        if missing_qwen:
            raise RuntimeError(
                f"开启了 --require-qwen，但当前环境缺少可选依赖: {', '.join(missing_qwen)}。"
                "请先执行 `pip install -e .[qwen,dev]` 或 `uv sync --extra qwen --extra dev`。"
            )

    cmd = [
        str(pyinstaller_exe),
        "--noconfirm",
    ]
    if clean:
        cmd.append("--clean")
    cmd.extend([
        "--windowed",
        "--name", "XingJiangSubStudio",
        "--specpath", str(root / "build"),
        "--icon", str(icon_ico),
        "--paths", str(root / "src"),
        # Bundle LGPL FFmpeg + ffprobe & app icon
        "--add-binary", f"{ffmpeg_bin / 'ffmpeg.exe'};ffmpeg/bin",
        "--add-binary", f"{ffmpeg_bin / 'ffprobe.exe'};ffmpeg/bin",
        "--add-data", f"{root / 'assets'};assets",
    ])

    for mod in _REQUIRED_DATA_MODULES:
        cmd.extend(["--collect-data", mod])
    for mod in _OPTIONAL_DATA_MODULES:
        if has_module(mod):
            cmd.extend(["--collect-data", mod])
        else:
            warnings.append(f"可选模块 `{mod}` 未安装，跳过 --collect-data {mod}")

    cmd.extend(["--collect-all", "funasr"])

    for mod in _REQUIRED_SUBMODULES:
        cmd.extend(["--collect-submodules", mod])
    for mod in _OPTIONAL_SUBMODULES:
        if has_module(mod):
            cmd.extend(["--collect-submodules", mod])
        else:
            warnings.append(f"可选模块 `{mod}` 未安装，跳过 --collect-submodules {mod}")

    for dist in _REQUIRED_METADATA_DISTS:
        if has_dist(dist):
            cmd.extend(["--copy-metadata", dist])
        else:
            warnings.append(f"依赖元数据 `{dist}` 未找到，跳过 --copy-metadata {dist}")

    for dist in _OPTIONAL_METADATA_DISTS:
        if has_dist(dist):
            cmd.extend(["--copy-metadata", dist])
        else:
            warnings.append(f"可选依赖 `{dist}` 未安装，跳过 --copy-metadata {dist}")

    # Exclude unused heavyweight modules
    for excl in (
        "nagisa",
        "dynet38",
        "gradio",
        "fastapi",
        "uvicorn",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "matplotlib",
        "tkinter",
        # PyInstaller's initialize_modgraph mutates excludes with '__main__' in-place;
        # declaring it upfront prevents spurious 'excludes changed' cache misses.
        "__main__",
    ):
        cmd.extend(["--exclude-module", excl])

    cmd.append(str(entry))
    return cmd, warnings


def _load_analysis_toc(toc_path: Path) -> tuple[list[Any], list[Any], list[Any], list[Any], set[str]]:
    """Load scripts, pure, binaries, datas TOC lists and known modules from Analysis-00.toc."""
    raw = ast.literal_eval(toc_path.read_text(encoding="utf-8"))
    if not isinstance(raw, tuple) or len(raw) < 19:
        raise ValueError("Invalid Analysis-00.toc structure")
    scripts_toc = list(raw[13])
    pure_toc = list(raw[14])
    binaries_toc = list(raw[15])
    datas_toc = list(raw[18])
    extra_outside = set(raw[19]) if len(raw) > 19 and isinstance(raw[19], (list, tuple, set)) else set()
    known_modules = {str(entry[0]) for entry in pure_toc if len(entry) >= 1} | {str(m) for m in extra_outside}
    # Include compiled C extensions (.pyd) such as PySide6.QtCore / PySide6.QtWidgets
    for entry in binaries_toc:
        if len(entry) >= 1 and str(entry[0]).lower().endswith(".pyd"):
            stem = str(entry[0]).replace("\\", "/")[:-4]
            segs = stem.split("/")
            segs[-1] = segs[-1].split(".", 1)[0]
            known_modules.add(".".join(segs))
    return scripts_toc, pure_toc, binaries_toc, datas_toc, known_modules


def _extract_non_stdlib_imports(py_file: Path) -> set[str]:
    """Return dotted module names imported via non-relative statements outside stdlib and drama_studio."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module]
        else:
            continue
        for mod in mods:
            top = mod.split(".", 1)[0]
            if top not in stdlib and top != "drama_studio":
                found.add(mod)
    return found


def _can_use_cached_relink(root: Path, *, require_qwen: bool) -> tuple[bool, str]:
    """Check whether Analysis-00.toc + dist/_internal can be safely reused for fast incremental relink."""
    work_dir = root / "build" / "XingJiangSubStudio"
    toc_path = work_dir / "Analysis-00.toc"
    dist_app_dir = root / "dist" / "XingJiangSubStudio"
    dist_exe = dist_app_dir / "XingJiangSubStudio.exe"
    internal_dir = dist_app_dir / "_internal"

    for req_path, label in (
        (toc_path, "cached Analysis-00.toc missing"),
        (dist_exe, "dist EXE missing"),
        (internal_dir, "dist _internal missing"),
    ):
        if not req_path.exists():
            return False, label

    toc_mtime = toc_path.stat().st_mtime
    stamp_path = work_dir / ".relink_stamp.ok"
    deps_baseline = max(toc_mtime, stamp_path.stat().st_mtime) if stamp_path.is_file() else toc_mtime
    for meta_file in (
        root / "scripts" / "build_exe.py",
        root / "pyproject.toml",
        root / ".venv" / "Lib" / "site-packages",
    ):
        if meta_file.exists() and meta_file.stat().st_mtime > deps_baseline:
            return False, f"{meta_file.name} updated after last full Analysis"

    try:
        scripts_toc, pure_toc, binaries_toc, datas_toc, known_modules = _load_analysis_toc(toc_path)
    except Exception as exc:
        return False, f"failed to read Analysis-00.toc ({exc})"

    if require_qwen and "qwen_asr" not in known_modules:
        return False, "cached build lacks qwen_asr"

    work_prefix = os.path.normcase(os.path.abspath(work_dir)) + os.sep
    for entry in datas_toc:
        if len(entry) >= 2:
            src_norm = os.path.normcase(os.path.abspath(entry[1]))
            if src_norm.startswith(work_prefix) and not os.path.exists(entry[1]):
                return False, f"missing cached build resource {os.path.basename(entry[1])}"

    known_data_srcs = {
        os.path.normcase(os.path.abspath(entry[1]))
        for entry in (binaries_toc + datas_toc)
        if len(entry) >= 2
    }
    assets_dir = root / "assets"
    if assets_dir.is_dir():
        for asset in assets_dir.rglob("*"):
            if asset.is_file() and os.path.normcase(os.path.abspath(asset)) not in known_data_srcs:
                return False, f"new asset file added: {asset.name}"

    src_pkg_dir = root / "src" / "drama_studio"
    src_prefix = os.path.normcase(os.path.abspath(src_pkg_dir)) + os.sep
    cached_src_files = {
        os.path.normcase(os.path.abspath(entry[1]))
        for entry in (pure_toc + scripts_toc)
        if len(entry) >= 2 and os.path.normcase(os.path.abspath(entry[1])).startswith(src_prefix)
    }
    current_src_paths = [p for p in src_pkg_dir.rglob("*.py") if p.is_file()] if src_pkg_dir.is_dir() else []
    current_src_files = {os.path.normcase(os.path.abspath(p)) for p in current_src_paths}
    if not current_src_files or cached_src_files != current_src_files:
        return False, "source files added or removed under src/drama_studio"

    for py_path in current_src_paths:
        if py_path.stat().st_mtime <= toc_mtime:
            continue
        try:
            imports = _extract_non_stdlib_imports(py_path)
        except Exception as exc:
            return False, f"AST parse failed on {py_path.name} ({exc})"
        untracked = sorted(mod for mod in imports if mod not in known_modules)
        if untracked:
            return False, f"new dependency {', '.join(untracked)} in {py_path.name}"

    return True, "cached module graph up to date"


def _sync_incremental_dist_files(root: Path, toc_path: Path, dist_app_dir: Path) -> int:
    """Copy rebuilt EXE and any modified binaries/datas into dist/XingJiangSubStudio without wiping torch/CUDA."""
    built_exe = root / "build" / "XingJiangSubStudio" / "XingJiangSubStudio.exe"
    if not built_exe.is_file():
        raise FileNotFoundError(f"Missing rebuilt EXE at {built_exe}")
    shutil.copy2(built_exe, dist_app_dir / "XingJiangSubStudio.exe")

    _, _, binaries_toc, datas_toc, _ = _load_analysis_toc(toc_path)
    internal_dir = dist_app_dir / "_internal"
    copied = 0
    for entry in binaries_toc + datas_toc:
        if len(entry) < 3 or entry[2] in ("DEPENDENCY", "SYMLINK"):
            continue
        src = Path(entry[1])
        if not src.is_file():
            continue
        dst = internal_dir / entry[0]
        src_st = src.stat()
        if dst.is_file():
            dst_st = dst.stat()
            if dst_st.st_size == src_st.st_size and src_st.st_mtime <= dst_st.st_mtime + 1e-6:
                continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def _invalidate_stale_exe_if_icon_changed(root: Path) -> None:
    """Force PyInstaller EXE rebuild when app_icon.ico is newer than EXE-00.toc (PyInstaller only checks icon path string)."""
    icon_ico = root / "assets" / "app_icon.ico"
    exe_toc = root / "build" / "XingJiangSubStudio" / "EXE-00.toc"
    if icon_ico.is_file() and exe_toc.is_file() and icon_ico.stat().st_mtime > exe_toc.stat().st_mtime:
        exe_toc.unlink()


def _run_cached_incremental_relink(root: Path) -> int:
    """Rebuild PYZ + EXE from cached Analysis-00.toc and sync only changed files into dist/."""
    pyinstaller_exe = root / ".venv" / "Scripts" / "pyinstaller.exe"
    toc_path = root / "build" / "XingJiangSubStudio" / "Analysis-00.toc"
    icon_ico = root / "assets" / "app_icon.ico"
    spec_dir = root / "build" / "_incremental"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / "XingJiangSubStudio.spec"

    icon_arg = f"[{icon_ico.as_posix()!r}]" if icon_ico.is_file() else "None"
    spec_text = (
        "import ast\n"
        f"with open({str(toc_path)!r}, 'r', encoding='utf-8') as _f:\n"
        "    _toc = ast.literal_eval(_f.read())\n"
        "_scripts, _pure = _toc[13], _toc[14]\n"
        "pyz = PYZ(_pure)\n"
        "exe = EXE(\n"
        "    pyz,\n"
        "    _scripts,\n"
        "    [],\n"
        "    exclude_binaries=True,\n"
        "    name='XingJiangSubStudio',\n"
        "    debug=False,\n"
        "    bootloader_ignore_signals=False,\n"
        "    strip=False,\n"
        "    upx=True,\n"
        "    console=False,\n"
        "    disable_windowed_traceback=False,\n"
        "    argv_emulation=False,\n"
        "    target_arch=None,\n"
        "    codesign_identity=None,\n"
        "    entitlements_file=None,\n"
        f"    icon={icon_arg},\n"
        ")\n"
    )
    spec_path.write_text(spec_text, encoding="utf-8")
    _invalidate_stale_exe_if_icon_changed(root)

    cmd = [
        str(pyinstaller_exe),
        "--noconfirm",
        "--distpath",
        str(root / "dist"),
        "--workpath",
        str(root / "build"),
        str(spec_path),
    ]
    print("[build_exe] INFO: Fast incremental relink (reusing Analysis-00.toc -> PYZ + EXE)...")
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(root))
    if proc.returncode != 0:
        return proc.returncode

    copied = _sync_incremental_dist_files(root, toc_path, root / "dist" / "XingJiangSubStudio")
    print(f"[build_exe] INFO: Synced updated EXE and {copied} changed resource file(s).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build XingJiangSubStudio standalone Windows package.")
    parser.add_argument(
        "--require-qwen",
        action="store_true",
        help="Fail build if optional Qwen dependencies (qwen-asr/transformers/accelerate) are missing.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Reuse PyInstaller build/ & dist/ caches without --clean for fast development rebuilds.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    ffmpeg_bin = root / "vendor" / "ffmpeg" / "bin"
    if not (ffmpeg_bin / "ffmpeg.exe").is_file() or not (ffmpeg_bin / "ffprobe.exe").is_file():
        print("ERROR: vendor/ffmpeg/bin/ffmpeg.exe or ffprobe.exe missing!", file=sys.stderr)
        return 1

    dist_dir = root / "dist"
    build_dir = root / "build"
    clean_build = not args.incremental

    try:
        cmd, warnings = build_pyinstaller_cmd(root, require_qwen=args.require_qwen, clean=clean_build)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for w in warnings:
        print(f"[build_exe] INFO: {w}")

    ran_relink = False
    if clean_build:
        # Clean previous build artifacts (including legacy DramaSubStudio bundle if present)
        for d in (dist_dir / "XingJiangSubStudio", dist_dir / "DramaSubStudio", build_dir):
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
    else:
        can_relink, reason = _can_use_cached_relink(root, require_qwen=args.require_qwen)
        if can_relink:
            rc = _run_cached_incremental_relink(root)
            if rc != 0:
                return rc
            ran_relink = True
        else:
            print(f"[build_exe] INFO: --incremental full Analysis needed ({reason}).")

    if not ran_relink:
        _invalidate_stale_exe_if_icon_changed(root)
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(root))
        if proc.returncode != 0:
            return proc.returncode
        work_app_dir = build_dir / "XingJiangSubStudio"
        if work_app_dir.is_dir():
            (work_app_dir / ".relink_stamp.ok").write_text("ok\n", encoding="utf-8")

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
