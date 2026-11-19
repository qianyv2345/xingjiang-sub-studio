"""Model download, disk & VRAM inspection, and stage cache management dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from drama_studio.align.model_manager import (
    MODEL_REGISTRY,
    check_model_vram_risk,
    delete_cached_model,
    download_model,
    get_cached_model_size_mb,
    get_total_models_size_mb,
    is_model_cached,
)
from drama_studio.align.runtime import (
    get_disk_free_mb,
    get_gpu_vram_info,
    get_system_memory_info,
)
from drama_studio.cache import clear_stage_cache, get_stage_cache_size_mb
from drama_studio.config import ConfigStore
from drama_studio.gui.theme import get_palette
from drama_studio.gui.widgets.themed_controls import apply_win11_dwm_chrome


class ModelDownloadWorker(QThread):
    logLine = Signal(str)
    progressPct = Signal(int)
    finishedOk = Signal(str)
    failed = Signal(str, str)  # model_id, error

    def __init__(self, model_id: str, cache_dir: Path, use_modelscope: bool) -> None:
        super().__init__()
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.use_modelscope = use_modelscope

    def run(self) -> None:
        try:
            download_model(
                self.model_id,
                self.cache_dir,
                use_modelscope=self.use_modelscope,
                progress_cb=lambda m: self.logLine.emit(m),
                percent_cb=lambda pct: self.progressPct.emit(pct),
            )
            self.finishedOk.emit(self.model_id)
        except Exception as exc:
            self.failed.emit(self.model_id, str(exc))


class ModelManagerDialog(QDialog):
    """GUI dialog for downloading, verifying, and cleaning AI models and stage cache."""

    def __init__(
        self,
        store: ConfigStore,
        parent: QWidget | None = None,
        *,
        highlight_model_id: str = "",
    ) -> None:
        super().__init__(parent)
        self.store = store
        self._highlight_model_id = highlight_model_id
        self._worker: ModelDownloadWorker | None = None
        self._last_failed_model_id: str = ""
        self.setWindowTitle("模型下载与硬盘缓存管理")
        self.resize(1040, 580)
        self.setMinimumSize(860, 480)
        self._build_ui()
        self.refresh_table()
        pal = get_palette(self.store.preferences.get("theme_mode", "light"))
        apply_win11_dwm_chrome(
            self,
            border_hex=pal.line,
            is_dark=pal.is_dark,
            round_corners=False,
            caption_hex=pal.header_bg,
            text_hex=pal.text,
        )

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setSpacing(8)

        # Hardware & Disk Banner
        vram = get_gpu_vram_info()
        ram = get_system_memory_info()
        ram_str = (
            f"系统内存 {ram['avail_mb'] / 1024:.1f} / {ram['total_mb'] / 1024:.1f} GB"
            if ram["total_mb"] > 0
            else "系统内存: 未知"
        )
        if vram["available"]:
            hw_txt = (
                f"GPU：{vram['device_name']}（可用显存 {vram['free_mb']} MB / 共 {vram['total_mb']} MB）"
                f" · {ram_str} · 显存不足时支持自动降级或 CPU 回退"
            )
        else:
            hw_txt = f"未检测到 CUDA GPU，将使用 CPU 执行 · {ram_str}"
        self.lbl_hw = QLabel(hw_txt)
        self.lbl_hw.setObjectName("HardwareBanner")
        self.lbl_hw.setWordWrap(True)
        lay.addWidget(self.lbl_hw)

        # Model directory selector
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("模型存储目录:"))
        self.edit_dir = QLineEdit(self.store.preferences["model_cache_dir"])
        self.edit_dir.setReadOnly(True)
        btn_dir = QPushButton("更改目录")
        btn_dir.clicked.connect(self._change_model_dir)
        self.lbl_model_disk = QLabel("")
        self.lbl_model_disk.setObjectName("MutedLabel")
        self.chk_ms = QCheckBox("优先 ModelScope 镜像")
        self.chk_ms.setChecked(bool(self.store.preferences.get("use_modelscope_mirror", True)))
        self.chk_ms.toggled.connect(self._on_ms_toggled)
        dir_row.addWidget(self.edit_dir, 1)
        dir_row.addWidget(btn_dir)
        dir_row.addWidget(self.lbl_model_disk)
        dir_row.addWidget(self.chk_ms)
        lay.addLayout(dir_row)

        # Temp stage cache directory selector & cleanup
        cache_row = QHBoxLayout()
        cache_row.addWidget(QLabel("临时缓存目录:"))
        self.edit_work_dir = QLineEdit(self.store.preferences["work_cache_dir"])
        self.edit_work_dir.setReadOnly(True)
        btn_work_dir = QPushButton("更改目录")
        btn_work_dir.clicked.connect(self._change_work_dir)
        self.lbl_cache_usage = QLabel("")
        self.lbl_cache_usage.setObjectName("MutedLabel")
        self.btn_clear_cache = QPushButton("清理任务临时缓存")
        self.btn_clear_cache.clicked.connect(self._clear_work_cache)
        cache_row.addWidget(self.edit_work_dir, 1)
        cache_row.addWidget(btn_work_dir)
        cache_row.addWidget(self.lbl_cache_usage)
        cache_row.addWidget(self.btn_clear_cache)
        lay.addLayout(cache_row)

        # Model Table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["模型名称", "预计体积", "安装状态", "适用语言", "用途说明", "操作"])
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 235)
        self.table.setColumnWidth(5, 118)
        lay.addWidget(self.table, 1)

        # Download progress & retry bar
        dl_row = QHBoxLayout()
        self.dl_progress = QProgressBar()
        self.dl_progress.setRange(0, 100)
        self.dl_progress.setValue(0)
        self.dl_progress.setMaximumWidth(260)
        self.btn_retry = QPushButton("重试下载")
        self.btn_retry.setVisible(False)
        self.btn_retry.clicked.connect(self._retry_last)
        self.lbl_status = QLabel("提示：按需下载所需模型（不强制全量下载），软件升级后自动复用已有模型。")
        self.lbl_status.setObjectName("MutedLabel")
        self.lbl_status.setWordWrap(True)
        dl_row.addWidget(self.lbl_status, 1)
        dl_row.addWidget(self.dl_progress)
        dl_row.addWidget(self.btn_retry)
        lay.addLayout(dl_row)

        bottom = QHBoxLayout()
        bottom.addStretch()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        bottom.addWidget(btn_close)
        lay.addLayout(bottom)

    def _cache_dir(self) -> Path:
        return Path(self.store.preferences["model_cache_dir"])

    def _work_cache_dir(self) -> Path:
        return Path(self.store.preferences["work_cache_dir"])

    def _change_model_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择模型存储目录", str(self._cache_dir()))
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)
            self.store.preferences["model_cache_dir"] = folder
            self.store.save_prefs()
            self.edit_dir.setText(folder)
            self.refresh_table()

    def _change_work_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择临时任务缓存目录", str(self._work_cache_dir()))
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)
            self.store.preferences["work_cache_dir"] = folder
            self.store.save_prefs()
            self.edit_work_dir.setText(folder)
            self.refresh_table()

    def _clear_work_cache(self) -> None:
        wdir = self._work_cache_dir()
        sz = get_stage_cache_size_mb(wdir)
        ans = QMessageBox.question(
            self,
            "清理临时缓存",
            f"确定清理阶段临时缓存（当前约 {sz:.1f} MB）？清理后不会影响已下载的 AI 模型与已导出结果。",
        )
        if ans == QMessageBox.StandardButton.Yes:
            freed = clear_stage_cache(wdir)
            self.lbl_status.setText(f"已释放临时缓存空间 {freed:.1f} MB。")
            self.refresh_table()

    def _on_ms_toggled(self, checked: bool) -> None:
        self.store.preferences["use_modelscope_mirror"] = checked
        self.store.save_prefs()

    def refresh_table(self) -> None:
        cdir = self._cache_dir()
        wdir = self._work_cache_dir()

        models_used_mb = get_total_models_size_mb(cdir)
        models_free_gb = get_disk_free_mb(cdir) / 1024.0
        self.lbl_model_disk.setText(f"已占 {models_used_mb:.0f} MB · 剩余 {models_free_gb:.1f} GB")

        cache_used_mb = get_stage_cache_size_mb(wdir)
        cache_free_gb = get_disk_free_mb(wdir) / 1024.0
        max_cap_mb = self.store.preferences.get("cache_max_size_mb", 4096)
        self.lbl_cache_usage.setText(f"缓存 {cache_used_mb:.1f} / {max_cap_mb} MB · 剩余 {cache_free_gb:.1f} GB")

        specs = list(MODEL_REGISTRY.values())
        self.table.setRowCount(len(specs))

        pal = get_palette(self.store.preferences.get("theme_mode", "light"))
        c_ok = QColor(pal.ok_text)
        c_warn = QColor("#FBBF24" if pal.is_dark else "#D97706")
        for row, spec in enumerate(specs):
            cached = is_model_cached(spec.model_id, cdir)
            actual_mb = get_cached_model_size_mb(spec.model_id, cdir)

            name_item = QTableWidgetItem(spec.display_name)
            if spec.model_id == self._highlight_model_id:
                name_item.setForeground(QColor(pal.accent))
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(f"{spec.approx_size_mb} MB"))
            st_str = "内置已就绪" if spec.is_bundled else (f"已安装 ({actual_mb} MB)" if cached else "未下载")
            st_item = QTableWidgetItem(st_str)
            st_item.setForeground(c_ok if cached else c_warn)
            self.table.setItem(row, 2, st_item)

            lang_lbl = "中文 / English" if "en" in spec.languages else "仅中文"
            self.table.setItem(row, 3, QTableWidgetItem(f"{lang_lbl}\n({spec.license_name})"))
            self.table.setItem(row, 4, QTableWidgetItem(spec.description))

            action_w = QWidget()
            action_w.setObjectName("TransparentContainer")
            a_lay = QHBoxLayout(action_w)
            a_lay.setContentsMargins(6, 5, 6, 5)
            a_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if spec.is_bundled:
                lbl = QLabel("随软件内置")
                lbl.setObjectName("MutedLabel")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                a_lay.addWidget(lbl)
            elif cached:
                btn_del = QPushButton("删除模型")
                btn_del.setFixedHeight(30)
                btn_del.setMinimumWidth(90)
                btn_del.setStyleSheet("padding: 2px 10px;")
                btn_del.clicked.connect(lambda _=False, mid=spec.model_id: self._delete_model(mid))
                a_lay.addWidget(btn_del)
            else:
                btn_dl = QPushButton("下载模型")
                btn_dl.setObjectName("PrimaryButton")
                btn_dl.setFixedHeight(30)
                btn_dl.setMinimumWidth(90)
                btn_dl.setStyleSheet("padding: 2px 10px;")
                btn_dl.clicked.connect(lambda _=False, mid=spec.model_id: self._download_model(mid))
                a_lay.addWidget(btn_dl)
            self.table.setCellWidget(row, 5, action_w)
            self.table.setRowHeight(row, 56)

    def _download_model(self, model_id: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "下载中", "当前已有模型正在下载，请等待完成后再试。")
            return
        vram_warn = check_model_vram_risk(model_id)
        if vram_warn:
            self.lbl_status.setText(f"⚠ {vram_warn}")
        else:
            self.lbl_status.setText(f"正在准备下载 {model_id}...")

        self.btn_retry.setVisible(False)
        self.dl_progress.setValue(0)
        self._worker = ModelDownloadWorker(
            model_id,
            self._cache_dir(),
            bool(self.store.preferences.get("use_modelscope_mirror", True)),
        )
        self._worker.logLine.connect(self.lbl_status.setText)
        self._worker.progressPct.connect(self.dl_progress.setValue)
        self._worker.finishedOk.connect(self._on_dl_done)
        self._worker.failed.connect(self._on_dl_err)
        self._worker.start()

    def _retry_last(self) -> None:
        if self._last_failed_model_id:
            self._download_model(self._last_failed_model_id)

    def _on_dl_done(self, model_id: str) -> None:
        self._last_failed_model_id = ""
        self.btn_retry.setVisible(False)
        self.dl_progress.setValue(100)
        self.lbl_status.setText(f"模型 {model_id} 下载并完整性验证通过！")
        self.refresh_table()

    def _on_dl_err(self, model_id: str, err: str) -> None:
        self._last_failed_model_id = model_id
        self.btn_retry.setVisible(True)
        self.lbl_status.setText(f"下载失败 ({model_id})：{err}")
        QMessageBox.warning(self, "下载失败", err)
        self.refresh_table()

    def _delete_model(self, model_id: str) -> None:
        ans = QMessageBox.question(self, "删除本地模型", f"确定删除 {model_id} 的本地权重文件？")
        if ans == QMessageBox.StandardButton.Yes:
            delete_cached_model(model_id, self._cache_dir())
            self.refresh_table()
