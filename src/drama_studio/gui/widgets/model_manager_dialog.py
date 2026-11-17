"""Model download, local cache inspection, and hardware acceleration settings dialog."""

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
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from drama_studio.align.model_manager import (
    MODEL_REGISTRY,
    delete_cached_model,
    download_model,
    get_cached_model_size_mb,
    is_model_cached,
)
from drama_studio.gui.theme import get_palette
from drama_studio.gui.widgets.themed_controls import apply_win11_dwm_chrome
from drama_studio.config import ConfigStore


class ModelDownloadWorker(QThread):
    logLine = Signal(str)
    finishedOk = Signal(str)
    failed = Signal(str)

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
            )
            self.finishedOk.emit(self.model_id)
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelManagerDialog(QDialog):
    """GUI dialog for downloading, verifying, and cleaning local AI alignment models."""

    def __init__(self, store: ConfigStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self._worker: ModelDownloadWorker | None = None
        self.setWindowTitle("本地模型下载与缓存管理")
        self.resize(980, 500)
        self.setMinimumSize(820, 440)
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
        lay.setSpacing(10)

        # Hardware Banner
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            free_mb, total_mb = (round(x / 1024 / 1024) for x in torch.cuda.mem_get_info(0))
            hw_txt = f"GPU 加速可用：{name}（当前可用显存 {free_mb} MB / 共 {total_mb} MB）· 显存不足时将自动回退 CPU"
        else:
            hw_txt = "未检测到 CUDA GPU，将使用 CPU 执行（推荐使用轻量 FunASR fa-zh 模型，CPU 速度极快）"
        lbl_hw = QLabel(hw_txt)
        lbl_hw.setObjectName("HardwareBanner")
        lbl_hw.setWordWrap(True)
        lay.addWidget(lbl_hw)

        # Cache directory selector
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("模型缓存目录:"))
        self.edit_dir = QLineEdit(self.store.preferences["model_cache_dir"])
        self.edit_dir.setReadOnly(True)
        btn_dir = QPushButton("更改目录")
        btn_dir.clicked.connect(self._change_dir)
        self.chk_ms = QCheckBox("优先使用国内 ModelScope 镜像下载")
        self.chk_ms.setChecked(bool(self.store.preferences.get("use_modelscope_mirror", True)))
        self.chk_ms.toggled.connect(self._on_ms_toggled)
        dir_row.addWidget(self.edit_dir, 1)
        dir_row.addWidget(btn_dir)
        dir_row.addWidget(self.chk_ms)
        lay.addLayout(dir_row)

        # Model Table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["模型名称", "预计体积", "缓存状态", "许可证", "用途说明", "操作"])
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 215)
        self.table.setColumnWidth(3, 145)
        self.table.setColumnWidth(5, 112)
        lay.addWidget(self.table, 1)

        self.lbl_status = QLabel("提示：首次下载完成后，所有对齐、断句与音频处理均可完全离线运行。")
        self.lbl_status.setObjectName("MutedLabel")
        lay.addWidget(self.lbl_status)

        bottom = QHBoxLayout()
        bottom.addStretch()
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        bottom.addWidget(btn_close)
        lay.addLayout(bottom)

    def _cache_dir(self) -> Path:
        return Path(self.store.preferences["model_cache_dir"])

    def _change_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择模型缓存目录", str(self._cache_dir()))
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)
            self.store.preferences["model_cache_dir"] = folder
            self.store.save_prefs()
            self.edit_dir.setText(folder)
            self.refresh_table()

    def _on_ms_toggled(self, checked: bool) -> None:
        self.store.preferences["use_modelscope_mirror"] = checked
        self.store.save_prefs()

    def refresh_table(self) -> None:
        cdir = self._cache_dir()
        specs = list(MODEL_REGISTRY.values())
        self.table.setRowCount(len(specs))

        pal = get_palette(self.store.preferences.get("theme_mode", "light"))
        c_ok = QColor(pal.ok_text)
        c_warn = QColor("#FBBF24" if pal.is_dark else "#D97706")
        for row, spec in enumerate(specs):
            cached = is_model_cached(spec.model_id, cdir)
            actual_mb = get_cached_model_size_mb(spec.model_id, cdir)

            self.table.setItem(row, 0, QTableWidgetItem(spec.display_name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{spec.approx_size_mb} MB"))
            st_str = "内置已就绪" if spec.is_bundled else (f"已缓存 ({actual_mb} MB)" if cached else "未下载")
            st_item = QTableWidgetItem(st_str)
            st_item.setForeground(c_ok if cached else c_warn)
            self.table.setItem(row, 2, st_item)
            self.table.setItem(row, 3, QTableWidgetItem(spec.license_name))
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
                btn_del = QPushButton("清理缓存")
                btn_del.setFixedHeight(30)
                btn_del.setMinimumWidth(88)
                btn_del.setStyleSheet("padding: 2px 10px;")
                btn_del.clicked.connect(lambda _=False, mid=spec.model_id: self._delete_model(mid))
                a_lay.addWidget(btn_del)
            else:
                btn_dl = QPushButton("下载模型")
                btn_dl.setObjectName("PrimaryButton")
                btn_dl.setFixedHeight(30)
                btn_dl.setMinimumWidth(88)
                btn_dl.setStyleSheet("padding: 2px 10px;")
                btn_dl.clicked.connect(lambda _=False, mid=spec.model_id: self._download_model(mid))
                a_lay.addWidget(btn_dl)
            self.table.setCellWidget(row, 5, action_w)
            self.table.setRowHeight(row, 54)

    def _download_model(self, model_id: str) -> None:
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "下载中", "当前已有模型正在下载，请等待完成后再试。")
            return
        self.lbl_status.setText(f"正在下载 {model_id}，请勿关闭窗口...")
        self._worker = ModelDownloadWorker(
            model_id,
            self._cache_dir(),
            bool(self.store.preferences.get("use_modelscope_mirror", True)),
        )
        self._worker.logLine.connect(self.lbl_status.setText)
        self._worker.finishedOk.connect(self._on_dl_done)
        self._worker.failed.connect(self._on_dl_err)
        self._worker.start()

    def _on_dl_done(self, model_id: str) -> None:
        self.lbl_status.setText(f"模型 {model_id} 下载完成并已验证！")
        self.refresh_table()

    def _on_dl_err(self, err: str) -> None:
        self.lbl_status.setText(f"下载失败：{err}")
        QMessageBox.warning(self, "下载失败", err)
        self.refresh_table()

    def _delete_model(self, model_id: str) -> None:
        ans = QMessageBox.question(self, "清理模型缓存", f"确定删除 {model_id} 的本地缓存文件？")
        if ans == QMessageBox.StandardButton.Yes:
            delete_cached_model(model_id, self._cache_dir())
            self.refresh_table()
