"""Windows runtime safeguards, hardware telemetry, and stage-scoped GPU/CPU model lifecycle pool."""

from __future__ import annotations

import ctypes
import gc
from pathlib import Path
import shutil
import sys
import threading
import types
from typing import Any


def ensure_qwen_windows_safe() -> None:
    """Stub unused Japanese `nagisa` module before importing `qwen_asr` to prevent dynet crash on CJK paths."""
    if "nagisa" not in sys.modules:
        stub = types.ModuleType("nagisa")
        stub.tagging = lambda *args, **kwargs: types.SimpleNamespace(words=[])
        sys.modules["nagisa"] = stub


def get_system_memory_info() -> dict[str, int]:
    """Return host system RAM statistics in MB (`total_mb`, `avail_mb`, `used_mb`)."""
    if sys.platform == "win32":
        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_mb = int(stat.ullTotalPhys // (1024 * 1024))
            avail_mb = int(stat.ullAvailPhys // (1024 * 1024))
            return {
                "total_mb": total_mb,
                "avail_mb": avail_mb,
                "used_mb": max(0, total_mb - avail_mb),
            }
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            vals: dict[str, int] = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    vals[parts[0].rstrip(":")] = int(parts[1])
            total_mb = vals.get("MemTotal", 0) // 1024
            avail_mb = vals.get("MemAvailable", vals.get("MemFree", 0)) // 1024
            return {
                "total_mb": total_mb,
                "avail_mb": avail_mb,
                "used_mb": max(0, total_mb - avail_mb),
            }
    except Exception:
        return {"total_mb": 0, "avail_mb": 0, "used_mb": 0}


def get_gpu_vram_info() -> dict[str, Any]:
    """Return CUDA device availability, name, and free/total/peak VRAM in MB."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "available": False,
                "device_name": "",
                "free_mb": 0,
                "total_mb": 0,
                "peak_allocated_mb": 0,
            }
        name = torch.cuda.get_device_name(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        peak_b = torch.cuda.max_memory_allocated(0)
        return {
            "available": True,
            "device_name": name,
            "free_mb": int(free_b // (1024 * 1024)),
            "total_mb": int(total_b // (1024 * 1024)),
            "peak_allocated_mb": int(peak_b // (1024 * 1024)),
        }
    except Exception:
        return {
            "available": False,
            "device_name": "",
            "free_mb": 0,
            "total_mb": 0,
            "peak_allocated_mb": 0,
        }


def get_disk_free_mb(path: Path | str) -> float:
    """Return free disk space in MB for the partition containing `path`."""
    p = Path(path)
    probe = p
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
        return round(usage.free / (1024 * 1024), 1)
    except Exception:
        return 0.0


def resolve_torch_device(pref: str = "auto", min_free_vram_mb: int = 800) -> str:
    """Select 'cuda:0' or 'cpu' safely based on user preference and current free VRAM."""
    if pref == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info(0)
                if free_bytes / (1024 * 1024) >= min_free_vram_mb or pref == "cuda":
                    return "cuda:0"
            except Exception:
                if pref == "cuda":
                    return "cuda:0"
    except Exception:
        pass
    return "cpu"


def is_oom_error(exc: BaseException) -> bool:
    """Detect CUDA or host-RAM out-of-memory exceptions reliably."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    keywords = (
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "not enough memory",
        "can't allocate memory",
    )
    return any(kw in msg for kw in keywords)


def release_torch_memory() -> None:
    """Trigger Python GC and CUDA cache flush without raising on CPU-only machines."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


class ModelSessionPool:
    """Manages stage-scoped heavyweight AI models so ASR and ForcedAligner never co-reside in VRAM."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[str, tuple[str, Any, str]] = {}  # role -> (cache_key, model_obj, device)

    def get(self, role: str, cache_key: str) -> tuple[Any, str] | None:
        with self._lock:
            item = self._slots.get(role)
            if item is not None and item[0] == cache_key:
                return item[1], item[2]
            return None

    def put(self, role: str, cache_key: str, model_obj: Any, device: str, *, exclusive_gpu: bool = True) -> None:
        evicted: list[Any] = []
        with self._lock:
            if exclusive_gpu and device.startswith("cuda"):
                for other_role in list(self._slots.keys()):
                    if other_role != role:
                        _, other_obj, other_dev = self._slots.pop(other_role)
                        if other_dev.startswith("cuda"):
                            evicted.append(other_obj)
            old = self._slots.get(role)
            if old is not None and old[0] != cache_key:
                evicted.append(old[1])
            self._slots[role] = (cache_key, model_obj, device)
        self._dispose_objects(evicted)

    def unload_role(self, role: str) -> None:
        evicted: list[Any] = []
        with self._lock:
            old = self._slots.pop(role, None)
            if old is not None:
                evicted.append(old[1])
        self._dispose_objects(evicted)

    def unload_all(self) -> None:
        evicted: list[Any] = []
        with self._lock:
            for _, obj, _ in self._slots.values():
                evicted.append(obj)
            self._slots.clear()
        self._dispose_objects(evicted)

    def loaded_roles(self) -> dict[str, tuple[str, str]]:
        with self._lock:
            return {role: (key, dev) for role, (key, _obj, dev) in self._slots.items()}

    @staticmethod
    def _dispose_objects(objs: list[Any]) -> None:
        if not objs:
            return
        for obj in objs:
            try:
                inner = getattr(obj, "model", None)
                if inner is not None and hasattr(inner, "to"):
                    inner.to("cpu")
                elif hasattr(obj, "to"):
                    obj.to("cpu")
            except Exception:
                pass
        objs.clear()
        release_torch_memory()


MODEL_POOL = ModelSessionPool()


def snapshot_resource_usage(cache_dir: Path | None = None, work_cache_dir: Path | None = None) -> dict[str, Any]:
    """Capture current VRAM, RAM, and disk usage diagnostics for reporting."""
    vram = get_gpu_vram_info()
    ram = get_system_memory_info()
    snap: dict[str, Any] = {
        "gpu_available": vram["available"],
        "gpu_name": vram["device_name"],
        "vram_free_mb": vram["free_mb"],
        "vram_total_mb": vram["total_mb"],
        "vram_peak_allocated_mb": vram["peak_allocated_mb"],
        "ram_total_mb": ram["total_mb"],
        "ram_avail_mb": ram["avail_mb"],
        "ram_used_mb": ram["used_mb"],
    }
    if cache_dir is not None:
        snap["model_dir_free_mb"] = get_disk_free_mb(cache_dir)
    if work_cache_dir is not None:
        snap["work_cache_free_mb"] = get_disk_free_mb(work_cache_dir)
    return snap
