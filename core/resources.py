"""GPU/CPU/disk inspection, GPU capability probing and per-process resource monitoring."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

import psutil

from core.settings import load_settings, workspaces_root


def _nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def gpu_info() -> list[dict[str, Any]]:
    nv = _nvml()
    if nv is None:
        return []
    out = []
    for i in range(nv.nvmlDeviceGetCount()):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        try:
            util = nv.nvmlDeviceGetUtilizationRates(h).gpu
        except Exception:
            util = None
        try:
            temp = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        name = nv.nvmlDeviceGetName(h)
        out.append({
            "index": i,
            "name": name.decode() if isinstance(name, bytes) else name,
            "memory_total_mb": mem.total // 2**20,
            "memory_used_mb": mem.used // 2**20,
            "utilization_pct": util,
            "temperature_c": temp,
        })
    return out


def torch_cuda_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    info = {"installed": True, "version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info.update(cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                    bf16=torch.cuda.is_bf16_supported())
    return info


def disk_free_gb(path: Path | None = None) -> float:
    return shutil.disk_usage(path or workspaces_root()).free / 2**30


def system_snapshot() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "cpu_count": os.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "ram_total_gb": round(vm.total / 2**30, 1),
        "ram_available_gb": round(vm.available / 2**30, 1),
        "disk_free_gb": round(disk_free_gb(), 1),
        "gpus": gpu_info(),
    }


CAPS_FILE = ".gpu_capabilities.json"


def gpu_capabilities(refresh: bool = False) -> dict[str, bool]:
    """Which libraries can actually train on this GPU. Probed once with tiny fits and cached;
    a wheel may import fine and still lack kernels for a new architecture."""
    path = workspaces_root() / CAPS_FILE
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    caps = {"xgboost": False, "catboost": False, "lightgbm": False, "torch": False}
    if gpu_info():
        import numpy as np
        rng = np.random.default_rng(0)
        X, y = rng.random((2000, 8)), rng.integers(0, 2, 2000)
        try:
            import xgboost as xgb
            xgb.XGBClassifier(n_estimators=5, device="cuda", tree_method="hist").fit(X, y)
            caps["xgboost"] = True
        except Exception:
            pass
        try:
            import catboost as cb
            cb.CatBoostClassifier(iterations=5, task_type="GPU", verbose=0, allow_writing_files=False).fit(X, y)
            caps["catboost"] = True
        except Exception:
            pass
        try:
            import lightgbm as lgb
            lgb.LGBMClassifier(n_estimators=5, device="gpu", verbose=-1).fit(X, y)
            caps["lightgbm"] = True
        except Exception:
            pass
        caps["torch"] = bool(torch_cuda_info().get("cuda_available"))
    path.write_text(json.dumps(caps))
    return caps


def gpu_enabled() -> bool:
    setting = str(load_settings()["resources"].get("use_gpu", "auto")).lower()
    if setting in ("false", "0", "no"):
        return False
    return bool(gpu_info())


class ResourceMonitor(threading.Thread):
    """Samples peak RSS (process tree) and GPU memory attributed to this process; optional heartbeat."""

    def __init__(self, interval: float = 2.0, heartbeat: Callable[[], None] | None = None,
                 heartbeat_every: float = 30.0):
        super().__init__(daemon=True)
        self.interval, self.heartbeat, self.heartbeat_every = interval, heartbeat, heartbeat_every
        self.peak_rss_mb = 0.0
        self.peak_gpu_mb = 0.0
        self._stop_event = threading.Event()
        self._proc = psutil.Process()
        self._nv = _nvml()
        self._baseline_mb = self._device_used_mb()

    def _device_used_mb(self) -> float:
        if self._nv is None:
            return 0.0
        try:
            return sum(self._nv.nvmlDeviceGetMemoryInfo(self._nv.nvmlDeviceGetHandleByIndex(i)).used / 2**20
                       for i in range(self._nv.nvmlDeviceGetCount()))
        except Exception:
            return 0.0

    def _gpu_mb(self) -> float:
        if self._nv is None:
            return 0.0
        pids = {self._proc.pid} | {c.pid for c in self._proc.children(recursive=True)}
        total = 0.0
        try:
            for i in range(self._nv.nvmlDeviceGetCount()):
                h = self._nv.nvmlDeviceGetHandleByIndex(i)
                for p in self._nv.nvmlDeviceGetComputeRunningProcesses(h):
                    if p.pid in pids and p.usedGpuMemory:
                        total += p.usedGpuMemory / 2**20
        except Exception:
            pass
        if total == 0.0:
            # Windows WDDM hides per-process usage: use device memory growth since this worker started
            # (an upper bound if other programs allocate VRAM concurrently).
            total = max(0.0, self._device_used_mb() - self._baseline_mb)
        if total == 0.0:
            try:
                import sys
                if "torch" in sys.modules:
                    import torch
                    if torch.cuda.is_available():
                        total = torch.cuda.max_memory_reserved() / 2**20
            except Exception:
                pass
        return total

    def sample(self) -> None:
        try:
            rss = self._proc.memory_info().rss + sum(c.memory_info().rss for c in self._proc.children(recursive=True))
            self.peak_rss_mb = max(self.peak_rss_mb, rss / 2**20)
        except psutil.Error:
            pass
        self.peak_gpu_mb = max(self.peak_gpu_mb, self._gpu_mb())

    def run(self) -> None:
        last_beat = 0.0
        while not self._stop_event.is_set():
            self.sample()
            if self.heartbeat and time.time() - last_beat > self.heartbeat_every:
                try:
                    self.heartbeat()
                except Exception:
                    pass
                last_beat = time.time()
            self._stop_event.wait(self.interval)

    def stop(self) -> None:
        self._stop_event.set()
        self.sample()
