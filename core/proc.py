"""Subprocess helpers: PID lock files, worker spawning and process-tree termination (Windows-safe)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import psutil

from core.settings import PROJECT_ROOT


class LockError(RuntimeError):
    pass


class PidLock:
    """Exclusive lock owned by a live PID. Stale locks (dead owner) are taken over."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.held = False

    def owner(self) -> int | None:
        try:
            pid = int(self.path.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None
        return pid if psutil.pid_exists(pid) else None

    def acquire(self) -> bool:
        if self.held:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self.held = True
                return True
            except FileExistsError:
                owner = self.owner()
                if owner == os.getpid():
                    self.held = True
                    return True
                if owner is not None:
                    return False
                self.path.unlink(missing_ok=True)  # stale
        return False

    def release(self) -> None:
        if self.held and self.owner() in (os.getpid(), None):
            self.path.unlink(missing_ok=True)
        self.held = False

    def __enter__(self):
        if not self.acquire():
            raise LockError(f"Lock {self.path} is held by PID {self.owner()}")
        return self

    def __exit__(self, *exc):
        self.release()


def spawn_worker(args: list[str], log_path: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start `python main.py <args>` detached from our console's Ctrl+C, logging to a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    log = open(log_path, "ab")
    try:
        return subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "main.py"), *args],
            cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env={**os.environ, **(env or {})}, creationflags=flags,
            start_new_session=sys.platform != "win32",
        )
    finally:
        log.close()


def kill_tree(pid: int, timeout: float = 10.0) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def is_worker_process(pid: int, exp_id: int) -> bool:
    """True only if pid is a `main.py worker --exp-id <exp_id>` process (guards against PID reuse)."""
    try:
        cmd = psutil.Process(pid).cmdline()
    except psutil.Error:
        return False
    return "worker" in cmd and str(exp_id) in cmd and any(c.endswith("main.py") for c in cmd)


def run_command(cmd: list[str], input_text: str | None = None, timeout: float = 300,
                cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run an external program with a hard timeout. No shell, so arguments are never interpreted."""
    return subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout,
                          cwd=cwd, shell=False, check=False)
