from __future__ import annotations

import subprocess
from functools import lru_cache

from core.settings import PROJECT_ROOT


@lru_cache(maxsize=1)
def code_version() -> str:
    """Git commit of the framework, suffixed with +dirty when there are uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=10).stdout.strip()
        if not sha:
            return "nogit"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{sha}+dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        return "nogit"
