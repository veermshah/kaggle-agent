"""Global settings: defaults.yaml merged with optional config/local.yaml and env overrides."""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env support: KEY=VALUE lines; real environment variables take precedence."""
    path = path or PROJECT_ROOT / ".env"
    if os.environ.get("KAGGLE_AGENT_NO_DOTENV") or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@lru_cache(maxsize=1)
def load_settings() -> dict[str, Any]:
    settings = yaml.safe_load((CONFIG_DIR / "defaults.yaml").read_text()) or {}
    local = CONFIG_DIR / "local.yaml"
    if local.exists() and not os.environ.get("KAGGLE_AGENT_IGNORE_LOCAL_CONFIG"):
        settings = deep_merge(settings, yaml.safe_load(local.read_text()) or {})
    # KAGGLE_AGENT_HOME relocates workspaces + database (used by tests and remote setups).
    home = os.environ.get("KAGGLE_AGENT_HOME")
    if home:
        settings["paths"]["workspaces"] = str(Path(home) / "workspaces")
        settings["paths"]["database"] = str(Path(home) / "workspaces" / "kaggle_agent.db")
    return settings


def reset_settings_cache() -> None:
    load_settings.cache_clear()


def workspaces_root() -> Path:
    root = _resolve(load_settings()["paths"]["workspaces"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def database_path() -> Path:
    path = _resolve(load_settings()["paths"]["database"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
