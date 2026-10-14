"""Per-competition workspace layout and config persistence."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from core.settings import workspaces_root

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,99}$")


class WorkspaceError(RuntimeError):
    pass


def validate_slug(slug: str) -> str:
    slug = slug.strip().lower()
    if not SLUG_RE.match(slug):
        raise WorkspaceError(f"Invalid competition slug {slug!r}: use lowercase letters, digits, '-' or '_'.")
    return slug


class Workspace:
    def __init__(self, slug: str, root: Path | None = None):
        self.slug = validate_slug(slug)
        self.root = (root or workspaces_root()) / self.slug

    # Layout
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def submissions_dir(self) -> Path:
        return self.root / "submissions"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def custom_features_dir(self) -> Path:
        return self.root / "custom_features"

    @property
    def config_path(self) -> Path:
        return self.root / "competition.yaml"

    @property
    def profile_path(self) -> Path:
        return self.root / "profile.json"

    @property
    def folds_path(self) -> Path:
        return self.root / "folds.csv"

    @property
    def validation_path(self) -> Path:
        return self.root / "validation.json"

    @property
    def optuna_url(self) -> str:
        return f"sqlite:///{(self.root / 'optuna.db').as_posix()}"

    def experiment_dir(self, exp_id: int) -> Path:
        return self.experiments_dir / f"exp_{exp_id:05d}"

    def create(self) -> "Workspace":
        for d in (self.data_dir, self.experiments_dir, self.submissions_dir, self.logs_dir, self.custom_features_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def exists(self) -> bool:
        return self.config_path.exists()

    # Config
    def load_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise WorkspaceError(
                f"Competition '{self.slug}' is not initialized. Run: python main.py start --competition {self.slug}"
            )
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}

    def save_config(self, config: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        tmp.replace(self.config_path)

    def load_profile(self) -> dict[str, Any]:
        return json.loads(self.profile_path.read_text()) if self.profile_path.exists() else {}

    def save_profile(self, profile: dict[str, Any]) -> None:
        self.profile_path.write_text(json.dumps(profile, indent=2, default=str))

    def load_validation(self) -> dict[str, Any]:
        if not self.validation_path.exists():
            raise WorkspaceError(f"No validation scheme for '{self.slug}'. Re-run onboarding.")
        return json.loads(self.validation_path.read_text())


ACTIVE_FILE = ".active_competition"


def set_active(slug: str) -> None:
    (workspaces_root() / ACTIVE_FILE).write_text(validate_slug(slug))


def get_active() -> str | None:
    path = workspaces_root() / ACTIVE_FILE
    return path.read_text().strip() if path.exists() else None


def resolve_workspace(slug: str | None) -> Workspace:
    slug = slug or get_active()
    if not slug:
        raise WorkspaceError("No competition given and none active. Pass --competition SLUG.")
    return Workspace(slug)
