import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from competitions.kaggle_client import LocalDirectoryClient  # noqa: E402
from competitions.mock import create_mock_competition  # noqa: E402
from competitions.onboarding import onboard  # noqa: E402
from core.settings import reset_settings_cache  # noqa: E402
from core.workspace import Workspace  # noqa: E402
from tracking.db import Tracker  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated workspaces + database; GPU probing disabled so tests are hardware-independent."""
    monkeypatch.setenv("KAGGLE_AGENT_HOME", str(tmp_path))
    monkeypatch.setenv("KAGGLE_AGENT_IGNORE_LOCAL_CONFIG", "1")  # tests use defaults only
    reset_settings_cache()
    root = tmp_path / "workspaces"
    root.mkdir()
    (root / ".gpu_capabilities.json").write_text(json.dumps(
        {"xgboost": False, "catboost": False, "lightgbm": False, "torch": False}))
    yield tmp_path
    reset_settings_cache()


@pytest.fixture
def tracker(home):
    return Tracker()


@pytest.fixture
def make_comp(home, tracker):
    def make(kind="binary", n_train=800, n_test=300, slug=None, seed=0):
        src = create_mock_competition(home / "src" / f"{kind}-{seed}", kind, n_train=n_train, n_test=n_test, seed=seed)
        slug = slug or f"t-{kind.replace('_', '-')}"
        onboard(slug, LocalDirectoryClient(src), tracker, data_dir=str(src))
        return Workspace(slug)
    return make
