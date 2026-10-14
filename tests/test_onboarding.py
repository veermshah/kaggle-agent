import numpy as np
import pandas as pd
import pytest

from competitions.kaggle_client import LocalDirectoryClient, safe_extract, KaggleError
from competitions.mock import create_mock_competition
from competitions.onboarding import onboard
from competitions.profiler import profile_dataset
from core.workspace import Workspace


@pytest.mark.parametrize("kind,task,fmt,metric", [
    ("binary", "binary", "probability", "roc_auc"),
    ("binary_label", "binary", "label", "accuracy"),
    ("multiclass", "multiclass", "class_probabilities", "logloss"),
    ("regression", "regression", "value", "rmse"),
])
def test_onboard_detects_task_and_format(make_comp, kind, task, fmt, metric):
    ws = make_comp(kind)
    cfg = ws.load_config()
    assert cfg["task_type"] == task
    assert cfg["submission"]["format"] == fmt
    assert cfg["metric"] == metric and cfg["metric_source"] == "kaggle"
    assert cfg["id_column"] == "id"
    assert "id" not in sum(cfg["features"].values(), [])
    assert "signup_date" in cfg["features"]["datetime"]
    assert set(cfg["features"]["categorical"]) == {"cat_a", "cat_b"}
    assert cfg["rules"]["external_data"] == "unknown"
    assert ws.validation_path.exists() and ws.folds_path.exists()
    # the answer key served by the mock source never lands in the workspace
    assert not (ws.data_dir / "solution.csv").exists()


def test_multiclass_classes_follow_submission_columns(make_comp):
    cfg = make_comp("multiclass").load_config()
    assert cfg["classes"] == ["C", "CL", "D"]
    assert cfg["submission"]["prediction_columns"] == ["Status_C", "Status_CL", "Status_D"]


def test_reonboard_preserves_user_decisions(make_comp, tracker, home):
    ws = make_comp("binary")
    cfg = ws.load_config()
    cfg["rules"]["external_data"] = "forbidden"
    cfg["validation"] = {"n_splits": 3}
    ws.save_config(cfg)
    version = ws.load_validation()["version"]
    src = home / "src" / "binary-0"
    onboard(ws.slug, LocalDirectoryClient(src), tracker, data_dir=str(src))
    cfg2 = ws.load_config()
    assert cfg2["rules"]["external_data"] == "forbidden"
    assert ws.load_validation()["version"] == version  # folds are not silently regenerated


def test_profiler_flags_leakage_and_contamination(home):
    src = create_mock_competition(home / "leaky", "binary", n_train=1000, n_test=300)
    train = pd.read_csv(src / "train.csv")
    test = pd.read_csv(src / "test.csv")
    train["leak"] = train["target"] * 10 + np.random.default_rng(0).random(len(train)) * 0.01
    test["leak"] = 0.0
    test.iloc[:5, 1:-1] = train.iloc[:5, 1:-2].to_numpy()  # copy feature rows from train
    test.loc[:4, "leak"] = train.loc[:4, "leak"].to_numpy()
    train.to_csv(src / "train.csv", index=False)
    test.to_csv(src / "test.csv", index=False)
    profile = profile_dataset(src)
    assert [s["column"] for s in profile["leakage_suspects"]] == ["leak"]
    assert profile["train_test_overlap_rows"] >= 5
    assert any("leakage" in w for w in profile["warnings"])


class FakeKaggleApi:
    """Stands in for the official API client: real (non-simulated) metadata, local files."""
    simulated = False

    def __init__(self, src, **meta):
        self.local = LocalDirectoryClient(src)
        self.meta = meta
        self.submitted = []

    def competition_metadata(self, slug):
        return {"slug": slug, "title": "Fake", "evaluation_metric": "Area Under the ROC Curve",
                "max_daily_submissions": 3, "deadline": "2099-01-01T00:00:00Z", "user_has_entered": True,
                "is_kernels_submissions_only": False, "submissions_disabled": False, **self.meta}

    def list_files(self, slug):
        return self.local.list_files(slug)

    def download(self, slug, dest):
        return self.local.download(slug, dest)

    def submit(self, slug, path, message):
        self.submitted.append((path, message))
        return {"ref": f"real-{len(self.submitted)}", "message": "ok"}

    def submissions(self, slug):
        return [{"ref": f"real-{i + 1}", "date": "2000-01-01", "description": m, "status": "complete",
                 "public_score": 0.9} for i, (_, m) in enumerate(self.submitted)]


def test_onboard_with_mocked_kaggle_api_records_rules(home, tracker):
    src = create_mock_competition(home / "api", "binary", n_train=600, n_test=200)
    out = onboard("fake-comp", FakeKaggleApi(src, is_kernels_submissions_only=True), tracker)
    cfg = Workspace("fake-comp").load_config()
    assert cfg["data_source"] == "kaggle"
    assert cfg["max_daily_submissions"] == 3
    assert cfg["rules"]["code_competition"] is True
    assert out["simulated_data_source"] is False


def test_safe_extract_rejects_traversal(tmp_path):
    import zipfile
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(KaggleError):
        safe_extract(archive, tmp_path / "out")
