import json
import os

import numpy as np
import pytest

from core.workspace import Workspace
from experiments import service
from experiments.spec import SpecError, fingerprint, normalize_spec
from experiments.trainer import cross_validate, load_context
from tracking.db import Tracker


def test_spec_validation_rejects_unsafe_or_unknown():
    with pytest.raises(SpecError):
        normalize_spec({"model": "lgbm", "code": "import os"})
    with pytest.raises(ValueError):
        normalize_spec({"model": "made_up_model"})
    with pytest.raises(SpecError):
        normalize_spec({"model": "lgbm", "params": {"x": [1, 2]}})
    with pytest.raises(SpecError):
        normalize_spec({"model": "lgbm", "params": {"__class__": 1}})
    with pytest.raises(ValueError):
        normalize_spec({"model": "lgbm", "features": ["nope"]})
    with pytest.raises(SpecError):
        normalize_spec({"model": "lgbm", "target_transform": "log1p"}, task_type="binary")


def test_fingerprint_ignores_device_and_order():
    a = normalize_spec({"model": "xgb", "features": ["row_stats", "count_encoding"], "device": "gpu"})
    b = normalize_spec({"model": "xgb", "features": ["count_encoding", "row_stats"], "device": "cpu"})
    assert fingerprint(a) == fingerprint(b)
    c = normalize_spec({"model": "xgb", "features": ["count_encoding"], "params": {"max_depth": 4}})
    assert fingerprint(a) != fingerprint(c)


def test_queue_dedupes_and_requires_hypothesis(make_comp, tracker):
    ws = make_comp("binary")
    with pytest.raises(service.ExperimentError):
        service.queue_experiment(ws, tracker, {"model": "lgbm"}, "  ")
    first = service.queue_experiment(ws, tracker, {"model": "lgbm"}, "baseline")
    again = service.queue_experiment(ws, tracker, {"model": "lgbm", "device": "cpu"}, "same thing")
    assert not first["duplicate"] and again["duplicate"]
    assert again["experiment_id"] == first["experiment_id"]


def test_compliance_blocks_unconfirmed_requirements(make_comp, tracker):
    ws = make_comp("binary")
    out = service.queue_experiment(ws, tracker, {"model": "lgbm", "requires": ["external_data"]}, "uses extra data")
    assert out["status"] == "blocked"
    assert "unconfirmed" in out["blocked_reasons"][0]
    cfg = ws.load_config()
    cfg["rules"]["external_data"] = "forbidden"
    ws.save_config(cfg)
    exp_id = out["experiment_id"]
    tracker.update_experiment(exp_id, status="queued")  # even if forced into the queue, the worker re-checks
    result = service.execute(exp_id, tracker)
    assert result["status"] == "blocked"


def test_execute_train_records_everything(make_comp, tracker):
    ws = make_comp("binary", n_train=2500)  # large enough for a holdout
    out = service.queue_experiment(ws, tracker, {"model": "lgbm", "features": ["count_encoding"]}, "baseline")
    result = service.execute(out["experiment_id"], tracker)
    assert result["status"] == "succeeded"
    exp = tracker.get_experiment(out["experiment_id"])
    assert 0.8 < exp["cv_score"] <= 1.0
    assert len(exp["fold_scores"]) == 5
    assert exp["holdout_score"] is not None
    assert exp["validation_version"] == ws.load_validation()["version"]
    assert exp["dataset_version"] and exp["code_version"]
    for name in ("oof.npy", "test.npy", "holdout.npy", "result.json", "train.log"):
        assert (ws.experiment_dir(exp["id"]) / name).exists()
    assert (ws.experiment_dir(exp["id"]) / "models" / "fold_0.joblib").exists()
    # holdout is hidden in normal views and revealed by promote
    assert "holdout_score" not in service.public_view(exp)
    promoted = service.promote(ws, tracker, exp["id"])
    assert promoted["holdout_score"] == exp["holdout_score"]


def test_fold_checkpoints_resume_without_retraining(make_comp, tracker, monkeypatch):
    ws = make_comp("binary")
    ctx = load_context(ws)
    spec = normalize_spec({"model": "lgbm"}, "binary")
    save_dir = ws.experiment_dir(999)
    save_dir.mkdir(parents=True)
    first = cross_validate(ctx, spec, {}, save_dir=save_dir)

    import models.tabular as mt
    def boom(*a, **k):
        raise AssertionError("should have used the checkpoint")
    monkeypatch.setattr(mt.FAMILIES["lgbm"], "fit", boom)
    second = cross_validate(ctx, spec, {}, save_dir=save_dir)
    assert second.cv_score == pytest.approx(first.cv_score)
    np.testing.assert_allclose(second.test_pred, first.test_pred)


@pytest.mark.parametrize("kind,model", [("multiclass", "xgb"), ("regression", "linear"), ("binary_label", "rf")])
def test_other_tasks_train(make_comp, tracker, kind, model):
    ws = make_comp(kind, n_train=600)
    out = service.queue_experiment(ws, tracker, {"model": model}, "smoke")
    assert service.execute(out["experiment_id"], tracker)["status"] == "succeeded"


def test_target_encoding_and_tuning(make_comp, tracker):
    ws = make_comp("regression", n_train=700)
    t = service.queue_experiment(ws, tracker, {"model": "lgbm", "features": ["target_encoding"],
                                               "target_transform": "log1p"}, "te")
    assert service.execute(t["experiment_id"], tracker)["status"] == "succeeded"
    tune = service.queue_experiment(ws, tracker, {"kind": "tune", "model": "lgbm", "n_trials": 3, "timeout": 120},
                                    "tuning helps")
    result = service.execute(tune["experiment_id"], tracker)
    assert result["status"] == "succeeded"
    info = json.loads((ws.experiment_dir(tune["experiment_id"]) / "result.json").read_text())
    assert info["n_trials"] >= 3 and info["best_params"]


def test_ensemble_and_compare(make_comp, tracker):
    ws = make_comp("binary")
    ids = []
    for m in ("lgbm", "linear", "xgb"):
        out = service.queue_experiment(ws, tracker, {"model": m}, f"{m} baseline")
        service.execute(out["experiment_id"], tracker)
        ids.append(out["experiment_id"])
    ens = service.queue_experiment(ws, tracker, {"kind": "ensemble"}, "blend")
    assert service.execute(ens["experiment_id"], tracker)["status"] == "succeeded"
    exp = tracker.get_experiment(ens["experiment_id"])
    weights = exp["params"]["weights"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    cmp = service.compare(ws, tracker, [ids[0], ens["experiment_id"]])
    assert cmp["experiments"][1]["vs_first"]["n_folds"] == 5


def test_recover_stale_requeues_dead_worker(make_comp, tracker):
    ws = make_comp("binary")
    out = service.queue_experiment(ws, tracker, {"model": "lgbm"}, "x")
    tracker.update_experiment(out["experiment_id"], status="running", pid=999999, attempts=1, heartbeat_at=0)
    actions = service.recover_stale(ws, tracker, max_attempts=3)
    assert actions == [{"experiment_id": out["experiment_id"], "action": "requeued"}]
    assert tracker.get_experiment(out["experiment_id"])["status"] == "queued"
    tracker.update_experiment(out["experiment_id"], status="running", pid=999999, attempts=3, heartbeat_at=0)
    assert service.recover_stale(ws, tracker, max_attempts=3)[0]["action"] == "failed"


def test_custom_feature_requires_approval(make_comp, tracker):
    from experiments import custom_code
    ws = make_comp("binary")
    code = ws.custom_features_dir / "ratio.py"
    code.write_text("import pandas as pd\n\ndef transform(df):\n    return pd.DataFrame({'r': df['num_0'] / (df['num_1'].abs() + 1)})\n")
    out = service.queue_experiment(ws, tracker, {"model": "lgbm", "features": ["custom:ratio"]}, "ratio helps")
    assert service.execute(out["experiment_id"], tracker)["status"] == "blocked"
    custom_code.approve(ws, "ratio")
    tracker.update_experiment(out["experiment_id"], status="queued")
    assert service.execute(out["experiment_id"], tracker)["status"] == "succeeded"
    code.write_text(code.read_text() + "\n# edited\n")
    assert not custom_code.list_custom(ws)[0]["approved"]


def test_tuning_warm_start_is_clipped_into_search_space():
    from experiments.tuning import fit_to_space
    from models.tabular import FAMILIES
    seeded = fit_to_space({"reg_alpha": 0.0, "num_leaves": 1000, "max_bin": 100, "learning_rate": 0.05,
                           "unknown": 1}, FAMILIES["lgbm"], "binary")
    assert seeded["reg_alpha"] == pytest.approx(1e-4)
    assert seeded["num_leaves"] == 256
    assert "max_bin" not in seeded and "unknown" not in seeded
    assert seeded["learning_rate"] == 0.05
