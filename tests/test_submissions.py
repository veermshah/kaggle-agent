import pandas as pd
import pytest

from competitions.kaggle_client import LocalDirectoryClient
from experiments import service
from submissions.builder import SubmissionError
from submissions.manager import SubmissionManager


def _trained(ws, tracker, model="lgbm", seed=42):
    out = service.queue_experiment(ws, tracker, {"model": model, "seed": seed}, f"{model} {seed}")
    assert service.execute(out["experiment_id"], tracker)["status"] == "succeeded"
    return out["experiment_id"]


@pytest.mark.parametrize("kind", ["binary", "binary_label", "multiclass", "regression"])
def test_prepare_matches_sample_format(make_comp, tracker, kind):
    ws = make_comp(kind, n_train=600)
    exp_id = _trained(ws, tracker, "xgb")
    mgr = SubmissionManager(ws, tracker)
    prepared = mgr.prepare(exp_id)
    sub = pd.read_csv(prepared["file"])
    sample = pd.read_csv(ws.data_dir / "sample_submission.csv")
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample) and (sub["id"] == sample["id"]).all()
    assert not sub.isna().any().any()
    if kind == "binary_label":
        assert set(sub["Transported"].astype(str)) <= {"True", "False"}
    if kind == "multiclass":
        assert sub.iloc[:, 1:].sum(axis=1).round(6).eq(1).all()
    assert prepared["status"] == "pending_approval"


def test_simulated_submission_flow_and_daily_budget(make_comp, tracker):
    ws = make_comp("binary")
    cfg = ws.load_config()
    cfg["max_daily_submissions"] = 1
    ws.save_config(cfg)
    a, b = _trained(ws, tracker, "lgbm"), _trained(ws, tracker, "linear")
    mgr = SubmissionManager(ws, tracker)
    first = mgr.prepare(a)
    result = mgr.approve(first["submission_id"])
    assert result["simulated"] is True and result["status"] == "scored"
    assert 0.5 < result["public_score"] <= 1.0  # scored against the mock answer key
    second = mgr.prepare(b)
    with pytest.raises(SubmissionError, match="budget exhausted"):
        mgr.approve(second["submission_id"])
    assert tracker.get_submission(second["submission_id"])["status"] == "pending_approval"
    assert mgr.usage_today()["remaining"] == 0


def test_near_duplicate_submissions_refused(make_comp, tracker):
    ws = make_comp("binary")
    a = _trained(ws, tracker, "lgbm")
    mgr = SubmissionManager(ws, tracker)
    mgr.prepare(a)
    with pytest.raises(SubmissionError, match="identical"):
        mgr.prepare(a)
    assert mgr.prepare(a, force=True)["status"] == "pending_approval"


def test_autonomous_submission_needs_explicit_authorization(make_comp, tracker):
    ws = make_comp("binary")
    exp = tracker.get_experiment(_trained(ws, tracker))
    mgr = SubmissionManager(ws, tracker)
    from evaluation.metrics import get_metric
    assert mgr.maybe_autosubmit(exp, get_metric("roc_auc")) is None
    cfg = ws.load_config()
    cfg["autosubmit"] = {"enabled": True, "max_daily": 1}
    ws.save_config(cfg)
    assert SubmissionManager(ws, tracker).maybe_autosubmit(exp, get_metric("roc_auc")) is None  # rules unconfirmed
    cfg["rules"]["automated_submissions"] = "allowed"
    ws.save_config(cfg)
    out = SubmissionManager(ws, tracker).maybe_autosubmit(exp, get_metric("roc_auc"))
    assert out["status"] == "scored"
    assert tracker.list_submissions(ws.slug)[0]["autonomous"] == 1
    # budget of 1 is now used
    assert SubmissionManager(ws, tracker).maybe_autosubmit(exp, get_metric("roc_auc")) is None


def test_code_competition_blocks_api_submission(make_comp, tracker):
    ws = make_comp("binary")
    cfg = ws.load_config()
    cfg["rules"]["code_competition"] = True
    ws.save_config(cfg)
    mgr = SubmissionManager(ws, tracker)
    prepared = mgr.prepare(_trained(ws, tracker))
    with pytest.raises(SubmissionError, match="code competition"):
        mgr.approve(prepared["submission_id"])


def test_tampered_file_is_not_submitted(make_comp, tracker):
    ws = make_comp("binary")
    mgr = SubmissionManager(ws, tracker)
    prepared = mgr.prepare(_trained(ws, tracker))
    with open(prepared["file"], "a") as fh:
        fh.write("999,0.5\n")
    with pytest.raises(SubmissionError, match="changed"):
        mgr.approve(prepared["submission_id"])


def test_cli_approval_requires_a_human(make_comp, tracker, monkeypatch):
    import main
    ws = make_comp("binary")
    prepared = SubmissionManager(ws, tracker).prepare(_trained(ws, tracker))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    code, payload = main.run(["submit", "approve", "--id", str(prepared["submission_id"]), "-c", ws.slug])
    assert code == 1 and "interactive terminal" in payload["error"]
    assert tracker.get_submission(prepared["submission_id"])["status"] == "pending_approval"
