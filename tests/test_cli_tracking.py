import threading

import main
from tracking.db import Tracker


def test_cli_end_to_end_json(home):
    src = home / "src"
    code, out = main.run(["mock", "create", "--kind", "binary", "--out", str(src), "--rows", "600"])
    assert code == 0 and out["ok"]
    code, out = main.run(["start", "--competition", "cli-comp", "--data-dir", str(src)])
    assert code == 0, out
    assert out["result"]["task_type"] == "binary"
    spec = '{"model": "lgbm"}'
    code, out = main.run(["exp", "propose", "--spec", spec, "--hypothesis", "baseline"])
    assert code == 0 and out["result"]["status"] == "queued"
    exp_id = out["result"]["experiment_id"]
    code, out = main.run(["exp", "run", "--id", str(exp_id)])
    assert code == 0 and out["result"]["status"] == "succeeded", out
    code, out = main.run(["status"])
    assert out["result"]["best"]["id"] == exp_id
    assert "holdout_score" not in out["result"]["best"]
    code, out = main.run(["report"])
    assert code == 0 and "What worked" in open(out["result"]["path"], encoding="utf-8").read()
    code, out = main.run(["submit", "prepare", "--exp-id", str(exp_id)])
    assert code == 0 and out["result"]["status"] == "pending_approval"


def test_cli_errors_are_structured(home):
    code, out = main.run(["status", "-c", "does-not-exist"])
    assert code == 1 and out["ok"] is False and "not initialized" in out["error"]


def test_cli_guards_destructive_and_risky_actions(make_comp):
    ws = make_comp("binary")
    code, out = main.run(["validation", "reset", "--strategy", "kfold", "-c", ws.slug])
    assert code == 1 and "--yes" in out["error"]
    code, out = main.run(["cleanup", "-c", ws.slug])
    assert code == 1 and "--yes" in out["error"]
    code, out = main.run(["autosubmit", "enable", "--max-daily", "2", "-c", ws.slug])
    assert code == 1 and "authorized" in out["error"]
    code, out = main.run(["autosubmit", "enable", "--max-daily", "2", "--i-authorize", "-c", ws.slug])
    assert code == 1 and "automated-submissions" in out["error"]
    main.run(["rules", "set", "--automated-submissions", "allowed", "-c", ws.slug])
    code, out = main.run(["autosubmit", "enable", "--max-daily", "99", "--i-authorize", "-c", ws.slug])
    assert code == 1 and "competition limit" in out["error"]
    code, out = main.run(["autosubmit", "enable", "--max-daily", "2", "--i-authorize", "-c", ws.slug])
    assert code == 0 and out["result"]["enabled"] is True


def test_validation_reset_creates_new_version(make_comp):
    ws = make_comp("binary")
    old = ws.load_validation()["version"]
    code, out = main.run(["validation", "reset", "--strategy", "kfold", "--n-splits", "3", "--yes", "-c", ws.slug])
    assert code == 0 and out["result"]["version"] != old and out["result"]["n_splits"] == 3


def test_claim_next_is_atomic(home):
    tracker = Tracker()
    for i in range(20):
        tracker.create_experiment("c", {"kind": "train", "model": "lgbm"}, fingerprint=str(i), hypothesis="h",
                                  validation_version="v", resource="cpu")
    claimed, lock = [], threading.Lock()

    def worker():
        t = Tracker()
        while (exp := t.claim_next("c", "cpu")) is not None:
            with lock:
                claimed.append(exp["id"])

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed) == list(range(1, 21))
