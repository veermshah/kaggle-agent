import json
import sys
import time

import pytest

from agents.base import build_context
from agents.command import CommandAgent, CompositeAgent, parse_proposals
from agents.heuristic import HeuristicAgent
from experiments import service
from experiments.scheduler import Scheduler, request_stop


def test_heuristic_starts_with_baselines(make_comp, tracker):
    ws = make_comp("binary")
    props = HeuristicAgent().propose(build_context(ws, tracker), 10)
    assert props and all(p.arm.startswith("baseline:") for p in props)
    assert {p.spec["model"] for p in props} >= {"lgbm", "xgb", "linear"}
    assert all(p.hypothesis for p in props)


def test_heuristic_moves_on_after_baselines_without_repeats(make_comp, tracker):
    ws = make_comp("binary")
    for m in ("lgbm", "linear", "xgb", "rf", "extratrees", "catboost"):
        out = service.queue_experiment(ws, tracker, {"model": m}, "baseline", arm=f"baseline:{m}")
        if m in ("lgbm", "linear", "xgb"):
            service.execute(out["experiment_id"], tracker)
        else:
            tracker.update_experiment(out["experiment_id"], status="cancelled")
    ctx = build_context(ws, tracker)
    props = HeuristicAgent(numeric_columns=ws.load_config()["features"]["numeric"]).propose(ctx, 20)
    arms = {p.arm.split(":")[0] for p in props}
    assert {"tune", "features", "ensemble"} <= arms
    existing = {e["fingerprint"] for e in ctx.experiments}
    from experiments.spec import fingerprint, normalize_spec
    for p in props:
        if p.spec["kind"] != "ensemble":
            assert fingerprint(normalize_spec(p.spec, "binary")) not in existing


def test_arm_parking_and_revival():
    agent = HeuristicAgent()

    class Ctx:
        experiments = [
            {"arm": "features:lgbm:a", "status": "succeeded", "improved": False, "finished_at": "2026-01-01T00:00:01"},
            {"arm": "features:lgbm:b", "status": "succeeded", "improved": False, "finished_at": "2026-01-01T00:00:02"},
            {"arm": "features:lgbm:c", "status": "succeeded", "improved": False, "finished_at": "2026-01-01T00:00:03"},
        ]
    assert "features:lgbm" in agent._parked(Ctx, patience=3)
    Ctx.experiments.append({"arm": "tune:xgb", "status": "succeeded", "improved": True,
                            "finished_at": "2026-01-01T00:00:04"})
    assert "features:lgbm" not in agent._parked(Ctx, patience=3)  # new best => worth revisiting


def test_command_agent_validates_output(make_comp, tracker):
    ws = make_comp("binary")
    payload = {"proposals": [
        {"spec": {"model": "lgbm", "params": {"num_leaves": 15}}, "hypothesis": "fewer leaves generalize", "arm": "x"},
        {"spec": {"model": "lgbm", "params": {"evil": [1]}}, "hypothesis": "bad", "arm": "y"},
        {"spec": {"model": "rm -rf"}, "hypothesis": "bad", "arm": "z"},
    ]}
    script = f"import sys; sys.stdin.read(); print('here you go: ' + {json.dumps(payload)!r})"
    agent = CommandAgent([sys.executable, "-c", script], timeout=60, name="fake-llm")
    props = agent.propose(build_context(ws, tracker), 5)
    assert len(props) == 1 and props[0].spec["params"] == {"num_leaves": 15}
    assert props[0].arm.startswith("fake-llm:")

    failing = CompositeAgent(CommandAgent([sys.executable, "-c", "import sys; sys.exit(3)"]), HeuristicAgent())
    assert failing.propose(build_context(ws, tracker), 2)  # falls back to heuristic
    assert "exited 3" in failing.last_error


def test_parse_proposals_rejects_garbage():
    with pytest.raises(ValueError):
        parse_proposals("no json here", "binary", "x")


def test_scheduler_runs_queue_and_selects_best(make_comp, tracker):
    ws = make_comp("binary", n_train=600)
    for m in ("lgbm", "linear"):
        service.queue_experiment(ws, tracker, {"model": m}, f"{m} baseline")
    result = Scheduler(ws, tracker, propose=False, poll_seconds=0.5).run()
    assert result["dispatched"] == 2
    statuses = {e["model_family"]: e["status"] for e in tracker.list_experiments(ws.slug)}
    assert statuses == {"lgbm": "succeeded", "linear": "succeeded"}
    best = tracker.best_experiment(ws.slug, ws.load_validation()["version"], "maximize")
    assert result["best"]["experiment_id"] == best["id"]
    assert (ws.root / "daemon.json").exists()


def test_scheduler_autonomous_loop_respects_budget(make_comp, tracker):
    ws = make_comp("binary", n_train=500)
    cfg = ws.load_config()
    cfg["loop"] = {"tune_trials": 2, "tune_seconds": 60}
    ws.save_config(cfg)
    result = Scheduler(ws, tracker, max_experiments=4, poll_seconds=0.5).run()
    assert result["dispatched"] == 4
    assert "budget" in result["stop_reason"]
    assert all(e["hypothesis"] for e in tracker.list_experiments(ws.slug))


def test_scheduler_enforces_timeout(make_comp, tracker):
    ws = make_comp("binary", n_train=600)
    out = service.queue_experiment(ws, tracker, {"model": "catboost", "max_seconds": 1,
                                                 "params": {"learning_rate": 0.001}}, "too slow")
    Scheduler(ws, tracker, propose=False, poll_seconds=0.5).run()
    assert tracker.get_experiment(out["experiment_id"])["status"] == "timeout"


def test_scheduler_recovers_interrupted_experiment(make_comp, tracker):
    ws = make_comp("binary", n_train=600)
    out = service.queue_experiment(ws, tracker, {"model": "lgbm"}, "x")
    tracker.update_experiment(out["experiment_id"], status="running", pid=999999, attempts=1, heartbeat_at=time.time())
    result = Scheduler(ws, tracker, propose=False, poll_seconds=0.5).run()
    assert result["recovered"][0]["action"] == "requeued"
    assert tracker.get_experiment(out["experiment_id"])["status"] == "succeeded"


def test_graceful_stop_request(make_comp, tracker):
    ws = make_comp("binary", n_train=600)
    service.queue_experiment(ws, tracker, {"model": "lgbm"}, "x")
    sched = Scheduler(ws, tracker, propose=False, poll_seconds=0.5)
    sched.lock.acquire()
    request_stop(ws)
    sched.lock.release()
    # A STOP file left before start is cleared on start, so the run proceeds normally.
    assert sched.run()["dispatched"] == 1
