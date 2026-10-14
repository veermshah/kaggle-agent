"""Experiment operations shared by the CLI, the scheduler daemon and research agents."""
from __future__ import annotations

import gc
import logging
import os
import sys
import time
import traceback
from typing import Any

import numpy as np

from core.compliance import ComplianceError, check_spec, spec_violations
from core.resources import ResourceMonitor, gpu_capabilities
from core.settings import load_settings
from core.versioning import code_version
from core.workspace import Workspace
from evaluation.metrics import get_metric
from experiments.custom_code import CustomCodeError, declared_requirements
from experiments.spec import SpecError, fingerprint, normalize_spec, resolve_resource
from tracking.db import Tracker, utcnow

log = logging.getLogger("kaggle_agent.service")

ACTIVE = ("queued", "running", "succeeded")


class ExperimentError(RuntimeError):
    pass


def _validation_version(ws: Workspace) -> str:
    return ws.load_validation()["version"]


def auto_members(tracker: Tracker, ws: Workspace, top_k: int) -> list[int]:
    config = ws.load_config()
    rows = [e for e in tracker.list_experiments(ws.slug, "succeeded", _validation_version(ws))
            if e["kind"] != "ensemble" and e["cv_score"] is not None]
    rows.sort(key=lambda e: e["cv_score"], reverse=config["direction"] == "maximize")
    # Over-select; the ensemble step drops near-duplicate members by OOF correlation.
    return [e["id"] for e in rows[: top_k * 2]]


def queue_experiment(ws: Workspace, tracker: Tracker, raw_spec: dict[str, Any], hypothesis: str, *,
                     arm: str | None = None, proposed_by: str = "user", priority: float = 0.0,
                     parent_id: int | None = None, force: bool = False) -> dict[str, Any]:
    if not hypothesis or not hypothesis.strip():
        raise ExperimentError("Every experiment needs a stated hypothesis")
    config = ws.load_config()
    spec = normalize_spec(raw_spec, config["task_type"])
    for recipe in spec.get("features", []):
        if recipe.startswith("custom:"):
            try:
                spec["requires"] = sorted(set(spec["requires"]) | set(declared_requirements(ws, recipe[7:])))
            except FileNotFoundError:
                raise ExperimentError(f"Custom feature module {recipe[7:]}.py not found in {ws.custom_features_dir}")
    if spec["kind"] == "ensemble" and not spec["members"]:
        spec["members"] = auto_members(tracker, ws, spec["top_k"])
        if len(spec["members"]) < 2:
            raise ExperimentError("Need at least two successful experiments before ensembling")

    version = _validation_version(ws)
    fp = fingerprint(spec)
    existing = tracker.find_by_fingerprint(ws.slug, fp, version)
    if existing and not force:
        if existing["status"] in ACTIVE:
            return {"experiment_id": existing["id"], "duplicate": True, "status": existing["status"],
                    "message": "Identical experiment already exists; not queued again"}
        return {"experiment_id": existing["id"], "duplicate": True, "status": existing["status"],
                "message": f"Identical experiment previously ended as {existing['status']}: "
                           f"{(existing.get('error') or '')[:200]}. Use --force to retry."}

    violations = spec_violations(spec, config)
    n_rows = ws.load_profile().get("n_train_rows", 0)
    resource = resolve_resource(spec, n_rows, gpu_capabilities(), load_settings())
    exp_id = tracker.create_experiment(
        ws.slug, spec, fingerprint=fp, hypothesis=hypothesis.strip(), validation_version=version,
        status="blocked" if violations else "queued", arm=arm, proposed_by=proposed_by, priority=priority,
        parent_id=parent_id, resource=resource, error="; ".join(violations) or None,
    )
    tracker.log_event(ws.slug, "queued" if not violations else "blocked",
                      f"exp {exp_id} [{arm or spec['kind']}] {hypothesis.strip()[:160]}")
    return {"experiment_id": exp_id, "duplicate": False, "status": "blocked" if violations else "queued",
            "resource": resource, "fingerprint": fp, "blocked_reasons": violations}


def _setup_file_logging(path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return handler


def _is_oom(text: str) -> bool:
    t = text.lower()
    return any(s in t for s in ("out of memory", "cudaerrormemoryallocation", "cuda error: out of memory",
                                "bad_alloc", "memoryerror"))


def _warm_start(tracker: Tracker, exp: dict, ws: Workspace) -> dict[str, Any] | None:
    """Seed a tuning study with the best known params for this family on the current validation version."""
    rows = [e for e in tracker.list_experiments(ws.slug, "succeeded", exp["validation_version"])
            if e["model_family"] == exp["model_family"] and e["params"]]
    if not rows:
        return None
    direction = ws.load_config()["direction"]
    best = sorted(rows, key=lambda e: e["cv_score"], reverse=direction == "maximize")[0]
    return {k: v for k, v in best["params"].items() if k not in ("n_estimators", "iterations")}


def execute(exp_id: int, tracker: Tracker | None = None) -> dict[str, Any]:
    """Run one experiment to completion in this process (normally a worker subprocess)."""
    from experiments.ensemble import run_ensemble, select_members
    from experiments.trainer import cross_validate, load_context, save_result
    from experiments.tuning import run_tune

    tracker = tracker or Tracker()
    exp = tracker.get_experiment(exp_id)
    if exp is None:
        raise ExperimentError(f"Experiment {exp_id} not found")
    if exp["status"] == "queued":
        tracker.update_experiment(exp_id, status="running", attempts=exp["attempts"] + 1, started_at=utcnow(),
                                  heartbeat_at=time.time(), error=None)
    elif exp["status"] != "running":
        raise ExperimentError(f"Experiment {exp_id} is {exp['status']}; only queued/running experiments can run")
    tracker.update_experiment(exp_id, pid=os.getpid())

    ws = Workspace(exp["competition_id"])
    exp_dir = ws.experiment_dir(exp_id)
    handler = _setup_file_logging(exp_dir / "train.log")
    monitor = ResourceMonitor(heartbeat=lambda: tracker.update_experiment(exp_id, heartbeat_at=time.time()))
    monitor.start()
    t0 = time.time()
    spec = exp["spec"]
    device = "gpu" if exp["resource"] == "gpu" else "cpu"
    log.info("experiment %s start kind=%s model=%s device=%s attempt=%s", exp_id, spec["kind"], spec.get("model"),
             device, exp["attempts"] + (1 if exp["status"] == "queued" else 0))
    log.info("hypothesis: %s", exp["hypothesis"])
    try:
        config = ws.load_config()
        check_spec(spec, config)
        ctx = load_context(ws)
        if ctx.validation_version != exp["validation_version"]:
            raise ExperimentError("Validation scheme changed since this experiment was queued; re-queue it")
        extra: dict[str, Any] = {}
        if spec["kind"] == "train":
            result = cross_validate(ctx, spec, spec.get("params", {}), device=device, save_dir=exp_dir)
        elif spec["kind"] == "tune":
            result, extra = run_tune(ctx, spec, exp_id, device, exp_dir, _warm_start(tracker, exp, ws))
        else:
            candidates = [tracker.get_experiment(i) for i in spec["members"]]
            candidates = [c for c in candidates if c and c["status"] == "succeeded"
                          and c["validation_version"] == ctx.validation_version]
            members = select_members(ctx, candidates, spec["top_k"])
            result, extra = run_ensemble(ctx, spec, members)
        save_result(exp_dir, result, extra)
        monitor.stop()
        tracker.update_experiment(
            exp_id, status="succeeded", cv_score=result.cv_score, cv_std=result.cv_std,
            fold_scores=result.fold_scores, holdout_score=result.holdout_score, params=result.params,
            train_seconds=round(time.time() - t0, 2), peak_gpu_mb=round(monitor.peak_gpu_mb, 1),
            peak_rss_mb=round(monitor.peak_rss_mb, 1), artifact_dir=str(exp_dir), finished_at=utcnow(),
            dataset_version=config.get("dataset_version"), code_version=code_version(), error=None,
        )
        log.info("experiment %s succeeded: %s=%.6f (+/- %.6f) in %.1fs", exp_id, ctx.metric.name, result.cv_score,
                 result.cv_std, time.time() - t0)
        tracker.log_event(ws.slug, "succeeded", f"exp {exp_id} {ctx.metric.name}={result.cv_score:.6f}")
        return {"experiment_id": exp_id, "status": "succeeded", "cv_score": result.cv_score,
                "cv_std": result.cv_std, "train_seconds": round(time.time() - t0, 2)}
    except (ComplianceError, CustomCodeError) as exc:
        tracker.update_experiment(exp_id, status="blocked", error=str(exc), finished_at=utcnow())
        tracker.log_event(ws.slug, "blocked", f"exp {exp_id}: {exc}")
        log.error("blocked: %s", exc)
        return {"experiment_id": exp_id, "status": "blocked", "error": str(exc)}
    except BaseException as exc:  # noqa: BLE001 - record everything, including KeyboardInterrupt
        tb = traceback.format_exc()
        log.error("experiment %s failed:\n%s", exp_id, tb)
        if isinstance(exc, KeyboardInterrupt):
            tracker.update_experiment(exp_id, status="queued", error="interrupted; will resume from checkpoints")
            raise
        if device == "gpu" and _is_oom(tb):
            tracker.update_experiment(exp_id, status="queued", resource="cpu",
                                      error=f"GPU out of memory; re-queued on CPU. {str(exc)[:300]}")
            tracker.log_event(ws.slug, "requeued", f"exp {exp_id} GPU OOM -> CPU")
            return {"experiment_id": exp_id, "status": "queued", "error": "gpu_oom_requeued_cpu"}
        tracker.update_experiment(exp_id, status="failed", error=f"{type(exc).__name__}: {exc}\n{tb[-3000:]}",
                                  finished_at=utcnow(), train_seconds=round(time.time() - t0, 2))
        tracker.log_event(ws.slug, "failed", f"exp {exp_id}: {type(exc).__name__}: {str(exc)[:200]}")
        return {"experiment_id": exp_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        monitor.stop()
        logging.getLogger().removeHandler(handler)
        handler.close()
        gc.collect()
        if "torch" in sys.modules:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


# Queries used by the CLI / agents

PUBLIC_FIELDS = ("id", "status", "kind", "model_family", "feature_set", "arm", "hypothesis", "cv_score", "cv_std",
                 "train_seconds", "resource", "peak_gpu_mb", "attempts", "proposed_by", "parent_id", "promoted",
                 "created_at", "finished_at")


def public_view(exp: dict, show_holdout: bool = False) -> dict[str, Any]:
    out = {k: exp.get(k) for k in PUBLIC_FIELDS}
    if exp.get("error"):
        out["error"] = exp["error"].splitlines()[0][:300]
    if show_holdout or exp.get("promoted"):
        out["holdout_score"] = exp.get("holdout_score")
    return out


def leaderboard(ws: Workspace, tracker: Tracker, limit: int = 20) -> list[dict[str, Any]]:
    config = ws.load_config()
    rows = [e for e in tracker.list_experiments(ws.slug, "succeeded", _validation_version(ws))
            if e["cv_score"] is not None]
    rows.sort(key=lambda e: e["cv_score"], reverse=config["direction"] == "maximize")
    return [public_view(e) for e in rows[:limit]]


def compare(ws: Workspace, tracker: Tracker, ids: list[int]) -> dict[str, Any]:
    metric = get_metric(ws.load_config()["metric"])
    exps = [tracker.get_experiment(i) for i in ids]
    missing = [i for i, e in zip(ids, exps) if e is None or e["status"] != "succeeded"]
    if missing:
        raise ExperimentError(f"Experiments not found or not succeeded: {missing}")
    if len({e["validation_version"] for e in exps}) > 1:
        raise ExperimentError("Experiments use different validation versions; their scores are not comparable")
    base = exps[0]
    rows = []
    for e in exps:
        row = public_view(e)
        row["fold_scores"] = e["fold_scores"]
        if e is not base and e["fold_scores"] and base["fold_scores"]:
            diffs = np.array(e["fold_scores"]) - np.array(base["fold_scores"])
            sign = 1 if metric.greater_is_better else -1
            row["vs_first"] = {
                "mean_fold_delta": float(diffs.mean()),
                "folds_better": int((sign * diffs > 0).sum()),
                "n_folds": len(diffs),
                "delta_over_std": float(diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)) + 1e-12)),
            }
        rows.append(row)
    return {"metric": metric.name, "direction": metric.direction, "experiments": rows}


def promote(ws: Workspace, tracker: Tracker, exp_id: int) -> dict[str, Any]:
    exp = tracker.get_experiment(exp_id)
    if not exp or exp["competition_id"] != ws.slug or exp["status"] != "succeeded":
        raise ExperimentError(f"Experiment {exp_id} is not a succeeded experiment of {ws.slug}")
    metric = get_metric(ws.load_config()["metric"])
    with tracker.conn() as c:
        c.execute("UPDATE experiments SET promoted=0 WHERE competition_id=?", (ws.slug,))
    tracker.update_experiment(exp_id, promoted=1)
    out = {"experiment_id": exp_id, "cv_score": exp["cv_score"], "cv_std": exp["cv_std"],
           "holdout_score": exp["holdout_score"]}
    if exp["holdout_score"] is not None and exp["cv_std"]:
        gap = (exp["cv_score"] - exp["holdout_score"]) * (1 if metric.greater_is_better else -1)
        out["holdout_gap"] = gap
        if gap > 2 * max(exp["cv_std"], 1e-9):
            out["warning"] = ("Holdout is markedly worse than CV: the selection may be overfit to the CV folds. "
                              "Prefer a simpler or ensembled candidate.")
    tracker.log_event(ws.slug, "promoted", f"exp {exp_id} promoted (cv {exp['cv_score']:.6f}, "
                                           f"holdout {exp['holdout_score']})")
    return out


def recover_stale(ws: Workspace, tracker: Tracker, max_attempts: int, stale_seconds: float = 600) -> list[dict]:
    """Experiments left 'running' by a crash/reboot go back to the queue (fold checkpoints are reused)."""
    from core.proc import is_worker_process, kill_tree, pid_alive
    actions = []
    for e in tracker.list_experiments(ws.slug, "running"):
        alive = pid_alive(e["pid"])
        stale = (time.time() - (e["heartbeat_at"] or 0)) > stale_seconds
        if alive and not stale:
            continue
        if alive and is_worker_process(e["pid"], e["id"]):
            kill_tree(e["pid"])  # hung worker; never kill a PID we cannot confirm is ours
        if e["attempts"] >= max_attempts:
            tracker.update_experiment(e["id"], status="failed", finished_at=utcnow(),
                                      error=f"Worker died or hung {e['attempts']} times; giving up")
            actions.append({"experiment_id": e["id"], "action": "failed"})
        else:
            tracker.update_experiment(e["id"], status="queued", pid=None,
                                      error="Recovered after interruption; resuming from checkpoints")
            actions.append({"experiment_id": e["id"], "action": "requeued"})
    if actions:
        tracker.log_event(ws.slug, "recovered", f"recovered {len(actions)} interrupted experiments", actions)
    return actions
