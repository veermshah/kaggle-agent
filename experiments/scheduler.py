"""Long-running experiment daemon.

Each experiment runs in its own worker subprocess (crash isolation, VRAM fully released on exit, hard
timeouts). The daemon only schedules: it reaps workers, enforces budgets, asks the research agent for new
proposals when the queue runs dry, and dispatches to one GPU slot plus CPU slots. All state is in SQLite,
so killing the daemon at any point and restarting it resumes the work (fold checkpoints are reused).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from agents.base import build_context, loop_settings
from agents.command import make_agent
from core.proc import PidLock, kill_tree, spawn_worker
from core.resources import disk_free_gb
from core.settings import load_settings, workspaces_root
from core.workspace import Workspace
from evaluation.metrics import get_metric
from experiments.service import ExperimentError, queue_experiment, recover_stale
from tracking.db import Tracker, utcnow

log = logging.getLogger("kaggle_agent.scheduler")


@dataclass
class Running:
    exp_id: int
    proc: Any
    resource: str
    started: float
    max_seconds: float


def stop_file(ws: Workspace):
    return ws.root / "STOP"


def request_stop(ws: Workspace, now: bool = False) -> None:
    stop_file(ws).write_text("now" if now else "graceful")


def daemon_state(ws: Workspace) -> dict[str, Any] | None:
    path = ws.root / "daemon.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    state["alive"] = PidLock(ws.root / "daemon.lock").owner() is not None
    return state


def prune_models(ws: Workspace, tracker: Tracker, keep_top: int) -> dict[str, Any]:
    """Delete fold model binaries outside the top-N (and not promoted/submitted). Predictions are kept, so
    ensembles and submissions still work; pruned experiments can be re-run from their spec."""
    config = ws.load_config()
    rows = [e for e in tracker.list_experiments(ws.slug, "succeeded") if e["cv_score"] is not None]
    rows.sort(key=lambda e: e["cv_score"], reverse=config["direction"] == "maximize")
    submitted = {s["experiment_id"] for s in tracker.list_submissions(ws.slug)}
    freed, pruned = 0, []
    for e in rows[keep_top:]:
        if e["promoted"] or e["id"] in submitted:
            continue
        models = ws.experiment_dir(e["id"]) / "models"
        if models.exists():
            freed += sum(p.stat().st_size for p in models.rglob("*") if p.is_file())
            shutil.rmtree(models)
            pruned.append(e["id"])
    if pruned:
        tracker.log_event(ws.slug, "pruned", f"removed model files of {len(pruned)} experiments", pruned)
    return {"pruned_experiments": pruned, "freed_mb": round(freed / 2**20, 1)}


class Scheduler:
    def __init__(self, ws: Workspace, tracker: Tracker, *, max_hours: float | None = None,
                 max_experiments: int | None = None, propose: bool = True, stop_when_idle: bool = True,
                 agent=None, poll_seconds: float | None = None):
        self.ws, self.tracker = ws, tracker
        self.settings = load_settings()
        self.config = ws.load_config()
        self.loop = loop_settings(self.config)
        self.max_hours = float(max_hours if max_hours is not None else self.loop["max_hours"])
        self.max_experiments = int(max_experiments if max_experiments is not None else self.loop["max_experiments"])
        self.propose_enabled = propose
        self.stop_when_idle = stop_when_idle
        self.agent = agent or make_agent(ws)
        self.poll = float(poll_seconds if poll_seconds is not None else self.loop["poll_seconds"])
        res = self.settings["resources"]
        self.slots = {"gpu": int(res["gpu_slots"]), "cpu": int(res["cpu_slots"])}
        self.default_max_seconds = float(res["max_experiment_seconds"])
        self.min_free_gb = float(res["min_free_disk_gb"])
        self.metric = get_metric(self.config["metric"])
        self.running: dict[int, Running] = {}
        self.dispatched = 0
        self.started = time.time()
        self.exhausted = False
        self.stop_reason: str | None = None
        self.best_seen = None
        self.lock = PidLock(ws.root / "daemon.lock")
        self.gpu_lock = PidLock(workspaces_root() / ".gpu.lock")
        self._last_propose_count = -1

    # Helpers
    def _threads(self, resource: str) -> int:
        cpus = os.cpu_count() or 4
        gpu_threads = max(2, cpus // 4)
        if resource == "gpu":
            return gpu_threads
        return max(1, (cpus - (gpu_threads if self.slots["gpu"] else 0)) // max(1, self.slots["cpu"]))

    def _write_state(self, status: str) -> None:
        state = {
            "pid": os.getpid(), "status": status, "competition": self.ws.slug,
            "started_at": datetime.fromtimestamp(self.started, timezone.utc).isoformat(timespec="seconds"),
            "last_tick": utcnow(), "elapsed_hours": round((time.time() - self.started) / 3600, 3),
            "max_hours": self.max_hours, "dispatched": self.dispatched, "max_experiments": self.max_experiments,
            "running": [{"experiment_id": r.exp_id, "resource": r.resource,
                         "seconds": round(time.time() - r.started, 1)} for r in self.running.values()],
            "agent": self.agent.name, "stop_reason": self.stop_reason, "exhausted": self.exhausted,
        }
        (self.ws.root / "daemon.json").write_text(json.dumps(state, indent=2))

    def _budget_left(self) -> bool:
        if (time.time() - self.started) / 3600 >= self.max_hours:
            self.stop_reason = self.stop_reason or f"max runtime {self.max_hours}h reached"
            return False
        if self.dispatched >= self.max_experiments:
            self.stop_reason = self.stop_reason or f"experiment budget {self.max_experiments} reached"
            return False
        return True

    def _converged(self, ctx) -> bool:
        finished = [e for e in ctx.experiments if e["status"] == "succeeded"]
        limit = int(self.loop["stop_after_no_improvement"])
        if len(finished) < limit:
            return False
        return not any(e["improved"] for e in finished[-limit:])

    # Core loop pieces
    def reap(self) -> None:
        for exp_id, r in list(self.running.items()):
            code = r.proc.poll()
            elapsed = time.time() - r.started
            if code is None and elapsed > r.max_seconds:
                log.warning("experiment %s exceeded %ss; killing", exp_id, r.max_seconds)
                kill_tree(r.proc.pid)
                r.proc.wait(timeout=30)
                self.tracker.update_experiment(exp_id, status="timeout", finished_at=utcnow(),
                                               train_seconds=round(elapsed, 1),
                                               error=f"Exceeded max experiment duration ({r.max_seconds:.0f}s)")
                self.tracker.log_event(self.ws.slug, "timeout", f"exp {exp_id} killed after {elapsed:.0f}s")
                del self.running[exp_id]
                continue
            if code is None:
                continue
            exp = self.tracker.get_experiment(exp_id)
            if exp["status"] == "running":  # worker died without recording an outcome
                max_attempts = int(self.loop["max_attempts"])
                if exp["attempts"] < max_attempts:
                    self.tracker.update_experiment(exp_id, status="queued", pid=None,
                                                   error=f"Worker exited with code {code}; retrying")
                else:
                    self.tracker.update_experiment(exp_id, status="failed", finished_at=utcnow(),
                                                   error=f"Worker exited with code {code} on every attempt; see "
                                                         f"{self.ws.experiment_dir(exp_id) / 'worker.log'}")
                self.tracker.log_event(self.ws.slug, "worker_crash", f"exp {exp_id} worker exit code {code}")
            elif exp["status"] == "succeeded":
                self._on_success(exp)
            del self.running[exp_id]
        if not any(r.resource == "gpu" for r in self.running.values()):
            self.gpu_lock.release()

    def _on_success(self, exp: dict) -> None:
        if not self.metric.better(exp["cv_score"], self.best_seen):
            return
        self.best_seen = exp["cv_score"]
        auto = self.ws.load_config().get("autosubmit", {})
        if auto.get("enabled"):
            from submissions.manager import SubmissionManager
            try:
                outcome = SubmissionManager(self.ws, self.tracker).maybe_autosubmit(exp, self.metric)
                if outcome:
                    log.info("autosubmit: %s", outcome)
            except Exception as exc:
                log.warning("autosubmit failed: %s", exc)

    def propose(self) -> int:
        ctx = build_context(self.ws, self.tracker)
        if self._converged(ctx):
            self.exhausted = True
            self.stop_reason = self.stop_reason or (
                f"no improvement in the last {self.loop['stop_after_no_improvement']} experiments")
            return 0
        n = int(self.loop.get("proposals_per_round", 3))
        queued = 0
        for prop in self.agent.propose(ctx, n):
            try:
                out = queue_experiment(self.ws, self.tracker, prop.spec, prop.hypothesis, arm=prop.arm,
                                       proposed_by=self.agent.name, priority=prop.priority, parent_id=prop.parent_id)
            except (ExperimentError, ValueError) as exc:
                log.info("proposal rejected: %s", exc)
                continue
            if not out["duplicate"] and out["status"] == "queued":
                queued += 1
        self.exhausted = queued == 0
        if self.exhausted:
            self.stop_reason = self.stop_reason or "research agent has no new experiments to propose"
        return queued

    def dispatch(self) -> None:
        for resource in ("gpu", "cpu"):
            while sum(r.resource == resource for r in self.running.values()) < self.slots[resource]:
                if not self._budget_left():
                    return
                if resource == "gpu" and not self.gpu_lock.acquire():
                    break  # another daemon owns the GPU
                exp = self.tracker.claim_next(self.ws.slug, resource)
                if exp is None:
                    break
                max_seconds = float(exp["spec"].get("max_seconds") or self.default_max_seconds)
                if exp["kind"] == "tune":
                    max_seconds = max(max_seconds, float(exp["spec"].get("timeout", 0)) * 1.5 + 600)
                proc = spawn_worker(["worker", "--exp-id", str(exp["id"])],
                                    self.ws.experiment_dir(exp["id"]) / "worker.log",
                                    env={"KAGGLE_AGENT_THREADS": str(self._threads(resource)),
                                         "PYTHONUNBUFFERED": "1"})
                self.tracker.update_experiment(exp["id"], pid=proc.pid)
                self.running[exp["id"]] = Running(exp["id"], proc, resource, time.time(), max_seconds)
                self.dispatched += 1
                log.info("dispatched exp %s on %s (pid %s): %s", exp["id"], resource, proc.pid,
                         exp["hypothesis"][:100])

    def _stop_requested(self) -> str | None:
        path = stop_file(self.ws)
        return path.read_text().strip() or "graceful" if path.exists() else None

    def shutdown_workers(self, reason: str) -> None:
        for exp_id, r in list(self.running.items()):
            kill_tree(r.proc.pid)
            self.tracker.update_experiment(exp_id, status="queued", pid=None,
                                           error=f"Interrupted ({reason}); will resume from checkpoints")
        self.running.clear()
        self.gpu_lock.release()

    def run(self) -> dict[str, Any]:
        if not self.lock.acquire():
            raise ExperimentError(f"A daemon is already running for {self.ws.slug} (PID {self.lock.owner()})")
        stop_file(self.ws).unlink(missing_ok=True)
        recovered = recover_stale(self.ws, self.tracker, int(self.loop["max_attempts"]))
        best = self.tracker.best_experiment(self.ws.slug, self.ws.load_validation()["version"],
                                            self.config["direction"])
        self.best_seen = best["cv_score"] if best else None
        self.tracker.log_event(self.ws.slug, "daemon_start", f"agent={self.agent.name} max_hours={self.max_hours} "
                                                             f"max_experiments={self.max_experiments}")
        errors = 0
        try:
            while True:
                try:
                    self.reap()
                    stop = self._stop_requested()
                    if stop == "now":
                        self.stop_reason = "stop requested (immediate)"
                        self.shutdown_workers("stop requested")
                        break
                    if stop:
                        self.stop_reason = "stop requested (graceful)"
                    can_dispatch = not stop and self._budget_left()
                    if can_dispatch and disk_free_gb(self.ws.root) < self.min_free_gb:
                        self.stop_reason = f"disk free below {self.min_free_gb} GB; dispatch paused"
                        self.tracker.log_event(self.ws.slug, "disk_low", self.stop_reason)
                        can_dispatch = False
                    if can_dispatch:
                        queued = self.tracker.list_experiments(self.ws.slug, "queued")
                        if self.propose_enabled and len(queued) < sum(self.slots.values()):
                            finished_count = len(self.tracker.list_experiments(
                                self.ws.slug, ("succeeded", "failed", "timeout")))
                            if not self.exhausted or finished_count != self._last_propose_count:
                                self._last_propose_count = finished_count
                                self.propose()
                        self.dispatch()
                    self._write_state("running")
                    idle = not self.running and not self.tracker.list_experiments(self.ws.slug, "queued")
                    if not self.running and (stop or not self._budget_left()):
                        break
                    if idle and self.stop_when_idle and (self.exhausted or not self.propose_enabled
                                                         or not can_dispatch):
                        self.stop_reason = self.stop_reason or "queue empty"
                        break
                    errors = 0
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    errors += 1
                    log.exception("scheduler tick failed (%s in a row): %s", errors, exc)
                    self.tracker.log_event(self.ws.slug, "daemon_error", str(exc)[:500])
                    if errors >= 10:
                        self.stop_reason = f"10 consecutive scheduler errors; last: {exc}"
                        self.shutdown_workers("scheduler errors")
                        break
                time.sleep(self.poll)
        except KeyboardInterrupt:
            self.stop_reason = "interrupted (Ctrl+C)"
            self.shutdown_workers("Ctrl+C")
        finally:
            self._write_state("stopped")
            self.tracker.log_event(self.ws.slug, "daemon_stop", self.stop_reason or "stopped")
            stop_file(self.ws).unlink(missing_ok=True)
            self.gpu_lock.release()
            self.lock.release()
        best = self.tracker.best_experiment(self.ws.slug, self.ws.load_validation()["version"],
                                            self.config["direction"])
        return {"stop_reason": self.stop_reason, "dispatched": self.dispatched, "recovered": recovered,
                "elapsed_hours": round((time.time() - self.started) / 3600, 3),
                "best": {"experiment_id": best["id"], "cv_score": best["cv_score"], "model": best["model_family"]}
                if best else None}
