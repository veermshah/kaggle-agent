"""Optuna tuning on the fixed folds. The study lives in the workspace's optuna.db, so an interrupted tuning
experiment resumes where it stopped instead of starting over."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from experiments.trainer import CVContext, CVResult, cross_validate
from models.tabular import get_family

log = logging.getLogger("kaggle_agent.tuning")


class _SpaceRecorder:
    """Captures a family's search space by calling it with a fake trial."""

    def __init__(self):
        self.space: dict[str, tuple] = {}

    def suggest_float(self, name, low, high, log=False, step=None):
        self.space[name] = ("float", low, high)
        return low

    def suggest_int(self, name, low, high, log=False, step=1):
        self.space[name] = ("int", low, high)
        return low

    def suggest_categorical(self, name, choices):
        self.space[name] = ("cat", list(choices))
        return choices[0]


def fit_to_space(params: dict[str, Any], family, task: str) -> dict[str, Any]:
    """Clip warm-start params into the search space; Optuna rejects out-of-range enqueued values."""
    rec = _SpaceRecorder()
    family.space(rec, task)
    out = {}
    for name, bounds in rec.space.items():
        if name not in params or params[name] is None:
            continue
        value = params[name]
        if bounds[0] == "cat":
            if value in bounds[1]:
                out[name] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            clipped = min(max(value, bounds[1]), bounds[2])
            out[name] = int(round(clipped)) if bounds[0] == "int" else float(clipped)
    return out


def run_tune(ctx: CVContext, spec: dict[str, Any], exp_id: int, device: str, save_dir: Path,
             warm_start: dict[str, Any] | None = None) -> tuple[CVResult, dict[str, Any]]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    family = get_family(spec["model"])
    study = optuna.create_study(
        study_name=f"{ctx.ws.slug}-exp{exp_id}",
        storage=ctx.ws.optuna_url,
        load_if_exists=True,
        direction=ctx.metric.direction,
        sampler=optuna.samplers.TPESampler(seed=spec["seed"], multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    if not study.trials and warm_start:
        seed_params = fit_to_space(warm_start, family, ctx.task)
        if seed_params:
            study.enqueue_trial(seed_params, skip_if_exists=True)

    finished = [t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                         optuna.trial.TrialState.PRUNED)]
    remaining = max(0, spec["n_trials"] - len(finished))
    log.info("tuning %s: %d trials done, %d remaining, timeout %ss", spec["model"], len(finished), remaining,
             spec["timeout"])

    def objective(trial):
        params = {**spec.get("params", {}), **family.space(trial, ctx.task)}
        result = cross_validate(ctx, spec, params, device=device, trial=trial)
        trial.set_user_attr("cv_std", result.cv_std)
        trial.set_user_attr("best_iterations", [b for b in result.best_iterations if b is not None])
        return result.cv_score

    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=spec["timeout"], gc_after_trial=True)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No Optuna trial completed (all pruned or failed); check the training log")
    best_params = {**spec.get("params", {}), **study.best_params}
    log.info("best trial %s value %.6f; refitting with artifacts", study.best_trial.number, study.best_value)
    result = cross_validate(ctx, spec, best_params, device=device, save_dir=save_dir)
    info = {
        "n_trials": len(study.trials),
        "n_complete": len(completed),
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "best_trial_value": study.best_value,
        "best_params": study.best_params,
    }
    return result, info
