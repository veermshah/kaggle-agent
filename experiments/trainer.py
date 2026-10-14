"""Cross-validated training on the fixed competition folds, with per-fold checkpoints."""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from competitions.tabular import TabularData, load_tabular
from core.workspace import Workspace
from evaluation.metrics import Metric, fit_threshold, get_metric, score
from evaluation.validation import HOLDOUT, fold_ids, fold_split, load_folds
from experiments.custom_code import loader as custom_loader
from experiments.features import FeatureFrame, build_features, target_encode
from models.tabular import get_family

log = logging.getLogger("kaggle_agent.trainer")


@dataclass
class CVContext:
    ws: Workspace
    config: dict[str, Any]
    data: TabularData
    folds: np.ndarray
    scheme: dict[str, Any]
    metric: Metric
    _features: dict[str, FeatureFrame] = field(default_factory=dict)

    @property
    def task(self) -> str:
        return self.config["task_type"]

    @property
    def validation_version(self) -> str:
        return self.scheme["version"]

    @property
    def holdout_idx(self) -> np.ndarray:
        return np.where(self.folds == HOLDOUT)[0]

    def features(self, recipes: list[str], params: dict[str, Any]) -> FeatureFrame:
        key = json.dumps([recipes, params], sort_keys=True)
        if key not in self._features:
            self._features[key] = build_features(self.data, recipes, params, custom_loader(self.ws))
        return self._features[key]

    def label_range(self) -> tuple[int, int] | None:
        if self.task != "regression":
            return None
        return int(np.floor(self.data.y.min())), int(np.ceil(self.data.y.max()))

    def score(self, y: np.ndarray, raw: np.ndarray, threshold: float = 0.5) -> float:
        return score(self.metric, self.task, y, raw, threshold, self.label_range())


def load_context(ws: Workspace) -> CVContext:
    config = ws.load_config()
    folds, scheme = load_folds(ws)
    data = load_tabular(ws, config)
    if len(folds) != len(data.y):
        raise RuntimeError("folds.csv row count does not match the training data; dataset changed?")
    return CVContext(ws, config, data, folds, scheme, get_metric(config["metric"]))


@dataclass
class CVResult:
    oof: np.ndarray
    test_pred: np.ndarray
    holdout_pred: np.ndarray | None
    fold_scores: list[float]
    cv_score: float
    cv_std: float
    holdout_score: float | None
    threshold: float
    best_iterations: list[int | None]
    importance: dict[str, float]
    n_features: int
    params: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {"cv_score": self.cv_score, "cv_std": self.cv_std, "fold_scores": self.fold_scores,
                "threshold": self.threshold, "best_iterations": self.best_iterations, "n_features": self.n_features,
                "params": self.params, "top_features": dict(list(self.importance.items())[:30])}


def _empty_like(task: str, n: int, k: int, fill: float = np.nan) -> np.ndarray:
    return np.full((n, k), fill) if task == "multiclass" else np.full(n, fill)


def oof_mask(oof: np.ndarray) -> np.ndarray:
    return ~np.isnan(oof).any(axis=1) if oof.ndim == 2 else ~np.isnan(oof)


def postprocess(ctx: CVContext, pred: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    if spec.get("target_transform") == "log1p":
        pred = np.expm1(pred)
    if ctx.task == "regression" and ctx.config.get("target_non_negative"):
        pred = np.clip(pred, 0, None)
    return pred


def cross_validate(ctx: CVContext, spec: dict[str, Any], params: dict[str, Any], device: str = "cpu",
                   save_dir: Path | None = None, trial=None, keep_models: bool = True) -> CVResult:
    family = get_family(spec["model"])
    task, data = ctx.task, ctx.data
    full_params = {**family.defaults(task), **params}
    ff = ctx.features(spec.get("features", []), spec.get("feature_params", {}))
    y = data.y
    y_fit = np.log1p(y) if spec.get("target_transform") == "log1p" else y
    if spec.get("target_transform") == "log1p" and (y < 0).any():
        raise ValueError("log1p target transform needs a non-negative target")

    n, k = len(y), data.n_classes
    hold = ctx.holdout_idx
    oof = _empty_like(task, n, k)
    test_sum = _empty_like(task, len(ff.test), k, 0.0)
    hold_sum = _empty_like(task, len(hold), k, 0.0) if len(hold) else None
    fold_scores, best_iters, importances = [], [], []
    run_key = hashlib.sha256(json.dumps([spec, full_params], sort_keys=True, default=str).encode()).hexdigest()[:16]
    folds_list = fold_ids(ctx.folds)

    for i, f in enumerate(folds_list):
        ckpt = save_dir / f"fold_{f}.npz" if save_dir else None
        tr, va = fold_split(ctx.folds, f, ctx.scheme["strategy"])
        if ckpt and ckpt.exists():
            z = np.load(ckpt, allow_pickle=False)
            if str(z["run_key"]) == run_key:
                oof[va], test_part, hold_part = z["p_va"], z["p_te"], z["p_hold"]
                best_iters.append(int(z["best_iter"]) if int(z["best_iter"]) >= 0 else None)
                importances.append(json.loads(str(z["importance"])))
                log.info("fold %s restored from checkpoint", f)
            else:
                ckpt.unlink()
                z = None
        else:
            z = None
        if z is None:
            t0 = time.time()
            X_tr, X_va = ff.train.iloc[tr].reset_index(drop=True), ff.train.iloc[va].reset_index(drop=True)
            X_te = ff.test
            X_ho = ff.train.iloc[hold].reset_index(drop=True) if len(hold) else None
            if ff.target_encode:
                X_tr, X_va, X_te, X_ho = X_tr.copy(), X_va.copy(), X_te.copy(), (X_ho.copy() if X_ho is not None else None)
                for c in ff.target_encode:
                    others = [X_va[c], X_te[c]] + ([X_ho[c]] if X_ho is not None else [])
                    enc_tr, enc_o = target_encode(X_tr[c], y_fit[tr].astype(float), others, seed=spec["seed"])
                    X_tr[f"{c}_te"], X_va[f"{c}_te"], X_te[f"{c}_te"] = enc_tr, enc_o[0], enc_o[1]
                    if X_ho is not None:
                        X_ho[f"{c}_te"] = enc_o[2]
            fit = family.fit(X_tr, y_fit[tr], X_va, y_fit[va], full_params, task, k, ff.categorical, device,
                             spec["seed"] + f)
            p_va = family.predict(fit.model, X_va, task)
            test_part = family.predict(fit.model, X_te, task)
            hold_part = family.predict(fit.model, X_ho, task) if X_ho is not None else np.zeros(0)
            oof[va] = p_va
            best_iters.append(fit.best_iteration)
            importances.append(dict(list(fit.importance.items())[:200]))
            if ckpt:
                np.savez(ckpt, run_key=run_key, p_va=p_va, p_te=test_part, p_hold=hold_part,
                         best_iter=-1 if fit.best_iteration is None else fit.best_iteration,
                         importance=json.dumps(importances[-1]))
                if keep_models:
                    (save_dir / "models").mkdir(exist_ok=True)
                    joblib.dump(fit.model, save_dir / "models" / f"fold_{f}.joblib", compress=3)
            log.info("fold %s trained in %.1fs (best_iter=%s)", f, time.time() - t0, fit.best_iteration)
            del fit
            gc.collect()

        test_sum += test_part
        if hold_sum is not None:
            hold_sum += hold_part
        fold_pred = postprocess(ctx, oof[va], spec)
        fold_scores.append(ctx.score(y[va], fold_pred))
        log.info("fold %s %s=%.6f", f, ctx.metric.name, fold_scores[-1])
        if trial is not None:
            trial.report(float(np.mean(fold_scores)), i)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    oof = postprocess(ctx, oof, spec)
    test_pred = postprocess(ctx, test_sum / len(folds_list), spec)
    hold_pred = postprocess(ctx, hold_sum / len(folds_list), spec) if hold_sum is not None else None
    mask = oof_mask(oof)
    threshold = fit_threshold(ctx.metric, y[mask], oof[mask]) if task == "binary" else 0.5
    if threshold != 0.5:
        fold_scores = [ctx.score(y[fold_split(ctx.folds, f, ctx.scheme["strategy"])[1]],
                                 oof[fold_split(ctx.folds, f, ctx.scheme["strategy"])[1]], threshold)
                       for f in folds_list]
    cv = ctx.score(y[mask], oof[mask], threshold)
    hold_score = ctx.score(y[hold], hold_pred, threshold) if hold_pred is not None else None

    agg: dict[str, float] = {}
    for imp in importances:
        for name, v in imp.items():
            agg[name] = agg.get(name, 0.0) + v / len(importances)
    importance = dict(sorted(agg.items(), key=lambda t: -t[1]))
    return CVResult(oof, test_pred, hold_pred, [float(s) for s in fold_scores], float(cv),
                    float(np.std(fold_scores)), hold_score, threshold, best_iters, importance,
                    ff.train.shape[1] + len(ff.target_encode), full_params)


def save_result(save_dir: Path, result: CVResult, extra: dict[str, Any] | None = None) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "oof.npy", result.oof)
    np.save(save_dir / "test.npy", result.test_pred)
    if result.holdout_pred is not None:
        np.save(save_dir / "holdout.npy", result.holdout_pred)
    payload = {**result.summary(), "holdout_score": result.holdout_score, **(extra or {})}
    (save_dir / "result.json").write_text(json.dumps(payload, indent=2, default=float))


def load_predictions(exp_dir: Path) -> dict[str, Any]:
    out = {"oof": np.load(exp_dir / "oof.npy"), "test": np.load(exp_dir / "test.npy")}
    out["holdout"] = np.load(exp_dir / "holdout.npy") if (exp_dir / "holdout.npy").exists() else None
    out["result"] = json.loads((exp_dir / "result.json").read_text())
    return out
