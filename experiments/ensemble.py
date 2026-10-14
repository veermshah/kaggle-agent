"""Blending out-of-fold predictions.

Weights are fit with nested cross-validation over the same folds: for each fold, weights come from the
other folds' OOF rows only, so the reported ensemble CV score is not inflated by fitting weights on the
rows it is scored on. Final weights (for test/holdout) are then fit on all OOF rows.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from evaluation.validation import fold_ids
from experiments.trainer import CVContext, CVResult, load_predictions, oof_mask

log = logging.getLogger("kaggle_agent.ensemble")


def _blend(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    out = sum(w * p for w, p in zip(weights, preds) if w > 0)
    if out.ndim == 2:
        out = out / out.sum(axis=1, keepdims=True)
    return out


def _ranks(p: np.ndarray) -> np.ndarray:
    return (np.argsort(np.argsort(p)) + 0.5) / len(p)


def hill_climb(ctx: CVContext, preds: list[np.ndarray], y: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Caruana ensemble selection with replacement, starting from the best single model."""
    m = ctx.metric
    scores = [ctx.score(y, p) for p in preds]
    best = int(np.argmax(scores) if m.greater_is_better else np.argmin(scores))
    counts = np.zeros(len(preds))
    counts[best] = 1
    current, current_score = preds[best].copy(), scores[best]
    for _ in range(max_iter):
        total = counts.sum()
        cand_j, cand_score = None, None
        for j, p in enumerate(preds):
            s = ctx.score(y, (current * total + p) / (total + 1))
            if cand_score is None or m.better(s, cand_score):
                cand_j, cand_score = j, s
        if not m.better(cand_score, current_score, 1e-9):
            break
        current = (current * total + preds[cand_j]) / (total + 1)
        counts[cand_j] += 1
        current_score = cand_score
    return counts / counts.sum()


def fit_weights(ctx: CVContext, method: str, preds: list[np.ndarray], y: np.ndarray) -> np.ndarray:
    if method in ("mean", "rank_mean"):
        return np.full(len(preds), 1 / len(preds))
    return hill_climb(ctx, preds, y)


def _platt(p_fit: np.ndarray, y_fit: np.ndarray, p_apply: list[np.ndarray]) -> list[np.ndarray]:
    lr = LogisticRegression(C=1e4)
    lr.fit(logit(np.clip(p_fit, 1e-6, 1 - 1e-6)).reshape(-1, 1), y_fit)
    return [lr.predict_proba(logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1] for p in p_apply]


def run_ensemble(ctx: CVContext, spec: dict[str, Any], members: list[dict]) -> tuple[CVResult, dict[str, Any]]:
    if len(members) < 2:
        raise ValueError("An ensemble needs at least two successful member experiments")
    method = spec["method"]
    if method == "rank_mean" and ctx.metric.name != "roc_auc":
        raise ValueError("rank_mean is only meaningful for rank-based metrics (roc_auc)")
    if spec.get("calibrate") and not (ctx.task == "binary" and ctx.metric.needs == "proba"):
        raise ValueError("calibration applies to binary probability metrics only")

    loaded = [load_predictions(ctx.ws.experiment_dir(m["id"])) for m in members]
    y, folds = ctx.data.y, ctx.folds
    mask = np.logical_and.reduce([oof_mask(p["oof"]) for p in loaded])
    oofs = [p["oof"] for p in loaded]
    tests = [p["test"] for p in loaded]
    holds = [p["holdout"] for p in loaded]
    if method == "rank_mean":
        oofs = [np.full(len(y), np.nan) for _ in loaded]
        for o, p in zip(oofs, loaded):
            o[mask] = _ranks(p["oof"][mask])
        tests = [_ranks(t) for t in tests]
        holds = [_ranks(h) if h is not None else None for h in holds]

    # Nested CV: weights for fold f are learned on the other folds.
    ens_oof = np.full_like(oofs[0], np.nan, dtype=float)
    for f in fold_ids(folds):
        fit_rows = mask & (folds != f) & (folds >= 0)
        apply_rows = mask & (folds == f)
        w = fit_weights(ctx, method, [o[fit_rows] for o in oofs], y[fit_rows])
        blended_fit = _blend([o[fit_rows] for o in oofs], w)
        blended_apply = _blend([o[apply_rows] for o in oofs], w)
        if spec.get("calibrate"):
            blended_apply = _platt(blended_fit, y[fit_rows], [blended_apply])[0]
        ens_oof[apply_rows] = blended_apply

    weights = fit_weights(ctx, method, [o[mask] for o in oofs], y[mask])
    test_pred = _blend(tests, weights)
    hold_pred = _blend(holds, weights) if all(h is not None for h in holds) else None
    if spec.get("calibrate"):
        full = _blend([o[mask] for o in oofs], weights)
        calibrated = _platt(full, y[mask], [test_pred] + ([hold_pred] if hold_pred is not None else []))
        test_pred = calibrated[0]
        hold_pred = calibrated[1] if hold_pred is not None else None

    from evaluation.metrics import fit_threshold
    from evaluation.validation import fold_split
    threshold = fit_threshold(ctx.metric, y[mask], ens_oof[mask]) if ctx.task == "binary" else 0.5
    fold_scores = []
    for f in fold_ids(folds):
        va = fold_split(folds, f, ctx.scheme["strategy"])[1]
        va = va[mask[va]]
        fold_scores.append(ctx.score(y[va], ens_oof[va], threshold))
    cv = ctx.score(y[mask], ens_oof[mask], threshold)
    hold_idx = ctx.holdout_idx
    hold_score = ctx.score(y[hold_idx], hold_pred, threshold) if hold_pred is not None and len(hold_idx) else None
    weight_map = {str(m["id"]): round(float(w), 4) for m, w in zip(members, weights)}
    result = CVResult(ens_oof, test_pred, hold_pred, [float(s) for s in fold_scores], float(cv),
                      float(np.std(fold_scores)), hold_score, threshold, [], {}, len(members),
                      {"weights": weight_map, "method": method, "calibrate": bool(spec.get("calibrate"))})
    member_scores = {str(m["id"]): m["cv_score"] for m in members}
    return result, {"weights": weight_map, "member_cv": member_scores}


def select_members(ctx: CVContext, candidates: list[dict], top_k: int, max_corr: float = 0.995) -> list[dict]:
    """Best experiments first, skipping near-duplicates (highly correlated OOF) to keep the blend diverse."""
    ranked = sorted(candidates, key=lambda e: e["cv_score"], reverse=ctx.metric.greater_is_better)
    chosen, chosen_oof = [], []
    for exp in ranked:
        path = ctx.ws.experiment_dir(exp["id"]) / "oof.npy"
        if not path.exists():
            continue
        oof = np.load(path)
        flat = oof.reshape(len(oof), -1)
        mask = ~np.isnan(flat).any(axis=1)
        if any(np.corrcoef(flat[mask].ravel(), c[mask].ravel())[0, 1] > max_corr for c in chosen_oof):
            continue
        chosen.append(exp)
        chosen_oof.append(flat)
        if len(chosen) >= top_k:
            break
    return chosen
