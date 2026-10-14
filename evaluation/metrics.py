"""Metric registry. Models always emit raw predictions (probabilities or values); metrics that need labels
convert internally, using a decision threshold fitted on out-of-fold predictions for binary tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    name: str
    direction: str            # maximize | minimize
    needs: str                # proba | label | value
    tasks: tuple[str, ...]
    fn: Callable[[np.ndarray, np.ndarray], float]

    @property
    def greater_is_better(self) -> bool:
        return self.direction == "maximize"

    def better(self, a: float | None, b: float | None, min_delta: float = 0.0) -> bool:
        """True if score a beats score b by more than min_delta."""
        if a is None or not np.isfinite(a):
            return False
        if b is None or not np.isfinite(b):
            return True
        return a > b + min_delta if self.greater_is_better else a < b - min_delta


def _logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    if p.ndim == 2:
        p = p / p.sum(axis=1, keepdims=True)
        return float(skm.log_loss(y, p, labels=np.arange(p.shape[1])))
    return float(skm.log_loss(y, p, labels=[0, 1]))


def _auc(y, p):
    if p.ndim == 2:
        return float(skm.roc_auc_score(y, p, multi_class="ovr", average="macro", labels=np.arange(p.shape[1])))
    return float(skm.roc_auc_score(y, p))


def _rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2)))


def _qwk(y, p):
    return float(skm.cohen_kappa_score(y, p, weights="quadratic"))


CLS = ("binary", "multiclass")
REG = ("regression",)

METRICS: dict[str, Metric] = {m.name: m for m in [
    Metric("roc_auc", "maximize", "proba", CLS, _auc),
    Metric("logloss", "minimize", "proba", CLS, _logloss),
    Metric("accuracy", "maximize", "label", CLS, lambda y, p: float(skm.accuracy_score(y, p))),
    Metric("balanced_accuracy", "maximize", "label", CLS, lambda y, p: float(skm.balanced_accuracy_score(y, p))),
    Metric("f1", "maximize", "label", ("binary",), lambda y, p: float(skm.f1_score(y, p))),
    Metric("f1_macro", "maximize", "label", CLS, lambda y, p: float(skm.f1_score(y, p, average="macro"))),
    Metric("f1_micro", "maximize", "label", CLS, lambda y, p: float(skm.f1_score(y, p, average="micro"))),
    Metric("mcc", "maximize", "label", CLS, lambda y, p: float(skm.matthews_corrcoef(y, p))),
    Metric("qwk", "maximize", "label", CLS + REG, _qwk),
    Metric("rmse", "minimize", "value", REG, lambda y, p: float(np.sqrt(skm.mean_squared_error(y, p)))),
    Metric("mse", "minimize", "value", REG, lambda y, p: float(skm.mean_squared_error(y, p))),
    Metric("mae", "minimize", "value", REG, lambda y, p: float(skm.mean_absolute_error(y, p))),
    Metric("rmsle", "minimize", "value", REG, _rmsle),
    Metric("mape", "minimize", "value", REG, lambda y, p: float(skm.mean_absolute_percentage_error(y, p))),
    Metric("r2", "maximize", "value", REG, lambda y, p: float(skm.r2_score(y, p))),
    Metric("medae", "minimize", "value", REG, lambda y, p: float(skm.median_absolute_error(y, p))),
]}

# Ordered: more specific phrases first.
KAGGLE_NAME_HINTS: list[tuple[str, str]] = [
    ("root mean squared logarithmic", "rmsle"), ("rmsle", "rmsle"),
    ("root mean squared", "rmse"), ("root mean square", "rmse"), ("rmse", "rmse"),
    ("mean squared error", "mse"), ("mse", "mse"),
    ("mean absolute percentage", "mape"), ("mape", "mape"),
    ("median absolute", "medae"),
    ("mean absolute error", "mae"), ("mae", "mae"),
    ("quadratic weighted kappa", "qwk"), ("kappa", "qwk"),
    ("matthews", "mcc"), ("mcc", "mcc"),
    ("balanced accuracy", "balanced_accuracy"),
    ("micro", "f1_micro"), ("macro f", "f1_macro"), ("f1", "f1"), ("f-score", "f1"), ("fscore", "f1"),
    ("area under", "roc_auc"), ("auc", "roc_auc"), ("roc", "roc_auc"),
    ("log loss", "logloss"), ("logloss", "logloss"), ("logarithmic loss", "logloss"), ("cross-entropy", "logloss"),
    ("cross entropy", "logloss"),
    ("accuracy", "accuracy"),
    ("r2", "r2"), ("r^2", "r2"), ("coefficient of determination", "r2"),
]

DEFAULT_METRIC = {"binary": "roc_auc", "multiclass": "logloss", "regression": "rmse"}


class MetricError(ValueError):
    pass


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise MetricError(f"Unknown metric {name!r}. Known: {', '.join(sorted(METRICS))}") from None


def metric_from_kaggle_name(name: str | None, task_type: str) -> str | None:
    if not name:
        return None
    lowered = name.lower()
    for hint, metric in KAGGLE_NAME_HINTS:
        if hint in lowered:
            if metric == "f1" and task_type == "multiclass":
                return "f1_macro"
            if task_type in get_metric(metric).tasks:
                return metric
    return None


def to_labels(raw: np.ndarray, task_type: str, threshold: float = 0.5) -> np.ndarray:
    if task_type == "binary":
        return (raw >= threshold).astype(int)
    if task_type == "multiclass":
        return np.argmax(raw, axis=1)
    return np.rint(raw).astype(int)


def score(metric: Metric, task_type: str, y_true: np.ndarray, raw: np.ndarray,
          threshold: float = 0.5, label_range: tuple[int, int] | None = None) -> float:
    if metric.needs == "label":
        pred = to_labels(raw, task_type, threshold)
        if task_type == "regression" and label_range is not None:
            pred = np.clip(pred, *label_range)
        y = np.rint(y_true).astype(int) if task_type == "regression" else y_true
        return metric.fn(y, pred)
    return metric.fn(y_true, raw)


def fit_threshold(metric: Metric, y_true: np.ndarray, raw: np.ndarray) -> float:
    """Pick the binary decision threshold that maximizes a label metric on out-of-fold predictions."""
    if metric.needs != "label" or raw.ndim != 1:
        return 0.5
    candidates = np.unique(np.quantile(raw, np.linspace(0.02, 0.98, 97)))
    best_t, best_s = 0.5, metric.fn(y_true, (raw >= 0.5).astype(int))
    for t in candidates:
        s = metric.fn(y_true, (raw >= t).astype(int))
        if metric.better(s, best_s):
            best_t, best_s = float(t), s
    return best_t


def score_submission_frame(sub, solution, metric: Metric) -> float:
    """Score a submission CSV against a solution CSV (used by simulated submissions)."""
    key = solution.columns[0]
    merged = solution.merge(sub, on=key, suffixes=("_true", "_pred"))
    if len(merged) != len(solution):
        raise MetricError("Submission ids do not match the solution file.")
    true_cols = [c for c in merged.columns if c.endswith("_true")]
    if len(true_cols) != 1:
        raise MetricError("Simulated scoring supports single-target solutions only.")
    target = true_cols[0][: -len("_true")]
    y, p = merged[f"{target}_true"].to_numpy(), merged[f"{target}_pred"].to_numpy()
    if metric.needs == "label":
        cats = {v: i for i, v in enumerate(sorted(set(map(str, y)) | set(map(str, p))))}
        y, p = np.array([cats[str(v)] for v in y]), np.array([cats[str(v)] for v in p])
    return metric.fn(y.astype(float) if metric.needs != "label" else y, p.astype(float) if metric.needs != "label" else p)
