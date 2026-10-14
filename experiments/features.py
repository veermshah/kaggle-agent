"""Named, deterministic feature recipes.

Unsupervised recipes run on train+test together (they never see the target). `target_encoding` is
fold-aware and is applied inside the CV loop by the trainer. `custom:<name>` loads an approved module from
the workspace's custom_features directory (see experiments/custom_code.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from competitions.tabular import TabularData

RECIPES = {
    "missing_indicators": "binary is-missing flags plus a per-row missing count",
    "count_encoding": "frequency of each categorical / low-cardinality value across train+test",
    "datetime_parts": "year, month, day, weekday, hour, day-of-year from datetime columns",
    "row_stats": "row-wise mean/std/min/max/zero-count over numeric columns",
    "numeric_interactions": "pairwise products and ratios of the most important numeric columns",
    "log_numeric": "log1p of skewed non-negative numeric columns",
    "text_basic": "character and word counts of free-text columns",
    "target_encoding": "out-of-fold smoothed target mean for categorical columns (binary/regression)",
}
RARE_MIN_COUNT = 3


@dataclass
class FeatureFrame:
    train: pd.DataFrame
    test: pd.DataFrame
    categorical: list[str]
    target_encode: list[str] = field(default_factory=list)


def validate_recipes(recipes: list[str]) -> list[str]:
    bad = [r for r in recipes if r not in RECIPES and not r.startswith("custom:")]
    if bad:
        raise ValueError(f"Unknown feature recipes {bad}. Available: {sorted(RECIPES)} or custom:<name>")
    return sorted(set(recipes))


def build_features(data: TabularData, recipes: list[str], params: dict[str, Any] | None = None,
                   custom_loader=None) -> FeatureFrame:
    params = params or {}
    recipes = validate_recipes(recipes)
    ctypes = data.column_types
    n_train = len(data.train)
    df = pd.concat([data.train, data.test], ignore_index=True)
    out: dict[str, pd.Series] = {}
    numeric = [c for c in ctypes.get("numeric", []) if c in df.columns]
    categorical = [c for c in ctypes.get("categorical", []) if c in df.columns]

    for c in numeric:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    cat_cols = []
    for c in categorical:
        s = df[c].astype("string")
        counts = s.value_counts()
        rare = counts.index[counts < RARE_MIN_COUNT]
        if len(rare):
            s = s.where(~s.isin(rare), "__rare__")
        out[c] = s.astype("category")
        cat_cols.append(c)

    for c in ctypes.get("datetime", []):
        if c not in df.columns:
            continue
        dt = pd.to_datetime(df[c], errors="coerce", format="mixed")
        out[f"{c}_ts"] = (dt.astype("int64") // 10**9).where(dt.notna()).astype("float32")
        if "datetime_parts" in recipes:
            out[f"{c}_year"] = dt.dt.year.astype("float32")
            out[f"{c}_month"] = dt.dt.month.astype("float32")
            out[f"{c}_day"] = dt.dt.day.astype("float32")
            out[f"{c}_weekday"] = dt.dt.weekday.astype("float32")
            out[f"{c}_dayofyear"] = dt.dt.dayofyear.astype("float32")
            if (dt.dt.hour.fillna(0) != 0).any():
                out[f"{c}_hour"] = dt.dt.hour.astype("float32")

    if "text_basic" in recipes:
        for c in ctypes.get("text", []):
            s = df[c].fillna("").astype(str)
            out[f"{c}_len"] = s.str.len().astype("float32")
            out[f"{c}_words"] = s.str.split().str.len().astype("float32")

    if "missing_indicators" in recipes:
        miss_cols = [c for c in numeric + categorical if df[c].isna().any()]
        for c in miss_cols:
            out[f"{c}_isna"] = df[c].isna().astype("uint8")
        if miss_cols:
            out["row_n_missing"] = df[miss_cols].isna().sum(axis=1).astype("float32")

    if "count_encoding" in recipes:
        for c in categorical + [c for c in numeric if df[c].nunique() <= 255]:
            out[f"{c}_count"] = df[c].map(df[c].value_counts(dropna=False)).astype("float32")

    if "row_stats" in recipes and len(numeric) >= 3:
        m = df[numeric].astype("float32")
        out["row_mean"] = m.mean(axis=1)
        out["row_std"] = m.std(axis=1)
        out["row_min"] = m.min(axis=1)
        out["row_max"] = m.max(axis=1)
        out["row_n_zero"] = (m == 0).sum(axis=1).astype("float32")

    if "numeric_interactions" in recipes:
        cols = [c for c in params.get("interaction_columns", []) if c in numeric] or numeric[:6]
        for a, b in combinations(cols[:6], 2):
            x, y = out[a], out[b]
            out[f"{a}_x_{b}"] = (x * y).astype("float32")
            out[f"{a}_div_{b}"] = (x / y.where(y != 0)).astype("float32")

    if "log_numeric" in recipes:
        for c in numeric:
            s = out[c]
            if s.min() >= 0 and s.skew() > 2:
                out[f"{c}_log1p"] = np.log1p(s).astype("float32")

    for r in recipes:
        if r.startswith("custom:"):
            if custom_loader is None:
                raise ValueError(f"Recipe {r} needs a custom feature loader")
            extra = custom_loader(r.split(":", 1)[1])(df.copy())
            if len(extra) != len(df):
                raise ValueError(f"Custom feature {r} returned {len(extra)} rows, expected {len(df)}")
            for c in extra.columns:
                out[f"{r.split(':', 1)[1]}__{c}"] = extra[c].reset_index(drop=True)

    X = pd.DataFrame(out)
    te = cat_cols if "target_encoding" in recipes and data.task_type in ("binary", "regression") else []
    return FeatureFrame(X.iloc[:n_train].reset_index(drop=True), X.iloc[n_train:].reset_index(drop=True),
                        cat_cols, te)


def _te_stats(cat: pd.Series, y: np.ndarray, prior: float, smoothing: float) -> pd.Series:
    stats = pd.DataFrame({"c": cat.astype("string").fillna("__nan__").to_numpy(), "y": y}).groupby("c")["y"]
    agg = stats.agg(["sum", "count"])
    return (agg["sum"] + prior * smoothing) / (agg["count"] + smoothing)


def target_encode(train_col: pd.Series, y: np.ndarray, others: list[pd.Series], smoothing: float = 20.0,
                  n_inner: int = 5, seed: int = 0) -> tuple[np.ndarray, list[np.ndarray]]:
    """Encode the training rows out-of-fold (inner K-fold) so the model never sees its own row's target;
    other frames (validation/holdout/test) use statistics from all training rows."""
    prior = float(np.mean(y))
    enc_train = np.full(len(train_col), prior, dtype="float32")
    keys = train_col.astype("string").fillna("__nan__").reset_index(drop=True)
    for tr, va in KFold(n_inner, shuffle=True, random_state=seed).split(keys):
        table = _te_stats(keys.iloc[tr], y[tr], prior, smoothing)
        enc_train[va] = keys.iloc[va].map(table).fillna(prior).to_numpy(dtype="float32")
    table = _te_stats(keys, y, prior, smoothing)
    enc_others = [o.astype("string").fillna("__nan__").map(table).fillna(prior).to_numpy(dtype="float32")
                  for o in others]
    return enc_train, enc_others
