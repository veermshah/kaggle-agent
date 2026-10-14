"""Validation schemes. Folds are created once per competition, written to folds.csv and hashed; every
experiment records that hash, and scores are only ever compared within the same validation version.

Fold codes: k >= 0 validation fold, -1 final holdout (never used for selection), -2 train-only rows
(earliest block in time-series validation, which has no earlier data to be validated against).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)

from core.workspace import Workspace

STRATEGIES = ("stratified", "kfold", "group", "time", "holdout")
HOLDOUT, TRAIN_ONLY = -1, -2


class ValidationError(RuntimeError):
    pass


def choose_scheme(task_type: str, profile: dict[str, Any], overrides: dict[str, Any],
                  defaults: dict[str, Any]) -> dict[str, Any]:
    scheme = {
        "n_splits": int(overrides.get("n_splits") or defaults["n_splits"]),
        "seed": int(overrides.get("seed", defaults["seed"])),
        "group_column": overrides.get("group_column") or profile.get("suggested_group_column"),
        "time_column": overrides.get("time_column") or profile.get("suggested_time_column"),
    }
    n_rows = profile.get("n_train_rows", 0)
    if overrides.get("strategy"):
        strategy, reason = overrides["strategy"], "set in competition config"
    elif scheme["time_column"] and profile.get("test_after_train"):
        strategy, reason = "time", f"test rows are later in time than train ({scheme['time_column']})"
    elif scheme["group_column"]:
        strategy, reason = "group", f"'{scheme['group_column']}' repeats within train and is disjoint from test"
    elif n_rows and n_rows < 200:
        strategy, reason = "kfold" if task_type == "regression" else "stratified", "small dataset"
    elif task_type in ("binary", "multiclass"):
        strategy, reason = "stratified", "classification target; preserve class balance per fold"
    else:
        strategy, reason = "kfold", "regression target"
    if strategy not in STRATEGIES:
        raise ValidationError(f"Unknown validation strategy {strategy!r}; choose from {STRATEGIES}")
    if strategy == "group" and not scheme["group_column"]:
        raise ValidationError("Group validation requires validation.group_column in competition.yaml")
    if strategy == "time" and not scheme["time_column"]:
        raise ValidationError("Time validation requires validation.time_column in competition.yaml")

    frac = overrides.get("holdout_fraction", defaults["holdout_fraction"])
    if n_rows < defaults.get("holdout_min_rows", 0) and "holdout_fraction" not in overrides:
        frac = 0.0
    scheme.update(strategy=strategy, reason=reason, holdout_fraction=float(frac))
    return scheme


def make_folds(train: pd.DataFrame, y: np.ndarray, scheme: dict[str, Any], task_type: str) -> np.ndarray:
    n = len(train)
    folds = np.full(n, -9, dtype=int)
    rng_seed = scheme["seed"]
    strategy = scheme["strategy"]
    frac = scheme.get("holdout_fraction", 0.0)
    stratify = y if task_type in ("binary", "multiclass") else None
    idx = np.arange(n)
    groups = train[scheme["group_column"]].astype(str).to_numpy() if strategy == "group" else None

    # 1) final holdout
    if frac > 0:
        if strategy == "time":
            order = np.argsort(_time_values(train, scheme["time_column"]), kind="stable")
            hold = order[int(round(n * (1 - frac))):]
        elif strategy == "group":
            gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=rng_seed)
            _, hold = next(gss.split(idx, groups=groups))
        else:
            _, hold = train_test_split(idx, test_size=frac, random_state=rng_seed,
                                       stratify=stratify if _can_stratify(stratify, frac) else None)
        folds[hold] = HOLDOUT
    rest = idx[folds != HOLDOUT]

    # 2) CV folds on the remaining rows
    k = scheme["n_splits"]
    if strategy == "holdout":
        tr, va = train_test_split(rest, test_size=0.2, random_state=rng_seed,
                                  stratify=stratify[rest] if _can_stratify(stratify, 0.2) else None)
        folds[tr], folds[va] = TRAIN_ONLY, 0
    elif strategy == "time":
        order = rest[np.argsort(_time_values(train.iloc[rest], scheme["time_column"]), kind="stable")]
        blocks = np.array_split(order, k + 1)
        folds[blocks[0]] = TRAIN_ONLY
        for f, block in enumerate(blocks[1:]):
            folds[block] = f
    else:
        if strategy == "group":
            splitter = (StratifiedGroupKFold(k, shuffle=True, random_state=rng_seed)
                        if stratify is not None else GroupKFold(k))
            splits = splitter.split(rest, stratify[rest] if stratify is not None else None, groups[rest])
        elif strategy == "stratified" and stratify is not None:
            splits = StratifiedKFold(k, shuffle=True, random_state=rng_seed).split(rest, stratify[rest])
        else:
            splits = KFold(k, shuffle=True, random_state=rng_seed).split(rest)
        for f, (_, va) in enumerate(splits):
            folds[rest[va]] = f
    if (folds == -9).any():
        raise ValidationError("Internal error: some rows were not assigned a fold")
    return folds


def _can_stratify(y: np.ndarray | None, frac: float) -> bool:
    if y is None:
        return False
    _, counts = np.unique(y, return_counts=True)
    return counts.min() >= 2 and counts.min() * frac >= 1


def _time_values(df: pd.DataFrame, col: str) -> np.ndarray:
    values = df[col]
    if not pd.api.types.is_numeric_dtype(values):
        values = pd.to_datetime(values, errors="coerce")
    return values.to_numpy()


def fold_ids(folds: np.ndarray) -> list[int]:
    return sorted(int(f) for f in np.unique(folds) if f >= 0)


def fold_split(folds: np.ndarray, k: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Training and validation row indices for fold k. Holdout rows never appear in either."""
    va = np.where(folds == k)[0]
    if strategy == "time":
        tr = np.where((folds == TRAIN_ONLY) | ((folds >= 0) & (folds < k)))[0]
    else:
        tr = np.where((folds != k) & (folds != HOLDOUT))[0]
    return tr, va


def version_hash(scheme: dict[str, Any], folds: np.ndarray) -> str:
    keys = ("strategy", "n_splits", "seed", "group_column", "time_column", "holdout_fraction")
    h = hashlib.sha256(json.dumps({k: scheme.get(k) for k in keys}, sort_keys=True, default=str).encode())
    h.update(folds.astype(np.int16).tobytes())
    return h.hexdigest()[:12]


def save_validation(ws: Workspace, scheme: dict[str, Any], folds: np.ndarray, force: bool = False) -> str:
    if ws.validation_path.exists() and not force:
        raise ValidationError(
            "A validation scheme already exists. Changing folds makes earlier scores incomparable; "
            "pass --force to create a new validation version deliberately."
        )
    version = version_hash(scheme, folds)
    pd.DataFrame({"row": np.arange(len(folds)), "fold": folds}).to_csv(ws.folds_path, index=False)
    counts = {str(k): int(v) for k, v in zip(*np.unique(folds, return_counts=True))}
    ws.validation_path.write_text(json.dumps({**scheme, "version": version, "fold_counts": counts}, indent=2))
    return version


def load_folds(ws: Workspace) -> tuple[np.ndarray, dict[str, Any]]:
    scheme = ws.load_validation()
    folds = pd.read_csv(ws.folds_path)["fold"].to_numpy()
    if version_hash(scheme, folds) != scheme["version"]:
        raise ValidationError("folds.csv does not match validation.json (modified on disk?). Refusing to train.")
    return folds, scheme
