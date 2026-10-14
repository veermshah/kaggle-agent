import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import fit_threshold, get_metric, metric_from_kaggle_name, score
from evaluation.validation import (HOLDOUT, TRAIN_ONLY, ValidationError, choose_scheme, fold_ids, fold_split,
                                   load_folds, make_folds, save_validation)

DEFAULTS = {"n_splits": 5, "seed": 42, "holdout_fraction": 0.1, "holdout_min_rows": 0}


@pytest.mark.parametrize("name,task,expected", [
    ("Area Under Receiver Operating Characteristic Curve", "binary", "roc_auc"),
    ("Root Mean Squared Logarithmic Error", "regression", "rmsle"),
    ("Root Mean Squared Error", "regression", "rmse"),
    ("Mean Absolute Error", "regression", "mae"),
    ("Multi-class Log Loss", "multiclass", "logloss"),
    ("Categorization Accuracy", "binary", "accuracy"),
    ("F1 Score", "multiclass", "f1_macro"),
    ("Quadratic Weighted Kappa", "multiclass", "qwk"),
    ("Something Custom", "binary", None),
])
def test_kaggle_metric_names(name, task, expected):
    assert metric_from_kaggle_name(name, task) == expected


def test_metric_direction_and_better():
    auc, rmse = get_metric("roc_auc"), get_metric("rmse")
    assert auc.better(0.9, 0.8) and not auc.better(0.8, 0.9)
    assert rmse.better(1.0, 2.0) and not rmse.better(2.0, 1.0)
    assert not auc.better(0.8001, 0.8, min_delta=0.001)
    assert auc.better(0.5, None)


def test_label_metric_threshold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.6, 0, 1)  # positives shifted up, best threshold != 0.5
    f1 = get_metric("f1")
    t = fit_threshold(f1, y, p)
    assert score(f1, "binary", y, p, t) >= score(f1, "binary", y, p, 0.5)


def _frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(size=n), "grp": rng.integers(0, 50, n).astype(str),
                         "date": pd.date_range("2020-01-01", periods=n, freq="h").astype(str)}), rng.integers(0, 2, n)


def test_stratified_folds_deterministic_with_holdout():
    df, y = _frame()
    scheme = choose_scheme("binary", {"n_train_rows": len(df)}, {}, DEFAULTS)
    assert scheme["strategy"] == "stratified"
    a, b = make_folds(df, y, scheme, "binary"), make_folds(df, y, scheme, "binary")
    assert (a == b).all()
    assert (a == HOLDOUT).sum() == 100
    assert fold_ids(a) == [0, 1, 2, 3, 4]
    for f in fold_ids(a):
        tr, va = fold_split(a, f, "stratified")
        assert not set(tr) & set(va)
        assert not set(tr) & set(np.where(a == HOLDOUT)[0])


def test_group_folds_keep_groups_together():
    df, y = _frame()
    scheme = choose_scheme("binary", {"n_train_rows": len(df)}, {"strategy": "group", "group_column": "grp"}, DEFAULTS)
    folds = make_folds(df, y, scheme, "binary")
    for g, sub in pd.Series(folds).groupby(df["grp"]):
        assert sub.nunique() == 1, f"group {g} split across folds"


def test_time_folds_train_on_past_only():
    df, y = _frame()
    scheme = choose_scheme("binary", {"n_train_rows": len(df), "suggested_time_column": "date",
                                      "test_after_train": True}, {}, DEFAULTS)
    assert scheme["strategy"] == "time"
    folds = make_folds(df, y, scheme, "binary")
    times = pd.to_datetime(df["date"]).to_numpy()
    assert (folds == TRAIN_ONLY).any()
    assert times[folds == HOLDOUT].min() >= times[folds != HOLDOUT].max()
    for f in fold_ids(folds):
        tr, va = fold_split(folds, f, "time")
        assert times[tr].max() < times[va].min()


def test_validation_is_frozen_and_tamper_evident(home):
    from core.workspace import Workspace
    ws = Workspace("frozen").create()
    df, y = _frame()
    scheme = choose_scheme("binary", {"n_train_rows": len(df)}, {}, DEFAULTS)
    folds = make_folds(df, y, scheme, "binary")
    save_validation(ws, scheme, folds)
    with pytest.raises(ValidationError):
        save_validation(ws, scheme, folds)  # changing folds needs an explicit force
    loaded, _ = load_folds(ws)
    assert (loaded == folds).all()
    tampered = pd.read_csv(ws.folds_path)
    tampered.loc[0, "fold"] = (tampered.loc[0, "fold"] + 1) % 5
    tampered.to_csv(ws.folds_path, index=False)
    with pytest.raises(ValidationError):
        load_folds(ws)
