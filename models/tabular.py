"""Tabular model families: defaults, Optuna search spaces, fit/predict with early stopping."""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

EARLY_STOP = 200


def _threads() -> int:
    return int(os.environ.get("KAGGLE_AGENT_THREADS", 0)) or max(1, (os.cpu_count() or 2) - 1)


@dataclass
class FitResult:
    model: Any
    best_iteration: int | None
    importance: dict[str, float]


@dataclass
class ModelFamily:
    name: str
    gpu_capable: bool
    defaults: Callable[[str], dict]
    space: Callable[[Any, str], dict]
    fit: Callable[..., FitResult]
    predict: Callable[[Any, pd.DataFrame, str], np.ndarray]
    max_rows: int | None = None       # skip in baselines above this many rows
    description: str = ""


def _proba(model, X, task):
    if task == "regression":
        return np.asarray(model.predict(X), dtype=float)
    p = np.asarray(model.predict_proba(X), dtype=float)
    return p[:, 1] if task == "binary" else p


def _importance(names, values) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    total = values.sum() or 1.0
    return {n: float(v / total) for n, v in sorted(zip(names, values), key=lambda t: -t[1])}


# LightGBM

def _lgbm_defaults(task):
    return dict(n_estimators=10000, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0)


def _lgbm_space(trial, task):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 256, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]),
    )


def _lgbm_fit(X_tr, y_tr, X_va, y_va, params, task, n_classes, cat_cols, device, seed):
    import lightgbm as lgb
    params = {**params, "random_state": seed, "n_jobs": _threads(), "verbose": -1}
    cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    if task == "multiclass":
        params.setdefault("objective", "multiclass")
    model = cls(**params)
    eval_kwargs = ({"eval_X": X_va, "eval_y": y_va} if "eval_X" in inspect.signature(model.fit).parameters
                   else {"eval_set": [(X_va, y_va)]})
    model.fit(X_tr, y_tr, **eval_kwargs, categorical_feature=cat_cols or "auto",
              callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    return FitResult(model, model.best_iteration_, _importance(X_tr.columns, model.booster_.feature_importance("gain")))


# XGBoost

def _xgb_defaults(task):
    return dict(n_estimators=10000, learning_rate=0.05, max_depth=6, min_child_weight=1.0, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0, max_bin=256)


def _xgb_space(trial, task):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 50, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        gamma=trial.suggest_float("gamma", 1e-4, 5.0, log=True),
    )


def _xgb_fit(X_tr, y_tr, X_va, y_va, params, task, n_classes, cat_cols, device, seed):
    import xgboost as xgb
    params = {**params, "random_state": seed, "tree_method": "hist", "device": "cuda" if device == "gpu" else "cpu",
              "enable_categorical": True, "max_cat_to_onehot": 1, "early_stopping_rounds": EARLY_STOP,
              "n_jobs": _threads()}
    cls = xgb.XGBRegressor if task == "regression" else xgb.XGBClassifier
    model = cls(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    if device == "gpu":
        model.set_params(device="cpu")  # predict on CPU: avoids host/device copy warnings for pandas input
    score = model.get_booster().get_score(importance_type="gain")
    return FitResult(model, int(model.best_iteration), _importance(list(score), list(score.values())))


# CatBoost

def _cat_prepare(X, cat_cols):
    if not cat_cols:
        return X
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__nan__").astype(str)
    return X


def _cb_defaults(task):
    return dict(iterations=10000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0)


def _cb_space(trial, task):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
        random_strength=trial.suggest_float("random_strength", 1e-3, 10, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        border_count=trial.suggest_categorical("border_count", [64, 128, 254]),
    )


def _cb_fit(X_tr, y_tr, X_va, y_va, params, task, n_classes, cat_cols, device, seed):
    import catboost as cb
    loss = {"binary": "Logloss", "multiclass": "MultiClass", "regression": "RMSE"}[task]
    params = {"loss_function": loss, **params, "random_seed": seed, "od_type": "Iter", "od_wait": EARLY_STOP,
              "use_best_model": True, "verbose": 0, "allow_writing_files": False,
              "task_type": "GPU" if device == "gpu" else "CPU"}
    if device != "gpu":
        params["thread_count"] = _threads()
    cls = cb.CatBoostRegressor if task == "regression" else cb.CatBoostClassifier
    model = cls(**params, cat_features=cat_cols or None)
    model.fit(_cat_prepare(X_tr, cat_cols), y_tr, eval_set=(_cat_prepare(X_va, cat_cols), y_va))
    model._agent_cat_cols = cat_cols
    return FitResult(model, int(model.get_best_iteration() or 0), _importance(X_tr.columns, model.get_feature_importance()))


def _cb_predict(model, X, task):
    return _proba(model, _cat_prepare(X, getattr(model, "_agent_cat_cols", [])), task)


# scikit-learn families (random forest, extra trees, linear)

def _sk_matrix(columns, cat_cols, scale: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    numeric_cols = [c for c in columns if c not in cat_cols]
    if scale:
        num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
        cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="__nan__"),
                            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, max_categories=50,
                                          sparse_output=False))
    else:
        num = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="__nan__"),
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    return ColumnTransformer([("num", num, numeric_cols), ("cat", cat, list(cat_cols))])


def _sk_X(X, cat_cols):
    if not cat_cols:
        return X
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype("object").where(X[c].notna(), "__nan__").astype(str)
    return X


def _forest_fit_factory(extra: bool):
    def fit(X_tr, y_tr, X_va, y_va, params, task, n_classes, cat_cols, device, seed):
        from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier,
                                      RandomForestRegressor)
        from sklearn.pipeline import make_pipeline
        if task == "regression":
            cls = ExtraTreesRegressor if extra else RandomForestRegressor
        else:
            cls = ExtraTreesClassifier if extra else RandomForestClassifier
        est = cls(**params, random_state=seed, n_jobs=_threads())
        model = make_pipeline(_sk_matrix(X_tr.columns, cat_cols, scale=False), est)
        model.fit(_sk_X(X_tr, cat_cols), y_tr)
        model._agent_cat_cols = cat_cols
        names = model[0].get_feature_names_out()
        return FitResult(model, None, _importance(names, est.feature_importances_))
    return fit


def _forest_defaults(task):
    return dict(n_estimators=400, max_features="sqrt" if task != "regression" else 0.5, min_samples_leaf=2)


def _forest_space(trial, task):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        max_features=trial.suggest_float("max_features", 0.1, 1.0),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30, log=True),
        max_depth=trial.suggest_categorical("max_depth", [None, 8, 16, 32]),
    )


def _linear_defaults(task):
    return dict(C=1.0) if task != "regression" else dict(alpha=1.0)


def _linear_space(trial, task):
    if task == "regression":
        return dict(alpha=trial.suggest_float("alpha", 1e-3, 1e3, log=True))
    return dict(C=trial.suggest_float("C", 1e-3, 1e2, log=True))


def _linear_fit(X_tr, y_tr, X_va, y_va, params, task, n_classes, cat_cols, device, seed):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    est = Ridge(**params) if task == "regression" else LogisticRegression(**params, max_iter=3000)
    model = make_pipeline(_sk_matrix(X_tr.columns, cat_cols, scale=True), est)
    model.fit(_sk_X(X_tr, cat_cols), y_tr)
    model._agent_cat_cols = cat_cols
    coef = np.abs(np.atleast_2d(est.coef_)).mean(axis=0)
    return FitResult(model, None, _importance(model[0].get_feature_names_out(), coef))


def _sk_predict(model, X, task):
    return _proba(model, _sk_X(X, getattr(model, "_agent_cat_cols", [])), task)


FAMILIES: dict[str, ModelFamily] = {f.name: f for f in [
    ModelFamily("lgbm", False, _lgbm_defaults, _lgbm_space, _lgbm_fit, _proba,
                description="LightGBM gradient boosting (CPU build)"),
    ModelFamily("xgb", True, _xgb_defaults, _xgb_space, _xgb_fit, _proba,
                description="XGBoost hist, CUDA when available"),
    ModelFamily("catboost", True, _cb_defaults, _cb_space, _cb_fit, _cb_predict, max_rows=3_000_000,
                description="CatBoost with native categoricals, GPU when available"),
    ModelFamily("rf", False, _forest_defaults, _forest_space, _forest_fit_factory(False), _sk_predict,
                max_rows=300_000, description="Random forest"),
    ModelFamily("extratrees", False, _forest_defaults, _forest_space, _forest_fit_factory(True), _sk_predict,
                max_rows=300_000, description="Extremely randomized trees"),
    ModelFamily("linear", False, _linear_defaults, _linear_space, _linear_fit, _sk_predict, max_rows=2_000_000,
                description="Logistic / ridge regression with one-hot + scaling"),
]}


def get_family(name: str) -> ModelFamily:
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(f"Unknown model family {name!r}. Available: {sorted(FAMILIES)}") from None


def baseline_families(n_rows: int, n_features: int) -> list[str]:
    """Candidate families for a first pass, cheapest-strong first."""
    out = []
    for name in ("lgbm", "xgb", "catboost", "linear", "extratrees", "rf"):
        fam = FAMILIES[name]
        if fam.max_rows and n_rows > fam.max_rows:
            continue
        if name in ("rf", "extratrees") and n_rows * max(n_features, 1) > 3e7:
            continue
        out.append(name)
    return out
