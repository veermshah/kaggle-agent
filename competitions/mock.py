"""Synthetic Kaggle-style competitions for tests and offline demos (served by LocalDirectoryClient)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = {
    "binary": ("Area Under Receiver Operating Characteristic Curve", "roc_auc"),
    "multiclass": ("Multi-class Log Loss", "logloss"),
    "regression": ("Root Mean Squared Error", "rmse"),
    "binary_label": ("Categorization Accuracy", "accuracy"),
}


def create_mock_competition(dest: Path, kind: str = "binary", n_train: int = 3000, n_test: int = 1000,
                            seed: int = 0) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    x = rng.normal(size=(n, 6))
    cat_a = rng.choice(["red", "green", "blue", "black"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    cat_b = rng.choice([f"lvl{i}" for i in range(30)], size=n)
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 700, size=n), unit="D")
    effect_a = pd.Series(cat_a).map({"red": 0.8, "green": -0.4, "blue": 0.1, "black": -1.0}).to_numpy()
    effect_b = (np.array([int(s[3:]) for s in cat_b]) % 5 - 2) * 0.3
    signal = 1.2 * x[:, 0] - 0.8 * x[:, 1] + 0.6 * x[:, 2] * x[:, 3] + np.sin(2 * x[:, 4]) + effect_a + effect_b
    signal += 0.3 * np.isin(dates.month, [6, 7, 8])

    df = pd.DataFrame({f"num_{i}": x[:, i] for i in range(6)})
    df["cat_a"], df["cat_b"], df["signup_date"] = cat_a, cat_b, dates.strftime("%Y-%m-%d")
    df.loc[rng.random(n) < 0.08, "num_5"] = np.nan
    df.loc[rng.random(n) < 0.05, "cat_a"] = None
    df.insert(0, "id", np.arange(100000, 100000 + n))

    noise = rng.normal(scale=0.7, size=n)
    if kind in ("binary", "binary_label"):
        target_name = "Transported" if kind == "binary_label" else "target"
        y = (signal + noise > 0.3)
        if kind == "binary":
            y = y.astype(int)
    elif kind == "multiclass":
        target_name = "Status"
        score = signal + noise
        y = np.where(score < -0.8, "C", np.where(score < 0.9, "CL", "D"))
    else:
        target_name = "price"
        y = np.round(np.exp(1.5 + 0.35 * (signal + noise)) * 10, 2)
    df[target_name] = y

    dest.mkdir(parents=True, exist_ok=True)
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    train.to_csv(dest / "train.csv", index=False)
    test.drop(columns=[target_name]).to_csv(dest / "test.csv", index=False)

    if kind == "multiclass":
        classes = ["C", "CL", "D"]
        sample = pd.DataFrame({"id": test["id"], **{f"Status_{c}": 1 / 3 for c in classes}})
        solution = test[["id", target_name]]
    elif kind == "binary":
        sample = pd.DataFrame({"id": test["id"], target_name: 0.5})
        solution = test[["id", target_name]]
    elif kind == "binary_label":
        sample = pd.DataFrame({"id": test["id"], target_name: False})
        solution = test[["id", target_name]]
    else:
        sample = pd.DataFrame({"id": test["id"], target_name: float(np.median(train[target_name]))})
        solution = test[["id", target_name]]
    sample.to_csv(dest / "sample_submission.csv", index=False)
    solution.to_csv(dest / "solution.csv", index=False)

    metric_name, metric_key = KINDS[kind]
    (dest / "metadata.json").write_text(json.dumps({
        "title": f"Mock {kind} competition",
        "evaluation_metric": metric_name,
        "metric": metric_key,
        "max_daily_submissions": 5,
        "deadline": "2099-12-31T23:59:59+00:00",
        "user_has_entered": True,
        "solution_file": "solution.csv",
    }, indent=2))
    return dest
