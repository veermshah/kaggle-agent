"""Build submission frames from raw test predictions and validate them against the sample submission."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from competitions.profiler import read_table
from core.workspace import Workspace
from evaluation.metrics import to_labels


class SubmissionError(RuntimeError):
    pass


def load_sample(ws: Workspace, config: dict[str, Any]) -> pd.DataFrame:
    name = config["data"].get("sample_submission_file")
    if not name:
        raise SubmissionError("No sample submission file was found during onboarding; cannot verify format")
    return read_table(ws.data_dir / name)


def test_ids(ws: Workspace, config: dict[str, Any]) -> pd.Series:
    id_col = config["submission"]["id_column"]
    test = read_table(ws.data_dir / config["data"]["test_file"])
    if id_col not in test.columns:
        raise SubmissionError(f"Id column {id_col!r} missing from the test file")
    return test[id_col]


def build_submission(ws: Workspace, config: dict[str, Any], pred: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    sample = load_sample(ws, config)
    sub_cfg = config["submission"]
    id_col, pred_cols, fmt = sub_cfg["id_column"], sub_cfg["prediction_columns"], sub_cfg["format"]
    ids = test_ids(ws, config)
    if len(ids) != len(pred):
        raise SubmissionError(f"{len(pred)} predictions for {len(ids)} test rows")
    df = pd.DataFrame({id_col: ids.to_numpy()})
    if fmt == "probability":
        df[pred_cols[0]] = np.clip(pred, 0, 1)
    elif fmt == "label":
        labels = to_labels(pred, config["task_type"], threshold)
        classes = config["classes"]
        df[pred_cols[0]] = [classes[i] for i in labels]
    elif fmt == "class_probabilities":
        if pred.ndim != 2 or pred.shape[1] != len(pred_cols):
            raise SubmissionError(f"Expected {len(pred_cols)} class columns, predictions have shape {pred.shape}")
        probs = np.clip(pred, 1e-15, 1)
        probs = probs / probs.sum(axis=1, keepdims=True)
        for j, col in enumerate(pred_cols):
            df[col] = probs[:, j]
    elif fmt == "value":
        df[pred_cols[0]] = pred
    else:
        raise SubmissionError(f"Unknown submission format {fmt!r}")

    # Match the sample's row order exactly.
    if id_col in sample.columns:
        indexed = df.set_index(id_col)
        missing = set(sample[id_col]) - set(indexed.index)
        if missing:
            raise SubmissionError(f"{len(missing)} sample-submission ids have no prediction (e.g. {list(missing)[:3]})")
        df = indexed.loc[sample[id_col].to_numpy()].reset_index()
    return df[list(sample.columns)]


def validate_submission(df: pd.DataFrame, ws: Workspace, config: dict[str, Any]) -> list[str]:
    sample = load_sample(ws, config)
    sub_cfg = config["submission"]
    errors = []
    if list(df.columns) != list(sample.columns):
        errors.append(f"Columns {list(df.columns)} differ from sample {list(sample.columns)}")
        return errors
    if len(df) != len(sample):
        errors.append(f"Row count {len(df)} != sample {len(sample)}")
    id_col = sub_cfg["id_column"]
    if id_col in df.columns:
        if df[id_col].duplicated().any():
            errors.append("Duplicate ids")
        if len(df) == len(sample) and not (df[id_col].astype(str).to_numpy() == sample[id_col].astype(str).to_numpy()).all():
            errors.append("Id order differs from the sample submission")
    pred_cols = [c for c in df.columns if c != id_col]
    if df[pred_cols].isna().any().any():
        errors.append("Predictions contain missing values")
    fmt = sub_cfg["format"]
    if fmt in ("probability", "class_probabilities", "value"):
        values = df[pred_cols].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append("Predictions contain inf/NaN")
        if fmt != "value" and ((values < 0).any() or (values > 1).any()):
            errors.append("Probabilities outside [0, 1]")
        if fmt == "class_probabilities" and not np.allclose(values.sum(axis=1), 1, atol=1e-4):
            errors.append("Class probabilities do not sum to 1")
    elif fmt == "label":
        allowed = {str(c) for c in config["classes"]}
        bad = set(df[pred_cols[0]].astype(str)) - allowed
        if bad:
            errors.append(f"Labels {sorted(bad)[:5]} not among known classes {sorted(allowed)}")
        sample_vals = sample[pred_cols[0]].dropna()
        if len(sample_vals) and pd.api.types.is_numeric_dtype(sample_vals) != pd.api.types.is_numeric_dtype(df[pred_cols[0]]) \
                and not pd.api.types.is_bool_dtype(sample_vals):
            errors.append(f"Label dtype {df[pred_cols[0]].dtype} differs from sample dtype {sample_vals.dtype}")
    return errors
