"""Dataset inspection: file roles, target/task detection, column typing, leakage and shift checks."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import METRICS

TABULAR_EXT = (".csv", ".parquet", ".tsv")
GROUP_NAME_HINT = re.compile(r"(group|user|customer|patient|subject|session|player|store|device|account|household)",
                             re.I)
TIME_NAME_HINT = re.compile(r"(date|time|timestamp|datetime|day|month|year)", re.I)


class ProfileError(RuntimeError):
    pass


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    sep = "\t" if path.suffix == ".tsv" else ","
    return pd.read_csv(path, nrows=nrows, sep=sep, low_memory=False)


def find_files(data_dir: Path) -> dict[str, Path | None]:
    files = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR_EXT
             and "cache" not in p.parts]

    def pick(patterns: list[str], exclude: tuple[str, ...] = ()) -> Path | None:
        for pat in patterns:
            hits = [p for p in files if re.search(pat, p.stem, re.I) and not any(e in p.stem.lower() for e in exclude)]
            if hits:
                return min(hits, key=lambda p: (len(p.stem), str(p)))
        return None

    sample = pick([r"sample_?submission", r"^sample", r"submission"])
    train = pick([r"^train$", r"^train", r"train"], exclude=("sample",))
    test = pick([r"^test$", r"^test", r"test"], exclude=("sample", "solution"))
    return {"train": train, "test": test, "sample_submission": sample}


def file_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def _is_categorical(s: pd.Series) -> bool:
    return (isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s))


def _looks_datetime(s: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(500)
    if sample.empty or sample.str.len().mean() < 6:
        return False
    if not (TIME_NAME_HINT.search(name) or sample.str.contains(r"\d{4}-\d{2}-\d{2}").mean() > 0.9):
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.95


def _is_text(s: pd.Series) -> bool:
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(2000)
    return not sample.empty and sample.str.len().mean() > 40 and sample.str.contains(" ").mean() > 0.5


def _id_like(s: pd.Series, n: int) -> bool:
    if s.nunique(dropna=False) != n:
        return False
    return (pd.api.types.is_integer_dtype(s) or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s))


def detect_task(y: pd.Series, metric_name: str | None, sample_pred_cols: list[str]) -> str:
    nunique = y.nunique(dropna=True)
    if nunique <= 1:
        raise ProfileError("Target has a single value; nothing to learn.")
    if metric_name and metric_name in METRICS:
        tasks = METRICS[metric_name].tasks
        if tasks == ("regression",):
            return "regression"
        if "regression" not in tasks:
            return "binary" if nunique == 2 else "multiclass"
    if nunique == 2:
        return "binary"
    if len(sample_pred_cols) > 1:
        return "multiclass"
    if _is_categorical(y):
        return "multiclass"
    values = y.dropna().to_numpy(dtype=float)
    if np.allclose(values, np.round(values)) and nunique <= 20:
        return "multiclass"
    return "regression"


def _match_class_columns(classes: list[Any], columns: list[str]) -> list[str] | None:
    """Map each class label to a sample-submission column (exact, or column ending in _<label>)."""
    mapping = []
    for c in classes:
        label = str(c)
        hits = [col for col in columns if col == label] or [col for col in columns if col.endswith(f"_{label}")]
        if len(hits) != 1:
            return None
        mapping.append(hits[0])
    return mapping if len(set(mapping)) == len(mapping) else None


def single_feature_leakage(train: pd.DataFrame, y: pd.Series, task_type: str, cols: list[str]) -> list[dict]:
    """Columns that alone almost perfectly predict the target are suspicious."""
    from sklearn.metrics import roc_auc_score
    flagged = []
    yv = pd.factorize(y, sort=True)[0] if task_type == "binary" else y.to_numpy()
    for col in cols:
        s = train[col]
        if not pd.api.types.is_numeric_dtype(s) or s.nunique() < 2:
            continue
        x = s.fillna(s.median()).to_numpy(dtype=float)
        try:
            if task_type == "binary":
                auc = roc_auc_score(yv, x)
                strength = max(auc, 1 - auc)
            elif task_type == "regression":
                strength = abs(pd.Series(x).corr(pd.Series(yv.astype(float)), method="spearman"))
            else:
                continue
        except ValueError:
            continue
        if strength > 0.995:
            flagged.append({"column": col, "strength": round(float(strength), 5)})
    return flagged


def adversarial_validation(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], seed: int = 0) -> dict:
    """Train vs test classifier. AUC near 0.5 means similar distributions; high AUC signals shift."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    if not cols:
        return {"auc": None}
    n = min(len(train), len(test), 20000)
    a = train[cols].sample(n, random_state=seed)
    b = test[cols].sample(n, random_state=seed)
    X = pd.concat([a, b], ignore_index=True)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    y = np.r_[np.zeros(n), np.ones(n)]
    model = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    pred = cross_val_predict(model, X, y, cv=3, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, pred))
    model.fit(X, y)
    drift = []
    if auc > 0.6:
        for c in cols:
            try:
                single = roc_auc_score(y, X[c].fillna(-999))
                drift.append((c, max(single, 1 - single)))
            except ValueError:
                pass
        drift.sort(key=lambda t: -t[1])
    return {"auc": round(auc, 4), "top_shifted_columns": [{"column": c, "auc": round(v, 4)} for c, v in drift[:10]]}


def profile_dataset(data_dir: Path, metric_name: str | None = None, seed: int = 42) -> dict[str, Any]:
    files = find_files(data_dir)
    if not files["train"] or not files["test"]:
        raise ProfileError(
            f"Could not identify train/test tables in {data_dir}. Found: "
            f"{[p.name for p in data_dir.rglob('*') if p.is_file()][:20]}. "
            "This competition may not be tabular; set train_file/test_file in competition.yaml if it is."
        )
    train, test = read_table(files["train"]), read_table(files["test"])
    sample = read_table(files["sample_submission"]) if files["sample_submission"] else None

    id_col = None
    if sample is not None and sample.columns[0] in test.columns:
        id_col = sample.columns[0]
    elif "id" in test.columns:
        id_col = "id"

    pred_cols = [c for c in sample.columns if c != id_col] if sample is not None else []
    train_only = [c for c in train.columns if c not in test.columns]
    target_candidates = [c for c in pred_cols if c in train.columns] or [c for c in train_only if c != id_col]
    if len(target_candidates) != 1:
        raise ProfileError(
            f"Could not determine a single target column (candidates: {target_candidates}). "
            "Multi-target competitions are not supported yet; set `target` in competition.yaml to override."
        )
    target = target_candidates[0]
    y = train[target]
    if y.isna().any():
        raise ProfileError(f"Target '{target}' has {int(y.isna().sum())} missing values; inspect the data.")
    task_type = detect_task(y, metric_name, pred_cols)

    # Submission format and class ordering.
    classes = None
    submission_format = "value"
    if task_type in ("binary", "multiclass"):
        classes = sorted(y.unique().tolist(), key=lambda v: (str(type(v)), v))
        if task_type == "multiclass" and len(pred_cols) > 1:
            mapped = _match_class_columns(classes, pred_cols)
            if mapped is None:
                raise ProfileError(f"Cannot map target classes {classes[:10]} to submission columns {pred_cols[:10]}")
            submission_format = "class_probabilities"
            order = sorted(range(len(classes)), key=lambda i: pred_cols.index(mapped[i]))
            classes = [classes[i] for i in order]
        else:
            needs_proba = metric_name in METRICS and METRICS[metric_name].needs == "proba"
            sample_vals = sample[pred_cols[0]] if sample is not None and pred_cols else None
            float_sample = sample_vals is not None and pd.api.types.is_float_dtype(sample_vals) and \
                not np.allclose(sample_vals.dropna(), np.round(sample_vals.dropna()))
            if task_type == "binary" and (needs_proba or float_sample):
                submission_format = "probability"
            elif task_type == "multiclass" and needs_proba:
                raise ProfileError("Probability metric with a single submission column on a multiclass task.")
            else:
                submission_format = "label"

    feature_cols = [c for c in train.columns if c not in (target, id_col) and c in test.columns]
    n = len(train)
    column_types: dict[str, list[str]] = {"numeric": [], "categorical": [], "datetime": [], "text": [], "drop": []}
    column_info = {}
    for col in feature_cols:
        s = train[col]
        nunique = int(s.nunique(dropna=True))
        info = {"dtype": str(s.dtype), "missing_rate": round(float(s.isna().mean()), 4), "nunique": nunique}
        if nunique <= 1 and test[col].nunique(dropna=True) <= 1:
            kind, info["reason"] = "drop", "constant"
        elif _id_like(s, n) and not pd.api.types.is_float_dtype(s):
            kind, info["reason"] = "drop", "unique per row (id-like)"
        elif _looks_datetime(s, col):
            kind = "datetime"
        elif _is_text(s):
            kind = "text"
        elif _is_categorical(s):
            kind = "categorical"
        else:
            kind = "numeric"
        column_types[kind].append(col)
        column_info[col] = {**info, "type": kind}

    usable = column_types["numeric"] + column_types["categorical"]

    # Target distribution
    if task_type == "regression":
        yv = y.astype(float)
        target_stats = {"mean": float(yv.mean()), "std": float(yv.std()), "min": float(yv.min()),
                        "max": float(yv.max()), "skew": float(yv.skew()), "non_negative": bool((yv >= 0).all())}
    else:
        counts = y.value_counts(normalize=True).head(50)
        target_stats = {"class_balance": {str(k): round(float(v), 4) for k, v in counts.items()},
                        "n_classes": int(y.nunique())}

    # Leakage / contamination
    leakage = single_feature_leakage(train, y, task_type, column_types["numeric"])
    dup_within_train = int(train[feature_cols].duplicated().sum()) if feature_cols else 0
    overlap = 0
    if feature_cols:
        key = lambda df: pd.util.hash_pandas_object(df[feature_cols], index=False)
        overlap = int(key(test).isin(set(key(train))).sum())
    adv = adversarial_validation(train, test, usable, seed=seed)

    # Group / time structure
    group_col = None
    for col in column_types["categorical"] + column_types["numeric"]:
        s = train[col]
        nunique = s.nunique()
        if GROUP_NAME_HINT.search(col) and 20 < nunique < 0.5 * n:
            shared = len(set(s.dropna().unique()) & set(test[col].dropna().unique()))
            if shared == 0:
                group_col = col
                break
    time_col, test_after_train = None, False
    for col in column_types["datetime"] + [c for c in column_types["numeric"] if TIME_NAME_HINT.search(c)]:
        tr = train[col] if col not in column_types["datetime"] else pd.to_datetime(train[col], errors="coerce")
        te = test[col] if col not in column_types["datetime"] else pd.to_datetime(test[col], errors="coerce")
        if tr.notna().mean() > 0.95 and te.notna().mean() > 0.95:
            time_col = time_col or col
            if te.min() >= tr.quantile(0.98):
                time_col, test_after_train = col, True
                break

    warnings = []
    if leakage:
        warnings.append(f"Possible target leakage in: {[l['column'] for l in leakage]}")
    if overlap:
        warnings.append(f"{overlap} test rows have identical features to train rows (train/test contamination)")
    if adv.get("auc") and adv["auc"] > 0.75:
        warnings.append(f"Strong train/test distribution shift (adversarial AUC {adv['auc']})")
    if column_types["text"]:
        warnings.append(f"Free-text columns {column_types['text']} get only basic length features in tabular mode")

    return {
        "files": {k: (str(v.relative_to(data_dir)) if v else None) for k, v in files.items()},
        "dataset_version": file_hash([p for p in (files["train"], files["test"]) if p]),
        "n_train_rows": n, "n_test_rows": len(test), "n_features": len(feature_cols),
        "id_column": id_col, "target": target, "task_type": task_type,
        "classes": classes, "submission_format": submission_format,
        "prediction_columns": pred_cols or [target],
        "target_stats": target_stats,
        "column_types": column_types, "columns": column_info,
        "train_only_columns": [c for c in train_only if c != target],
        "leakage_suspects": leakage,
        "duplicate_rows_in_train": dup_within_train,
        "train_test_overlap_rows": overlap,
        "adversarial_validation": adv,
        "suggested_group_column": group_col,
        "suggested_time_column": time_col,
        "test_after_train": test_after_train,
        "memory_mb": round(float(train.memory_usage(deep=True).sum() + test.memory_usage(deep=True).sum()) / 2**20, 1),
        "warnings": warnings,
    }
