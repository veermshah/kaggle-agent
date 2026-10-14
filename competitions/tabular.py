"""Tabular dataset loading with a parquet cache keyed by dataset version."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from competitions.profiler import read_table
from core.workspace import Workspace


@dataclass
class TabularData:
    train: pd.DataFrame          # raw feature columns (target and id removed)
    test: pd.DataFrame
    y: np.ndarray                # encoded: class index for classification, float for regression
    test_ids: pd.Series | None
    classes: list[Any] | None
    task_type: str
    column_types: dict[str, list[str]]

    @property
    def n_classes(self) -> int:
        return len(self.classes) if self.classes else 0


def _cached(ws: Workspace, name: str, source: str, version: str) -> pd.DataFrame:
    cache = ws.data_dir / "cache" / f"{name}-{version}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = read_table(ws.data_dir / source)
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache, index=False)
    except Exception:
        pass  # mixed-type object columns can fail to serialize; just skip caching
    return df


def load_tabular(ws: Workspace, config: dict[str, Any]) -> TabularData:
    data_cfg = config["data"]
    version = config.get("dataset_version", "v0")
    train = _cached(ws, "train", data_cfg["train_file"], version)
    test = _cached(ws, "test", data_cfg["test_file"], version)
    target, id_col = config["target"], config.get("id_column")
    task = config["task_type"]
    raw_y = train[target]
    classes = config.get("classes")
    if task in ("binary", "multiclass"):
        lookup = {_key(c): i for i, c in enumerate(classes)}
        try:
            y = np.array([lookup[_key(v)] for v in raw_y], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Target value {exc} not among configured classes {classes}") from None
    else:
        y = raw_y.to_numpy(dtype=float)
    ctypes = config["features"]
    keep = [c for kind in ("numeric", "categorical", "datetime", "text") for c in ctypes.get(kind, [])
            if c not in ctypes.get("drop", [])]
    return TabularData(
        train=train[keep].reset_index(drop=True),
        test=test[keep].reset_index(drop=True),
        y=y,
        test_ids=test[id_col].reset_index(drop=True) if id_col and id_col in test.columns else None,
        classes=classes,
        task_type=task,
        column_types=ctypes,
    )


def _key(v: Any) -> str:
    """Normalize labels so YAML round-trips (e.g. 1 vs 1.0 vs '1', True vs 'True') still match."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v)
