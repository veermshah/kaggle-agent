"""Experiment specifications: validated, declarative descriptions of what to run.

Specs are data, not code. Anything an agent proposes goes through `normalize_spec`, which rejects unknown
model families, recipes, kinds and parameter names, so a proposal can never smuggle in executable content.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from experiments.features import validate_recipes
from models.tabular import FAMILIES, get_family

KINDS = ("train", "tune", "ensemble")
ENSEMBLE_METHODS = ("hill_climb", "mean", "rank_mean")
FINGERPRINT_EXCLUDE = {"device", "max_seconds", "n_trials", "timeout", "requires"}
PARAM_TYPES = (int, float, str, bool, type(None))
PARAM_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")


class SpecError(ValueError):
    pass


def _check_params(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise SpecError("params must be an object")
    for k, v in params.items():
        if not isinstance(k, str) or not PARAM_NAME.match(k):
            raise SpecError(f"Invalid parameter name {k!r}")
        if not isinstance(v, PARAM_TYPES):
            raise SpecError(f"Parameter {k} must be a scalar, got {type(v).__name__}")
    return params


def normalize_spec(raw: dict[str, Any], task_type: str | None = None) -> dict[str, Any]:
    spec = dict(raw)
    kind = spec.setdefault("kind", "train")
    if kind not in KINDS:
        raise SpecError(f"kind must be one of {KINDS}")
    allowed = {"kind", "model", "params", "features", "feature_params", "target_transform", "seed", "device",
               "requires", "max_seconds", "n_trials", "timeout", "members", "method", "calibrate", "top_k",
               "notes"}
    unknown = set(spec) - allowed
    if unknown:
        raise SpecError(f"Unknown spec fields: {sorted(unknown)}")
    spec["seed"] = int(spec.get("seed", 42))
    spec["device"] = spec.get("device", "auto")
    if spec["device"] not in ("auto", "cpu", "gpu"):
        raise SpecError("device must be auto, cpu or gpu")
    spec["requires"] = sorted(set(spec.get("requires", [])))

    if kind in ("train", "tune"):
        if "model" not in spec:
            raise SpecError("train/tune specs need a 'model'")
        get_family(spec["model"])
        spec["params"] = _check_params(spec.get("params", {}))
        spec["features"] = validate_recipes(spec.get("features", []))
        spec["feature_params"] = spec.get("feature_params", {})
        if not isinstance(spec["feature_params"], dict):
            raise SpecError("feature_params must be an object")
        spec["target_transform"] = spec.get("target_transform", "none")
        if spec["target_transform"] not in ("none", "log1p"):
            raise SpecError("target_transform must be none or log1p")
        if spec["target_transform"] == "log1p" and task_type and task_type != "regression":
            raise SpecError("log1p target transform only applies to regression")
        if kind == "tune":
            spec["n_trials"] = int(spec.get("n_trials", 40))
            spec["timeout"] = int(spec.get("timeout", 1800))
        for key in ("members", "method", "calibrate", "top_k"):
            if key in spec:
                raise SpecError(f"'{key}' is only valid for ensemble specs")
    else:
        spec["method"] = spec.get("method", "hill_climb")
        if spec["method"] not in ENSEMBLE_METHODS:
            raise SpecError(f"method must be one of {ENSEMBLE_METHODS}")
        spec["members"] = sorted(int(m) for m in spec.get("members", []))
        spec["top_k"] = int(spec.get("top_k", 8))
        spec["calibrate"] = bool(spec.get("calibrate", False))
        spec["features"] = []
        spec["model"] = "ensemble"
    return spec


def fingerprint(spec: dict[str, Any]) -> str:
    canonical = {k: v for k, v in spec.items() if k not in FINGERPRINT_EXCLUDE and k != "notes"}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, default=str).encode()).hexdigest()[:16]


def resolve_resource(spec: dict[str, Any], n_rows: int, gpu_caps: dict[str, bool], settings: dict) -> str:
    """'gpu' only when the family has working GPU kernels here and the data is big enough to benefit."""
    if spec["kind"] == "ensemble":
        return "cpu"
    family = FAMILIES[spec["model"]]
    if spec["device"] == "cpu" or not family.gpu_capable:
        return "cpu"
    lib = {"xgb": "xgboost", "catboost": "catboost"}.get(spec["model"])
    if not gpu_caps.get(lib, False):
        return "cpu"
    if spec["device"] == "gpu":
        return "gpu"
    res = settings["resources"]
    if str(res.get("use_gpu", "auto")).lower() in ("false", "0", "no"):
        return "cpu"
    return "gpu" if n_rows >= int(res.get("gpu_min_rows", 50000)) else "cpu"
