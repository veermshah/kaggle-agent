"""Run screened agent-written Python on the competition data: feature-module tests and analysis scripts.

The research agent cannot run arbitrary Python. Instead it writes code into custom_features/ or analysis/,
and these commands run it after the static safety screen, in a separate process with a hard timeout and
with secret-looking environment variables (tokens, keys, passwords) removed:

    main.py features test --name <module>   # does it run? what does it produce? does it help on fold 0?
    main.py analyze --name <script>         # analysis/<script>.py: analyze(train, test) -> JSON-able result
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from core.settings import PROJECT_ROOT
from core.workspace import Workspace
from experiments import custom_code

SECRET_ENV = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL|AUTH)", re.I)
MAX_OUTPUT = 20000


class SandboxError(RuntimeError):
    pass


def analysis_dir(ws: Workspace) -> Path:
    path = ws.root / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitized_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    env["KAGGLE_AGENT_ROLE"] = "sandbox"
    env["KAGGLE_AGENT_NO_DOTENV"] = "1"  # the child must not re-load .env secrets
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _check(ws: Workspace, kind: str, name: str, human: bool) -> None:
    if kind == "feature":
        path, entry = custom_code._path(ws, name), "transform"
    else:
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise SandboxError(f"Invalid analysis script name {name!r}")
        path, entry = analysis_dir(ws) / f"{name}.py", "analyze"
    if not path.exists():
        raise SandboxError(f"{path} not found")
    problems = custom_code.screen_path(path, entry)
    if problems:
        raise SandboxError(f"Safety screen rejected {path.name}: {problems}")
    if kind == "feature":
        approved = any(m["name"] == name and m["approved"] for m in custom_code.list_custom(ws))
        if not approved:
            if custom_code.auto_approve_enabled(ws):
                custom_code.approve(ws, name)
            elif not human:
                raise SandboxError(f"custom:{name} is not approved and auto-approve is off")
    elif not (human or custom_code.auto_approve_enabled(ws)):
        raise SandboxError("Running agent analysis scripts requires `features auto-approve --state on`")


def run_sandboxed(ws: Workspace, kind: str, name: str, timeout: float = 600, human: bool = False) -> dict[str, Any]:
    _check(ws, kind, name, human)
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py"), "sandbox-exec", "--kind", kind, "--name", name,
           "-c", ws.slug]
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=sanitized_env(), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise SandboxError(f"{kind} {name} exceeded {timeout:.0f}s") from None
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        raise SandboxError(f"{kind} {name} crashed:\n{proc.stderr[-3000:]}") from None
    if not payload.get("ok"):
        raise SandboxError(payload.get("error", "unknown error"))
    result = payload["result"]
    result["seconds"] = round(time.time() - started, 1)
    return result


# Executed inside the sandbox child process

def exec_feature_test(ws: Workspace, name: str) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from evaluation.validation import fold_split
    from experiments.spec import normalize_spec
    from experiments.trainer import load_context
    from models.tabular import get_family

    ctx = load_context(ws)
    base = ctx.features([], {})
    full = ctx.features([f"custom:{name}"], {})
    new_cols = [c for c in full.train.columns if c not in base.train.columns]
    combined = pd.concat([full.train[new_cols], full.test[new_cols]], ignore_index=True)
    columns = {}
    flags = []
    for c in new_cols:
        s = combined[c]
        info = {"dtype": str(s.dtype), "null_frac": round(float(s.isna().mean()), 4), "nunique": int(s.nunique())}
        if info["nunique"] <= 1:
            flags.append(f"{c} is constant")
        if info["null_frac"] > 0.5:
            flags.append(f"{c} is mostly null")
        columns[c] = info
    # Suspiciously perfect single features usually mean target leakage.
    if ctx.task == "binary":
        from sklearn.metrics import roc_auc_score
        for c in new_cols:
            x = full.train[c]
            if pd.api.types.is_numeric_dtype(x) and x.nunique() > 1:
                auc = roc_auc_score(ctx.data.y, x.fillna(x.median()))
                if max(auc, 1 - auc) > 0.99:
                    flags.append(f"{c} alone gives AUC {max(auc, 1 - auc):.4f}: likely target leakage")

    family = get_family("lgbm")
    params = {**family.defaults(ctx.task), "learning_rate": 0.1}
    tr, va = fold_split(ctx.folds, 0, ctx.scheme["strategy"])
    scores = {}
    for label, ff in (("base", base), ("with_feature", full)):
        fit = family.fit(ff.train.iloc[tr], ctx.data.y[tr], ff.train.iloc[va], ctx.data.y[va], params, ctx.task,
                         ctx.data.n_classes, ff.categorical, "cpu", 0)
        scores[label] = ctx.score(ctx.data.y[va], family.predict(fit.model, ff.train.iloc[va], ctx.task))
        if label == "with_feature":
            imp = {k: round(v, 4) for k, v in fit.importance.items() if k in new_cols}
    delta = scores["with_feature"] - scores["base"]
    return {"module": name, "rows_ok": True, "new_columns": columns, "flags": flags,
            "quick_check": {"model": "lgbm lr=0.1", "fold": 0, "metric": ctx.metric.name, **scores,
                            "delta": delta, "better": ctx.metric.better(scores["with_feature"], scores["base"]),
                            "note": "one fold, untuned: a smoke signal, not evidence; queue a full experiment"},
            "importance_share": imp}


def exec_analysis(ws: Workspace, name: str) -> dict[str, Any]:
    from competitions.profiler import read_table
    config = ws.load_config()
    train = read_table(ws.data_dir / config["data"]["train_file"])
    test = read_table(ws.data_dir / config["data"]["test_file"])
    path = analysis_dir(ws) / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"analysis_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.analyze(train, test)
    text = json.dumps(result, default=str)
    if len(text) > MAX_OUTPUT:
        return {"truncated": True, "output": text[:MAX_OUTPUT]}
    return {"truncated": False, "output": json.loads(text)}
