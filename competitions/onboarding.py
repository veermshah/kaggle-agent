"""`start --competition SLUG`: download, inspect, configure, and freeze the validation scheme."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from competitions.kaggle_client import KaggleClient, LocalDirectoryClient, KaggleApiClient
from competitions.profiler import profile_dataset, read_table
from competitions.tabular import _key
from core.compliance import default_rules
from core.settings import load_settings
from core.workspace import Workspace, set_active
from evaluation.metrics import DEFAULT_METRIC, METRICS, get_metric, metric_from_kaggle_name
from evaluation.validation import choose_scheme, make_folds, save_validation
from tracking.db import Tracker

log = logging.getLogger("kaggle_agent.onboarding")

PRESERVE_KEYS = ("rules", "autosubmit", "validation", "agent", "submission_budget")


def client_for(config: dict[str, Any]) -> KaggleClient:
    source = config.get("data_source", "kaggle")
    if source.startswith("local:"):
        return LocalDirectoryClient(Path(source[len("local:"):]))
    return KaggleApiClient()


def _metric_hint(name: str | None) -> str | None:
    for task in ("binary", "multiclass", "regression"):
        m = metric_from_kaggle_name(name, task)
        if m:
            return m
    return None


def _data_source(client: KaggleClient, data_dir: str | None, previous: dict[str, Any]) -> str:
    if not client.simulated:
        return "kaggle"
    if data_dir:
        return f"local:{Path(data_dir).resolve()}"
    return previous.get("data_source", f"local:{getattr(client, 'source_dir', '')}")


def onboard(slug: str, client: KaggleClient, tracker: Tracker, *, data_dir: str | None = None,
            force: bool = False, metric: str | None = None, redownload: bool = False) -> dict[str, Any]:
    settings = load_settings()
    ws = Workspace(slug).create()
    previous = ws.load_config() if ws.exists() else {}

    meta = client.competition_metadata(slug)
    meta["simulated"] = client.simulated
    remote_files = client.list_files(slug)
    local_tables = [p for p in ws.data_dir.glob("*") if p.is_file()]
    if redownload or not local_tables:
        log.info("downloading %s", slug)
        client.download(slug, ws.data_dir)

    user_metric = metric or (previous.get("metric") if previous.get("metric_source") == "user" else None)
    metric_hint = user_metric or _metric_hint(meta.get("evaluation_metric"))
    profile = profile_dataset(ws.data_dir, metric_hint, seed=settings["validation"]["seed"])
    profile["remote_files"] = remote_files
    task = profile["task_type"]

    if metric:
        metric_name, source = get_metric(metric).name, "user"
    elif previous.get("metric_source") == "user":
        metric_name, source = previous["metric"], "user"
    else:
        metric_name = metric_from_kaggle_name(meta.get("evaluation_metric"), task)
        source = "kaggle" if metric_name else "default"
        metric_name = metric_name or DEFAULT_METRIC[task]
    if task not in METRICS[metric_name].tasks:
        raise ValueError(f"Metric {metric_name} does not apply to a {task} task")

    if profile["submission_format"] == "probability" and METRICS[metric_name].needs == "label":
        profile["submission_format"] = "label"

    rules = default_rules(meta)
    config: dict[str, Any] = {
        "slug": slug,
        "title": meta.get("title", slug),
        "url": meta.get("url"),
        "data_source": _data_source(client, data_dir, previous),
        "task_type": task,
        "target": profile["target"],
        "id_column": profile["id_column"],
        "classes": profile["classes"],
        "metric": metric_name,
        "metric_source": source,
        "kaggle_metric_name": meta.get("evaluation_metric"),
        "direction": METRICS[metric_name].direction,
        "deadline": meta.get("deadline"),
        "max_daily_submissions": meta.get("max_daily_submissions") or 5,
        "dataset_version": profile["dataset_version"],
        "target_non_negative": bool(profile["target_stats"].get("non_negative", False)),
        "data": {"train_file": profile["files"]["train"], "test_file": profile["files"]["test"],
                 "sample_submission_file": profile["files"]["sample_submission"]},
        "submission": {"format": profile["submission_format"], "id_column": profile["id_column"],
                       "prediction_columns": profile["prediction_columns"]},
        "features": {**profile["column_types"]},
        "rules": rules,
        "autosubmit": {"enabled": False, "max_daily": 0, "authorized_at": None, "min_improvement": None},
        "validation": {},
    }
    for key in PRESERVE_KEYS:
        if key in previous:
            config[key] = previous[key] if key != "rules" else {**rules, **{
                k: v for k, v in previous["rules"].items()
                if k in ("external_data", "pretrained_models", "automated_submissions", "confirmed_by_user", "notes")}}
    if previous.get("features", {}).get("drop"):
        config["features"]["drop"] = sorted(set(config["features"]["drop"]) | set(previous["features"]["drop"]))

    ws.save_config(config)
    ws.save_profile(profile)

    # Freeze validation folds (only regenerated with --force).
    scheme = choose_scheme(task, profile, config["validation"], settings["validation"])
    validation_note = "kept existing folds"
    if force or not ws.validation_path.exists():
        train = read_table(ws.data_dir / profile["files"]["train"])
        lookup = {_key(c): i for i, c in enumerate(profile["classes"] or [])}
        y = (train[profile["target"]].map(lambda v: lookup[_key(v)]).to_numpy()
             if lookup else train[profile["target"]].to_numpy(dtype=float))
        folds = make_folds(train, y, scheme, task)
        version = save_validation(ws, scheme, folds, force=True)
        validation_note = "created new folds"
    else:
        version = ws.load_validation()["version"]

    tracker.upsert_competition(slug, title=config["title"], task_type=task, metric=metric_name,
                               direction=config["direction"], status="ready")
    tracker.log_event(slug, "onboarded", f"{task} task, metric {metric_name} ({source}), validation {version}",
                      {"warnings": profile["warnings"]})
    set_active(slug)
    return {
        "competition": slug,
        "workspace": str(ws.root),
        "task_type": task,
        "target": profile["target"],
        "metric": metric_name,
        "metric_source": source,
        "submission_format": profile["submission_format"],
        "rows": {"train": profile["n_train_rows"], "test": profile["n_test_rows"]},
        "column_types": {k: len(v) for k, v in profile["column_types"].items()},
        "validation": {**{k: v for k, v in ws.load_validation().items() if k != "fold_counts"},
                       "note": validation_note},
        "rules": config["rules"],
        "warnings": profile["warnings"]
        + ([] if source != "default" else [f"Metric could not be read from Kaggle; defaulted to {metric_name}. "
                                           "Override with: start --metric NAME"])
        + ([] if rules.get("joined_competition") is not False else
           ["You have not joined this competition on kaggle.com; submissions will fail until you do"]),
        "simulated_data_source": client.simulated,
    }
