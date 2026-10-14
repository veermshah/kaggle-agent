"""Research agent interface and the shared research context.

An agent sees a JSON-serializable summary of the competition and experiment history and returns
proposals: declarative specs with a hypothesis. Specs are validated before anything runs, so agents
(heuristic, LLM via CLI, Devin, MCP clients) never execute code directly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from core.settings import load_settings
from core.workspace import Workspace
from evaluation.metrics import Metric, get_metric
from tracking.db import Tracker


@dataclass
class Proposal:
    spec: dict[str, Any]
    hypothesis: str
    arm: str
    priority: float = 0.0
    parent_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchContext:
    competition: str
    task_type: str
    metric: str
    direction: str
    n_train_rows: int
    n_features: int
    column_types: dict[str, int]
    has_datetime: bool
    has_text: bool
    target_stats: dict[str, Any]
    validation: dict[str, Any]
    experiments: list[dict[str, Any]]           # finished + active, current validation version, chronological
    best: dict[str, Any] | None
    best_by_family: dict[str, dict[str, Any]]
    queued: int
    running: int
    loop: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def metric_obj(self) -> Metric:
        return get_metric(self.metric)

    def succeeded(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [e for e in self.experiments if e["status"] == "succeeded" and (kind is None or e["kind"] == kind)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchAgent(Protocol):
    name: str

    def propose(self, ctx: ResearchContext, n: int) -> list[Proposal]: ...


def loop_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Global loop settings with per-competition overrides from competition.yaml `loop:`."""
    return {**load_settings()["loop"], **(config.get("loop") or {})}


def build_context(ws: Workspace, tracker: Tracker) -> ResearchContext:
    import json

    config, profile = ws.load_config(), ws.load_profile()
    validation = ws.load_validation()
    metric = get_metric(config["metric"])
    loop = loop_settings(config)
    min_delta = float(loop["min_improvement"])
    rows = tracker.list_experiments(ws.slug, validation_version=validation["version"])
    best_so_far = None
    experiments = []
    for e in sorted(rows, key=lambda r: (r["finished_at"] or "9999", r["id"])):
        improved = None
        if e["status"] == "succeeded" and e["cv_score"] is not None:
            improved = metric.better(e["cv_score"], best_so_far, min_delta * max(1.0, abs(best_so_far or 0.0)))
            prev = best_so_far
            if metric.better(e["cv_score"], best_so_far):
                best_so_far = e["cv_score"]
        else:
            prev = best_so_far
        top_features = None
        if e["status"] == "succeeded" and e["kind"] != "ensemble" and e["artifact_dir"]:
            try:
                result = json.loads((ws.experiment_dir(e["id"]) / "result.json").read_text())
                top_features = list(result.get("top_features", {}))[:15]
            except (OSError, ValueError):
                pass
        experiments.append({
            "id": e["id"], "kind": e["kind"], "model": e["model_family"], "status": e["status"],
            "fingerprint": e["fingerprint"],
            "arm": e["arm"], "hypothesis": e["hypothesis"], "features": e["spec"].get("features", []),
            "feature_params": e["spec"].get("feature_params", {}),
            "target_transform": e["spec"].get("target_transform", "none"), "seed": e["spec"].get("seed"),
            "spec_params": e["spec"].get("params", {}), "params": e["params"],
            "members": e["spec"].get("members"), "method": e["spec"].get("method"),
            "calibrate": e["spec"].get("calibrate"),
            "cv_score": e["cv_score"], "cv_std": e["cv_std"], "train_seconds": e["train_seconds"],
            "improved": improved, "best_before": prev, "proposed_by": e["proposed_by"],
            "error": (e["error"] or "").splitlines()[0][:200] if e["error"] else None,
            "top_features": top_features, "finished_at": e["finished_at"],
        })
    done = [x for x in experiments if x["status"] == "succeeded" and x["cv_score"] is not None]
    order = sorted(done, key=lambda x: x["cv_score"], reverse=metric.greater_is_better)
    best = order[0] if order else None
    best_by_family: dict[str, dict] = {}
    for x in order:
        if x["kind"] != "ensemble":
            best_by_family.setdefault(x["model"], x)
    return ResearchContext(
        competition=ws.slug, task_type=config["task_type"], metric=metric.name, direction=metric.direction,
        n_train_rows=profile.get("n_train_rows", 0), n_features=profile.get("n_features", 0),
        column_types={k: len(v) for k, v in config["features"].items()},
        has_datetime=bool(config["features"].get("datetime")), has_text=bool(config["features"].get("text")),
        target_stats=profile.get("target_stats", {}),
        validation={k: validation[k] for k in ("strategy", "n_splits", "holdout_fraction", "version")},
        experiments=experiments, best=best, best_by_family=best_by_family,
        queued=sum(x["status"] == "queued" for x in experiments),
        running=sum(x["status"] == "running" for x in experiments),
        loop=loop, warnings=profile.get("warnings", []),
    )
