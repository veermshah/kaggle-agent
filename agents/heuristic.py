"""Rule-based research strategist.

Candidate experiments are grouped into arms (baseline, tune, features, target, revisit, ensemble, seed,
retry). Each candidate is scored with a UCB-style value: a prior for the arm type, the arm type's observed
improvement rate, an exploration bonus for rarely tried types, and a penalty for expected compute. An arm
is parked after `arm_patience` consecutive non-improving runs and unparked when the global best moves
(new evidence). Identical specs are never proposed twice.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from agents.base import Proposal, ResearchContext
from experiments.features import RECIPES
from experiments.spec import fingerprint, normalize_spec
from models.tabular import baseline_families

PRIORS = {"baseline": 1.0, "retry": 0.8, "ensemble": 0.7, "tune": 0.6, "target": 0.6, "features": 0.5,
          "calibrate": 0.4, "revisit": 0.3, "seed": 0.2}
EXPLORATORY = {"baseline", "revisit", "retry"}
CARRY_EXCLUDE = {"random_state", "random_seed", "n_jobs", "verbose"}


def _arm_type(arm: str | None) -> str:
    return (arm or "manual").split(":")[0]


def _park_key(arm: str | None) -> str:
    parts = (arm or "manual").split(":")
    return ":".join(parts[:2])


class HeuristicAgent:
    name = "heuristic"

    def __init__(self, numeric_columns: list[str] | None = None, has_categorical: bool = True):
        self.numeric_columns = numeric_columns or []
        self.has_categorical = has_categorical

    # Bookkeeping
    def _stats(self, ctx: ResearchContext) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = defaultdict(lambda: {"runs": 0, "improved": 0, "seconds": 0.0})
        for e in ctx.experiments:
            if e["status"] in ("succeeded", "failed", "timeout"):
                s = stats[_arm_type(e["arm"])]
                s["runs"] += 1
                s["improved"] += bool(e["improved"])
                s["seconds"] += e["train_seconds"] or 0.0
        return stats

    def _parked(self, ctx: ResearchContext, patience: int) -> set[str]:
        by_key: dict[str, list[dict]] = defaultdict(list)
        for e in ctx.experiments:
            if e["status"] in ("succeeded", "failed", "timeout"):
                by_key[_park_key(e["arm"])].append(e)
        last_best_change = max((e["finished_at"] or "" for e in ctx.experiments if e["improved"]), default="")
        parked = set()
        for key, runs in by_key.items():
            trailing = 0
            for e in reversed(runs):
                if e["improved"]:
                    break
                trailing += 1
            if trailing >= patience and (runs[-1]["finished_at"] or "") >= last_best_change:
                parked.add(key)
        return parked

    def _family_seconds(self, ctx: ResearchContext) -> dict[str, float]:
        acc: dict[str, list[float]] = defaultdict(list)
        for e in ctx.succeeded():
            if e["kind"] == "train" and e["train_seconds"]:
                acc[e["model"]].append(e["train_seconds"])
        return {k: sum(v) / len(v) for k, v in acc.items()}

    @staticmethod
    def _carry(params: dict[str, Any] | None) -> dict[str, Any]:
        return {k: v for k, v in (params or {}).items() if k not in CARRY_EXCLUDE}

    # Candidate generation
    def candidates(self, ctx: ResearchContext) -> list[tuple[Proposal, float]]:
        """Returns (proposal, estimated_seconds) pairs before scoring."""
        out: list[tuple[Proposal, float]] = []
        secs = self._family_seconds(ctx)
        default_secs = 60.0
        tried_families = {e["model"] for e in ctx.experiments if _arm_type(e["arm"]) == "baseline"
                          or (e["kind"] == "train" and e["status"] != "cancelled")}
        log_target = ctx.metric in ("rmsle",) and ctx.target_stats.get("non_negative")

        for fam in baseline_families(ctx.n_train_rows, ctx.n_features):
            if fam in tried_families:
                continue
            spec = {"kind": "train", "model": fam, "features": []}
            if log_target:
                spec["target_transform"] = "log1p"
            out.append((Proposal(spec, f"Default {fam} on base features gives a reference score for this model "
                                       "family and shows whether it suits this data.", f"baseline:{fam}"),
                        secs.get(fam, default_secs)))

        # Retry GPU failures on CPU once.
        for e in ctx.experiments:
            if e["status"] == "failed" and e["error"] and any(s in e["error"].lower() for s in ("cuda", "gpu")):
                spec = {"kind": e["kind"], "model": e["model"], "features": e["features"],
                        "feature_params": e["feature_params"], "params": e["spec_params"],
                        "target_transform": e["target_transform"], "device": "cpu", "seed": e["seed"] + 1000}
                out.append((Proposal(spec, f"Experiment {e['id']} failed on GPU ({e['error'][:80]}); the same "
                                           "configuration on CPU should run.", f"retry:{e['model']}",
                                     parent_id=e["id"]), secs.get(e["model"], default_secs) * 3))

        fams = ctx.best_by_family
        if len(fams) < 2:
            return out
        ranked = sorted(fams.values(), key=lambda e: e["cv_score"], reverse=ctx.metric_obj.greater_is_better)
        top = ranked[0]
        best_single = top

        # Optuna tuning for the two strongest families on their best feature set.
        tuned = {(e["model"], tuple(e["features"])) for e in ctx.experiments if e["kind"] == "tune"
                 and e["status"] != "cancelled"}
        for fam_best in ranked[:2]:
            key = (fam_best["model"], tuple(fam_best["features"]))
            if key in tuned:
                continue
            spec = {"kind": "tune", "model": fam_best["model"], "features": fam_best["features"],
                    "feature_params": fam_best["feature_params"], "target_transform": fam_best["target_transform"],
                    "n_trials": ctx.loop["tune_trials"], "timeout": ctx.loop["tune_seconds"]}
            out.append((Proposal(spec, f"Optuna search over {fam_best['model']} hyperparameters on features "
                                       f"{fam_best['features'] or ['base']} beats the current "
                                       f"{fam_best['cv_score']:.6f} from exp {fam_best['id']}.",
                                 f"tune:{fam_best['model']}", parent_id=fam_best["id"]),
                        min(ctx.loop["tune_seconds"], secs.get(fam_best["model"], default_secs) * ctx.loop["tune_trials"])))

        # Feature recipes on top of the leading single model.
        for recipe in self._applicable_recipes(ctx, top["model"]):
            if recipe in top["features"]:
                continue
            feats = sorted(set(top["features"]) | {recipe})
            fparams = dict(top["feature_params"])
            if recipe == "numeric_interactions":
                fparams["interaction_columns"] = [c for c in (top.get("top_features") or [])
                                                  if c in self.numeric_columns][:6] or self.numeric_columns[:6]
            spec = {"kind": "train", "model": top["model"], "params": self._carry(top["params"]), "features": feats,
                    "feature_params": fparams, "target_transform": top["target_transform"], "seed": top["seed"]}
            out.append((Proposal(spec, f"Adding {recipe} ({RECIPES[recipe]}) to the best {top['model']} "
                                       f"configuration (exp {top['id']}, {top['cv_score']:.6f}) adds signal the "
                                       "model cannot build from raw columns.",
                                 f"features:{top['model']}:{recipe}", parent_id=top["id"]),
                        secs.get(top["model"], default_secs)))

        # Regression target transform.
        if ctx.task_type == "regression" and ctx.target_stats.get("non_negative") \
                and abs(ctx.target_stats.get("skew", 0)) > 1 and top["target_transform"] == "none":
            spec = {"kind": "train", "model": top["model"], "params": self._carry(top["params"]),
                    "features": top["features"], "feature_params": top["feature_params"], "target_transform": "log1p",
                    "seed": top["seed"]}
            out.append((Proposal(spec, f"Target skew is {ctx.target_stats['skew']:.2f}; fitting log1p(target) "
                                       "stabilises the loss on large values.", f"target:{top['model']}:log1p",
                                 parent_id=top["id"]), secs.get(top["model"], default_secs)))

        # Carry the winning feature set to other families.
        if top["features"]:
            for fam_best in ranked[1:4]:
                if set(fam_best["features"]) == set(top["features"]):
                    continue
                spec = {"kind": "train", "model": fam_best["model"], "params": self._carry(fam_best["params"]),
                        "features": top["features"], "feature_params": top["feature_params"],
                        "target_transform": top["target_transform"], "seed": fam_best["seed"]}
                out.append((Proposal(spec, f"Features {top['features']} improved {top['model']}; they may also "
                                           f"lift {fam_best['model']} ({fam_best['cv_score']:.6f}) and diversify the "
                                           "ensemble.", f"revisit:{fam_best['model']}", parent_id=fam_best["id"]),
                            secs.get(fam_best["model"], default_secs)))

        # Ensembling once there are enough diverse successes.
        singles = [e for e in ctx.succeeded() if e["kind"] != "ensemble"]
        ensembles = ctx.succeeded("ensemble")
        last_ens = max((e["id"] for e in ctx.experiments if e["kind"] == "ensemble"), default=0)
        new_since = sum(1 for e in singles if e["id"] > last_ens)
        if len(singles) >= 3 and (not ensembles or new_since >= ctx.loop["ensemble_every"]) \
                and not any(e["kind"] == "ensemble" and e["status"] in ("queued", "running") for e in ctx.experiments):
            out.append((Proposal({"kind": "ensemble", "method": "hill_climb", "top_k": 8, "seed": len(ensembles)},
                                 f"A hill-climbed blend of the top diverse models beats the best single model "
                                 f"({best_single['cv_score']:.6f}) because their errors are not fully correlated.",
                                 "ensemble:hill_climb"), 30.0))
            if ctx.task_type == "binary" and ctx.metric == "logloss":
                out.append((Proposal({"kind": "ensemble", "method": "hill_climb", "top_k": 8, "calibrate": True,
                                      "seed": len(ensembles)},
                                     "Platt-calibrating the blend (fit in nested CV) lowers log loss.",
                                     "calibrate:ensemble"), 30.0))

        # Seed bagging of the best single model for ensemble diversity.
        if ensembles or len(singles) >= 6:
            used_seeds = {e["seed"] for e in ctx.experiments if e["model"] == top["model"]
                          and e["features"] == top["features"]}
            for k in range(1, 4):
                seed = top["seed"] + k
                if seed in used_seeds:
                    continue
                spec = {"kind": "train", "model": top["model"], "params": self._carry(top["params"]),
                        "features": top["features"], "feature_params": top["feature_params"],
                        "target_transform": top["target_transform"], "seed": seed}
                out.append((Proposal(spec, f"Re-training the best {top['model']} config with seed {seed} gives a "
                                           "decorrelated member for blending.", f"seed:{top['model']}",
                                     parent_id=top["id"]), secs.get(top["model"], default_secs)))
                break
        return out

    def _applicable_recipes(self, ctx: ResearchContext, family: str) -> list[str]:
        ct = ctx.column_types
        out = ["count_encoding", "missing_indicators"]
        if ct.get("numeric", 0) >= 3:
            out.append("row_stats")
        if ct.get("numeric", 0) >= 2:
            out.append("numeric_interactions")
        if ctx.has_datetime:
            out.append("datetime_parts")
        if ctx.has_text:
            out.append("text_basic")
        if ctx.task_type in ("binary", "regression") and ct.get("categorical", 0):
            out.append("target_encoding")
        if family == "linear":
            out.append("log_numeric")
        return out

    # Scoring
    def propose(self, ctx: ResearchContext, n: int) -> list[Proposal]:
        patience = int(ctx.loop["arm_patience"])
        c = float(ctx.loop["exploration_weight"])
        stats = self._stats(ctx)
        parked = self._parked(ctx, patience)
        existing = {e.get("fingerprint") for e in ctx.experiments if e["status"] != "cancelled"}
        total_runs = sum(s["runs"] for s in stats.values())
        explore_cap = float(ctx.loop.get("exploration_budget", 0.35)) * float(ctx.loop["max_experiments"])
        explored = sum(stats[t]["runs"] for t in EXPLORATORY if t != "baseline")

        scored = []
        seen = set()
        for prop, est_seconds in self.candidates(ctx):
            atype = _arm_type(prop.arm)
            if _park_key(prop.arm) in parked:
                continue
            if atype in EXPLORATORY - {"baseline"} and explored >= explore_cap:
                continue
            try:
                spec = normalize_spec(prop.spec, ctx.task_type)
            except ValueError:
                continue
            fp = fingerprint(spec) if spec["kind"] != "ensemble" else f"ens-{prop.arm}-{len(ctx.experiments)}"
            if fp in existing or fp in seen:
                continue
            seen.add(fp)
            s = stats.get(atype, {"runs": 0, "improved": 0})
            success = (s["improved"] + 1) / (s["runs"] + 2)
            bonus = c * math.sqrt(math.log(total_runs + 2) / (s["runs"] + 1))
            cost = 0.1 * math.log1p(est_seconds / 60.0)
            prop.priority = round(PRIORS.get(atype, 0.3) + success + bonus - cost, 4)
            scored.append(prop)
        scored.sort(key=lambda p: -p.priority)
        return scored[:n]
