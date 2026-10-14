"""Markdown research summary: what was tried, what worked, what failed, what comes next."""
from __future__ import annotations

from collections import defaultdict

from agents.base import build_context
from agents.command import make_agent
from agents.heuristic import HeuristicAgent
from core.workspace import Workspace
from tracking.db import Tracker, utcnow


def _fmt(x, digits: int = 6) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def generate_report(ws: Workspace, tracker: Tracker) -> str:
    config, profile = ws.load_config(), ws.load_profile()
    ctx = build_context(ws, tracker)
    val = ws.load_validation()
    exps = ctx.experiments
    done = ctx.succeeded()
    status_counts = defaultdict(int)
    for e in exps:
        status_counts[e["status"]] += 1
    hours = sum(e["train_seconds"] or 0 for e in exps) / 3600
    lines = [
        f"# Research summary: {config.get('title', ws.slug)}",
        "",
        f"_Generated {utcnow()} for `{ws.slug}`._",
        "",
        f"- **Task:** {config['task_type']}, target `{config['target']}`, "
        f"{profile.get('n_train_rows')} train / {profile.get('n_test_rows')} test rows",
        f"- **Metric:** {config['metric']} ({config['direction']}; source: {config['metric_source']})",
        f"- **Validation:** {val['strategy']}, {val['n_splits']} folds, holdout {val['holdout_fraction']:.0%} "
        f"(version `{val['version']}`; {val.get('reason', '')})",
        f"- **Experiments:** {len(exps)} total, " + ", ".join(f"{v} {k}" for k, v in sorted(status_counts.items()))
        + f"; {hours:.2f} compute-hours",
    ]
    if ctx.best:
        b = ctx.best
        lines.append(f"- **Best CV:** {_fmt(b['cv_score'])} ± {_fmt(b['cv_std'])} (exp {b['id']}, {b['model']}, "
                     f"features {b['features'] or ['base']})")
    promoted = [e for e in tracker.list_experiments(ws.slug, "succeeded") if e["promoted"]]
    if promoted:
        p = promoted[-1]
        lines.append(f"- **Promoted:** exp {p['id']} (CV {_fmt(p['cv_score'])}, holdout {_fmt(p['holdout_score'])})")

    lines += ["", "## What worked", "", "Experiments that set a new best CV score:", "",
              "| exp | arm | CV | gain | hypothesis |", "|---|---|---|---|---|"]
    for e in done:
        if e["improved"]:
            gain = "" if e["best_before"] is None else _fmt(abs(e["cv_score"] - e["best_before"]))
            lines.append(f"| {e['id']} | {e['arm'] or e['kind']} | {_fmt(e['cv_score'])} | {gain} | "
                         f"{e['hypothesis'][:120]} |")

    lines += ["", "## Model families", "", "| family | best CV | runs | mean seconds |", "|---|---|---|---|"]
    fam_runs = defaultdict(list)
    for e in done:
        fam_runs[e["model"]].append(e)
    for fam, runs in sorted(fam_runs.items(), key=lambda kv: kv[0]):
        best = max(runs, key=lambda r: r["cv_score"]) if ctx.metric_obj.greater_is_better else min(
            runs, key=lambda r: r["cv_score"])
        mean_s = sum(r["train_seconds"] or 0 for r in runs) / len(runs)
        lines.append(f"| {fam} | {_fmt(best['cv_score'])} | {len(runs)} | {mean_s:.0f} |")

    lines += ["", "## What did not work", ""]
    flops = [e for e in done if e["improved"] is False and e["best_before"] is not None]
    for e in flops[-15:]:
        delta = e["cv_score"] - e["best_before"]
        lines.append(f"- exp {e['id']} [{e['arm'] or e['kind']}]: {_fmt(e['cv_score'])} ({delta:+.6f} vs best then) "
                     f"— {e['hypothesis'][:110]}")
    parked = HeuristicAgent()._parked(ctx, int(ctx.loop["arm_patience"]))
    if parked:
        lines.append(f"- Parked approaches (no gain in {ctx.loop['arm_patience']} consecutive tries): "
                     + ", ".join(sorted(parked)))
    failed = [e for e in exps if e["status"] in ("failed", "timeout")]
    if failed:
        lines += ["", "### Failures", ""]
        lines += [f"- exp {e['id']} ({e['model']}, {e['status']}): {e['error']}" for e in failed[-10:]]
    blocked = [e for e in exps if e["status"] == "blocked"]
    if blocked:
        lines += ["", "### Blocked by competition rules", ""]
        lines += [f"- exp {e['id']}: {e['error']}" for e in blocked]

    subs = tracker.list_submissions(ws.slug)
    lines += ["", "## Submissions", ""]
    if subs:
        lines += ["| id | exp | status | CV | public LB |", "|---|---|---|---|---|"]
        lines += [f"| {s['id']} | {s['experiment_id']} | {s['status']} | {_fmt(s['cv_score'])} | "
                  f"{_fmt(s['public_score'])} |" for s in subs]
        lines.append("")
        lines.append("_Public leaderboard scores are recorded for reference only; experiment selection uses local CV._")
    else:
        lines.append("None yet.")

    lines += ["", "## Next steps", ""]
    try:
        upcoming = make_agent(ws).propose(ctx, 5)
    except Exception as exc:
        upcoming = []
        lines.append(f"(agent unavailable: {exc})")
    queued = [e for e in exps if e["status"] == "queued"]
    for e in queued[:5]:
        lines.append(f"- queued exp {e['id']} [{e['arm']}]: {e['hypothesis'][:140]}")
    for p in upcoming:
        lines.append(f"- [{p.arm}] (priority {p.priority:.2f}) {p.hypothesis[:140]}")
    if not queued and not upcoming:
        lines.append("- The research agent has no further proposals: the search has converged for the current "
                     "recipes. Consider new feature ideas (custom features) or an LLM agent.")

    if profile.get("warnings"):
        lines += ["", "## Data warnings", ""] + [f"- {w}" for w in profile["warnings"]]
    return "\n".join(lines) + "\n"
