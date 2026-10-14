"""Kaggle agent CLI. Every command prints one JSON object to stdout ({"ok": ..., "result"|"error": ...});
logs go to stderr. Run `python main.py -h` or `python main.py <command> -h` for usage."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

COMMANDS: dict[str, Callable[[argparse.Namespace], Any]] = {}


class UserError(RuntimeError):
    pass


def command(name: str):
    def wrap(fn):
        COMMANDS[name] = fn
        return fn
    return wrap


def _ws(args):
    from core.workspace import resolve_workspace
    return resolve_workspace(getattr(args, "competition", None))


def _tracker():
    from tracking.db import Tracker
    return Tracker()


def _load_json_arg(value: str) -> Any:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


# Onboarding / info

@command("start")
def cmd_start(args):
    from competitions.kaggle_client import make_client
    from competitions.onboarding import onboard
    return onboard(args.competition, make_client(args.data_dir), _tracker(), data_dir=args.data_dir,
                   force=args.force_validation, metric=args.metric, redownload=args.redownload)


@command("mock")
def cmd_mock(args):
    from competitions.mock import create_mock_competition
    path = create_mock_competition(Path(args.out), args.kind, n_train=args.rows, n_test=max(200, args.rows // 3),
                                   seed=args.seed)
    return {"data_dir": str(path.resolve()), "kind": args.kind,
            "next_step": f"python main.py start --competition mock-{args.kind.replace('_', '-')} --data-dir {path}"}


@command("status")
def cmd_status(args):
    from core.resources import gpu_info
    from experiments.scheduler import daemon_state
    from experiments.service import public_view
    from submissions.manager import SubmissionManager
    ws, tracker = _ws(args), _tracker()
    config = ws.load_config()
    version = ws.load_validation()["version"]
    exps = tracker.list_experiments(ws.slug, validation_version=version)
    counts: dict[str, int] = {}
    for e in exps:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    best = tracker.best_experiment(ws.slug, version, config["direction"])
    out = {
        "competition": ws.slug, "title": config.get("title"), "task_type": config["task_type"],
        "metric": config["metric"], "direction": config["direction"], "validation_version": version,
        "experiments": counts,
        "best": public_view(best) if best else None,
        "running": [public_view(e) for e in exps if e["status"] == "running"],
        "daemon": daemon_state(ws),
        "gpus": gpu_info(),
        "rules": {k: config["rules"].get(k) for k in ("external_data", "pretrained_models", "automated_submissions",
                                                       "code_competition")},
        "autosubmit": config.get("autosubmit"),
    }
    if args.with_submissions:
        out["submission_usage"] = SubmissionManager(ws, tracker).usage_today()
    return out


@command("info")
def cmd_info(args):
    ws = _ws(args)
    out = {"config": ws.load_config(), "validation": ws.load_validation()}
    if args.profile:
        out["profile"] = ws.load_profile()
    return out


@command("gpu")
def cmd_gpu(args):
    from core.resources import gpu_capabilities, system_snapshot, torch_cuda_info
    return {"system": system_snapshot(), "torch": torch_cuda_info(), "capabilities": gpu_capabilities(args.refresh)}


# Rules, validation, autosubmit

@command("rules")
def cmd_rules(args):
    from core.compliance import PERMISSIONS, VALUES
    ws = _ws(args)
    config = ws.load_config()
    if args.action == "show":
        return config["rules"]
    changes = {}
    for perm in PERMISSIONS:
        value = getattr(args, perm)
        if value is not None:
            if value not in VALUES:
                raise UserError(f"--{perm.replace('_', '-')} must be one of {VALUES}")
            changes[perm] = value
    if not changes and not args.notes:
        raise UserError("Nothing to set. Example: rules set --external-data forbidden --pretrained-models allowed")
    config["rules"].update(changes)
    if args.notes:
        config["rules"]["notes"] = args.notes
    config["rules"]["confirmed_by_user"] = True
    ws.save_config(config)
    _tracker().log_event(ws.slug, "rules_updated", json.dumps(changes))
    return config["rules"]


@command("autosubmit")
def cmd_autosubmit(args):
    from tracking.db import utcnow
    ws = _ws(args)
    config = ws.load_config()
    if args.action == "disable":
        config["autosubmit"] = {**config.get("autosubmit", {}), "enabled": False, "max_daily": 0}
    else:
        if not args.i_authorize:
            raise UserError("Autonomous submission must be explicitly authorized: add --i-authorize")
        if args.max_daily is None or args.max_daily < 1:
            raise UserError("Set a daily budget with --max-daily N (N >= 1)")
        kaggle_max = int(config.get("max_daily_submissions") or 5)
        if args.max_daily > kaggle_max:
            raise UserError(f"--max-daily cannot exceed the competition limit ({kaggle_max})")
        if config["rules"].get("automated_submissions") != "allowed":
            raise UserError("Confirm the competition allows automated API submissions first: "
                            "rules set --automated-submissions allowed")
        config["autosubmit"] = {"enabled": True, "max_daily": args.max_daily, "authorized_at": utcnow(),
                                "min_improvement": args.min_improvement}
    ws.save_config(config)
    _tracker().log_event(ws.slug, "autosubmit", json.dumps(config["autosubmit"]))
    return config["autosubmit"]


@command("validation")
def cmd_validation(args):
    ws = _ws(args)
    if args.action == "show":
        return ws.load_validation()
    from competitions.profiler import read_table
    from competitions.tabular import _key
    from core.settings import load_settings
    from evaluation.validation import choose_scheme, make_folds, save_validation
    if not args.yes:
        raise UserError("Resetting validation creates a new version; earlier scores become incomparable and the "
                        "agent restarts its search. Re-run with --yes to confirm.")
    config, profile = ws.load_config(), ws.load_profile()
    overrides = {k: v for k, v in {"strategy": args.strategy, "n_splits": args.n_splits,
                                   "group_column": args.group_column, "time_column": args.time_column,
                                   "holdout_fraction": args.holdout_fraction}.items() if v is not None}
    config["validation"] = {**config.get("validation", {}), **overrides}
    scheme = choose_scheme(config["task_type"], profile, config["validation"], load_settings()["validation"])
    train = read_table(ws.data_dir / config["data"]["train_file"])
    lookup = {_key(c): i for i, c in enumerate(config.get("classes") or [])}
    y = (train[config["target"]].map(lambda v: lookup[_key(v)]).to_numpy() if lookup
         else train[config["target"]].to_numpy(dtype=float))
    folds = make_folds(train, y, scheme, config["task_type"])
    version = save_validation(ws, scheme, folds, force=True)
    ws.save_config(config)
    _tracker().log_event(ws.slug, "validation_reset", f"new validation version {version}", scheme)
    return ws.load_validation()


# Experiments

@command("exp")
def cmd_exp(args):
    from experiments import service
    ws, tracker = _ws(args), _tracker()
    action = args.action
    if action == "list":
        statuses = tuple(args.status.split(",")) if args.status else None
        version = None if args.all_versions else ws.load_validation()["version"]
        rows = tracker.list_experiments(ws.slug, statuses, version)
        return [service.public_view(e, args.show_holdout) for e in rows[-args.limit:]]
    if action == "leaderboard":
        return service.leaderboard(ws, tracker, args.limit)
    if action == "show":
        exp = tracker.get_experiment(args.id)
        if not exp or exp["competition_id"] != ws.slug:
            raise UserError(f"Experiment {args.id} not found")
        out = service.public_view(exp, args.show_holdout)
        out.update(spec=exp["spec"], params=exp["params"], fold_scores=exp["fold_scores"],
                   validation_version=exp["validation_version"], code_version=exp["code_version"],
                   dataset_version=exp["dataset_version"], artifact_dir=exp["artifact_dir"], error=exp["error"],
                   peak_rss_mb=exp["peak_rss_mb"])
        result = ws.experiment_dir(args.id) / "result.json"
        if result.exists():
            data = json.loads(result.read_text())
            data.pop("holdout_score", None)
            out["result"] = data
        return out
    if action == "propose":
        if not args.spec or not args.hypothesis:
            raise UserError("exp propose needs --spec JSON and --hypothesis TEXT")
        return service.queue_experiment(ws, tracker, _load_json_arg(args.spec), args.hypothesis,
                                        arm=args.arm or "manual", proposed_by=args.proposed_by,
                                        priority=args.priority, force=args.force)
    if action == "suggest":
        from agents.base import build_context
        from agents.command import make_agent
        return [p.to_dict() for p in make_agent(ws).propose(build_context(ws, tracker), args.n)]
    if action == "run":
        if args.inline:
            return service.execute(args.id, tracker)
        from core.proc import spawn_worker
        exp = tracker.get_experiment(args.id)
        if not exp or exp["status"] != "queued":
            raise UserError(f"Experiment {args.id} is not queued")
        proc = spawn_worker(["worker", "--exp-id", str(args.id)], ws.experiment_dir(args.id) / "worker.log",
                            env={"PYTHONUNBUFFERED": "1"})
        try:
            proc.wait(timeout=args.timeout)
        except Exception:
            from core.proc import kill_tree
            kill_tree(proc.pid)
            tracker.update_experiment(args.id, status="timeout", error=f"exceeded --timeout {args.timeout}s")
        return service.public_view(tracker.get_experiment(args.id))
    if action == "cancel":
        exp = tracker.get_experiment(args.id)
        if not exp or exp["status"] not in ("queued", "blocked"):
            raise UserError("Only queued or blocked experiments can be cancelled (stop the daemon for running ones)")
        tracker.update_experiment(args.id, status="cancelled")
        return {"experiment_id": args.id, "status": "cancelled"}
    if action == "requeue":
        exp = tracker.get_experiment(args.id)
        if not exp or exp["status"] not in ("failed", "timeout", "blocked", "cancelled"):
            raise UserError("Only failed/timeout/blocked/cancelled experiments can be re-queued")
        tracker.update_experiment(args.id, status="queued", attempts=0, error=None)
        return {"experiment_id": args.id, "status": "queued"}
    if action == "compare":
        return service.compare(ws, tracker, args.ids)
    if action == "logs":
        path = ws.experiment_dir(args.id) / ("worker.log" if args.worker else "train.log")
        if not path.exists():
            raise UserError(f"No log at {path}")
        return {"path": str(path), "lines": path.read_text(encoding="utf-8", errors="replace").splitlines()[-args.tail:]}
    if action == "failures":
        rows = tracker.list_experiments(ws.slug, ("failed", "timeout", "blocked"))
        return [{"id": e["id"], "status": e["status"], "model": e["model_family"], "arm": e["arm"],
                 "hypothesis": e["hypothesis"], "error": (e["error"] or "")[:1500]} for e in rows[-args.limit:]]
    raise UserError(f"Unknown exp action {action}")


def _run_scheduler(ws, tracker, *, propose: bool, max_hours=None, max_experiments=None, stop_when_idle=True):
    from experiments.scheduler import Scheduler
    return Scheduler(ws, tracker, propose=propose, max_hours=max_hours, max_experiments=max_experiments,
                     stop_when_idle=stop_when_idle).run()


@command("baseline")
def cmd_baseline(args):
    from agents.base import build_context
    from agents.heuristic import HeuristicAgent
    from experiments.service import queue_experiment
    ws, tracker = _ws(args), _tracker()
    ctx = build_context(ws, tracker)
    proposals = [p for p, _ in HeuristicAgent().candidates(ctx) if p.arm.startswith("baseline:")]
    if args.models:
        proposals = [p for p in proposals if p.spec["model"] in args.models]
    queued = [queue_experiment(ws, tracker, p.spec, p.hypothesis, arm=p.arm, proposed_by="baseline",
                               priority=10 - i) for i, p in enumerate(proposals)]
    out = {"queued": queued}
    if args.run:
        out["run"] = _run_scheduler(ws, tracker, propose=False)
    return out


@command("tune")
def cmd_tune(args):
    from experiments.service import queue_experiment
    ws, tracker = _ws(args), _tracker()
    spec = {"kind": "tune", "model": args.model, "features": args.features or [], "n_trials": args.trials,
            "timeout": args.timeout}
    out = {"queued": queue_experiment(ws, tracker, spec, args.hypothesis or
                                      f"Optuna tuning of {args.model} improves on its default parameters",
                                      arm=f"tune:{args.model}", proposed_by="user", priority=5)}
    if args.run:
        out["run"] = _run_scheduler(ws, tracker, propose=False)
    return out


@command("ensemble")
def cmd_ensemble(args):
    from experiments.service import queue_experiment
    ws, tracker = _ws(args), _tracker()
    spec = {"kind": "ensemble", "method": args.method, "top_k": args.top_k, "calibrate": args.calibrate}
    if args.members:
        spec["members"] = args.members
    out = {"queued": queue_experiment(ws, tracker, spec, "Blending top models beats the best single model",
                                      arm="ensemble:manual", proposed_by="user", priority=5, force=args.force)}
    if args.run:
        out["run"] = _run_scheduler(ws, tracker, propose=False)
    return out


@command("loop")
def cmd_loop(args):
    ws, tracker = _ws(args), _tracker()
    return _run_scheduler(ws, tracker, propose=not args.no_propose, max_hours=args.max_hours,
                          max_experiments=args.max_experiments, stop_when_idle=not args.keep_alive)


@command("resume")
def cmd_resume(args):
    from agents.base import loop_settings
    from experiments.service import recover_stale
    ws, tracker = _ws(args), _tracker()
    recovered = recover_stale(ws, tracker, int(loop_settings(ws.load_config())["max_attempts"]))
    out = {"recovered": recovered}
    if not args.recover_only:
        out["run"] = _run_scheduler(ws, tracker, propose=args.continue_research)
    return out


@command("stop")
def cmd_stop(args):
    from experiments.scheduler import daemon_state, request_stop
    ws = _ws(args)
    state = daemon_state(ws)
    if not state or not state.get("alive"):
        return {"message": "No daemon running"}
    request_stop(ws, now=args.now)
    return {"message": "Stop requested" + (" (workers killed and re-queued)" if args.now else
                                           " (finishes running experiments first)"), "daemon_pid": state["pid"]}


@command("promote")
def cmd_promote(args):
    from experiments.service import promote
    return promote(_ws(args), _tracker(), args.id)


# Submissions

def _confirm_interactively(sub: dict) -> None:
    """Human approval gate. Requires a real terminal so automated agents cannot self-approve."""
    if not sys.stdin.isatty():
        raise UserError("Submission approval requires an interactive terminal (a human). Run this command "
                        "yourself in PowerShell, or approve from the dashboard.")
    print(f"About to submit {sub['file_path']} (CV {sub['cv_score']}) to Kaggle.", file=sys.stderr)
    answer = input(f"Type the submission id ({sub['id']}) to confirm: ").strip()
    if answer != str(sub["id"]):
        raise UserError("Confirmation did not match; nothing submitted")


@command("submit")
def cmd_submit(args):
    from submissions.manager import SubmissionManager
    ws, tracker = _ws(args), _tracker()
    mgr = SubmissionManager(ws, tracker)
    if args.action == "prepare":
        if args.exp_id is None:
            raise UserError("submit prepare needs --exp-id")
        return mgr.prepare(args.exp_id, args.message or "", force=args.force)
    if args.action == "list":
        return tracker.list_submissions(ws.slug)
    if args.action == "usage":
        return mgr.usage_today()
    if args.action == "refresh":
        return mgr.refresh()
    if args.id is None:
        raise UserError(f"submit {args.action} needs --id")
    if args.action == "approve":
        sub = tracker.get_submission(args.id)
        if not sub:
            raise UserError(f"Submission {args.id} not found")
        _confirm_interactively(sub)
        return mgr.approve(args.id, approver="user")
    if args.action == "reject":
        return mgr.reject(args.id, args.message or "")
    raise UserError(f"Unknown submit action {args.action}")


# Custom features

@command("features")
def cmd_features(args):
    from experiments import custom_code
    from experiments.features import RECIPES
    ws = _ws(args)
    if args.action == "list":
        return {"recipes": RECIPES, "custom": custom_code.list_custom(ws)}
    if args.action == "auto-approve":
        if args.state not in ("on", "off"):
            raise UserError("features auto-approve needs --state on|off")
        config = ws.load_config()
        config["custom_code"] = {**(config.get("custom_code") or {}), "auto_approve": args.state == "on"}
        ws.save_config(config)
        _tracker().log_event(ws.slug, "custom_code_policy", f"auto_approve={args.state}")
        return config["custom_code"]
    if not args.name:
        raise UserError(f"features {args.action} needs --name")
    if args.action == "test":
        import os
        from experiments.sandbox import run_sandboxed
        return run_sandboxed(ws, "feature", args.name, human=os.environ.get("KAGGLE_AGENT_ROLE") != "agent")
    if not sys.stdin.isatty() and not args.yes:
        raise UserError("Approving code requires an interactive terminal or --yes after you have reviewed it")
    problems = custom_code.screen(ws, args.name)
    if problems and not args.force:
        raise UserError(f"Safety screen flagged custom:{args.name}: {problems}. Review it; --force approves anyway.")
    return custom_code.approve(ws, args.name)


@command("data")
def cmd_data(args):
    from competitions.profiler import read_table
    ws = _ws(args)
    config = ws.load_config()
    df = read_table(ws.data_dir / config["data"]["train_file" if args.split == "train" else "test_file"])
    if args.column:
        if args.column not in df.columns:
            raise UserError(f"No column {args.column!r}. Columns: {list(df.columns)}")
        s = df[args.column]
        out = {"dtype": str(s.dtype), "missing": int(s.isna().sum()), "nunique": int(s.nunique())}
        out["top_values"] = {str(k): int(v) for k, v in s.value_counts(dropna=False).head(25).items()}
        if s.dtype.kind in "if":
            out["describe"] = {k: float(v) for k, v in s.describe().items()}
        target = config["target"]
        if args.split == "train" and target in df.columns and args.column != target:
            top = s.astype(str).value_counts().head(15).index
            sub = df[s.astype(str).isin(top)]
            rates = pd_crosstab(sub[args.column].astype(str), sub[target])
            out["target_distribution_by_value"] = rates
        return out
    return {"shape": list(df.shape), "columns": list(df.columns),
            "rows": json.loads(df.head(args.rows).to_json(orient="records"))}


def pd_crosstab(a, b) -> dict:
    import pandas as pd
    tab = pd.crosstab(a, b, normalize="index").round(3)
    return {str(i): {str(c): float(v) for c, v in row.items()} for i, row in tab.iterrows()}


@command("analyze")
def cmd_analyze(args):
    import os
    from experiments.sandbox import run_sandboxed
    return run_sandboxed(_ws(args), "analysis", args.name, timeout=args.timeout,
                         human=os.environ.get("KAGGLE_AGENT_ROLE") != "agent")


@command("sandbox-exec")
def cmd_sandbox_exec(args):
    import os
    from experiments import sandbox
    if os.environ.get("KAGGLE_AGENT_ROLE") != "sandbox":
        raise UserError("sandbox-exec is internal; use `features test` or `analyze`")
    ws = _ws(args)
    return sandbox.exec_feature_test(ws, args.name) if args.kind == "feature" else sandbox.exec_analysis(ws, args.name)


@command("research")
def cmd_research(args):
    from agents.claude_research import supervise
    ws, tracker = _ws(args), _tracker()
    return supervise(ws, tracker, with_daemon=args.with_daemon, max_sessions=args.max_sessions,
                     every_minutes=args.every_minutes, once=args.once)


# Reporting & maintenance

@command("report")
def cmd_report(args):
    from tracking.report import generate_report
    ws = _ws(args)
    text = generate_report(ws, _tracker())
    path = Path(args.out) if args.out else ws.root / "REPORT.md"
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "markdown": text if args.print else None}


@command("events")
def cmd_events(args):
    ws = _ws(args)
    return _tracker().list_events(ws.slug, args.limit, args.kind)


@command("cleanup")
def cmd_cleanup(args):
    from core.settings import load_settings
    from experiments.scheduler import prune_models
    ws, tracker = _ws(args), _tracker()
    keep = args.keep_top or int(load_settings()["loop"]["keep_top_models"])
    if not args.yes:
        raise UserError(f"This deletes fold model files of experiments outside the top {keep} (predictions are "
                        "kept). Re-run with --yes to confirm.")
    return prune_models(ws, tracker, keep)


@command("dashboard")
def cmd_dashboard(args):
    import subprocess
    app = Path(__file__).parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
           "--server.address", args.address, "--server.headless", "true"]
    print(json.dumps({"ok": True, "result": {"url": f"http://{args.address}:{args.port}"}}), flush=True)
    raise SystemExit(subprocess.call(cmd))


@command("serve")
def cmd_serve(args):
    from agents.api_server import serve
    serve(args.host, args.port)


@command("worker")
def cmd_worker(args):
    from experiments.service import execute
    return execute(args.exp_id)


# Parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description="Autonomous Kaggle competition agent")
    p.add_argument("-v", "--verbose", action="store_true", help="info-level logs on stderr")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_text):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("-c", "--competition", help="competition slug (default: active competition)")
        return sp

    sp = add("start", "onboard a competition (download, inspect, configure, freeze folds)")
    sp.add_argument("--data-dir", help="serve files from a local directory instead of the Kaggle API (simulated)")
    sp.add_argument("--metric", help="override the evaluation metric")
    sp.add_argument("--force-validation", action="store_true", help="regenerate folds (new validation version)")
    sp.add_argument("--redownload", action="store_true")

    sp = sub.add_parser("mock", help="generate a synthetic competition for offline testing")
    sp.add_argument("action", choices=["create"])
    sp.add_argument("--kind", default="binary", choices=["binary", "multiclass", "regression", "binary_label"])
    sp.add_argument("--out", required=True)
    sp.add_argument("--rows", type=int, default=3000)
    sp.add_argument("--seed", type=int, default=0)

    sp = add("status", "competition, best model, running work, daemon state")
    sp.add_argument("--with-submissions", action="store_true", help="include today's submission usage (API call)")
    sp = add("info", "competition config and validation scheme")
    sp.add_argument("--profile", action="store_true", help="include the full dataset profile")
    sp = sub.add_parser("gpu", help="hardware, CUDA and per-library GPU capability")
    sp.add_argument("--refresh", action="store_true", help="re-probe GPU support")

    sp = add("rules", "show or record competition rule decisions")
    sp.add_argument("action", choices=["show", "set"])
    for perm in ("external_data", "pretrained_models", "automated_submissions"):
        sp.add_argument(f"--{perm.replace('_', '-')}", dest=perm, choices=["allowed", "forbidden", "unknown"])
    sp.add_argument("--notes")

    sp = add("autosubmit", "enable/disable autonomous submissions for one competition")
    sp.add_argument("action", choices=["enable", "disable"])
    sp.add_argument("--max-daily", type=int)
    sp.add_argument("--min-improvement", type=float, help="CV gain over the last submitted CV required")
    sp.add_argument("--i-authorize", action="store_true")

    sp = add("validation", "show or deliberately reset the validation scheme")
    sp.add_argument("action", choices=["show", "reset"])
    sp.add_argument("--strategy", choices=["stratified", "kfold", "group", "time", "holdout"])
    sp.add_argument("--n-splits", type=int)
    sp.add_argument("--group-column")
    sp.add_argument("--time-column")
    sp.add_argument("--holdout-fraction", type=float)
    sp.add_argument("--yes", action="store_true")

    sp = add("exp", "experiment operations")
    sp.add_argument("action", choices=["list", "leaderboard", "show", "propose", "suggest", "run", "cancel", "requeue",
                                       "compare", "logs", "failures"])
    sp.add_argument("--id", type=int)
    sp.add_argument("--ids", type=int, nargs="+")
    sp.add_argument("--status", help="comma-separated statuses")
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--all-versions", action="store_true")
    sp.add_argument("--show-holdout", action="store_true", help="reveal holdout scores (do not select on them)")
    sp.add_argument("--spec", help="JSON spec or @file.json")
    sp.add_argument("--hypothesis")
    sp.add_argument("--arm")
    sp.add_argument("--proposed-by", default="user")
    sp.add_argument("--priority", type=float, default=1.0)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--n", type=int, default=5)
    sp.add_argument("--inline", action="store_true", help="run in this process (debugging)")
    sp.add_argument("--timeout", type=float, default=7200)
    sp.add_argument("--tail", type=int, default=60)
    sp.add_argument("--worker", action="store_true", help="show the worker stdout log instead of train.log")

    sp = add("baseline", "queue baseline models for every suitable family")
    sp.add_argument("--models", nargs="+")
    sp.add_argument("--run", action="store_true", help="run the queue to completion")
    sp = add("tune", "queue an Optuna tuning experiment")
    sp.add_argument("--model", required=True)
    sp.add_argument("--features", nargs="*")
    sp.add_argument("--trials", type=int, default=40)
    sp.add_argument("--timeout", type=int, default=1800)
    sp.add_argument("--hypothesis")
    sp.add_argument("--run", action="store_true")
    sp = add("ensemble", "queue an ensemble of the best experiments")
    sp.add_argument("--members", type=int, nargs="+")
    sp.add_argument("--method", default="hill_climb", choices=["hill_climb", "mean", "rank_mean"])
    sp.add_argument("--top-k", type=int, default=8)
    sp.add_argument("--calibrate", action="store_true")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--run", action="store_true")

    sp = add("loop", "run the autonomous research loop (daemon)")
    sp.add_argument("--max-hours", type=float)
    sp.add_argument("--max-experiments", type=int)
    sp.add_argument("--no-propose", action="store_true", help="only drain the existing queue")
    sp.add_argument("--keep-alive", action="store_true", help="keep polling when idle (for remote agents)")
    sp = add("resume", "recover interrupted experiments and finish the queue")
    sp.add_argument("--recover-only", action="store_true")
    sp.add_argument("--continue-research", action="store_true", help="also let the agent propose new work")
    sp = add("stop", "ask a running daemon to stop")
    sp.add_argument("--now", action="store_true", help="kill running workers (they resume later)")
    sp = add("promote", "mark an experiment as the selected model (reveals its holdout score)")
    sp.add_argument("--id", type=int, required=True)

    sp = add("submit", "submission workflow")
    sp.add_argument("action", choices=["prepare", "list", "approve", "reject", "refresh", "usage"])
    sp.add_argument("--exp-id", type=int)
    sp.add_argument("--id", type=int)
    sp.add_argument("--message")
    sp.add_argument("--force", action="store_true")

    sp = add("features", "feature recipes and approval of custom feature code")
    sp.add_argument("action", choices=["list", "approve", "auto-approve", "test"])
    sp.add_argument("--name")
    sp.add_argument("--state", choices=["on", "off"], help="for auto-approve")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--force", action="store_true", help="approve despite safety-screen findings")

    sp = add("data", "read-only peek at the competition data")
    sp.add_argument("--split", choices=["train", "test"], default="train")
    sp.add_argument("--rows", type=int, default=10)
    sp.add_argument("--column", help="value counts / stats for one column")

    sp = add("analyze", "run a screened analysis script (analysis/<name>.py) in a sandbox")
    sp.add_argument("--name", required=True)
    sp.add_argument("--timeout", type=float, default=600)

    sp = add("sandbox-exec", "(internal) execute screened code inside the sandbox process")
    sp.add_argument("--kind", choices=["feature", "analysis"], required=True)
    sp.add_argument("--name", required=True)

    sp = add("research", "autonomous Claude Code research sessions (optionally with the daemon)")
    sp.add_argument("--with-daemon", action="store_true", help="also keep the scheduler daemon running")
    sp.add_argument("--max-sessions", type=int)
    sp.add_argument("--every-minutes", type=float)
    sp.add_argument("--once", action="store_true", help="run a single session now and exit")

    sp = add("report", "write a Markdown research summary")
    sp.add_argument("--out")
    sp.add_argument("--print", action="store_true")
    sp = add("events", "audit log")
    sp.add_argument("--limit", type=int, default=50)
    sp.add_argument("--kind")
    sp = add("cleanup", "delete model binaries outside the top-N")
    sp.add_argument("--keep-top", type=int)
    sp.add_argument("--yes", action="store_true")

    sp = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    sp.add_argument("--port", type=int, default=8501)
    sp.add_argument("--address", default="127.0.0.1")
    sp = sub.add_parser("serve", help="HTTP JSON API for remote agents (token required)")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp = sub.add_parser("worker", help="(internal) run one experiment")
    sp.add_argument("--exp-id", type=int, required=True)
    return p


AGENT_FORBIDDEN = {("features", "approve"), ("features", "auto-approve"), ("submit", "approve"), ("rules", "set"),
                   ("autosubmit", None), ("validation", "reset"), ("cleanup", None), ("start", None),
                   ("research", None), ("sandbox-exec", None)}


def _agent_forbidden(args) -> bool:
    """Research agents (KAGGLE_AGENT_ROLE=agent) may not approve, change rules/validation or delete."""
    import os
    if os.environ.get("KAGGLE_AGENT_ROLE") != "agent":
        return False
    action = getattr(args, "action", None)
    return (args.cmd, None) in AGENT_FORBIDDEN or (args.cmd, action) in AGENT_FORBIDDEN


def run(argv: list[str] | None = None) -> tuple[int, dict]:
    from core.settings import load_dotenv
    load_dotenv()
    args = build_parser().parse_args(argv)
    if _agent_forbidden(args):
        return 1, {"ok": False, "command": args.cmd,
                   "error": "This command is reserved for the human operator and is refused for research agents."}
    logging.basicConfig(level=logging.INFO if args.verbose or args.cmd in ("worker", "loop", "resume", "baseline", "research")
                        else logging.WARNING, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        result = COMMANDS[args.cmd](args)
        return 0, {"ok": True, "command": args.cmd, "result": result}
    except KeyboardInterrupt:
        return 130, {"ok": False, "command": args.cmd, "error": "interrupted"}
    except Exception as exc:  # every failure becomes a structured error for agents
        logging.getLogger("kaggle_agent").debug("command failed", exc_info=True)
        return 1, {"ok": False, "command": args.cmd, "error": str(exc), "error_type": type(exc).__name__}


def main() -> None:
    code, payload = run()
    print(json.dumps(payload, indent=2, default=str))
    sys.exit(code)


if __name__ == "__main__":
    main()
