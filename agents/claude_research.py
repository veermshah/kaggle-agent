"""Autonomous research supervisor: periodic headless Claude Code sessions alongside the scheduler daemon.

    python main.py research -c <slug> --with-daemon

The daemon keeps the GPU busy without any LLM. Every `every_minutes` (once at least `min_new_results`
experiments have finished since the previous session) a fresh `claude -p` session starts. It reads the
results and its own notes, studies the data, may write feature modules into custom_features/, queues
experiments, and appends to research_notes.md for the next session. Each session starts with clean
context, so this can run for days.

Sandbox: --permission-mode dontAsk with an allow-list (main.py commands, read-only tools, writes only to
custom_features/ and the notes file). KAGGLE_AGENT_ROLE=agent makes main.py refuse approvals, rule changes,
validation resets and cleanup. Feature code runs only after approval: a human `features approve`, or, if
the user opted in with `custom_code.auto_approve: true`, after the static safety screen passes.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from core.settings import PROJECT_ROOT
from core.workspace import Workspace
from tracking.db import Tracker, utcnow

log = logging.getLogger("kaggle_agent.research")

DENIED_SUBCOMMANDS = ("features approve", "submit approve", "rules set", "autosubmit", "validation reset",
                      "cleanup", "start", "stop")

BRIEF = """You are the autonomous research lead for the Kaggle competition `{slug}` ({title}).
Goal: maximise the local cross-validation {metric} ({direction}). A scheduler daemon runs queued experiments on
the GPU in the background; your job is to give it better ideas than its built-in heuristic.

Run commands exactly like this (Windows; the bare `python` lacks the dependencies):
  {py} main.py <command> -c {slug}
Useful: `status`, `report --print`, `exp leaderboard`, `exp list --limit 30`, `exp show --id N`,
`exp failures`, `exp suggest`, `info --profile`, `data --rows 15`, `data --column COL`, `features list`,
`exp propose --spec '<json>' --hypothesis '<why it should help>'`.

Procedure for this session:
1. Read {notes} (your memory from earlier sessions) and `report --print`.
2. Look at the data (`info --profile`, `data ...`) and the best experiments. Form 1-3 concrete hypotheses,
   preferring domain feature engineering that the built-in recipes ({recipes}) cannot express.
3. For each new feature idea, write `{features_dir}/<name>.py` defining
   `transform(df: pandas.DataFrame) -> pandas.DataFrame`: df is train+test raw columns plus the id column
   `{id_column}` (no target);
   return ONLY new columns, same row count and order. Imports limited to pandas, numpy, math, re, itertools,
   collections, string. No file, network or process access. Must not use the target.
4. Queue experiments that use it, building on the best configuration, e.g.
   --spec '{{"model": "<best model>", "params": {{...best params...}}, "features": ["<recipes>", "custom:<name>"]}}'
   At most 5 experiments per session. Every experiment needs a falsifiable hypothesis.
5. Append to {notes}: date, what you looked at, ideas queued (with experiment ids), what earlier ideas
   showed, and what to try next. Keep it concise.

Rules: never optimise for the public leaderboard; never try to change validation folds, competition rules,
approvals or submissions (those commands are refused for you); do not repeat ideas your notes say failed;
custom code that is not yet approved shows as `blocked` until it is approved — that is expected.
{approval_note}
Finish with a 3-line summary of what you did."""


def research_settings(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {"every_minutes": 45, "min_new_results": 3, "session_timeout": 1800, "max_sessions": 50,
                "model": None, "command": None}
    return {**defaults, **(config.get("research") or {})}


def notes_path(ws: Workspace) -> Path:
    return ws.root / "research_notes.md"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "//" + path.resolve().as_posix().lstrip("/")


def build_brief(ws: Workspace) -> str:
    from experiments.features import RECIPES
    config = ws.load_config()
    auto = (config.get("custom_code") or {}).get("auto_approve")
    return BRIEF.format(
        slug=ws.slug, title=config.get("title", ws.slug), id_column=config.get("id_column"), metric=config["metric"], direction=config["direction"],
        py=_python_cmd(), notes=_rel(notes_path(ws)), features_dir=_rel(ws.custom_features_dir),
        recipes=", ".join(sorted(RECIPES)),
        approval_note=("Feature modules are auto-approved after a static safety screen once you finish the session; "
                       "their experiments then run automatically." if auto else
                       "Feature modules need the human's approval before their experiments run."))


def _python_cmd() -> str:
    exe = Path(sys.executable)
    try:
        return exe.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return exe.as_posix()


def claude_command(ws: Workspace, settings: dict[str, Any]) -> list[str]:
    if settings.get("command"):
        return list(settings["command"])  # tests / alternative CLIs
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("The `claude` CLI was not found on PATH. Install Claude Code first.")
    py = _python_cmd()
    allowed = [f"Bash({py} main.py *)", "Read", "Grep", "Glob",
               f"Write({_rel(ws.custom_features_dir)}/**)", f"Edit({_rel(ws.custom_features_dir)}/**)",
               f"Write({_rel(notes_path(ws))})", f"Edit({_rel(notes_path(ws))})"]
    denied = [f"Bash({py} main.py {sub}*)" for sub in DENIED_SUBCOMMANDS]
    cmd = [exe, "-p", build_brief(ws), "--permission-mode", "dontAsk", "--output-format", "json",
           "--no-session-persistence", "--allowedTools", *allowed, "--disallowedTools", *denied]
    if not str(ws.root.resolve()).startswith(str(PROJECT_ROOT)):
        cmd += ["--add-dir", str(ws.root.resolve())]
    if settings.get("model"):
        cmd += ["--model", settings["model"]]
    return cmd


def auto_approve_new_code(ws: Workspace, tracker: Tracker) -> list[dict[str, Any]]:
    """Opt-in: approve agent-written modules that pass the safety screen, then re-queue their experiments."""
    from experiments import custom_code
    config = ws.load_config()
    if not (config.get("custom_code") or {}).get("auto_approve"):
        return []
    results = []
    for mod in custom_code.list_custom(ws):
        if mod["approved"]:
            continue
        problems = custom_code.screen(ws, mod["name"])
        if problems:
            results.append({"name": mod["name"], "approved": False, "problems": problems})
            tracker.log_event(ws.slug, "code_rejected", f"custom:{mod['name']} failed screen", problems)
            continue
        custom_code.approve(ws, mod["name"])
        results.append({"name": mod["name"], "approved": True})
        tracker.log_event(ws.slug, "code_auto_approved", f"custom:{mod['name']} passed screen")
    approved = {r["name"] for r in results if r["approved"]}
    for exp in tracker.list_experiments(ws.slug, "blocked"):
        if any(f"custom:{n}" in exp["spec"].get("features", []) for n in approved):
            tracker.update_experiment(exp["id"], status="queued", attempts=0, error=None)
    return results


def run_session(ws: Workspace, tracker: Tracker, settings: dict[str, Any], index: int) -> dict[str, Any]:
    import os
    out_dir = ws.logs_dir / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not notes_path(ws).exists():
        notes_path(ws).write_text(f"# Research notes: {ws.slug}\n\n", encoding="utf-8")
    cmd = claude_command(ws, settings)
    env = {**os.environ, "KAGGLE_AGENT_ROLE": "agent", "PYTHONUNBUFFERED": "1"}
    started = time.time()
    tracker.log_event(ws.slug, "research_start", f"session {index}")
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=float(settings["session_timeout"]),
                              stdin=subprocess.DEVNULL)
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        status, stdout, stderr = "timeout", exc.stdout or "", exc.stderr or ""
        stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
    summary = stdout
    cost = None
    try:
        parsed = json.loads(stdout)
        summary, cost = parsed.get("result", stdout), parsed.get("total_cost_usd")
    except (ValueError, AttributeError):
        pass
    record = {"session": index, "status": status, "seconds": round(time.time() - started, 1), "summary": summary,
              "reported_cost_usd": cost, "stderr_tail": (stderr or "")[-2000:], "finished_at": utcnow()}
    (out_dir / f"session_{index:04d}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    record["auto_approved"] = auto_approve_new_code(ws, tracker)
    tracker.log_event(ws.slug, "research_end", f"session {index}: {status}", {"summary": (summary or "")[:1000]})
    return record


def _finished_count(ws: Workspace, tracker: Tracker) -> int:
    return len(tracker.list_experiments(ws.slug, ("succeeded", "failed", "timeout")))


def _ensure_daemon(ws: Workspace) -> None:
    from core.proc import spawn_worker
    from experiments.scheduler import daemon_state
    state = daemon_state(ws)
    if not state or not state.get("alive"):
        spawn_worker(["loop", "-c", ws.slug, "--keep-alive"], ws.logs_dir / "daemon.log")
        log.info("started scheduler daemon")


def supervise(ws: Workspace, tracker: Tracker, *, with_daemon: bool = False, max_sessions: int | None = None,
              every_minutes: float | None = None, once: bool = False) -> dict[str, Any]:
    settings = research_settings(ws.load_config())
    max_sessions = int(max_sessions or settings["max_sessions"])
    every = float(every_minutes if every_minutes is not None else settings["every_minutes"]) * 60
    sessions, last_count, last_time = [], -10**9, 0.0
    stop_path = ws.root / "STOP_RESEARCH"
    stop_path.unlink(missing_ok=True)
    try:
        while len(sessions) < max_sessions:
            if stop_path.exists():
                break
            if with_daemon:
                _ensure_daemon(ws)
            finished = _finished_count(ws, tracker)
            due = time.time() - last_time >= every
            enough = finished - last_count >= int(settings["min_new_results"])
            if once or (due and enough):
                rec = run_session(ws, tracker, settings, len(sessions) + 1)
                sessions.append({k: rec[k] for k in ("session", "status", "seconds", "reported_cost_usd")})
                log.info("research session %s: %s", rec["session"], rec["status"])
                last_count, last_time = finished, time.time()
                if once:
                    break
            time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        stop_path.unlink(missing_ok=True)
    return {"sessions": sessions, "notes": str(notes_path(ws))}
