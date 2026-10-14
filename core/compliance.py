"""Competition rule gating. The Kaggle API does not expose rule text, so anything it cannot confirm starts as
'unknown' and actions that depend on it are blocked until the user records a decision with `rules set`.
Checks run at queue time, again in the worker before training, and before every submission."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

PERMISSIONS = ("external_data", "pretrained_models", "automated_submissions")
VALUES = ("allowed", "forbidden", "unknown")


class ComplianceError(RuntimeError):
    pass


def default_rules(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "external_data": "unknown",
        "pretrained_models": "unknown",
        "automated_submissions": "unknown",
        "code_competition": bool(meta.get("is_kernels_submissions_only")),
        "submissions_disabled": bool(meta.get("submissions_disabled")),
        "joined_competition": meta.get("user_has_entered"),
        "max_team_size": meta.get("max_team_size"),
        "source": "kaggle_api" if not meta.get("simulated") else "local_data",
        "confirmed_by_user": False,
        "notes": "Kaggle's API does not expose competition rules. Read the rules page and record decisions "
                 "with: python main.py rules set --competition <slug> --external-data allowed|forbidden ...",
    }


def spec_violations(spec: dict[str, Any], config: dict[str, Any]) -> list[str]:
    rules = config.get("rules", {})
    problems = []
    for need in spec.get("requires", []):
        if need not in PERMISSIONS:
            problems.append(f"Unknown requirement '{need}'")
            continue
        status = rules.get(need, "unknown")
        if status == "forbidden":
            problems.append(f"Experiment needs {need}, which this competition forbids")
        elif status != "allowed":
            problems.append(f"Experiment needs {need}, but compliance is unconfirmed (status: {status}). "
                            f"Check the rules, then: python main.py rules set --{need.replace('_', '-')} allowed")
    return problems


def check_spec(spec: dict[str, Any], config: dict[str, Any]) -> None:
    problems = spec_violations(spec, config)
    if problems:
        raise ComplianceError("; ".join(problems))


def submission_violations(config: dict[str, Any], autonomous: bool) -> list[str]:
    rules = config.get("rules", {})
    problems = []
    if rules.get("submissions_disabled"):
        problems.append("Submissions are disabled for this competition")
    if rules.get("code_competition"):
        problems.append("This is a code competition: predictions must be produced by a Kaggle Notebook, "
                        "so file submission through the API is not possible")
    if rules.get("joined_competition") is False:
        problems.append("You have not joined this competition / accepted its rules on kaggle.com")
    deadline = config.get("deadline")
    if autonomous and deadline:
        try:
            ends = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
            if ends.tzinfo is None:
                ends = ends.replace(tzinfo=timezone.utc)
            if ends < datetime.now(timezone.utc):
                problems.append(f"Deadline {deadline} has passed; late submissions require manual approval")
        except ValueError:
            pass
    if autonomous:
        auto = config.get("autosubmit", {})
        if not auto.get("enabled"):
            problems.append("Autonomous submission is not enabled for this competition")
        if rules.get("automated_submissions") != "allowed":
            problems.append("Automated submissions are not confirmed as allowed (rules set --automated-submissions)")
        if int(auto.get("max_daily", 0)) <= 0:
            problems.append("Autonomous daily submission budget is zero")
    return problems
