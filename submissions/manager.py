"""Submission queue: prepare -> (human approval | authorized autonomous mode) -> submit -> score refresh.

Daily usage is counted per UTC day (Kaggle's reset) from both our records and, for real competitions, the
submissions Kaggle reports, so submissions made by hand on the website also count against the budget.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from competitions.kaggle_client import KaggleClient, KaggleError
from competitions.onboarding import client_for
from core.compliance import submission_violations
from core.settings import load_settings
from core.workspace import Workspace
from submissions.builder import SubmissionError, build_submission, validate_submission
from tracking.db import Tracker, utcnow

log = logging.getLogger("kaggle_agent.submissions")

USED_STATUSES = ("submitting", "submitted", "scored")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _tag(sub_id: int) -> str:
    return f"[ka:{sub_id}]"


class SubmissionManager:
    def __init__(self, ws: Workspace, tracker: Tracker, client: KaggleClient | None = None):
        self.ws, self.tracker = ws, tracker
        self.config = ws.load_config()
        self._client = client
        self.settings = load_settings()["submissions"]

    @property
    def client(self) -> KaggleClient:
        if self._client is None:
            self._client = client_for(self.config)
        return self._client

    # Budget
    def usage_today(self, autonomous: bool = False) -> dict[str, Any]:
        today = _today()
        local = [s for s in self.tracker.list_submissions(self.ws.slug, USED_STATUSES)
                 if (s["submitted_at"] or "").startswith(today)]
        remote_count = None
        try:
            remote_count = sum(1 for s in self.client.submissions(self.ws.slug) if (s.get("date") or "").startswith(today))
        except Exception as exc:  # offline or auth problems should not hide the local count
            log.warning("could not list remote submissions: %s", exc)
        used = max(len(local), remote_count or 0)
        kaggle_limit = int(self.config.get("max_daily_submissions") or 5)
        limit = kaggle_limit
        if self.config.get("submission_budget"):
            limit = min(limit, int(self.config["submission_budget"]))
        if autonomous:
            limit = min(limit, int(self.config.get("autosubmit", {}).get("max_daily", 0)))
        return {"date_utc": today, "used": used, "used_by_agent": len(local), "used_reported_by_kaggle": remote_count,
                "limit": limit, "kaggle_limit": kaggle_limit, "remaining": max(0, limit - used)}

    # Duplicate detection
    def _near_duplicate(self, df: pd.DataFrame, sha: str) -> dict | None:
        id_col = self.config["submission"]["id_column"]
        threshold = float(self.settings.get("duplicate_correlation", 0.9995))
        cols = [c for c in df.columns if c != id_col]
        for prev in self.tracker.list_submissions(self.ws.slug, ("pending_approval",) + USED_STATUSES):
            if prev["file_sha256"] == sha:
                return {"submission_id": prev["id"], "reason": "identical file"}
            try:
                other = pd.read_csv(prev["file_path"])
            except (OSError, ValueError):
                continue
            if list(other.columns) != list(df.columns) or len(other) != len(df):
                continue
            if self.config["submission"]["format"] == "label":
                agree = float((other[cols[0]].astype(str).to_numpy() == df[cols[0]].astype(str).to_numpy()).mean())
                if agree >= threshold:
                    return {"submission_id": prev["id"], "reason": f"{agree:.4%} identical labels"}
            else:
                a, b = other[cols].to_numpy(dtype=float).ravel(), df[cols].to_numpy(dtype=float).ravel()
                if a.std() > 0 and b.std() > 0:
                    corr = float(np.corrcoef(a, b)[0, 1])
                    if corr >= threshold:
                        return {"submission_id": prev["id"], "reason": f"prediction correlation {corr:.6f}"}
        return None

    # Workflow
    def prepare(self, exp_id: int, message: str = "", force: bool = False, autonomous: bool = False) -> dict[str, Any]:
        exp = self.tracker.get_experiment(exp_id)
        if not exp or exp["competition_id"] != self.ws.slug or exp["status"] != "succeeded":
            raise SubmissionError(f"Experiment {exp_id} is not a succeeded experiment of {self.ws.slug}")
        exp_dir = self.ws.experiment_dir(exp_id)
        pred = np.load(exp_dir / "test.npy")
        threshold = json.loads((exp_dir / "result.json").read_text()).get("threshold", 0.5)
        df = build_submission(self.ws, self.config, pred, threshold)
        errors = validate_submission(df, self.ws, self.config)
        if errors:
            raise SubmissionError("Submission failed validation: " + "; ".join(errors))
        self.ws.submissions_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = self.ws.submissions_dir / f"sub_exp{exp_id:05d}_{stamp}.csv"
        df.to_csv(path, index=False)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        dup = self._near_duplicate(df, sha)
        if dup and not force:
            path.unlink()
            raise SubmissionError(f"Nearly identical to submission {dup['submission_id']} ({dup['reason']}); "
                                  "not queued. Use --force to override.")
        cv = exp["cv_score"]
        text = message or f"exp {exp_id} {exp['model_family']} cv={cv:.6f}"
        sub_id = self.tracker.create_submission(
            self.ws.slug, experiment_id=exp_id, file_path=str(path), file_sha256=sha, status="pending_approval",
            message=text, cv_score=cv, autonomous=int(autonomous))
        self.tracker.log_event(self.ws.slug, "submission_prepared", f"submission {sub_id} from exp {exp_id}")
        return {"submission_id": sub_id, "file": str(path), "rows": len(df), "status": "pending_approval",
                "cv_score": cv, "message": text,
                "next_step": f"python main.py submit approve --id {sub_id}  (or: submit reject --id {sub_id})"}

    def approve(self, sub_id: int, approver: str = "user", autonomous: bool = False) -> dict[str, Any]:
        sub = self.tracker.get_submission(sub_id)
        if not sub or sub["competition_id"] != self.ws.slug:
            raise SubmissionError(f"Submission {sub_id} not found for {self.ws.slug}")
        if sub["status"] != "pending_approval":
            raise SubmissionError(f"Submission {sub_id} is {sub['status']}, not pending approval")
        problems = submission_violations(self.config, autonomous)
        if problems:
            raise SubmissionError("Submission not allowed: " + "; ".join(problems))
        usage = self.usage_today(autonomous)
        if usage["remaining"] <= 0:
            raise SubmissionError(f"Daily submission budget exhausted ({usage['used']}/{usage['limit']} used today, "
                                  "resets 00:00 UTC). Submission stays pending.")
        with open(sub["file_path"], "rb") as fh:
            if hashlib.sha256(fh.read()).hexdigest() != sub["file_sha256"]:
                raise SubmissionError("Submission file changed after it was prepared; prepare it again")
        self.tracker.update_submission(sub_id, status="submitting", approved_by=approver, submitted_at=utcnow())
        try:
            resp = self.client.submit(self.ws.slug, sub["file_path"], f"{sub['message']} {_tag(sub_id)}")
        except KaggleError as exc:
            self.tracker.update_submission(sub_id, status="error", error=str(exc))
            self.tracker.log_event(self.ws.slug, "submission_error", f"submission {sub_id}: {exc}")
            raise SubmissionError(f"Kaggle rejected the submission: {exc}") from exc
        self.tracker.update_submission(sub_id, status="submitted", kaggle_ref=resp.get("ref"))
        self.tracker.log_event(self.ws.slug, "submitted", f"submission {sub_id} by {approver}"
                               + (" (SIMULATED)" if self.client.simulated else ""))
        out = {"submission_id": sub_id, "status": "submitted", "kaggle_ref": resp.get("ref"),
               "simulated": self.client.simulated, "kaggle_message": resp.get("message")}
        out.update(self.refresh().get(str(sub_id), {}))
        return out

    def reject(self, sub_id: int, reason: str = "") -> dict[str, Any]:
        sub = self.tracker.get_submission(sub_id)
        if not sub or sub["status"] != "pending_approval":
            raise SubmissionError(f"Submission {sub_id} is not pending approval")
        self.tracker.update_submission(sub_id, status="rejected", error=reason or None)
        return {"submission_id": sub_id, "status": "rejected"}

    def refresh(self) -> dict[str, Any]:
        pending = self.tracker.list_submissions(self.ws.slug, "submitted")
        if not pending:
            return {}
        remote = self.client.submissions(self.ws.slug)
        updates = {}
        for sub in pending:
            match = next((r for r in remote if _tag(sub["id"]) in (r.get("description") or "")
                          or (sub["kaggle_ref"] and r.get("ref") == sub["kaggle_ref"])), None)
            if not match:
                continue
            status = (match.get("status") or "").lower()
            if status == "complete":
                self.tracker.update_submission(sub["id"], status="scored", public_score=match.get("public_score"),
                                               private_score=match.get("private_score"), scored_at=utcnow())
                updates[str(sub["id"])] = {"status": "scored", "public_score": match.get("public_score")}
            elif status == "error":
                self.tracker.update_submission(sub["id"], status="error", error=match.get("error"))
                updates[str(sub["id"])] = {"status": "error", "error": match.get("error")}
        return updates

    def maybe_autosubmit(self, exp: dict[str, Any], metric) -> dict[str, Any] | None:
        """Called by the daemon when a new best appears. Only acts in an explicitly authorized mode."""
        auto = self.config.get("autosubmit", {})
        if submission_violations(self.config, autonomous=True):
            return None
        if self.usage_today(autonomous=True)["remaining"] <= 0:
            return None
        submitted = [s for s in self.tracker.list_submissions(self.ws.slug, USED_STATUSES) if s["cv_score"] is not None]
        if submitted:
            best_cv = max(s["cv_score"] for s in submitted) if metric.greater_is_better else min(
                s["cv_score"] for s in submitted)
            min_gain = float(auto.get("min_improvement") or load_settings()["loop"]["min_improvement"])
            if not metric.better(exp["cv_score"], best_cv, min_gain):
                return None
        try:
            prepared = self.prepare(exp["id"], autonomous=True)
            return self.approve(prepared["submission_id"], approver="autonomous", autonomous=True)
        except SubmissionError as exc:
            log.info("autosubmit skipped: %s", exc)
            return {"skipped": str(exc)}
