"""SQLite experiment store. Shared by the daemon, workers, CLI and dashboard (WAL mode)."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from core.settings import database_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS competitions (
    id TEXT PRIMARY KEY,
    title TEXT,
    task_type TEXT,
    metric TEXT,
    direction TEXT,
    status TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    kind TEXT NOT NULL,                 -- train | tune | ensemble
    status TEXT NOT NULL,               -- queued | running | succeeded | failed | timeout | cancelled | blocked
    hypothesis TEXT NOT NULL,
    arm TEXT,                           -- research arm (strategy bookkeeping)
    proposed_by TEXT,
    parent_id INTEGER,
    model_family TEXT,
    feature_set TEXT,
    spec_json TEXT NOT NULL,
    params_json TEXT,                   -- final params actually used (after tuning)
    resource TEXT DEFAULT 'cpu',        -- cpu | gpu
    priority REAL DEFAULT 0,
    attempts INTEGER DEFAULT 0,
    pid INTEGER,
    validation_version TEXT,
    dataset_version TEXT,
    code_version TEXT,
    cv_score REAL,
    cv_std REAL,
    fold_scores_json TEXT,
    holdout_score REAL,                 -- hidden from selection; revealed on promote
    train_seconds REAL,
    peak_gpu_mb REAL,
    peak_rss_mb REAL,
    artifact_dir TEXT,
    error TEXT,
    promoted INTEGER DEFAULT 0,
    created_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    heartbeat_at REAL
);
CREATE INDEX IF NOT EXISTS ix_exp_comp_status ON experiments(competition_id, status);
CREATE INDEX IF NOT EXISTS ix_exp_fingerprint ON experiments(competition_id, fingerprint);

CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id TEXT NOT NULL,
    experiment_id INTEGER,
    file_path TEXT NOT NULL,
    file_sha256 TEXT,
    status TEXT NOT NULL,               -- pending_approval | rejected | submitting | submitted | scored | error
    message TEXT,
    cv_score REAL,
    public_score REAL,
    private_score REAL,
    kaggle_ref TEXT,
    autonomous INTEGER DEFAULT 0,
    approved_by TEXT,
    error TEXT,
    created_at TEXT,
    submitted_at TEXT,
    scored_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_sub_comp ON submissions(competition_id, status);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    competition_id TEXT,
    ts TEXT,
    kind TEXT,
    message TEXT,
    data_json TEXT
);
"""

TERMINAL = ("succeeded", "failed", "timeout", "cancelled", "blocked")
JSON_FIELDS = ("spec_json", "params_json", "fold_scores_json", "data_json")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for key in JSON_FIELDS:
        if key in out and out[key]:
            out[key[: -len("_json")]] = json.loads(out.pop(key))
        elif key in out:
            out[key[: -len("_json")]] = None
            out.pop(key)
    return out


class Tracker:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=60000")
        try:
            yield con
        finally:
            con.close()

    # Competitions
    def upsert_competition(self, slug: str, **fields: Any) -> None:
        now = utcnow()
        with self.conn() as c:
            exists = c.execute("SELECT 1 FROM competitions WHERE id=?", (slug,)).fetchone()
            if exists:
                if fields:
                    sets = ", ".join(f"{k}=?" for k in fields)
                    c.execute(f"UPDATE competitions SET {sets}, updated_at=? WHERE id=?", (*fields.values(), now, slug))
            else:
                cols = ["id", "created_at", "updated_at", *fields]
                c.execute(
                    f"INSERT INTO competitions ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    (slug, now, now, *fields.values()),
                )

    def get_competition(self, slug: str) -> dict | None:
        with self.conn() as c:
            return _row_to_dict(c.execute("SELECT * FROM competitions WHERE id=?", (slug,)).fetchone())

    # Experiments
    def find_by_fingerprint(self, competition: str, fingerprint: str, validation_version: str) -> dict | None:
        with self.conn() as c:
            row = c.execute(
                "SELECT * FROM experiments WHERE competition_id=? AND fingerprint=? AND validation_version=? "
                "AND status NOT IN ('cancelled') ORDER BY id DESC LIMIT 1",
                (competition, fingerprint, validation_version),
            ).fetchone()
            return _row_to_dict(row)

    def create_experiment(self, competition: str, spec: dict, *, fingerprint: str, hypothesis: str,
                          validation_version: str, status: str = "queued", **fields: Any) -> int:
        record = {
            "competition_id": competition,
            "fingerprint": fingerprint,
            "kind": spec.get("kind", "train"),
            "status": status,
            "hypothesis": hypothesis,
            "model_family": spec.get("model"),
            "feature_set": ",".join(spec.get("features", [])) or "base",
            "spec_json": json.dumps(spec, sort_keys=True),
            "validation_version": validation_version,
            "created_at": utcnow(),
            **fields,
        }
        cols = ", ".join(record)
        with self.conn() as c:
            cur = c.execute(f"INSERT INTO experiments ({cols}) VALUES ({', '.join('?' * len(record))})",
                            tuple(record.values()))
            return int(cur.lastrowid)

    def get_experiment(self, exp_id: int) -> dict | None:
        with self.conn() as c:
            return _row_to_dict(c.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone())

    def list_experiments(self, competition: str, status: str | tuple[str, ...] | None = None,
                         validation_version: str | None = None, limit: int | None = None) -> list[dict]:
        sql, args = "SELECT * FROM experiments WHERE competition_id=?", [competition]
        if status:
            statuses = (status,) if isinstance(status, str) else status
            sql += f" AND status IN ({', '.join('?' * len(statuses))})"
            args += list(statuses)
        if validation_version:
            sql += " AND validation_version=?"
            args.append(validation_version)
        sql += " ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self.conn() as c:
            return [_row_to_dict(r) for r in c.execute(sql, args).fetchall()]

    def update_experiment(self, exp_id: int, **fields: Any) -> None:
        for key in ("params", "fold_scores"):
            if key in fields:
                fields[f"{key}_json"] = json.dumps(fields.pop(key), default=float)
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE experiments SET {sets} WHERE id=?", (*fields.values(), exp_id))

    def claim_next(self, competition: str, resource: str) -> dict | None:
        """Atomically move the highest-priority queued experiment for a resource to running."""
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                row = c.execute(
                    "SELECT id FROM experiments WHERE competition_id=? AND status='queued' AND resource=? "
                    "ORDER BY priority DESC, id ASC LIMIT 1",
                    (competition, resource),
                ).fetchone()
                if row is None:
                    c.execute("COMMIT")
                    return None
                c.execute(
                    "UPDATE experiments SET status='running', attempts=attempts+1, started_at=?, heartbeat_at=?, "
                    "error=NULL WHERE id=?",
                    (utcnow(), time.time(), row["id"]),
                )
                c.execute("COMMIT")
            except Exception:
                c.execute("ROLLBACK")
                raise
        return self.get_experiment(row["id"])

    def best_experiment(self, competition: str, validation_version: str, direction: str,
                        kind: str | None = None) -> dict | None:
        order = "DESC" if direction == "maximize" else "ASC"
        sql = ("SELECT * FROM experiments WHERE competition_id=? AND validation_version=? AND status='succeeded' "
               "AND cv_score IS NOT NULL")
        args: list[Any] = [competition, validation_version]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        sql += f" ORDER BY cv_score {order}, id ASC LIMIT 1"
        with self.conn() as c:
            return _row_to_dict(c.execute(sql, args).fetchone())

    # Submissions
    def create_submission(self, competition: str, **fields: Any) -> int:
        record = {"competition_id": competition, "created_at": utcnow(), **fields}
        with self.conn() as c:
            cur = c.execute(
                f"INSERT INTO submissions ({', '.join(record)}) VALUES ({', '.join('?' * len(record))})",
                tuple(record.values()),
            )
            return int(cur.lastrowid)

    def get_submission(self, sub_id: int) -> dict | None:
        with self.conn() as c:
            return _row_to_dict(c.execute("SELECT * FROM submissions WHERE id=?", (sub_id,)).fetchone())

    def list_submissions(self, competition: str, status: str | tuple[str, ...] | None = None) -> list[dict]:
        sql, args = "SELECT * FROM submissions WHERE competition_id=?", [competition]
        if status:
            statuses = (status,) if isinstance(status, str) else status
            sql += f" AND status IN ({', '.join('?' * len(statuses))})"
            args += list(statuses)
        with self.conn() as c:
            return [_row_to_dict(r) for r in c.execute(sql + " ORDER BY id", args).fetchall()]

    def update_submission(self, sub_id: int, **fields: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE submissions SET {sets} WHERE id=?", (*fields.values(), sub_id))

    # Events
    def log_event(self, competition: str | None, kind: str, message: str, data: Any = None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO events (competition_id, ts, kind, message, data_json) VALUES (?, ?, ?, ?, ?)",
                (competition, utcnow(), kind, message, json.dumps(data, default=str) if data is not None else None),
            )

    def list_events(self, competition: str, limit: int = 100, kind: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM events WHERE competition_id=?", [competition]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        with self.conn() as c:
            rows = c.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
            return [_row_to_dict(r) for r in rows]
