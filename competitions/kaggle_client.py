"""Kaggle access behind a small interface so onboarding and submission logic can be tested offline.

KaggleApiClient wraps the official `kaggle` package. LocalDirectoryClient serves a competition from a
local folder (mock data, pre-downloaded data, tests) and simulates submissions without touching Kaggle.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class KaggleError(RuntimeError):
    pass


class KaggleAuthError(KaggleError):
    pass


class KaggleClient(Protocol):
    simulated: bool

    def competition_metadata(self, slug: str) -> dict[str, Any]: ...
    def list_files(self, slug: str) -> list[dict[str, Any]]: ...
    def download(self, slug: str, dest: Path) -> list[Path]: ...
    def submit(self, slug: str, file_path: Path, message: str) -> dict[str, Any]: ...
    def submissions(self, slug: str) -> list[dict[str, Any]]: ...


def safe_extract(archive: Path, dest: Path) -> list[Path]:
    """Extract a zip refusing absolute paths and '..' traversal. Nothing extracted is ever executed."""
    dest = dest.resolve()
    out = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                raise KaggleError(f"Refusing unsafe path in archive: {member.filename}")
            if member.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", str(value))
    return name.split("_")[-1].lower() if "_" in name else name.lower()


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


class KaggleApiClient:
    """Official Kaggle API. Credentials come from KAGGLE_API_TOKEN, ~/.kaggle/access_token or kaggle.json."""

    simulated = False

    def __init__(self) -> None:
        self._api = None

    @property
    def api(self):
        if self._api is None:
            # `import kaggle` authenticates at import time and prints to stdout; keep our JSON output clean.
            with contextlib.redirect_stdout(sys.stderr):
                try:
                    from kaggle.api.kaggle_api_extended import KaggleApi
                except ImportError as exc:
                    raise KaggleError("The 'kaggle' package is not installed: pip install kaggle") from exc
                api = KaggleApi()
                try:
                    api.authenticate()
                except (Exception, SystemExit) as exc:
                    raise KaggleAuthError(
                        "Kaggle credentials not found. Run `kaggle auth login`, set KAGGLE_API_TOKEN, "
                        "or place an API token in ~/.kaggle/access_token (see README)."
                    ) from exc
            self._api = api
        return self._api

    def _call(self, fn, *args, **kwargs):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                return fn(*args, **kwargs)
        except KaggleError:
            raise
        except Exception as exc:  # the SDK raises a mix of HTTP and value errors
            text = str(exc)
            if "403" in text or "Forbidden" in text:
                raise KaggleError(
                    "Kaggle returned 403. You must join the competition and accept its rules on kaggle.com "
                    "before the API allows downloads or submissions."
                ) from exc
            if "401" in text:
                raise KaggleAuthError("Kaggle rejected the credentials (401). Regenerate your API token.") from exc
            raise KaggleError(f"Kaggle API call failed: {text}") from exc

    def competition_metadata(self, slug: str) -> dict[str, Any]:
        resp = self._call(self.api.competitions_list, search=slug, page_size=50)
        comps = getattr(resp, "competitions", None) or []
        match = None
        for comp in comps:
            ref = str(getattr(comp, "ref", "") or "")
            if ref.rstrip("/").split("/")[-1] == slug:
                match = comp
                break
        if match is None:
            raise KaggleError(f"Competition '{slug}' not found via the Kaggle API search.")
        return {
            "slug": slug,
            "title": getattr(match, "title", slug),
            "url": getattr(match, "url", None),
            "description": getattr(match, "description", None),
            "category": getattr(match, "category", None),
            "deadline": _iso(getattr(match, "deadline", None)),
            "evaluation_metric": getattr(match, "evaluation_metric", None),
            "max_daily_submissions": getattr(match, "max_daily_submissions", None),
            "max_team_size": getattr(match, "max_team_size", None),
            "user_has_entered": getattr(match, "user_has_entered", None),
            "is_kernels_submissions_only": getattr(match, "is_kernels_submissions_only", None),
            "submissions_disabled": getattr(match, "submissions_disabled", None),
            "reward": getattr(match, "reward", None),
            "tags": [getattr(t, "name", str(t)) for t in (getattr(match, "tags", None) or [])],
        }

    def list_files(self, slug: str) -> list[dict[str, Any]]:
        resp = self._call(self.api.competition_list_files, slug, page_size=200)
        files = getattr(resp, "files", None) or []
        return [{"name": f.name, "size": getattr(f, "total_bytes", None)} for f in files]

    def download(self, slug: str, dest: Path) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        self._call(self.api.competition_download_files, slug, path=str(dest), quiet=True)
        extracted: list[Path] = []
        for archive in dest.glob("*.zip"):
            extracted += safe_extract(archive, dest)
            archive.unlink()
        return extracted or [p for p in dest.iterdir() if p.is_file()]

    def submit(self, slug: str, file_path: Path, message: str) -> dict[str, Any]:
        resp = self._call(self.api.competition_submit, str(file_path), message, slug, quiet=True)
        return {"ref": str(getattr(resp, "ref", "") or ""), "message": getattr(resp, "message", None)}

    def submissions(self, slug: str) -> list[dict[str, Any]]:
        subs = self._call(self.api.competition_submissions, slug, page_size=100) or []
        out = []
        for s in subs:
            if s is None:
                continue
            out.append({
                "ref": str(getattr(s, "ref", "")),
                "date": _iso(getattr(s, "date", None)),
                "description": getattr(s, "description", None),
                "status": _enum_name(getattr(s, "status", None)),
                "public_score": _to_float(getattr(s, "public_score", None)),
                "private_score": _to_float(getattr(s, "private_score", None)),
                "error": getattr(s, "error_description", None),
                "file_name": getattr(s, "file_name", None),
            })
        return out


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


class LocalDirectoryClient:
    """Serves competition files from a local directory; submissions are simulated and never leave the machine.

    Optional `metadata.json` in the directory mimics Kaggle competition metadata. If it contains
    `solution_file` (a CSV with the true test targets), simulated submissions receive a score.
    """

    simulated = True

    def __init__(self, source_dir: Path):
        self.source_dir = Path(source_dir)
        if not self.source_dir.is_dir():
            raise KaggleError(f"Data directory not found: {self.source_dir}")
        self._log = self.source_dir / "_simulated_submissions.json"

    def _meta(self) -> dict[str, Any]:
        path = self.source_dir / "metadata.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def competition_metadata(self, slug: str) -> dict[str, Any]:
        meta = {"slug": slug, "title": slug, "evaluation_metric": None, "max_daily_submissions": 5,
                "deadline": None, "is_kernels_submissions_only": False, "submissions_disabled": False}
        meta.update({k: v for k, v in self._meta().items() if k != "solution_file"})
        return meta

    def list_files(self, slug: str) -> list[dict[str, Any]]:
        return [{"name": p.name, "size": p.stat().st_size} for p in sorted(self.source_dir.iterdir())
                if p.is_file() and not p.name.startswith("_") and p.name != "metadata.json"]

    def download(self, slug: str, dest: Path) -> list[Path]:
        dest.mkdir(parents=True, exist_ok=True)
        out = []
        solution = self._meta().get("solution_file")
        for f in self.list_files(slug):
            if f["name"] == solution:
                continue  # the answer key never enters the workspace
            src = self.source_dir / f["name"]
            if src.suffix == ".zip":
                out += safe_extract(src, dest)
            else:
                shutil.copy2(src, dest / f["name"])
                out.append(dest / f["name"])
        return out

    def _read_log(self) -> list[dict[str, Any]]:
        return json.loads(self._log.read_text()) if self._log.exists() else []

    def submit(self, slug: str, file_path: Path, message: str) -> dict[str, Any]:
        log = self._read_log()
        digest = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:12]
        ref = f"sim-{len(log) + 1}-{digest}"
        score = self._score(Path(file_path))
        log.append({"ref": ref, "date": datetime.now(timezone.utc).isoformat(), "description": message,
                    "status": "complete", "public_score": score, "private_score": None, "error": None,
                    "file_name": Path(file_path).name})
        self._log.write_text(json.dumps(log, indent=2))
        return {"ref": ref, "message": "Simulated submission (local data source, not sent to Kaggle)"}

    def submissions(self, slug: str) -> list[dict[str, Any]]:
        return list(reversed(self._read_log()))

    def _score(self, file_path: Path) -> float | None:
        meta = self._meta()
        solution = meta.get("solution_file")
        if not solution or not (self.source_dir / solution).exists():
            return None
        import pandas as pd
        from evaluation.metrics import get_metric, score_submission_frame
        sub = pd.read_csv(file_path)
        sol = pd.read_csv(self.source_dir / solution)
        try:
            return score_submission_frame(sub, sol, get_metric(meta.get("metric", "roc_auc")))
        except Exception:
            return None


def make_client(data_dir: str | Path | None = None) -> KaggleClient:
    return LocalDirectoryClient(Path(data_dir)) if data_dir else KaggleApiClient()
