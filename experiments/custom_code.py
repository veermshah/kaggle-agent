"""Approval-gated custom feature code.

An agent (or you) may write `workspaces/<slug>/custom_features/<name>.py` defining
`transform(df: pd.DataFrame) -> pd.DataFrame` (new columns, same row count/order) and optionally
`REQUIRES = ["external_data"]`. The file only runs after a human approves its exact SHA-256 with
`python main.py features approve --name <name>`; any later edit invalidates the approval. It always runs
inside the time-limited worker subprocess, never in the daemon.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from typing import Any, Callable

from core.workspace import Workspace


class CustomCodeError(RuntimeError):
    pass


def _path(ws: Workspace, name: str):
    if not name.replace("_", "").isalnum():
        raise CustomCodeError(f"Invalid custom feature name {name!r}")
    return ws.custom_features_dir / f"{name}.py"


def _registry(ws: Workspace) -> dict[str, str]:
    path = ws.custom_features_dir / "approved.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def list_custom(ws: Workspace) -> list[dict[str, Any]]:
    reg = _registry(ws)
    out = []
    for p in sorted(ws.custom_features_dir.glob("*.py")):
        sha = _sha(p)
        out.append({"name": p.stem, "sha256": sha, "approved": reg.get(p.stem) == sha,
                    "requires": declared_requirements(ws, p.stem)})
    return out


def declared_requirements(ws: Workspace, name: str) -> list[str]:
    """Read REQUIRES statically (no execution) so compliance can be checked before approval/running."""
    tree = ast.parse(_path(ws, name).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "REQUIRES" for t in node.targets):
            try:
                return list(ast.literal_eval(node.value))
            except ValueError:
                raise CustomCodeError(f"{name}.py: REQUIRES must be a literal list of strings") from None
    return []


def approve(ws: Workspace, name: str) -> dict[str, Any]:
    path = _path(ws, name)
    if not path.exists():
        raise CustomCodeError(f"{path} does not exist")
    ast.parse(path.read_text(encoding="utf-8"))  # refuse to approve code that does not even parse
    reg = _registry(ws)
    reg[name] = _sha(path)
    (ws.custom_features_dir / "approved.json").write_text(json.dumps(reg, indent=2))
    return {"name": name, "sha256": reg[name], "requires": declared_requirements(ws, name)}


def loader(ws: Workspace) -> Callable[[str], Callable]:
    def load(name: str) -> Callable:
        path = _path(ws, name)
        if not path.exists():
            raise CustomCodeError(f"Custom feature module {path} not found")
        if _registry(ws).get(name) != _sha(path):
            raise CustomCodeError(f"custom:{name} is not approved or changed since approval. Review it, then run "
                                  f"`python main.py features approve --name {name}`")
        spec = importlib.util.spec_from_file_location(f"custom_features_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "transform", None)):
            raise CustomCodeError(f"{path} must define transform(df) -> DataFrame")
        return module.transform
    return load
