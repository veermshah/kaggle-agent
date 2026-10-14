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
                    "requires": declared_requirements(ws, p.stem), "screen_problems": screen(ws, p.stem)})
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


ALLOWED_IMPORTS = {"pandas", "numpy", "math", "re", "itertools", "collections", "string", "functools", "statistics"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals",
                   "getattr", "setattr", "delattr", "vars"}
FORBIDDEN_ATTRS = {"read_csv", "read_parquet", "read_pickle", "read_json", "read_excel", "read_sql", "to_csv",
                   "to_parquet", "to_pickle", "to_json", "to_sql", "system", "popen", "remove", "unlink", "rmtree"}


def screen(ws: Workspace, name: str) -> list[str]:
    return screen_path(_path(ws, name), "transform")


def screen_path(path, entry: str = "transform") -> list[str]:
    """Static safety screen for agent-written code. Defence in depth: code also only ever runs in a separate,
    time-limited process without secrets in its environment. Not a sandbox; it rejects the obvious ways to
    touch files, network, processes or dynamic code."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            mods = []
        problems += [f"import of {m!r} not allowed" for m in mods if m not in ALLOWED_IMPORTS]
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"use of {node.id!r} not allowed")
        if isinstance(node, ast.Attribute) and (node.attr in FORBIDDEN_ATTRS or node.attr.startswith("__")):
            problems.append(f"attribute {node.attr!r} not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == entry for n in tree.body):
        problems.append(f"no top-level {entry}() function")
    return sorted(set(problems))


def auto_approve_enabled(ws: Workspace) -> bool:
    """Competition setting `custom_code.auto_approve`, else the global default in config/*.yaml."""
    from core.settings import load_settings
    local = (ws.load_config().get("custom_code") or {}).get("auto_approve")
    if local is not None:
        return bool(local)
    return bool((load_settings().get("custom_code") or {}).get("auto_approve", False))


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
