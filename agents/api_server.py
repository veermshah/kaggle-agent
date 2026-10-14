"""HTTP JSON API so a remote agent (Devin, a cloud Claude Code session, an MCP bridge) can drive the
experiment loop on this GPU machine without the GPU being local to the agent.

Every route maps onto a CLI command, so behaviour and safety checks are identical. Requires a bearer token
(KAGGLE_AGENT_API_TOKEN). There is deliberately no route to approve submissions or code: those stay with a
human at this machine. Binds to 127.0.0.1 by default; expose it through an SSH tunnel or similar, not
directly to the internet.

Routes (all return the CLI's JSON envelope):
  GET  /health
  GET  /status | /info | /leaderboard | /experiments | /experiments/<id> | /failures | /suggest | /report
  GET  /events | /submissions | /submissions/usage
  POST /experiments              {"spec": {...}, "hypothesis": "...", "arm": "...", "priority": 1}
  POST /experiments/<id>/cancel | /experiments/<id>/requeue
  POST /compare                  {"ids": [1, 2]}
  POST /promote                  {"id": 3}
  POST /submissions/prepare      {"exp_id": 3, "message": "..."}
  POST /loop/start               {"max_hours": 12, "max_experiments": 50}   (spawns a detached daemon)
  POST /loop/stop                {"now": false}
Use ?competition=<slug> on any route to target a non-active competition.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MAX_BODY = 1 << 20


def _argv_for(method: str, path: str, body: dict) -> list[str] | None:
    def rid(pattern: str) -> int | None:
        m = re.fullmatch(pattern, path)
        return int(m.group(1)) if m else None

    if method == "GET":
        simple = {"/status": ["status"], "/info": ["info"], "/leaderboard": ["exp", "leaderboard"],
                  "/experiments": ["exp", "list"], "/failures": ["exp", "failures"], "/suggest": ["exp", "suggest"],
                  "/report": ["report", "--print"], "/events": ["events"], "/submissions": ["submit", "list"],
                  "/submissions/usage": ["submit", "usage"]}
        if path in simple:
            return simple[path]
        if (i := rid(r"/experiments/(\d+)")) is not None:
            return ["exp", "show", "--id", str(i)]
        if (i := rid(r"/experiments/(\d+)/logs")) is not None:
            return ["exp", "logs", "--id", str(i)]
        return None
    if path == "/experiments":
        return ["exp", "propose", "--spec", json.dumps(body.get("spec", {})),
                "--hypothesis", str(body.get("hypothesis", "")), "--arm", str(body.get("arm", "remote")),
                "--proposed-by", str(body.get("proposed_by", "remote-agent")),
                "--priority", str(float(body.get("priority", 1.0)))]
    if (i := rid(r"/experiments/(\d+)/cancel")) is not None:
        return ["exp", "cancel", "--id", str(i)]
    if (i := rid(r"/experiments/(\d+)/requeue")) is not None:
        return ["exp", "requeue", "--id", str(i)]
    if path == "/compare":
        return ["exp", "compare", "--ids", *[str(int(x)) for x in body.get("ids", [])]]
    if path == "/promote":
        return ["promote", "--id", str(int(body["id"]))]
    if path == "/submissions/prepare":
        argv = ["submit", "prepare", "--exp-id", str(int(body["exp_id"]))]
        return argv + (["--message", str(body["message"])] if body.get("message") else [])
    if path == "/loop/stop":
        return ["stop"] + (["--now"] if body.get("now") else [])
    return None


def _start_loop(body: dict, competition: str | None) -> dict:
    from core.proc import spawn_worker
    from core.workspace import resolve_workspace
    ws = resolve_workspace(competition)
    argv = ["loop", "-c", ws.slug]
    if body.get("max_hours"):
        argv += ["--max-hours", str(float(body["max_hours"]))]
    if body.get("max_experiments"):
        argv += ["--max-experiments", str(int(body["max_experiments"]))]
    proc = spawn_worker(argv, ws.logs_dir / "daemon.log")
    return {"ok": True, "command": "loop", "result": {"pid": proc.pid, "log": str(ws.logs_dir / "daemon.log")}}


class Handler(BaseHTTPRequestHandler):
    token: str = ""

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        return header.startswith("Bearer ") and hmac.compare_digest(header[7:], self.token)

    def _handle(self, method: str) -> None:
        url = urlparse(self.path)
        path = url.path.rstrip("/") or "/"
        if path == "/health":
            return self._send(200, {"ok": True})
        if not self._authorized():
            return self._send(401, {"ok": False, "error": "missing or invalid bearer token"})
        body = {}
        if method == "POST":
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, {"ok": False, "error": "body too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._send(400, {"ok": False, "error": "invalid JSON body"})
        competition = (parse_qs(url.query).get("competition") or [None])[0]
        try:
            if method == "POST" and path == "/loop/start":
                return self._send(200, _start_loop(body, competition))
            argv = _argv_for(method, path, body)
        except (KeyError, ValueError, TypeError) as exc:
            return self._send(400, {"ok": False, "error": f"bad request: {exc}"})
        if argv is None:
            return self._send(404, {"ok": False, "error": f"no route {method} {path}"})
        if competition:
            argv += ["-c", competition]
        import main
        code, payload = main.run(argv)
        self._send(200 if code == 0 else 400, payload)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def log_message(self, fmt, *args):
        sys.stderr.write("api: " + fmt % args + "\n")


def make_server(host: str, port: int, token: str) -> ThreadingHTTPServer:
    if len(token) < 16:
        raise RuntimeError("Set KAGGLE_AGENT_API_TOKEN to a random string of at least 16 characters")
    handler = type("AuthedHandler", (Handler,), {"token": token})
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str, port: int) -> None:
    server = make_server(host, port, os.environ.get("KAGGLE_AGENT_API_TOKEN", ""))
    print(json.dumps({"ok": True, "result": {"listening": f"http://{host}:{port}"}}), flush=True)
    server.serve_forever()
