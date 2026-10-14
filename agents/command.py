"""Bridge to any external research agent that can be run as a command (an LLM CLI such as `claude -p`,
a Devin wrapper, a script calling some API...).

The command receives the research context as JSON on stdin, followed by the instructions below, and must
print JSON: {"proposals": [{"spec": {...}, "hypothesis": "...", "arm": "..."}]}. Proposals are validated as
declarative specs; anything invalid is dropped and the heuristic agent fills the gap.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agents.base import Proposal, ResearchContext
from core.proc import run_command
from experiments.features import RECIPES
from experiments.spec import KINDS, SpecError, normalize_spec
from models.tabular import FAMILIES

log = logging.getLogger("kaggle_agent.agents")

INSTRUCTIONS = """You are the research lead for a Kaggle competition. Above is JSON describing the competition and
every experiment so far (scores are local cross-validation; never optimise for the public leaderboard).
Propose up to {n} NEW experiments that are most likely to improve the CV {metric} ({direction}).
Each needs a falsifiable hypothesis. Do not repeat experiments already listed.
Allowed spec fields: kind ({kinds}), model ({models}), params (scalar hyperparameters), features ({recipes}),
feature_params, target_transform (none|log1p, regression only), seed, n_trials/timeout (tune), method/top_k/calibrate
(ensemble).
Respond with ONLY a JSON object: {{"proposals": [{{"spec": {{...}}, "hypothesis": "...", "arm": "short-label"}}]}}"""


def build_prompt(ctx: ResearchContext, n: int) -> str:
    payload = ctx.to_dict()
    payload["experiments"] = payload["experiments"][-60:]
    return json.dumps(payload, default=str) + "\n\n" + INSTRUCTIONS.format(
        n=n, metric=ctx.metric, direction=ctx.direction, kinds="|".join(KINDS), models="|".join(FAMILIES),
        recipes="|".join(RECIPES))


def parse_proposals(text: str, task_type: str, source: str) -> list[Proposal]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in agent output")
    data = json.loads(match.group(0))
    out = []
    for item in data.get("proposals", []):
        try:
            spec = normalize_spec(item["spec"], task_type)
            hyp = str(item["hypothesis"]).strip()
            if not hyp:
                raise SpecError("empty hypothesis")
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("dropping invalid proposal from %s: %s", source, exc)
            continue
        arm = re.sub(r"[^a-zA-Z0-9:_\-]", "", str(item.get("arm") or "llm"))[:60] or "llm"
        out.append(Proposal(spec, hyp, f"{source}:{arm}", priority=float(item.get("priority", 1.0))))
    return out


class CommandAgent:
    def __init__(self, command: list[str], timeout: float = 600, name: str = "command"):
        if not command:
            raise ValueError("agent.command must be a non-empty list, e.g. [\"claude\", \"-p\"]")
        self.command, self.timeout, self.name = command, timeout, name

    def propose(self, ctx: ResearchContext, n: int) -> list[Proposal]:
        result = run_command(self.command, input_text=build_prompt(ctx, n), timeout=self.timeout)
        if result.returncode != 0:
            raise RuntimeError(f"agent command exited {result.returncode}: {result.stderr[-500:]}")
        return parse_proposals(result.stdout, ctx.task_type, self.name)[:n]


class CompositeAgent:
    """Asks the primary agent first; the fallback fills remaining slots or covers failures."""

    def __init__(self, primary: Any, fallback: Any):
        self.primary, self.fallback = primary, fallback
        self.name = f"{primary.name}+{fallback.name}"
        self.last_error: str | None = None

    def propose(self, ctx: ResearchContext, n: int) -> list[Proposal]:
        props: list[Proposal] = []
        try:
            props = self.primary.propose(ctx, n)
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            log.warning("primary agent %s failed: %s", self.primary.name, exc)
        if len(props) < n:
            props += self.fallback.propose(ctx, n - len(props))
        return props[:n]


def make_agent(ws, agent_cfg: dict[str, Any] | None = None):
    from agents.heuristic import HeuristicAgent
    config = ws.load_config()
    heuristic = HeuristicAgent(numeric_columns=config["features"].get("numeric", []),
                               has_categorical=bool(config["features"].get("categorical")))
    agent_cfg = agent_cfg or config.get("agent") or {}
    if agent_cfg.get("type") == "command":
        return CompositeAgent(CommandAgent(agent_cfg["command"], agent_cfg.get("timeout", 600)), heuristic)
    return heuristic
