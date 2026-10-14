"""Research agent backed directly by the Claude API (optional; `pip install anthropic`).

Enable per competition in competition.yaml:
    agent: {type: anthropic, model: claude-opus-5-5, effort: high}
Credentials come from the environment (ANTHROPIC_API_KEY or an `ant auth login` profile).
Its proposals go through the same spec validation as every other agent and are mixed with the heuristic
agent's (CompositeAgent), so a failed or refused call never stalls the loop.
"""
from __future__ import annotations

from typing import Any

from agents.base import Proposal, ResearchContext
from agents.command import build_prompt, parse_proposals

SYSTEM = ("You are a senior Kaggle grandmaster acting as research lead. You only propose experiments as JSON; "
          "an automated system validates and runs them on fixed cross-validation folds.")


class AnthropicAgent:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", max_tokens: int = 16000,
                 client: Any = None):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.name = "claude"
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError("agent.type=anthropic needs `pip install anthropic`") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def propose(self, ctx: ResearchContext, n: int) -> list[Proposal]:
        # Server-side fallback reroutes a safety-classifier refusal to another model inside the same call.
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": build_prompt(ctx, n)}],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude declined to propose experiments")
        text = "".join(block.text for block in response.content if block.type == "text")
        return parse_proposals(text, ctx.task_type, self.name)[:n]
