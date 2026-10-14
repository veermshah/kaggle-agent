"""Research agent backed by a model running on this machine (Ollama, or any OpenAI-compatible server such
as LM Studio or llama.cpp's llama-server). No API costs, no new dependencies.

competition.yaml:
    agent:
      type: local
      model: qwen3:8b                 # any model you have pulled
      provider: ollama                # or: openai (OpenAI-compatible /v1/chat/completions)
      base_url: http://localhost:11434
      keep_alive: 0                   # Ollama: unload right after each call so training gets the VRAM back
      defer_while_gpu_busy: true      # don't load the LLM while a GPU experiment is running
      think: true                     # qwen3-style reasoning: ~2x slower, more valid proposals
      timeout: 600

Proposals are validated exactly like any other agent's, and the heuristic agent fills gaps or covers failures.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from agents.base import Proposal, ResearchContext
from agents.command import build_prompt, parse_proposals

SYSTEM = ("You are an expert Kaggle competitor acting as research lead. Reply with a single JSON object and "
          "nothing else.")


class LocalLLMAgent:
    uses_local_gpu = True

    def __init__(self, model: str = "qwen3:8b", provider: str = "ollama", base_url: str | None = None,
                 keep_alive: int | str = 0, think: bool = True, timeout: float = 600,
                 defer_while_gpu_busy: bool = True, temperature: float = 0.7):
        if provider not in ("ollama", "openai"):
            raise ValueError("agent.provider must be 'ollama' or 'openai'")
        self.model, self.provider, self.keep_alive, self.think = model, provider, keep_alive, think
        self.base_url = (base_url or ("http://localhost:11434" if provider == "ollama"
                                      else "http://localhost:1234")).rstrip("/")
        self.timeout, self.temperature = timeout, temperature
        self.defer_while_gpu_busy = defer_while_gpu_busy
        self.name = f"local-{model.split(':')[0].split('/')[-1]}"

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Local LLM server at {self.base_url} unreachable ({exc}). Is Ollama running?") from exc

    def complete(self, prompt: str) -> str:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        if self.provider == "ollama":
            out = self._post("/api/chat", {
                "model": self.model, "messages": messages, "stream": False, "format": "json",
                "think": self.think, "keep_alive": self.keep_alive,
                "options": {"temperature": self.temperature, "num_ctx": 16384},
            })
            return out.get("message", {}).get("content", "")
        out = self._post("/v1/chat/completions", {
            "model": self.model, "messages": messages, "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        })
        return out["choices"][0]["message"]["content"]

    def propose(self, ctx: ResearchContext, n: int) -> list[Proposal]:
        text = self.complete(build_prompt(ctx, n, compact=True))
        return parse_proposals(text, ctx.task_type, self.name)[:n]
