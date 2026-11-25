#!/usr/bin/env bash
# Prepares a Claude Code cloud session (claude.ai/code) to run kaggle-agent.
# Runs from the SessionStart hook in .claude/settings.json; safe to run again (it skips work already done).
#
#  1. a virtualenv at ~/.venvs/kaggle-agent with requirements-cloud.txt (outside the repo, so an
#     environment setup script can pre-build it and the cloud snapshot keeps it)
#  2. puts that venv first on PATH for Claude's shell (via CLAUDE_ENV_FILE), so `python main.py ...` works
#  3. writes cloud-sized defaults to config/local.yaml (git-ignored) if you have none
#  4. reports whether Kaggle credentials are present, without printing them
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ] && [ "${1:-}" != "--force" ]; then
  exit 0  # local machine: follow the README instead
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${KAGGLE_AGENT_VENV:-$HOME/.venvs/kaggle-agent}"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c "import lightgbm, xgboost, catboost, optuna, sklearn, kaggle" 2>/dev/null; then
  echo "kaggle-agent: installing Python packages (first session only, a few minutes)..." >&2
  "$VENV/bin/pip" install --quiet --disable-pip-version-check -r "$ROOT/requirements-cloud.txt" >&2
fi

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$VENV/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi

if [ ! -f "$ROOT/config/local.yaml" ]; then
  cat > "$ROOT/config/local.yaml" <<'YAML'
# Written by scripts/cloud-setup.sh for a Claude Code cloud session (4 vCPU, 16 GB RAM, ~30 GB disk, no GPU).
resources:
  use_gpu: false
  cpu_slots: 1
  max_experiment_seconds: 3600
  min_free_disk_gb: 3
loop:
  max_hours: 3
  max_experiments: 60
  tune_trials: 25
  tune_seconds: 900
YAML
fi

if [ -n "${KAGGLE_API_TOKEN:-}" ] || { [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; } \
   || [ -f "$HOME/.kaggle/access_token" ] || [ -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "kaggle-agent: ready (Kaggle credentials found)."
else
  echo "kaggle-agent: ready, but no Kaggle credentials. Add KAGGLE_API_TOKEN to this cloud environment's variables to use real competitions; mock competitions work without it."
fi
