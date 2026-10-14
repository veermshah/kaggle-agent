# kaggle-agent

An autonomous, reusable agent for Kaggle competitions. Point it at a competition slug and it downloads the
data, profiles it, freezes a validation scheme, trains baselines, then keeps proposing, running and
evaluating experiments (hyperparameter search, feature engineering, model selection, ensembling) for as long
as you let it. It tracks everything in SQLite and only submits to Kaggle with your approval.

Tabular classification and regression are production-ready. Image classification has a working GPU
training adapter; NLP is a planned adapter.

```
                 ┌─────────────── research agents ───────────────┐
                 │ heuristic (default) · any LLM CLI · Claude API │
                 └───────────────────────┬───────────────────────┘
                              proposals (validated JSON specs)
 CLI / HTTP API / dashboard ──► SQLite queue ◄── scheduler daemon ──► worker subprocesses (1 GPU + N CPU)
   (Claude Code, Devin, you)        │                                    │ fixed folds, fold checkpoints
                                    ▼                                    ▼
                         experiments · submissions · events      oof / test / holdout predictions
```

## Setup (Windows PowerShell)

```powershell
cd C:\Users\<you>\cs\kaggle-agent
py -3.13 -m venv .venv --system-site-packages   # reuses an existing CUDA torch install
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py gpu        # hardware, CUDA, and which libraries can really train on your GPU
python -m pytest tests    # ~3 minutes, no credentials needed
```

`--system-site-packages` lets the venv reuse an already-installed CUDA build of PyTorch. If you have none,
install one that matches your driver (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu128`
for RTX 50-series). PyTorch is only needed for the vision adapter.

**Kaggle credentials** (never committed): run `kaggle auth login`, or click **Generate New Token** under
"API Tokens (Recommended)" at https://www.kaggle.com/settings/api (not the legacy kaggle.json key) and either set `KAGGLE_API_TOKEN` (in a git-ignored `.env`; see
`.env.example`) or save it to `%USERPROFILE%\.kaggle\access_token`. You must join each competition and
accept its rules on kaggle.com yourself. The agent never automates the website.

## Quick start

Offline, with a synthetic competition (simulated submissions scored against a hidden answer key):

```powershell
python main.py mock create --kind binary --out data\mock-binary
python main.py start --competition mock-binary --data-dir data\mock-binary
python main.py baseline --run                 # every suitable model family
python main.py loop --max-experiments 20      # autonomous research
python main.py report --print                 # what worked, what failed, what is next
python main.py dashboard                      # http://127.0.0.1:8501
```

A real competition:

```powershell
python main.py start --competition playground-series-s5e10
python main.py rules set --external-data forbidden --pretrained-models allowed   # after reading the rules
python main.py loop --max-hours 48
python main.py exp leaderboard
python main.py promote --id 42                # reveals the holdout score for that model
python main.py submit prepare --exp-id 42     # validated CSV, queued for approval
python main.py submit approve --id 1          # human-only: needs an interactive terminal
python main.py submit refresh                 # pulls the public LB score
```

Every command prints one JSON object (`{"ok": true, "result": ...}` or `{"ok": false, "error": ...}`) and
exits non-zero on failure. Logs go to stderr. `-c SLUG` targets a competition; otherwise the last started
one is used.

## What happens on `start`

1. Metadata comes from the Kaggle API: title, metric name, deadline, daily submission limit, whether it is a
   code competition, and whether you have joined.
2. Files are downloaded and zip archives are extracted safely (path traversal refused). Nothing downloaded is executed.
3. The profiler identifies train/test/sample files, id and target columns, the task (binary / multiclass /
   regression), submission format (probability / label / per-class probabilities / value), class order, and
   column types (numeric, categorical, datetime, text, dropped id-like/constant).
4. Integrity checks run: single-column target leakage, duplicate rows, train/test row overlap (contamination),
   adversarial validation for distribution shift, group columns disjoint from test, test-after-train time ordering.
5. The metric is mapped from Kaggle's metric name (falls back to a task default with a warning; override with `--metric`).
6. The validation scheme is chosen and frozen (see below).
7. `workspaces/<slug>/competition.yaml` is written. Edit it to override anything, then re-run `start`.
   Your rule decisions, validation overrides and autosubmit settings survive re-onboarding.

## Validation integrity

- **Strategy** is chosen from the data: time-series when test is later than train, group K-fold when a
  group-like column is disjoint between train and test, stratified K-fold for classification, K-fold for regression.
  It can be overridden in `competition.yaml` (`validation: {strategy, n_splits, group_column, time_column, holdout_fraction}`).
- **Folds are frozen** in `folds.csv` and hashed. Every experiment records the hash; scores are only
  compared within one validation version, and a tampered `folds.csv` stops training. Changing folds is a
  deliberate `validation reset --yes` that creates a new version.
- **Final holdout** (default 10% when train has ≥2000 rows) is never used for selection. Each experiment
  computes a holdout score, but it is hidden from listings, the research context and agents. It is shown
  only when you `promote` a model, with a warning if it falls well short of CV.
- Early stopping uses the validation fold (standard practice). Target encoding is fit out-of-fold inside
  each training fold. Ensemble weights and calibration are fit with nested CV, so ensemble scores are honest.
- Out-of-fold predictions are kept for every experiment, and ensembles blend them.
- The public leaderboard is recorded for reference only. Selection always uses local CV.

## The research loop

`python main.py loop` runs a daemon that:

1. recovers interrupted experiments (dead workers are re-queued and resume from per-fold checkpoints),
2. asks the research agent for proposals when the queue is low; every proposal has a stated hypothesis,
3. dispatches each experiment to its own subprocess (one GPU-heavy job at a time, CPU jobs alongside),
4. kills experiments that exceed `max_experiment_seconds`, retries crashed ones up to `max_attempts`, re-queues GPU OOMs on CPU,
5. stops on its budget (`--max-hours`, `--max-experiments`), when nothing improves for `stop_after_no_improvement`
   experiments, when the agent has nothing new, on `python main.py stop`, or when free disk falls below `min_free_disk_gb`.

The LLM does not need to stay connected: all state is in SQLite, so you can kill the daemon or reboot and
run `python main.py resume`.

**Default heuristic strategy** (`agents/heuristic.py`). Candidate experiments fall into arms:

| arm | what it does |
|---|---|
| `baseline:<family>` | default LightGBM, XGBoost, CatBoost, linear, ExtraTrees, RF (chosen by data size) |
| `tune:<family>` | Optuna TPE + median pruning on the two strongest families (study persisted, resumable, warm-started) |
| `features:<family>:<recipe>` | adds one feature recipe to the best model (count/target encoding, missing flags, datetime parts, row stats, interactions, log transforms, text lengths) |
| `target:` | log1p target for skewed non-negative regression targets |
| `revisit:<family>` | carries a winning feature set to the other families |
| `ensemble`, `calibrate` | hill-climbing blend of diverse top models; Platt calibration for log loss |
| `seed:` | seed-bagged copies of the best model for ensemble diversity |
| `retry:` | re-runs GPU failures on CPU |

Candidates are scored by arm prior + observed improvement rate + UCB exploration bonus − expected compute.
An arm that fails to improve `arm_patience` times in a row is parked, and it's unparked when the global
best moves (new evidence). Identical specs are never run twice (fingerprints). Failures are remembered and
not retried unless you `exp requeue` them. Exploratory arms are capped at `exploration_budget`.

## Agent control interface (Claude Code, Devin, MCP)

Everything an agent needs is a CLI command with JSON output:

| need | command |
|---|---|
| competition info | `status`, `info --profile`, `rules show`, `validation show` |
| history / metrics | `exp list`, `exp leaderboard`, `exp show --id N`, `exp compare --ids 3 7`, `events` |
| logs / failures | `exp logs --id N`, `exp failures` |
| what to try next | `exp suggest` (heuristic agent's ranked ideas) |
| propose | `exp propose --spec '{"model":"xgb","params":{"max_depth":4},"features":["count_encoding"]}' --hypothesis "..."` |
| run | `exp run --id N`, `loop`, `resume`, `stop` |
| select / submit | `promote --id N`, `submit prepare --exp-id N` (approval stays with a human) |

Specs are declarative and validated: known model families, feature recipes and scalar parameters only.
For new feature ideas, an agent can write `workspaces/<slug>/custom_features/<name>.py` defining
`transform(df) -> DataFrame` and reference it as `"features": ["custom:<name>"]`. It runs only after a
human reviews it and runs `features approve --name <name>`; any edit revokes the approval. A module can
declare `REQUIRES = ["external_data"]`, and the rule gates apply to it too.

**Remote control without a local GPU:** run `python main.py serve` on the GPU machine with
`KAGGLE_AGENT_API_TOKEN` set (≥16 chars). The HTTP API (see `agents/api_server.py`) mirrors the CLI
(`GET /status`, `POST /experiments`, `POST /loop/start`, ...). It binds to 127.0.0.1 by default; reach it
over an SSH tunnel or similar, and never expose it directly to the internet. It has no submission-approval route.

**Plugging in an LLM as the research agent** (per competition, in `competition.yaml`):

```yaml
agent:
  type: command                 # any program: gets the research context JSON on stdin,
  command: ["claude", "-p"]     # prints {"proposals": [{"spec": ..., "hypothesis": ..., "arm": ...}]}
  timeout: 600
# or
agent:
  type: anthropic               # direct Claude API (pip install anthropic; ANTHROPIC_API_KEY)
  model: claude-opus-5-5
  effort: high
```

**Free, local alternative** (Ollama or any OpenAI-compatible server such as LM Studio or llama.cpp):

```yaml
agent:
  type: local
  model: qwen3:8b          # `ollama pull qwen3:8b`
  think: true
```

The model is unloaded after each call (`keep_alive: 0`) and is never loaded while a GPU experiment runs
(`defer_while_gpu_busy`), so it does not compete with training for VRAM. Those rounds fall back to the
heuristic. A small context is sent so 8B-class models stay accurate. Measured on this machine:
qwen3:8b with thinking gave 4/4 valid, novel proposals in ~12 s; llama3.1:8b gave 3/4.

LLM proposals are validated like any others, and the heuristic agent fills any gap or covers a failure.
The `anthropic` agent enables Claude API server-side refusal fallbacks (`fallbacks: "default"`). Calls are
billed to your Anthropic account.

## Autonomous Claude Code research loop

```powershell
python main.py research -c <slug> --with-daemon      # runs for days; Ctrl+C or create workspaces/<slug>/STOP_RESEARCH
python main.py research -c <slug> --once             # a single session now
```

The daemon keeps the GPU busy without any LLM. About every 45 minutes, once at least 3 new results exist
(set `research: {every_minutes, min_new_results, session_timeout, max_sessions, model}` in competition.yaml),
a fresh headless `claude -p` session starts. It reads the report and its own `research_notes.md`, studies the
data (`main.py data --column X`), writes feature modules into `custom_features/`, queues experiments with
hypotheses, and updates the notes for the next session. Fresh context each session, with memory in SQLite plus
the notes file, so it can run for days. It uses your Claude Code plan; each session's transcript summary and
reported cost are saved in `workspaces/<slug>/logs/research/`.

Sandbox: `--permission-mode dontAsk` with an allow-list (main.py commands, read-only tools, and writes only to
`custom_features/` and the notes file). With `KAGGLE_AGENT_ROLE=agent`, main.py refuses approvals, rule and
validation changes, autosubmit, and cleanup. New feature code is screened statically (pandas/numpy-style
imports only; no file, network, process or `eval`/`exec` access) and runs only after approval:

```powershell
python main.py features list -c <slug>                       # shows approval state + screen findings
python main.py features approve --name cabin_group -c <slug> # after reading the file
python main.py features auto-approve --state on -c <slug>    # opt-in: approve code that passes the screen
```

With auto-approve on, nothing waits for you, but agent-written code then runs on your machine after only a
static screen (inside a time-limited worker process). It's your call per competition.

## Submissions

- Files are built from test predictions in the exact sample-submission layout: column names and order,
  id order, row count, labels mapped back to original values (e.g. `True`/`False`), class probabilities
  normalized. They are validated before queueing.
- Submissions that are nearly identical to an earlier one (correlation ≥ 0.9995, or label agreement) are refused unless you pass `--force`.
- Daily usage is counted per UTC day from both the local records and Kaggle's own list, so manual website
  submissions also count. Limit = min(Kaggle limit, optional `submission_budget`).
- **Approval**: `submit approve` needs an interactive terminal (or the dashboard button), so automated agents cannot approve.
- **Autonomous mode** is per competition and opt-in:
  `rules set --automated-submissions allowed` then `autosubmit enable --max-daily 2 --i-authorize`.
  The daemon then submits only when a new best CV beats the best submitted CV, within that budget.
- Code competitions, disabled submissions, or not having joined the competition block API submission.
  Simulated submissions (local data source) are labelled as simulated and never touch Kaggle.

## Competition rules and safety

The Kaggle API does not expose rule text, so external data, pretrained models and automated submissions
start as `unknown`, and anything that needs them is **blocked** until you record a decision with `rules set`.
Checks run when an experiment is queued, again in the worker before training, and before every submission.
Other safeguards: no credentials in code or git; no execution of downloaded files; subprocess timeouts;
human confirmation for `validation reset`, `cleanup` and code approval; detailed logs (`train.log`,
`worker.log`, `events`).

## Resources (tuned for an RTX 5070 Ti, 16 GB)

`python main.py gpu` probes real GPU support once and caches it. On this machine, XGBoost (CUDA) and CatBoost
(GPU) train on the GPU. The LightGBM pip wheel is CPU-only. GPU is used automatically for GPU-capable families
when train has ≥ `gpu_min_rows` rows (smaller data is faster on CPU). Workers record peak RSS and GPU memory,
heartbeat to the DB, and release VRAM by exiting. The vision adapter uses bf16 autocast and channels-last.
`cleanup --yes` deletes model binaries outside the top N (predictions are kept). Disk headroom is checked
before each dispatch.

## Defaults (config/defaults.yaml; override in config/local.yaml or competition.yaml `loop:`)

| setting | default | why |
|---|---|---|
| folds / seed | 5 / 42 | standard bias–variance trade-off |
| holdout | 10% if ≥2000 rows | detects selection overfitting without starving small datasets |
| GPU / CPU slots | 1 / 1 | one GPU-heavy job at a time; boosting libraries already use all cores |
| max experiment duration | 2 h | tune experiments get their Optuna timeout × 1.5 |
| loop budget | 24 h / 200 experiments | |
| improvement threshold | 0.0002 × max(1, \|best\|) | scale-aware "real" gain |
| arm patience | 3 | |
| stop after no improvement | 25 experiments | |
| Optuna | 40 trials / 30 min per tune | |
| submissions | approval required; duplicate corr 0.9995 | |

## Project layout

```
main.py                CLI (JSON in/out)
config/                defaults.yaml (+ optional local.yaml)
core/                  settings, workspace, compliance gates, resources/GPU, process control, versioning
competitions/          Kaggle client (real + local/simulated), onboarding, profiler, tabular loader, mock data, adapters
evaluation/            metrics registry, validation schemes
models/                tabular families (LightGBM/XGBoost/CatBoost/sklearn), vision (PyTorch)
experiments/           specs, features, trainer, tuning, ensembling, service, scheduler daemon, custom code gate
agents/                research context, heuristic strategist, command/Claude agents, HTTP API
submissions/           builder/validator, manager (approval, budget, dedupe, scoring)
tracking/              SQLite store, Markdown report
dashboard/             Streamlit app
tests/                 pytest suite (mock data + mocked Kaggle API; no credentials)
workspaces/<slug>/     data, folds, experiments/<id>/{oof,test,holdout}.npy, logs, submissions (git-ignored)
```

## Limitations and next steps

- Multi-target / multi-label competitions are detected but not supported yet.
- The vision adapter (image classification) trains and checkpoints on GPU, but onboarding does not
  auto-detect image competitions and the heuristic agent does not propose vision experiments yet.
  NLP has only the adapter interface.
- Kaggle rule text cannot be read through the API, so rule decisions are yours to record.
- Code competitions (notebook-only submission) are detected and blocked, not automated.
