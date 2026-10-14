import json
import sys
from pathlib import Path

import main
from agents.claude_research import build_brief, claude_command, research_settings, run_session
from experiments import custom_code, service

FAKE_SESSION = r'''
import json, os, subprocess, sys
from pathlib import Path
slug, features_dir = sys.argv[1], Path(sys.argv[2])
assert os.environ["KAGGLE_AGENT_ROLE"] == "agent"
(features_dir / "num_ratio.py").write_text(
    "import numpy as np\nimport pandas as pd\n\n"
    "def transform(df):\n    return pd.DataFrame({'r': df['num_0'] / (df['num_1'].abs() + 1),\n"
    "                         'id_mod': df['id'] % 7})\n")
def cli(*args):
    out = subprocess.run([sys.executable, "main.py", *args, "-c", slug], capture_output=True, text=True)
    return json.loads(out.stdout)
spec = json.dumps({"model": "lgbm", "features": ["custom:num_ratio"]})
queued = cli("exp", "propose", "--spec", spec, "--hypothesis", "ratio of the two strongest columns adds signal")
refused = cli("features", "approve", "--name", "num_ratio", "--yes")
print(json.dumps({"result": f"queued {queued['result']['experiment_id']}; self-approval ok={refused['ok']}",
                  "total_cost_usd": 0.0}))
'''


def _fake(tmp_path, ws):
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_SESSION)
    return [sys.executable, str(script), ws.slug, str(ws.custom_features_dir)]


def test_safety_screen(make_comp):
    ws = make_comp("binary")
    good = ws.custom_features_dir / "good.py"
    good.write_text("import numpy as np\nimport pandas as pd\n\ndef transform(df):\n    return pd.DataFrame({'a': np.log1p(df['num_0'].abs())})\n")
    assert custom_code.screen(ws, "good") == []
    bad = ws.custom_features_dir / "bad.py"
    bad.write_text("import os\nimport pandas as pd\n\ndef transform(df):\n    open('x','w')\n    os.system('calc')\n"
                   "    pd.read_csv('secret.csv')\n    return df.__class__\n")
    problems = " ".join(custom_code.screen(ws, "bad"))
    for needle in ("'os'", "'open'", "read_csv", "__class__"):
        assert needle in problems


def test_agent_role_cannot_use_human_commands(make_comp, monkeypatch):
    ws = make_comp("binary")
    monkeypatch.setenv("KAGGLE_AGENT_ROLE", "agent")
    for argv in (["features", "approve", "--name", "x", "--yes"], ["rules", "set", "--external-data", "allowed"],
                 ["autosubmit", "enable", "--max-daily", "1", "--i-authorize"], ["validation", "reset", "--yes"],
                 ["cleanup", "--yes"], ["features", "auto-approve", "--state", "on"]):
        code, out = main.run(argv + ["-c", ws.slug])
        assert code == 1 and "reserved for the human" in out["error"], argv
    code, out = main.run(["exp", "list", "-c", ws.slug])  # read/propose commands still work
    assert code == 0


def test_claude_command_is_sandboxed(make_comp, monkeypatch):
    ws = make_comp("binary")
    monkeypatch.setattr("shutil.which", lambda name: "claude")
    cmd = claude_command(ws, research_settings(ws.load_config()))
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    allowed = cmd[cmd.index("--allowedTools") + 1: cmd.index("--disallowedTools")]
    assert not any(t.startswith("Bash(") and "main.py" not in t for t in allowed)
    assert any("custom_features/**" in t and t.startswith("Write(") for t in allowed)
    assert "bypassPermissions" not in cmd
    assert "id_column" not in build_brief(ws) and "`id`" in build_brief(ws)


def test_research_session_with_human_approval(make_comp, tracker, tmp_path):
    ws = make_comp("binary", n_train=600)
    settings = {**research_settings(ws.load_config()), "command": _fake(tmp_path, ws)}
    rec = run_session(ws, tracker, settings, 1)
    assert rec["status"] == "ok" and "self-approval ok=False" in rec["summary"]
    assert rec["auto_approved"] == []  # auto-approval is off by default
    exp = tracker.list_experiments(ws.slug)[-1]
    assert exp["proposed_by"] == "user" or exp["hypothesis"].startswith("ratio")
    assert service.execute(exp["id"], tracker)["status"] == "blocked"  # waits for the human
    assert (ws.root / "research_notes.md").exists()
    assert json.loads((ws.logs_dir / "research" / "session_0001.json").read_text())["status"] == "ok"


def test_research_session_with_opt_in_auto_approval(make_comp, tracker, tmp_path):
    ws = make_comp("binary", n_train=600)
    main.run(["features", "auto-approve", "--state", "on", "-c", ws.slug])
    settings = {**research_settings(ws.load_config()), "command": _fake(tmp_path, ws)}
    rec = run_session(ws, tracker, settings, 1)
    assert rec["auto_approved"] == [{"name": "num_ratio", "approved": True}]
    exp = tracker.list_experiments(ws.slug)[-1]
    if exp["status"] == "blocked":  # blocked at queue time is fine; auto-approval re-queues it
        exp = tracker.get_experiment(exp["id"])
    assert exp["status"] == "queued"
    assert service.execute(exp["id"], tracker)["status"] == "succeeded"  # used the id column too


def test_feature_test_and_analysis_sandbox(make_comp, monkeypatch):
    import pytest
    from experiments.sandbox import SandboxError, run_sandboxed, sanitized_env
    ws = make_comp("binary", n_train=600)
    (ws.custom_features_dir / "ratio.py").write_text(
        "import pandas as pd\n\ndef transform(df):\n    return pd.DataFrame({'r': df['num_0'] * df['num_2'],"
        " 'const': 1.0})\n")
    monkeypatch.setenv("KAGGLE_AGENT_ROLE", "agent")
    with pytest.raises(SandboxError, match="auto-approve is off"):
        run_sandboxed(ws, "feature", "ratio")
    main.run(["exp", "list", "-c", ws.slug])
    monkeypatch.delenv("KAGGLE_AGENT_ROLE")
    main.run(["features", "auto-approve", "--state", "on", "-c", ws.slug])
    monkeypatch.setenv("KAGGLE_AGENT_ROLE", "agent")
    out = run_sandboxed(ws, "feature", "ratio")
    assert set(out["new_columns"]) == {"ratio__r", "ratio__const"}
    assert any("constant" in f for f in out["flags"])
    assert out["quick_check"]["metric"] == "roc_auc"

    analysis = ws.root / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "balance.py").write_text("def analyze(train, test):\n"
                                         "    return {'rows': len(train), 'rate': float(train['target'].mean())}\n")
    assert run_sandboxed(ws, "analysis", "balance")["output"]["rows"] == 600
    (analysis / "evil.py").write_text("import os\n\ndef analyze(train, test):\n    return os.environ\n")
    with pytest.raises(SandboxError, match="Safety screen"):
        run_sandboxed(ws, "analysis", "evil")
    code, out = main.run(["sandbox-exec", "--kind", "analysis", "--name", "balance", "-c", ws.slug])
    assert code == 1  # agents cannot call the internal executor directly

    monkeypatch.setenv("KAGGLE_API_TOKEN", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    env = sanitized_env()
    assert "KAGGLE_API_TOKEN" not in env and "ANTHROPIC_API_KEY" not in env
    assert env["KAGGLE_AGENT_NO_DOTENV"] == "1"
