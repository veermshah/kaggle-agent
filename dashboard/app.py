"""Local dashboard: `python main.py dashboard` (http://127.0.0.1:8501). Read-only except for the human
submission approval controls."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.resources import gpu_info  # noqa: E402
from core.settings import load_dotenv, workspaces_root  # noqa: E402
from core.workspace import Workspace, get_active  # noqa: E402
from experiments.scheduler import daemon_state  # noqa: E402
from tracking.db import Tracker  # noqa: E402

load_dotenv()
st.set_page_config(page_title="Kaggle Agent", layout="wide")
tracker = Tracker()

slugs = sorted(p.name for p in workspaces_root().iterdir() if (p / "competition.yaml").exists())
if not slugs:
    st.info("No competitions yet. Run `python main.py start --competition <slug>`.")
    st.stop()
active = get_active()
slug = st.sidebar.selectbox("Competition", slugs, index=slugs.index(active) if active in slugs else 0)
refresh = st.sidebar.slider("Auto-refresh (seconds, 0 = off)", 0, 120,
                            int(os.environ.get("KAGGLE_AGENT_DASHBOARD_REFRESH", 15)))
ws = Workspace(slug)
config = ws.load_config()
validation = ws.load_validation()
greater = config["direction"] == "maximize"

st.title(config.get("title", slug))
st.caption(f"{config['task_type']} · metric **{config['metric']}** ({config['direction']}) · validation "
           f"{validation['strategy']} × {validation['n_splits']}, holdout {validation['holdout_fraction']:.0%} "
           f"(version `{validation['version']}`)")

exps = pd.DataFrame(tracker.list_experiments(slug, validation_version=validation["version"]))
state = daemon_state(ws) or {}
done = exps[exps["status"] == "succeeded"] if not exps.empty else exps
best = (done.sort_values("cv_score", ascending=not greater).iloc[0] if not done.empty else None)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Best CV", f"{best['cv_score']:.6f}" if best is not None else "-",
          help=f"exp {int(best['id'])} ({best['model_family']})" if best is not None else None)
c2.metric("Experiments", len(exps), f"{int((exps['status'] == 'queued').sum()) if not exps.empty else 0} queued")
c3.metric("Daemon", "running" if state.get("alive") else "stopped",
          state.get("stop_reason") or "", delta_color="off")
budget_left = (state.get("max_experiments", 0) - state.get("dispatched", 0)) if state else None
c4.metric("Experiment budget left", budget_left if budget_left is not None else "-")
hours_left = round(state["max_hours"] - state["elapsed_hours"], 2) if state.get("alive") else None
c5.metric("Hours left", hours_left if hours_left is not None else "-")

st.subheader("GPU")
gpus = gpu_info()
if gpus:
    for g in gpus:
        st.progress(min(1.0, g["memory_used_mb"] / g["memory_total_mb"]),
                    text=f"{g['name']}: {g['memory_used_mb']}/{g['memory_total_mb']} MB, "
                         f"util {g['utilization_pct']}%, {g['temperature_c']}°C")
else:
    st.write("No NVIDIA GPU detected.")

if not exps.empty:
    running = exps[exps["status"] == "running"]
    if not running.empty:
        st.subheader("Running now")
        for _, r in running.iterrows():
            log = ws.experiment_dir(int(r["id"])) / "train.log"
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-5:] if log.exists() else []
            st.markdown(f"**exp {int(r['id'])}** {r['model_family']} [{r['arm']}] — {r['hypothesis']}")
            st.code("\n".join(tail) or "(starting)")

    st.subheader("Score progression")
    if not done.empty:
        prog = done.sort_values("finished_at")[["id", "cv_score"]].copy()
        prog["best so far"] = prog["cv_score"].cummax() if greater else prog["cv_score"].cummin()
        st.line_chart(prog.set_index("id")[["cv_score", "best so far"]])

    left, right = st.columns(2)
    with left:
        st.subheader("Model families")
        if not done.empty:
            agg = done.groupby("model_family").agg(best_cv=("cv_score", "max" if greater else "min"),
                                                   runs=("id", "count"), mean_seconds=("train_seconds", "mean"))
            st.dataframe(agg.sort_values("best_cv", ascending=not greater), width="stretch")
    with right:
        st.subheader("Status")
        st.bar_chart(exps["status"].value_counts())

    st.subheader("Experiment history")
    cols = ["id", "status", "kind", "model_family", "feature_set", "arm", "cv_score", "cv_std", "train_seconds",
            "resource", "peak_gpu_mb", "hypothesis", "error"]
    st.dataframe(exps[cols].sort_values("id", ascending=False), width="stretch", hide_index=True)

st.subheader("Submissions")
subs = pd.DataFrame(tracker.list_submissions(slug))
if subs.empty:
    st.write("None yet. Prepare one with `python main.py submit prepare --exp-id N`.")
else:
    st.dataframe(subs[["id", "experiment_id", "status", "cv_score", "public_score", "message", "created_at",
                       "submitted_at"]], width="stretch", hide_index=True)
    pending = subs[subs["status"] == "pending_approval"]
    for _, s in pending.iterrows():
        with st.expander(f"Pending submission {int(s['id'])} (exp {int(s['experiment_id'])}, CV {s['cv_score']:.6f})"):
            st.write(s["file_path"])
            confirm = st.text_input("Type the submission id to confirm", key=f"confirm{s['id']}")
            a, r = st.columns(2)
            if a.button("Approve and submit", key=f"approve{s['id']}", disabled=confirm != str(int(s["id"]))):
                from submissions.manager import SubmissionManager
                try:
                    st.success(SubmissionManager(ws, tracker).approve(int(s["id"]), approver="dashboard"))
                except Exception as exc:
                    st.error(str(exc))
            if r.button("Reject", key=f"reject{s['id']}"):
                from submissions.manager import SubmissionManager
                SubmissionManager(ws, tracker).reject(int(s["id"]), "rejected in dashboard")
                st.rerun()

with st.expander("Research report"):
    from tracking.report import generate_report
    st.markdown(generate_report(ws, tracker))

with st.expander("Recent events"):
    st.dataframe(pd.DataFrame(tracker.list_events(slug, 100)), width="stretch", hide_index=True)

if refresh:
    import time
    time.sleep(refresh)
    st.rerun()
