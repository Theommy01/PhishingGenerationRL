"""Browse the loop's runs.

    streamlit run dashboard/app.py

Reads MongoDB live, so a run in progress is browsable while it generates: the
messages are stored as they are produced and scored afterwards, and every view
here is written to handle a round that is half generated or generated but not
yet scored.
"""

import os
import sys

import pandas as pd
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dashboard import charts, data  # noqa: E402
from dashboard.views import compare, messages, overview, provenance, transfer  # noqa: E402

st.set_page_config(page_title="PhishingGenerationRL", page_icon="🎣", layout="wide")


def pick_run() -> int:
    """The run selector, with its state summarised beside it."""
    runs = data.list_runs()
    if runs.empty:
        st.warning("No runs in the database yet. Start one with `python run_loop.py`.")
        st.stop()

    def label(run_id: int) -> str:
        row = runs[runs["run_id"] == run_id].iloc[0]
        mark = "● running" if row["running"] else "○"
        return (
            f"{run_id}  {mark}  {row['algorithm']}/{row['ref_mode']}  "
            f"{row['decoding']} T={row['temperature']}  "
            f"{row['rounds']} rounds  {row['messages']} msgs"
        )

    with st.sidebar:
        st.header("Run")
        run_id = st.selectbox(
            "run", runs["run_id"], format_func=label, label_visibility="collapsed"
        )
        if st.button("↻ refresh", use_container_width=True):
            data.refresh()
            st.rerun()
        st.caption(f"cached for {data.CACHE_TTL}s")

        row = runs[runs["run_id"] == run_id].iloc[0]
        st.divider()
        st.metric("Messages", int(row["messages"]))
        if row["unscored"]:
            st.metric("Awaiting scoring", int(row["unscored"]))
        st.caption(
            f"{row['prompts']} prompts ({row['holdout']} held out) "
            f"× {row['n_samples']} samples"
        )
        st.caption(f"started {row['created_at']:%Y-%m-%d %H:%M}")

        with st.expander("config"):
            st.json(data.run_config(run_id), expanded=False)

    return int(run_id)


def main() -> None:
    st.title("PhishingGenerationRL")
    run_id = pick_run()

    live = data.progress(run_id)
    if any(r["unscored"] for r in live["rounds"]):
        pending = [r for r in live["rounds"] if r["unscored"]]
        for record in pending:
            done, expected = record["messages"], record["expected"]
            st.progress(
                min(done / expected, 1.0) if expected else 0.0,
                text=(
                    f"round {record['round']}: {done}/{expected} generated, "
                    f"{record['unscored']} awaiting scoring — metrics appear once "
                    "the round is scored"
                ),
            )

    tabs = st.tabs(["Overview", "Messages", "Transfer", "Compare", "Provenance"])
    with tabs[0]:
        overview.render(run_id)
    with tabs[1]:
        messages.render(run_id)
    with tabs[2]:
        transfer.render(run_id)
    with tabs[3]:
        compare.render(run_id)
    with tabs[4]:
        provenance.render(run_id)


main()
