"""Two runs side by side — built for the decoding sweep."""

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import data

HEADLINE = [
    ("evasion_rate", "Evasion rate", "%"),
    ("asr_at_n", "ASR@n", "%"),
    ("mean_score", "Mean score", "%"),
    ("kl_per_token", "KL / token", ""),
    ("url_compliance", "URL flag followed", "%"),
    ("attachment_compliance", "Attachment flag followed", "%"),
    ("cos_subject", "Similarity to subject", "%"),
    ("duplicates", "Duplicate bodies", ""),
]


def _final(summary: pd.DataFrame, split: str, column: str):
    if summary.empty or column not in summary:
        return None
    rows = summary[(summary["split"] == split) & summary[column].notna()]
    return None if rows.empty else rows.iloc[-1][column]


def render(run_id: int) -> None:
    runs = data.list_runs()
    if len(runs) < 2:
        st.info(
            "Only one run so far. The second decoding run (`--decoding sampling`) "
            "is what this view is for: identical prompts and identical training, "
            "different temperature."
        )
        return

    left_default = int(run_id)
    others = [r for r in runs["run_id"] if r != left_default]
    columns = st.columns(2)
    a = columns[0].selectbox("run A", runs["run_id"], index=list(runs["run_id"]).index(left_default))
    b = columns[1].selectbox("run B", others, index=0)

    split = st.radio("split", ["train", "holdout"], horizontal=True, key="compare-split")

    summaries = {run: data.split_summary(run) for run in (a, b)}
    configs = {run: data.run_config(run) for run in (a, b)}

    st.caption(
        "The runs differ in what their configs differ in — usually decoding. "
        "Everything else (prompts, subjects, algorithm, seed) is shared, which "
        "is what makes the comparison a controlled one."
    )

    differences = {
        key: (configs[a].get(key), configs[b].get(key))
        for key in sorted(set(configs[a]) | set(configs[b]))
        if configs[a].get(key) != configs[b].get(key)
    }
    if differences:
        st.dataframe(
            pd.DataFrame(
                [{"setting": k, f"run {a}": v[0], f"run {b}": v[1]} for k, v in differences.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("the two configs are identical")

    st.divider()
    rows = []
    for column, label, unit in HEADLINE:
        first, second = (_final(summaries[run], split, column) for run in (a, b))
        rows.append(
            {
                "metric": label,
                f"run {a}": None if first is None else round(first, 2),
                f"run {b}": None if second is None else round(second, 2),
                "difference": (
                    None if first is None or second is None else round(second - first, 2)
                ),
                "unit": unit,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"final round of each run, {split} split")

    frames = []
    for run in (a, b):
        summary = summaries[run]
        if summary.empty:
            continue
        subset = summary[summary["split"] == split].copy()
        subset["run"] = str(run)
        frames.append(subset)
    if not frames:
        return

    combined = pd.concat(frames)
    metric = st.selectbox(
        "trajectory", [c for c, _, _ in HEADLINE if c in combined], key="compare-metric"
    )
    chart = (
        alt.Chart(combined.dropna(subset=[metric]))
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=False)),
            color=alt.Color("run:N", title="Run"),
            tooltip=["run", "round", alt.Tooltip(f"{metric}:Q", format=".2f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
