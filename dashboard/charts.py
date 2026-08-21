"""Two ways to draw the same thing.

Interactive Altair for exploring, and the project's own matplotlib figures for
the write-up. `figure_toggle` puts both behind one control: the interactive
chart is the default, and switching to "thesis figure" calls
`visualisation.charts`, which writes the PNG to output/figures on the way — so
looking at the publication version is also how you produce it.
"""

import os
from typing import Callable, Optional

import altair as alt
import pandas as pd
import streamlit as st

# The project's palette, so the interactive charts and the figures agree.
from visualisation.style import PALETTE_ORANGE

SPLIT_COLORS = {"train": PALETTE_ORANGE[1], "holdout": PALETTE_ORANGE[2]}


def figure_toggle(key: str) -> str:
    """The interactive / thesis-figure switch. Returns the chosen mode."""
    return st.radio(
        "view",
        ["interactive", "thesis figure"],
        horizontal=True,
        label_visibility="collapsed",
        key=key,
    )


def thesis_figure(plot: Callable, *args, download_label: str = "figure", **kwargs):
    """Render a `visualisation.charts` figure and offer the PNG it wrote."""
    try:
        path = plot(*args, **kwargs)
    except Exception as exc:  # a half-populated live run can miss a column
        st.info(f"figure not available yet: {exc}")
        return

    st.image(path)
    with open(path, "rb") as handle:
        st.download_button(
            f"download {download_label}.png",
            handle.read(),
            file_name=os.path.basename(path),
            mime="image/png",
            key=f"download-{download_label}",
        )
    st.caption(f"written to {path}")


def _line(df: pd.DataFrame, y: str, title: str, y_title: str, domain=None):
    scale = alt.Scale(domain=domain) if domain else alt.Scale(zero=False)
    return (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y(f"{y}:Q", title=y_title, scale=scale),
            color=alt.Color(
                "split:N",
                title="Split",
                scale=alt.Scale(
                    domain=list(SPLIT_COLORS), range=list(SPLIT_COLORS.values())
                ),
            ),
            tooltip=["round", "split", alt.Tooltip(f"{y}:Q", format=".1f"), "messages"],
        )
        .properties(title=title, height=280)
    )


def evasion_by_split(summary: pd.DataFrame):
    """The headline: evasion per round, training against held out.

    The gap between the two lines is the question the held-out split exists to
    answer — whether the policy learned to evade or learned these 150 subjects.
    """
    if summary.empty:
        st.info("no scored rounds yet")
        return

    long = summary.melt(
        id_vars=["round", "split", "messages"],
        value_vars=[c for c in ("evasion_rate", "asr_at_n") if c in summary],
        var_name="metric",
        value_name="percent",
    )
    chart = (
        alt.Chart(long)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y("percent:Q", title="Percent", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(
                    domain=list(SPLIT_COLORS), range=list(SPLIT_COLORS.values())
                ),
            ),
            strokeDash=alt.StrokeDash("metric:N", title="Metric"),
            tooltip=[
                "round",
                "split",
                "metric",
                alt.Tooltip("percent:Q", format=".1f"),
                "messages",
            ],
        )
        .properties(height=320)
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [50]}))
        .mark_rule(strokeDash=[4, 4], color="#888")
        .encode(y="y:Q")
    )
    st.altair_chart(chart + rule, use_container_width=True)


def guardrail(summary: pd.DataFrame, column: str, title: str, y_title: str, domain=None):
    """One guardrail metric per round and split, round 0 marked as the baseline."""
    if summary.empty or column not in summary or summary[column].isna().all():
        st.info(f"{title.lower()} not measured for this run")
        return

    data = summary.dropna(subset=[column])
    chart = _line(data, column, title, y_title, domain)

    baseline = data[data["round"] == data["round"].min()]
    marks = (
        alt.Chart(baseline)
        .mark_rule(strokeDash=[2, 3], opacity=0.6)
        .encode(
            y=f"{column}:Q",
            color=alt.Color(
                "split:N",
                scale=alt.Scale(
                    domain=list(SPLIT_COLORS), range=list(SPLIT_COLORS.values())
                ),
                legend=None,
            ),
        )
    )
    st.altair_chart(chart + marks, use_container_width=True)


def score_distribution(df: pd.DataFrame, threshold: float = 0.5):
    """Where the scores sit, per round — a mean can hide a bimodal split."""
    if df.empty or "score" not in df:
        st.info("no scored messages yet")
        return

    chart = (
        alt.Chart(df)
        .transform_density("score", groupby=["round"], as_=["score", "density"], extent=[0, 1])
        .mark_area(opacity=0.45)
        .encode(
            x=alt.X("score:Q", title="ScamLLM score (higher = evaded)"),
            y=alt.Y("density:Q", title="Density", stack=None),
            color=alt.Color("round:O", title="Round"),
            tooltip=["round"],
        )
        .properties(height=260)
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [threshold]}))
        .mark_rule(strokeDash=[4, 4], color="#888")
        .encode(x="x:Q")
    )
    st.altair_chart(chart + rule, use_container_width=True)


def breakdown_bars(df: pd.DataFrame, group: str, value: str = "evaded"):
    """Evasion by category or generator, per round."""
    if df.empty or group not in df:
        st.info(f"no {group} data yet")
        return

    data = (
        df.groupby(["round", group])[value]
        .mean()
        .mul(100)
        .reset_index(name="percent")
    )
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{group}:N", title=group.title()),
            y=alt.Y("percent:Q", title="Evasion rate (%)"),
            color=alt.Color("round:O", title="Round"),
            xOffset="round:O",
            tooltip=[group, "round", alt.Tooltip("percent:Q", format=".1f")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)
