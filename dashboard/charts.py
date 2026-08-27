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

# Spelled out in the legend rather than left as column names, and pinned to a
# dash pattern below. Vega sorts a nominal domain alphabetically, so leaving it
# implicit handed the solid line to `asr_at_n` while the caption claimed it for
# the evasion rate — a mapping the chart never actually controlled.
METRIC_LABELS = {
    "evasion_rate": "evasion rate (per message)",
    "asr_at_n": "ASR@n (per prompt)",
}
SOLID, DASHED = [1, 0], [6, 4]


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
    long["metric"] = long["metric"].map(METRIC_LABELS)
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
            strokeDash=alt.StrokeDash(
                "metric:N",
                title="Metric",
                # explicit, so the legend cannot drift from what is drawn
                scale=alt.Scale(
                    domain=[METRIC_LABELS["evasion_rate"], METRIC_LABELS["asr_at_n"]],
                    range=[SOLID, DASHED],
                ),
            ),
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


# Diverging about the 0.5 threshold: caught below, evaded above, and the pale
# shade on each side for the band where the detector is barely committed.
BAND_LABELS = {
    "caught": "caught (0 – .25)",
    "caught_weak": "caught, weakly (.25 – .5)",
    "evaded_weak": "evaded, weakly (.5 – .75)",
    "evaded": "evaded (.75 – 1)",
}
BAND_COLORS = ["#2c7fb8", "#a6cee3", "#fdbf6f", "#e67e22"]


def score_bands(summary: pd.DataFrame, split: str = "train"):
    """The share of messages in each fixed score band, stacked, per round."""
    if summary.empty or "score_bands" not in summary:
        st.info("no scored messages yet")
        return

    rows = summary[(summary["split"] == split) & summary["score_bands"].notna()]
    if rows.empty:
        st.info("no scored messages yet")
        return

    data = pd.DataFrame(
        [
            {"round": row["round"], "band": BAND_LABELS[name], "percent": percent}
            for _, row in rows.iterrows()
            for name, percent in (row["score_bands"] or {}).items()
        ]
    )
    chart = (
        alt.Chart(data)
        .mark_area()
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y("percent:Q", title="Messages (%)", stack="zero"),
            color=alt.Color(
                "band:N",
                title="Detector score",
                # the order the bands sit in, not alphabetical
                sort=list(BAND_LABELS.values()),
                scale=alt.Scale(domain=list(BAND_LABELS.values()), range=BAND_COLORS),
            ),
            order=alt.Order("color_band_sort_index:Q"),
            tooltip=["round", "band", alt.Tooltip("percent:Q", format=".1f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


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


def emit_when_asked(store_frame: pd.DataFrame, field: str):
    """Emit-when-asked compliance per round — the honest half of compliance.

    `store_frame` is one row per round with a `compliance` column, as
    metrics.paired.compliance_when_requested returns.
    """
    if store_frame.empty:
        st.info(f"no prompts requested a {field} in this run")
        return
    chart = (
        alt.Chart(store_frame)
        .mark_line(point=True, strokeWidth=2.5, color=PALETTE_ORANGE[1])
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y("compliance:Q", title="% emitting when asked", scale=alt.Scale(domain=[0, 100])),
            tooltip=["round", alt.Tooltip("compliance:Q", format=".1f"), "prompts"],
        )
        .properties(height=260, title=f"{field} placeholder emitted when requested")
    )
    baseline = store_frame.iloc[[0]]
    mark = alt.Chart(baseline).mark_rule(strokeDash=[2, 3], opacity=0.6, color=PALETTE_ORANGE[1]).encode(y="compliance:Q")
    st.altair_chart(chart + mark, use_container_width=True)


def trajectory_by_flag(traj: pd.DataFrame, group: str, outcome: str, title: str):
    """One line per value of a per-prompt flag — e.g. evasion split by url_requested."""
    if traj.empty:
        st.info("not enough data yet")
        return
    plot = traj.copy()
    plot[group] = plot[group].map({True: "requested", False: "not requested"}).fillna(plot[group].astype(str))
    chart = (
        alt.Chart(plot)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y(f"{outcome}:Q", title=title, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(f"{group}:N", title=group.replace("_", " ")),
            tooltip=["round", group, alt.Tooltip(f"{outcome}:Q", format=".3f"), "prompts"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def evasion_kept_vs_dropped(traj: pd.DataFrame, field: str):
    """Per-message evasion split by whether the placeholder was kept.

    `traj` is metrics.paired.evasion_by_presence output: rows of
    (round, kept in {"kept","dropped"}, evasion_rate, messages).
    """
    if traj.empty:
        st.info(f"no prompts requested a {field} in this run")
        return
    chart = (
        alt.Chart(traj)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("round:O", title="Round"),
            y=alt.Y("evasion_rate:Q", title="Evasion rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "kept:N",
                title=f"<{field.upper()}>",
                scale=alt.Scale(domain=["kept", "dropped"],
                                range=[PALETTE_ORANGE[1], PALETTE_ORANGE[0]]),
            ),
            tooltip=["round", "kept", alt.Tooltip("evasion_rate:Q", format=".3f"), "messages"],
        )
        .properties(height=300, title=f"Evasion when the {field} placeholder is kept vs dropped")
    )
    st.altair_chart(chart, use_container_width=True)
