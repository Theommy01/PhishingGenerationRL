"""The three questions, answered for one run.

Leads with paired, per-prompt change rather than a difference of two means.
Every round generates from the identical subject list, so each prompt is its own
control — which matters here because the SFT baseline is uneven, and a marginal
mean mixes real movement with how hard the prompts happen to be.
"""

import pandas as pd
import streamlit as st

from dashboard import charts, data
from metrics import paired


def _delta(summary, split: str, column: str):
    """First round to last, for one split — the shape of every headline here."""
    rows = summary[(summary["split"] == split) & summary[column].notna()]
    if rows.empty:
        return None, None
    first, last = rows.iloc[0][column], rows.iloc[-1][column]
    return last, (last - first if len(rows) > 1 else None)


def _paired_headline(frame: pd.DataFrame, rounds) -> None:
    """Flip table and paired deltas, per prompt, against round 0."""
    columns = st.columns([1, 1, 2])
    baseline = columns[0].selectbox("baseline", rounds, index=0, key="paired-baseline")
    later = [r for r in rounds if r > baseline] or [rounds[-1]]
    target = columns[1].selectbox("compared with", later, index=len(later) - 1, key="paired-round")
    split = columns[2].radio(
        "split", ["train", "holdout"], horizontal=True, key="paired-split"
    )

    table = paired.flip_table(frame, target, baseline, split)
    if not table:
        st.info("not enough scored rounds for a paired comparison yet")
        return

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Prompts that changed** — did *any* of the n samples evade?")
        grid = pd.DataFrame(
            {
                f"round {target}: evaded": [table["both"], table["gained"]],
                f"round {target}: blocked": [table["lost"], table["neither"]],
            },
            index=[f"round {baseline}: evaded", f"round {baseline}: blocked"],
        )
        st.dataframe(grid, use_container_width=True)

        verdict = f"**{table['gained']} gained, {table['lost']} lost** (net {table['net']:+d})"
        if "p_value" in table:
            verdict += f" — McNemar p = {table['p_value']:.4f} on {table['discordant']} discordant"
        st.markdown(verdict)
        st.caption(
            "Only the off-diagonal counts: a prompt that evaded both times, or "
            "neither time, says nothing about the change."
        )

    with right:
        st.markdown("**Paired change per prompt**")
        report = paired.report(
            frame, target, baseline, split,
            outcomes=["evasion_rate", "mean_score", "url_ok", "attachment_ok", "cos_subject"],
        )
        if report.empty:
            st.info("no paired outcomes available")
        else:
            display = report[
                ["outcome", "median_delta", "improved", "worsened", "unchanged", "prompts"]
            ].copy()
            display["median_delta"] = display["median_delta"].round(3)
            st.dataframe(display, hide_index=True, use_container_width=True)
            st.caption(
                "Median rather than mean, and counts rather than magnitudes: a "
                "baseline that fails on some prompts moves a mean around far "
                "more than it moves a sign."
            )

    did = paired.difference_in_differences(frame, target, baseline)
    if did:
        st.markdown(
            f"**Generalisation** — median paired change in evasion rate: "
            f"train {did['train']['median']:+.3f} "
            f"({did['train']['prompts']} prompts), held out "
            f"{did['holdout']['median']:+.3f} ({did['holdout']['prompts']}). "
            f"Gap {did['gap_median']:+.3f}."
        )
        st.caption(
            "The gap is the memorisation estimate: how much more the policy "
            "improved on subjects it trained on than on subjects it did not."
        )

    joined = paired.paired(frame, target, baseline, split)
    with st.expander("the prompts that moved most — read these first"):
        outcome = st.selectbox(
            "outcome", ["mean_score", "evasion_rate", "cos_subject"], key="movers-outcome"
        )
        best, worst = st.columns(2)
        with best:
            st.caption("largest gain")
            st.dataframe(paired.movers(joined, outcome), hide_index=True)
        with worst:
            st.caption("largest loss")
            st.dataframe(paired.movers(joined, outcome, worst=True), hide_index=True)


def render(run_id: int) -> None:
    summary = data.split_summary(run_id)
    frame = data.scored(data.messages_frame(run_id))

    if summary.empty:
        st.info(
            "Nothing scored yet. Messages are stored as they are generated and "
            "scored when the round finishes, so text is readable in the Messages "
            "tab before any number appears here."
        )
        return

    # -- headline ----------------------------------------------------------
    st.subheader("Did evasion improve?")

    rounds = sorted(summary["round"].unique())
    if len(rounds) > 1:
        _paired_headline(frame, rounds)
        st.divider()
        st.markdown("**Round by round**")
    columns = st.columns(4)
    for column, (split, metric, label) in zip(
        columns,
        [
            ("train", "evasion_rate", "Evasion (train)"),
            ("holdout", "evasion_rate", "Evasion (held out)"),
            ("train", "asr_at_n", "ASR@n (train)"),
            ("holdout", "asr_at_n", "ASR@n (held out)"),
        ],
    ):
        value, change = _delta(summary, split, metric)
        column.metric(
            label,
            "—" if value is None else f"{value:.1f}%",
            None if change is None else f"{change:+.1f} pts",
        )

    mode = charts.figure_toggle("evasion-mode")
    if mode == "interactive":
        charts.evasion_by_split(summary)
    else:
        from visualisation.charts import plot_round_trajectory

        charts.thesis_figure(
            plot_round_trajectory,
            data.trajectory(run_id),
            name=f"run{run_id}_round_trajectory",
            download_label="round_trajectory",
        )
    st.caption(
        "Solid: evasion rate, per message. Dashed: ASR@n, the fraction of "
        "prompts where at least one of the n samples evaded. A gap that widens "
        "between train and held out is the policy fitting these subjects rather "
        "than learning to evade."
    )

    st.divider()

    # -- guardrail 1: degeneration -----------------------------------------
    st.subheader("Is it degenerating?")
    left, right = st.columns(2)
    with left:
        charts.guardrail(
            summary,
            "logratio_per_token",
            "Log ratio against the SFT baseline",
            "Per token (median)",
        )
        st.caption(
            "Measured on each round's own text, always against the pinned SFT "
            "checkpoint; round 0 is the anchor, so it is zero by construction. "
            "This is a **log ratio, not a KL divergence** — negative values mean "
            "the policy assigns its own output lower probability than SFT does, "
            "which is likelihood displacement and a real thing to watch for."
        )
    with right:
        charts.score_distribution(frame)
        st.caption(
            "A rising mean can hide a split distribution: a few messages "
            "evading completely reads the same as all of them improving."
        )

    duplicates = summary[["round", "split", "duplicates"]].pivot(
        index="round", columns="split", values="duplicates"
    )
    st.caption("Exact-duplicate bodies per round (mode collapse, if it climbs):")
    st.dataframe(duplicates, use_container_width=True)

    st.divider()

    # -- guardrail 2: instruction following ---------------------------------
    st.subheader("Does it still follow the prompt?")
    mode = charts.figure_toggle("instruction-mode")
    if mode == "interactive":
        left, right = st.columns(2)
        with left:
            charts.guardrail(
                summary, "url_compliance", "URL flag followed", "Percent", [0, 100]
            )
        with right:
            charts.guardrail(
                summary,
                "attachment_compliance",
                "Attachment flag followed",
                "Percent",
                [0, 100],
            )
        charts.guardrail(
            summary, "cos_subject", "Similarity to the subject line", "Cosine (%)"
        )
    else:
        from visualisation.charts import plot_instruction_following

        train = summary[summary["split"] == "train"].rename(
            columns={"url_compliance": "url_ok", "attachment_compliance": "attachment_ok"}
        )
        charts.thesis_figure(
            plot_instruction_following,
            train,
            name=f"run{run_id}_instruction_following",
            download_label="instruction_following",
        )
    st.caption(
        "Compliance is two-sided: producing a <URL> that was not asked for "
        "counts against it, as does omitting one that was. The dotted line is "
        "the round-0 level — the question is movement from there, not the "
        "absolute value."
    )

    st.divider()
    st.subheader("By category and generator")
    left, right = st.columns(2)
    with left:
        charts.breakdown_bars(frame, "category")
    with right:
        charts.breakdown_bars(frame, "generator")
