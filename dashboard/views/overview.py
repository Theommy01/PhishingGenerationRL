"""The three questions, answered for one run."""

import streamlit as st

from dashboard import charts, data


def _delta(summary, split: str, column: str):
    """First round to last, for one split — the shape of every headline here."""
    rows = summary[(summary["split"] == split) & summary[column].notna()]
    if rows.empty:
        return None, None
    first, last = rows.iloc[0][column], rows.iloc[-1][column]
    return last, (last - first if len(rows) > 1 else None)


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
            "kl_per_token",
            "KL from the SFT baseline",
            "Nats per token",
        )
        st.caption(
            "Measured on each round's own text, always against the pinned SFT "
            "checkpoint. Round 0 is the anchor, so it is zero by construction."
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
