"""The three questions, answered for one run.

Leads with paired, per-prompt change rather than a difference of two means.
Every round generates from the identical subject list, so each prompt is its own
control — which matters here because the SFT baseline is uneven, and a marginal
mean mixes real movement with how hard the prompts happen to be.
"""

import pandas as pd
import streamlit as st

from dashboard import charts, data
from loop.store import TRAIN_SPLIT
from metrics import paired


def _delta(summary, split: str, column: str):
    """Latest scored round for one split, and its change from the first.

    Returns (value, change, latest_round, first_round) — the round numbers come
    back so the tiles can say which rounds they are talking about rather than
    leaving the reader to assume.
    """
    rows = summary[(summary["split"] == split) & summary[column].notna()]
    if rows.empty:
        return None, None, None, None
    first, last = rows.iloc[0], rows.iloc[-1]
    change = (last[column] - first[column]) if len(rows) > 1 else None
    return last[column], change, int(last["round"]), int(first["round"])


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
    shown = []
    for column, (split, metric, label) in zip(
        columns,
        [
            ("train", "evasion_rate", "Evasion (train)"),
            ("holdout", "evasion_rate", "Evasion (held out)"),
            ("train", "asr_at_n", "ASR@n (train)"),
            ("holdout", "asr_at_n", "ASR@n (held out)"),
        ],
    ):
        value, change, latest, first = _delta(summary, split, metric)
        if latest is not None:
            shown.append((latest, first))
        column.metric(
            label,
            "—" if value is None else f"{value:.1f}%",
            None if change is None else f"{change:+.1f} pts",
            help=(
                f"Round {latest}. The change is against round {first}."
                if latest is not None
                else "no scored round for this split yet"
            ),
        )

    if shown:
        latest = max(r for r, _ in shown)
        first = min(f for _, f in shown)
        st.caption(
            f"Figures are **round {latest}** — the latest scored round; the "
            f"change beneath each is against **round {first}**. These are "
            "marginal means over all of a round's messages: the paired, "
            "per-prompt view below is the one that says whether a difference is "
            "real."
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
        "**Solid: evasion rate** — the share of individual messages that got "
        "through. **Dashed: ASR@n** — the share of *prompts* where at least one "
        "of the n samples got through, which is the attacker who generates n and "
        "keeps whichever lands. ASR@n is at least the evasion rate by "
        "construction, so the dashed line always sits above the solid one; it is "
        "also only comparable at a fixed n. A gap that widens between train and "
        "held out is the policy fitting these subjects rather than learning to "
        "evade."
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
    st.caption(
        "The pooled compliance rate is misleading: omitting a placeholder "
        "nobody asked for is free, and most prompts ask for neither, so the "
        "pooled figure stays flat while the side that costs the model something "
        "— emitting when asked — can erode. These show only the requested "
        "subset, which is the honest half."
    )
    left, right = st.columns(2)
    with left:
        charts.emit_when_asked(paired.compliance_when_requested(frame, "url", TRAIN_SPLIT), "URL")
    with right:
        charts.emit_when_asked(
            paired.compliance_when_requested(frame, "attachment", TRAIN_SPLIT), "attachment"
        )
    charts.guardrail(
        summary, "cos_subject", "Similarity to the subject line", "Cosine (%)"
    )

    st.divider()
    st.subheader("Does evasion improve for emails that must carry a URL / attachment?")
    st.caption(
        "The hardest test of genuine evasion: a prompt that requests a URL must "
        "keep a phishing signal and still slip past the detector, rather than "
        "evading by having nothing to flag. If the requested subset improves — "
        "and improves while the placeholder is kept — the gain is real quality, "
        "not signal-stripping."
    )
    left, right = st.columns(2)
    with left:
        charts.trajectory_by_flag(
            paired.trajectory_by(frame, "url_requested", "evasion_rate", TRAIN_SPLIT),
            "url_requested", "evasion_rate", "Evasion rate (URL requested vs not)",
        )
    with right:
        charts.trajectory_by_flag(
            paired.trajectory_by(frame, "attachment_requested", "evasion_rate", TRAIN_SPLIT),
            "attachment_requested", "evasion_rate", "Evasion rate (attachment requested vs not)",
        )

    st.divider()
    st.subheader("By category and generator")
    left, right = st.columns(2)
    with left:
        charts.breakdown_bars(frame, "category")
    with right:
        charts.breakdown_bars(frame, "generator")
