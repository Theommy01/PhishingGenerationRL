"""Artefact or genuine improvement — and the checks that decide if it is readable.

The scatter is the point of this tab. Each dot is a prompt, positioned by how
much it improved against the detector in the loop (x) and against a detector
that was never in it (y). The quadrants are the decomposition:

    upper right   improved against both  -> transferred
    lower right   improved in loop only  -> detector-specific
    upper left    only the held-out one moved -> noise
    lower left    nothing moved
"""

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import data
from metrics import paired, transfer


def _delta_frame(df, in_loop, held_out, target, baseline, split):
    """Per prompt: the paired score change under each detector."""
    frames = {}
    for detector in (in_loop, held_out):
        column = f"{transfer.SCORE_PREFIX}{detector}"
        if column not in df:
            return pd.DataFrame()
        frame = df.copy()
        frame["score"] = frame[column]
        frame = frame.dropna(subset=["score"])
        joined = paired.paired(frame, target, baseline, split)
        if joined.empty or "mean_score_delta" not in joined:
            return pd.DataFrame()
        frames[detector] = joined

    out = pd.DataFrame(
        {
            "in_loop_delta": frames[in_loop]["mean_score_delta"],
            "held_out_delta": frames[held_out]["mean_score_delta"],
        }
    ).dropna()
    for column in ("subject_text", "category", "generator", "split"):
        if column in frames[in_loop]:
            out[column] = frames[in_loop][column]
    return out.reset_index()


def render(run_id: int) -> None:
    store = data.get_store()
    scored = store.scored_detectors(run_id)
    # Which detector supplied the reward is a property of the run, not a
    # constant: its verdict is the message's own `score`, so reading the wrong
    # name off it would label one detector's scores with another's.
    reward = data.run_config(run_id).get("detector") or "scamllm"
    df = transfer.with_detector_columns(data.messages_frame(run_id), in_loop=reward)
    present = transfer.detectors_present(df)

    st.caption(
        "The detector in the loop supplies the reward, so it cannot testify "
        "about itself: a policy that learned its quirks and one that learned to "
        "write better phishing both show the same rising evasion. A detector "
        "that was never optimised against can tell them apart."
    )

    if len(present) < 2:
        # suggest a detector this run does not already have — which is not
        # always bert-phishing, since it can be the one in the loop
        suggestion = "scamllm" if reward != "scamllm" else "bert-phishing"
        st.warning(
            f"Only one detector has scored this run ({present or 'none'}). "
            "Add another with:\n\n"
            f"```\npython -m detectors.backfill {run_id} --detector {suggestion}\n```\n"
            "It reads stored bodies only — no GPU, no regeneration."
        )
        return

    columns = st.columns(4)
    in_loop = columns[0].selectbox(
        "in the loop", present, index=present.index(reward) if reward in present else 0
    )
    others = [d for d in present if d != in_loop]
    held_out = columns[1].selectbox("held out", others)

    rounds = sorted(df["round"].dropna().unique())
    if len(rounds) < 2:
        st.info(
            "Only one round so far. Transfer is a comparison of *changes*, so it "
            "needs a trained round to compare against the baseline."
        )
        _preconditions(df, in_loop, held_out, int(rounds[0]) if rounds else 0)
        return

    baseline = columns[2].selectbox("baseline", rounds, index=0, key="transfer-baseline")
    later = [r for r in rounds if r > baseline] or [rounds[-1]]

    # A round is comparable only where the held-out detector has scored it too.
    # During a backfill the latest round often is not scored yet, so default to
    # the latest one that IS rather than to a round that would render empty.
    held_col = f"{transfer.LABEL_PREFIX}{held_out}"
    scored_rounds = set()
    if held_col in df:
        scored_rounds = set(df.loc[df[held_col].notna(), "round"].unique())
    comparable = [r for r in later if r in scored_rounds]
    default = later.index(comparable[-1]) if comparable else len(later) - 1
    target = columns[3].selectbox("compared with", later, index=default)
    if target not in scored_rounds:
        st.warning(
            f"Round {target} has not been scored by **{held_out}** yet, so the "
            "transfer comparison is unavailable for it. Score it with "
            f"`python -m detectors.backfill {run_id} --detector {held_out}`. "
            + (f"Rounds already comparable: {sorted(comparable)}." if comparable else "")
        )
    split = st.radio("split", [None, "train", "holdout"],
                     format_func=lambda s: s or "both", horizontal=True, key="transfer-split")

    result = transfer.report(df, target, in_loop, held_out, baseline, split)
    correlation = result.get("correlation") or {}
    levels = result.get("level_correlation") or {}

    # -- the headline -------------------------------------------------------
    st.subheader("Did the gain transfer?")
    if not correlation:
        st.info("not enough paired prompts scored by both detectors yet")
    else:
        left, middle, right = st.columns(3)
        ci = correlation.get("ci95") or (None, None)
        left.metric(
            "Rank correlation of changes",
            f"{correlation['spearman']:.3f}",
            help="Spearman of per-prompt paired score deltas. Threshold-free.",
        )
        if ci[0] is not None:
            left.caption(f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}] · p = {correlation['p_value']:.3g}")
        middle.metric(
            f"Median change, {in_loop}", f"{correlation[f'median_delta_{in_loop}']:+.3f}"
        )
        right.metric(
            f"Median change, {held_out}", f"{correlation[f'median_delta_{held_out}']:+.3f}"
        )

        if levels:
            ceiling = levels["spearman"]
            st.info(
                f"Read against the ceiling: these two detectors rank the *same "
                f"messages* at ρ = {ceiling:.2f} on the baseline round. Two "
                "detectors that only agree that much about phishing in general "
                "cannot agree more than that about changes in it, so "
                f"{correlation['spearman']:.2f} should be read against "
                f"{ceiling:.2f}, not against 1.0."
            )

    # -- the scatter --------------------------------------------------------
    points = _delta_frame(df, in_loop, held_out, target, baseline, split)
    if not points.empty:
        chart = (
            alt.Chart(points)
            .mark_circle(size=70, opacity=0.65)
            .encode(
                x=alt.X("in_loop_delta:Q", title=f"change against {in_loop} (in the loop)"),
                y=alt.Y("held_out_delta:Q", title=f"change against {held_out} (held out)"),
                color=alt.Color("split:N") if "split" in points else alt.value("#3498db"),
                tooltip=[
                    c for c in ("subject_text", "category", "generator", "split",
                                "in_loop_delta", "held_out_delta") if c in points
                ],
            )
            .properties(height=420)
        )
        rules = (
            alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color="#888", strokeDash=[4, 4])
        )
        st.altair_chart(
            chart + rules.encode(x="v:Q") + rules.encode(y="v:Q"), use_container_width=True
        )
        st.caption(
            "Upper right: improved against both — transferred. Lower right: "
            "improved only against the detector it trained on — specific to it. "
            "Upper left: the held-out detector moved on its own, which is the "
            "noise floor for reading the other quadrants."
        )

    # -- secondary, and the checks -----------------------------------------
    with st.expander("label-based decomposition (threshold-dependent)"):
        decomposition = result.get("decomposition") or {}
        if not decomposition:
            st.info("not available")
        else:
            st.dataframe(
                pd.DataFrame(
                    {
                        f"{held_out}: improved": [decomposition["transferred"], decomposition["noise"]],
                        f"{held_out}: not": [decomposition["artefact"], decomposition["nothing"]],
                    },
                    index=[f"{in_loop}: improved", f"{in_loop}: not"],
                ),
                use_container_width=True,
            )
            # every one of these is None in a real case: no prompt gained, no
            # headroom left, or the two detectors never both labelled a round
            def show(value, spec=".2f"):
                return "n/a" if value is None else format(value, spec)

            fraction = decomposition.get("transfer_fraction")
            adjusted = decomposition.get("transfer_fraction_with_headroom")
            st.markdown(
                f"transfer fraction **{show(fraction)}**"
                + (f" (with headroom: **{show(adjusted)}**)" if adjusted is not None else "")
                + f" · {decomposition['held_out_ceiling']} of the gained prompts were "
                "already failing on the held-out detector at the baseline, so they "
                "had nowhere to improve."
            )
            if fraction is None:
                st.caption(
                    "No prompt improved against the in-loop detector between "
                    "these rounds, so there is no gain to decompose."
                )
            st.warning(
                "This reading depends on where both thresholds sit, and the two "
                "detectors agree on labels at kappa = "
                f"{show(decomposition.get('baseline_kappa'))}. The rank "
                "correlation above does not depend on either threshold, which is "
                "why it leads."
            )

    _preconditions(df, in_loop, held_out, baseline)


def _preconditions(df, in_loop, held_out, baseline) -> None:
    with st.expander("preconditions — check these before believing any of it"):
        ranges = pd.DataFrame(
            [
                transfer.dynamic_range(df, detector, baseline)
                for detector in (in_loop, held_out)
            ]
        )
        st.markdown("**Dynamic range on this text**")
        st.dataframe(ranges, hide_index=True, use_container_width=True)
        st.caption(
            "A detector that says the same thing about everything has no "
            "headroom. Plausible here: the corpus is entity-anonymised, so a "
            "detector fitted on real email never sees the links it keys on."
        )

        agreement = transfer.baseline_agreement(df, in_loop, held_out, baseline)
        levels = transfer.level_correlation(df, in_loop, held_out, baseline)
        if agreement:
            left, right = st.columns(2)
            left.metric("Label agreement (kappa)", f"{agreement['kappa']:.3f}")
            left.caption(
                f"{in_loop} calls {agreement[f'{in_loop}_evaded']:.0%} evaded, "
                f"{held_out} {agreement[f'{held_out}_evaded']:.0%} — at the same "
                "arbitrary 0.5 cut."
            )
            if levels:
                right.metric("Rank correlation of levels", f"{levels['spearman']:.3f}")
                right.caption(
                    "Poor label agreement with a decent rank correlation is a "
                    "calibration difference, not a disagreement about the text."
                )
