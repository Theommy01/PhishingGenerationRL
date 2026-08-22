"""Round-over-round change, paired on the prompt.

Every round generates from the identical subject list, so each prompt is its own
control and the comparison can be paired. That matters more here than it usually
would: the SFT baseline is uneven — some prompts produce corpus regurgitation
rather than an on-topic email — and a marginal mean over rounds mixes real
movement with how hard the prompts happen to be. A paired delta cancels the
difficulty, because the same prompt appears on both sides.

Three outcomes per prompt, in increasing sensitivity:

    any_evaded     did at least one of the n samples get through? The ASR@n
                   framing, and the one that yields a flip table.
    evasion_rate   k/n. Sees partial movement (1/4 -> 3/4) that the binary
                   scores as no change.
    mean_score     the ScamLLM probability itself, no threshold. Sees pressure
                   below the 0.5 line that neither of the others can.

Reported together on purpose. Agreement across all three is a robust claim;
movement in the means alone is usually a composition artefact.

The statistics are deliberately sign-based — flip counts, medians, the fraction
of prompts that improved. A median and a sign survive a baseline where some
prompts produce nonsense; a mean does not.
"""

from typing import Dict, List, Optional

import pandas as pd

from metrics import config

# Per-prompt outcomes and how a prompt's samples collapse into one number.
OUTCOMES = {
    "any_evaded": ("evaded", "any"),
    "evasion_rate": ("evaded", "mean"),
    "mean_score": ("score", "mean"),
    "url_ok": ("url_ok", "mean"),
    "attachment_ok": ("attachment_ok", "mean"),
    "cos_subject": ("cos_subject", "mean"),
    "logratio_per_token": ("logratio_per_token", "mean"),
}

# Outcomes where a higher number is the policy doing better at evading. The
# guardrails are not in here: for those, "improved" is not the point — movement
# away from the baseline in either direction is what matters.
HIGHER_IS_MORE_EVASION = {"any_evaded", "evasion_rate", "mean_score"}


def per_prompt(
    df: pd.DataFrame,
    round_index: int,
    split: Optional[str] = None,
    threshold: float = config.SAFE_THRESHOLD,
) -> pd.DataFrame:
    """Collapse one round's messages to one row per prompt."""
    rows = df[df["round"] == round_index]
    if split is not None and "split" in rows:
        rows = rows[rows["split"] == split]
    if rows.empty:
        return pd.DataFrame()

    rows = rows.copy()
    if "evaded" not in rows and "score" in rows:
        rows["evaded"] = rows["score"] >= threshold

    aggregations = {
        name: pd.NamedAgg(column=column, aggfunc=how)
        for name, (column, how) in OUTCOMES.items()
        if column in rows
    }
    aggregations["samples"] = pd.NamedAgg(column="body", aggfunc="size")

    out = rows.groupby("prompt_id").agg(**aggregations)
    for column in ("category", "generator", "subject_text", "split"):
        if column in rows:
            out[column] = rows.groupby("prompt_id")[column].first()
    return out


def paired(
    df: pd.DataFrame,
    round_index: int,
    baseline: int = 0,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """Per-prompt change from `baseline` to `round_index`.

    Inner-joined on prompt_id, so a prompt missing from either round is dropped
    rather than counted as a change.
    """
    before = per_prompt(df, baseline, split)
    after = per_prompt(df, round_index, split)
    if before.empty or after.empty:
        return pd.DataFrame()

    shared = [name for name in OUTCOMES if name in before and name in after]
    joined = before[shared].join(
        after[shared], lsuffix="_before", rsuffix="_after", how="inner"
    )
    for name in shared:
        # any_evaded is boolean, and numpy refuses to subtract those. Cast
        # first, so its delta is -1 / 0 / +1 — lost, unchanged, gained — which
        # is the same event the flip table counts.
        before_values = joined[f"{name}_before"].astype(float)
        after_values = joined[f"{name}_after"].astype(float)
        joined[f"{name}_delta"] = after_values - before_values

    for column in ("category", "generator", "subject_text", "split"):
        if column in after:
            joined[column] = after[column]
    return joined


def flip_table(
    df: pd.DataFrame,
    round_index: int,
    baseline: int = 0,
    split: Optional[str] = None,
    outcome: str = "any_evaded",
) -> Dict:
    """The 2x2 of a binary outcome before and after, with McNemar's test.

    Only the discordant cells carry information: prompts that evaded both times
    or neither time say nothing about the change. The test is an exact binomial
    on those two counts, which is the right one at these sample sizes.
    """
    joined = paired(df, round_index, baseline, split)
    if joined.empty or f"{outcome}_before" not in joined:
        return {}

    before = joined[f"{outcome}_before"].astype(bool)
    after = joined[f"{outcome}_after"].astype(bool)

    gained = int((~before & after).sum())
    lost = int((before & ~after).sum())
    kept = int((before & after).sum())
    never = int((~before & ~after).sum())

    result = {
        "outcome": outcome,
        "baseline": baseline,
        "round": round_index,
        "prompts": int(len(joined)),
        "gained": gained,
        "lost": lost,
        "both": kept,
        "neither": never,
        "net": gained - lost,
    }

    discordant = gained + lost
    if discordant:
        from scipy.stats import binomtest

        result["p_value"] = binomtest(gained, discordant, 0.5).pvalue
        result["discordant"] = discordant
    return result


def summarise(joined: pd.DataFrame, outcome: str) -> Dict:
    """Sign-based summary of one outcome's paired deltas."""
    column = f"{outcome}_delta"
    if joined.empty or column not in joined:
        return {}

    deltas = joined[column].dropna()
    if deltas.empty:
        return {}

    improved = int((deltas > 0).sum())
    worsened = int((deltas < 0).sum())
    result = {
        "outcome": outcome,
        "prompts": int(len(deltas)),
        "median_delta": float(deltas.median()),
        "iqr": (float(deltas.quantile(0.25)), float(deltas.quantile(0.75))),
        "mean_delta": float(deltas.mean()),
        "improved": improved,
        "worsened": worsened,
        "unchanged": int((deltas == 0).sum()),
        "fraction_improved": improved / len(deltas),
    }

    if improved + worsened:
        from scipy.stats import wilcoxon

        try:
            result["p_value"] = float(wilcoxon(deltas, zero_method="wilcox").pvalue)
        except ValueError:
            result["p_value"] = None
    return result


def report(
    df: pd.DataFrame,
    round_index: int,
    baseline: int = 0,
    split: Optional[str] = None,
    outcomes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """One row per outcome: median delta, how many prompts moved, and which way."""
    joined = paired(df, round_index, baseline, split)
    if joined.empty:
        return pd.DataFrame()

    names = outcomes or [name for name in OUTCOMES if f"{name}_delta" in joined]
    rows = [summarise(joined, name) for name in names]
    return pd.DataFrame([row for row in rows if row])


def difference_in_differences(
    df: pd.DataFrame,
    round_index: int,
    baseline: int = 0,
    outcome: str = "evasion_rate",
) -> Dict:
    """Held-out improvement minus training improvement, both paired.

    The memorisation estimate. If the policy learned to evade in general, the
    two move together; if it fitted these subjects, the training split moves
    further. Paired on both sides, so prompt difficulty cancels within each
    split — though the two splits still hold different prompts, which is why
    this compares *changes* rather than levels.
    """
    changes = {}
    for split in ("train", "holdout"):
        joined = paired(df, round_index, baseline, split)
        column = f"{outcome}_delta"
        if joined.empty or column not in joined:
            return {}
        changes[split] = {
            "median": float(joined[column].median()),
            "mean": float(joined[column].mean()),
            "prompts": int(len(joined)),
        }

    return {
        "outcome": outcome,
        "round": round_index,
        "baseline": baseline,
        "train": changes["train"],
        "holdout": changes["holdout"],
        "gap_median": changes["train"]["median"] - changes["holdout"]["median"],
        "gap_mean": changes["train"]["mean"] - changes["holdout"]["mean"],
    }


def movers(joined: pd.DataFrame, outcome: str, n: int = 5, worst: bool = False):
    """The prompts that moved most — what to read when a number surprises you."""
    column = f"{outcome}_delta"
    if joined.empty or column not in joined:
        return pd.DataFrame()

    ordered = joined.sort_values(column, ascending=worst)
    keep = [c for c in ("subject_text", "category", "generator", "split") if c in ordered]
    return ordered[keep + [f"{outcome}_before", f"{outcome}_after", column]].head(n)
