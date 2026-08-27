"""Did the policy learn about phishing, or about the detector it trained on?

Both are consistent with evasion going up, and they have opposite consequences.
If the gain is specific to the detector in the loop, an attack needs access to
the detector it means to defeat. If it transfers to a detector that was never
optimised against, then any surrogate works as a training signal and the
resulting generator outlives the detector it was trained on.

So the question is not "which one" but "how much of each", and the measure is a
decomposition of the *gain*, paired per prompt against a baseline round:

                        held out: improved    held out: not
    in loop: improved       transferred          artefact
    in loop: not               noise             nothing

    transfer fraction = transferred / (transferred + artefact)

Near 0 means detector-specific; near 1 means it transfers; the middle is the
usual answer and the reason for measuring a proportion rather than picking a
side.

Two things to check before believing any of it, both provided here:

`dynamic_range` — a held-out detector fitted on real corpora may be out of
distribution on this text, which is entity-anonymised (`<URL>`, `<ORG>` where a
real link would be). One that calls everything phishing, or nothing, has no
headroom and its transfer number is meaningless.

`baseline_agreement` — if the two detectors already disagree wildly on round 0,
they are not measuring the same construct, and "did not transfer" would say
more about that than about the policy.
"""

from typing import Dict, List, Optional

import pandas as pd

from metrics import config, paired

SCORE_PREFIX = "det_score_"
LABEL_PREFIX = "det_label_"


def with_detector_columns(
    df: pd.DataFrame, in_loop: str = "scamllm", threshold: float = config.SAFE_THRESHOLD
) -> pd.DataFrame:
    """Flatten `detector_scores` / `detector_labels` into columns.

    The in-loop detector's verdict also lives in the message's own `score` and
    `label` — it is the reward, so it has a place of its own — and is used as a
    fallback when the backfill has not written it into the dict form.
    """
    if df.empty:
        return df

    out = df.copy()
    names = set()
    for column in ("detector_scores", "detector_labels"):
        if column in out:
            for value in out[column].dropna():
                if isinstance(value, dict):
                    names.update(value)

    for name in sorted(names):
        if "detector_scores" in out:
            out[f"{SCORE_PREFIX}{name}"] = out["detector_scores"].apply(
                lambda d, n=name: (d or {}).get(n) if isinstance(d, dict) else None
            )
        if "detector_labels" in out:
            out[f"{LABEL_PREFIX}{name}"] = out["detector_labels"].apply(
                lambda d, n=name: (d or {}).get(n) if isinstance(d, dict) else None
            )

    if f"{SCORE_PREFIX}{in_loop}" not in out and "score" in out:
        out[f"{SCORE_PREFIX}{in_loop}"] = out["score"]
    if f"{LABEL_PREFIX}{in_loop}" not in out and "score" in out:
        out[f"{LABEL_PREFIX}{in_loop}"] = out["score"] >= threshold

    return out


def detectors_present(df: pd.DataFrame) -> List[str]:
    return sorted(
        column[len(SCORE_PREFIX) :]
        for column in df.columns
        if column.startswith(SCORE_PREFIX)
    )


def dynamic_range(df: pd.DataFrame, detector: str, round_index: Optional[int] = None) -> Dict:
    """Does this detector discriminate on this text at all?"""
    rows = df if round_index is None else df[df["round"] == round_index]
    score_column, label_column = f"{SCORE_PREFIX}{detector}", f"{LABEL_PREFIX}{detector}"
    if rows.empty or score_column not in rows:
        return {}

    scores = rows[score_column].dropna()
    if scores.empty:
        return {}

    labels = rows[label_column].dropna() if label_column in rows else pd.Series(dtype=bool)
    evaded = float(labels.mean()) if not labels.empty else None
    return {
        "detector": detector,
        "messages": int(len(scores)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "evaded_fraction": evaded,
        "usable": bool(scores.std() > 0.01 and evaded not in (0.0, 1.0)),
    }


def baseline_agreement(
    df: pd.DataFrame, in_loop: str, held_out: str, round_index: int = 0
) -> Dict:
    """How far the two detectors agree at the baseline, before any training."""
    rows = df[df["round"] == round_index]
    first, second = f"{LABEL_PREFIX}{in_loop}", f"{LABEL_PREFIX}{held_out}"
    if rows.empty or first not in rows or second not in rows:
        return {}

    rows = rows[[first, second]].dropna()
    if rows.empty:
        return {}

    a, b = rows[first].astype(bool), rows[second].astype(bool)
    agree = float((a == b).mean())

    # Cohen's kappa: agreement above what the two marginals would give by chance
    p_a, p_b = a.mean(), b.mean()
    chance = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = (agree - chance) / (1 - chance) if chance < 1 else None

    return {
        "round": round_index,
        "messages": int(len(rows)),
        "agreement": agree,
        "kappa": None if kappa is None else float(kappa),
        f"{in_loop}_evaded": float(p_a),
        f"{held_out}_evaded": float(p_b),
    }


def _paired_for(
    df: pd.DataFrame, detector: str, round_index: int, baseline: int, split
) -> pd.DataFrame:
    """One detector's paired per-prompt change, as `metrics.paired` computes it."""
    score_column, label_column = f"{SCORE_PREFIX}{detector}", f"{LABEL_PREFIX}{detector}"
    if df.empty or score_column not in df or label_column not in df:
        return pd.DataFrame()  # this detector has not scored the run

    frame = df.copy()
    frame["score"] = frame[score_column]
    frame["evaded"] = frame[label_column].astype("boolean")
    frame = frame.dropna(subset=["score", "evaded"])
    if frame.empty:
        return pd.DataFrame()

    return paired.paired(frame, round_index, baseline, split)


def decomposition(
    df: pd.DataFrame,
    round_index: int,
    in_loop: str = "scamllm",
    held_out: str = "bert-phishing",
    baseline: int = 0,
    split: Optional[str] = None,
    outcome: str = "evasion_rate",
) -> Dict:
    """Split the gain into transferred and detector-specific.

    Paired per prompt on both detectors, then cross-tabulated. `outcome` picks
    how "improved" is defined: `evasion_rate` counts any increase in the share
    of a prompt's samples that evaded, `any_evaded` only counts prompts that
    crossed from never evading to evading at all.
    """
    in_loop_pairs = _paired_for(df, in_loop, round_index, baseline, split)
    held_out_pairs = _paired_for(df, held_out, round_index, baseline, split)
    delta, before = f"{outcome}_delta", f"{outcome}_before"
    if (
        in_loop_pairs.empty
        or held_out_pairs.empty
        or delta not in in_loop_pairs
        or delta not in held_out_pairs
    ):
        return {}

    shared = in_loop_pairs.index.intersection(held_out_pairs.index)
    first = (in_loop_pairs.loc[shared, delta] > 0)
    second = (held_out_pairs.loc[shared, delta] > 0)

    # A prompt the held-out detector already failed on at the baseline cannot
    # improve there, so counting it as "did not transfer" would understate
    # transfer. It is reported and excluded from the adjusted fraction rather
    # than silently dropped, because how many there are is itself informative.
    saturated = held_out_pairs.loc[shared, before] >= 1.0

    transferred = int((first & second).sum())
    artefact = int((first & ~second).sum())
    noise = int((~first & second).sum())
    nothing = int((~first & ~second).sum())

    gained = transferred + artefact
    ceiling = int((first & saturated).sum())
    headroom = gained - ceiling
    result = {
        "in_loop": in_loop,
        "held_out": held_out,
        "outcome": outcome,
        "round": round_index,
        "baseline": baseline,
        "split": split,
        "prompts": int(len(shared)),
        "transferred": transferred,
        "artefact": artefact,
        "noise": noise,
        "nothing": nothing,
        "gained_in_loop": gained,
        "transfer_fraction": (transferred / gained) if gained else None,
        # gained prompts where the held-out detector was already fooled at the
        # baseline, and the fraction with those removed from the denominator
        "held_out_ceiling": ceiling,
        "transfer_fraction_with_headroom": (
            (transferred / headroom) if headroom else None
        ),
    }

    if gained:
        from scipy.stats import binomtest

        # Is transfer better than the held-out detector's own base rate of
        # improving? Compared against how often it improved on prompts the
        # in-loop detector did *not* gain on — otherwise a detector that drifts
        # on its own would look like transfer.
        base = (noise / (noise + nothing)) if (noise + nothing) else 0.0
        result["holdout_base_rate"] = base
        result["p_value"] = float(
            binomtest(transferred, gained, base or 1e-9, alternative="greater").pvalue
        )
    return result


def level_correlation(
    df: pd.DataFrame,
    in_loop: str = "scamllm",
    held_out: str = "bert-phishing",
    round_index: int = 0,
) -> Dict:
    """Do the two detectors *rank* the same messages alike, thresholds aside?

    The companion to `baseline_agreement`, and the more informative of the two
    when the label agreement is poor. Low agreement with a decent rank
    correlation means the detectors broadly agree and are merely calibrated
    differently — a threshold problem. Low agreement *and* a near-zero rank
    correlation means they disagree about the text itself, and a transfer result
    against this detector would say as much about the detector as the policy.
    """
    rows = df[df["round"] == round_index]
    first, second = f"{SCORE_PREFIX}{in_loop}", f"{SCORE_PREFIX}{held_out}"
    if rows.empty or first not in rows or second not in rows:
        return {}

    rows = rows[[first, second]].dropna()
    if len(rows) < 3:
        return {}

    from scipy.stats import spearmanr

    result = spearmanr(rows[first], rows[second])
    return {
        "round": round_index,
        "messages": int(len(rows)),
        "spearman": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _bootstrap_spearman(x, y, samples: int = 2000, seed: int = 0):
    """Percentile interval for a rank correlation, since n is ~135 prompts."""
    import numpy as np
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    values = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(values[0])
    estimates = []
    for _ in range(samples):
        index = rng.integers(0, n, n)
        a, b = values[0][index], values[1][index]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        estimates.append(spearmanr(a, b).statistic)
    if not estimates:
        return None, None
    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def score_correlation(
    df: pd.DataFrame,
    round_index: int,
    in_loop: str = "scamllm",
    held_out: str = "bert-phishing",
    baseline: int = 0,
    split: Optional[str] = None,
    bootstrap: bool = True,
) -> Dict:
    """Rank correlation of the two detectors' paired per-prompt score changes.

    **The primary transfer measure.** It uses the size of each prompt's
    movement rather than only its sign, and being rank-based it is indifferent
    to the two detectors' scores living on different scales and to where either
    threshold sits — which is what makes it usable when the label agreement is
    poor, as it is here.

    Read it with the two medians beside it, which `report` puts there: a
    positive correlation means prompts that improved against one detector
    improved against the other, but if the held-out median barely moved then
    little transferred in absolute terms however well the ranks line up.
    """
    changes = {}
    for detector in (in_loop, held_out):
        score_column = f"{SCORE_PREFIX}{detector}"
        if df.empty or score_column not in df:
            return {}
        frame = df.copy()
        frame["score"] = frame[score_column]
        frame = frame.dropna(subset=["score"])
        if frame.empty:
            return {}
        joined = paired.paired(frame, round_index, baseline, split)
        if joined.empty or "mean_score_delta" not in joined:
            return {}
        changes[detector] = joined["mean_score_delta"]

    shared = changes[in_loop].index.intersection(changes[held_out].index)
    if len(shared) < 3:
        return {}

    from scipy.stats import spearmanr

    first, second = changes[in_loop].loc[shared], changes[held_out].loc[shared]
    result = spearmanr(first, second)
    out = {
        "prompts": int(len(shared)),
        "spearman": float(result.statistic),
        "p_value": float(result.pvalue),
        f"median_delta_{in_loop}": float(first.median()),
        f"median_delta_{held_out}": float(second.median()),
        f"improved_{in_loop}": int((first > 0).sum()),
        f"improved_{held_out}": int((second > 0).sum()),
    }
    if bootstrap:
        low, high = _bootstrap_spearman(first, second)
        out["ci95"] = (low, high)
    return out


def report(
    df: pd.DataFrame,
    round_index: int,
    in_loop: str = "scamllm",
    held_out: str = "bert-phishing",
    baseline: int = 0,
    split: Optional[str] = None,
) -> Dict:
    """Everything needed to read the transfer result, checks included.

    `correlation` is the headline: it is threshold-free, which matters because
    the two detectors here agree on labels barely better than chance, and a
    label-based decomposition inherits that disagreement. `decomposition` is
    kept as a secondary reading and carries `threshold_dependent`, so nothing
    downstream can present it as if it were free of that assumption.
    """
    agreement = baseline_agreement(df, in_loop, held_out, baseline)
    levels = level_correlation(df, in_loop, held_out, baseline)

    label_based = decomposition(df, round_index, in_loop, held_out, baseline, split)
    if label_based:
        label_based["threshold_dependent"] = True
        label_based["baseline_kappa"] = agreement.get("kappa")

    return {
        "correlation": score_correlation(
            df, round_index, in_loop, held_out, baseline, split
        ),
        "level_correlation": levels,
        "baseline_agreement": agreement,
        "dynamic_range": {
            detector: dynamic_range(df, detector) for detector in (in_loop, held_out)
        },
        "decomposition": label_based,
    }
