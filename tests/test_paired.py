"""Paired, per-prompt change — the analysis the uneven baseline calls for."""

import pandas as pd
import pytest

from metrics import paired


def frame(rows):
    """rows: (round, prompt_id, score, split) -> the columns the module needs."""
    return pd.DataFrame(
        [
            {
                "round": r,
                "prompt_id": p,
                "score": s,
                "evaded": s >= 0.5,
                "split": split,
                "body": f"r{r}p{p}",
                "subject_text": f"subject {p}",
            }
            for r, p, s, split in rows
        ]
    )


def test_per_prompt_collapses_samples():
    df = frame([(0, 0, 0.9, "train"), (0, 0, 0.1, "train"), (0, 1, 0.2, "train")])

    out = paired.per_prompt(df, 0)
    assert out.loc[0, "samples"] == 2
    assert out.loc[0, "any_evaded"]           # one of two got through
    assert out.loc[0, "evasion_rate"] == 0.5
    assert out.loc[0, "mean_score"] == pytest.approx(0.5)
    assert not out.loc[1, "any_evaded"]


def test_paired_deltas_are_per_prompt():
    df = frame(
        [
            (0, 0, 0.2, "train"), (1, 0, 0.8, "train"),   # improved
            (0, 1, 0.9, "train"), (1, 1, 0.3, "train"),   # regressed
            (0, 2, 0.4, "train"), (1, 2, 0.4, "train"),   # unchanged
        ]
    )

    joined = paired.paired(df, round_index=1)
    assert joined.loc[0, "mean_score_delta"] == pytest.approx(0.6)
    assert joined.loc[1, "mean_score_delta"] == pytest.approx(-0.6)
    assert joined.loc[2, "mean_score_delta"] == 0


def test_a_prompt_missing_from_either_round_is_dropped():
    """Not counted as a change — it was never compared."""
    df = frame([(0, 0, 0.2, "train"), (1, 0, 0.8, "train"), (1, 1, 0.9, "train")])

    assert list(paired.paired(df, round_index=1).index) == [0]


def test_flip_table_counts_only_discordant_prompts():
    df = frame(
        [
            (0, 0, 0.2, "train"), (1, 0, 0.8, "train"),   # gained
            (0, 1, 0.1, "train"), (1, 1, 0.9, "train"),   # gained
            (0, 2, 0.8, "train"), (1, 2, 0.2, "train"),   # lost
            (0, 3, 0.9, "train"), (1, 3, 0.9, "train"),   # both
            (0, 4, 0.1, "train"), (1, 4, 0.1, "train"),   # neither
        ]
    )

    table = paired.flip_table(df, round_index=1)
    assert (table["gained"], table["lost"]) == (2, 1)
    assert (table["both"], table["neither"]) == (1, 1)
    assert table["net"] == 1
    assert table["discordant"] == 3
    assert 0 < table["p_value"] <= 1


def test_a_prompt_already_evading_cannot_inflate_the_gain():
    """The ceiling case: 4/4 in round 0 has nowhere to go."""
    df = frame([(0, 0, 0.9, "train"), (1, 0, 0.95, "train")])

    table = paired.flip_table(df, round_index=1)
    assert table["gained"] == 0 and table["lost"] == 0
    assert table["both"] == 1
    assert "p_value" not in table, "no discordant pairs means no test"


def test_summarise_is_sign_based():
    df = frame(
        [
            (0, 0, 0.1, "train"), (1, 0, 0.9, "train"),
            (0, 1, 0.2, "train"), (1, 1, 0.3, "train"),
            (0, 2, 0.5, "train"), (1, 2, 0.4, "train"),
        ]
    )

    summary = paired.summarise(paired.paired(df, 1), "mean_score")
    assert summary["improved"] == 2
    assert summary["worsened"] == 1
    assert summary["fraction_improved"] == pytest.approx(2 / 3)
    assert summary["median_delta"] == pytest.approx(0.1)


def test_one_wild_prompt_moves_the_mean_but_not_the_median():
    """Why the summary leads with signs: the baseline is uneven."""
    rows = [(0, p, 0.4, "train") for p in range(9)] + [(1, p, 0.45, "train") for p in range(9)]
    rows += [(0, 9, 0.05, "train"), (1, 9, 0.99, "train")]  # one prompt goes wild

    summary = paired.summarise(paired.paired(frame(rows), 1), "mean_score")
    assert summary["median_delta"] == pytest.approx(0.05)
    assert summary["mean_delta"] > 0.12
    assert summary["improved"] == 10


def test_score_can_move_without_any_flip():
    """Pressure without breakthrough — the case the binary cannot see."""
    df = frame([(0, p, 0.10, "train") for p in range(5)]
               + [(1, p, 0.45, "train") for p in range(5)])

    table = paired.flip_table(df, round_index=1)
    summary = paired.summarise(paired.paired(df, 1), "mean_score")

    assert table["gained"] == 0
    assert summary["median_delta"] == pytest.approx(0.35)
    assert summary["fraction_improved"] == 1.0


def test_report_covers_every_available_outcome():
    df = frame([(0, 0, 0.2, "train"), (1, 0, 0.8, "train")])

    out = paired.report(df, round_index=1)
    assert set(out["outcome"]) >= {"any_evaded", "evasion_rate", "mean_score"}


def test_difference_in_differences_measures_the_memorisation_gap():
    rows = []
    for p in range(4):  # training prompts improve a lot
        rows += [(0, p, 0.2, "train"), (1, p, 0.9, "train")]
    for p in range(4, 6):  # held out improves less
        rows += [(0, p, 0.2, "holdout"), (1, p, 0.5, "holdout")]

    did = paired.difference_in_differences(frame(rows), 1, outcome="mean_score")
    assert did["train"]["median"] == pytest.approx(0.7)
    assert did["holdout"]["median"] == pytest.approx(0.3)
    assert did["gap_median"] == pytest.approx(0.4)


def test_movers_surface_the_prompts_to_read():
    df = frame(
        [
            (0, 0, 0.5, "train"), (1, 0, 0.5, "train"),
            (0, 1, 0.1, "train"), (1, 1, 0.99, "train"),
        ]
    )

    best = paired.movers(paired.paired(df, 1), "mean_score", n=1)
    assert best.iloc[0]["subject_text"] == "subject 1"

    worst = paired.movers(paired.paired(df, 1), "mean_score", n=1, worst=True)
    assert worst.iloc[0]["subject_text"] == "subject 0"


def test_everything_survives_an_empty_or_single_round_frame():
    empty = pd.DataFrame(columns=["round", "prompt_id", "score", "split", "body"])
    assert paired.per_prompt(empty, 0).empty
    assert paired.paired(empty, 1).empty
    assert paired.flip_table(empty, 1) == {}
    assert paired.report(empty, 1).empty
    assert paired.difference_in_differences(empty, 1) == {}

    one_round = frame([(0, 0, 0.5, "train")])
    assert paired.paired(one_round, round_index=1).empty
