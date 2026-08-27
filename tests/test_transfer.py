"""Artefact or genuine improvement — the decomposition and its preconditions."""

import pandas as pd
import pytest

from metrics import transfer


def frame(rows):
    """rows: (round, prompt_id, in_loop_score, held_out_score)."""
    return pd.DataFrame(
        [
            {
                "round": r,
                "prompt_id": p,
                "sample_idx": 0,
                "split": "train",
                "body": f"r{r}p{p}",
                "detector_scores": {"scamllm": a, "bert-phishing": b},
                "detector_labels": {"scamllm": a >= 0.5, "bert-phishing": b >= 0.5},
            }
            for r, p, a, b in rows
        ]
    )


def test_detector_columns_are_flattened():
    df = transfer.with_detector_columns(frame([(0, 0, 0.2, 0.3)]))

    assert df.loc[0, "det_score_scamllm"] == 0.2
    assert df.loc[0, "det_score_bert-phishing"] == 0.3
    assert transfer.detectors_present(df) == ["bert-phishing", "scamllm"]


def test_the_in_loop_score_falls_back_to_the_reward_field():
    """The reward has a field of its own; the backfill may not have run."""
    df = pd.DataFrame([{"round": 0, "prompt_id": 0, "sample_idx": 0, "score": 0.8,
                        "body": "b", "split": "train"}])

    out = transfer.with_detector_columns(df)
    assert out.loc[0, "det_score_scamllm"] == 0.8
    assert bool(out.loc[0, "det_label_scamllm"]) is True


def test_dynamic_range_flags_a_detector_with_no_headroom():
    """The precondition: a detector that says the same thing about everything."""
    flat = transfer.with_detector_columns(
        frame([(0, p, 0.4, 0.99) for p in range(10)])
    )

    assert transfer.dynamic_range(flat, "scamllm")["usable"] is False, "no variance"
    held = transfer.dynamic_range(flat, "bert-phishing")
    assert held["usable"] is False, "everything evaded — nothing left to gain"
    assert held["evaded_fraction"] == 1.0


def test_dynamic_range_accepts_a_discriminating_detector():
    df = transfer.with_detector_columns(
        frame([(0, p, 0.1 * p, 0.1 * p) for p in range(10)])
    )

    assert transfer.dynamic_range(df, "scamllm")["usable"] is True


def test_baseline_agreement_reports_kappa():
    """Do the two detectors measure the same thing before any training?"""
    df = transfer.with_detector_columns(
        frame([(0, p, 0.9 if p < 5 else 0.1, 0.9 if p < 5 else 0.1) for p in range(10)])
    )

    agreement = transfer.baseline_agreement(df, "scamllm", "bert-phishing")
    assert agreement["agreement"] == 1.0
    assert agreement["kappa"] == pytest.approx(1.0)


def test_pure_transfer_is_measured_as_such():
    """Both detectors improve on the same prompts."""
    rows = [(0, p, 0.1, 0.1) for p in range(10)] + [(1, p, 0.9, 0.9) for p in range(10)]

    result = transfer.decomposition(transfer.with_detector_columns(frame(rows)), 1)
    assert result["transferred"] == 10
    assert result["artefact"] == 0
    assert result["transfer_fraction"] == 1.0


def test_pure_artefact_is_measured_as_such():
    """The in-loop detector is fooled; the held-out one is not."""
    rows = [(0, p, 0.1, 0.1) for p in range(10)] + [(1, p, 0.9, 0.1) for p in range(10)]

    result = transfer.decomposition(transfer.with_detector_columns(frame(rows)), 1)
    assert result["transferred"] == 0
    assert result["artefact"] == 10
    assert result["transfer_fraction"] == 0.0


def test_a_partial_split_is_the_usual_answer():
    rows = []
    for p in range(10):
        rows.append((0, p, 0.1, 0.1))
        rows.append((1, p, 0.9, 0.9 if p < 4 else 0.1))   # 4 of 10 transfer

    result = transfer.decomposition(transfer.with_detector_columns(frame(rows)), 1)
    assert (result["transferred"], result["artefact"]) == (4, 6)
    assert result["transfer_fraction"] == pytest.approx(0.4)
    assert 0 <= result["p_value"] <= 1


def test_prompts_that_did_not_gain_are_excluded_from_the_fraction():
    """Only the gain is decomposed — the rest is context, not denominator."""
    rows = []
    for p in range(5):
        rows += [(0, p, 0.1, 0.1), (1, p, 0.9, 0.9)]      # gained, transferred
    for p in range(5, 10):
        rows += [(0, p, 0.1, 0.1), (1, p, 0.1, 0.1)]      # never moved

    result = transfer.decomposition(transfer.with_detector_columns(frame(rows)), 1)
    assert result["gained_in_loop"] == 5
    assert result["nothing"] == 5
    assert result["transfer_fraction"] == 1.0


def test_held_out_drift_is_kept_out_of_the_transfer_claim():
    """A detector drifting on its own must not read as transfer."""
    rows = []
    for p in range(5):
        rows += [(0, p, 0.1, 0.1), (1, p, 0.9, 0.9)]      # in-loop gained too
    for p in range(5, 15):
        rows += [(0, p, 0.1, 0.1), (1, p, 0.1, 0.9)]      # only the held-out moved

    result = transfer.decomposition(transfer.with_detector_columns(frame(rows)), 1)
    assert result["noise"] == 10
    assert result["holdout_base_rate"] == 1.0
    assert result["p_value"] > 0.05, "transfer should not look significant here"


def test_score_correlation_uses_the_size_of_the_movement():
    rows = []
    for p in range(10):
        rows += [(0, p, 0.1, 0.1), (1, p, 0.1 + p * 0.08, 0.1 + p * 0.08)]

    result = transfer.score_correlation(transfer.with_detector_columns(frame(rows)), 1)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["prompts"] == 10


def test_report_leads_with_the_threshold_free_measure():
    rows = [(0, p, 0.1 + 0.01 * p, 0.1 + 0.01 * p) for p in range(8)]
    rows += [(1, p, 0.9, 0.5 + 0.01 * p) for p in range(8)]

    out = transfer.report(transfer.with_detector_columns(frame(rows)), 1)

    # the correlation is first, and the label-based reading is marked as
    # inheriting the threshold it depends on
    assert list(out)[0] == "correlation"
    assert out["decomposition"]["threshold_dependent"] is True
    assert "baseline_kappa" in out["decomposition"]
    assert set(out["dynamic_range"]) == {"scamllm", "bert-phishing"}


def test_the_correlation_carries_an_interval_and_both_medians():
    rows = []
    for p in range(20):
        rows += [(0, p, 0.1, 0.1), (1, p, 0.1 + p * 0.04, 0.1 + p * 0.03)]

    out = transfer.score_correlation(transfer.with_detector_columns(frame(rows)), 1)
    assert out["spearman"] == pytest.approx(1.0)
    low, high = out["ci95"]
    assert low <= out["spearman"] <= high
    assert out["median_delta_scamllm"] > 0
    assert out["median_delta_bert-phishing"] > 0


def test_level_correlation_separates_calibration_from_disagreement():
    """Same ranking, different thresholds — a calibration problem, not a real one."""
    calibrated = [(0, p, 0.1 * p, 0.1 * p + 0.4) for p in range(10)]
    assert transfer.level_correlation(
        transfer.with_detector_columns(frame(calibrated))
    )["spearman"] == pytest.approx(1.0)

    # opposite rankings: the detectors disagree about the text itself
    opposed = [(0, p, 0.1 * p, 1.0 - 0.1 * p) for p in range(10)]
    assert transfer.level_correlation(
        transfer.with_detector_columns(frame(opposed))
    )["spearman"] == pytest.approx(-1.0)


def test_it_degrades_quietly_when_a_detector_is_missing():
    df = transfer.with_detector_columns(frame([(0, 0, 0.2, 0.3), (1, 0, 0.8, 0.9)]))

    assert transfer.decomposition(df, 1, held_out="not-scored-yet") == {}
    assert transfer.score_correlation(df, 1, held_out="not-scored-yet") == {}


def test_a_ceiling_on_the_held_out_detector_is_reported_not_hidden():
    """A prompt it already failed at baseline cannot improve — so it is not
    evidence of "did not transfer"."""
    rows = []
    for p in range(4):  # held-out already fully fooled at the baseline
        rows += [(0, p, 0.1, 0.9), (1, p, 0.9, 0.95)]
    for p in range(4, 8):  # held-out had headroom, and transferred
        rows += [(0, p, 0.1, 0.1), (1, p, 0.9, 0.9)]

    result = transfer.decomposition(transfer.with_detector_columns(frame(rows)), 1)

    assert result["gained_in_loop"] == 8
    assert result["held_out_ceiling"] == 4
    assert result["transferred"] == 4
    # the naive fraction understates; the adjusted one does not
    assert result["transfer_fraction"] == pytest.approx(0.5)
    assert result["transfer_fraction_with_headroom"] == pytest.approx(1.0)
