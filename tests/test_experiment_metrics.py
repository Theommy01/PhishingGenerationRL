"""The three questions the experiment has to answer.

1. how the evasion rate moves across rounds
2. whether the policy is degenerating — measured as KL from the SFT baseline,
   since the baseline itself is imperfect and the question is "worse than it
   was", not "good in absolute terms"
3. whether it still follows the prompt: the <URL> and <ATTACHMENT> placeholders
   the corpus uses, and relatedness to the subject line
"""

import pytest

from loop import report
from policy_kl import summarise
from reference_model import training_stats


# -- 1. evasion across rounds -----------------------------------------------


def test_round_metrics_separates_per_message_and_per_prompt_evasion():
    """evasion_rate counts messages; asr_at_n counts prompts with any hit."""
    messages = [
        {"prompt_id": 0, "score": 0.9, "body": "a"},  # prompt 0 evades once
        {"prompt_id": 0, "score": 0.1, "body": "b"},
        {"prompt_id": 1, "score": 0.2, "body": "c"},  # prompt 1 never evades
        {"prompt_id": 1, "score": 0.3, "body": "d"},
    ]

    metrics = report.round_metrics(messages, threshold=0.5)
    assert metrics["evasion_rate"] == 25.0
    assert metrics["asr_at_n"] == 50.0
    assert metrics["prompts"] == 2


def test_evasion_trajectory_across_rounds(store, prompts, make_records, make_checkpoint):
    """The headline: one row per round, evasion rising or falling."""
    run_id = store.create_run(prompts, {})
    checkpoint = store.upsert_checkpoint(make_checkpoint())

    for round_index, score in enumerate([0.2, 0.6, 0.9]):
        records = make_records(round_index)
        for record in records:
            record["score"] = score
            record["label"] = score >= 0.5
        store.add_messages(run_id, round_index, records, checkpoint=checkpoint)
        store.record_round(
            run_id,
            round_index,
            checkpoint_path=checkpoint["path"],
            metrics=report.round_metrics(
                store.get_messages(run_id, round_index=round_index, with_subject=False)
            ),
        )

    df = report.trajectory(store, run_id)
    assert list(df["evasion_rate"]) == [0.0, 100.0, 100.0]
    assert list(df["mean_score"]) == [20.0, 60.0, 90.0]

    deltas = report.compare(store, run_id)
    assert deltas.loc["evasion_rate", "delta"] == 100.0


def test_round_summary_matches_round_metrics(store, prompts, make_records, make_checkpoint):
    """The long-frame path and the round-document path must not disagree."""
    from metrics.analysis import load_run, round_summary

    run_id = store.create_run(prompts, {})
    checkpoint = store.upsert_checkpoint(make_checkpoint())
    store.add_messages(run_id, 0, make_records(0), checkpoint=checkpoint)

    stored = report.round_metrics(store.get_messages(run_id, round_index=0, with_subject=False))
    summary = round_summary(load_run(store, run_id)).iloc[0]

    assert summary["evasion_rate"] == pytest.approx(stored["evasion_rate"])
    assert summary["asr_at_n"] == pytest.approx(stored["asr_at_n"])
    assert summary["mean_score"] == pytest.approx(stored["mean_score"])


# -- 2. degeneration, as KL from the baseline -------------------------------


def test_training_stats_pull_out_the_kl_trl_logs():
    log_history = [
        {"loss": 1.0, "kl": 0.01, "step": 1},
        {"loss": 0.8, "kl": 0.05, "step": 2},
        {"loss": 0.6, "kl": 0.03, "step": 3},
        {"train_runtime": 12.0},  # the summary entry, which has neither
    ]

    stats = training_stats(log_history)
    assert stats["steps"] == 3
    assert stats["loss_final"] == 0.6
    assert stats["kl_mean"] == pytest.approx(0.03)
    assert stats["kl_max"] == 0.05


def test_training_stats_survive_a_run_without_kl():
    """BCO's loss has no per-step KL for TRL to log."""
    stats = training_stats([{"loss": 1.0, "step": 1}])

    assert stats["loss_final"] == 1.0
    assert "kl_mean" not in stats


def test_policy_kl_summary_reports_the_tail():
    """One message diverging hard is the interesting case, and a mean hides it."""
    records = [{"kl_per_token": v, "kl_k3_per_token": v / 2} for v in [0.1] * 19 + [5.0]]

    summary = summarise(records)
    assert summary["kl_messages"] == 20
    assert summary["kl_per_token"] == pytest.approx(0.345)
    assert summary["kl_max"] == 5.0
    assert summary["kl_p95"] == 5.0


def test_policy_kl_summary_of_nothing_is_empty():
    assert summarise([]) == {}
    assert summarise([{"kl_per_token": None}]) == {}


def test_round_metrics_aggregate_the_policy_kl():
    messages = [
        {"prompt_id": 0, "score": 0.5, "body": "a", "kl_per_token": 0.2},
        {"prompt_id": 1, "score": 0.5, "body": "b", "kl_per_token": 0.4},
    ]

    metrics = report.round_metrics(messages)
    assert metrics["kl_per_token"] == pytest.approx(0.3)
    assert metrics["kl_p95"] == 0.4


# -- 3. instruction following -----------------------------------------------


@pytest.mark.parametrize(
    "body, expected",
    [
        ("Please confirm at <URL> today.", True),
        ("Follow <url> to continue", True),  # case-insensitive
        ("Visit <URL_1> now", True),  # numbered variant
        ("Please reply to this message.", False),
        ("Contact <EMAIL> or call <PHONE>", False),  # other placeholders
    ],
)
def test_url_placeholder_detection(body, expected):
    assert report.has_url(body) is expected


@pytest.mark.parametrize(
    "body, expected",
    [
        ("See <ATTACHMENT> for details.", True),
        ("<ATTACH> enclosed", True),
        ("Please find the invoice attached.", False),  # prose, not the placeholder
        ("Nothing here.", False),
    ],
)
def test_attachment_placeholder_detection(body, expected):
    assert report.mentions_attachment(body) is expected


def test_a_literal_link_is_flagged_separately():
    """Off-distribution for this corpus: the training data has none."""
    assert report.has_literal_url("go to https://evil.example.com now")
    assert not report.has_literal_url("go to <URL> now")


def test_compliance_is_two_sided(prompts):
    """Producing a URL that was not asked for is a failure too."""
    wants_url = next(i for i, spec in enumerate(prompts) if spec["urls"])
    no_url = next(i for i, spec in enumerate(prompts) if not spec["urls"])

    records = [
        {"prompt_id": wants_url, "body": "Confirm at <URL>."},
        {"prompt_id": wants_url, "body": "Confirm by replying."},
        {"prompt_id": no_url, "body": "Confirm at <URL>."},
        {"prompt_id": no_url, "body": "Confirm by replying."},
    ]
    report.attach_compliance(records, prompts, sim_model=None)

    assert [r["url_ok"] for r in records] == [True, False, False, True]
    assert [r["url_requested"] for r in records] == [True, True, False, False]


def test_compliance_without_sbert_skips_only_the_similarity(prompts):
    records = [{"prompt_id": 0, "body": "Confirm at <URL>."}]
    report.attach_compliance(records, prompts, sim_model=None)

    assert "url_ok" in records[0]
    assert "cos_subject" not in records[0]


def test_compliance_uses_sbert_for_subject_relatedness(prompts):
    """cos_subject compares against the subject line, not the whole prompt block."""

    class StubEmbedder:
        def __init__(self):
            self.encoded = []

        def encode(self, texts, convert_to_tensor=False):
            import torch

            self.encoded.append(list(texts))
            return torch.ones(len(texts), 4)

    embedder = StubEmbedder()
    records = [{"prompt_id": 0, "body": "Confirm at <URL>."}]
    report.attach_compliance(records, prompts, sim_model=embedder)

    assert embedder.encoded[0] == [prompts[0]["subject"]]
    assert records[0]["cos_subject"] == pytest.approx(100.0)


def test_round_metrics_report_compliance_as_a_percentage():
    messages = [
        {"prompt_id": 0, "score": 0.5, "body": "a", "url_ok": True, "attachment_ok": True},
        {"prompt_id": 1, "score": 0.5, "body": "b", "url_ok": False, "attachment_ok": True},
    ]

    metrics = report.round_metrics(messages)
    assert metrics["url_compliance"] == 50.0
    assert metrics["attachment_compliance"] == 100.0


def test_the_real_corpus_uses_placeholders_not_links():
    """Guards the assumption the whole check rests on."""
    import json
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Dataset",
        "master_training_dataset.jsonl",
    )
    if not os.path.isfile(path):
        pytest.skip("master_training_dataset.jsonl is not in this checkout")

    with open(path) as handle:
        completions = [json.loads(line)["completion"] for line in handle]

    assert any(report.has_url(text) for text in completions)
    assert not any(report.has_literal_url(text) for text in completions)


# -- policy_kl's model-free paths -------------------------------------------
#
# The forward passes need a GPU and an adapter, so what is testable here is
# when it declines to run at all — which is also where a mistake would silently
# cost a round of GPU time or, worse, record a KL of zero as if measured.


def test_policy_kl_skips_when_the_policy_is_the_reference():
    """Round 0 against SFT: the KL is identically zero, so do not load a model."""
    from policy_kl import attach_policy_kl

    records = [{"prompt_text": "subject: x\n->", "body": "hello"}]
    attach_policy_kl(records, "same/path", "same/path")

    assert "kl_per_token" not in records[0]


def test_policy_kl_skips_without_a_reference():
    from policy_kl import attach_policy_kl

    records = [{"prompt_text": "subject: x\n->", "body": "hello"}]
    attach_policy_kl(records, "policy/path", None)

    assert "kl_per_token" not in records[0]


def test_policy_kl_of_no_records_is_empty():
    from policy_kl import attach_policy_kl, measure_policy_kl

    assert attach_policy_kl([], "a", "b") == []
    assert measure_policy_kl("a", "b", []) == []
