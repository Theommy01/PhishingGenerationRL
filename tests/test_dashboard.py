"""The dashboard renders — against an empty run, a live one, and a finished one.

Streamlit's own harness runs the script in-process and collects whatever it
raised, so this catches the failure mode a dashboard actually has: a column that
is missing while a run is still generating.
"""

import os

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import REPO_ROOT  # noqa: E402

# absolute: AppTest resolves a relative path against *this* file's directory
APP = os.path.join(REPO_ROOT, "dashboard", "app.py")
TIMEOUT = 120


@pytest.fixture
def app(store, monkeypatch):
    """AppTest pointed at the test's scratch database.

    Every cache is cleared, not just the store's: the loaders are keyed on
    run_id, and two tests creating a run in the same second get the *same* id in
    their separate scratch databases — so a stale entry from the previous test
    would answer for this one.
    """
    import streamlit as st

    from dashboard import data

    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setattr(data, "get_store", lambda: store)

    yield lambda: AppTest.from_file(APP, default_timeout=TIMEOUT).run()

    st.cache_data.clear()


def test_it_renders_with_no_runs_at_all(app):
    at = app()

    assert not at.exception
    assert any("No runs" in w.value for w in at.warning)


def test_it_renders_a_run_that_is_still_generating(app, store, prompts, make_records):
    """Messages stored, none scored — what the first hour of a run looks like."""
    run_id = store.create_run(prompts, {"n_samples": 2, "algorithm": "kto"})
    records = [
        {k: v for k, v in record.items() if k not in ("score", "label")}
        for record in make_records()
    ]
    store.add_messages(run_id, 0, records)

    at = app()

    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == [
        "Overview", "Messages", "Transfer", "Compare", "Provenance"
    ]
    # the overview says so rather than dividing by zero
    assert any("Nothing scored yet" in block.value for block in at.info)


def test_it_renders_a_scored_run(app, store, prompts, make_records, make_checkpoint):
    from loop import report

    run_id = store.create_run(prompts, {"n_samples": 2, "algorithm": "kto"})
    checkpoint = store.upsert_checkpoint(make_checkpoint())
    records = report.attach_compliance(make_records(), prompts, sim_model=None)
    store.add_messages(run_id, 0, records, checkpoint=checkpoint)
    store.record_round(
        run_id,
        0,
        checkpoint_path=checkpoint["path"],
        checkpoint=store.checkpoint_ref(checkpoint),
        checkpoint_hash=checkpoint["weights_hash"],
        metrics=report.round_metrics(
            store.get_messages(run_id, round_index=0, with_subject=False)
        ),
    )

    at = app()

    assert not at.exception, [e.value for e in at.exception]
    # the headline metrics are rendered rather than skipped
    assert any("Evasion" in metric.label for metric in at.metric)


def test_the_transfer_tab_asks_for_a_second_detector_when_there_is_one(
    app, store, prompts, make_records, make_checkpoint
):
    """With only the in-loop detector scored, it says how to add another."""
    run_id = store.create_run(prompts, {"n_samples": 2})
    store.add_messages(run_id, 0, make_records(), checkpoint=store.upsert_checkpoint(make_checkpoint()))

    at = app()

    assert not at.exception, [e.value for e in at.exception]
    assert any("backfill" in w.value for w in at.warning)


def test_the_transfer_tab_renders_with_two_detectors(
    app, store, prompts, make_records, make_checkpoint
):
    run_id = store.create_run(prompts, {"n_samples": 2})
    checkpoint = store.upsert_checkpoint(make_checkpoint())
    for round_index in (0, 1):
        store.add_messages(run_id, round_index, make_records(round_index), checkpoint=checkpoint)
        verdicts = [
            {
                "round": round_index,
                "prompt_id": m["prompt_id"],
                "sample_idx": m["sample_idx"],
                "score": 0.3 + 0.2 * round_index + 0.05 * m["prompt_id"],
                "label": round_index > 0,
            }
            for m in store.get_messages(run_id, round_index=round_index, with_subject=False)
        ]
        store.set_detector_verdicts(run_id, "bert-phishing", verdicts)

    at = app()

    assert not at.exception, [e.value for e in at.exception]
    assert any("Did the gain transfer" in h.value for h in at.subheader)


def test_the_evasion_chart_pins_which_line_is_which():
    """The legend must not depend on Vega's alphabetical sort of the domain.

    Left implicit, `asr_at_n` sorted before `evasion_rate` and silently took the
    solid pattern the caption claimed for the evasion rate.
    """
    import json

    import pandas as pd

    from dashboard import charts

    summary = pd.DataFrame(
        {
            "round": [0, 1],
            "split": ["train", "train"],
            "messages": [540, 540],
            "evasion_rate": [31.3, 34.8],
            "asr_at_n": [63.0, 67.4],
        }
    )

    import streamlit as st

    drawn = {}
    st.altair_chart = lambda chart, **kwargs: drawn.setdefault("chart", chart)
    charts.evasion_by_split(summary)

    spec = json.loads(drawn["chart"].to_json())
    dash = next(
        layer["encoding"]["strokeDash"]
        for layer in spec["layer"]
        if "strokeDash" in layer.get("encoding", {})
    )
    domain = dash["scale"]["domain"]
    ranges = dash["scale"]["range"]

    assert domain[0] == charts.METRIC_LABELS["evasion_rate"]
    assert ranges[domain.index(charts.METRIC_LABELS["evasion_rate"])] == charts.SOLID
    assert ranges[domain.index(charts.METRIC_LABELS["asr_at_n"])] == charts.DASHED


def test_transfer_tab_defaults_to_a_round_the_held_out_detector_scored(
    app, store, prompts, make_records, make_checkpoint
):
    """During a backfill the latest round is often unscored; the default must
    land on the latest COMPARABLE round rather than rendering empty."""
    run_id = store.create_run(prompts, {"n_samples": 2})
    checkpoint = store.upsert_checkpoint(make_checkpoint())
    for round_index in range(3):
        store.add_messages(run_id, round_index, make_records(round_index), checkpoint=checkpoint)
        # only rounds 0 and 1 get a held-out verdict; round 2 is "still backfilling"
        if round_index < 2:
            store.set_detector_verdicts(
                run_id, "bert-phishing",
                [{"round": round_index, "prompt_id": m["prompt_id"],
                  "sample_idx": m["sample_idx"], "score": 0.4 + 0.05 * m["prompt_id"],
                  "label": round_index > 0}
                 for m in store.get_messages(run_id, round_index=round_index, with_subject=False)],
            )

    at = app()

    assert not at.exception, [e.value for e in at.exception]
    # it defaulted to a comparable round and produced the headline, rather than
    # defaulting to round 2 and showing "not enough"
    assert any("Did the gain transfer" in h.value for h in at.subheader)
