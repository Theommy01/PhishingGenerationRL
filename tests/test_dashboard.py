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
    assert [t.label for t in at.tabs] == ["Overview", "Messages", "Compare", "Provenance"]
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
