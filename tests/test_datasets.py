"""Datasets: a query plus an `as_of`, and the content hash that proves it."""

from datetime import datetime, timedelta, timezone

import pytest

from loop.store import as_utc, dataset_fingerprint, utc_now


@pytest.fixture
def run_with_rounds(store, prompts, make_records):
    """Three rounds, stamped an hour apart, oldest first."""
    run_id = store.create_run(prompts, {})
    stamps = [utc_now() - timedelta(hours=hours) for hours in (3, 2, 1)]
    for round_index, stamp in enumerate(stamps):
        store.add_messages(run_id, round_index, make_records(round_index), added_at=stamp)
    return run_id, stamps


# -- time -------------------------------------------------------------------


def test_as_utc_accepts_datetime_epoch_and_iso():
    aware = datetime(2026, 8, 21, 12, 30, tzinfo=timezone.utc)
    naive = aware.replace(tzinfo=None)

    assert as_utc(aware) == naive
    assert as_utc(naive) == naive
    assert as_utc(aware.timestamp()) == naive
    assert as_utc(naive.isoformat()) == naive
    assert as_utc("2026-08-21T12:30:00Z") == naive


def test_as_utc_converts_a_non_utc_zone():
    berlin = datetime(2026, 8, 21, 14, 30, tzinfo=timezone(timedelta(hours=2)))
    assert as_utc(berlin) == datetime(2026, 8, 21, 12, 30)


def test_as_utc_rejects_nonsense():
    with pytest.raises(TypeError):
        as_utc(object())


def test_a_batch_shares_one_timestamp(store, prompts, make_records):
    """One stamp per round, so no `as_of` can cut a round in half."""
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records(n_samples=3))

    stamps = store.messages.distinct("added_at", {"run_id": run_id})
    assert len(stamps) == 1


# -- as_of ------------------------------------------------------------------


def test_as_of_selects_a_point_in_time(store, run_with_rounds):
    run_id, stamps = run_with_rounds
    query = store.pool_query(run_id, max_round=2)

    assert store.materialise(query, as_of=stamps[0])["count"] == 8
    assert store.materialise(query, as_of=stamps[1])["count"] == 16
    assert store.materialise(query, as_of=stamps[2])["count"] == 24


def test_as_of_defaults_to_now(store, run_with_rounds):
    run_id, _ = run_with_rounds
    query = store.pool_query(run_id, max_round=2)

    assert store.materialise(query)["count"] == 24


def test_a_slice_is_stable_under_later_appends(store, run_with_rounds, make_records):
    """The property round numbers alone cannot promise."""
    run_id, stamps = run_with_rounds
    query = store.pool_query(run_id, max_round=9)
    before = store.materialise(query, as_of=stamps[2])

    store.add_messages(run_id, 3, make_records(3), added_at=utc_now())
    after = store.materialise(query, as_of=stamps[2])

    assert after["content_hash"] == before["content_hash"]
    assert after["count"] == before["count"]
    assert store.materialise(query)["count"] > before["count"]


def test_equivalent_as_of_forms_agree(store, run_with_rounds):
    run_id, stamps = run_with_rounds
    query = store.pool_query(run_id, max_round=2)
    cutoff = stamps[1]

    by_datetime = store.materialise(query, as_of=cutoff)
    by_iso = store.materialise(query, as_of=cutoff.isoformat())
    by_epoch = store.materialise(query, as_of=cutoff.replace(tzinfo=timezone.utc).timestamp())

    assert by_iso["content_hash"] == by_datetime["content_hash"]
    assert by_epoch["content_hash"] == by_datetime["content_hash"]


def test_materialise_maps_fields_and_refuses_unscored_messages(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    records = make_records()
    del records[0]["score"]
    del records[0]["label"]
    store.add_messages(run_id, 0, records)

    rows = store.materialise(
        store.pool_query(run_id, 0), fields={"prompt": "prompt_text", "completion": "body"}
    )["rows"]
    assert set(rows[0]) == {"prompt", "completion"}

    with pytest.raises(ValueError, match="never scored"):
        store.materialise(store.pool_query(run_id, 0))


# -- the content hash -------------------------------------------------------


def test_hash_ignores_row_order_but_not_content(store, run_with_rounds):
    run_id, _ = run_with_rounds
    resolved = store.materialise(store.pool_query(run_id, max_round=2))
    rows = resolved["rows"]

    assert dataset_fingerprint(list(reversed(rows))) == resolved["content_hash"]
    assert dataset_fingerprint(rows[:-1]) != resolved["content_hash"]

    edited = [dict(rows[0], completion="something else")] + rows[1:]
    assert dataset_fingerprint(edited) != resolved["content_hash"]


def test_hash_keeps_duplicate_rows(store):
    row = {"prompt": "p", "completion": "c", "label": True}

    assert dataset_fingerprint([row, row]) != dataset_fingerprint([row])


# -- stored datasets --------------------------------------------------------


def test_create_dataset_records_and_exports(store, run_with_rounds, tmp_path):
    run_id, stamps = run_with_rounds
    export = tmp_path / "pool.jsonl"

    dataset = store.create_dataset(
        store.pool_query(run_id, max_round=1),
        as_of=stamps[1],
        name="round1_pool",
        export_path=str(export),
    )

    assert dataset["count"] == 16
    assert dataset["name"] == "round1_pool"
    assert len(export.read_text().strip().splitlines()) == 16
    assert store.get_dataset(dataset["_id"])["content_hash"] == dataset["content_hash"]
    # exported rows are the training schema, not raw message documents
    import json

    assert set(json.loads(export.read_text().splitlines()[0])) == {
        "prompt",
        "completion",
        "label",
    }


def test_recreating_the_same_spec_reuses_the_document(store, run_with_rounds):
    run_id, stamps = run_with_rounds
    query = store.pool_query(run_id, max_round=1)

    first = store.create_dataset(query, as_of=stamps[1])
    second = store.create_dataset(query, as_of=stamps[1])

    assert first["_id"] == second["_id"]
    assert store.datasets.count_documents({}) == 1


def test_the_same_spec_written_differently_is_one_dataset(store, run_with_rounds):
    """MongoDB compares embedded documents by key order; the spec hash does not."""
    run_id, stamps = run_with_rounds

    first = store.create_dataset({"run_id": run_id, "round": {"$lte": 1}}, as_of=stamps[1])
    second = store.create_dataset({"round": {"$lte": 1}, "run_id": run_id}, as_of=stamps[1])

    assert first["_id"] == second["_id"]


def test_a_different_cutoff_is_a_different_dataset(store, run_with_rounds):
    run_id, stamps = run_with_rounds
    query = store.pool_query(run_id, max_round=2)

    first = store.create_dataset(query, as_of=stamps[1])
    second = store.create_dataset(query, as_of=stamps[2])

    assert first["_id"] != second["_id"]
    assert first["content_hash"] != second["content_hash"]


def test_verify_dataset_passes_when_nothing_moved(store, run_with_rounds):
    run_id, stamps = run_with_rounds
    dataset = store.create_dataset(store.pool_query(run_id, max_round=1), as_of=stamps[1])

    assert store.verify_dataset(dataset["_id"])["ok"]


@pytest.mark.parametrize("tamper", ["edit", "delete"])
def test_verify_dataset_catches_rewritten_history(store, run_with_rounds, tamper):
    run_id, stamps = run_with_rounds
    dataset = store.create_dataset(store.pool_query(run_id, max_round=1), as_of=stamps[1])

    if tamper == "edit":
        store.messages.update_one({"run_id": run_id}, {"$set": {"body": "rewritten"}})
    else:
        store.messages.delete_one({"run_id": run_id})

    check = store.verify_dataset(dataset["_id"])
    assert not check["ok"]
    assert check["recorded"]["content_hash"] != check["actual"]["content_hash"]


def test_recreating_a_changed_slice_raises(store, run_with_rounds):
    run_id, stamps = run_with_rounds
    query = store.pool_query(run_id, max_round=1)
    store.create_dataset(query, as_of=stamps[1])

    store.messages.update_one({"run_id": run_id}, {"$set": {"body": "rewritten"}})

    with pytest.raises(ValueError, match="messages under it have changed"):
        store.create_dataset(query, as_of=stamps[1])


def test_dataset_rows_rematerialise_from_the_stored_spec(store, run_with_rounds):
    run_id, stamps = run_with_rounds
    dataset = store.create_dataset(store.pool_query(run_id, max_round=1), as_of=stamps[1])

    rows = store.dataset_rows(dataset["_id"])
    assert len(rows) == dataset["count"]
    assert set(rows[0]) == {"prompt", "completion", "label"}


def test_unknown_dataset_raises(store):
    from bson import ObjectId

    with pytest.raises(KeyError):
        store.verify_dataset(ObjectId())


def test_drop_run_takes_its_datasets(store, run_with_rounds):
    run_id, stamps = run_with_rounds
    store.create_dataset(store.pool_query(run_id, max_round=1), as_of=stamps[1])

    store.drop_run(run_id)

    assert store.datasets.count_documents({}) == 0


def test_training_pool_is_the_dataset_rows(store, run_with_rounds):
    run_id, _ = run_with_rounds
    pool = store.training_pool(run_id, max_round=1)

    assert len(pool) == 16
    assert set(pool[0]) == {"prompt", "completion", "label"}
    assert store.datasets.count_documents({}) == 0  # records nothing
