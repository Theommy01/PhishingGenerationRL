"""Exporting runs to an archive and importing them into another database.

The point of these is the *joins*: a message that arrives without a resolvable
subject ref, or a dataset whose rows no longer hash the same, has travelled as
bytes but not as data. So most of what is asserted is dereferencing on the far
side, not document counts.
"""

import gzip
import json
import os
import uuid

import pytest
from bson import DBRef

from loop import archive
from loop.store import LoopStore


@pytest.fixture
def other_store(mongo_client):
    """A second, empty database — the recipient."""
    db_name = f"phishnet_rl_test_{uuid.uuid4().hex[:12]}"
    store = LoopStore(db_name=db_name, client=mongo_client)
    yield store
    mongo_client.drop_database(db_name)


@pytest.fixture
def populated(store, prompts, make_records, make_checkpoint):
    """A two-round run with checkpoints, a pinned dataset and held-out messages."""
    run_id = store.create_run(prompts, {"n_samples": 2, "algorithm": "kto"})

    for round_index in range(2):
        checkpoint = store.upsert_checkpoint(
            make_checkpoint(f"round{round_index}", weights=f"adapter {round_index}".encode())
        )
        store.add_messages(
            run_id, round_index, make_records(round_index), checkpoint=checkpoint
        )
        store.add_messages(
            run_id,
            round_index,
            make_records(round_index, n_samples=1),
            checkpoint=checkpoint,
            split="holdout",
        )
        dataset = store.create_dataset(
            store.pool_query(run_id, round_index), name=f"pool-{round_index}"
        )
        store.record_round(
            run_id,
            round_index,
            checkpoint=store.checkpoint_ref(checkpoint),
            checkpoint_hash=checkpoint["weights_hash"],
            dataset=store.dataset_ref(dataset),
            metrics={"asr": 0.5},
        )

    return run_id


def test_round_trip_carries_the_run(store, other_store, populated, tmp_path):
    path = str(tmp_path / "run.tar.gz")
    archive.export_archive(store, path, [populated])
    summary = archive.import_archive(other_store, path)

    assert summary["runs"] == [populated]
    assert other_store.get_run(populated)["config"] == store.get_run(populated)["config"]
    assert len(other_store.get_rounds(populated)) == 2

    here = store.get_messages(populated, with_subject=False)
    there = other_store.get_messages(populated, with_subject=False)
    assert len(there) == len(here)
    assert {m["body"] for m in there} == {m["body"] for m in here}


def test_subjects_dereference_on_the_far_side(store, other_store, populated, tmp_path):
    """The refs are the export's whole job: a message must still find its spec."""
    path = str(tmp_path / "run.tar.gz")
    archive.export_archive(store, path, [populated])
    archive.import_archive(other_store, path)

    messages = other_store.get_messages(populated)
    assert messages
    for message in messages:
        assert isinstance(message["subject"], DBRef)
        assert message["subject_text"]
        assert message["category"]

    original = {m["subject_text"] for m in store.get_messages(populated)}
    assert {m["subject_text"] for m in messages} == original


def test_checkpoint_refs_survive(store, other_store, populated, tmp_path):
    path = str(tmp_path / "run.tar.gz")
    archive.export_archive(store, path, [populated])
    archive.import_archive(other_store, path)

    for round_document in other_store.get_rounds(populated):
        checkpoint = other_store.get_checkpoint(round_document["checkpoint"])
        assert checkpoint is not None
        assert checkpoint["weights_hash"] == round_document["checkpoint_hash"]

    message = other_store.get_messages(populated, round_index=0)[0]
    assert other_store.checkpoint_for_message(message)["weights_hash"] == message["checkpoint_hash"]


def test_imported_datasets_still_hash_the_same(store, other_store, populated, tmp_path):
    """The end-to-end integrity claim: same query, same cut-off, same rows."""
    path = str(tmp_path / "run.tar.gz")
    archive.export_archive(store, path, [populated])
    summary = archive.import_archive(other_store, path)

    assert summary["datasets"]
    assert all(check["ok"] for check in summary["datasets"])


def test_import_is_idempotent(store, other_store, populated, tmp_path):
    path = str(tmp_path / "run.tar.gz")
    archive.export_archive(store, path, [populated])
    archive.import_archive(other_store, path)
    before = other_store.messages.count_documents({})

    again = archive.import_archive(other_store, path)

    assert again["skipped"] == [populated]
    assert other_store.messages.count_documents({}) == before
    assert other_store.runs.count_documents({"run_id": populated}) == 1


def test_shared_subjects_are_not_duplicated(store, other_store, populated, prompts, tmp_path):
    """The recipient already ran the same prompts.json; one subject, not two."""
    other_store.sync_subjects(prompts)
    existing = {doc["spec_hash"]: doc["_id"] for doc in other_store.subjects.find({})}

    path = str(tmp_path / "run.tar.gz")
    archive.export_archive(store, path, [populated])
    archive.import_archive(other_store, path)

    assert other_store.subjects.count_documents({}) == len(existing)
    # and the messages point at the documents that were already there
    for message in other_store.get_messages(populated, with_subject=False):
        assert message["subject"].id in existing.values()


def test_remap_imports_alongside_a_colliding_run(store, other_store, populated, prompts, tmp_path):
    """Two researchers, one afternoon, the same unix-timestamp run id."""
    other_store.create_run(prompts, {"n_samples": 9, "algorithm": "bco"})
    other_store.runs.update_one({}, {"$set": {"run_id": populated}})

    path = str(tmp_path / "run.tar.gz")
    archive.export_archive(store, path, [populated])
    summary = archive.import_archive(other_store, path, on_conflict="remap")

    fresh = summary["remapped"][populated]
    assert fresh != populated
    # the local run is untouched, the imported one is complete beside it
    assert other_store.get_run(populated)["config"]["algorithm"] == "bco"
    assert other_store.get_run(fresh)["config"]["algorithm"] == "kto"
    assert len(other_store.get_messages(fresh, with_subject=False)) == len(
        store.get_messages(populated, with_subject=False)
    )
    assert all(check["ok"] for check in summary["datasets"])


def test_remapped_datasets_query_the_new_run(store, other_store, populated, prompts, tmp_path):
    other_store.create_run(prompts, {})
    other_store.runs.update_one({}, {"$set": {"run_id": populated}})

    path = str(tmp_path / "run.tar.gz")
    archive.export_archive(store, path, [populated])
    summary = archive.import_archive(other_store, path, on_conflict="remap")
    fresh = summary["remapped"][populated]

    for dataset in other_store.datasets.find({}):
        assert dataset["query"]["run_id"] == fresh
        assert other_store.verify_dataset(dataset["_id"])["ok"]


def test_replace_overwrites_the_local_run(store, other_store, populated, prompts, tmp_path):
    other_store.create_run(prompts, {"algorithm": "bco"})
    other_store.runs.update_one({}, {"$set": {"run_id": populated}})

    path = str(tmp_path / "run.tar.gz")
    archive.export_archive(store, path, [populated])
    summary = archive.import_archive(other_store, path, on_conflict="replace")

    assert summary["replaced"] == [populated]
    assert other_store.runs.count_documents({"run_id": populated}) == 1
    assert other_store.get_run(populated)["config"]["algorithm"] == "kto"


def test_export_to_a_directory(store, populated, tmp_path):
    path = str(tmp_path / "archive")
    archive.export_archive(store, path, [populated])

    assert os.path.isfile(os.path.join(path, "manifest.json"))
    for collection in archive.COLLECTIONS:
        assert os.path.isfile(os.path.join(path, f"{collection}.jsonl.gz"))

    read = archive.read_archive(path)
    assert read["manifest"]["runs"][0]["run_id"] == populated


def test_a_damaged_member_is_refused(store, populated, tmp_path):
    """The manifest hashes the documents, so a truncated file is caught."""
    path = str(tmp_path / "archive")
    archive.export_archive(store, path, [populated])

    member = os.path.join(path, "messages.jsonl.gz")
    documents = archive._decode(gzip.decompress(open(member, "rb").read()))
    with open(member, "wb") as handle:
        handle.write(gzip.compress(archive._encode(documents[:-1])))

    with pytest.raises(ValueError, match="damaged"):
        archive.read_archive(path)


def test_export_names_a_missing_run(store, tmp_path):
    with pytest.raises(KeyError, match="no such run"):
        archive.export_archive(store, str(tmp_path / "x.tar.gz"), [1])


def test_only_the_named_run_travels(store, other_store, populated, prompts, make_records, tmp_path):
    other_run = store.create_run(prompts, {"algorithm": "bco"})
    store.add_messages(other_run, 0, make_records(0))

    path = str(tmp_path / "run.tar.gz")
    archive.export_archive(store, path, [populated])
    archive.import_archive(other_store, path)

    assert other_store.get_run(other_run) is None
    assert other_store.messages.count_documents({"run_id": other_run}) == 0


def test_manifest_describes_what_is_inside(store, populated, tmp_path):
    path = str(tmp_path / "archive")
    manifest = archive.export_archive(store, path, [populated], note="for the write-up")

    assert manifest["format"] == archive.FORMAT
    assert manifest["note"] == "for the write-up"
    assert manifest["source_db"] == store.db.name
    assert manifest["runs"][0]["messages"] == store.messages.count_documents(
        {"run_id": populated}
    )
    on_disk = json.load(open(os.path.join(path, "manifest.json")))
    assert on_disk["collections"]["messages"]["count"] == manifest["runs"][0]["messages"]
