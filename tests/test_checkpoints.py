"""Checkpoints: which adapter generated a message, and is it still that one."""

import json
import os
import shutil

import pytest
from bson import DBRef, ObjectId

from loop.store import CHECKPOINT_IDENTITY_FILES, checkpoint_files, checkpoint_fingerprint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_CHECKPOINT = os.path.join(REPO_ROOT, "checkpoint-2122")


# -- identity ---------------------------------------------------------------


def test_identity_covers_weights_config_and_tokenizer(make_checkpoint):
    path = make_checkpoint()
    names = [entry["name"] for entry in checkpoint_files(path)]

    assert "adapter_model.safetensors" in names
    assert "adapter_config.json" in names
    assert set(names) <= set(CHECKPOINT_IDENTITY_FILES)


def test_training_state_is_not_part_of_identity(store, make_checkpoint):
    """Optimizer and RNG state differ without changing what is generated."""
    path = make_checkpoint()
    before = store.upsert_checkpoint(path)

    with open(os.path.join(path, "optimizer.pt"), "wb") as handle:
        handle.write(os.urandom(64))

    assert store.verify_checkpoint(before["_id"])["ok"]


def test_different_weights_are_a_different_checkpoint(store, make_checkpoint):
    first = store.upsert_checkpoint(make_checkpoint("a", weights=b"one"))
    second = store.upsert_checkpoint(make_checkpoint("b", weights=b"two"))

    assert first["_id"] != second["_id"]
    assert first["weights_hash"] != second["weights_hash"]


def test_the_same_adapter_at_two_paths_is_one_checkpoint(store, make_checkpoint, tmp_path):
    original = make_checkpoint("original")
    copy = str(tmp_path / "copy")
    shutil.copytree(original, copy)

    first = store.upsert_checkpoint(original)
    second = store.upsert_checkpoint(copy)

    assert first["_id"] == second["_id"]
    assert sorted(second["paths"]) == sorted([original, copy])
    assert store.checkpoints.count_documents({}) == 1


def test_a_missing_directory_is_still_recorded(store, tmp_path):
    document = store.upsert_checkpoint(str(tmp_path / "never-existed"))

    assert document["weights_hash"] is None
    assert document["key"].startswith("path:")
    assert document["files"] == []


def test_two_missing_paths_do_not_collide(store, tmp_path):
    first = store.upsert_checkpoint(str(tmp_path / "gone-a"))
    second = store.upsert_checkpoint(str(tmp_path / "gone-b"))

    assert first["_id"] != second["_id"]


def test_base_model_is_read_from_the_adapter_config(store, make_checkpoint):
    path = make_checkpoint(base_model="unsloth/Meta-Llama-3.1-8B-bnb-4bit")

    assert store.upsert_checkpoint(path)["base_model"] == "unsloth/Meta-Llama-3.1-8B-bnb-4bit"


def test_an_unreadable_config_does_not_break_the_upsert(store, make_checkpoint):
    path = make_checkpoint()
    with open(os.path.join(path, "adapter_config.json"), "w") as handle:
        handle.write("{not json")

    document = store.upsert_checkpoint(path)
    assert document["base_model"] is None
    assert document["weights_hash"] is not None


def test_produced_by_is_not_overwritten_by_a_later_run(store, make_checkpoint):
    path = make_checkpoint()
    first = store.upsert_checkpoint(path, produced_by={"run_id": 1, "round": 2})
    second = store.upsert_checkpoint(path, produced_by={"run_id": 99, "round": 0})

    assert second["_id"] == first["_id"]
    assert second["produced_by"] == {"run_id": 1, "round": 2}


def test_fingerprint_of_nothing_is_none():
    assert checkpoint_fingerprint([]) is None


@pytest.mark.skipif(
    not os.path.isdir(SFT_CHECKPOINT), reason="checkpoint-2122 is gitignored"
)
def test_a_real_adapter_hashes(store):
    document = store.upsert_checkpoint(SFT_CHECKPOINT)

    assert document["weights_hash"]
    assert document["base_model"] == "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    assert {entry["name"] for entry in document["files"]} == set(CHECKPOINT_IDENTITY_FILES)
    assert store.verify_checkpoint(document["_id"])["ok"]


# -- provenance on a message ------------------------------------------------


def test_messages_are_stamped_with_the_checkpoint(store, prompts, make_records, make_checkpoint):
    run_id = store.create_run(prompts, {})
    path = make_checkpoint()
    store.add_messages(run_id, 0, make_records(), checkpoint=path)

    message = store.messages.find_one({"run_id": run_id}, {"_id": 0})
    assert isinstance(message["checkpoint"], DBRef)
    assert message["checkpoint_hash"] == store.upsert_checkpoint(path)["weights_hash"]


def test_checkpoint_can_be_given_as_path_document_or_ref(
    store, prompts, make_records, make_checkpoint
):
    run_id = store.create_run(prompts, {})
    path = make_checkpoint()
    document = store.upsert_checkpoint(path)

    for round_index, form in enumerate([path, document, store.checkpoint_ref(document)]):
        store.add_messages(run_id, round_index, make_records(round_index), checkpoint=form)

    hashes = store.messages.distinct("checkpoint_hash", {"run_id": run_id})
    assert hashes == [document["weights_hash"]]


def test_messages_without_a_checkpoint_are_unstamped(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())

    message = store.messages.find_one({"run_id": run_id}, {"_id": 0})
    assert "checkpoint" not in message
    assert store.checkpoint_for_message(message) is None


def test_checkpoint_for_message_resolves(store, prompts, make_records, make_checkpoint):
    run_id = store.create_run(prompts, {})
    path = make_checkpoint()
    store.add_messages(run_id, 0, make_records(), checkpoint=path)

    message = store.get_messages(run_id, round_index=0)[0]
    assert store.checkpoint_for_message(message)["path"] == path


def test_messages_for_checkpoint_splits_by_adapter(
    store, prompts, make_records, make_checkpoint
):
    run_id = store.create_run(prompts, {})
    first = make_checkpoint("first", weights=b"one")
    second = make_checkpoint("second", weights=b"two")
    store.add_messages(run_id, 0, make_records(0), checkpoint=first)
    store.add_messages(run_id, 1, make_records(1), checkpoint=second)

    assert len(store.messages_for_checkpoint(first)) == 8
    assert len(store.messages_for_checkpoint(second)) == 8
    assert {m["round"] for m in store.messages_for_checkpoint(second)} == {1}


def test_messages_for_an_unknown_checkpoint_is_empty(store):
    assert store.messages_for_checkpoint(ObjectId()) == []


# -- verification -----------------------------------------------------------


def test_verify_passes_on_an_untouched_checkpoint(store, make_checkpoint):
    document = store.upsert_checkpoint(make_checkpoint())

    check = store.verify_checkpoint(document["_id"])
    assert check["ok"] and check["present"]
    assert check["recorded"] == check["actual"]


def test_verify_catches_an_overwritten_path(store, make_checkpoint):
    """In place, in the same session: the digest cache must not hide this."""
    path = make_checkpoint(weights=b"original")
    document = store.upsert_checkpoint(path)

    make_checkpoint(weights=b"a different adapter")  # same name, new contents

    check = store.verify_checkpoint(document["_id"])
    assert not check["ok"]
    assert check["present"]
    assert check["actual"] != check["recorded"]


def test_verify_catches_a_deleted_checkpoint(store, make_checkpoint):
    path = make_checkpoint()
    document = store.upsert_checkpoint(path)
    shutil.rmtree(path)

    check = store.verify_checkpoint(document["_id"])
    assert not check["ok"]
    assert not check["present"]
    assert check["actual"] is None


def test_verify_can_be_pointed_at_a_restored_copy(store, make_checkpoint, tmp_path):
    path = make_checkpoint("original")
    document = store.upsert_checkpoint(path)
    backup = str(tmp_path / "backup")
    shutil.copytree(path, backup)
    shutil.rmtree(path)

    assert not store.verify_checkpoint(document["_id"])["ok"]
    assert store.verify_checkpoint(document["_id"], path=backup)["ok"]


def test_verify_of_an_unknown_checkpoint_raises(store):
    with pytest.raises(KeyError):
        store.verify_checkpoint(ObjectId())


def test_digests_are_cached_but_invalidate_on_a_rewrite(store, make_checkpoint):
    path = make_checkpoint(weights=b"first")
    store.upsert_checkpoint(path)
    assert len(store._digest_cache) == 1

    store.upsert_checkpoint(path)  # unchanged: served from the cache
    assert len(store._digest_cache) == 1

    make_checkpoint(weights=b"second")  # rewritten in place: must re-hash
    store.upsert_checkpoint(path)
    assert len(store._digest_cache) == 2
