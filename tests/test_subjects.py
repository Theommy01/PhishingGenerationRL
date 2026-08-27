"""Subjects: content addressing, the DBRef on a message, and the join back."""

import pytest
from bson import DBRef

from loop.store import SUBJECT_FIELDS, subject_document, subject_fingerprint


# -- identity ---------------------------------------------------------------


def test_fingerprint_ignores_key_order_and_unknown_keys(prompts):
    spec = prompts[0]
    reordered = dict(reversed(list(spec.items())))
    annotated = dict(spec, notes="a key the model never reads")

    assert subject_fingerprint(reordered) == subject_fingerprint(spec)
    assert subject_fingerprint(annotated) == subject_fingerprint(spec)
    assert "notes" not in subject_document(annotated)


@pytest.mark.parametrize("field", SUBJECT_FIELDS)
def test_fingerprint_covers_every_spec_field(prompts, field):
    """Every field of the spec is part of identity, not just the subject line."""
    spec = prompts[0]
    changed = dict(spec)
    changed[field] = (
        not spec[field] if isinstance(spec[field], bool) else f"{spec[field]} (changed)"
    )

    assert subject_fingerprint(changed) != subject_fingerprint(spec)


def test_incomplete_spec_is_refused(prompts):
    with pytest.raises(ValueError, match="missing"):
        subject_fingerprint({"subject": prompts[0]["subject"]})


def test_upsert_is_idempotent(store, prompts):
    first = store.upsert_subject(prompts[0])
    second = store.upsert_subject(dict(prompts[0]))

    assert first == second
    assert store.subjects.count_documents({}) == 1


def test_flipping_a_flag_writes_a_new_subject(store, prompts):
    """Specs are never edited in place: an old run's refs must stay valid."""
    original = store.upsert_subject(prompts[0])
    variant = store.upsert_subject(dict(prompts[0], urls=not prompts[0]["urls"]))

    assert variant != original
    assert store.subjects.count_documents({}) == 2
    assert store.get_subject(original)["urls"] == prompts[0]["urls"]


def test_two_runs_share_one_set_of_subjects(store, prompts):
    store.create_run(prompts, {})
    store.create_run(prompts, {})

    assert store.subjects.count_documents({}) == len(prompts)


# -- the run's ordered refs -------------------------------------------------


def test_run_records_refs_in_prompt_order(store, prompts):
    run_id = store.create_run(prompts, {})
    refs = store.run_subject_refs(run_id)

    assert len(refs) == len(prompts)
    assert all(isinstance(ref, DBRef) for ref in refs)
    assert [store.get_subject(ref)["subject"] for ref in refs] == [
        spec["subject"] for spec in prompts
    ]


def test_run_prompts_round_trips(store, prompts):
    """A resumed run rebuilds its specs from the database, not from the file."""
    run_id = store.create_run(prompts, {})
    recovered = store.run_prompts(run_id)

    assert recovered == [{field: spec[field] for field in SUBJECT_FIELDS} for spec in prompts]
    store.check_prompts(run_id, recovered)  # must not raise


def test_check_prompts_rejects_an_edited_spec(store, prompts):
    run_id = store.create_run(prompts, {})
    edited = [dict(prompts[0], urls=not prompts[0]["urls"])] + prompts[1:]

    with pytest.raises(ValueError, match="prompt 0 changed"):
        store.check_prompts(run_id, edited)


def test_check_prompts_rejects_a_different_count(store, prompts):
    run_id = store.create_run(prompts, {})

    with pytest.raises(ValueError, match="was created with"):
        store.check_prompts(run_id, prompts[:2])


def test_check_prompts_rejects_reordered_prompts(store, prompts):
    """prompt_id indexes into this list, so order is part of the contract."""
    run_id = store.create_run(prompts, {})

    with pytest.raises(ValueError, match="changed since run"):
        store.check_prompts(run_id, list(reversed(prompts)))


# -- messages ---------------------------------------------------------------


def test_message_stores_a_ref_not_a_copy(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())

    raw = store.messages.find_one({"run_id": run_id}, {"_id": 0})
    assert isinstance(raw["subject"], DBRef)
    assert "category" not in raw
    assert "generator" not in raw
    assert store.get_subject(raw["subject"])["subject"] == prompts[raw["prompt_id"]]["subject"]


def test_out_of_range_prompt_id_is_refused(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    record = dict(make_records()[0], prompt_id=len(prompts))

    with pytest.raises(ValueError, match="not a prompt of run"):
        store.add_messages(run_id, 0, [record])
    assert store.messages.count_documents({}) == 0


def test_get_messages_joins_the_subject_back_in(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())

    joined = store.get_messages(run_id, round_index=0)[0]
    spec = prompts[joined["prompt_id"]]
    assert joined["subject_text"] == spec["subject"]
    assert joined["category"] == spec["category"]
    assert joined["generator"] == spec["generator"]
    assert isinstance(joined["subject"], DBRef)


def test_with_subject_false_skips_the_join(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())

    lean = store.get_messages(run_id, round_index=0, with_subject=False)[0]
    assert "category" not in lean
    assert "subject_text" not in lean


def test_editing_a_subject_shows_through_to_old_messages(store, prompts, make_records):
    """The point of the ref: one edit, not one per message."""
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())

    store.subjects.update_one(
        {"spec_hash": subject_fingerprint(prompts[0])},
        {"$set": {"category": "Recategorised"}},
    )

    joined = store.get_messages(run_id, round_index=0)
    assert all(
        message["category"] == "Recategorised"
        for message in joined
        if message["prompt_id"] == 0
    )


def test_messages_for_subject_spans_runs(store, prompts, make_records):
    run_a = store.create_run(prompts, {})
    run_b = store.create_run(prompts, {})
    store.add_messages(run_a, 0, make_records())
    store.add_messages(run_b, 0, make_records())

    subject_id = store.subjects.find_one({"spec_hash": subject_fingerprint(prompts[0])})["_id"]
    across = store.messages_for_subject(subject_id)
    within = store.messages_for_subject(subject_id, run_id=run_a)

    assert len(across) == 4  # 2 samples x 2 runs
    assert len(within) == 2
    assert {message["prompt_id"] for message in across} == {0}


# -- housekeeping -----------------------------------------------------------


def test_drop_run_leaves_shared_subjects_alone(store, prompts, make_records):
    run_a = store.create_run(prompts, {})
    run_b = store.create_run(prompts, {})
    store.add_messages(run_b, 0, make_records())

    store.drop_run(run_b)

    assert store.subjects.count_documents({}) == len(prompts)
    assert store.messages.count_documents({}) == 0
    assert store.get_run(run_a) is not None


def test_prune_subjects_removes_only_unreferenced_ones(store, prompts):
    store.create_run(prompts, {})
    store.upsert_subject(dict(prompts[0], urls=not prompts[0]["urls"]))

    assert store.prune_subjects() == 1
    assert store.subjects.count_documents({}) == len(prompts)


def test_run_ids_do_not_collide_within_a_second(store, prompts):
    """run_id is a unix timestamp, and it is a unique index."""
    ids = [store.create_run(prompts, {}) for _ in range(3)]

    assert len(set(ids)) == 3
    assert store.runs.count_documents({}) == 3
