"""Prompt structure: the shape the model was shown, values stripped."""

import pytest

from loop.store import prompt_structure, structure_fingerprint
from tests.conftest import render

RENDERED = (
    "subject: URGENT: Immediate action required to avoid service interruption.\n"
    "urls: True\n"
    "attachments: False\n"
    "sentiment: urgent, warning\n"
    "->"
)
STRUCTURE = "subject:\nurls:\nattachments:\nsentiment:\n->"


def test_structure_is_field_names_and_markers():
    assert prompt_structure(RENDERED) == STRUCTURE


def test_values_do_not_reach_the_structure(prompts):
    """Including a subject line that itself contains a colon."""
    structures = {prompt_structure(render(spec)) for spec in prompts}

    assert structures == {STRUCTURE}


def test_a_wrapped_value_cannot_leak_into_the_structure():
    wrapped = "subject: first line\nsecond line is content\nurls: True\n->"

    structure = prompt_structure(wrapped)
    assert "second line" not in structure
    assert structure == "subject:\n...\nurls:\n->"


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("subject: x\nattachments: False\nurls: True\n->", id="reordered"),
        pytest.param("subject: x\nurls: True\ntone: urgent\n->", id="renamed field"),
        pytest.param("subject: x\nurls: True\n->", id="dropped field"),
        pytest.param("subject: x\nurls: True\nattachments: False", id="no marker"),
        pytest.param("subject: x\nurls: True\n=>", id="different marker"),
    ],
)
def test_a_template_change_changes_the_structure(variant):
    reference = "subject: x\nurls: True\nattachments: False\n->"

    assert prompt_structure(variant) != prompt_structure(reference)
    assert structure_fingerprint(prompt_structure(variant)) != structure_fingerprint(
        prompt_structure(reference)
    )


def test_blank_lines_and_padding_are_ignored():
    padded = "  subject: x  \n\n   urls: True\n\n->  \n"

    assert prompt_structure(padded) == "subject:\nurls:\n->"


# -- the guard on a run -----------------------------------------------------


def test_the_first_round_records_the_structure(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    assert store.get_run(run_id)["prompt_structures"] is None

    store.add_messages(run_id, 0, make_records())

    recorded = store.get_run(run_id)["prompt_structures"]
    assert [entry["structure"] for entry in recorded] == [STRUCTURE]
    assert recorded[0]["hash"] == structure_fingerprint(STRUCTURE)


def test_every_message_carries_its_structure_hash(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())

    hashes = store.messages.distinct("prompt_structure_hash", {"run_id": run_id})
    assert hashes == [structure_fingerprint(STRUCTURE)]


def test_a_later_round_may_not_introduce_a_new_shape(store, prompts, make_records):
    """What a hash of the specs cannot catch: generate_prompt changing."""
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())

    changed = "subject: x\nurls: True\nattachments: False\ntone: urgent\n->"
    with pytest.raises(ValueError, match="not comparable as model inputs"):
        store.add_messages(run_id, 1, make_records(1, prompt_text=changed))


def test_a_refused_round_does_not_land(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())

    with pytest.raises(ValueError):
        store.add_messages(run_id, 1, make_records(1, prompt_text="a: 1\nb: 2\n->"))

    assert store.messages.count_documents({"run_id": run_id, "round": 1}) == 0


def test_a_mixed_prompt_set_is_allowed_and_pinned(store, prompts, make_records):
    """generate_prompt drops the sentiment line when sentiment is empty."""
    run_id = store.create_run(prompts, {})
    records = make_records()
    records[0]["prompt_text"] = "subject: x\nurls: True\nattachments: False\n->"
    store.add_messages(run_id, 0, records)

    recorded = store.get_run(run_id)["prompt_structures"]
    assert len(recorded) == 2

    # both shapes stay allowed in later rounds; a third does not
    store.add_messages(run_id, 1, make_records(1))
    with pytest.raises(ValueError):
        store.add_messages(run_id, 2, make_records(2, prompt_text="q: 1\n->"))


def test_the_structure_check_is_independent_of_the_specs(store, prompts, make_records):
    """Same specs, different rendering — the spec check alone would pass."""
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())
    store.check_prompts(run_id, prompts)  # unchanged, so this passes

    restyled = [
        dict(record, prompt_text=render(spec).replace("urls:", "links:"))
        for record, spec in zip(make_records(1), prompts * 2)
    ]
    with pytest.raises(ValueError, match="not comparable"):
        store.add_messages(run_id, 1, restyled)
