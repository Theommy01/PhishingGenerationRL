"""Shared fixtures.

The tests run against a real mongod on localhost, not a mock. The store is a
thin wrapper over pymongo, so most of what is worth testing *is* server
behaviour: `$`-prefixed keys surviving a round trip, indexes on a DBRef's `$id`
subfield, `$setOnInsert` upserts, BSON dates coming back naive. A mock would
either reimplement or quietly fake all of that.

Each test gets its own scratch database, dropped afterwards, so they neither
see each other nor touch `phishnet_rl`. If mongod is not running the whole
suite skips rather than fails — it is a dependency of the loop, not of anyone
reading the code.
"""

import json
import os
import sys
import uuid

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from loop.store import LoopStore  # noqa: E402  (after the path fix)

PROMPTS_PATH = os.path.join(REPO_ROOT, "prompts.json")


@pytest.fixture(scope="session")
def mongo_client():
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1500)
    try:
        client.server_info()
    except PyMongoError as exc:
        pytest.skip(f"no mongod on localhost:27017 ({type(exc).__name__})")

    yield client
    client.close()


@pytest.fixture
def store(mongo_client):
    """A LoopStore on a scratch database, dropped when the test ends."""
    db_name = f"phishnet_rl_test_{uuid.uuid4().hex[:12]}"
    store = LoopStore(db_name=db_name, client=mongo_client)
    yield store
    mongo_client.drop_database(db_name)


@pytest.fixture(scope="session")
def all_prompts():
    with open(PROMPTS_PATH) as handle:
        return json.load(handle)


@pytest.fixture
def prompts(all_prompts):
    """A handful of real prompt specs — three categories, two generators."""
    return [dict(spec) for spec in all_prompts[:4]]


def render(spec):
    """The prompt text `generate_prompt` produces for a spec, plus the marker.

    Kept here rather than importing phishnet_inference, so the tests exercise
    the store's own parsing without pulling in the generation stack. It has to
    match the real field order — subject, urls, attachments, sentiment — because
    that order is exactly what the structure hash pins.
    """
    return (
        f"subject: {spec['subject']}\n"
        f"urls: {spec['urls']}\n"
        f"attachments: {spec['attachments']}\n"
        f"sentiment: {', '.join(spec['sentiment'])}\n"
        "->"
    )


@pytest.fixture
def make_records(prompts):
    """Build a round's worth of scored records, as the generator would."""

    def build(round_index=0, n_samples=2, prompt_text=None, specs=None):
        specs = specs if specs is not None else prompts
        return [
            {
                "prompt_id": prompt_id,
                "sample_idx": sample_idx,
                "prompt_text": prompt_text or render(spec),
                "body": f"round {round_index} prompt {prompt_id} sample {sample_idx}",
                "score": 0.3 + 0.2 * sample_idx,
                "label": sample_idx > 0,
                # the generator emits these; the store should drop them in
                # favour of the subject ref
                "category": spec["category"],
                "generator": spec["generator"],
            }
            for prompt_id, spec in enumerate(specs)
            for sample_idx in range(n_samples)
        ]

    return build


@pytest.fixture
def make_checkpoint(tmp_path):
    """A directory shaped like a LoRA adapter, without the 168 MB of weights."""

    def build(name="adapter", weights=b"adapter weights", base_model="unsloth/x"):
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        (path / "adapter_model.safetensors").write_bytes(weights)
        (path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": base_model})
        )
        (path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 512}))
        # training state, which must NOT be part of the checkpoint's identity
        (path / "optimizer.pt").write_bytes(os.urandom(64))
        return str(path)

    return build
