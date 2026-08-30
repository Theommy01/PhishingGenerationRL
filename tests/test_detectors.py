"""The detector registry, and the transfer analysis that needs two of them."""

import pandas as pd
import pytest

from detectors import base


@pytest.fixture
def registry(monkeypatch):
    """An isolated registry, so tests do not disturb the real one."""
    monkeypatch.setattr(base, "_REGISTRY", {})
    return base


class Stub:
    """A detector that calls a message evaded when its body contains a marker."""

    def __init__(self, marker="EVADE", built=None):
        self.marker = marker
        self.calls = 0
        if built is not None:
            built.append(self)

    def score_messages(self, bodies):
        self.calls += 1
        return [0.9 if self.marker in b else 0.1 for b in bodies]

    def label_messages(self, bodies):
        return [s >= 0.5 for s in self.score_messages(bodies)]


def test_a_detector_is_built_once_and_only_when_used(registry):
    built = []
    registry.register(
        base.DetectorSpec(name="stub", build=lambda: Stub(built=built), description="")
    )

    assert built == [], "registering must not build"
    registry.get_detector("stub")
    registry.get_detector("stub")
    assert len(built) == 1, "built more than once"


def test_unavailable_detectors_are_registered_but_not_listed(registry):
    registry.register(
        base.DetectorSpec(name="here", build=Stub, description="")
    )
    registry.register(
        base.DetectorSpec(
            name="absent", build=Stub, description="", is_available=lambda: False
        )
    )

    assert sorted(registry.registry()) == ["absent", "here"]
    assert registry.available() == ["here"]


def test_the_in_loop_detector_is_identified(registry):
    registry.register(base.DetectorSpec(name="held", build=Stub, description=""))
    registry.register(
        base.DetectorSpec(name="reward", build=Stub, description="", in_loop=True)
    )

    assert registry.in_loop_detector() == "reward"


def test_an_unknown_detector_says_what_it_has(registry):
    registry.register(base.DetectorSpec(name="stub", build=Stub, description=""))

    with pytest.raises(KeyError, match="stub"):
        registry.get_detector("nope")


def test_score_with_runs_several_detectors(registry):
    registry.register(base.DetectorSpec(name="a", build=lambda: Stub("EVADE"), description=""))
    registry.register(base.DetectorSpec(name="b", build=lambda: Stub("OTHER"), description=""))

    out = registry.score_with(["a", "b"], ["please EVADE now", "plain text"])
    assert out["a"]["labels"] == [True, False]
    assert out["b"]["labels"] == [False, False]


def test_the_reward_can_come_from_any_registered_detector(registry):
    """`score_with_detector` is what makes the in-loop detector a parameter."""
    from loop.runner import score_with_detector

    registry.register(
        base.DetectorSpec(name="stub", build=lambda: Stub("EVADE"), description="")
    )
    records = [{"body": "please EVADE now"}, {"body": "plain text"}]

    scored = score_with_detector("stub")(records, 0.5)

    assert [r["score"] for r in scored] == [0.9, 0.1]
    assert [r["label"] for r in scored] == [True, False]


def test_the_reward_label_uses_the_runs_threshold_not_the_detectors(registry):
    """One run, one threshold, whatever default a detector carries."""
    from loop.runner import score_with_detector

    registry.register(
        base.DetectorSpec(name="stub", build=lambda: Stub("EVADE"), description="")
    )

    scored = score_with_detector("stub")([{"body": "please EVADE now"}], 0.95)

    assert scored[0]["label"] is False, "0.9 is below the 0.95 asked for"


def test_an_unavailable_detector_cannot_supply_the_reward(registry):
    from loop.runner import score_with_detector

    registry.register(
        base.DetectorSpec(
            name="absent", build=Stub, description="", is_available=lambda: False
        )
    )

    with pytest.raises(RuntimeError, match="not available"):
        score_with_detector("absent")([{"body": "anything"}], 0.5)


def test_the_real_registry_has_one_in_loop_detector():
    import detectors

    specs = detectors.registry()
    assert sum(spec.in_loop for spec in specs.values()) == 1
    assert detectors.in_loop_detector() == "scamllm"
    assert "bert-phishing" in specs


# -- storage ----------------------------------------------------------------


def test_verdicts_are_stored_per_detector(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())

    verdicts = [
        {"round": 0, "prompt_id": m["prompt_id"], "sample_idx": m["sample_idx"],
         "score": 0.7, "label": True}
        for m in store.get_messages(run_id, with_subject=False)
    ]
    store.set_detector_verdicts(run_id, "bert-phishing", verdicts)

    assert store.scored_detectors(run_id) == ["bert-phishing"]
    assert store.messages_missing_detector(run_id, "bert-phishing") == []
    assert len(store.messages_missing_detector(run_id, "svm")) == len(verdicts)

    message = store.get_messages(run_id, with_subject=False)[0]
    assert message["detector_scores"]["bert-phishing"] == 0.7
    assert message["detector_labels"]["bert-phishing"] is True


def test_adding_a_detector_does_not_disturb_another(store, prompts, make_records):
    run_id = store.create_run(prompts, {})
    store.add_messages(run_id, 0, make_records())
    keys = [
        {"round": 0, "prompt_id": m["prompt_id"], "sample_idx": m["sample_idx"]}
        for m in store.get_messages(run_id, with_subject=False)
    ]

    store.set_detector_verdicts(run_id, "one", [{**k, "score": 0.2, "label": False} for k in keys])
    store.set_detector_verdicts(run_id, "two", [{**k, "score": 0.8, "label": True} for k in keys])

    scores = store.get_messages(run_id, with_subject=False)[0]["detector_scores"]
    assert scores == {"one": 0.2, "two": 0.8}
    assert store.scored_detectors(run_id) == ["one", "two"]
