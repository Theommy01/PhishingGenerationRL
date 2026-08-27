"""The loop end to end, with generation, scoring and training stubbed out.

No GPU and no models: the point is the sequence and what it records, not what
the adapters learn.
"""

import pytest

from loop import report
from loop.runner import LoopRunner
from tests.conftest import render


@pytest.fixture
def stubs(make_checkpoint):
    """Stand-ins for generate/score/train that record how they were called."""

    class Stubs:
        def __init__(self):
            self.calls = []
            self.sft = make_checkpoint("sft", weights=b"the sft adapter")

        def generate(self, checkpoint, prompts, gen_args, n_samples, on_prompt=None):
            self.calls.append(("generate", checkpoint, dict(gen_args), n_samples))
            produced = [
                {
                    "prompt_id": prompt_id,
                    "sample_idx": sample_idx,
                    "prompt_text": render(spec),
                    "body": f"{checkpoint}::{prompt_id}::{sample_idx}::{len(self.calls)}",
                    "category": spec["category"],
                    "generator": spec["generator"],
                }
                for prompt_id, spec in enumerate(prompts)
                for sample_idx in range(n_samples)
            ]
            if on_prompt is not None:
                for prompt_id in range(len(prompts)):
                    on_prompt([r for r in produced if r["prompt_id"] == prompt_id])
            return produced

        def score(self, records, threshold):
            for index, record in enumerate(records):
                record["score"] = 0.3 + 0.2 * (index % 2)
                record["label"] = bool(record["score"] >= threshold)
            return records

        def train(self, algorithm, base, dataset_path, output_dir, ref_mode, sft_path, epochs, seed):
            with open(dataset_path) as handle:
                rows = len(handle.read().strip().splitlines())
            self.calls.append(("train", algorithm, base, rows, epochs, seed))
            checkpoint = make_checkpoint(
                output_dir.rsplit("/", 1)[-1], weights=f"{seed}:{rows}:{base}".encode()
            )
            # what TRL's log history summarises to, for KTO
            return checkpoint, {"steps": rows, "loss_final": 0.5, "kl_mean": 0.02 * rows}

        def policy_kl(self, records, checkpoint_path, reference_path):
            self.calls.append(("policy_kl", checkpoint_path, reference_path))
            for index, record in enumerate(records):
                record["logratio_per_token"] = -0.01 * (index + 1)
                record["kl_k3_per_token"] = 0.005 * (index + 1)
            return records

    return Stubs()


@pytest.fixture
def runner_for(store, prompts, stubs):
    def build(**overrides):
        settings = dict(
            prompts=prompts,
            store=store,
            n_samples=2,
            gen_args={"do_sample": True, "max_new_tokens": 256},
            generate_fn=stubs.generate,
            score_fn=stubs.score,
            train_fn=stubs.train,
            policy_kl_fn=stubs.policy_kl,
            measure_drift=False,
            sft_path=stubs.sft,
        )
        settings.update(overrides)
        return LoopRunner(**settings)

    return build


# -- validation -------------------------------------------------------------


def test_unknown_algorithm_is_refused(runner_for):
    with pytest.raises(ValueError, match="unknown algorithm"):
        runner_for(algorithm="dpo")


def test_unknown_ref_mode_is_refused(runner_for):
    with pytest.raises(ValueError, match="unknown ref_mode"):
        runner_for(ref_mode="sideways")


def test_greedy_decoding_with_several_samples_is_refused(runner_for):
    """It would return n identical messages."""
    with pytest.raises(ValueError, match="requires sampling"):
        runner_for(n_samples=4, gen_args={"do_sample": False})


# -- a round ----------------------------------------------------------------


def test_round_zero_is_a_baseline(store, runner_for, prompts, stubs):
    run_id = runner_for().start()

    round_zero = store.get_round(run_id, 0)
    assert round_zero["base_checkpoint"] is None
    assert round_zero["checkpoint_path"] == stubs.sft
    assert round_zero["dataset"] is None
    assert round_zero["training"] is None
    assert round_zero["dataset_size"] == 0
    assert not any(call[0] == "train" for call in stubs.calls)
    assert store.messages.count_documents({"run_id": run_id}) == len(prompts) * 2


def test_a_training_round_pins_its_pool(store, runner_for, prompts):
    runner = runner_for()
    run_id = runner.start()
    runner.step(run_id)

    round_one = store.get_round(run_id, 1)
    dataset = store.get_dataset(round_one["dataset"])
    assert dataset["count"] == len(prompts) * 2  # round 0's messages only
    assert round_one["dataset_hash"] == dataset["content_hash"]
    assert store.verify_dataset(dataset["_id"])["ok"]


def test_the_pool_is_cumulative(store, runner_for, prompts, stubs):
    runner = runner_for()
    run_id = runner.run(rounds=2)

    trained = [call for call in stubs.calls if call[0] == "train"]
    assert [call[3] for call in trained] == [len(prompts) * 2, len(prompts) * 4]
    assert store.get_round(run_id, 2)["dataset_size"] == len(prompts) * 4


def test_each_round_is_generated_by_its_own_checkpoint(store, runner_for):
    run_id = runner_for().run(rounds=2)

    stamps = []
    for round_index in range(3):
        messages = store.get_messages(run_id, round_index=round_index, with_subject=False)
        round_stamps = {message["checkpoint_hash"] for message in messages}
        assert len(round_stamps) == 1, "a round was generated by more than one adapter"
        stamps.append(round_stamps.pop())

    assert len(set(stamps)) == 3


def test_the_checkpoint_a_round_produced_is_attributed_to_it(store, runner_for):
    run_id = runner_for().run(rounds=1)

    baseline = store.get_checkpoint(store.get_round(run_id, 0)["checkpoint"])
    trained = store.get_checkpoint(store.get_round(run_id, 1)["checkpoint"])

    assert baseline["produced_by"] is None, "the SFT adapter is not this run's work"
    assert trained["produced_by"] == {"run_id": run_id, "round": 1}


def test_the_seed_reaches_the_trainer_and_the_round(store, runner_for, stubs):
    run_id = runner_for(seed=4242, epochs=2).run(rounds=1)

    trained = [call for call in stubs.calls if call[0] == "train"][0]
    assert trained[4:] == (2, 4242)
    training = store.get_round(run_id, 1)["training"]
    assert training["epochs"] == 2 and training["seed"] == 4242


def test_the_training_kl_is_captured_on_the_round(store, runner_for):
    """TRL logs it every step and then drops it; the round keeps it."""
    run_id = runner_for().run(rounds=1)

    training = store.get_round(run_id, 1)["training"]
    assert training["kl_mean"] == pytest.approx(0.02 * 8)
    assert training["loss_final"] == 0.5
    assert training["steps"] == 8


def test_a_train_fn_returning_only_a_path_still_works(store, runner_for, make_checkpoint):
    """The stats are an addition; an older train_fn must not break."""

    def legacy_train(algorithm, base, dataset_path, output_dir, ref_mode, sft_path, epochs, seed):
        return make_checkpoint(output_dir.rsplit("/", 1)[-1], weights=b"legacy")

    run_id = runner_for(train_fn=legacy_train).run(rounds=1)

    training = store.get_round(run_id, 1)["training"]
    assert training == {"epochs": 3, "seed": 3407}


def test_policy_kl_is_measured_against_the_sft_baseline(store, runner_for, stubs):
    """Anchored to SFT whatever ref_mode training used, so rounds compare."""
    run_id = runner_for(ref_mode="previous").run(rounds=2)

    calls = [call for call in stubs.calls if call[0] == "policy_kl"]
    assert [call[2] for call in calls] == [stubs.sft, stubs.sft, stubs.sft]

    messages = store.get_messages(run_id, round_index=2, with_subject=False)
    assert all(m["logratio_per_token"] is not None for m in messages)
    assert store.get_round(run_id, 2)["metrics"]["logratio_per_token"] is not None


def test_the_scorers_are_evicted_before_the_kl_pass(store, runner_for, monkeypatch):
    """It loads an 8B adapter plus the reference; the card has to be free."""
    order = []

    from loop.runner import LoopRunner

    monkeypatch.setattr(
        LoopRunner, "free_auxiliary_models", lambda self: order.append("free")
    )

    runner = runner_for(policy_kl_fn=lambda records, *_: order.append("kl") or records)
    runner.run(rounds=1)

    # once per round before the KL pass, plus once before training
    assert order[: order.index("kl") + 1][-2:] == ["free", "kl"]


def test_policy_kl_can_be_turned_off(store, runner_for, stubs):
    runner_for(measure_policy_kl=False).run(rounds=1)

    assert not [call for call in stubs.calls if call[0] == "policy_kl"]


def test_every_round_records_how_it_generated(store, runner_for):
    run_id = runner_for(n_samples=3, gen_args={"do_sample": True, "max_new_tokens": 128}).run(
        rounds=1
    )

    for round_index in (0, 1):
        generation = store.get_round(run_id, round_index)["generation"]
        assert generation["n_samples"] == 3
        assert generation["gen_args"]["max_new_tokens"] == 128


def test_messages_are_stored_as_they_are_generated(store, runner_for, prompts):
    """Generation is the expensive part; a crash after it must not lose it."""
    seen = []

    def generate(checkpoint, specs, gen_args, n_samples, on_prompt=None):
        produced = []
        for prompt_id, spec in enumerate(specs):
            batch = [
                {
                    "prompt_id": prompt_id,
                    "sample_idx": i,
                    "prompt_text": render(spec),
                    "body": f"p{prompt_id} s{i}",
                }
                for i in range(n_samples)
            ]
            produced += batch
            if on_prompt is not None:
                on_prompt(batch)
                # what the database holds *during* generation
                seen.append(store.messages.count_documents({}))
        return produced

    runner_for(generate_fn=generate).start()

    assert seen == [2, 4, 6, 8], "messages did not accumulate per prompt"


def test_messages_are_stored_unscored_then_filled_in(store, runner_for, prompts):
    """The two phases: generation persists, scoring updates in place."""
    during = {}

    def generate(checkpoint, specs, gen_args, n_samples, on_prompt=None):
        produced = []
        for prompt_id, spec in enumerate(specs):
            batch = [
                {
                    "prompt_id": prompt_id,
                    "sample_idx": 0,
                    "prompt_text": render(spec),
                    "body": f"p{prompt_id}",
                }
            ]
            produced += batch
            if on_prompt is not None:
                on_prompt(batch)
                during["unscored"] = store.messages.count_documents(
                    {"score": {"$exists": False}}
                )
        return produced

    run_id = runner_for(generate_fn=generate, n_samples=1).start()

    assert during["unscored"] > 0, "messages were scored before they were stored"
    assert store.unscored_count(run_id) == 0, "scores were not filled in afterwards"

    message = store.get_messages(run_id, with_subject=False)[0]
    assert "score" in message and "label" in message
    # the insert-time fields survived the update
    assert message["split"] == "train" and message["checkpoint_hash"]


def test_a_generate_fn_without_on_prompt_still_works(store, runner_for):
    """A caller's own generator keeps its four-argument signature."""

    def legacy_generate(checkpoint, specs, gen_args, n_samples):
        return [
            {
                "prompt_id": prompt_id,
                "sample_idx": 0,
                "prompt_text": render(spec),
                "body": f"legacy p{prompt_id}",
            }
            for prompt_id, spec in enumerate(specs)
        ]

    run_id = runner_for(generate_fn=legacy_generate, n_samples=1).start()

    messages = store.get_messages(run_id, with_subject=False)
    assert len(messages) == 4
    assert all("score" in m for m in messages)


def test_a_round_shares_one_timestamp_despite_incremental_writes(store, runner_for):
    run_id = runner_for().start()

    stamps = store.messages.distinct("added_at", {"run_id": run_id})
    assert len(stamps) == 1, "an as_of could cut this round in half"


# -- the held-out split -----------------------------------------------------


def test_held_out_prompts_are_generated_but_never_trained_on(store, runner_for, all_prompts):
    """The whole point: scored every round, absent from every pool."""
    held = [dict(spec) for spec in all_prompts[40:42]]
    runner = runner_for(holdout=held)
    run_id = runner.run(rounds=1)

    from loop.store import HOLDOUT_SPLIT, TRAIN_SPLIT

    held_messages = store.get_messages(run_id, split=HOLDOUT_SPLIT, with_subject=False)
    train_messages = store.get_messages(run_id, split=TRAIN_SPLIT, with_subject=False)
    assert len(held_messages) == 2 * 2 * 2  # 2 prompts x 2 samples x 2 rounds
    assert len(train_messages) == 4 * 2 * 2

    pool = store.training_pool(run_id, max_round=1)
    held_bodies = {m["body"] for m in held_messages}
    assert not held_bodies & {row["completion"] for row in pool}
    assert store.get_round(run_id, 1)["dataset_size"] == 8  # round 0's train split only


def test_held_out_prompts_share_the_run_id_space(store, runner_for, all_prompts):
    held = [dict(spec) for spec in all_prompts[40:42]]
    run_id = runner_for(holdout=held).start()

    from loop.store import HOLDOUT_SPLIT

    ids = {m["prompt_id"] for m in store.get_messages(run_id, split=HOLDOUT_SPLIT)}
    assert ids == {4, 5}, "held-out ids must continue after the training prompts"
    assert len(store.run_subject_refs(run_id)) == 6

    # and each resolves to the right subject
    message = store.get_messages(run_id, split=HOLDOUT_SPLIT)[0]
    assert message["subject_text"] == held[message["prompt_id"] - 4]["subject"]


def test_the_two_splits_are_scored_separately(store, runner_for, all_prompts):
    held = [dict(spec) for spec in all_prompts[40:42]]
    run_id = runner_for(holdout=held).start()

    record = store.get_round(run_id, 0)
    assert record["metrics"]["messages"] == 8
    assert record["holdout_metrics"]["messages"] == 4


def test_a_run_without_holdout_records_none(store, runner_for):
    run_id = runner_for().start()

    assert store.get_round(run_id, 0)["holdout_metrics"] is None


# -- resuming ---------------------------------------------------------------


def test_a_run_resumes_from_its_stored_subjects(store, runner_for, prompts):
    run_id = runner_for().run(rounds=1)

    recovered = store.run_prompts(run_id)
    runner_for(prompts=recovered).run(rounds=1, run_id=run_id)

    assert store.get_round(run_id, 2) is not None
    assert store.subjects.count_documents({}) == len(prompts)


def test_resuming_with_different_prompts_is_refused(store, runner_for, prompts, all_prompts):
    run_id = runner_for().run(rounds=1)
    others = [dict(spec) for spec in all_prompts[10:14]]

    with pytest.raises(ValueError, match="changed since run"):
        runner_for(prompts=others).step(run_id)


def test_stepping_a_run_with_no_baseline_is_refused(store, runner_for, prompts):
    run_id = store.create_run(prompts, {})

    with pytest.raises(RuntimeError, match="no round 0"):
        runner_for().step(run_id)


def test_config_drift_is_reported_not_refused(store, runner_for, capsys):
    """Changing --n-samples mid-run is legitimate, but never silent."""
    run_id = runner_for(n_samples=2).run(rounds=1)
    capsys.readouterr()

    resumed = runner_for(n_samples=4, gen_args={"do_sample": True, "max_new_tokens": 512})
    drift = resumed.warn_on_config_drift(run_id)
    printed = capsys.readouterr().out

    assert set(drift) == {"n_samples", "gen_args"}
    assert "generates differently" in printed
    assert "n_samples" in printed


def test_a_drifted_round_records_what_it_actually_used(store, runner_for, prompts):
    run_id = runner_for(n_samples=2).run(rounds=1)
    runner_for(n_samples=4, seed=77).run(rounds=1, run_id=run_id)

    assert store.get_round(run_id, 2)["generation"]["n_samples"] == 4
    assert store.get_round(run_id, 2)["training"]["seed"] == 77
    # the run's config is its original intent, and stays put
    assert store.get_run(run_id)["config"]["n_samples"] == 2
    assert len(store.get_messages(run_id, round_index=2)) == len(prompts) * 4


# -- reporting --------------------------------------------------------------


def test_trajectory_has_a_row_per_round(store, runner_for):
    run_id = runner_for().run(rounds=2)

    df = report.trajectory(store, run_id)
    assert list(df["round"]) == [0, 1, 2]
    assert df["pool_hash"].isna().sum() == 1  # only round 0 trains on nothing
    assert df.loc[df["round"] == 2, "pool"].item() > df.loc[df["round"] == 1, "pool"].item()


def test_provenance_passes_on_an_untouched_run(store, runner_for):
    run_id = runner_for().run(rounds=2)

    df = report.provenance(store, run_id)
    assert df["ckpt_ok"].all()
    assert list(df["ckpt_stamps"]) == [1, 1, 1]
    assert df.loc[df["round"] > 0, "pool_ok"].all()
    assert df.loc[df["round"] == 0, "pool_ok"].isna().all()


def test_provenance_flags_a_missing_checkpoint(store, runner_for):
    import shutil

    run_id = runner_for().run(rounds=1)
    shutil.rmtree(store.get_round(run_id, 1)["checkpoint_path"])

    df = report.provenance(store, run_id)
    row = df[df["round"] == 1].iloc[0]
    assert not row["ckpt_ok"]
    assert row["ckpt_note"] == "missing"


def test_provenance_flags_a_rewritten_pool(store, runner_for):
    run_id = runner_for().run(rounds=1)
    store.messages.update_one({"run_id": run_id, "round": 0}, {"$set": {"body": "edited"}})

    df = report.provenance(store, run_id)
    assert not df[df["round"] == 1].iloc[0]["pool_ok"]


def test_round_metrics_are_computed_per_round(store, runner_for, prompts):
    run_id = runner_for().run(rounds=1)

    metrics = store.get_round(run_id, 1)["metrics"]
    assert metrics["messages"] == len(prompts) * 2
    assert metrics["prompts"] == len(prompts)
    assert 0 <= metrics["evasion_rate"] <= 100


def test_load_run_flattens_the_refs(store, runner_for):
    from metrics.analysis import load_run, round_breakdown, round_summary

    run_id = runner_for().run(rounds=1)
    df = load_run(store, run_id)

    assert df["subject_id"].map(type).eq(str).all()
    assert df["checkpoint_id"].map(type).eq(str).all()
    assert "subject" not in df.columns and "checkpoint" not in df.columns
    # the joined spec fields are what the analysis groups by
    assert not round_breakdown(df, group_col="generator").empty
    assert list(round_summary(df)["round"]) == [0, 1]


def test_load_run_of_an_empty_run_is_empty(store, prompts):
    from metrics.analysis import load_run

    run_id = store.create_run(prompts, {})
    assert load_run(store, run_id).empty
