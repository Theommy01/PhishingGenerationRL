"""run_loop.py's read-only entry points and their exit codes.

The generating paths are covered through LoopRunner in test_runner.py; what is
tested here is argument handling and what the CLI reports, so nothing is
generated.
"""

import pytest

import run_loop


@pytest.fixture(autouse=True)
def cli_store(store, monkeypatch):
    """Point the CLI at the test's scratch database."""
    monkeypatch.setattr(run_loop, "LoopStore", lambda: store)
    return store


@pytest.fixture
def finished_run(store, prompts, make_records, make_checkpoint):
    """A two-round run, recorded the way LoopRunner records one."""
    run_id = store.create_run(prompts, {"n_samples": 2, "algorithm": "kto"})

    for round_index in range(2):
        checkpoint = store.upsert_checkpoint(
            make_checkpoint(f"round{round_index}", weights=f"adapter {round_index}".encode())
        )
        store.add_messages(
            run_id, round_index, make_records(round_index), checkpoint=checkpoint
        )
        fields = dict(
            checkpoint_path=checkpoint["path"],
            checkpoint=store.checkpoint_ref(checkpoint),
            checkpoint_hash=checkpoint["weights_hash"],
            ref_mode="sft",
            generation={"n_samples": 2, "gen_args": {"max_new_tokens": 256}},
        )
        if round_index:
            pool = store.create_dataset(store.pool_query(run_id, round_index - 1))
            fields.update(
                dataset=store.dataset_ref(pool),
                dataset_size=pool["count"],
                dataset_hash=pool["content_hash"],
                training={"epochs": 1, "seed": 3407},
            )
        store.record_round(run_id, round_index, **fields)

    return run_id


# -- argument parsing -------------------------------------------------------


def test_seed_defaults_to_the_shared_constant():
    from metrics import config

    assert run_loop.parse_args([]).seed == config.DEFAULT_TRAINING_SEED


def test_flags_are_parsed():
    args = run_loop.parse_args(
        ["--algorithm", "bco", "--ref-mode", "previous", "--seed", "7", "--limit", "3"]
    )

    assert (args.algorithm, args.ref_mode, args.seed, args.limit) == ("bco", "previous", 7, 3)


def test_the_expensive_measurements_can_be_skipped():
    """Both cost real time per round, so both need a way off."""
    args = run_loop.parse_args(["--no-drift", "--no-policy-kl"])

    assert args.no_drift and args.no_policy_kl
    assert not run_loop.parse_args([]).no_policy_kl


def test_an_unknown_algorithm_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        run_loop.parse_args(["--algorithm", "dpo"])


# -- --report ---------------------------------------------------------------


def test_report_prints_the_trajectory(finished_run, capsys):
    assert run_loop.main(["--report", str(finished_run)]) == 0

    printed = capsys.readouterr().out
    assert "pool_hash" in printed
    assert "evasion_rate" in printed


def test_report_of_an_unknown_run_fails(capsys):
    assert run_loop.main(["--report", "1"]) == 1
    assert "no such run: 1" in capsys.readouterr().err


# -- --verify ---------------------------------------------------------------


def test_verify_passes_on_an_untouched_run(finished_run, capsys):
    assert run_loop.main(["--verify", str(finished_run)]) == 0

    printed = capsys.readouterr().out
    assert "ckpt_hash" in printed and "seed" in printed


def test_verify_fails_when_a_checkpoint_is_gone(store, finished_run, capsys):
    import shutil

    shutil.rmtree(store.get_round(finished_run, 1)["checkpoint_path"])

    assert run_loop.main(["--verify", str(finished_run)]) == 1
    captured = capsys.readouterr()
    assert "missing" in captured.out
    assert "ckpt_ok" in captured.err


def test_verify_fails_when_the_pool_was_rewritten(store, finished_run, capsys):
    store.messages.update_one(
        {"run_id": finished_run, "round": 0}, {"$set": {"body": "rewritten"}}
    )

    assert run_loop.main(["--verify", str(finished_run)]) == 1
    assert "pool_ok" in capsys.readouterr().err


def test_verify_of_an_unknown_run_fails(capsys):
    assert run_loop.main(["--verify", "1"]) == 1
    assert "no such run: 1" in capsys.readouterr().err


# -- --resume ---------------------------------------------------------------


def test_resume_reads_the_prompts_from_the_database(finished_run, prompts, capsys, monkeypatch):
    """An edited prompts.json must not reach a half-finished run."""

    def fail_if_called(*args, **kwargs):
        raise AssertionError("resume read the prompts file")

    monkeypatch.setattr(run_loop, "load_prompts", fail_if_called)

    captured = {}

    class SpyRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, rounds, run_id=None):
            captured["rounds"] = rounds
            return run_id

    monkeypatch.setattr(run_loop, "LoopRunner", SpyRunner)

    assert run_loop.main(["--resume", str(finished_run), "--rounds", "1"]) == 0
    assert [spec["subject"] for spec in captured["prompts"]] == [
        spec["subject"] for spec in prompts
    ]
    assert "stored subjects" in capsys.readouterr().out


def test_resume_of_an_unknown_run_fails(capsys):
    assert run_loop.main(["--resume", "1", "--rounds", "1"]) == 1
    assert "no such run: 1" in capsys.readouterr().err


def test_an_empty_prompts_file_fails(tmp_path, capsys):
    empty = tmp_path / "empty.json"
    empty.write_text("[]")

    assert run_loop.main(["--prompts", str(empty)]) == 1
    assert "no prompts in" in capsys.readouterr().err


def test_an_impossible_decoding_combination_is_a_usage_error(capsys):
    """Greedy plus several samples: reported, not a traceback."""
    code = run_loop.main(["--greedy", "--n-samples", "4", "--limit", "2"])

    assert code == 2
    assert "requires sampling" in capsys.readouterr().err
