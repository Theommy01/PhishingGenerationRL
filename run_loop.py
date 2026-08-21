"""Command-line entry point for the adversarial fine-tuning loop.

One round is: generate n messages per prompt with the current checkpoint,
score them with ScamLLM, add them to the run's cumulative pool, train BCO or
KTO on everything collected so far, and regenerate over the same prompts.

    # baseline plus three KTO rounds over the first 20 prompts
    python run_loop.py --rounds 3 --limit 20

    # BCO, anchoring the KL term to the pinned SFT checkpoint
    python run_loop.py --algorithm bco --ref-mode sft --rounds 3

    # carry an interrupted run on for two more rounds
    python run_loop.py --resume 1786641452 --rounds 2

    # inspect a finished run without generating anything
    python run_loop.py --report 1786641452

Prompt specs are stored as `subjects` documents when a run is created, and the
run keeps an ordered list of DBRefs to them. Resuming reads the prompts back
from those refs rather than from --prompts, so an edited prompts.json cannot
quietly change what a half-finished run measures. The fingerprint of the same
specs is checked on every round as a second guard.
"""

import argparse
import sys

from generate_dataset import DEFAULT_PROMPTS_PATH, load_prompts
from loop import report
from loop.runner import ALGORITHMS, REF_MODES, LoopRunner
from loop.store import LoopStore
from metrics import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the generate -> label -> fine-tune loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--prompts",
        # absolute, so the CLI works from any working directory
        default=DEFAULT_PROMPTS_PATH,
        help="prompt spec file; ignored with --resume, which reads the run's subjects",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use only the first N prompts (a full 150-prompt round is slow)",
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="kto")
    parser.add_argument(
        "--ref-mode",
        choices=REF_MODES,
        default="sft",
        help=(
            "what the KL term is anchored to. 'sft' pins it to the SFT "
            "checkpoint, 'previous' lets it move with each round, 'base' "
            "anchors to the raw pretrained Llama as the notebook did"
        ),
    )
    parser.add_argument("--rounds", type=int, default=1, help="training rounds to run")
    parser.add_argument(
        "--n-samples", type=int, default=4, help="messages generated per prompt"
    )
    parser.add_argument("--epochs", type=int, default=1, help="epochs per round")
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_TRAINING_SEED,
        help=(
            "training seed (data order, dropout), recorded on each round. Does "
            "not make a round bit-reproducible — 4-bit training runs "
            "non-deterministic kernels — but completes the record of its inputs"
        ),
    )
    parser.add_argument(
        "--sft-path",
        default=config.PATH_SFT,
        help="checkpoint round 0 generates from, and the 'sft' KL anchor",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="generation length cap"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="greedy decoding; incompatible with --n-samples > 1, which would "
        "return n identical messages",
    )
    parser.add_argument(
        "--no-drift",
        action="store_true",
        help="skip the SBERT semantic-drift metrics",
    )
    parser.add_argument(
        "--resume", type=int, default=None, metavar="RUN_ID", help="continue a run"
    )
    parser.add_argument(
        "--report",
        type=int,
        default=None,
        metavar="RUN_ID",
        help="print an existing run's trajectory and exit",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=None,
        metavar="RUN_ID",
        help=(
            "check each round's provenance and exit: that the checkpoint on "
            "disk is the one that generated the messages, and that the pool it "
            "trained on still hashes the same"
        ),
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    store = LoopStore()

    if args.report is not None:
        if store.get_run(args.report) is None:
            print(f"no such run: {args.report}", file=sys.stderr)
            return 1
        report.print_trajectory(store, args.report)
        return 0

    if args.verify is not None:
        if store.get_run(args.verify) is None:
            print(f"no such run: {args.verify}", file=sys.stderr)
            return 1
        df = report.print_provenance(store, args.verify)
        # a failed check is the answer to the question, so it sets the exit code
        failed = [
            column
            for column in ("ckpt_ok", "pool_ok")
            if column in df and (df[column] == False).any()  # noqa: E712 — NA-safe
        ]
        if failed:
            print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
            return 1
        return 0

    if args.resume is not None:
        # a resumed run generates from the subjects it was created with, so it
        # keeps measuring what it started measuring whatever prompts.json says now
        if store.get_run(args.resume) is None:
            print(f"no such run: {args.resume}", file=sys.stderr)
            return 1
        prompts = store.run_prompts(args.resume)
        print(f"resuming run {args.resume} from its {len(prompts)} stored subjects")
    else:
        prompts = load_prompts(args.prompts)
        if args.limit is not None:
            prompts = prompts[: args.limit]
        if not prompts:
            print(f"no prompts in {args.prompts}", file=sys.stderr)
            return 1

    gen_args = {"max_new_tokens": args.max_new_tokens, "do_sample": not args.greedy}

    # LoopRunner validates the decoding/n_samples combination up front, so a
    # bad one is a usage error to report plainly rather than a traceback.
    try:
        runner = LoopRunner(
            prompts=prompts,
            store=store,
            algorithm=args.algorithm,
            ref_mode=args.ref_mode,
            n_samples=args.n_samples,
            gen_args=gen_args,
            epochs=args.epochs,
            seed=args.seed,
            sft_path=args.sft_path,
            measure_drift=not args.no_drift,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"{args.algorithm.upper()} / ref_mode={args.ref_mode} / "
        f"{len(prompts)} prompts x {args.n_samples} samples / "
        f"{args.rounds} round(s)"
    )

    run_id = runner.run(rounds=args.rounds, run_id=args.resume)
    print(f"\nrun_id {run_id} — re-inspect with: python run_loop.py --report {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
