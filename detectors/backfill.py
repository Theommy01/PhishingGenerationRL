"""Score a finished run's messages with any detector.

    python -m detectors.backfill RUN_ID --detector bert-phishing

Scoring is a separate pass over stored bodies, not part of generation, so a
detector can be added long after a run finished — including one that did not
exist when it ran. Nothing is regenerated and no checkpoint is loaded.

It also reports the detector's **dynamic range** on the run, which decides
whether its verdicts are usable at all. A held-out detector fitted on real
corpora may be out of distribution on this text — the corpus is
entity-anonymised, so `<URL>` and `<ORG>` stand where a real link would be — and
one that calls every message phishing, or none, has no headroom and cannot
measure transfer. Better to see that in the output than to discover it inside a
conclusion.
"""

import argparse
import sys
from typing import List, Optional

from detectors import available, get_detector, get_spec
from loop.store import LoopStore
from metrics import config

BATCH = 32


def backfill(
    store: LoopStore,
    run_id: int,
    detector: str,
    threshold: float = config.SAFE_THRESHOLD,
    rescore: bool = False,
    progress: bool = True,
) -> dict:
    """Score the run's messages and store the verdicts. Returns a summary."""
    spec = get_spec(detector)
    if not spec.is_available():
        raise RuntimeError(f"detector {detector!r} is registered but not available here")

    if rescore:
        messages = store.get_messages(run_id, with_subject=False)
    else:
        messages = store.messages_missing_detector(run_id, detector)

    if not messages:
        return {"detector": detector, "scored": 0, "note": "nothing to score"}

    model = get_detector(detector)
    verdicts = []
    for start in range(0, len(messages), BATCH):
        chunk = messages[start : start + BATCH]
        scores = model.score_messages([m["body"] for m in chunk])
        batch = [
            {
                "round": message["round"],
                "prompt_id": message["prompt_id"],
                "sample_idx": message["sample_idx"],
                "score": float(score),
                "label": bool(score >= threshold),
            }
            for message, score in zip(chunk, scores)
        ]
        # written per batch, so an interrupted backfill keeps what it scored and
        # resumes where it stopped — `messages_missing_detector` skips these
        store.set_detector_verdicts(run_id, detector, batch)
        verdicts += batch
        if progress:
            print(
                f"  scored {min(start + BATCH, len(messages))} of {len(messages)}",
                flush=True,
            )

    return summarise(detector, verdicts, threshold)


def summarise(detector: str, verdicts: List[dict], threshold: float) -> dict:
    """Dynamic range, per round — the check that decides if this is usable."""
    scores = [v["score"] for v in verdicts]
    evaded = [v["label"] for v in verdicts]

    by_round = {}
    for verdict in verdicts:
        bucket = by_round.setdefault(verdict["round"], [])
        bucket.append(verdict["label"])

    return {
        "detector": detector,
        "scored": len(verdicts),
        "threshold": threshold,
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores),
        "evaded_fraction": sum(evaded) / len(evaded),
        "per_round_evaded": {
            round_index: sum(labels) / len(labels)
            for round_index, labels in sorted(by_round.items())
        },
        "degenerate": min(scores) == max(scores)
        or sum(evaded) in (0, len(evaded)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id", type=int)
    parser.add_argument(
        "--detector",
        default="bert-phishing",
        help=f"one of the registered detectors (available here: {', '.join(available())})",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="score every message again, not only the ones without a verdict",
    )
    args = parser.parse_args(argv)

    store = LoopStore()
    if store.get_run(args.run_id) is None:
        print(f"no such run: {args.run_id}", file=sys.stderr)
        return 1

    summary = backfill(store, args.run_id, args.detector, rescore=args.rescore)
    print()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if summary.get("degenerate"):
        print(
            "\nWARNING: this detector gave the same verdict to everything. It has "
            "no dynamic range on this text — probably out of distribution for it — "
            "so transfer measured against it would be meaningless.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
