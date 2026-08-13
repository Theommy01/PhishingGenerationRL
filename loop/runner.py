"""Orchestration: generate -> label -> train -> generate -> ...

Round 0 is the baseline: generate with the SFT checkpoint and score, without
training. Every round after that trains on the *cumulative* pool of every
message scored so far, then regenerates over the same prompts so the rounds
stay comparable.

Generation, scoring and training are injected, so the whole sequence can be
driven by stubs without a GPU. The defaults do the real thing.
"""

import json
import os
from typing import Callable, Dict, List, Optional

from metrics import config
from loop import report
from loop.store import LoopStore

ALGORITHMS = ("bco", "kto")
REF_MODES = ("base", "sft", "previous")


# =============================================================================
# Default implementations of the injected steps
# =============================================================================


def default_generate(checkpoint_path, prompts, gen_args, n_samples) -> List[Dict]:
    from generate_dataset import generate_messages

    return generate_messages(
        prompts, path_sft=checkpoint_path, gen_args=gen_args, n_samples=n_samples
    )


def default_score(records, threshold) -> List[Dict]:
    from generate_dataset import score_messages

    return score_messages(records, threshold)


def default_train(
    algorithm, base_model, dataset_path, output_dir, ref_mode, sft_path, epochs
) -> str:
    """Run one round of BCO or KTO and return the checkpoint directory."""
    if algorithm == "bco":
        from bco_trainer import train_bco as train
    elif algorithm == "kto":
        from kto_trainer import train_kto as train
    else:
        raise ValueError(f"unknown algorithm: {algorithm!r} (bco | kto)")

    train(
        num_epochs=epochs,
        base_model=base_model,
        dataset_path=dataset_path,
        output_dir=output_dir,
        ref_mode=ref_mode,
        sft_path=sft_path,
    )

    # the trainers swallow load failures and return early, so confirm the
    # checkpoint actually landed rather than carrying on with a stale path
    if not os.path.isdir(output_dir):
        raise RuntimeError(
            f"{algorithm} training produced no checkpoint at {output_dir}; "
            "check the training log above"
        )
    return output_dir


# =============================================================================
# Runner
# =============================================================================


class LoopRunner:
    def __init__(
        self,
        prompts: List[Dict],
        store: Optional[LoopStore] = None,
        algorithm: str = "kto",
        ref_mode: str = "sft",
        n_samples: int = 4,
        gen_args: Optional[dict] = None,
        threshold: float = config.SAFE_THRESHOLD,
        epochs: int = 3,
        sft_path: str = config.PATH_SFT,
        generate_fn: Callable = default_generate,
        score_fn: Callable = default_score,
        train_fn: Callable = default_train,
        sim_model=None,
        measure_drift: bool = True,
    ):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm: {algorithm!r} {ALGORITHMS}")
        if ref_mode not in REF_MODES:
            raise ValueError(f"unknown ref_mode: {ref_mode!r} {REF_MODES}")

        self.prompts = prompts
        self.store = store or LoopStore()
        self.algorithm = algorithm
        self.ref_mode = ref_mode
        self.n_samples = n_samples
        self.threshold = threshold
        self.epochs = epochs
        self.sft_path = sft_path
        self.generate_fn = generate_fn
        self.score_fn = score_fn
        self.train_fn = train_fn
        self.measure_drift = measure_drift
        self._sim_model = sim_model
        self._baselines: Dict[int, Dict] = {}

        # validated up front so a bad combination fails before any GPU time
        from generate_dataset import resolve_gen_args

        self.gen_args = resolve_gen_args(gen_args, n_samples)

    # -- drift ----------------------------------------------------------------

    def sim_model(self):
        """Loaded on first use so a drift-free run never pays for SBERT."""
        if self._sim_model is None:
            from metrics.models import get_similarity_model

            self._sim_model = get_similarity_model()
        return self._sim_model

    def free_auxiliary_models(self) -> None:
        """Evict the scorers from VRAM so training has the card to itself.

        A caller-supplied `sim_model` is left alone: it was not ours to unload,
        and dropping only our reference would not free it anyway.
        """
        from metrics import models

        if self._sim_model is not None and self._sim_model is models._CACHE.get("sbert"):
            self._sim_model = None

        models.unload_auxiliary_models()

    def baselines(self, run_id: int):
        """Round-0 embeddings for this run, computed once and cached."""
        if run_id not in self._baselines:
            round_zero = self.store.get_messages(run_id, round_index=0)
            self._baselines[run_id] = report.baseline_embeddings(
                round_zero, self.sim_model()
            )
        return self._baselines[run_id]

    # -- paths --------------------------------------------------------------

    def run_dir(self, run_id: int) -> str:
        path = os.path.join(config.OUTPUT_DIR, "runs", str(run_id))
        os.makedirs(path, exist_ok=True)
        return path

    def dataset_path(self, run_id: int, round_index: int) -> str:
        return os.path.join(self.run_dir(run_id), f"pool_round{round_index}.jsonl")

    def checkpoint_dir(self, run_id: int, round_index: int) -> str:
        return os.path.join(
            config.MODELS_DIR, f"run{run_id}_round{round_index}_{self.algorithm}"
        )

    # -- steps --------------------------------------------------------------

    def _config(self) -> Dict:
        return {
            "algorithm": self.algorithm,
            "ref_mode": self.ref_mode,
            "n_samples": self.n_samples,
            "gen_args": self.gen_args,
            "threshold": self.threshold,
            "epochs": self.epochs,
            "sft_path": self.sft_path,
        }

    def generate_and_score(self, run_id: int, round_index: int, checkpoint: str) -> List[Dict]:
        """Generate over every prompt with `checkpoint`, score, and store."""
        records = self.generate_fn(checkpoint, self.prompts, self.gen_args, self.n_samples)
        records = self.score_fn(records, self.threshold)

        if self.measure_drift:
            # round 0 IS the baseline, so it only gets prompt coherence
            baselines = None if round_index == 0 else self.baselines(run_id)
            records = report.attach_drift(records, self.sim_model(), baselines)

        self.store.add_messages(run_id, round_index, records)
        return records

    def write_pool(self, run_id: int, round_index: int) -> tuple:
        """Materialise the cumulative pool as jsonl for the trainers.

        The trainers read a jsonl path, so each round's pool is written out
        rather than changing their interface.
        """
        pool = self.store.training_pool(run_id, max_round=round_index)
        path = self.dataset_path(run_id, round_index)
        with open(path, "w") as f:
            for row in pool:
                f.write(json.dumps(row) + "\n")
        return path, pool

    def evaluate_round(self, run_id: int, round_index: int) -> Dict:
        messages = self.store.get_messages(run_id, round_index=round_index)
        metrics = report.round_metrics(messages, self.threshold)
        self.store.record_round(run_id, round_index, metrics=metrics)
        return metrics

    # -- the loop -----------------------------------------------------------

    def start(self) -> int:
        """Round 0: the SFT baseline, generated and scored but not trained."""
        run_id = self.store.create_run(self.prompts, self._config())
        print(f"\n{'=' * 60}\nRUN {run_id} — round 0 (baseline)\n{'=' * 60}")

        self.generate_and_score(run_id, 0, self.sft_path)
        self.store.record_round(
            run_id,
            0,
            base_checkpoint=None,
            checkpoint_path=self.sft_path,
            ref_mode=self.ref_mode,
            dataset_size=0,
            pool_counts=None,
            label_counts=self.store.label_counts(run_id, round_index=0),
        )
        metrics = self.evaluate_round(run_id, 0)
        print(f"round 0: {_fmt(metrics)}")
        return run_id

    def step(self, run_id: int) -> int:
        """Train on everything so far, then regenerate over the same prompts."""
        self.store.check_prompts(run_id, self.prompts)

        previous = self.store.latest_round(run_id)
        if previous is None:
            raise RuntimeError(f"run {run_id} has no round 0; call start() first")

        round_index = previous["round"] + 1
        base = previous["checkpoint_path"]

        print(f"\n{'=' * 60}\nRUN {run_id} — round {round_index} ({self.algorithm})\n{'=' * 60}")

        dataset_path, pool = self.write_pool(run_id, round_index - 1)
        counts = self.store.label_counts(run_id, max_round=round_index - 1)
        print(
            f"training pool: {len(pool)} messages "
            f"({counts['desirable']} desirable / {counts['undesirable']} undesirable)"
        )
        if not counts["desirable"] or not counts["undesirable"]:
            print(
                "WARNING: the pool has only one class. KTO/BCO contrast desirable "
                "against undesirable, so this round cannot learn a useful signal."
            )

        # Scoring and drift leave ScamLLM, SBERT and the AI detector resident,
        # and training needs the whole card for the 8B adapter (plus a second
        # copy of it when ref_mode != "base"). Without this, accelerate offloads
        # layers to CPU and the load fails outright on an 11 GB card. They
        # reload lazily when the next round scores.
        self.free_auxiliary_models()

        checkpoint = self.train_fn(
            self.algorithm,
            base,
            dataset_path,
            self.checkpoint_dir(run_id, round_index),
            self.ref_mode,
            self.sft_path,
            self.epochs,
        )

        # The trainers clean up before returning, but their frame is only gone
        # once train_fn has actually returned, so a second pass here reclaims
        # whatever was still referenced from it. Generation loads another 8B
        # adapter next and needs the room.
        config.free_vram()

        self.generate_and_score(run_id, round_index, checkpoint)
        self.store.record_round(
            run_id,
            round_index,
            base_checkpoint=base,
            checkpoint_path=checkpoint,
            ref_mode=self.ref_mode,
            dataset_size=len(pool),
            pool_counts=counts,
            label_counts=self.store.label_counts(run_id, round_index=round_index),
        )
        metrics = self.evaluate_round(run_id, round_index)
        print(f"round {round_index}: {_fmt(metrics)}")
        return round_index

    def run(self, rounds: int = 1, run_id: Optional[int] = None) -> int:
        """Run the baseline plus `rounds` training rounds, or resume a run."""
        if run_id is None:
            run_id = self.start()
        else:
            self.store.check_prompts(run_id, self.prompts)
            print(f"resuming run {run_id} from round {self.store.latest_round(run_id)['round']}")

        for _ in range(rounds):
            self.step(run_id)

        print(f"\n{'=' * 60}\nRUN {run_id} — trajectory\n{'=' * 60}")
        report.print_trajectory(self.store, run_id)
        return run_id


def _fmt(metrics: Dict) -> str:
    if not metrics:
        return "(no messages)"
    return (
        f"mean score {metrics['mean_score']:.1f}%  "
        f"evasion {metrics['evasion_rate']:.1f}%  "
        f"ASR@n {metrics['asr_at_n']:.1f}%  "
        f"dupes {metrics['duplicates']}"
    )
