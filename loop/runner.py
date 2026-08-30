"""Orchestration: generate -> label -> train -> generate -> ...

Round 0 is the baseline: generate with the SFT checkpoint and score, without
training. Every round after that trains on the *cumulative* pool of every
message scored so far, then regenerates over the same prompts so the rounds
stay comparable.

Generation, scoring and training are injected, so the whole sequence can be
driven by stubs without a GPU. The defaults do the real thing.
"""

import os
from typing import Callable, Dict, List, Optional

from metrics import config
from loop import report
from loop.store import HOLDOUT_SPLIT, TRAIN_SPLIT, LoopStore, utc_now

ALGORITHMS = ("bco", "kto")
from training.reference_model import REF_MODES  # ("sft", "previous", "base")

# The detector whose score is the reward unless a run says otherwise. Every
# published result so far used this one, so it stays the default; the parameter
# exists because "did the policy learn about phishing or about ScamLLM" is
# answered differently depending on which detector is in the loop, and running
# the same setup against a second one is the cleanest way to ask.
DEFAULT_DETECTOR = "scamllm"


# =============================================================================
# Default implementations of the injected steps
# =============================================================================


def default_generate(checkpoint_path, prompts, gen_args, n_samples, on_prompt=None):
    from generate_dataset import generate_messages

    return generate_messages(
        prompts,
        path_sft=checkpoint_path,
        gen_args=gen_args,
        n_samples=n_samples,
        on_prompt=on_prompt,
    )


def supports_on_prompt(generate_fn) -> bool:
    """Whether a generate_fn can report progress as it goes.

    Checked rather than assumed, so a caller's own generator — a notebook, a
    test stub — keeps working with the four arguments it was written for.
    """
    import inspect

    try:
        return "on_prompt" in inspect.signature(generate_fn).parameters
    except (TypeError, ValueError):
        return False


def score_with_detector(name: str) -> Callable:
    """A `score_fn` that rewards against one registered detector.

    Whichever detector this is, its score lands on the message as `score` and
    its verdict as `label` — the desirable/undesirable class BCO and KTO train
    on. That is what "in the loop" means, and it is the one thing that changes
    when the reward detector does: everything downstream reads those two fields
    without caring which model wrote them.

    The label is computed here from the runner's threshold rather than by the
    detector's own `label_messages`, so one run has one threshold whatever
    default a detector carries.
    """

    def score(records, threshold) -> List[Dict]:
        from detectors import get_detector, get_spec

        spec = get_spec(name)
        if not spec.is_available():
            raise RuntimeError(
                f"detector {name!r} is registered but not available here, so it "
                "cannot supply the reward"
            )

        print(f"\nScoring {len(records)} messages with {name}...")
        scores = get_detector(name).score_messages([r["body"] for r in records])
        for record, value in zip(records, scores):
            record["score"] = float(value)
            record["label"] = bool(value >= threshold)
        return records

    return score


def default_score(records, threshold) -> List[Dict]:
    """Reward against ScamLLM, which is what every run did before this."""
    return score_with_detector(DEFAULT_DETECTOR)(records, threshold)


def default_train(
    algorithm, base_model, dataset_path, output_dir, ref_mode, sft_path, epochs, seed
):
    """Run one round of BCO or KTO.

    Returns (checkpoint directory, training stats). The stats carry the losses
    and — for KTO, which is the one TRL logs it for — the KL between the policy
    and the reference, the penalty term the algorithm applied.
    """
    if algorithm == "bco":
        from training.bco_trainer import train_bco as train
    elif algorithm == "kto":
        from training.kto_trainer import train_kto as train
    else:
        raise ValueError(f"unknown algorithm: {algorithm!r} (bco | kto)")

    stats = train(
        num_epochs=epochs,
        base_model=base_model,
        dataset_path=dataset_path,
        output_dir=output_dir,
        ref_mode=ref_mode,
        sft_path=sft_path,
        seed=seed,
    )

    # the trainers swallow load failures and return early, so confirm the
    # checkpoint actually landed rather than carrying on with a stale path
    if not os.path.isdir(output_dir):
        raise RuntimeError(
            f"{algorithm} training produced no checkpoint at {output_dir}; "
            "check the training log above"
        )
    return output_dir, stats


def default_policy_kl(records, checkpoint_path, reference_path):
    """Attach the policy-vs-reference KL to a round's messages."""
    from training.policy_kl import attach_policy_kl

    return attach_policy_kl(records, checkpoint_path, reference_path)


def unpack_train_result(result):
    """Accept either a checkpoint path or (path, stats) from a `train_fn`.

    The stats are an addition, and a caller with its own trainer — a notebook,
    a test stub — should not have to grow a second return value to keep working.
    """
    if isinstance(result, tuple):
        checkpoint, stats = result
        return checkpoint, stats or {}
    return result, {}


# =============================================================================
# Runner
# =============================================================================


class LoopRunner:
    def __init__(
        self,
        prompts: List[Dict],
        store: Optional[LoopStore] = None,
        holdout: Optional[List[Dict]] = None,
        algorithm: str = "kto",
        ref_mode: str = "sft",
        detector: str = DEFAULT_DETECTOR,
        n_samples: int = 4,
        gen_args: Optional[dict] = None,
        decoding: str = "default",
        threshold: float = config.SAFE_THRESHOLD,
        epochs: int = 3,
        seed: int = config.DEFAULT_TRAINING_SEED,
        sft_path: str = config.PATH_SFT,
        generate_fn: Callable = default_generate,
        score_fn: Optional[Callable] = None,
        train_fn: Callable = default_train,
        policy_kl_fn: Callable = default_policy_kl,
        sim_model=None,
        measure_drift: bool = True,
        measure_compliance: bool = True,
        measure_policy_kl: bool = True,
    ):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm: {algorithm!r} {ALGORITHMS}")
        if ref_mode not in REF_MODES:
            raise ValueError(f"unknown ref_mode: {ref_mode!r} {REF_MODES}")
        # An unusable reward detector is a run that generates for hours and then
        # cannot label anything, so it fails here rather than after round 0.
        # A caller-supplied score_fn is trusted and skips the check: that is the
        # injection point the stubs use, and it need not touch the registry.
        if score_fn is None:
            from detectors import available, get_spec

            try:
                spec = get_spec(detector)
            except KeyError as exc:
                # the CLI reports ValueError as a usage error; an unknown
                # detector is one, not a traceback
                raise ValueError(str(exc)) from exc
            if not spec.is_available():
                raise ValueError(
                    f"detector {detector!r} is registered but not available here "
                    f"(available: {', '.join(available())})"
                )

        self.prompts = prompts
        self.holdout = list(holdout or [])
        # One subject list per run, training prompts first: `prompt_id` indexes
        # into this, so a held-out prompt's id is its position after the offset
        # and the two splits share one unambiguous id space.
        self.all_prompts = list(prompts) + self.holdout
        self.store = store or LoopStore()
        self.algorithm = algorithm
        self.ref_mode = ref_mode
        self.detector = detector
        self.n_samples = n_samples
        self.threshold = threshold
        self.epochs = epochs
        self.seed = seed
        self.sft_path = sft_path
        self.generate_fn = generate_fn
        self.score_fn = score_fn or score_with_detector(detector)
        self.train_fn = train_fn
        self.policy_kl_fn = policy_kl_fn
        self.measure_drift = measure_drift
        self.measure_compliance = measure_compliance
        self.measure_policy_kl = measure_policy_kl
        self._sim_model = sim_model
        self._baselines: Dict[int, Dict] = {}

        # Resolved once, here, so every round of the run shares one complete
        # decoding spec and a bad combination fails before any GPU time.
        from generate_dataset import resolve_gen_args

        self.decoding = decoding
        self.gen_args = resolve_gen_args(gen_args, n_samples, decoding)

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
        import detectors
        from metrics import models

        if self._sim_model is not None and self._sim_model is models._CACHE.get("sbert"):
            self._sim_model = None

        models.unload_auxiliary_models()
        # The registry caches every detector it has built, the reward one
        # included. Dropping ScamLLM's module singleton is not enough once the
        # loop scores through the registry — the spec would still hold it, and
        # the VRAM with it.
        detectors.unload_all()

    def baselines(self, run_id: int):
        """Round-0 embeddings for this run, computed once and cached."""
        if run_id not in self._baselines:
            round_zero = self.store.get_messages(
                run_id, round_index=0, with_subject=False
            )
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
        """What the run was started with — its intent."""
        return {
            "algorithm": self.algorithm,
            "ref_mode": self.ref_mode,
            # which detector's verdict became the training labels. Runs made
            # before this was a parameter have no such key and were all ScamLLM,
            # so readers fall back to that rather than guessing.
            "detector": self.detector,
            "sft_path": self.sft_path,
            **self._generation_config(),
            **self._training_config(),
        }

    def _generation_config(self) -> Dict:
        """How this round's messages were produced.

        Recorded per round, not just per run: a resumed run builds a fresh
        LoopRunner from whatever flags were passed that time, so these can
        legitimately differ between rounds — and if they do, the round document
        is the only honest record of what actually happened.
        """
        return {
            "decoding": self.decoding,
            "holdout_prompts": len(self.holdout),
            "gen_args": self.gen_args,
            "n_samples": self.n_samples,
            "threshold": self.threshold,
        }

    def _training_config(self) -> Dict:
        """What a training round needs to be re-runnable.

        With the dataset pinned by content hash and the base checkpoint by
        weights hash, these are the remaining inputs. `ref_mode` is recorded at
        the top level of the round, so it is not repeated here.
        """
        return {"epochs": self.epochs, "seed": self.seed}

    def warn_on_config_drift(self, run_id: int) -> Dict:
        """Print a warning if this runner generates differently from the run.

        Not an error: changing `--n-samples` or the decoding length part way
        through a run is a legitimate thing to do. But it changes what the later
        rounds mean — `asr_at_n` is per prompt over n samples, so it is not
        comparable across a change in n — so it should never happen silently.
        """
        drift = self.store.config_drift(run_id, self._generation_config())
        if drift:
            print(
                "WARNING: this round generates differently from the run's config:"
            )
            for field, (was, now) in sorted(drift.items()):
                print(f"  {field}: run says {was!r}, this round uses {now!r}")
            print("  the round document records what was actually used.")
        return drift

    def generate_and_score(self, run_id: int, round_index: int, checkpoint: str) -> tuple:
        """Generate over every prompt with `checkpoint`, score, and store.

        The checkpoint is content-addressed before generating, so every message
        carries a DBRef to the adapter that wrote it. Returns (records,
        checkpoint document).
        """
        # round 0 generates from the pinned SFT adapter, which this run did not
        # produce and must not claim
        produced_by = (
            None if round_index == 0 else {"run_id": run_id, "round": round_index}
        )
        checkpoint_doc = self.store.upsert_checkpoint(checkpoint, produced_by=produced_by)

        # One timestamp for the whole round, taken before any of it is written,
        # so incremental inserts do not let an `as_of` cut a round in half.
        stamp = utc_now()
        streaming = supports_on_prompt(self.generate_fn)

        def generate(prompts, split, offset=0):
            """Generate for one split, persisting each prompt as it lands.

            The messages go in unscored: generation is the expensive part — hours
            for a full round — and a crash after it should not throw that away.
            The scoring and metric passes below fill the rest in.
            """

            def persist(batch):
                for record in batch:
                    record["prompt_id"] += offset
                    record["split"] = split
                self.store.add_messages(
                    run_id, round_index, batch, checkpoint=checkpoint_doc, added_at=stamp
                )

            if streaming:
                produced = self.generate_fn(
                    checkpoint, prompts, self.gen_args, self.n_samples, on_prompt=persist
                )
                # persist() has already offset and tagged what it was handed
                return produced

            produced = self.generate_fn(checkpoint, prompts, self.gen_args, self.n_samples)
            persist(produced)
            return produced

        records = generate(self.prompts, TRAIN_SPLIT)
        if self.holdout:
            # Same checkpoint, same decoding, same scoring — the only difference
            # is that these never reach the training pool.
            records = records + generate(self.holdout, HOLDOUT_SPLIT, len(self.prompts))

        records = self.score_fn(records, self.threshold)

        if self.measure_drift:
            # round 0 IS the baseline, so it only gets prompt coherence
            baselines = None if round_index == 0 else self.baselines(run_id)
            records = report.attach_drift(records, self.sim_model(), baselines)

        if self.measure_compliance:
            # did it still do what the prompt asked? The placeholder checks are
            # free; cos_subject needs SBERT, so it rides along only when drift
            # has already paid for the model.
            records = report.attach_compliance(
                records, self.all_prompts, self.sim_model() if self.measure_drift else None
            )

        if self.measure_policy_kl:
            # Scoring and drift leave ScamLLM, SBERT and the AI detector
            # resident, and the KL pass loads the 8B policy plus the reference
            # adapter — the same 11 GB squeeze training faces, so it needs the
            # same eviction. They reload lazily on the next round's scoring.
            self.free_auxiliary_models()

            # How far this round's policy has moved from the SFT baseline, on
            # the text it just produced. Always anchored to SFT whatever the
            # training-time ref_mode was, so rounds and ref_modes are on one
            # axis. Skipped for round 0, whose policy *is* the anchor.
            records = self.policy_kl_fn(records, checkpoint, self.sft_path)

        # the messages are already stored; this writes the scores and metrics on
        self.store.update_message_metrics(run_id, round_index, records)
        return records, checkpoint_doc

    def write_pool(self, run_id: int, round_index: int) -> tuple:
        """Pin the cumulative pool as a dataset and export it for the trainers.

        The trainers read a jsonl path, so each round's pool is written out
        rather than changing their interface. It is recorded as a dataset first
        — query plus an `as_of` cut-off taken now — so what the round trained on
        stays nameable and checkable after later rounds have appended to the
        same collection.

        Returns (path, dataset), where `dataset` carries `count`,
        `content_hash` and the materialised `rows`.
        """
        path = self.dataset_path(run_id, round_index)
        dataset = self.store.create_dataset(
            self.store.pool_query(run_id, round_index),
            name=f"run{run_id}_pool_round{round_index}",
            export_path=path,
            run_id=run_id,
            round=round_index,
        )
        return path, dataset

    def evaluate_round(self, run_id: int, round_index: int) -> Dict:
        """Metrics for the round, computed separately for each split.

        Kept apart rather than pooled: the training split says how well the
        policy does on subjects it has trained on, the held-out split whether
        that generalises, and averaging them together would hide the gap that
        is the whole point of having two.
        """
        messages = self.store.get_messages(
            run_id, round_index=round_index, with_subject=False
        )
        train = [m for m in messages if m.get("split", TRAIN_SPLIT) == TRAIN_SPLIT]
        held = [m for m in messages if m.get("split") == HOLDOUT_SPLIT]

        metrics = report.round_metrics(train, self.threshold)
        holdout_metrics = report.round_metrics(held, self.threshold) if held else None
        self.store.record_round(
            run_id, round_index, metrics=metrics, holdout_metrics=holdout_metrics
        )
        return metrics

    # -- the loop -----------------------------------------------------------

    def start(self) -> int:
        """Round 0: the SFT baseline, generated and scored but not trained."""
        run_id = self.store.create_run(self.all_prompts, self._config())
        print(f"\n{'=' * 60}\nRUN {run_id} — round 0 (baseline)\n{'=' * 60}")

        _, checkpoint_doc = self.generate_and_score(run_id, 0, self.sft_path)
        self.store.record_round(
            run_id,
            0,
            base_checkpoint=None,
            checkpoint_path=self.sft_path,
            checkpoint=self.store.checkpoint_ref(checkpoint_doc),
            checkpoint_hash=checkpoint_doc.get("weights_hash"),
            ref_mode=self.ref_mode,
            generation=self._generation_config(),
            # round 0 trains on nothing, so it pins neither a dataset nor a seed
            training=None,
            dataset=None,
            dataset_size=0,
            dataset_hash=None,
            pool_counts=None,
            label_counts=self.store.label_counts(run_id, round_index=0),
        )
        metrics = self.evaluate_round(run_id, 0)
        print(f"round 0: {_fmt(metrics)}")
        _fmt_holdout(self, run_id, 0)
        return run_id

    def step(self, run_id: int) -> int:
        """Train on everything so far, then regenerate over the same prompts."""
        self.store.check_prompts(run_id, self.all_prompts)
        self.warn_on_config_drift(run_id)

        previous = self.store.latest_round(run_id)
        if previous is None:
            raise RuntimeError(f"run {run_id} has no round 0; call start() first")

        round_index = previous["round"] + 1
        base = previous["checkpoint_path"]

        print(f"\n{'=' * 60}\nRUN {run_id} — round {round_index} ({self.algorithm})\n{'=' * 60}")

        dataset_path, pool = self.write_pool(run_id, round_index - 1)
        counts = self.store.label_counts(run_id, max_round=round_index - 1)
        print(
            f"training pool: {pool['count']} messages "
            f"({counts['desirable']} desirable / {counts['undesirable']} undesirable) "
            f"[{pool['content_hash']} as of {pool['as_of']:%Y-%m-%d %H:%M:%S}Z]"
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

        checkpoint, training_stats = unpack_train_result(
            self.train_fn(
                self.algorithm,
                base,
                dataset_path,
                self.checkpoint_dir(run_id, round_index),
                self.ref_mode,
                self.sft_path,
                self.epochs,
                self.seed,
            )
        )

        # The trainers clean up before returning, but their frame is only gone
        # once train_fn has actually returned, so a second pass here reclaims
        # whatever was still referenced from it. Generation loads another 8B
        # adapter next and needs the room.
        config.free_vram()

        _, checkpoint_doc = self.generate_and_score(run_id, round_index, checkpoint)
        self.store.record_round(
            run_id,
            round_index,
            base_checkpoint=base,
            checkpoint_path=checkpoint,
            checkpoint=self.store.checkpoint_ref(checkpoint_doc),
            checkpoint_hash=checkpoint_doc.get("weights_hash"),
            ref_mode=self.ref_mode,
            generation=self._generation_config(),
            training={**self._training_config(), **training_stats},
            dataset=self.store.dataset_ref(pool),
            # denormalised so a trajectory reads without a second query; the
            # dataset document is the authority, and it never changes
            dataset_size=pool["count"],
            dataset_hash=pool["content_hash"],
            pool_counts=counts,
            label_counts=self.store.label_counts(run_id, round_index=round_index),
        )
        metrics = self.evaluate_round(run_id, round_index)
        print(f"round {round_index}: {_fmt(metrics)}")
        _fmt_holdout(self, run_id, round_index)
        return round_index

    def run(self, rounds: int = 1, run_id: Optional[int] = None) -> int:
        """Run the baseline plus `rounds` training rounds, or resume a run."""
        if run_id is None:
            run_id = self.start()
        else:
            self.store.check_prompts(run_id, self.all_prompts)
            print(f"resuming run {run_id} from round {self.store.latest_round(run_id)['round']}")

        for _ in range(rounds):
            self.step(run_id)

        print(f"\n{'=' * 60}\nRUN {run_id} — trajectory\n{'=' * 60}")
        report.print_trajectory(self.store, run_id)
        return run_id


def _fmt_holdout(runner, run_id: int, round_index: int) -> None:
    record = runner.store.get_round(run_id, round_index) or {}
    holdout = record.get("holdout_metrics")
    if holdout:
        print(f"  held out: {_fmt(holdout)}")


def _fmt(metrics: Dict) -> str:
    if not metrics:
        return "(no messages)"
    return (
        f"mean score {metrics['mean_score']:.1f}%  "
        f"evasion {metrics['evasion_rate']:.1f}%  "
        f"ASR@n {metrics['asr_at_n']:.1f}%  "
        f"dupes {metrics['duplicates']}"
    )
