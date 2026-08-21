# PhishingGenerationRL

Adversarial fine-tuning of a phishing-email generator against a detector.
A LoRA-tuned Llama 3.1 8B writes emails from prompt specs, ScamLLM scores each
one, and BCO or KTO trains the generator on those scores. Repeat, and watch
whether evasion improves and how far the text drifts from where it started.

The research question is whether the choice of **KL anchor** changes the
outcome — see `--ref-mode` below.

---

## Layout

| Path | What it is |
|---|---|
| `run_loop.py` | **CLI for the whole loop.** Start here. |
| `generate_dataset.py` | Generate messages from prompt specs, score them, build a training set. Also a CLI. |
| `loop/` | The loop: `store.py` (MongoDB), `runner.py` (the cycle), `report.py` (per-round metrics). |
| `kto_trainer.py`, `bco_trainer.py` | One training round each. Alternatives — a run uses one or the other. |
| `reference_model.py` | What the KL term is anchored to, shared by both trainers. |
| `ScamAuxiliaryModel.py`, `ScamLabeller.py`, `ScamLabel.py` | ScamLLM, following the phishnet AuxiliaryModel/Labeller/Label pattern. |
| `metrics/` | `config.py` (paths, text helpers), `models.py` (AI detector, SBERT), `analysis.py`, `generation.py`. |
| `visualisation/charts.py` | Every figure and text report. Writes to `output/`. |
| `prompts.json` | 150 prompt specs: subject, sentiment, urls, attachments, category, generator. |
| `PORTING_NOTES.md` | What was found while porting the original notebook. **Read §1 before trusting any pre-existing result.** |

`reference/` holds the original notebook and is not imported by anything.

### Where ScamLLM lives

Exactly one place. `ScamLabeller.parse_model_output` is the only code in the
project that turns ScamLLM's labels into a number, and `get_scam_labeller()`
returns a single shared instance so the model loads into VRAM once:

```python
from ScamLabeller import get_scam_labeller
scores = get_scam_labeller().score_messages(bodies)   # safe probability, 0-1
```

**Higher means safer**, i.e. the filter was evaded. `LABEL_0` is safe and
`LABEL_1` is phishing, per the model card. This mattered: the codebase used to
carry three copies of that mapping, two of which disagreed, and the published
BCO/KTO results were computed with the inverted one. See `PORTING_NOTES.md` §1.

---

## Setup

Dependencies are managed with **uv**, not pip:

```bash
uv sync
```

You also need:

- **MongoDB** on `localhost:27017` (the loop stores runs in the `phishnet_rl` database)
- **An SFT checkpoint.** Adapters are gitignored — a few hundred MB each — so
  they are not in the repo. `checkpoint-2122` is the current one. Override with
  `PHISHNET_SFT_CHECKPOINT`, or pass `--sft-path`.
- **A GPU.** Developed on an RTX 2080 Ti (11 GB), which is enough but not roomy;
  see *VRAM* below.

Other environment overrides: `THESIS_BASE_DIR` (repo root by default),
`PHISHNET_BCO_CHECKPOINT`, `PHISHNET_KTO_CHECKPOINT`, `PHISHNET_SVM_MODEL`.

---

## The loop

One **round** is:

1. Generate `n` messages for every prompt with the current checkpoint
2. Score each with ScamLLM
3. Add them to the run's message pool
4. Train BCO or KTO on the **cumulative** pool
5. Regenerate over the **same** prompts with the new checkpoint

Round 0 is the baseline: generated and scored from the SFT checkpoint, no
training. The pool is cumulative because ScamLLM is frozen, so labelled
messages stay valid across rounds and each round only adds `N` new ones.
Evaluation is round-scoped, so metrics always describe the current checkpoint.

A run cannot silently change what it is measuring. `--resume` reads the prompts
back out of the database rather than off disk, so an edited `prompts.json`
cannot reach a half-finished run at all; the specs are re-checked against the
run's subject documents every round, and the *rendered* prompt structure — the
field names and markers the model was shown — is checked too, which catches
`generate_prompt` changing under a run in a way a hash of the specs cannot.

### Data model

Five collections in `phishnet_rl`:

| Collection | One document per | Key fields |
|---|---|---|
| `subjects` | prompt spec | `subject`, `category`, `generator`, `sentiment`, `urls`, `attachments`, `spec_hash` |
| `checkpoints` | adapter | `weights_hash`, `base_model`, `path`, `paths`, `files`, `produced_by` |
| `runs` | loop invocation | `run_id`, `config`, `subjects` (ordered DBRefs), `prompt_structures` |
| `rounds` | round of a run | `round`, `base_checkpoint`, `checkpoint` (DBRef), `dataset` (DBRef), `generation`, `training`, `pool_counts`, `metrics` |
| `messages` | generated message | `run_id`, `round`, `prompt_id`, `sample_idx`, `prompt_text`, `body`, `score`, `label`, `added_at`, `subject` (DBRef), `checkpoint` (DBRef) |
| `datasets` | point-in-time slice | `query`, `as_of`, `fields`, `count`, `content_hash`, `export_path` |

`messages` is append-only, and everything else is either an entity it points at
or a way of naming a subset of it.

**Subjects and checkpoints are content-addressed entities.** Both are shared —
the same 150 specs are regenerated every round, and one adapter generates a
whole round — so a message points at them with a DBRef instead of copying their
fields. Subject identity is a hash of the whole spec and checkpoint identity is
a hash of the adapter weights, so neither document is ever mutated: flipping
`urls` writes a new subject, and the same adapter found at a second path is
recognised as the same checkpoint rather than duplicated.

**Datasets are a query plus an `as_of` cut-off**, resolved against `added_at`.
Since messages are only appended, that pair names the same rows however many
rounds run afterwards — which round numbers alone cannot promise. Each is
stored with the `content_hash` of the rows it resolved to, so it can be checked
later rather than merely re-run.

### What can actually be verified

Generation is stochastic and unseeded, so no hash of the *inputs* makes a run
reproducible. What the model does give you is provenance you can check:

```bash
python run_loop.py --verify 1786641452
```

Per round, that answers two questions from what the messages themselves
recorded, not from the round document:

- **which checkpoint wrote these messages** — every message carries a DBRef and
  the adapter's `weights_hash`, stamped at insert time, so it holds even if the
  run dies before the round is written. `--verify` re-hashes the adapter now on
  disk: `differs` means the path was overwritten, `missing` means it is gone.
- **whether the pool a round trained on still holds** — the dataset is
  re-materialised from its query and cut-off and re-hashed, so an edited or
  deleted message shows up as a mismatch instead of a quietly different
  training set.

A failed check sets the exit code, so it can gate a report build.

Each round also records the config it *actually* used — `generation`
(`gen_args`, `n_samples`, `threshold`) and `training` (`epochs`, `seed`) —
rather than relying on the run's config, which is only what the run was
started with. Resuming builds a fresh runner from whatever flags were passed
that time, so those can legitimately differ; when they do, the loop warns and
the round document is the record. This matters for reading results: `asr_at_n`
is per prompt over n samples, so it is not comparable across a round where
`--n-samples` changed.

Note what a seed does and does not buy. There is no *decoding* seed, and none
is needed: every generated message is stored, so a dataset is reproduced by
retrieving it, not by replaying the sampler — which 4-bit kernels would not
replay faithfully anyway. The training seed is recorded because training is the
step worth re-running: with the dataset pinned by content hash and the base
checkpoint by weights hash, the seed completes the list of a round's inputs. It
still will not give bit-identical adapters without
`torch.use_deterministic_algorithms(True)` and a matching environment.

### Reading messages back

`store.get_messages()` joins the subject back in by default — `subject_text`
plus the spec fields — which is the shape `metrics.analysis` groups by. Pass
`with_subject=False` on paths that only need bodies and labels. Going the other
way, `store.messages_for_subject(id)` and `store.messages_for_checkpoint(id)`
collect every message a subject line or an adapter ever produced, across runs;
`store.checkpoint_for_message(m)` goes from one message to the adapter that
wrote it.

### Running it

```bash
# baseline plus three KTO rounds over the first 20 prompts
python run_loop.py --rounds 3 --limit 20

# BCO instead, with a moving KL anchor
python run_loop.py --algorithm bco --ref-mode previous --rounds 3

# carry an interrupted run on for two more rounds
python run_loop.py --resume 1786641452 --rounds 2

# inspect a finished run without generating anything
python run_loop.py --report 1786641452

# check each round's checkpoint and training pool still hash as recorded
python run_loop.py --verify 1786641452
```

| Flag | Default | Notes |
|---|---|---|
| `--prompts` | `prompts.json` | prompt spec file; ignored with `--resume` |
| `--limit` | all | first N prompts. Use it — a full round is slow |
| `--algorithm` | `kto` | `kto` or `bco`; alternatives, not stages |
| `--ref-mode` | `sft` | KL anchor; see below |
| `--rounds` | 1 | training rounds after the baseline |
| `--n-samples` | 4 | messages generated per prompt |
| `--epochs` | 1 | epochs per round |
| `--seed` | 3407 | training seed, recorded per round |
| `--sft-path` | `config.PATH_SFT` | round 0's checkpoint, and the `sft` anchor |
| `--max-new-tokens` | 256 | generation length cap |
| `--greedy` | off | greedy decoding; rejected with `--n-samples > 1`, which would return N identical messages |
| `--no-drift` | off | skip the SBERT drift metrics |
| `--resume` / `--report` / `--verify` | — | run id |

### `--ref-mode`: what the KL term is anchored to

BCO and KTO both regularise the policy against a reference policy, so the
trainer needs the reference's logprobs on every step. This is the comparison
the project is about:

| Mode | Anchor | Cost |
|---|---|---|
| `sft` *(default)* | the SFT checkpoint, pinned every round — divergence is always measured from the tuned baseline | +0.15 GiB |
| `previous` | the checkpoint this round trains from, so the anchor moves and divergence from SFT compounds | +0.15 GiB |
| `base` | the raw pretrained Llama, which is what the notebook did implicitly | free |

At round 1 `sft` and `previous` coincide, since that round trains from SFT.

The reference is loaded as a **second LoRA adapter on the same base model**,
and TRL swaps between them. Policy and reference share the identical frozen
4-bit base, so a second base model is never needed — handing TRL a checkpoint
*path* makes it build one anyway, which costs 5.5 GB and OOMs on an 11 GB card.

### Metrics per round

| Metric | Meaning |
|---|---|
| `mean_score` | mean ScamLLM safe probability |
| `evasion_rate` | share of messages scoring above `SAFE_THRESHOLD` (0.5) |
| `asr_at_n` | share of *prompts* with at least one evading sample — the number that matters when an attacker only needs one to land |
| `duplicates` | identical bodies, i.e. mode collapse |
| `cos_prompt` | semantic similarity to the prompt: did it stay on topic |
| `cos_baseline`, `kl_baseline` | drift from this run's round 0 |

Drift is computed per message and then averaged, **not** against a centroid;
the two differ materially (pairwise 85.4 cos / 0.037 KL versus centroid 90.6 /
0.023 on the same data).

---

## Building a training set on its own

Without the loop, for a one-shot dataset:

```bash
python generate_dataset.py            # writes Dataset/master_training_dataset.jsonl
```

Then `python kto_trainer.py` or `python bco_trainer.py`.

---

## Reading the output

Figures, tables and text reports land in `output/` (gitignored). The loop's
per-round charts are `plot_round_trajectory`, `plot_drift_trajectory` and
`plot_round_breakdown` in `visualisation/charts.py`; `metrics/analysis.py` has
`load_run`, `round_summary` and `round_breakdown` for the long-format data.

`metrics/analysis.py` also still carries the wide notebook-era analysis (one
row per prompt, one column per model). That path reads
`Dataset/final_evaluation_report.csv`, **whose BCO and KTO columns are
inverted**. Use `final_evaluation_report_corrected.csv` instead.

---

## VRAM

A round loads three 8B models in sequence — generate, train, generate again —
and 11 GB only fits one at a time, so each phase must fully release before the
next. Two things make that work, and both are easy to undo by accident:

- **Rebind to `None` before calling `free_vram()`.** Passing a model *to* it
  does nothing: `del` on a parameter drops only that function's reference.
- **Collect before emptying the cache.** The trainer, model and optimiser
  reference each other, so `del` only makes them collectable; `empty_cache()`
  before `gc.collect()` runs while everything is still alive and returns
  nothing to the driver. `config.free_vram()` does it in the right order.

`LoopRunner.free_auxiliary_models()` evicts ScamLLM, SBERT and the AI detector
before training; they reload lazily for the next round's scoring.

Symptom when this regresses: *"Some modules are dispatched on the CPU or the
disk"* from accelerate, or a plain CUDA OOM.

---

## Known state

- The loop runs end to end on real hardware, verified with a real KTO round
  under both `--ref-mode base` and `--ref-mode sft`.
- `--ref-mode previous` uses the identical code path as `sft` (a different
  checkpoint into the same adapter slot) but has not been run.
- BCO has not been run end to end; only KTO has.
- Per-epoch checkpoints duplicate the reference adapter (~160 MB each).

Open issues are tracked in [`TODO.md`](TODO.md).
- Generation is ~4.6 s/message, so 150 prompts × 4 samples ≈ 45 min per round
  before training. Batching measured *slower* (0.9×), because a batch runs to
  its longest member. vLLM is the only real speedup identified, and is not
  installed.
- `Models/bco_model_ep3` and `Models/kto_model_ep3` were trained on
  `master_training_dataset.jsonl`, whose labels are largely inverted. They are
  historical artefacts — not a warm start, not a reference model.
