# Open issues

Work still outstanding, plus the caveats that belong in the write-up.
`PORTING_NOTES.md` records what was found while porting the notebook; this file
is for what is left to do and what not to over-claim.

The first real end-to-end run is `1787343134`: KTO, `ref_mode=sft`,
`decoding=default` (temperature 0.9), 135 train + 15 held-out prompts × 4
samples, rounds 0–4. Numbers below refer to it.

---

## Analysis still to do (no training needed)

All of this runs post-hoc over stored messages and checkpoints — the reason the
provenance work exists — so none of it needs the GPU loop.

### The held-out split is underpowered — the biggest gap

15 held-out prompts, deltas quantised to multiples of 0.25, so generalisation
(the second research question) cannot be answered cleanly: round 0→2 held-out
was 7 improved / 4 worsened, p ≈ 0.12. Every round's adapter is saved and
content-addressed, so the fix is to **generate a larger held-out prompt set from
each stored checkpoint** and score it — generation time, not training. This is
the highest-value remaining task.

### No trivial-alternative baseline

Round 0 already reaches ASR@4 = 63% just by sampling four times. A reader will
ask how much of round 4's gain is the fine-tuning versus more sampling.
Run **best-of-n on the SFT checkpoint** with a larger n as a baseline arm — from
the saved checkpoint, no training.

### Only one held-out detector

Transfer is measured against `bert-phishing` alone, and it agrees with ScamLLM
only weakly (baseline κ = 0.12, rank ceiling ρ = 0.33). Non-transfer to one
detector is suggestive; to two independent ones, especially the bag-of-words
SVM, would be much harder to argue with. The SVM is registered but unavailable
(no fitted `.pkl` in the checkout, and `DetectionDataSources/` absent). Get or
refit the pipeline, then `python -m detectors.backfill 1787343134 --detector svm`.

### Placeholder rehydration for a fair transfer test (needs design)

`bert-phishing` was fitted on real email; our corpus is entity-anonymised
(`<URL>`, `<ORG>`), which is out of distribution for it and may explain the
non-transfer as much as the policy would. Option 3 from the session: substitute
plausible URLs/orgs before held-out scoring. Needs decisions — what to inject,
and whether ScamLLM is re-scored on the same rehydrated text — so it is design
work, not a quick pass.

### Final round-4 backfill sweep

`bert-phishing` is scored on rounds 0–3 in full and most of round 4. Once the
loop finishes, run `python -m detectors.backfill 1787343134 --detector
bert-phishing` once more to catch round 4's tail (it skips already-scored
messages).

---

## What the run showed, for the write-up

Not tasks — but claims that need their caveats stated.

- **Evasion improved and it is significant.** Round 0→2 (train), paired per
  prompt: 31 gained / 8 lost, McNemar p = 2.9e-4; median evasion +0.25, Wilcoxon
  p = 7.6e-8. By round 4, evasion 31% → 52%, ASR@n 63% → 88%.
- **It is genuine, not signal-stripping.** Among prompts that kept the `<URL>`
  placeholder, evasion still rose 18% → 48% — the model learned to evade with
  the phishing signal present. The URL-required subset improved *most*.
- **Instruction-following on requested fields erodes, modestly.** URL
  emit-when-asked fell 79% → 70%, paired Wilcoxon p = 0.0065. Small next to the
  evasion gain, but real, and in the direction that helps evasion.
- **Round 1 was a null; improvement started at round 2.** Report it as
  likelihood displacement on a cold policy (both reward series fell together,
  KL term clamped to 0 all of round 1), resolving once training accumulated.
- **Single run, one seed.** Every claim is "this run did this", not "KTO does
  this". State it plainly; do not make method-level claims.
- **Evaluation prompts are the training prompts** (bar the 15 held out), so
  generalisation rests on that small split — see above.

---

## Degeneration is measured only as a log ratio and duplicates

`logratio_per_token` bounds distribution movement and `duplicates` catches exact
collapse, but neither sees within-message repetition or near-duplicates. On this
run the text looks fine (duplicates fell 19 → 6–8, log ratio recovered −0.23 →
−0.04), so it was not needed — but if a later run's log ratio looks fine while
the text reads wrong, add `repeat_2gram_ratio` / `distinct_2` and a
near-duplicate rate in `report.attach_compliance` (it already walks every
message). Cheap, no model.

---

## The training KL is KTO-only, and was clamped early

`training.kl_mean` comes from what TRL logs, and only KTOTrainer logs a `kl`.
Worse, TRL clamps that KL at 0 when its mismatched-pair estimate is negative,
which it was for all of round 1 and most of round 2 — so the training-time KL is
literally 0 there, and only becomes informative from round 3. Lead with the
eval-time `policy_kl` (covers both algorithms, on generated text) and treat the
training KL as a secondary, KTO-only diagnostic. Do not put `training.kl_mean`
next to a BCO round's blank cell as if comparable.

---

## Sentiment is not checked

The prompt asks for four things; three are verified (subject relatedness, `<URL>`
flag, `<ATTACHMENT>` flag). Nothing checks whether `sentiment: urgent,
threatening` actually reads that way. A classifier is the obvious route; the
sentiments are a long tail (20 prompts share `["urgent"]`, most others are
one-offs), so per-sentiment accuracy will be noisy at 150 prompts.

---

## `<ATTACH...>` variants are assumed, not confirmed

`ATTACHMENT_PLACEHOLDER` matches `<ATTACH>`/`<ATTACHMENT>` plus anything starting
`<ATTACH`. The corpus check found only `<ATTACHMENT>` (24 in 150 completions);
the variants are speculative tolerance. Harmless, but confirm before trusting a
sudden jump in attachment compliance.

---

## Per-epoch checkpoints duplicate the reference adapter

**Cost:** ~160 MB of redundant weights per epoch checkpoint, per round.

`KTOConfig/BCOConfig(save_strategy="epoch")` makes the HF Trainer write
intermediate checkpoints through a path that calls `save_pretrained` with no
`selected_adapters`, so peft writes every attached adapter — the policy at the
root and the reference into a `reference/` subdirectory. Only affects
`--ref-mode sft` and `previous`, which attach a reference; `base` attaches none.

Fix (option 2 from the original note): make `save_strategy` a parameter,
defaulting to `"no"` from the loop and `"epoch"` for standalone
`python -m training.kto_trainer`. Does **not** affect checkpoint identity —
`checkpoint_fingerprint` hashes only the five root identity files, so a stray
`reference/` changes no hash.

---

## `--ref-mode previous` / `base` unverified; BCO unrun

Only KTO with `ref_mode=sft` has run end to end. Per the session, the KL-anchor
comparison is a mechanism question, not the thesis's headline, so this is
future work rather than a blocker. If pursued, the strong version is a **fork**:
a `previous` arm that reuses this run's rounds 0–1 (identical for `sft` and
`previous` by construction) and diverges only from round 2 — a controlled,
paired comparison rather than two independent runs. The runner has no
"start from another run's round N" path yet; the data model supports it (round
0/1 messages and subjects are shareable). BCO is likewise unrun; it received the
same changes as KTO (ref_mode, adapter swap, seed, training stats) but has not
been exercised.

---

## Generation and measurement speed

Measured on the 2080 Ti (11 GB, 4-bit, no bf16): ~9 s/message under
`decoding=default` with `--max-new-tokens 256`, so ~90 min to generate a
600-message round, ~5½ h for a 5-round run including training and scoring.
Greedy is roughly half that. The `policy_kl` pass adds two forward passes per
message (no sampling loop) — a few minutes per round; `--no-policy-kl` skips it.
vLLM is the only real generation multiplier identified and is not installed.
Batching generation was measured *slower* (ragged early-stopping); batching the
`policy_kl` forward passes, by contrast, should help, since every sequence is
already complete.

---

## The legacy wide-format analysis reads the uncorrected report

`metrics/config.py` points `FINAL_EVALUATION_REPORT` at
`Dataset/final_evaluation_report.csv`, whose BCO/KTO columns are inverted
(`PORTING_NOTES.md` §1). Anything run through the wide path in `metrics/analysis`
regenerates the wrong figures. Build reports from the loop path
(`analysis.load_run`, `report.*`, `metrics.paired`, `metrics.transfer`) and the
legacy path is never touched; if it is kept, point it at
`final_evaluation_report_corrected.csv`. That path also still writes `*_KL_Div`
column names for what is now `embedding_distance` (a distance over SBERT
coordinates, not a KL) — the names were left because the CSVs on disk use them.

---

## Data files left where they are

`Dataset/archive/` holds everything no code reads. Three things were left alone
because guessing wrong is worse than the mess:

- **Three different `master_training_dataset.jsonl`** — under `Dataset/`,
  `Dataset/Experiment/SoloChat/` and `Models/`, all different checksums. Only
  `Dataset/`'s is read. Someone who knows the other two should name them.
- **`Models/master_training_dataset.jsonl`** — a dataset in the checkpoint
  directory; probably belongs in `Dataset/` but would collide with the name
  there.
- **`Dataset/Experiment/SoloChat/`** — one file, two directory levels, no
  explanation of the experiment.
