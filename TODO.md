# Open issues

Things known to be wrong or unfinished, with enough detail to pick up cold.
`PORTING_NOTES.md` records what was found while porting the notebook; this
file is only for work still outstanding.

---

## Per-epoch checkpoints duplicate the reference adapter

**Cost:** ~160 MB of redundant weights per epoch checkpoint, per round.

The trainers save the final model with
`save_pretrained(save_dir, selected_adapters=[policy_adapter])`, so `save_dir`
itself is clean. But `KTOConfig/BCOConfig(save_strategy="epoch")` makes the HF
Trainer write its own intermediate checkpoints through a different code path,
which calls `save_pretrained` with no `selected_adapters`. peft then defaults
to `list(self.peft_config.keys())` — every attached adapter — writing the
policy at the checkpoint root and the reference into a `reference/`
subdirectory:

```
Models/run…_round1_kto/
    adapter_config.json              <- clean, policy only
    checkpoint-0/
        adapter_config.json          <- policy
        reference/                   <- redundant copy of the anchor
```

Only affects `--ref-mode sft` and `previous`, which are the modes that attach
a reference adapter. `base` attaches none, so nothing is duplicated.

**Options:**

1. `save_strategy="no"` in both trainers. The loop only ever consumes the
   final `save_dir`, so intermediate checkpoints are pure waste for it. This
   changes behaviour for standalone `python kto_trainer.py` use, where
   per-epoch checkpoints may be wanted — hence not done unilaterally.
2. Make `save_strategy` a parameter, defaulting to `"no"` from the loop and
   `"epoch"` standalone.
3. A `TrainerCallback` that prunes `reference/` after each save. Works, but
   more moving parts than the problem deserves.

Option 2 is probably right.

---

## `--ref-mode previous` is unverified

It uses the identical code path as `sft` — a different checkpoint resolved
into the same adapter slot — and `sft` is verified on a real KTO round, so
this is expected to work. It has simply not been run. Needs a two-round run to
exercise it properly, since `previous` and `sft` only diverge from round 2.

---

## BCO is unverified on a real round

Only KTO has been run end to end. `bco_trainer.py` received the same changes
(ref_mode, adapter swap, `selected_adapters`, VRAM ordering) but has not been
exercised.

---

## Generation speed

~4.6 s/message, so a full 150-prompt round at `--n-samples 4` is roughly 45
minutes before training even starts. Batching was measured and is *slower*
(0.9x), because a batch runs until its longest member finishes and the token
lengths vary a lot (27/86/79/127 in the sample measured). `for_inference` is
already enabled. vLLM is the only real multiplier identified and is not
installed.

---

## The legacy wide-format analysis still reads the uncorrected report

`metrics/config.py` points `FINAL_EVALUATION_REPORT` at
`Dataset/final_evaluation_report.csv`, whose BCO and KTO columns are inverted
(see `PORTING_NOTES.md` §1). Anything run through the wide path in
`metrics/analysis.py` will regenerate the original wrong figures. Either point
it at `final_evaluation_report_corrected.csv` or delete the legacy path, now
that the loop supersedes it.
