# Porting notes: `Generate_Message copy.ipynb` → `metrics/` + `visualisation/`

Observations collected while moving the notebook's 61 cells into modules. Nothing
in this list has been "fixed" in the port unless the entry says so explicitly —
the goal was to preserve behaviour, and several of these need your judgement
because they change published numbers.

Original notebook preserved at `reference/Generate_Message.ipynb`.

---

## 0. Cell → file map

Use this to delete cells with confidence. Every cell is accounted for.

| Cell(s) | Content | Now lives in |
|---|---|---|
| 0, 5 | `pip install`, `drive.mount` | not ported (environment) |
| 1, 2 | `GenModel`, `LLama31GenModel` | already in `phishnet_lab/phishnet_sft/` |
| 3 | `Prompt`/`OutputMessage`, prompt builders | already in `phishnet_inference/prompt_generation/` |
| 4 | `MessageGenerator` | already in `phishnet_inference/` |
| 6 | one-off generation demo | not ported |
| 8 | ScamLLM pipeline + reward fn | `metrics/models.py` |
| 9 | safe-percentage print | `metrics.models.report_safe_percentage` |
| 11 | 90-prompt list | already `prompts.json` + `generate_dataset.py` |
| 12 | dataset viewer | `charts.show_dataset_records` (text) |
| 13, 15 | BCO / KTO training | already `bco_trainer.py`, `kto_trainer.py` |
| 14, 16, 18 | `%cd` + `!python` runners | `__main__` blocks |
| 17 | `evaluation.py` | `metrics/generation.py`, `metrics/models.py` |
| 19 | metric grid + summary tables | `analysis.averages_by_side` / `semantic_comparison_table`, `charts.plot_metric_grid` |
| 20 | distribution grid | `charts.plot_distribution_grid` |
| 21 | evaluation viewer | `charts.show_comparison` (text) |
| 22 | test-set run | `metrics.generation.run_test_set` |
| 23 | test-set viewer | `charts.show_test_set_records` (text) |
| 25, 28 | earlier / duplicate dashboards | not ported (superseded by 27) |
| 27 | live dashboard, FINAL | `metrics.generation.run_adversarial_pipeline` + `charts.show_pipeline_result` |
| 30 | evaded vs blocked | `analysis.evasion_stats`, `charts.plot_evaded_vs_blocked` |
| 31 | score distribution by generator | `charts.plot_scores_by_generator` |
| 32 | mean score by generator | `analysis.mean_scores_by`, `charts.plot_mean_scores_by_generator` |
| 33 | alignment-tax regression | `analysis.score_vs_similarity`, `charts.plot_alignment_tax` |
| 34 | malicious vs safe start | `analysis.split_by_sft_outcome`, `charts.plot_malicious_vs_safe` |
| 35 | same, distributions | `charts.plot_scores_by_generator_split` |
| 37 | 4-quadrant link analysis | `analysis.add_quadrants` / `score_table`, `charts.plot_quadrants` |
| 38 | RoBERTa AI detection | `metrics.models.run_ai_detection` |
| 39 | AI-detection viewer + bar | `charts.show_ai_detection_records`, `charts.plot_mean_ai_probability` |
| 40 | prompt-length correlation | `models.prompt_length_correlation`, `charts.plot_prompt_length_correlation` |
| 42 | AI-detection boxplot | `charts.plot_ai_probability_distribution` |
| 41, 60 | thesis prose | not ported |
| 44 | stray `SVM` code cell | not ported (would raise `NameError`) |
| 45-52, 54-57 | SVM train / eval / CV / features / save | `metrics/analysis.py`, `visualisation/charts.py` |
| 48 | SVM EDA plots | `charts.plot_class_and_source_distribution` / `plot_text_length_distribution` |
| 53 | SVM evaluation | `analysis.evaluate_model`, `charts.plot_confusion_matrix` |
| 58, 59 | SVM vs ScamLLM | `metrics/analysis.py` (`run_full_comparison`) |


---

## 1. Critical: the two ScamLLM scorers disagree about which label means "safe"

There are **three** definitions of `scam_evasion_reward` in the codebase, and two
of them contradict each other:

| Source | Reads | Treats as *safe* |
|---|---|---|
| Cell 8 (the one live in the notebook namespace) | full `top_k=None` list, `scores.get('LABEL_0')` | **LABEL_0** |
| Cell 17 (written out to `evaluation.py`) | `result[0]`, `score` if label is `LABEL_1`/`safe` else `1 - score` | **LABEL_1** |
| `ScamAuxiliaryModel.py` | copy of the cell-8 body | LABEL_0 (but see §4) |

This matters because the two scorers feed the *same CSV*:

- `SFT_ScamLLM_Score` / `SFT_Is_Safe` come from the dataset built with the
  **cell-8** scorer (`generate_dataset.py`).
- `BCO_ScamLLM_Score` / `KTO_ScamLLM_Score` are computed by `run_evaluation()`
  with the **cell-17** scorer.

If one of them has the polarity backwards, every SFT-vs-BCO/KTO comparison in
the thesis is comparing an evasion score against its own complement. Worth
resolving before anything else: run both over the same 20 texts and check they
agree.

Secondary issue in the cell-17 version: with `top_k=None` the pipeline returns a
*list* per input sorted by score, so `result[0]` is "whichever label won", not a
fixed label. The `1 - score` branch then converts the loser's confidence, which
is only equivalent to the safe probability for a strictly binary head.

Also: its `threshold=0.50` parameter is accepted and never used.

**In the port:** both are kept, under distinct names, in `metrics/models.py` —
`scam_evasion_reward` (cell 8) and `scam_evasion_score` (cell 17). Call sites
were wired to whichever the notebook actually used at that point.

## 2. Same name, different behaviour

| Name | Copies | Divergence |
|---|---|---|
| `scam_evasion_reward` | 3 | see §1 |
| `extract_body` | 3 | cells 22/25/27/28 split on `"->\n"`; cells 17/38 split on `"->\nbody:"`. Different outputs on the same text. |
| `free_vram` | 4 | byte-identical (cells 22, 25, 27, 28) |
| `map_generator` | 5 | function in cells 30/32/34, inline nested lambda in 31/35. Same result. |
| `extract_subject` | 2 | identical (cells 12, 21) |
| bar annotators | 4 | `annota_barre` (19), `add_labels` (30, %+n/total), `add_labels` (34, %-only colour-coded), `add_score_labels` (32, 37). Three distinct behaviours under two names. |
| dropdown viewer | 4 | cells 12, 21, 23, 39 — same widget, four copies |
| `get_status_html` / `run_pipeline` | 3 | cells 27 and 28 are byte-identical; cell 25 lacks the `"skipped"` state |
| `pipeline` | — | sklearn `Pipeline` object (cells 45-59) vs `transformers.pipeline` function (cells 8, 17, 38), both live in one namespace |

**In the port:** the two `extract_body` variants are kept as
`extract_body_after_arrow` / `extract_body_after_body_tag`
(`metrics/config.py`); the three annotators as `annotate_bars`,
`annotate_scores`, `annotate_counts` (`visualisation/style.py`); the four
viewers became the text reports in `visualisation/charts.py` (§12); only the final
dashboard is kept (cell 27, per your call).

## 3. Globals reused for different data

- **`df`** is the biggest one. Across the notebook it is: the training dataset
  (cell 12), the evaluation report (19, 20, 21, 30-37), the AI-detection report
  (39, 40, 42), one source CSV inside the SVM load loop (46), and the combined
  SVM frame. **Cells 40 and 55 have no `read_csv` of their own** — they operate
  on whatever `df` was last bound, so running cells out of order silently
  changes what those figures show.
- **`prompts`**: the 90-entry prompt list (cell 11), `df['prompt'].tolist()`
  (cell 17), the reward-function parameter, and `test_prompts` (cell 22).
- **`model`**: a `LLama31GenModel` instance (cells 6, 22, 25-28) vs the
  `(model, tokenizer)` tuple from `FastLanguageModel.from_pretrained`
  (cell 17 and the trainers).
- **`counts`**: per-generator sizes (30, 32, 34), per-quadrant sizes (37), and a
  spam/non-spam dict (46).
- **`threshold`**: 0.50, re-declared in cells 30, 34, 35, 37 and in the dataset
  builder. Now `metrics.paths.SAFE_THRESHOLD`.

## 4. Bugs found in the existing root-level files

These are in the files you had already started, not the notebook:

- **`ScamAuxiliaryModel.py` — FIXED.** Was: `Any` used in the `classifier`
  return annotation but never imported (`NameError` at class definition);
  `scam_evasion_reward` calling a global `scam_detector` that does not exist in
  that module; `print("Dowload fine!")` running at import.
  Now imports `Any`/`Dict`/`List`, carries a `model_args` dict and builds the
  pipeline from it (matching `SentimentAuxiliaryModel`), and `predict` takes
  `message_bodies` to match the signature `Labeller.label_messages` calls. The
  broken `scam_evasion_reward` was **deleted** rather than repaired — nothing
  imported it, and two working versions already live in `metrics/models.py`.
- **`ScamLabel.py` — unchanged behaviour**, docstring added recording that
  `value` is the LABEL_1 probability and pointing at the §1 polarity caveat.
- **`ScamLabeller.py` — FIXED.** Was: `auxiliary_model = ScamAuxiliaryModel = ScamAuxiliaryModel()`
  rebinding the class name to an instance; `parse_model_output` defined at module
  level (with a `self` parameter) *outside* the class, so never a method.
  A third bug surfaced while fixing it: `parse_model_output` did
  `output.get("LABEL_1", 0.0)` on each entry, but with `top_k=None` the pipeline
  returns a **list of per-label dicts** per message, not a `{label: score}`
  mapping — so it would have raised `AttributeError` on real output. It now
  builds the mapping first (`{item["label"]: item["score"] for item in label}`),
  as the reward function always did. `label_messages` added to match
  `SentimentLabeller`.
- **`generate_dataset.py` — FIXED.** Was: `message.append(...)` instead of
  `messages.append`; `texts_sft`, `prompt_structures`, `pd`, `os` and
  `scam_evasion_reward` all undefined; `generate_messages()` defined but never
  called, so the dataset build ran at import against names that did not exist.
  Two further bugs surfaced while fixing it: `LLama31GenModel.load(...)` is not
  a real method (there is no `load` classmethod — construction is
  `LLama31GenModel(checkpoint_path=...)`), and `MessageGenerator(gen_model)` was
  positional, which a pydantic model rejects.
  It is now a module of functions (`load_prompts`, `generate_messages`,
  `build_dataset`, `save_dataset`, `main`) with nothing running at import,
  reading `prompts.json` next to the file rather than from the CWD and taking
  its paths from `metrics.config`. This closes the cell 11 gap in §9a.
- **`generate_dataset.py`** — `message.append(...)` should be `messages.append`;
  `texts_sft`, `prompt_structures`, `pd` and `os` are undefined; the
  `generate_messages()` function is defined but never called, so the script
  cannot run as-is.
- **`kto_trainer.py`** — the `%%writefile /content/.../train_kto.py` magic that
  made this file invalid Python **had already been removed** by the time the
  trainers were fixed; an earlier revision of this note said otherwise and was
  stale.
- **`bco_trainer.py` / `kto_trainer.py` — FIXED.** Both passed
  `tokenizer=tokenizer` to `BCOTrainer`/`KTOTrainer`. TRL removed that argument
  in favour of `processing_class` around 0.16, and **the venv here has trl
  0.18.1**, so both scripts would have died with
  `TypeError: __init__() got an unexpected keyword argument 'tokenizer'` before
  a single training step. Now `processing_class=tokenizer`, which 0.15.2 and
  0.18.1 both accept.
  Also: Colab paths replaced with `metrics.paths` (so `THESIS_BASE_DIR` works),
  unused `json`/`os` imports dropped, `import inspect` moved to the top, and the
  `Red`→`Read` print typo fixed.
  Left alone deliberately: the unsloth `_get_train_sampler` patch is a no-op on
  trl 0.18.1 (the method already takes `dataset`), but the guard inspects the
  signature at runtime, so it stays correct on older versions.

## 5. Bugs in the notebook's analysis cells

- **Cell 59, `df_comp`** — the BCO and KTO columns both zip against
  `df['SVM_SFT']`. The BCO/KTO SVM predictions are computed and then discarded,
  so those two columns show the SFT verdict three times. Reproduced verbatim in
  `metrics.analysis.comparison_table`, with a comment.
- **Cell 59, concordance** — `scamllm_is_spam = df[score_col] > 0.5`, but that
  score is a *safe* probability everywhere else. As written the metric compares
  "ScamLLM says safe" against "SVM says spam", so the headline 50-56%
  concordance is likely the discordance. Combined with §1 this number should be
  recomputed before it goes in the thesis. Reproduced verbatim, with a comment.
- **`plt.savefig` after `plt.show()`** (cells 30, 32, 34, 37) — `show()` releases
  the figure, so the four saved PNGs are blank outside the inline backend.
  **This one is fixed in the port**: `visualisation/charts.py` saves before
  showing. It is the only behaviour change I made without asking, because the
  feature was simply not working.
- **Cell 33** — `plt.figure(figsize=(12,8))` immediately before `sns.lmplot`,
  which creates its own `FacetGrid`. The bare figure is emitted empty. Dropped
  in the port.
- **Cell 38** — assumes the detector's positive label is the literal string
  `'Fake'`. Depending on the `roberta-base-openai-detector` revision the pipeline
  returns `Real`/`Fake` *or* `LABEL_0`/`LABEL_1`; on the latter every score
  silently becomes `100 - score`. Kept as-is, but worth an assert.
- **Cell 44** is a *code* cell containing the single word `SVM` — it raises
  `NameError` on run. Clearly meant to be markdown.
- **Cell 22** hardcodes `10` (`[""]*10`, `range(10)`) instead of
  `len(test_prompts)`, so adding a prompt silently truncates the scoring. The
  port uses `len()`; identical for the current 10 prompts.
- **Cells 31, 33, 39, 40, 42** use `pd`, `sns`, `plt` or `df_filtered` without
  importing/defining them in that cell. They only work top-to-bottom.
- **Cell 45** runs `pip uninstall`/`pip install` for scikit-learn mid-notebook
  without restarting the runtime, so the already-imported version stays in
  memory for the rest of the session.

## 6. Unit and convention inconsistencies

- Cosine similarity is stored **already multiplied by 100**; the ScamLLM score is
  stored **0-1** and multiplied at display time; KL divergence is raw. The same
  CSV therefore mixes three scales, and
  `plot_dynamic_distribution` has to sniff the column name
  (`'ScamLLM_Score' in cols[0]`) to decide whether to rescale.
- `scam_average()` multiplies by 100, `cos_average()` does not — correct given
  the above, but only if you remember the storage convention.
- Three different SFT/BCO/KTO palettes: `["#bdc3c7","#3498db","#2ecc71"]` (metric
  grids), `["#95a5a6","#3498db","#2ecc71"]` (AI detection),
  `["#95a5a6","#3498db","#e67e22"]` (generators/quadrants). All three preserved
  in `visualisation/style.py` so the figures still match the thesis.
- `SFT_Is_Safe` is the *dataset* label produced at generation time, while
  `BCO_ScamLLM_Score`/`KTO_ScamLLM_Score` are produced at evaluation time by a
  different scorer — see §1.
- Comments, printed labels and chart titles mix Italian and English, sometimes
  in the same f-string. Left as-is; the printed output is what the thesis
  screenshots show.

## 7. Artifact / path issues

- Cell 52 saves `svm_email_classifier.pkl` to the **current working directory**;
  cell 57 saves `svm_phishing_detector.pkl` to **`/kaggle/working/models`**, a
  path that does not exist on Colab. Cells 58/59 load the first one. Two
  artifacts for one model; both paths are in `metrics/config.py`.
- Every Colab path is now in `metrics/config.py`, overridable with the
  `THESIS_BASE_DIR` environment variable.
- `%cd` + `!python script.py` cells (14, 16, 18) are gone: the modules have
  `if __name__ == "__main__"` blocks instead.

## 8. Redundancy with the rest of the phishnet workspace

- Cells 1-4 (`GenModel`, `LLama31GenModel`, the `Prompt`/`OutputMessage`
  pydantic models, `MessageGenerator`) are re-pasted copies of code that already
  lives in `phishnet_lab/phishnet_sft/`, `phishnet_inference/` and
  `phishnet_context_manager/`. Not ported — `metrics/generation.py` and
  `visualisation/live_dashboard.py` import from those packages, matching what
  `generate_dataset.py` already does.
- Cells 13 and 15 (BCO/KTO training) are already at the repo root as
  `bco_trainer.py` / `kto_trainer.py`. Not re-ported.
- `metrics_by prompt.py` at the repo root was a verbatim copy of cell 19, and
  `metrics_1.py` was empty. Both are now superseded by
  `metrics/analysis.py` + `visualisation.charts.plot_metric_grid`.

## 9. Verification against `reference/Generate_Message.ipynb`

`Generate_Message copy.ipynb` and `reference/Generate_Message.ipynb` are
byte-identical (same md5), so the reference is the authority for this audit.

All **78** `def`/`class` definitions in the notebook's code cells were
enumerated and matched to a destination — none unaccounted for. Cells 1-4 were
diffed line-by-line against the packages they were pasted from:

| Cell | Package file | Difference |
|---|---|---|
| 1 `GenModel` | `phishnet_sft/GenModel.py` | package adds a `FineTuningArguments` import; otherwise identical |
| 2 `LLama31GenModel` | `phishnet_sft/LLama31GenModel.py` | package adds a `GenModel` import; one blank line, one newline-at-EOF |
| 3 prompt models | `phishnet_inference/prompt_generation/generate_prompt.py` | **byte-identical** |
| 4 `MessageGenerator` | `phishnet_inference/MessageGenerator.py` | package adds imports and `-> str` annotations; otherwise identical |

### Two gaps found by this audit

**(a) Cell 11's dataset-generation loop is only half-ported.** The cell has no
`def`, so it does not appear in the definition audit, but it contains real
logic that exists nowhere in `metrics/`:

- the prompt string, built inline (see §10 for how it differs from
  `MessageGenerator`);
- the raw-tokenizer generation loop (`FastLanguageModel.from_pretrained` →
  `for_inference` → `generate(max_new_tokens=256, temperature=0.7)`), which is
  the same shape as `generate_for_model` but on the SFT checkpoint;
- the `dataset_records` build and
  `df.to_json(save_path, orient="records", lines=True)`.

**This gap is now closed**: `generate_dataset.py` implements the pipeline (see
§4). It routes through `MessageGenerator` rather than the notebook's inline
format string, per the decision recorded in §10, which has two consequences
worth knowing:

- Decoding defaults to `GREEDY_GEN_ARGS`, reproducing what the notebook actually
  did and therefore the existing dataset. Pass `gen_args` for anything else.
- The prompt stored in the `prompt` column is now `generate_prompt(...) +
  "\n->\n"` — it ends at the arrow, where the notebook's ended at
  `->\nbody:`. Downstream parsers are unaffected (`clean_prompt` splits on
  `"->"`, `extract_subject` / `extract_url_flag` read individual lines).
- Because `MessageGenerator` stops before `body: `, the model has to emit that
  prefix itself. `_extract_completion` splits on `->\nbody:` when present and
  falls back to the bare `->\n` otherwise, so a missing prefix strips the
  prompt anyway instead of silently storing it as part of the completion. The
  run prints a warning counting how often the fallback fired.

**(b) `prompts.json` was not valid JSON. — FIXED.** The content was complete and
correct all along (150 prompts, same order as the notebook, verified field by
field), but the file had kept two Python-isms from the paste: 18 `#` comment
lines and a trailing comma before the closing `]`, either of which makes
`json.load()` fail. Both removed; the file now parses and still matches the
notebook list exactly.

---

## 10. `MessageGenerator` vs the notebook's inline prompt

The decision is to **use `MessageGenerator`** going forward. The two paths are
not equivalent, so here is exactly what changes.

### What is identical

Key order is the **same** in both — `subject`, `urls`, `attachments`,
`sentiment`. (An earlier draft of these notes claimed otherwise; that was
wrong.) Booleans render as Python `True`/`False` in both, and sentiment is
`", ".join(sentiment)` in both.

### Difference 1 — the `body: ` primer

```
notebook (cell 11)                    MessageGenerator.generate_message
──────────────────────────────        ─────────────────────────────────
subject: {subject}                    subject: {subject}
urls: {urls}                          urls: {urls}
attachments: {attachments}            attachments: {attachments}
sentiment: {a, b}                     sentiment: {a, b}
->                                    ->
body:                                 ⏎ (prompt ends here)
```

The SFT training samples were built by `generate_prompt_output_pair` as
`{prompt}\n->\nbody: {body}`. So the notebook's prompt reproduces the training
prefix **through the `body: ` token**, priming the model to continue the body
directly. `MessageGenerator` stops one token short, and the model has to emit
`body: ` itself.

Consequences:

- **Extraction.** `extract_body_after_arrow` splits on `"->\n"` and takes `[1]`,
  so with `MessageGenerator` the returned text **retains a leading `body: `**.
  The notebook's own cell 22 and the dashboard already had this behaviour, which
  is why the test-set texts carry a `body: ` prefix that the evaluation-report
  texts (split on `"->\nbody:"`) do not. ScamLLM and SBERT therefore see a stray
  token on one path and not the other.
  If you want them consistent, run `extract_body_after_body_tag` over the result.
- **Reliability.** Nothing forces the model to emit `body: `. When it does not,
  the split falls through to `text.strip()` and the completion silently keeps
  the whole echoed prompt.
- **Stored prompts.** The `prompt` column of `master_training_dataset.jsonl`
  currently ends with `->\nbody:`; regenerated via `MessageGenerator` it will end
  at `->`. Downstream parsers are unaffected — `clean_prompt` splits on `"->"`,
  and `extract_subject` / `extract_url_flag` read individual lines.

### Difference 2 — decoding (the bigger one) — NOW SELECTABLE

Decoding is no longer hardcoded. `MessageGenerator` takes a `gen_args` dict, on
the instance or per call, merged over the defaults:

```python
from phishnet_inference.MessageGenerator import (
    MessageGenerator, DEFAULT_GEN_ARGS, GREEDY_GEN_ARGS, SAMPLING_GEN_ARGS,
)

MessageGenerator(gen_model=m)                            # sampling @ 0.9 (unchanged default)
MessageGenerator(gen_model=m, gen_args=GREEDY_GEN_ARGS)  # notebook parity
gen.generate_message(subject=..., gen_args={"temperature": 0.7})   # partial, per call
```

`metrics.test_set.run_test_set(...)` and `visualisation.live_dashboard.build_dashboard(...)`
both take the same `gen_args` and apply it to all three checkpoints.

Two enabling changes in `phishnet_sft/LLama31GenModel.generate` (both backward
compatible — callers passing only `max_length` are unaffected):

- `max_new_tokens` is supported and takes precedence over `max_length`; passing
  both forwards only `max_new_tokens`, instead of making transformers warn.
- the sampling knobs are dropped rather than forwarded when `do_sample=False`,
  so `{"do_sample": False}` alone is enough to get true greedy decoding without
  a leftover `temperature` triggering warnings.
- it also no longer tokenizes the prompt twice.

The table below is what the two presets encode.



| | notebook (cells 11, 17) | `MessageGenerator.generate_message` |
|---|---|---|
| sampling | `do_sample` **not passed** → defaults to `False` → **greedy** | `do_sample=True` → **stochastic** |
| temperature | `0.7`, but **ignored** (no sampling; HF warns) | `0.9`, applied |
| top_k / top_p | n/a | `50` / `0.95` |
| length | `max_new_tokens=256` (completion only) | `max_length=384` (**prompt + completion**) |
| reproducibility | deterministic | varies per run unless seeded |

Two things to watch:

- `max_length=384` counts the prompt. These prompts run ~60-90 tokens, so the
  effective completion budget is ~290-320 tokens — usually more than 256, but a
  long subject line eats into it, and a very long one can leave almost nothing.
  `max_new_tokens` has no such coupling.
- The dataset was built greedily; regenerating it through `MessageGenerator`
  samples at temperature 0.9. Expect different completions, different ScamLLM
  scores, and a different safe/malicious split than the current
  `master_training_dataset.jsonl`. That is a re-baseline, not a refactor.

### Difference 3 — minor

- `gen_args` passes `num_return_sequences: 1`, but `LLama31GenModel.generate`
  never reads it — only `max_length`, `do_sample`, `temperature`, `top_p`,
  `top_k`. Inert.
- `LLama31GenModel.generate` tokenizes the prompt **twice** (once for
  `input_ids`, once for `attention_mask`).
- `generate_prompt` **omits the sentiment line entirely** when `sentiment` is
  empty/`None`; the notebook's f-string always emits `sentiment: ` (possibly
  blank). Only matters for prompts with no sentiment — none in `prompts.json`.

### If you re-port cell 11 on top of `MessageGenerator`

The loop becomes: `generate_message(subject=…, urls=…, attachments=…,
sentiment=…)` per entry in `prompts.json`, then `extract_body_after_body_tag`
on the result (not `extract_body_after_arrow`, so the `body: ` prefix is
stripped), then the existing `dataset_records` build. Store the prompt actually
sent, via `generate_prompt(...) + "\n->\n"`, so the CSV's `prompt` column stays
truthful.

## 11. Not ported (deliberately)

- Cells 0, 5 (`pip install unsloth`, `drive.mount`) — environment setup.
- Cells 6, 9 — one-off demos. Cell 9's logic survives as
  `metrics.models.report_safe_percentage`.
- Cells 14, 16, 18 — `%cd` + `!python` runners.
- Cells 41, 60 — thesis prose, no code.
- Cell 25 — earlier dashboard, strict subset of cell 27.
- Cell 28 — byte-identical to cell 27.
- The commented-out `predict_proba` variant of `evaluate_model` (cell 53) — the
  SVC is built with `probability=False`, so it could never have run.

---

## 12. Consolidation and the move off notebooks

23 modules collapsed to 6. The merges were driven by shared shape, not just
file count:

| Was | Now | Why |
|---|---|---|
| `paths.py` + `text_utils.py` | `config.py` | both are dependency-free constants/helpers |
| `scam_llm.py` + `ai_detection.py` + `semantic.py` | `models.py` | three lazily-loaded auxiliary models behind one `_cached` helper |
| `evaluation_report.py` + `test_set.py` + `live_dashboard.py` | `generation.py` | everything that loads a checkpoint and produces text |
| `aggregates.py` + `generators.py` + `link_analysis.py` + `svm_detector.py` + `detector_agreement.py` | `analysis.py` | all pandas over the same report |
| `bar_charts.py` + `distributions.py` + `ai_detection_plots.py` + `svm_plots.py` + 3 widget modules | `charts.py` | one plotting module |

Duplication actually removed (not just relocated):

- **Four near-identical grouped-bar charts** (cells 30, 32, 34, 37) each
  hardcoded `x - width` / `x` / `x + width`, the palette, the 50% line and the
  annotation loop. They now share `grouped_bar()` and `threshold_line()`.
- **Three box+strip blocks** share `box_strip()`.
- **`mean_scores_by_generator` and the quadrant means** were the same group-by;
  now one `mean_scores_by(df, group_col)`.
- **`print_score_table` and `print_quadrant_table`** were the same table with
  different column widths; now one `score_table(df, group_col, order, title)`
  that prints, optionally saves a CSV, and returns the DataFrame.
- **Four dropdown viewers** shared one widget skeleton; they are now four
  formatters over one `_render` helper.

### Behaviour changes this entailed

- **No `plt.show()` anywhere.** Matplotlib is pinned to the Agg backend at
  import; every plot function writes a PNG to `output/figures` and returns the
  path. Tables go to `output/tables` as CSV, text reports to `output/reports`.
  `OUTPUT_DIR` defaults to `<repo>/output` and honours `PHISHNET_OUTPUT_DIR`.
- **ipywidgets and IPython are gone**, and dropped from `requirements.txt`. The
  four viewers became `show_dataset_records`, `show_comparison`,
  `show_test_set_records` and `show_ai_detection_records`, which print a
  formatted record and can save it. They take an int, a list of indices, or
  `None` for every row.
- **The live dashboard's logic survives** as
  `metrics.generation.run_adversarial_pipeline(subject=...)`, which returns
  `{"subject", "sft", "bco", "kto"}` with the same early exit — `"skipped"` for
  BCO/KTO when SFT already evades. Only the button and the HTML went.
  `charts.show_pipeline_result()` renders it.
- **Heavy imports are now function-local.** `metrics.config`, `models`,
  `analysis` and `generation` all import with just pandas/numpy; torch,
  transformers, sentence-transformers, unsloth, sklearn and joblib load only
  when the function that needs them runs. Previously `import metrics.test_set`
  pulled in unsloth.
- The old figure-path constants (`GENERATORS_SPLIT_PLOT` etc.) are gone; each
  plot function takes a `name` argument instead and resolves it under
  `output/figures`.
- `save_model_and_metrics` now defaults to `output/` rather than
  `/kaggle/working/models`, which does not exist off Kaggle (§7).

### Verified

Analysis and reports were exercised on a synthetic 12-row report: the
phishing/safe split, `averages_by_side`, generator filtering, `score_table` for
both groupings, `evasion_stats` (evaded + blocked == total), quadrant
assignment, long-format melts, and the CSV/text writers. `grouped_bar` was
checked to place bars at exactly `x-w, x, x+w`, matching the four originals.

**Not verified: the rendered figures.** matplotlib and seaborn are not
installed in any environment on this machine and the venv has no pip, so no
plot function has actually been executed — only imported with stubs and
type-checked by eye. Run one chart before trusting the batch.
