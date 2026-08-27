# Superseded artefacts

Nothing in this directory is read by any code in the repository — checked by
grepping every `*.py` for each filename. They are kept because the thesis
figures and tables were produced from them.

| What | Why it is here |
|---|---|
| `dpo_dataset*.jsonl` | DPO was tried and dropped; the project uses BCO and KTO. |
| `kto_dataset*.jsonl` | Superseded by `master_training_dataset.jsonl`, which carries the same pairs plus the ScamLLM score and the label both trainers read. |
| `*_withlinks.*` | A variant run where the entity placeholders were replaced with literal URLs. Not the corpus the SFT checkpoint was trained on. |
| `adversarial_rl_dataset.jsonl`, `master_evaluation_dataset.jsonl` | Notebook-era intermediates. |
| `coherence_metrics_results.csv`, `report_with_ai_prob.csv`, `ai_detection_percentages_only.csv`, `qualitative_analysis.csv`, `tabella_tesi_finale.csv`, `test_set_results.csv` | Tables the notebook wrote. `visualisation/` now writes its equivalents to `output/tables/`. |
| `AI_Detection_*.png` | Figures the notebook wrote. `visualisation/charts.py` regenerates these into `output/figures/`. |

The files still in `Dataset/` are the four the code reads —
`master_training_dataset.jsonl`, `final_evaluation_report.csv`,
`test_set_full_results.csv`, `ai_detection.csv` — plus
`final_evaluation_report_corrected.csv` and `starting_dataset.jsonl`, which
`PORTING_NOTES.md` and `TODO.md` refer to.
