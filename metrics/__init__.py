"""Metrics extracted from Generate_Message.ipynb.

    config.py      paths, output locations, text helpers
    models.py      ScamLLM, the AI-text detector, SBERT — all loaded lazily
    generation.py  evaluation report, test set, adversarial pipeline
    analysis.py    report aggregation and the SVM baseline detector

Submodules are not imported here: `generation` pulls in unsloth and `analysis`
pulls in sklearn, so importing the package would drag in everything. Import what
you need, e.g. `from metrics import analysis`.
"""
