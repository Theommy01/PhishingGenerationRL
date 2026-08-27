"""Paths, output locations and small text helpers.

Inputs (datasets, checkpoints) keep the notebook's Colab layout, overridable
with THESIS_BASE_DIR. Everything the code *produces* — figures, tables, text
reports — goes under OUTPUT_DIR instead, so nothing depends on a notebook
display any more.
"""

import gc
import os

# --- inputs ---------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The notebook ran on Colab and mounted Drive at this path. The repo carries its
# own Dataset/ and Models/ directories, so those are the defaults now; set
# THESIS_BASE_DIR to point elsewhere (e.g. back at a Drive mount).
BASE_DIR = os.environ.get("THESIS_BASE_DIR", _REPO_ROOT)

DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DETECTION_SOURCES_DIR = os.path.join(BASE_DIR, "DetectionDataSources")

MASTER_TRAINING_DATASET = os.path.join(DATASET_DIR, "master_training_dataset.jsonl")
FINAL_EVALUATION_REPORT = os.path.join(DATASET_DIR, "final_evaluation_report.csv")
TEST_SET_FULL_RESULTS = os.path.join(DATASET_DIR, "test_set_full_results.csv")
AI_DETECTION_REPORT = os.path.join(DATASET_DIR, "ai_detection.csv")

# Checkpoints are overridable individually, because they are gitignored (a few
# hundred MB each) and so differ per checkout. checkpoint-2122 is the SFT
# adapter currently on disk; Models/checkpoint-2104 is the one the notebook
# used and is not in this checkout.
PATH_SFT = os.environ.get(
    "PHISHNET_SFT_CHECKPOINT", os.path.join(_REPO_ROOT, "checkpoint-2122")
)

# The legacy BCO/KTO adapters, used only by the wide notebook-era report in
# metrics.analysis. They were trained on master_training_dataset.jsonl, whose
# labels are largely inverted (see PORTING_NOTES.md §1), so they are historical
# artefacts: do not use them as a warm start or a reference model for the loop.
PATH_BCO = os.environ.get(
    "PHISHNET_BCO_CHECKPOINT", os.path.join(MODELS_DIR, "bco_model_ep3")
)
PATH_KTO = os.environ.get(
    "PHISHNET_KTO_CHECKPOINT", os.path.join(MODELS_DIR, "kto_model_ep3")
)

DETECTION_DATASET_PATHS = {
    name: os.path.join(DETECTION_SOURCES_DIR, filename)
    for name, filename in {
        "SpamAssasin": "SpamAssasin.csv",
        "Nigerian_Fraud": "Nigerian_Fraud.csv",
        "Phishing_Email": "phishing_email.csv",
        "CEAS_08": "CEAS_08.csv",
        "Enron": "Enron.csv",
        "Ling": "Ling.csv",
        "Nazario": "Nazario.csv",
    }.items()
}

SVM_MODEL_PATH = os.environ.get(
    "PHISHNET_SVM_MODEL", os.path.join(_REPO_ROOT, "svm_email_classifier.pkl")
)

# Score at or above which ScamLLM is taken to consider a message safe.
SAFE_THRESHOLD = 0.50

# Seed for a training round: data order, dropout, and anything else TRL draws.
# 3407 is what both trainers hardcoded, kept as the default so existing runs
# behave identically; the loop records the seed it used on each round.
#
# It does not buy bit-identical adapters. 4-bit quantised training runs
# non-deterministic CUDA kernels, and reproducing a round exactly would also
# need torch.use_deterministic_algorithms(True) and a matching environment.
# What it does buy is a round whose inputs are all written down: dataset hash,
# base checkpoint hash, config, seed.
DEFAULT_TRAINING_SEED = 3407


# --- outputs --------------------------------------------------------------

OUTPUT_DIR = os.environ.get("PHISHNET_OUTPUT_DIR", os.path.join(_REPO_ROOT, "output"))
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")


def figure_path(name: str) -> str:
    """Absolute path for a figure, creating the directory on the way."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)


def table_path(name: str) -> str:
    os.makedirs(TABLES_DIR, exist_ok=True)
    return os.path.join(TABLES_DIR, name)


def report_path(name: str) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    return os.path.join(REPORTS_DIR, name)


def save_table(df, name: str, index: bool = True) -> str:
    """Write a DataFrame to output/tables and return the path."""
    path = table_path(name if name.endswith(".csv") else f"{name}.csv")
    df.to_csv(path, index=index)
    print(f"saved {path}")
    return path


def save_text(text: str, name: str) -> str:
    """Write a text report to output/reports and return the path."""
    path = report_path(name if name.endswith(".txt") else f"{name}.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"saved {path}")
    return path


# --- text helpers ---------------------------------------------------------


def extract_body_after_arrow(text: str) -> str:
    """Body extraction used by the test-set run and the adversarial pipeline."""
    return text.split("->\n")[1].strip() if "->\n" in text else text.strip()


def extract_body_after_body_tag(text) -> str:
    """Body extraction used by the evaluation report and the AI detector.

    Tolerates NaN so it can be mapped straight over a pandas column.
    """
    if text is None or (isinstance(text, float) and text != text):  # NaN
        return ""
    if "->\nbody:" in text:
        return text.split("->\nbody:")[1].strip()
    return str(text).strip()


def extract_subject(prompt_text: str) -> str:
    """Pull the subject line out of a structured prompt, for display labels."""
    try:
        return prompt_text.split("subject:")[1].split("\n")[0].strip()
    except Exception:
        return prompt_text[:50] + "..."


def extract_url_flag(prompt_text: str) -> bool:
    """Whether the prompt asked for a URL ("urls: True")."""
    try:
        lines = prompt_text.split("\n")
        url_line = [line for line in lines if line.startswith("urls:")][0]
        return url_line.split(":")[1].strip() == "True"
    except Exception:
        return False


def free_vram(model_instance=None, generator_instance=None) -> None:
    """Collect garbage and empty the CUDA caches.

    The two arguments are vestigial and do almost nothing: `del` on a parameter
    drops only this function's reference, so whatever the caller still holds
    keeps the model alive and resident. **Callers must rebind their own
    references to None first**, then call this with no arguments. They are kept
    only so the notebook-era call sites still run.
    """
    import torch  # imported lazily so the pandas-only paths stay torch-free

    try:
        if model_instance is not None:
            del model_instance
        if generator_instance is not None:
            del generator_instance
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
