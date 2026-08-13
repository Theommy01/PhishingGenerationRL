"""Paths, output locations and small text helpers.

Inputs (datasets, checkpoints) keep the notebook's Colab layout, overridable
with THESIS_BASE_DIR. Everything the code *produces* — figures, tables, text
reports — goes under OUTPUT_DIR instead, so nothing depends on a notebook
display any more.
"""

import gc
import os

# --- inputs ---------------------------------------------------------------

BASE_DIR = os.environ.get("THESIS_BASE_DIR", "/content/drive/MyDrive/Thesisproject")

DATASET_DIR = f"{BASE_DIR}/Dataset"
MODELS_DIR = f"{BASE_DIR}/Models"
DETECTION_SOURCES_DIR = f"{BASE_DIR}/DetectionDataSources"

MASTER_TRAINING_DATASET = f"{DATASET_DIR}/master_training_dataset.jsonl"
FINAL_EVALUATION_REPORT = f"{DATASET_DIR}/final_evaluation_report.csv"
TEST_SET_FULL_RESULTS = f"{DATASET_DIR}/test_set_full_results.csv"
AI_DETECTION_REPORT = f"{DATASET_DIR}/ai_detection.csv"

PATH_SFT = f"{MODELS_DIR}/checkpoint-2104"
PATH_BCO = f"{MODELS_DIR}/bco_model_ep3"
PATH_KTO = f"{MODELS_DIR}/kto_model_ep3"

DETECTION_DATASET_PATHS = {
    "SpamAssasin": f"{DETECTION_SOURCES_DIR}/SpamAssasin.csv",
    "Nigerian_Fraud": f"{DETECTION_SOURCES_DIR}/Nigerian_Fraud.csv",
    "Phishing_Email": f"{DETECTION_SOURCES_DIR}/phishing_email.csv",
    "CEAS_08": f"{DETECTION_SOURCES_DIR}/CEAS_08.csv",
    "Enron": f"{DETECTION_SOURCES_DIR}/Enron.csv",
    "Ling": f"{DETECTION_SOURCES_DIR}/Ling.csv",
    "Nazario": f"{DETECTION_SOURCES_DIR}/Nazario.csv",
}

SVM_MODEL_PATH = os.environ.get("PHISHNET_SVM_MODEL", "svm_email_classifier.pkl")

# Score at or above which ScamLLM is taken to consider a message safe.
SAFE_THRESHOLD = 0.50


# --- outputs --------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    """Drop a model/generator and empty the CUDA caches."""
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
