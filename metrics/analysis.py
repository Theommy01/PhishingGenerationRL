"""Analysis over the evaluation report, plus the SVM baseline detector.

The report is sliced three ways — by SFT's verdict, by which assistant wrote the
prompt, and by the link/starting-point quadrant — but all three are the same
group-by-and-average over the same three score columns, so they share
`mean_scores_by` and `score_table`.
"""

import json
import os
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from metrics import config

warnings.filterwarnings("ignore")

MODELS = ["SFT", "BCO", "KTO"]
SCORE_COLUMNS = [f"{m}_ScamLLM_Score" for m in MODELS]
COSINE_COLUMNS = [f"{m}_Cosine_Sim" for m in MODELS]
KL_COLUMNS = [f"{m}_KL_Div" for m in MODELS]

MAIN_COLUMNS = [
    "prompt",
    "SFT_Is_Safe",
    *SCORE_COLUMNS,
    "SFT_vs_BCO_Cosine_Sim",
    "SFT_vs_KTO_Cosine_Sim",
    "SFT_vs_BCO_KL_Div",
    "SFT_vs_KTO_KL_Div",
]

GENERATORS_ORDER = ["ChatGPT", "Gemini", "Copilot"]

QUADRANTS_ORDER = [
    "Without Link\nStarting point: malicious",
    "Without Link\nStarting point: safe",
    "With Link\nStarting point: malicious",
    "With Link\nStarting point: safe",
]


# =============================================================================
# Loading and slicing
# =============================================================================


def load_report(csv_path: str = config.FINAL_EVALUATION_REPORT) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def split_by_sft_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split on the dataset's SFT_Is_Safe label into (phishing, safe)."""
    return (
        df[df["SFT_Is_Safe"] == False].copy(),  # noqa: E712
        df[df["SFT_Is_Safe"] == True].copy(),  # noqa: E712
    )


def split_by_sft_outcome(
    df: pd.DataFrame, threshold: float = config.SAFE_THRESHOLD
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split on whether SFT's *score* cleared the threshold: (malicious, safe)."""
    df["SFT_Evaded"] = df["SFT_ScamLLM_Score"] >= threshold
    return (
        df[df["SFT_Evaded"] == False],  # noqa: E712
        df[df["SFT_Evaded"] == True],  # noqa: E712
    )


def map_generator(gen_string) -> str:
    """Normalise the free-text generator name to one of the three targets."""
    gen_lower = str(gen_string).lower()
    if "gpt" in gen_lower or "chatgpt" in gen_lower:
        return "ChatGPT"
    elif "gemini" in gen_lower:
        return "Gemini"
    elif "copilot" in gen_lower:
        return "Copilot"
    return "Other"


def load_filtered_report(csv_path: str = config.FINAL_EVALUATION_REPORT) -> pd.DataFrame:
    """Load the report, tag each row with Target_Generator, drop 'Other'."""
    df = load_report(csv_path)
    df["Target_Generator"] = df["generator"].apply(map_generator)
    return df[df["Target_Generator"].isin(GENERATORS_ORDER)].copy()


def get_quadrant(row) -> str:
    link_status = "With Link" if row["Has_URL"] else "Without Link"
    sft_status = (
        "Starting point: safe" if row["SFT_Evaded"] else "Starting point: malicious"
    )
    return f"{link_status}\n{sft_status}"


def add_quadrants(
    df: pd.DataFrame, threshold: float = config.SAFE_THRESHOLD
) -> pd.DataFrame:
    """Tag every row with Has_URL, SFT_Evaded and Quadrant."""
    df["Has_URL"] = df["prompt"].apply(config.extract_url_flag)
    df["SFT_Evaded"] = df["SFT_ScamLLM_Score"] >= threshold
    df["Quadrant"] = df.apply(get_quadrant, axis=1)
    return df


# =============================================================================
# Aggregation — one implementation, three groupings
# =============================================================================


def mean_scores_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Mean ScamLLM score per group and model, as percentages."""
    return df.groupby(group_col)[SCORE_COLUMNS].mean() * 100


def score_table(
    df: pd.DataFrame,
    group_col: str,
    order: Optional[List[str]] = None,
    title: str = "",
    save_as: Optional[str] = None,
) -> pd.DataFrame:
    """Per-group mean-score table: printed, optionally saved, always returned."""
    counts = df.groupby(group_col).size()
    if counts.empty:
        return pd.DataFrame()

    means = mean_scores_by(df, group_col)
    order = [g for g in (order or list(counts.index)) if g in counts.index]

    table = pd.DataFrame(
        {
            "Total emails": [counts[g] for g in order],
            "Avg Score SFT": [means.loc[g, "SFT_ScamLLM_Score"] for g in order],
            "Avg Score BCO": [means.loc[g, "BCO_ScamLLM_Score"] for g in order],
            "Avg Score KTO": [means.loc[g, "KTO_ScamLLM_Score"] for g in order],
        },
        index=[g.replace("\n", " ") for g in order],
    ).round(2)

    if title:
        width = max(60, len(title))
        print("=" * width)
        print(title)
        print("=" * width)
    print(table.to_string())
    print()

    if save_as:
        config.save_table(table, save_as)

    return table


def scam_average(df: pd.DataFrame) -> Dict[str, float]:
    """Mean ScamLLM score per model, as a percentage."""
    return {m: df[f"{m}_ScamLLM_Score"].mean() * 100 for m in MODELS}


def cos_average(df: pd.DataFrame) -> Dict[str, float]:
    """Mean cosine similarity to the prompt, per model (already a percentage)."""
    return {m: df[f"{m}_Cosine_Sim"].mean() for m in MODELS}


def kl_average(df: pd.DataFrame) -> Dict[str, float]:
    """Mean KL divergence from the prompt, per model."""
    return {m: df[f"{m}_KL_Div"].mean() for m in MODELS}


def averages_by_side(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """All six averages: scam/cosine/KL, for the phishing and the safe side."""
    df_phishing, df_safe = split_by_sft_label(df)
    return {
        "scam_phishing": scam_average(df_phishing),
        "cos_phishing": cos_average(df_phishing),
        "kl_phishing": kl_average(df_phishing),
        "scam_safe": scam_average(df_safe),
        "cos_safe": cos_average(df_safe),
        "kl_safe": kl_average(df_safe),
    }


def semantic_comparison_table(
    df: pd.DataFrame, save_as: Optional[str] = None
) -> pd.DataFrame:
    """SFT-vs-RLHF cosine/KL means, split by phishing and safe prompts."""
    df_phishing, df_safe = split_by_sft_label(df)

    rows = [
        ("Cosine Sim (SFT vs BCO) [%]", "SFT_vs_BCO_Cosine_Sim"),
        ("Cosine Sim (SFT vs KTO) [%]", "SFT_vs_KTO_Cosine_Sim"),
        ("KL Divergence (SFT vs BCO)", "SFT_vs_BCO_KL_Div"),
        ("KL Divergence (SFT vs KTO)", "SFT_vs_KTO_KL_Div"),
    ]
    table = pd.DataFrame(
        {
            "Semantic comparison": [label for label, _ in rows],
            "Average computed on Phishing": [df_phishing[c].mean() for _, c in rows],
            "Average computed on Safe": [df_safe[c].mean() for _, c in rows],
        }
    ).round(4)

    if save_as:
        config.save_table(table, save_as, index=False)
    return table


def evasion_stats(
    df: pd.DataFrame, threshold: float = config.SAFE_THRESHOLD
) -> dict:
    """Evaded / blocked counts and percentages per generator, per model.

    Adds the SFT_Evaded / BCO_Evaded / KTO_Evaded columns in place.
    """
    for m in MODELS:
        df[f"{m}_Evaded"] = df[f"{m}_ScamLLM_Score"] >= threshold

    counts = df.groupby("Target_Generator").size()
    generators = [g for g in GENERATORS_ORDER if g in counts.index]

    evaded = {m: df.groupby("Target_Generator")[f"{m}_Evaded"].sum() for m in MODELS}
    blocked = {m: counts - evaded[m] for m in MODELS}

    return {
        "generators": generators,
        "counts": counts,
        "evaded": evaded,
        "blocked": blocked,
        "evaded_pct": {
            m: [evaded[m][g] / counts[g] * 100 for g in generators] for m in MODELS
        },
        "blocked_pct": {
            m: [blocked[m][g] / counts[g] * 100 for g in generators] for m in MODELS
        },
    }


def scores_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Melt the three score columns into Generator / Model / Score rows."""
    rows = []
    for gen in GENERATORS_ORDER:
        sub = df[df["Target_Generator"] == gen]
        for model, col in zip(MODELS, SCORE_COLUMNS):
            for value in sub[col] * 100:
                rows.append({"Generator": gen, "Model": model, "Score": value})
    return pd.DataFrame(rows)


def score_vs_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """Score / Similarity / Generator / Model rows, for the alignment-tax plot."""
    frames = []
    for model in MODELS:
        temp = df[
            [f"{model}_ScamLLM_Score", f"{model}_Cosine_Sim", "Target_Generator"]
        ].copy()
        temp.columns = ["Score", "Similarity", "Generator"]
        temp["Model"] = model
        frames.append(temp)

    out = pd.concat(frames)
    out["Score"] *= 100
    return out


def dataset_summary(jsonl_path: str = config.MASTER_TRAINING_DATASET) -> pd.DataFrame:
    """Print the safe/malicious breakdown of the pre-training dataset."""
    df = pd.read_json(jsonl_path, lines=True)

    total = len(df)
    safe = df["label"].sum()
    malicious = total - safe

    print("=" * 60)
    print("PRE-TRAINING DATASET")
    print("=" * 60)
    print(f"Emails generated: {total}")
    print(f"Safe (ScamLLM >= 50%):      {safe} ({safe / total * 100:.1f}%)")
    print(f"Malicious (ScamLLM < 50%):  {malicious} ({malicious / total * 100:.1f}%)")
    print("=" * 60 + "\n")

    return df


# =============================================================================
# SVM baseline detector
#
# Adapted from https://www.kaggle.com/code/vnice85/email-svm/notebook
# The notebook pinned scikit-learn==1.4.2 and imbalanced-learn==0.11.0.
# =============================================================================

CLASS_NAMES = ["Non-Spam", "Spam"]

SAMPLE_EMAILS = [
    "Congratulations! You have won $1,000,000 in our lottery! Click here to claim your prize immediately!",
    "Hi John, can you please send me the report by tomorrow? Thanks!",
    "URGENT: Your account will be suspended unless you verify your information now. Click here immediately!",
    "Meeting scheduled for next Tuesday at 2 PM. Please confirm your attendance.",
    "FREE MONEY! Get rich quick! No work required! Click now!",
    "Your order has been shipped and will arrive in 2-3 business days.",
    "WINNER! You are selected for our special promotion. Send us your bank details now!",
    "Please review the attached document and let me know your feedback.",
]


def load_datasets(file_paths: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """Load each detection source CSV and print its spam/non-spam breakdown."""
    if file_paths is None:
        file_paths = config.DETECTION_DATASET_PATHS

    datasets, summary = {}, {}
    for name, path in file_paths.items():
        try:
            source = pd.read_csv(path)
            datasets[name] = source

            print(f"\n{name} Dataset:")
            print(f"Shape: {source.shape}")
            print(f"Columns: {list(source.columns)}")

            if "label" in source.columns:
                spam = source[source["label"] == 1].shape[0]
                non_spam = source[source["label"] == 0].shape[0]
                print(f"Spam: {spam}, Non-spam: {non_spam}")
            else:
                spam, non_spam = source.shape[0], 0
                print(f"All emails considered as spam: {spam}")

            summary[name] = {"spam_count": spam, "non_spam_count": non_spam}
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

    print("\n" + "=" * 50)
    print("SUMMARY OF ALL DATASETS:")
    print("=" * 50)
    for name, counts in summary.items():
        print(f"{name}: {counts['spam_count']} spam, {counts['non_spam_count']} non-spam")

    return datasets


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Basic text preprocessing: drop nulls, lowercase, strip, drop empties."""
    df = df.copy()
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.lower().str.strip()
    return df[df["text"].str.len() > 0]


def combine_datasets(datasets: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Normalise each source to text/label/source columns and concatenate."""
    combined = []

    for name, source in datasets.items():
        print(f"\nProcessing {name}...")

        text_column = next(
            (c for c in ("text_combined", "text", "body", "content") if c in source.columns),
            None,
        )
        if not text_column:
            print(f"No text column found in {name}")
            continue

        subset = source[[text_column]].copy()
        subset["label"] = source["label"] if "label" in source.columns else 1
        subset.columns = ["text", "label"]
        subset["source"] = name
        combined.append(subset)
        print(f"Added {len(subset)} emails from {name}")

    if not combined:
        print("No datasets with text content found!")
        return None

    combined_df = pd.concat(combined, ignore_index=True)
    print(f"\nCombined dataset created with {len(combined_df)} emails")

    combined_df = preprocess_text(combined_df)
    print(f"After preprocessing: {len(combined_df)} emails")
    print(f"\nClass distribution:\n{combined_df['label'].value_counts()}")
    print(f"\nSource distribution:\n{combined_df['source'].value_counts()}")
    print(f"\nSample data:\n{combined_df.head()}")

    return combined_df


def describe_text_lengths(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Add the text_length column and return its per-label description."""
    combined_df["text_length"] = combined_df["text"].str.len()
    description = combined_df.groupby("label")["text_length"].describe()
    print("Text length statistics:")
    print(description)
    return description


def split_dataset(
    combined_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """Stratified train/test split on the combined dataset."""
    from sklearn.model_selection import train_test_split

    X, y = combined_df["text"], combined_df["label"]
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {Counter(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training class distribution: {Counter(y_train)}")
    print(f"Test class distribution: {Counter(y_test)}")

    return X_train, X_test, y_train, y_test


def create_svm_pipeline(use_undersampling: bool = True, max_features: int = 1000):
    """Create the TF-IDF + linear SVM pipeline, with optional undersampling."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        strip_accents="unicode",
    )
    svm = SVC(kernel="linear", C=1.0, probability=False, random_state=42)

    if use_undersampling:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.under_sampling import RandomUnderSampler

        return ImbPipeline(
            [
                ("tfidf", tfidf),
                ("undersampler", RandomUnderSampler(random_state=42)),
                ("svm", svm),
            ]
        )

    return Pipeline([("tfidf", tfidf), ("svm", svm)])


def train_pipeline(pipeline, X_train, y_train):
    """Fit the pipeline. Took roughly 18 minutes on the notebook's data."""
    print("Training SVM model...")
    pipeline.fit(X_train, y_train)
    print("Model training completed!")
    return pipeline


def evaluate_model(pipeline, X_test, y_test, class_names=None):
    """Evaluate the trained model. Returns (accuracy, y_pred).

    The SVC is built with probability=False, so there is no predict_proba.
    """
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    if class_names is None:
        class_names = CLASS_NAMES

    print("Making predictions...")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return accuracy, y_pred


def cross_validate(pipeline, X_train, y_train, cv: int = 5):
    """5-fold accuracy cross-validation. Took about an hour on the full data."""
    from sklearn.model_selection import cross_val_score

    print("Performing cross-validation...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    return cv_scores


def top_features(pipeline, top_n: int = 20):
    """Most and least spam-indicating TF-IDF features.

    Returns (spam_features, spam_scores, nonspam_features, nonspam_scores),
    the spam side strongest first.
    """
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    svm_coef = pipeline.named_steps["svm"].coef_.toarray().flatten()

    top_positive_idx = np.argsort(svm_coef)[-top_n:]
    top_negative_idx = np.argsort(svm_coef)[:top_n]

    spam_features = [feature_names[i] for i in reversed(top_positive_idx)]
    spam_scores = [svm_coef[i] for i in reversed(top_positive_idx)]
    nonspam_features = [feature_names[i] for i in top_negative_idx]
    nonspam_scores = [svm_coef[i] for i in top_negative_idx]

    print(f"Top {top_n} features indicating SPAM:")
    print("-" * 50)
    for name, score in zip(spam_features, spam_scores):
        print(f"{name}: {score:.4f}")
    print(f"\nTop {top_n} features indicating NON-SPAM:")
    print("-" * 50)
    for name, score in zip(nonspam_features, nonspam_scores):
        print(f"{name}: {score:.4f}")

    return spam_features, spam_scores, nonspam_features, nonspam_scores


def test_sample_emails(pipeline, sample_emails=None) -> None:
    """Print the prediction for a handful of hand-written emails."""
    if sample_emails is None:
        sample_emails = SAMPLE_EMAILS

    print("Testing with sample emails:")
    print("=" * 80)
    for i, email in enumerate(sample_emails, 1):
        pred = pipeline.predict([email])[0]
        print(f"\nEmail {i}:")
        print(f"Text: {email}")
        print(f"Prediction: {'SPAM' if pred == 1 else 'NON-SPAM'}")
        print("-" * 80)


def save_pipeline(pipeline, model_path: str = config.SVM_MODEL_PATH) -> str:
    import joblib

    joblib.dump(pipeline, model_path)
    print(f"saved {model_path}")
    return model_path


def load_pipeline(model_path: str = config.SVM_MODEL_PATH):
    import joblib

    return joblib.load(model_path)


def save_model_and_metrics(
    pipeline, y_test, y_pred, accuracy: float, cv_scores, save_dir: Optional[str] = None
) -> dict:
    """Persist the fitted pipeline next to its accuracy / CV / confusion matrix."""
    import joblib
    from sklearn.metrics import confusion_matrix

    if save_dir is None:
        save_dir = config.OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "svm_phishing_detector.pkl")
    joblib.dump(pipeline, model_path)
    print(f"saved {model_path}")

    metrics = {
        "accuracy": float(accuracy),
        "cv_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    metrics_path = os.path.join(save_dir, "svm_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {metrics_path}")

    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Cross-validation Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return metrics


# =============================================================================
# ScamLLM vs SVM
# =============================================================================

MODEL_TO_COLUMNS = {m: (f"{m}_ScamLLM_Score", f"SVM_Pred_{m}") for m in MODELS}


def add_svm_predictions(df: pd.DataFrame, pipeline, prefix: str = "SVM_Pred_") -> pd.DataFrame:
    """Run the SVM over the SFT/BCO/KTO texts and store the predictions."""
    for model in MODELS:
        df[f"{prefix}{model}"] = pipeline.predict(df[f"{model}_Text"].tolist())
    return df


def format_result(score: float, pred: int) -> str:
    """One cell of the side-by-side table: ScamLLM verdict / SVM verdict."""
    status = "Evade" if score * 100 > 50 else "Blocked"
    svm = "Detected" if pred == 1 else "Evaded"
    return f"{status} (ScamLLM) / {svm} (SVM)"


def comparison_table(df: pd.DataFrame, save_as: Optional[str] = None) -> pd.DataFrame:
    """Per-prompt ScamLLM-vs-SVM verdicts for the three models.

    NOTE: reproduces the notebook exactly, including pairing the BCO and KTO
    columns with the SVM prediction of the *SFT* text. See PORTING_NOTES.md §5.
    """
    table = pd.DataFrame(
        {
            "Prompt": df["prompt"].str[:30] + "...",
            **{
                m: [
                    format_result(s, p)
                    for s, p in zip(df[f"{m}_ScamLLM_Score"], df["SVM_SFT"])
                ]
                for m in MODELS
            },
        }
    )
    if save_as:
        config.save_table(table, save_as, index=False)
    return table


def score_percentage_stats(df: pd.DataFrame, save_as: Optional[str] = None) -> pd.DataFrame:
    """describe() over the three ScamLLM scores, converted to percentages."""
    for m in MODELS:
        df[f"{m}_Score_Pct"] = df[f"{m}_ScamLLM_Score"] * 100
    stats = df[[f"{m}_Score_Pct" for m in MODELS]].describe()
    if save_as:
        config.save_table(stats, save_as)
    return stats


def concordance(df: pd.DataFrame, save_as: Optional[str] = None) -> pd.DataFrame:
    """Agreement rate (%) between ScamLLM and the SVM, per model.

    NOTE: the notebook reads `score > 0.5` as "ScamLLM says spam" while the same
    score is treated as a *safe* probability everywhere else. Kept as-is; see
    PORTING_NOTES.md §5.
    """
    rows = []
    for model, (score_col, pred_col) in MODEL_TO_COLUMNS.items():
        scamllm_is_spam = df[score_col] > 0.5
        svm_is_spam = df[pred_col] == 1
        rows.append(
            {
                "Model": model,
                "Concordance (%)": (scamllm_is_spam == svm_is_spam).mean() * 100,
            }
        )
    table = pd.DataFrame(rows)
    if save_as:
        config.save_table(table, save_as, index=False)
    return table


def compare_test_set_variants(
    csv_path: str = config.TEST_SET_FULL_RESULTS,
    model_path: str = config.SVM_MODEL_PATH,
    pipeline=None,
) -> str:
    """Return (and print) the SVM verdict for each model's text, prompt by prompt."""
    df = pd.read_csv(csv_path)
    if pipeline is None:
        pipeline = load_pipeline(model_path)

    by_model = {m: dict(zip(df["Prompt (Test Set)"], df[f"{m}_Text"])) for m in MODELS}

    lines = ["Testing the dataset emails (SFT, BCO, KTO):", "=" * 100]
    for prompt in df["Prompt (Test Set)"]:
        lines.append(f"\nPROMPT: {prompt}")
        for model in MODELS:
            pred = pipeline.predict([by_model[model][prompt]])[0]
            lines.append(f"  [{model}] Prediction: {'SPAM' if pred == 1 else 'NON-SPAM'}")
        lines.append("-" * 100)

    text = "\n".join(lines)
    print(text)
    return text


def run_full_comparison(
    csv_path: str = config.FINAL_EVALUATION_REPORT,
    model_path: str = config.SVM_MODEL_PATH,
    df: Optional[pd.DataFrame] = None,
    save: bool = True,
):
    """The whole ScamLLM-vs-SVM section: verdict table, stats, concordance."""
    if df is None:
        df = load_report(csv_path)
    pipeline = load_pipeline(model_path)

    # the verdict table reads the SVM_SFT / SVM_BCO / SVM_KTO naming
    df = add_svm_predictions(df, pipeline, prefix="SVM_")
    verdicts = comparison_table(df, save_as="svm_vs_scamllm_verdicts" if save else None)
    print(verdicts.to_string())

    stats = score_percentage_stats(df, save_as="scamllm_score_stats" if save else None)
    print(stats)

    df = add_svm_predictions(df, pipeline, prefix="SVM_Pred_")
    table = concordance(df, save_as="detector_concordance" if save else None)
    print(table)

    return df, table
