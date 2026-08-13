"""Static figures and text reports, all written to disk.

There is no notebook any more, so nothing calls `plt.show()`: every plot
function saves a PNG under `output/figures` and returns its path. Matplotlib
runs on the Agg backend so this works headless.

The four grouped-bar charts and the three box/strip charts share the
`grouped_bar` and `box_strip` primitives rather than repeating the same
offset/annotate/threshold-line code.
"""

import matplotlib

matplotlib.use("Agg")

from typing import Dict, List, Optional, Sequence  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from metrics import analysis, config  # noqa: E402
from metrics.models import AI_PROB_COLUMNS, mean_ai_probabilities  # noqa: E402
from visualisation.style import (  # noqa: E402
    GENERATOR_PALETTE_FLAT,
    MODEL_LABELS,
    PALETTE_AI,
    PALETTE_BY_MODEL,
    PALETTE_GREEN,
    PALETTE_ORANGE,
    annotate_bars,
    annotate_counts,
    annotate_scores,
)

sns.set_theme(style="whitegrid")

AI_MODEL_LABELS = ["SFT (Base)", "BCO (RLHF)", "KTO (RLHF)"]


# =============================================================================
# Primitives
# =============================================================================


def save_figure(fig, name: str, dpi: int = 300) -> str:
    """Write a figure to output/figures, close it, return the path."""
    path = config.figure_path(name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


def grouped_bar(
    ax,
    values_by_series: Dict[str, Sequence[float]],
    xtick_labels: Sequence[str],
    colors: Sequence[str] = PALETTE_ORANGE,
    series_labels: Optional[Sequence[str]] = None,
    width: float = 0.25,
    fontsize: int = 12,
):
    """Draw one bar per series per group, centred on each tick.

    Returns the list of bar containers so the caller can annotate them.
    """
    series = list(values_by_series.items())
    x = np.arange(len(xtick_labels))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width

    rect_groups = []
    for i, (name, values) in enumerate(series):
        label = series_labels[i] if series_labels is not None else name
        rect_groups.append(
            ax.bar(
                x + offsets[i],
                values,
                width,
                label=label,
                color=colors[i % len(colors)],
                edgecolor="black",
            )
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=fontsize, fontweight="bold")
    return rect_groups


def threshold_line(ax, y: float = 50, label: str = "Safety threshold (50%)"):
    ax.axhline(y=y, color="#c0392b", linestyle="--", linewidth=2, alpha=0.8, label=label)


def box_strip(ax, df_long: pd.DataFrame, x: str, y: str, palette, title: str, ylabel: str):
    """Boxplot with the individual points jittered over it."""
    sns.boxplot(
        data=df_long, x=x, y=y, ax=ax, palette=palette, width=0.5,
        boxprops=dict(alpha=0.4),
    )
    sns.stripplot(
        data=df_long, x=x, y=y, ax=ax, color="black", alpha=0.3, jitter=0.2, size=3
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)


# =============================================================================
# Evaluation-report charts
# =============================================================================


def plot_metric_grid(df: pd.DataFrame, name: str = "metric_grid") -> str:
    """2x3 grid: evasion / cosine / KL, for phishing prompts then safe ones."""
    avg = analysis.averages_by_side(df)
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    models = analysis.MODELS

    panels = [
        (0, 0, avg["scam_phishing"], "1. Evasion Rate", "ScamLLM average score (%)", "darkred", (0, 100), "{:.1f}%"),
        (0, 1, avg["cos_phishing"], "2. Cosine Similarity", "Similarity (%)", "darkred", (0, 100), "{:.1f}%"),
        (0, 2, avg["kl_phishing"], "3. KL Divergence", "Divergence Score", "darkred", None, "{:.4f}"),
        (1, 0, avg["scam_safe"], "4. Safe Preservation", "ScamLLM average score (%)", "darkgreen", (0, 100), "{:.1f}%"),
        (1, 1, avg["cos_safe"], "5. Cosine Similarity", "Similarity (%)", "darkgreen", (0, 100), "{:.1f}%"),
        (1, 2, avg["kl_safe"], "6. KL Divergence", "Divergence Score", "darkgreen", None, "{:.4f}"),
    ]

    for row, col, values, title, ylabel, colour, ylim, fmt in panels:
        ax = axes[row, col]
        sns.barplot(x=models, y=[values[m] for m in models], palette=PALETTE_GREEN, ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold", color=colour)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim if ylim else (0, max(values.values()) * 1.2))
        annotate_bars(ax, fmt=fmt)

    fig.tight_layout(pad=3.0)
    return save_figure(fig, name)


def plot_distribution_grid(df: pd.DataFrame, name: str = "distribution_grid") -> str:
    """The 2x3 distribution counterpart of the metric grid."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    phishing, safe = analysis.split_by_sft_label(df)

    panels = [
        (0, 0, phishing, analysis.SCORE_COLUMNS, "1. Evasion Rate (Phishing)", "ScamLLM Score (%)"),
        (0, 1, phishing, analysis.COSINE_COLUMNS, "2. Cosine Similarity (Phishing)", "Similarity (%)"),
        (0, 2, phishing, analysis.KL_COLUMNS, "3. KL Divergence (Phishing)", "Div. Score"),
        (1, 0, safe, analysis.SCORE_COLUMNS, "4. Safe Preservation", "ScamLLM Score (%)"),
        (1, 1, safe, analysis.COSINE_COLUMNS, "5. Cosine Similarity (Safe)", "Similarity (%)"),
        (1, 2, safe, analysis.KL_COLUMNS, "6. KL Divergence (Safe)", "Div. Score"),
    ]

    for row, col, subset, cols, title, ylabel in panels:
        df_long = subset.melt(value_vars=list(cols), var_name="Model", value_name="Score")
        # only the ScamLLM columns are stored 0-1; cosine and KL are display-ready
        if "ScamLLM_Score" in cols[0]:
            df_long["Score"] *= 100
        df_long["Model"] = df_long["Model"].apply(
            lambda x: x.split("_")[0].replace("SFT", "SFT (Pre-training)")
        )
        box_strip(axes[row, col], df_long, "Model", "Score", PALETTE_GREEN, title, ylabel)

    fig.tight_layout(pad=3.0)
    return save_figure(fig, name)


def plot_evaded_vs_blocked(stats: dict, name: str = "generators_evaded_vs_blocked") -> str:
    """Side by side: emails that bypassed ScamLLM, and emails it detected."""
    generators, counts = stats["generators"], stats["counts"]
    xtick_labels = [f"{g}\n(n={counts[g]})" for g in generators]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Comparison: Bypassed vs Detected emails", fontsize=18, fontweight="bold", y=1.05)

    for ax, pct_key, abs_key, title in (
        (axes[0], "evaded_pct", "evaded", "Bypassed (emails that got past ScamLLM)"),
        (axes[1], "blocked_pct", "blocked", "Detected as malicious by ScamLLM"),
    ):
        rects = grouped_bar(
            ax, stats[pct_key], xtick_labels, series_labels=MODEL_LABELS
        )
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_ylabel("Safe percentage", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.legend(fontsize=11)
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
        for model, group in zip(analysis.MODELS, rects):
            annotate_counts(ax, group, stats[abs_key][model], generators, counts)

    fig.tight_layout()
    return save_figure(fig, name)


def plot_mean_scores_by_generator(
    df_filtered: pd.DataFrame, name: str = "generators_mean_scores"
) -> str:
    """Mean ScamLLM confidence per generator, with the 50% evasion line."""
    means = analysis.mean_scores_by(df_filtered, "Target_Generator")
    counts = df_filtered.groupby("Target_Generator").size()
    generators = [g for g in analysis.GENERATORS_ORDER if g in counts.index]

    fig, ax = plt.subplots(figsize=(12, 7))
    rects = grouped_bar(
        ax,
        {m: [means.loc[g, f"{m}_ScamLLM_Score"] for g in generators] for m in analysis.MODELS},
        [f"{g}\n(n={counts[g]})" for g in generators],
        series_labels=MODEL_LABELS,
        fontsize=13,
    )
    ax.set_ylabel("ScamLLM Confidence Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("ScamLLM Score: safeness detected", fontsize=16, pad=20, fontweight="bold")
    ax.set_ylim(0, 115)
    threshold_line(ax, label="Evasion rate (50%)")
    ax.legend(fontsize=11, loc="upper left")
    for group in rects:
        annotate_scores(ax, group)

    fig.tight_layout()
    return save_figure(fig, name)


def plot_malicious_vs_safe(
    df_malicious: pd.DataFrame,
    df_safe: pd.DataFrame,
    name: str = "generators_malicious_vs_safe",
) -> str:
    """Mean scores per generator, split by whether SFT started safe or malicious."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "ScamLLM score evolution by generator and starting state",
        fontsize=18, fontweight="bold", y=1.05,
    )

    for ax, subset, title in (
        (axes[0], df_malicious, "Malicious starting point\nAverage score"),
        (axes[1], df_safe, "Safe starting point\nAverage score"),
    ):
        counts = subset.groupby("Target_Generator").size()
        generators = [g for g in analysis.GENERATORS_ORDER if g in counts]
        if not generators:
            ax.set_title(f"{title}\n(no data)", fontsize=14)
            continue

        rects = grouped_bar(
            ax,
            {
                m: [
                    subset[subset["Target_Generator"] == g][f"{m}_ScamLLM_Score"].mean() * 100
                    for g in generators
                ]
                for m in analysis.MODELS
            },
            [f"{g}\n(n={counts[g]})" for g in generators],
            series_labels=MODEL_LABELS,
        )
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_ylabel("ScamLLM Confidence Score (%)", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.legend(fontsize=11)
        threshold_line(ax)
        for group in rects:
            annotate_scores(ax, group)

    fig.tight_layout()
    return save_figure(fig, name)


def plot_quadrants(
    mean_scores: pd.DataFrame, counts: pd.Series, name: str = "link_quadrants"
) -> str:
    """Mean ScamLLM score per link/starting-point quadrant."""
    fig, ax = plt.subplots(figsize=(15, 8))
    rects = grouped_bar(
        ax,
        {
            m: [
                mean_scores.loc[q, f"{m}_ScamLLM_Score"] if q in mean_scores.index else 0
                for q in analysis.QUADRANTS_ORDER
            ]
            for m in analysis.MODELS
        },
        [f"{q}\n(n={counts.get(q, 0)})" for q in analysis.QUADRANTS_ORDER],
        series_labels=MODEL_LABELS,
    )
    ax.set_ylabel("ScamLLM Confidence Score (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "ScamLLM Score by case: Recovery Rate vs Alignment Tax",
        fontsize=16, pad=20, fontweight="bold",
    )
    ax.set_ylim(0, 115)
    threshold_line(ax)
    ax.legend(fontsize=12, loc="upper right")
    for group in rects:
        annotate_scores(ax, group, offset=2)

    fig.tight_layout()
    return save_figure(fig, name)


def plot_scores_by_generator(
    df_filtered: pd.DataFrame, name: str = "generators_score_distribution"
) -> str:
    """Mean bar + per-email strip of the ScamLLM scores, grouped by generator."""
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle(
        "Evasion Performance: Distribution of ScamLLM Scores", fontsize=20, fontweight="bold"
    )

    df_plot = analysis.scores_long_format(df_filtered)
    sns.barplot(
        data=df_plot, x="Generator", y="Score", hue="Model",
        palette=GENERATOR_PALETTE_FLAT, edgecolor="black", alpha=0.9, ax=ax, errorbar=None,
    )
    sns.stripplot(
        data=df_plot, x="Generator", y="Score", hue="Model",
        dodge=True, color="black", alpha=0.25, size=3, ax=ax,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("ScamLLM Score (%)", fontsize=14)
    ax.set_xlabel("Generator", fontsize=14)
    ax.axhline(50, color="red", linestyle="--", alpha=0.5)

    # the stripplot doubles the legend entries; keep the first three
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], title="Model", loc="upper left", fontsize=12)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return save_figure(fig, name)


def plot_scores_by_generator_split(
    df_filtered: pd.DataFrame,
    threshold: float = config.SAFE_THRESHOLD,
    name: str = "generators_score_distribution_split",
) -> str:
    """Same distributions, side by side for malicious and safe starting points."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    for ax, kind, title in (
        (axes[0], "malicious", "Starting State: Malicious"),
        (axes[1], "safe", "Starting State: Safe"),
    ):
        subset = (
            df_filtered[df_filtered["SFT_ScamLLM_Score"] < threshold]
            if kind == "malicious"
            else df_filtered[df_filtered["SFT_ScamLLM_Score"] >= threshold]
        )
        df_plot = analysis.scores_long_format(subset)

        sns.barplot(
            data=df_plot, x="Generator", y="Score", hue="Model",
            palette=GENERATOR_PALETTE_FLAT, ax=ax, edgecolor="black", alpha=0.8, errorbar=None,
        )
        sns.stripplot(
            data=df_plot, x="Generator", y="Score", hue="Model",
            ax=ax, dodge=True, color="black", alpha=0.15, size=3,
        )
        ax.set_title(f"{title} (N={len(subset)})", fontsize=16, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.axhline(50, color="red", linestyle="--", alpha=0.5)
        ax.legend(title="Model", loc="upper left")

    fig.tight_layout()
    return save_figure(fig, name)


def plot_alignment_tax(df_scatter: pd.DataFrame, name: str = "alignment_tax") -> str:
    """Evasion score vs semantic similarity, with a regression line per model.

    A flat line means the model evades without sacrificing coherence.
    """
    grid = sns.lmplot(
        data=df_scatter, x="Score", y="Similarity", hue="Model",
        palette=PALETTE_BY_MODEL, height=7, aspect=1.3, markers=["o", "s", "^"],
        scatter_kws={"alpha": 0.3}, line_kws={"linewidth": 3},
    )
    grid.figure.suptitle(
        "Alignment Tax: Evasion Score vs Semantic Similarity",
        fontsize=16, fontweight="bold",
    )
    grid.set_axis_labels(
        "ScamLLM Evasion Score (%)", "Semantic Similarity to Base Prompt (%)", fontsize=13
    )
    for ax in grid.axes.flat:
        ax.axvline(50, color="red", linestyle="--", alpha=0.5)

    grid.figure.tight_layout()
    return save_figure(grid.figure, name)


# =============================================================================
# AI-detection charts
# =============================================================================


def plot_mean_ai_probability(df: pd.DataFrame, name: str = "ai_detection_mean") -> str:
    """Bar chart of the mean AI-detection probability, with exact values on top."""
    means = mean_ai_probabilities(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(AI_MODEL_LABELS, means, color=PALETTE_AI, edgecolor="black")
    ax.set_title("Mean AI-detection probability", fontsize=14)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 1, f"{y:.2f}%", ha="center", fontweight="bold")

    fig.tight_layout()
    return save_figure(fig, name)


def plot_ai_probability_distribution(
    df: pd.DataFrame, name: str = "ai_detection_distribution"
) -> str:
    """Boxplot + jitter of the per-email AI-detection probabilities."""
    df_melted = df[AI_PROB_COLUMNS].copy()
    df_melted.columns = AI_MODEL_LABELS
    df_melted = df_melted.melt(var_name="Model", value_name="AI probability (%)")

    fig, ax = plt.subplots(figsize=(10, 6))
    box_strip(
        ax, df_melted, "Model", "AI probability (%)", PALETTE_AI,
        "Distribution of AI-detection probabilities", "AI-generated probability (%)",
    )
    ax.set_ylim(-5, 105)

    fig.tight_layout()
    return save_figure(fig, name)


def plot_prompt_length_correlation(
    corr: pd.DataFrame, name: str = "prompt_length_correlation"
) -> str:
    """Heatmap of the prompt-length / AI-probability correlation matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Prompt length vs AI detection")
    fig.tight_layout()
    return save_figure(fig, name)


# =============================================================================
# SVM baseline charts
# =============================================================================


def plot_class_and_source_distribution(
    combined_df: pd.DataFrame, name: str = "svm_dataset_distribution"
) -> str:
    """Class balance and per-source counts of the combined detection dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    combined_df["label"].value_counts().plot(kind="bar", ax=axes[0])
    axes[0].set_title("Class Distribution")
    axes[0].set_xlabel("Label (0=Non-spam, 1=Spam)")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    combined_df["source"].value_counts().plot(kind="bar", ax=axes[1])
    axes[1].set_title("Source Distribution")
    axes[1].set_xlabel("Dataset Source")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return save_figure(fig, name)


def plot_text_length_distribution(
    combined_df: pd.DataFrame, name: str = "svm_text_lengths"
) -> str:
    """Histogram of text length per class. Requires the text_length column."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in (0, 1):
        ax.hist(
            combined_df[combined_df["label"] == label]["text_length"],
            bins=50, alpha=0.7, label=f"Label {label}",
        )
    ax.set_xlabel("Text Length")
    ax.set_ylabel("Frequency")
    ax.set_title("Text Length Distribution by Label")
    ax.legend()
    ax.set_xlim(0, 5000)  # limit for readability

    fig.tight_layout()
    return save_figure(fig, name)


def plot_confusion_matrix(y_test, y_pred, class_names=None, name: str = "svm_confusion_matrix") -> str:
    from sklearn.metrics import confusion_matrix

    if class_names is None:
        class_names = analysis.CLASS_NAMES

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    fig.tight_layout()
    return save_figure(fig, name)


def plot_cross_validation_scores(cv_scores, name: str = "svm_cv_scores") -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(cv_scores) + 1), cv_scores)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-Validation Scores")
    ax.set_ylim(0, 1)
    for i, score in enumerate(cv_scores):
        ax.text(i + 1, score + 0.01, f"{score:.3f}", ha="center")

    fig.tight_layout()
    return save_figure(fig, name)


def plot_feature_importance(
    spam_features, spam_scores, nonspam_features, nonspam_scores,
    name: str = "svm_feature_importance",
) -> str:
    """Horizontal bars of the strongest spam and non-spam TF-IDF features."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for ax, features, scores, title in (
        (axes[0], spam_features, spam_scores, "Top Features Indicating SPAM"),
        (axes[1], nonspam_features, nonspam_scores, "Top Features Indicating NON-SPAM"),
    ):
        ax.barh(range(len(features)), scores)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel("Feature Weight")
        ax.set_title(title)

    fig.tight_layout()
    return save_figure(fig, name)


# =============================================================================
# Text reports (replacing the notebook's ipywidgets viewers)
# =============================================================================

_RULE = "=" * 78
_THIN = "-" * 78


def _block(title: str, body: str) -> str:
    return f"{title}\n{_THIN}\n{body}\n"


def format_dataset_record(row) -> str:
    """One training-dataset email: prompt, verdict, completion."""
    verdict = "SAFE" if row["label"] else "MALICIOUS"
    return "\n".join(
        [
            _RULE,
            f"[{verdict}]  ScamLLM score: {row['score_scamllm'] * 100:.2f}%",
            f"Category: {row['category']}   Generated by: {row['generator']}",
            _RULE,
            _block("PROMPT", row["prompt"]),
            _block("COMPLETION", row["completion"]),
        ]
    )


def format_comparison(row) -> str:
    """One evaluation-report row: SFT, BCO and KTO stacked with their metrics."""
    header = "Safe w.r.t. SFT" if row["SFT_Is_Safe"] else "Malicious w.r.t. SFT"
    parts = [_RULE, header, _RULE, _block("PROMPT", row["prompt"])]

    parts.append(
        _block(
            "SFT",
            f"ScamLLM: {row['SFT_ScamLLM_Score'] * 100:.1f}%  |  "
            f"Cosine vs prompt: {row['SFT_Cosine_Sim']:.1f}%  |  "
            f"KL vs prompt: {row['SFT_KL_Div']:.4f}\n\n{row['SFT_Text']}",
        )
    )
    for model in ("BCO", "KTO"):
        parts.append(
            _block(
                model,
                f"ScamLLM: {row[f'{model}_ScamLLM_Score'] * 100:.1f}%  |  "
                f"Cosine vs SFT: {row[f'SFT_vs_{model}_Cosine_Sim']:.1f}%  |  "
                f"KL vs SFT: {row[f'SFT_vs_{model}_KL_Div']:.4f}\n\n{row[f'{model}_Text']}",
            )
        )
    return "\n".join(parts)


def format_test_set_record(row) -> str:
    """One held-out test-set row."""
    evaded = row["SFT Score (%)"] >= 50.0
    header = (
        "Evaded on the first try by SFT (safe)" if evaded else "Blocked by SFT (malicious)"
    )
    parts = [_RULE, header, _RULE, _block("PROMPT", str(row["Prompt"]))]

    parts.append(
        _block("SFT (baseline)", f"ScamLLM: {row['SFT Score (%)']:.2f}%\n\n{row['SFT_Text']}")
    )
    for model, label in (("BCO", "BCO (binary)"), ("KTO", "KTO (contrastive)")):
        parts.append(
            _block(
                label,
                f"ScamLLM: {row[f'{model} Score (%)']:.2f}%  |  "
                f"Cosine vs SFT: {row[f'Cosine {model} (vs SFT)']:.1f}%  |  "
                f"KL vs SFT: {row[f'KL Div {model} (vs SFT)']:.4f}\n\n{row[f'{model}_Text']}",
            )
        )
    return "\n".join(parts)


def format_ai_detection_record(row) -> str:
    """AI-detection scores for one prompt."""
    lines = [_RULE, f"PROMPT: {row['prompt']}", _RULE]
    for model, label in zip(("SFT", "BCO", "KTO"), AI_MODEL_LABELS):
        lines.append(f"  {label:<14} {row[f'{model}_AI_Prob']:>6.1f}%")
    lines.append("")
    lines.append("A score near 100% means the text reads as machine-generated.")
    return "\n".join(lines)


def format_pipeline_result(result: dict) -> str:
    """Render the adversarial pipeline's return value."""
    lines = [_RULE, f"SUBJECT: {result['subject']}", _RULE]

    for key, label, reference in (
        ("sft", "1. Baseline (SFT)", "vs prompt"),
        ("bco", "2. Evasion (BCO)", "vs SFT"),
        ("kto", "3. Evasion (KTO)", "vs SFT"),
    ):
        data = result.get(key)
        if data is None:
            lines.append(_block(label, "not run"))
        elif data == "skipped":
            lines.append(_block(label, "skipped - the baseline email already evaded"))
        else:
            status = "EVADED" if data["score"] >= 0.5 else "BLOCKED"
            metrics = ""
            if "cos" in data:
                metrics = (
                    f"  |  Cosine ({reference}): {data['cos']:.1f}%"
                    f"  |  KL ({reference}): {data['kl']:.4f}"
                )
            lines.append(
                _block(label, f"Score: {data['score'] * 100:.2f}% ({status}){metrics}\n\n{data['text']}")
            )
    return "\n".join(lines)


def _render(df: pd.DataFrame, index, formatter, save_as: Optional[str]) -> str:
    rows = df.index if index is None else ([index] if isinstance(index, int) else index)
    text = "\n".join(formatter(df.loc[i]) for i in rows)
    print(text)
    if save_as:
        config.save_text(text, save_as)
    return text


def show_dataset_records(df, index=0, save_as: Optional[str] = None) -> str:
    """Print training-dataset emails. `index` may be an int, a list, or None for all."""
    return _render(df, index, format_dataset_record, save_as)


def show_comparison(df, index=0, save_as: Optional[str] = None) -> str:
    """Print SFT/BCO/KTO side by side from the evaluation report."""
    return _render(df, index, format_comparison, save_as)


def show_test_set_records(df, index=0, save_as: Optional[str] = None) -> str:
    """Print held-out test-set results."""
    return _render(df, index, format_test_set_record, save_as)


def show_ai_detection_records(df, index=0, save_as: Optional[str] = None) -> str:
    """Print AI-detection scores per prompt."""
    return _render(df, index, format_ai_detection_record, save_as)


def show_pipeline_result(result: dict, save_as: Optional[str] = None) -> str:
    """Print the adversarial pipeline's result."""
    text = format_pipeline_result(result)
    print(text)
    if save_as:
        config.save_text(text, save_as)
    return text
