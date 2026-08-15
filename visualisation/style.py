"""Shared palettes and bar-annotation helpers.

The notebook used three different SFT/BCO/KTO palettes depending on the section;
all three are kept so each figure still looks the way it did.
"""

# Used by the metric grids and the AI-detection plots: grey / blue / green.
PALETTE_GREEN = ["#bdc3c7", "#3498db", "#2ecc71"]

# Used by the AI-detection bar and box plots: darker grey / blue / green.
PALETTE_AI = ["#95a5a6", "#3498db", "#2ecc71"]

# Used by the per-generator and quadrant charts: grey / blue / orange.
PALETTE_ORANGE = ["#95a5a6", "#3498db", "#e67e22"]

PALETTE_BY_MODEL = {"SFT": "#95a5a6", "BCO": "#3498db", "KTO": "#e67e22"}

# High-contrast palette for the per-generator distribution plots,
# three shades per generator in SFT / BCO / KTO order.
GENERATOR_COLOR_MAP = {
    "ChatGPT": ["#0077b6", "#00b4d8", "#90e0ef"],  # deep blue, sky blue, cyan
    "Gemini": ["#d00000", "#ffba08", "#ff006e"],  # dark red, ochre, fuchsia
    "Copilot": ["#004b23", "#40916c", "#b7e4c7"],  # forest, mint, pale green
}

GENERATOR_PALETTE_FLAT = [
    color
    for generator in ("ChatGPT", "Gemini", "Copilot")
    for color in GENERATOR_COLOR_MAP[generator]
]

MODEL_LABELS = ["SFT (Base)", "BCO (Binary)", "KTO (Contrastive)"]

SAFE_COLOR = "#27ae60"
UNSAFE_COLOR = "#c0392b"


def annotate_bars(ax, fmt: str = "{:.1f}%") -> None:
    """Print the value above every patch of an axis (used by the metric grids)."""
    for p in ax.patches:
        ax.annotate(
            fmt.format(p.get_height()),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )


def annotate_scores(ax, rects, offset: float = 1.5, fontsize: int = 11) -> None:
    """Label each bar with its score, green above the 50% threshold and red below."""
    for rect in rects:
        height = rect.get_height()
        if height == 0:
            continue
        color_text = SAFE_COLOR if height >= 50 else UNSAFE_COLOR
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + offset,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            color=color_text,
        )


def annotate_counts(ax, rects, absolute_data, generators, counts) -> None:
    """Label each bar with its percentage and the underlying n/total."""
    for i, rect in enumerate(rects):
        g = generators[i]
        height = rect.get_height()
        abs_val = absolute_data[g]
        tot = counts[g]
        text = f"{height:.1f}%\n({abs_val}/{tot})"
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + 1.5,
            text,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
