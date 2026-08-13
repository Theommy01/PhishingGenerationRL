"""The auxiliary models this package owns: the AI-text detector and SBERT.

Both are loaded lazily through `_cached`, so importing this module costs
nothing and a script only pays for the models it actually touches.

ScamLLM is the third auxiliary model but lives outside this package, in
ScamAuxiliaryModel/ScamLabeller/ScamLabel, following the phishnet
AuxiliaryModel/Labeller/Label pattern.

Citations
---------
ScamLLM — "From Chatbots to Phishbots?: Phishing Scam Generation in Commercial
Large Language Models", Roy, Thota, Naragam & Nilizadeh, IEEE S&P 2024.
https://www.computer.org/csdl/proceedings-article/sp/2024/313000a221/1WPcYLpYFHy

AI detector — "Release strategies and the social impacts of language models",
Solaiman et al., arXiv:1908.09203 (2019).
"""

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from metrics import config

_CACHE: Dict[str, Any] = {}


def _cached(key: str, factory: Callable[[], Any]) -> Any:
    """Build `key` once, then hand back the same object."""
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def unload_auxiliary_models() -> None:
    """Evict the cached auxiliary models and give their VRAM back.

    The RL loop scores and measures drift between rounds, so the AI detector
    and SBERT would otherwise stay resident while BCO/KTO tries to load an 8B
    adapter. On an 11 GB card that is the difference between training and
    accelerate silently offloading layers to CPU. They reload lazily on next
    use, so this is only a time/VRAM trade.
    """
    from ScamLabeller import unload_scam_labeller

    _CACHE.clear()
    unload_scam_labeller()
    config.free_vram()


# =============================================================================
# ScamLLM
# =============================================================================


# ScamLLM is not wired up in this module. It belongs to ScamAuxiliaryModel
# (the pipeline) and ScamLabeller (the label-to-score mapping), and callers
# reach it with `ScamLabeller.get_scam_labeller().score_messages(bodies)`.
# Keeping it out of here is deliberate: the inverted-score bug that invalidated
# the published BCO/KTO columns came from this module holding a second,
# divergent copy of that mapping. See PORTING_NOTES.md §1.


def report_safe_percentage(completions, prompts=None) -> List[float]:
    """Print the safe percentage of each completion and return the rewards."""
    from ScamLabeller import get_scam_labeller

    rewards = get_scam_labeller().score_messages(completions)

    lines = ["--- Test model output's safe percentage ---"]
    for text, reward in zip(completions, rewards):
        lines.append(f"\nText:\n{text}")
        lines.append("-" * 40)
        lines.append(f"Safe percentage: {reward:.4f} ({reward * 100:.2f}%)")
    print("\n".join(lines))

    return rewards


# =============================================================================
# AI-generated text detection
# =============================================================================

AI_PROB_COLUMNS = ["SFT_AI_Prob", "BCO_AI_Prob", "KTO_AI_Prob"]


def get_ai_detector():
    """roberta-base-openai-detector."""

    def build():
        import torch
        from transformers import pipeline

        print("Loading RoBERTa AI detector...")
        return pipeline(
            "text-classification",
            model="roberta-base-openai-detector",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512,
        )

    return _cached("ai_detector", build)


def get_ai_prob(text) -> float:
    """Probability (%) that a generated email reads as machine-written.

    Assumes the positive label is the literal string 'Fake'; some revisions of
    the model emit LABEL_0/LABEL_1 instead, which would invert this silently.
    """
    body = config.extract_body_after_body_tag(text)
    if not body:
        return 0.0

    res = get_ai_detector()(body[:512])[0]
    return round(
        (res["score"] * 100) if res["label"] == "Fake" else (100 - res["score"] * 100), 2
    )


def run_ai_detection(
    df: Optional[pd.DataFrame] = None,
    csv_path: str = config.FINAL_EVALUATION_REPORT,
    output_path: str = config.AI_DETECTION_REPORT,
    debug: bool = True,
) -> pd.DataFrame:
    """Score the SFT/BCO/KTO texts with the AI detector and save the report."""
    from tqdm import tqdm

    if df is None:
        df = pd.read_csv(csv_path)

    if debug:
        print("\n--- extracted text check (row 0) ---")
        print(f"raw:     {df['SFT_Text'][0][:50]}...")
        print(f"cleaned: {config.extract_body_after_body_tag(df['SFT_Text'][0])[:50]}...")
        print("------------------------------------\n")

    for model in ("SFT", "BCO", "KTO"):
        print(f"Scoring {model}...")
        df[f"{model}_AI_Prob"] = [get_ai_prob(t) for t in tqdm(df[f"{model}_Text"])]

    df.to_csv(output_path, index=False)
    print(f"saved {output_path}")
    return df


def load_ai_detection_report(csv_path: str = config.AI_DETECTION_REPORT) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def prompt_length_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between prompt length and each model's AI-detection score."""
    df["Prompt_Len"] = df["prompt"].apply(len)
    return df[["Prompt_Len"] + AI_PROB_COLUMNS].corr()


def mean_ai_probabilities(df: pd.DataFrame) -> pd.Series:
    return df[AI_PROB_COLUMNS].mean()


# =============================================================================
# Semantic coherence (SBERT)
# =============================================================================

# Temperature applied to the embeddings before softmax, for the KL divergence.
KL_TEMPERATURE = 0.1
SIMILARITY_MODEL_NAME = "all-MiniLM-L6-v2"


def get_similarity_model(
    model_name: str = SIMILARITY_MODEL_NAME, device: Optional[str] = None
):
    """all-MiniLM-L6-v2 sentence-transformer."""

    def build():
        import torch
        from sentence_transformers import SentenceTransformer

        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return SentenceTransformer(model_name, device=resolved)

    return _cached("sbert", build)


def cosine_similarity(embedding_a, embedding_b) -> float:
    """Cosine similarity between two embeddings, as a percentage.

    Uses torch rather than sentence_transformers.util.cos_sim — the value is
    identical for a single pair, and it keeps this function usable with any
    embedder (including a stub) instead of requiring sentence-transformers.
    """
    import torch.nn.functional as F

    return F.cosine_similarity(embedding_a, embedding_b, dim=-1).item() * 100


def kl_divergence(
    embedding, reference_embedding, temperature: float = KL_TEMPERATURE
) -> float:
    """KL divergence of `embedding` from `reference_embedding`."""
    import torch.nn.functional as F

    log_prob = F.log_softmax(embedding / temperature, dim=-1)
    reference_prob = F.softmax(reference_embedding / temperature, dim=-1)
    return F.kl_div(log_prob, reference_prob, reduction="sum").item()


def clean_prompt(prompt: str) -> str:
    """Strip the structured prompt down to the subject text."""
    return prompt.split("->")[0].replace("subject:", "").strip()


def add_semantic_columns(
    df: pd.DataFrame,
    sim_model=None,
    temperature: float = KL_TEMPERATURE,
) -> pd.DataFrame:
    """Add the cosine/KL columns for SFT, BCO and KTO to an evaluation report.

    Each model is compared against the prompt, and BCO/KTO additionally against
    SFT. Expects `prompt`, `SFT_Text`, `BCO_Text`, `KTO_Text`.
    """
    if sim_model is None:
        sim_model = get_similarity_model()

    cos = {"SFT": [], "BCO": [], "KTO": [], "SFT_vs_BCO": [], "SFT_vs_KTO": []}
    kl = {"SFT": [], "BCO": [], "KTO": [], "SFT_vs_BCO": [], "SFT_vs_KTO": []}

    for _, row in df.iterrows():
        embeddings = sim_model.encode(
            [
                clean_prompt(row["prompt"]),
                row["SFT_Text"],
                row["BCO_Text"],
                row["KTO_Text"],
            ],
            convert_to_tensor=True,
        )
        emb_prompt, emb_sft, emb_bco, emb_kto = embeddings[:4]

        for name, emb in (("SFT", emb_sft), ("BCO", emb_bco), ("KTO", emb_kto)):
            cos[name].append(cosine_similarity(emb_prompt, emb))
            kl[name].append(kl_divergence(emb, emb_prompt, temperature))

        for name, emb in (("SFT_vs_BCO", emb_bco), ("SFT_vs_KTO", emb_kto)):
            cos[name].append(cosine_similarity(emb_sft, emb))
            kl[name].append(kl_divergence(emb, emb_sft, temperature))

    for name in ("SFT", "BCO", "KTO"):
        df[f"{name}_Cosine_Sim"] = cos[name]
        df[f"{name}_KL_Div"] = kl[name]
    for name in ("SFT_vs_BCO", "SFT_vs_KTO"):
        df[f"{name}_Cosine_Sim"] = cos[name]
        df[f"{name}_KL_Div"] = kl[name]

    return df


def compare_against_baseline(
    baseline_texts: List[str],
    candidate_texts: List[str],
    sim_model=None,
    temperature: float = KL_TEMPERATURE,
):
    """Cosine similarity and KL divergence of each candidate against its baseline.

    Returns (cosine percentages, KL divergences), one entry per pair.
    """
    if sim_model is None:
        sim_model = get_similarity_model()

    cosines, divergences = [], []
    for baseline_text, candidate_text in zip(baseline_texts, candidate_texts):
        emb_baseline = sim_model.encode([baseline_text], convert_to_tensor=True)[0]
        emb_candidate = sim_model.encode([candidate_text], convert_to_tensor=True)[0]
        cosines.append(cosine_similarity(emb_baseline, emb_candidate))
        divergences.append(kl_divergence(emb_candidate, emb_baseline, temperature))

    return cosines, divergences
