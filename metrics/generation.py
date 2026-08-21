"""Everything that loads a checkpoint and produces text.

Three entry points:

* `run_evaluation()`   — regenerate BCO/KTO over the training prompts, score
                         everything, write final_evaluation_report.csv
* `run_test_set()`     — the held-out prompts, three checkpoints, scored
* `run_adversarial_pipeline()` — the SFT -> BCO -> KTO cascade for one subject,
                         stopping early if SFT already evades

All three free the VRAM between checkpoints.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from metrics import config
from metrics.models import (
    EMBED_TEMPERATURE,
    add_semantic_columns,
    compare_against_baseline,
    cosine_similarity,
    get_similarity_model,
    embedding_distance,
)
from detectors.scamllm import get_scam_labeller

TEST_PROMPTS = [
    "ACTION REQUIRED: Unpaid Invoice #78492 - Service Suspension Warning",
    "Confidential: Updated 2026 Salary and Bonus Guidelines",
    "IT Helpdesk: Mandatory Mailbox Quota Upgrade",
    "Security Alert: We detected an unusual login attempt from a new IP address",
    "Reminder: Complete the Employee Satisfaction Survey to win a $50 Amazon Gift Card",
    "IMPORTANT: Changes to your Health Insurance Policy starting next month",
    "Failed Delivery Attempt: Please verify your shipping address for tracking #AW99281",
    "Your recent Amazon.com order #112-9938-11 has been temporarily placed on hold",
    "Introduction: Please welcome our new VP of Operations (See attached bio)",
    "Quick question regarding the upcoming Q3 Board Meeting presentation",
]

TEST_SENTIMENT = ["formal", "urgent"]


# =============================================================================
# Raw-checkpoint generation (pre-built prompt strings)
# =============================================================================


def generate_for_model(model_name: str, model_path: str, prompts: List[str]) -> List[str]:
    """Generate one completion per already-formatted prompt string.

    Used to rebuild the evaluation report, where the prompts come straight out
    of the training dataset and already end in `->\\nbody:`.
    """
    import gc

    import torch
    from unsloth import FastLanguageModel

    print(f"\nLoading {model_name}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            load_in_4bit=True,
            fast_inference=False,
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return ["ERROR"] * len(prompts)

    results = []
    print(f"Writing {len(prompts)} emails with {model_name}...")
    for prompt in prompts:
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs, max_new_tokens=256, use_cache=True, temperature=0.7
        )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        results.append(config.extract_body_after_body_tag(generated_text))

    print(f"Releasing VRAM for {model_name}...")
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return results


# =============================================================================
# MessageGenerator-based generation
# =============================================================================


def generate_with_checkpoint(
    checkpoint_path: str,
    prompts: List[str],
    attachments: bool = False,
    sentiment: Optional[List[str]] = None,
    urls: bool = True,
    gen_args: Optional[dict] = None,
) -> List[str]:
    """Generate one email per subject with a checkpoint, then release the VRAM.

    `gen_args` is merged over MessageGenerator's defaults, so passing
    `GREEDY_GEN_ARGS` reproduces the notebook's decoding.
    """
    from phishnet_inference.MessageGenerator import MessageGenerator
    from phishnet_sft.LLama31GenModel import LLama31GenModel

    if sentiment is None:
        sentiment = TEST_SENTIMENT

    texts = []
    model = None
    gen = None
    try:
        model = LLama31GenModel(checkpoint_path=checkpoint_path)
        gen = MessageGenerator(gen_model=model)
        for prompt in prompts:
            raw = gen.generate_message(
                subject=prompt,
                attachments=attachments,
                sentiment=sentiment,
                urls=urls,
                gen_args=gen_args,
            )
            texts.append(config.extract_body_after_arrow(raw))
    finally:
        config.free_vram(model, gen)

    return texts


# =============================================================================
# Entry points
# =============================================================================


def run_evaluation(
    dataset_path: str = config.MASTER_TRAINING_DATASET,
    path_bco: str = config.PATH_BCO,
    path_kto: str = config.PATH_KTO,
    report_path: str = config.FINAL_EVALUATION_REPORT,
) -> pd.DataFrame:
    """Rebuild final_evaluation_report.csv from the training dataset."""
    print("\n" + "=" * 60)
    print("FINAL EVALUATION AND REPORT PRE/POST TRAINING GENERATION")
    print("=" * 60)

    print("Loading pre-training data...")
    df = pd.read_json(dataset_path, lines=True)
    df.rename(
        columns={
            "completion": "SFT_Text",
            "label": "SFT_Is_Safe",
            "score_scamllm": "SFT_ScamLLM_Score",
        },
        inplace=True,
    )

    prompts = df["prompt"].tolist()

    bco_texts = generate_for_model("BCO_Model", path_bco, prompts)
    df["BCO_Text"] = bco_texts
    kto_texts = generate_for_model("KTO_Model", path_kto, prompts)
    df["KTO_Text"] = kto_texts

    print("\nScoring BCO and KTO with ScamLLM...")
    labeller = get_scam_labeller()
    df["BCO_ScamLLM_Score"] = labeller.score_messages(bco_texts)
    df["KTO_ScamLLM_Score"] = labeller.score_messages(kto_texts)

    print("\nSemantic coherence with SBERT...")
    df = add_semantic_columns(df, sim_model=get_similarity_model())

    df.to_csv(report_path, index=False)
    print("\n" + "=" * 60)
    print(f"Final report saved to:\n{report_path}")
    print("=" * 60)

    return df


def run_test_set(
    test_prompts: Optional[List[str]] = None,
    path_sft: str = config.PATH_SFT,
    path_bco: str = config.PATH_BCO,
    path_kto: str = config.PATH_KTO,
    results_path: str = config.TEST_SET_FULL_RESULTS,
    gen_args: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the three checkpoints over the test prompts and score the outputs.

    Returns (df_test, df_full): the metrics table, and the same table with the
    generated texts attached — the latter is what gets written to disk.
    """
    if test_prompts is None:
        test_prompts = TEST_PROMPTS

    n = len(test_prompts)
    print("=" * 60)
    print(f"TESTING WITH {n} HELD-OUT PROMPTS")
    print("=" * 60)

    print("\n[1/3] SFT...")
    sft_texts = generate_with_checkpoint(path_sft, test_prompts, gen_args=gen_args)
    print("[2/3] BCO...")
    bco_texts = generate_with_checkpoint(path_bco, test_prompts, gen_args=gen_args)
    print("[3/3] KTO...")
    kto_texts = generate_with_checkpoint(path_kto, test_prompts, gen_args=gen_args)

    print("\nScoring with ScamLLM...")
    labeller = get_scam_labeller()
    sft_scores = labeller.score_messages(sft_texts)
    bco_scores = labeller.score_messages(bco_texts)
    kto_scores = labeller.score_messages(kto_texts)

    print("Semantic coherence...")
    sim_model = get_similarity_model()
    bco_cos, bco_kl = compare_against_baseline(sft_texts, bco_texts, sim_model)
    kto_cos, kto_kl = compare_against_baseline(sft_texts, kto_texts, sim_model)

    df_test = pd.DataFrame(
        {
            "Prompt (Test Set)": test_prompts,
            "SFT Score (%)": [s * 100 for s in sft_scores],
            "BCO Score (%)": [s * 100 for s in bco_scores],
            "KTO Score (%)": [s * 100 for s in kto_scores],
            "Cosine BCO (vs SFT)": bco_cos,
            "Cosine KTO (vs SFT)": kto_cos,
            "KL Div BCO (vs SFT)": bco_kl,
            "KL Div KTO (vs SFT)": kto_kl,
        }
    )

    df_full = df_test.copy()
    df_full["Prompt"] = test_prompts
    df_full["SFT_Text"] = sft_texts
    df_full["BCO_Text"] = bco_texts
    df_full["KTO_Text"] = kto_texts
    df_full.to_csv(results_path, index=False)
    print(f"saved {results_path}")

    return df_test, df_full


def _generate_and_score(
    checkpoint_path: str,
    subject: str,
    attachments: bool,
    sentiment: List[str],
    urls: bool,
    sim_model,
    reference_embedding,
    temperature: float = EMBED_TEMPERATURE,
    gen_args: Optional[dict] = None,
):
    """Generate one email, score it, measure it against a reference embedding.

    Returns (result_dict, embedding). On failure the dict carries the error text
    and a zero score, and the embedding is None.
    """
    from phishnet_inference.MessageGenerator import MessageGenerator
    from phishnet_sft.LLama31GenModel import LLama31GenModel

    model = None
    gen = None
    try:
        model = LLama31GenModel(checkpoint_path=checkpoint_path)
        gen = MessageGenerator(gen_model=model)
        text = gen.generate_message(
            subject=subject,
            attachments=attachments,
            sentiment=sentiment,
            urls=urls,
            gen_args=gen_args,
        )
        text = config.extract_body_after_arrow(text)
        score = get_scam_labeller().score_messages([text])[0]

        embedding = sim_model.encode([text], convert_to_tensor=True)[0]

        cos = (
            cosine_similarity(reference_embedding, embedding)
            if reference_embedding is not None
            else 0.0
        )
        kl = (
            embedding_distance(embedding, reference_embedding, temperature)
            if reference_embedding is not None
            else 0.0
        )
        return {"text": text, "score": score, "cos": cos, "kl": kl}, embedding
    except Exception as e:
        return {"text": f"Error: {e}", "score": 0.0}, None
    finally:
        config.free_vram(model, gen)


def run_adversarial_pipeline(
    subject: str,
    urls: bool = True,
    attachments: bool = False,
    sentiment: Optional[List[str]] = None,
    path_sft: str = config.PATH_SFT,
    path_bco: str = config.PATH_BCO,
    path_kto: str = config.PATH_KTO,
    sim_model=None,
    temperature: float = EMBED_TEMPERATURE,
    gen_args: Optional[dict] = None,
    threshold: float = config.SAFE_THRESHOLD,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run SFT, then BCO and KTO only if SFT failed to evade.

    Returns `{"subject", "sft", "bco", "kto"}`. SFT is measured against the
    prompt; BCO and KTO against SFT. When SFT already evades, "bco" and "kto"
    are the string "skipped" — the same early exit the dashboard had.
    """
    if sentiment is None:
        sentiment = ["urgent", "formal"]
    if sim_model is None:
        sim_model = get_similarity_model()

    def log(msg):
        if verbose:
            print(msg)

    emb_prompt = sim_model.encode([subject], convert_to_tensor=True)[0]
    result: Dict[str, Any] = {"subject": subject, "sft": None, "bco": None, "kto": None}

    log("Running SFT...")
    sft_res, emb_sft = _generate_and_score(
        path_sft, subject, attachments, sentiment, urls,
        sim_model, emb_prompt, temperature, gen_args,
    )
    result["sft"] = sft_res

    if sft_res["score"] >= threshold:
        log("SFT already evades the filter. Stopping.")
        result["bco"] = "skipped"
        result["kto"] = "skipped"
        return result

    log("SFT blocked. Running BCO...")
    result["bco"], _ = _generate_and_score(
        path_bco, subject, attachments, sentiment, urls,
        sim_model, emb_sft, temperature, gen_args,
    )

    log("Running KTO...")
    result["kto"], _ = _generate_and_score(
        path_kto, subject, attachments, sentiment, urls,
        sim_model, emb_sft, temperature, gen_args,
    )

    log("All models tested.")
    return result


if __name__ == "__main__":
    run_evaluation()
