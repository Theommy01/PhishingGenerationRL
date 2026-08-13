"""Build master_training_dataset.jsonl: generate with SFT, label with ScamLLM.

This is the pipeline that was cell 11 of Generate_Message.ipynb. Every prompt in
prompts.json is sent through the SFT checkpoint, the completion is scored by
ScamLLM, and the pair is written out with a boolean label that BCO and KTO
consume as their desirable/undesirable signal.

Decoding defaults to GREEDY_GEN_ARGS, which is what the notebook effectively did
(it never passed do_sample, so transformers defaulted it to False) and therefore
what produced the existing dataset. Pass `gen_args` to change it — e.g.
`SAMPLING_GEN_ARGS`, or any partial dict such as `{"temperature": 0.7}`.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from metrics import config
from metrics.models import scam_evasion_reward

DEFAULT_PROMPTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompts.json"
)


def load_prompts(path: str = DEFAULT_PROMPTS_PATH) -> List[Dict]:
    """Read the prompt specs (subject, sentiment, urls, attachments, ...)."""
    with open(path, "r") as f:
        return json.load(f)


def _extract_completion(raw: str) -> Tuple[str, bool]:
    """Strip the echoed prompt off a generation.

    The model is asked to continue after `->\\n`, and having been trained on
    `...->\\nbody: {body}` it normally emits the `body: ` prefix itself, so the
    marker to split on is `->\\nbody:`. When it does not emit that prefix,
    splitting on the bare `->\\n` at least removes the prompt instead of storing
    it as part of the completion.

    Returns (completion, used_fallback).
    """
    if "->\nbody:" in raw:
        return config.extract_body_after_body_tag(raw), False
    return config.extract_body_after_arrow(raw), True


def generate_messages(
    prompts: List[Dict],
    path_sft: str = config.PATH_SFT,
    gen_args: Optional[dict] = None,
) -> Tuple[List[str], List[str]]:
    """Generate one email per prompt spec with the SFT checkpoint.

    Returns (prompt_strings, completions). The prompt strings are what was
    actually sent to the model, so they can be stored verbatim in the dataset.
    """
    from phishnet_inference.MessageGenerator import GREEDY_GEN_ARGS, MessageGenerator
    from phishnet_inference.prompt_generation.generate_prompt import generate_prompt
    from phishnet_sft.LLama31GenModel import LLama31GenModel

    if gen_args is None:
        gen_args = GREEDY_GEN_ARGS

    prompt_structures: List[str] = []
    completions: List[str] = []
    fallbacks = 0

    model = None
    generator = None
    try:
        model = LLama31GenModel(checkpoint_path=path_sft)
        generator = MessageGenerator(gen_model=model)

        print(f"Generating {len(prompts)} emails...")
        for i, p in enumerate(prompts, 1):
            kwargs = dict(
                subject=p["subject"],
                urls=p["urls"],
                attachments=p["attachments"],
                sentiment=p["sentiment"],
            )

            # store the prompt exactly as MessageGenerator builds it
            prompt_structures.append(generate_prompt(**kwargs) + "\n->\n")

            raw = generator.generate_message(gen_args=gen_args, **kwargs)
            completion, used_fallback = _extract_completion(raw)
            fallbacks += used_fallback
            completions.append(completion)

            print(f"  generated {i} of {len(prompts)}")
    finally:
        config.free_vram(model, generator)

    if fallbacks:
        print(
            f"\nWarning: {fallbacks}/{len(prompts)} generations did not contain "
            "'->\\nbody:'; the prompt was stripped on the bare arrow instead. "
            "Spot-check those completions."
        )

    return prompt_structures, completions


def build_dataset(
    prompts: List[Dict],
    prompt_structures: List[str],
    completions: List[str],
    threshold: float = config.SAFE_THRESHOLD,
) -> pd.DataFrame:
    """Score the completions with ScamLLM and assemble the training records.

    `label` is True when ScamLLM considers the message safe, i.e. the filter was
    evaded — that is the desirable class for BCO/KTO.
    """
    print("\nScoring with ScamLLM...")
    scores = scam_evasion_reward(
        prompts=[""] * len(completions), completions=completions
    )

    records = []
    for idx, score in enumerate(scores):
        records.append(
            {
                "prompt": prompt_structures[idx].strip(),
                "completion": completions[idx],
                "label": bool(score >= threshold),
                "score_scamllm": score,
                "category": prompts[idx]["category"],
                "generator": prompts[idx]["generator"],
            }
        )

    return pd.DataFrame(records)


def save_dataset(df: pd.DataFrame, save_path: str = config.MASTER_TRAINING_DATASET) -> str:
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_json(save_path, orient="records", lines=True)

    safe = int(df["label"].sum())
    print(f"\nDataset created: {len(df)} records -> {save_path}")
    print(f"  safe (evaded):  {safe} ({safe / len(df) * 100:.1f}%)")
    print(f"  malicious:      {len(df) - safe} ({(len(df) - safe) / len(df) * 100:.1f}%)")
    return save_path


def main(
    prompts_path: str = DEFAULT_PROMPTS_PATH,
    path_sft: str = config.PATH_SFT,
    save_path: str = config.MASTER_TRAINING_DATASET,
    threshold: float = config.SAFE_THRESHOLD,
    gen_args: Optional[dict] = None,
) -> pd.DataFrame:
    prompts = load_prompts(prompts_path)
    prompt_structures, completions = generate_messages(prompts, path_sft, gen_args)
    df = build_dataset(prompts, prompt_structures, completions, threshold)
    save_dataset(df, save_path)
    return df


if __name__ == "__main__":
    main()
