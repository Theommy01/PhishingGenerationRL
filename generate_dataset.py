"""Build master_training_dataset.jsonl: generate with SFT, label with ScamLLM.

This is the pipeline that was cell 11 of Generate_Message.ipynb. Every prompt in
prompts.json is sent through the SFT checkpoint, the completion is scored by
ScamLLM, and the pair is written out with a boolean label that BCO and KTO
consume as their desirable/undesirable signal.

Decoding is chosen by preset — `default` (temperature 0.9), `sampling`
(temperature 0.7) or `greedy` — and `gen_args` overrides individual keys on top.
`resolve_gen_args` returns the complete resolved spec, which is what runs and
what gets stored on every message, so a run can always say what produced it.
The existing master_training_dataset.jsonl predates this and was generated
greedily.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from metrics import config
from detectors.scamllm import get_scam_labeller

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


DECODING_PRESETS = ("default", "sampling", "greedy")


def decoding_preset(name: str) -> dict:
    """One of MessageGenerator's named presets, by name."""
    from phishnet_inference.MessageGenerator import (
        DEFAULT_GEN_ARGS,
        GREEDY_GEN_ARGS,
        SAMPLING_GEN_ARGS,
    )

    presets = {
        "default": DEFAULT_GEN_ARGS,  # temperature 0.9, top_p 0.95, top_k 50
        "sampling": SAMPLING_GEN_ARGS,  # temperature 0.7, and a max_new_tokens cap
        "greedy": GREEDY_GEN_ARGS,  # deterministic; only valid at n_samples=1
    }
    if name not in presets:
        raise ValueError(f"unknown decoding preset: {name!r} {DECODING_PRESETS}")
    return dict(presets[name])


def resolve_gen_args(
    gen_args: Optional[dict], n_samples: int, preset: Optional[str] = "default"
) -> dict:
    """Produce the *complete* decoding spec, so what is recorded is what runs.

    This used to hand MessageGenerator a partial dict — typically just
    `max_new_tokens` and `do_sample` — which it then merged over its own
    `DEFAULT_GEN_ARGS`. The temperature, top_p and top_k actually in force came
    from that hidden merge and were recorded nowhere, so a stored run could not
    say what sampling produced it. Merging over the named preset here makes the
    result the whole truth, and it is what gets stored on every message.

    Greedy decoding is deterministic, so asking for several samples of the same
    prompt under it would return n identical messages. That is always a mistake,
    so it is refused rather than silently producing duplicates.
    """
    # preset=None means `gen_args` is already a complete spec — the loop
    # resolves once when the run is configured and passes the result down, so
    # re-merging a second base over it here would smuggle in stray keys.
    resolved = decoding_preset(preset) if preset else {}
    resolved.update(gen_args or {})

    # `max_length` caps prompt+completion, `max_new_tokens` caps the completion
    # alone; transformers takes the latter and warns. Keeping both would record
    # a number that has no effect.
    if "max_new_tokens" in resolved:
        resolved.pop("max_length", None)

    if n_samples > 1 and not resolved.get("do_sample", True):
        raise ValueError(
            f"n_samples={n_samples} requires sampling, but the decoding args have "
            "do_sample=False; greedy decoding would return identical messages. "
            "Use --decoding default or sampling."
        )
    return resolved


def generate_messages(
    prompts: List[Dict],
    path_sft: str = config.PATH_SFT,
    gen_args: Optional[dict] = None,
    n_samples: int = 1,
    preset: Optional[str] = None,
) -> List[Dict]:
    """Generate `n_samples` emails per prompt spec with one checkpoint.

    Returns one record per generated message, carrying enough context to store
    it and to regenerate the same prompt in a later round:

        {prompt_id, sample_idx, prompt_text, body, category, generator}

    `prompt_text` is what was actually sent to the model, so it can be stored
    verbatim and fed straight back to KTO/BCO. `prompt_id` is the index into
    `prompts`, which is also the index into a run's subject list — LoopStore
    turns it into the message's subject DBRef, and drops the `category` and
    `generator` copies, which are there for the standalone jsonl path only.

    Each record also carries `decoding`: the complete, resolved generation
    arguments this message was produced with. It is constant within a call, but
    a message is the unit that gets compared, filtered and exported, and two
    runs differing only in temperature are exactly what a decoding sweep looks
    like — so it belongs on the message, not only on the round.
    """
    from phishnet_inference.MessageGenerator import MessageGenerator
    from phishnet_inference.prompt_generation.generate_prompt import generate_prompt
    from phishnet_sft.LLama31GenModel import LLama31GenModel

    resolved_args = resolve_gen_args(gen_args, n_samples, preset)
    print(f"Decoding: {resolved_args}")

    records: List[Dict] = []
    fallbacks = 0
    total = len(prompts) * n_samples

    model = None
    generator = None
    try:
        model = LLama31GenModel(checkpoint_path=path_sft)
        # the resolved args become the generator's own defaults, so nothing is
        # merged over them behind our back and `resolved_args` is exactly what
        # every call runs with
        generator = MessageGenerator(gen_model=model, gen_args=resolved_args)

        print(f"Generating {total} emails ({len(prompts)} prompts x {n_samples})...")
        for prompt_id, p in enumerate(prompts):
            kwargs = dict(
                subject=p["subject"],
                urls=p["urls"],
                attachments=p["attachments"],
                sentiment=p["sentiment"],
            )
            prompt_text = generate_prompt(**kwargs) + "\n->\n"

            for sample_idx in range(n_samples):
                raw = generator.generate_message(**kwargs)
                body, used_fallback = _extract_completion(raw)
                fallbacks += used_fallback

                records.append(
                    {
                        "prompt_id": prompt_id,
                        "sample_idx": sample_idx,
                        "prompt_text": prompt_text.strip(),
                        "body": body,
                        "category": p["category"],
                        "generator": p["generator"],
                        "decoding": dict(resolved_args),
                    }
                )
                print(f"  generated {len(records)} of {total}")
    finally:
        # rebinding here is what actually releases the adapter: passing the
        # objects to free_vram only deletes ITS parameters, leaving these two
        # references alive, so the memory would not come back until this frame
        # died — by which point empty_cache() had already run.
        model = None
        generator = None
        config.free_vram()

    if fallbacks:
        print(
            f"\nWarning: {fallbacks}/{total} generations did not contain "
            "'->\\nbody:'; the prompt was stripped on the bare arrow instead. "
            "Spot-check those completions."
        )

    return records


def score_messages(
    records: List[Dict], threshold: float = config.SAFE_THRESHOLD
) -> List[Dict]:
    """Score each record's body with ScamLLM and attach `score` and `label`.

    `label` is True when ScamLLM considers the message safe, i.e. the filter was
    evaded — the desirable class for KTO/BCO. Mutates and returns the records so
    the loop can hand them straight to the store.
    """
    print(f"\nScoring {len(records)} messages with ScamLLM...")
    scores = get_scam_labeller().score_messages([r["body"] for r in records])

    for record, score in zip(records, scores):
        record["score"] = score
        record["label"] = bool(score >= threshold)

    return records


def build_dataset(
    records: List[Dict], threshold: float = config.SAFE_THRESHOLD
) -> pd.DataFrame:
    """Score the records if needed and lay them out in the training schema.

    Produces the columns BCO/KTO and the evaluation report expect:
    prompt, completion, label, score_scamllm, category, generator.
    """
    if any("score" not in r for r in records):
        records = score_messages(records, threshold)

    return pd.DataFrame(
        [
            {
                "prompt": r["prompt_text"],
                "completion": r["body"],
                "label": r["label"],
                "score_scamllm": r["score"],
                "category": r["category"],
                "generator": r["generator"],
            }
            for r in records
        ]
    )


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
    n_samples: int = 1,
) -> pd.DataFrame:
    prompts = load_prompts(prompts_path)
    records = generate_messages(prompts, path_sft, gen_args, n_samples)
    df = build_dataset(records, threshold)
    save_dataset(df, save_path)
    return df


if __name__ == "__main__":
    main()
