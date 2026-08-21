"""KL between a round's policy and its reference, on the text actually generated.

This is the RLHF penalty term, measured at evaluation time rather than during
training — the quantity `metrics.models.embedding_distance` is *not*, despite
what its old name suggested.

Two reasons to measure it here as well as inside the trainer:

* TRL only logs a KL for KTO, and only over the batches it trained on. This
  covers BCO too, and it looks at the messages the round actually produced.
* It answers the question the experiment is about — how far the policy has
  moved from the SFT baseline — on the same axis for every `ref_mode`, since
  the anchor here is always the pinned SFT checkpoint regardless of what the
  KL term was anchored to during training.

The estimator
-------------
The messages were sampled from the policy, so the Monte-Carlo estimator over
those samples is the standard one:

    k1 = E_y~policy [ log policy(y|x) - log reference(y|x) ]

which is unbiased for KL(policy || reference) and needs only the log
probabilities of the tokens that were actually produced — not the full vocab
distribution at every position, which would be 128k floats per token. `k3` is
Schulman's low-variance, non-negative variant of the same estimate,

    k3 = exp(-r) - 1 + r,   r = log policy - log reference

and is reported alongside because k1 can come out negative on a small sample.
http://joschu.net/blog/kl-approx.html

Cost is two forward passes per message with no sampling loop, on one model with
two adapters attached — about 0.15 GiB over the policy alone, the same trick
`reference_model.attach_reference` uses for training.
"""

import math
from typing import Dict, List, Optional, Sequence

from metrics import config
from reference_model import REF_ADAPTER_NAME

# Messages are generated with max_new_tokens=256 on a 512-token prompt budget.
MAX_LENGTH = 512


def _sequence_logprob(model, tokenizer, prompt_text: str, body: str, max_length: int):
    """Total and per-token logprob of `body` given `prompt_text`.

    Only the completion's tokens are scored: the prompt is identical between
    policy and reference, so including it would add the same constant to both
    and dilute the per-token average.
    """
    import torch

    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    full_ids = tokenizer(prompt_text + body, return_tensors="pt").input_ids
    full_ids = full_ids[:, :max_length].to(model.device)

    prompt_length = min(prompt_ids.shape[1], full_ids.shape[1])
    completion_length = full_ids.shape[1] - prompt_length
    if completion_length <= 0:
        return None, 0

    with torch.no_grad():
        logits = model(full_ids).logits

    # position i predicts token i+1, so the completion's logits start one earlier
    logits = logits[:, prompt_length - 1 : -1, :]
    targets = full_ids[:, prompt_length:]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    return token_logprobs.sum().item(), completion_length


def measure_policy_kl(
    policy_path: str,
    reference_path: str,
    records: Sequence[Dict],
    max_length: int = MAX_LENGTH,
    progress: bool = True,
) -> List[Dict]:
    """Per-message KL of `policy_path` from `reference_path`.

    `records` need `prompt_text` and `body`. Returns one dict per record with
    `kl_per_token`, `kl_k3_per_token`, `logp_policy`, `logp_reference` and
    `completion_tokens`, in the same order.

    Both adapters sit on one 4-bit base model and are swapped between passes.
    """
    if not records:
        return []

    # imported here, not at module scope: unsloth is a heavy import and a caller
    # that measures nothing should not pay for it
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=policy_path, max_seq_length=max_length, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)

    policy_adapter = model.active_adapter
    if isinstance(policy_adapter, list):
        policy_adapter = policy_adapter[0]
    model.load_adapter(reference_path, adapter_name=REF_ADAPTER_NAME)

    results = []
    try:
        iterator = records
        if progress:
            from tqdm import tqdm

            iterator = tqdm(records, desc="policy KL")

        for record in iterator:
            model.set_adapter(policy_adapter)
            logp_policy, length = _sequence_logprob(
                model, tokenizer, record["prompt_text"], record["body"], max_length
            )

            model.set_adapter(REF_ADAPTER_NAME)
            logp_reference, _ = _sequence_logprob(
                model, tokenizer, record["prompt_text"], record["body"], max_length
            )

            if not length or logp_policy is None or logp_reference is None:
                results.append(
                    {
                        "kl_per_token": None,
                        "kl_k3_per_token": None,
                        "logp_policy": None,
                        "logp_reference": None,
                        "completion_tokens": length,
                    }
                )
                continue

            ratio = (logp_policy - logp_reference) / length
            results.append(
                {
                    "kl_per_token": ratio,
                    "kl_k3_per_token": math.exp(-ratio) - 1 + ratio,
                    "logp_policy": logp_policy / length,
                    "logp_reference": logp_reference / length,
                    "completion_tokens": length,
                }
            )
    finally:
        model.set_adapter(policy_adapter)
        model = None
        tokenizer = None
        config.free_vram()

    return results


def attach_policy_kl(
    records: List[Dict],
    policy_path: str,
    reference_path: Optional[str],
    max_length: int = MAX_LENGTH,
    progress: bool = True,
) -> List[Dict]:
    """Add the KL columns to each record, in place.

    A round whose policy *is* the reference (round 0 against the SFT
    checkpoint) is skipped rather than measured: the KL is identically zero and
    the second forward pass would be wasted.
    """
    if not records or not reference_path:
        return records
    if policy_path == reference_path:
        return records

    for record, measured in zip(
        records, measure_policy_kl(policy_path, reference_path, records, max_length, progress)
    ):
        record.update(measured)

    return records


def summarise(records: Sequence[Dict]) -> Dict:
    """Round-level KL figures: the mean, and the tail that a mean would hide."""
    values = [r["kl_per_token"] for r in records if r.get("kl_per_token") is not None]
    if not values:
        return {}

    ordered = sorted(values)
    k3 = [r["kl_k3_per_token"] for r in records if r.get("kl_k3_per_token") is not None]

    return {
        "kl_per_token": sum(values) / len(values),
        "kl_k3_per_token": (sum(k3) / len(k3)) if k3 else None,
        "kl_p95": ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)],
        "kl_max": ordered[-1],
        "kl_messages": len(values),
    }
