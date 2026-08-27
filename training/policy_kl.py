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

The estimator, and what it is honestly called
---------------------------------------------
The quantity measured per message is the **log ratio**, per completion token:

    r = [ log policy(y|x) - log reference(y|x) ] / |y|

It needs only the log probabilities of the tokens actually produced, not the
full vocabulary distribution at every position, which would be 128k floats per
token.

`r` is *not* a KL divergence, and this module no longer calls it one. Averaged
over samples drawn from the policy it would be the k1 Monte-Carlo estimator of
KL(policy || reference) — but these samples were drawn with temperature 0.9,
top-k 50 and top-p 0.95, so the sampling distribution is not the policy
distribution and the estimate is biased. Measured on run 1787343134 it came out
negative for 99.7% of messages, which a KL cannot be: the policy assigns lower
probability to its own sampled text than the SFT reference does. That is a real
result — likelihood displacement, corroborated by both reward series falling
during training — but it is a log ratio, not a divergence.

`k3 = exp(-r) - 1 + r` is Schulman's non-negative variant
(http://joschu.net/blog/kl-approx.html). It is kept, but **summarised by its
median, never its mean**: exp(-r) explodes for the strongly negative r this
setup produces, and one message at r = -26 dragged the mean to 3.9e8 while the
median sat at 0.03.

Cost is two forward passes per message with no sampling loop, on one model with
two adapters attached — about 0.15 GiB over the policy alone, the same trick
`reference_model.attach_reference` uses for training.
"""

import math
from typing import Dict, List, Optional, Sequence

from metrics import config
from training.reference_model import REF_ADAPTER_NAME

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
    `logratio_per_token`, `kl_k3_per_token`, `logp_policy`, `logp_reference`
    and `completion_tokens`, in the same order.

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
                        "logratio_per_token": None,
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
                    "logratio_per_token": ratio,
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


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def summarise(records: Sequence[Dict]) -> Dict:
    """Round-level figures, summarised robustly.

    Medians and percentiles rather than means throughout: the per-message log
    ratio has a long negative tail (one message at -26 on the first real run),
    and k3 exponentiates it, so a mean of either says more about the worst
    message than about the round.
    """
    values = [
        r["logratio_per_token"]
        for r in records
        if r.get("logratio_per_token") is not None
    ]
    if not values:
        return {}

    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )

    k3 = sorted(
        r["kl_k3_per_token"] for r in records if r.get("kl_k3_per_token") is not None
    )

    summary = {
        "logratio_per_token": median,
        "logratio_mean": sum(values) / len(values),
        "logratio_p5": _percentile(ordered, 0.05),
        "logratio_p95": _percentile(ordered, 0.95),
        "logratio_min": ordered[0],
        "negative_fraction": sum(v < 0 for v in values) / len(values),
        "kl_messages": len(values),
    }
    if k3:
        k3_middle = len(k3) // 2
        summary["kl_k3_median"] = (
            k3[k3_middle] if len(k3) % 2 else (k3[k3_middle - 1] + k3[k3_middle]) / 2
        )
        summary["kl_k3_max"] = k3[-1]
    return summary
