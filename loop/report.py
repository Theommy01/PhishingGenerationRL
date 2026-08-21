"""Per-round metrics and the round-over-round comparison.

Metrics are always computed on a single round's *fresh* messages, never on the
cumulative pool — history would dilute the signal and make each round's
improvement look smaller than it is. The cumulative pool is for training only.
"""

import re
from typing import Dict, List, Optional

import pandas as pd

from metrics import config
from metrics.models import (
    EMBED_TEMPERATURE,
    clean_prompt,
    cosine_similarity,
    embedding_distance,
    get_similarity_model,
)


# =============================================================================
# Semantic drift
#
# Two references, because they answer different questions:
#
#   *_prompt    the message against its own prompt — is it still on topic?
#               Comparable across every round, including round 0.
#   *_baseline  the message against what round 0 produced for the same prompt —
#               how far the policy has moved. This is the trace that separates
#               ref_mode="sft" (anchored) from ref_mode="previous" (compounding).
# =============================================================================


def baseline_embeddings(messages: List[Dict], sim_model=None) -> Dict[int, "object"]:
    """Round-0 embeddings for each prompt, kept as a stack rather than averaged.

    With n samples per prompt there is no single baseline message, so a
    round-N message is compared against every round-0 sample for its prompt and
    the resulting metrics are averaged (see `attach_drift`).
    """
    if sim_model is None:
        sim_model = get_similarity_model()

    by_prompt: Dict[int, List[str]] = {}
    for message in messages:
        by_prompt.setdefault(message["prompt_id"], []).append(message["body"])

    return {
        prompt_id: sim_model.encode(bodies, convert_to_tensor=True)
        for prompt_id, bodies in by_prompt.items()
    }


def _drift_against(body_emb, reference_embs, temperature: float, reduction: str):
    """Cosine and embedding distance of one message against round-0 samples.

    reduction="pairwise" computes the metric against each baseline sample and
    averages the metrics — the aggregation happens after the metric, so a
    non-linear measure stays interpretable.

    reduction="centroid" averages the embeddings first and compares once. It is
    cheaper but not equivalent: mean(cos(x, y_j)) != cos(x, mean(y_j)).
    """
    import torch

    references = list(reference_embs)
    if not references:
        return None, None

    if reduction == "centroid":
        centroid = torch.stack(references).mean(dim=0)
        return (
            cosine_similarity(centroid, body_emb),
            embedding_distance(body_emb, centroid, temperature),
        )

    if reduction != "pairwise":
        raise ValueError(f"unknown reduction: {reduction!r} (pairwise | centroid)")

    cosines = [cosine_similarity(ref, body_emb) for ref in references]
    divergences = [embedding_distance(body_emb, ref, temperature) for ref in references]
    return sum(cosines) / len(cosines), sum(divergences) / len(divergences)


def attach_drift(
    records: List[Dict],
    sim_model=None,
    baselines: Optional[Dict[int, "object"]] = None,
    temperature: float = EMBED_TEMPERATURE,
    reduction: str = "pairwise",
) -> List[Dict]:
    """Add cosine and embedding distance vs the prompt and vs round 0.

    Every metric is per message; round-level figures are the mean of these.
    Mutates and returns the records. `baselines` is omitted for round 0, whose
    messages *are* the baseline.
    """
    if not records:
        return records
    if sim_model is None:
        sim_model = get_similarity_model()

    bodies = sim_model.encode([r["body"] for r in records], convert_to_tensor=True)
    prompts = sim_model.encode(
        [clean_prompt(r["prompt_text"]) for r in records], convert_to_tensor=True
    )

    for record, body_emb, prompt_emb in zip(records, bodies, prompts):
        record["cos_prompt"] = cosine_similarity(prompt_emb, body_emb)
        record["embed_dist_prompt"] = embedding_distance(body_emb, prompt_emb, temperature)

        if baselines is not None:
            references = baselines.get(record["prompt_id"])
            if references is not None:
                cos, distance = _drift_against(body_emb, references, temperature, reduction)
                if cos is not None:
                    record["cos_baseline"] = cos
                    record["embed_dist_baseline"] = distance

    return records


# =============================================================================
# Instruction following
#
# The prompt asks for four things: a subject to write about, whether to include
# a URL, whether to reference an attachment, and a sentiment. Three of those are
# checkable against the message itself, which is what this section does — the
# question being whether the policy trades instruction-following away for
# evasion as the rounds go on.
#
# `cos_prompt` does not answer it: it compares the body against the whole
# rendered prompt, including the `urls:`/`attachments:`/`sentiment:` lines,
# which are near-identical across all 150 prompts and dilute the signal.
# `cos_subject` compares against the subject line alone.
# =============================================================================

# The corpus is entity-anonymised, so the model was trained to emit placeholder
# tokens rather than real values: <URL>, <ATTACHMENT>, <ORG>, <PER>, <EMAIL>,
# <DATE>, <LOC>, <PHONE>. Following the `urls: True` instruction therefore means
# emitting <URL>, not writing out a link — matching real URLs would score every
# compliant message as a failure.
URL_PLACEHOLDER = re.compile(r"<URL[^>]*>", re.IGNORECASE)
ATTACHMENT_PLACEHOLDER = re.compile(r"<ATTACH(?:MENT)?[^>]*>", re.IGNORECASE)

# A literal link is *off*-distribution here: the training data contains none, so
# one appearing means the policy has drifted away from the corpus's conventions.
# Tracked separately from compliance, as a format signal rather than a failure.
LITERAL_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)


def has_url(text: str) -> bool:
    """Whether the body carries a <URL> placeholder."""
    return bool(URL_PLACEHOLDER.search(text or ""))


def mentions_attachment(text: str) -> bool:
    """Whether the body carries an <ATTACHMENT> placeholder."""
    return bool(ATTACHMENT_PLACEHOLDER.search(text or ""))


def has_literal_url(text: str) -> bool:
    return bool(LITERAL_URL.search(text or ""))


def attach_compliance(
    records: List[Dict], prompts: List[Dict], sim_model=None
) -> List[Dict]:
    """Add per-message instruction-following columns.

        url_requested / url_present / url_ok
        attachment_requested / attachment_present / attachment_ok
        url_literal     a real link instead of the <URL> placeholder
        cos_subject     body against the subject line alone

    Presence means the *placeholder* the corpus uses, `<URL>` / `<ATTACHMENT>`;
    see the patterns above. `*_ok` is two-sided: asking for no URL and getting
    one is as much a failure to follow the prompt as the reverse. Mutates and
    returns the records.
    """
    if not records:
        return records

    # The placeholder checks are pure string work. `sim_model=None` means skip
    # cos_subject and do only those, so --no-drift can mean "no SBERT" without
    # also meaning "no instruction-following check".
    if sim_model is None:
        subject_embs = body_embs = [None] * len(records)
    else:
        subjects = [prompts[record["prompt_id"]]["subject"] for record in records]
        subject_embs = sim_model.encode(subjects, convert_to_tensor=True)
        body_embs = sim_model.encode([r["body"] for r in records], convert_to_tensor=True)

    for record, subject_emb, body_emb in zip(records, subject_embs, body_embs):
        spec = prompts[record["prompt_id"]]

        url_present = has_url(record["body"])
        attachment_present = mentions_attachment(record["body"])

        record["url_requested"] = bool(spec["urls"])
        record["url_present"] = url_present
        record["url_ok"] = url_present == bool(spec["urls"])

        record["attachment_requested"] = bool(spec["attachments"])
        record["attachment_present"] = attachment_present
        record["attachment_ok"] = attachment_present == bool(spec["attachments"])
        record["url_literal"] = has_literal_url(record["body"])

        if subject_emb is not None:
            record["cos_subject"] = cosine_similarity(subject_emb, body_emb)

    return records


def round_metrics(messages: List[Dict], threshold: float = config.SAFE_THRESHOLD) -> Dict:
    """Evasion metrics for one round.

    evasion_rate is per message; asr_at_n is per prompt — the fraction of
    prompts where at least one of the n samples evaded. With n=1 the two
    collapse to the same number, so nothing is lost by reporting both.
    """
    if not messages:
        return {}

    scores = [m["score"] for m in messages]
    evaded = [m["score"] >= threshold for m in messages]

    by_prompt: Dict[int, List[bool]] = {}
    for message, hit in zip(messages, evaded):
        by_prompt.setdefault(message["prompt_id"], []).append(hit)

    bodies = [m["body"] for m in messages]

    def mean_of(key):
        values = [m[key] for m in messages if m.get(key) is not None]
        return sum(values) / len(values) if values else None

    def rate_of(key):
        """Percentage of messages where a boolean check passed."""
        values = [m[key] for m in messages if m.get(key) is not None]
        return sum(values) / len(values) * 100 if values else None

    def percentile_of(key, fraction):
        """The tail a mean hides — one message diverging hard is the interesting case."""
        values = sorted(m[key] for m in messages if m.get(key) is not None)
        if not values:
            return None
        return values[min(int(fraction * len(values)), len(values) - 1)]

    return {
        "messages": len(messages),
        "prompts": len(by_prompt),
        "mean_score": sum(scores) / len(scores) * 100,
        "evasion_rate": sum(evaded) / len(evaded) * 100,
        "asr_at_n": sum(any(v) for v in by_prompt.values()) / len(by_prompt) * 100,
        "duplicates": len(bodies) - len(set(bodies)),
        "cos_prompt": mean_of("cos_prompt"),
        "embed_dist_prompt": mean_of("embed_dist_prompt"),
        "cos_baseline": mean_of("cos_baseline"),
        "embed_dist_baseline": mean_of("embed_dist_baseline"),
        # instruction following
        "cos_subject": mean_of("cos_subject"),
        "url_compliance": rate_of("url_ok"),
        "attachment_compliance": rate_of("attachment_ok"),
        # KL from the SFT baseline, on this round's own text
        "kl_per_token": mean_of("kl_per_token"),
        "kl_k3_per_token": mean_of("kl_k3_per_token"),
        "kl_p95": percentile_of("kl_per_token", 0.95),
    }


def trajectory(store, run_id: int) -> pd.DataFrame:
    """One row per round: what it trained on and how it scored."""
    rows = []
    for record in store.get_rounds(run_id):
        metrics = record.get("metrics") or {}
        counts = record.get("label_counts") or {}
        pool = record.get("pool_counts") or {}
        rows.append(
            {
                "round": record["round"],
                "base": _short(record.get("base_checkpoint")),
                "checkpoint": _short(record.get("checkpoint_path")),
                "ref_mode": record.get("ref_mode"),
                # what this round trained on (cumulative pool)
                "pool": record.get("dataset_size"),
                # the content hash of that exact pool — the anchor for
                # `store.verify_dataset`, and the only part of a round that is
                # reproducible, generation being stochastic
                "pool_hash": record.get("dataset_hash"),
                "pool_desirable": pool.get("desirable"),
                "pool_undesirable": pool.get("undesirable"),
                # what this round then produced (its own messages)
                "out_desirable": counts.get("desirable"),
                "out_undesirable": counts.get("undesirable"),
                "mean_score": metrics.get("mean_score"),
                "evasion_rate": metrics.get("evasion_rate"),
                "asr_at_n": metrics.get("asr_at_n"),
                # has it moved? (KL from the SFT baseline on this round's text)
                "kl_per_token": metrics.get("kl_per_token"),
                "kl_p95": metrics.get("kl_p95"),
                "train_kl": (record.get("training") or {}).get("kl_mean"),
                # does it still follow the prompt?
                "cos_subject": metrics.get("cos_subject"),
                "url_ok": metrics.get("url_compliance"),
                "attach_ok": metrics.get("attachment_compliance"),
                # semantic drift
                "cos_prompt": metrics.get("cos_prompt"),
                "cos_baseline": metrics.get("cos_baseline"),
                "embed_dist_baseline": metrics.get("embed_dist_baseline"),
                "duplicates": metrics.get("duplicates"),
            }
        )
    return pd.DataFrame(rows)


def _short(path: Optional[str]) -> Optional[str]:
    return path.rsplit("/", 1)[-1] if path else path


def compare(store, run_id: int, first: int = 0, last: Optional[int] = None) -> pd.DataFrame:
    """Round-over-round deltas for the headline metrics."""
    df = trajectory(store, run_id).set_index("round")
    if last is None:
        last = int(df.index.max())

    metrics = ["mean_score", "evasion_rate", "asr_at_n", "cos_prompt", "cos_baseline"]
    return pd.DataFrame(
        {
            f"round {first}": [df.loc[first, m] for m in metrics],
            f"round {last}": [df.loc[last, m] for m in metrics],
            "delta": [
                (df.loc[last, m] - df.loc[first, m])
                if pd.notna(df.loc[last, m]) and pd.notna(df.loc[first, m])
                else None
                for m in metrics
            ],
        },
        index=metrics,
    ).round(2)


def print_trajectory(store, run_id: int, save_as: Optional[str] = None) -> pd.DataFrame:
    df = trajectory(store, run_id)
    print(df.to_string(index=False))
    if save_as:
        config.save_table(df, save_as, index=False)
    return df


# =============================================================================
# Provenance
# =============================================================================


def provenance(store, run_id: int) -> pd.DataFrame:
    """Per round: what generated it, and whether that still checks out.

    Two independent questions, answered from what the messages themselves
    recorded rather than from the round document:

      checkpoint  is the adapter now at that path the one that wrote these
                  messages? (`ok` false means moved, deleted or overwritten)
      dataset     does the slice this round trained on still hash the same?
    """
    rows = []
    for record in store.get_rounds(run_id):
        round_index = record["round"]
        messages = store.get_messages(
            run_id, round_index=round_index, with_subject=False
        )
        stamped = {m.get("checkpoint_hash") for m in messages if m.get("checkpoint_hash")}

        generation = record.get("generation") or {}
        training = record.get("training") or {}

        row = {
            "round": round_index,
            "checkpoint": _short(record.get("checkpoint_path")),
            "ckpt_hash": record.get("checkpoint_hash"),
            # what this round actually generated and trained with, which a
            # resumed run can legitimately change from the run's config
            "n_samples": generation.get("n_samples"),
            "max_new_tokens": (generation.get("gen_args") or {}).get("max_new_tokens"),
            "seed": training.get("seed"),
            "messages": len(messages),
            # every message of a round should carry the same stamp; more than
            # one means the round was generated by more than one adapter
            "ckpt_stamps": len(stamped),
            "ckpt_ok": None,
            "pool": record.get("dataset_size"),
            "pool_hash": record.get("dataset_hash"),
            "pool_ok": None,
        }

        if record.get("checkpoint") is not None:
            check = store.verify_checkpoint(record["checkpoint"])
            row["ckpt_ok"] = check["ok"]
            row["ckpt_note"] = (
                "" if check["ok"] else ("missing" if not check["present"] else "differs")
            )
        if record.get("dataset") is not None:
            row["pool_ok"] = store.verify_dataset(record["dataset"])["ok"]

        rows.append(row)

    return pd.DataFrame(rows)


def print_provenance(store, run_id: int, save_as: Optional[str] = None) -> pd.DataFrame:
    df = provenance(store, run_id)
    print(df.to_string(index=False))
    if save_as:
        config.save_table(df, save_as, index=False)
    return df
