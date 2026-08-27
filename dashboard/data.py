"""Everything the dashboard reads, cached, with in-progress runs in mind.

A round is written in two phases — messages land as they are generated, then a
scoring pass fills in `score`, `label` and the metric columns — so a run being
watched live will have messages with no score and rounds with no metrics. Every
loader here is written for that: `scored_only` filters where a number is needed,
and the raw frame keeps the unscored ones so the text is readable while the run
is still going.
"""

import os
import sys
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop import report  # noqa: E402
from loop.store import HOLDOUT_SPLIT, TRAIN_SPLIT, LoopStore  # noqa: E402
from metrics import config  # noqa: E402
from metrics.analysis import load_run  # noqa: E402

CACHE_TTL = 30


@st.cache_resource
def get_store() -> LoopStore:
    """One Mongo client for the session, not one per rerun."""
    return LoopStore()


def refresh() -> None:
    """Drop the cached reads so the next render sees the live database."""
    st.cache_data.clear()


@st.cache_data(ttl=CACHE_TTL)
def list_runs() -> pd.DataFrame:
    """Every run, newest first, with enough to pick one."""
    store = get_store()
    rows = []
    for run in store.runs.find({}, {"_id": 0, "subjects": 0}).sort("run_id", -1):
        run_id = run["run_id"]
        rounds = store.get_rounds(run_id)
        config_ = run.get("config") or {}
        messages = store.messages.count_documents({"run_id": run_id})
        unscored = store.unscored_count(run_id)
        rows.append(
            {
                "run_id": run_id,
                "created_at": pd.to_datetime(run["created_at"], unit="s"),
                "algorithm": config_.get("algorithm"),
                "ref_mode": config_.get("ref_mode"),
                "decoding": config_.get("decoding"),
                "temperature": (config_.get("gen_args") or {}).get("temperature"),
                "n_samples": config_.get("n_samples"),
                "prompts": run.get("prompt_count"),
                "holdout": config_.get("holdout_prompts", 0),
                "rounds": len(rounds),
                "messages": messages,
                "unscored": unscored,
                "running": unscored > 0 or not rounds,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL)
def run_config(run_id: int) -> Dict:
    run = get_store().get_run(run_id) or {}
    return run.get("config") or {}


@st.cache_data(ttl=CACHE_TTL)
def messages_frame(run_id: int) -> pd.DataFrame:
    """One row per message, subject fields joined in, DBRefs flattened.

    Unscored messages are kept — during a live round they are all there is —
    so callers that need a number must filter on `scored`.
    """
    df = load_run(get_store(), run_id)
    if df.empty:
        return df

    df["scored"] = df["score"].notna() if "score" in df else False
    if "split" not in df:
        df["split"] = TRAIN_SPLIT
    if "score" in df:
        df["evaded"] = df["score"] >= config.SAFE_THRESHOLD
    # the decoding dict is unwieldy in a table; keep the one field that varies
    if "decoding" in df:
        df["temperature"] = df["decoding"].apply(
            lambda d: (d or {}).get("temperature") if isinstance(d, dict) else None
        )
    return df


def scored(df: pd.DataFrame) -> pd.DataFrame:
    """The rows a metric can be computed from."""
    if df.empty or "scored" not in df:
        return df
    return df[df["scored"]]


@st.cache_data(ttl=CACHE_TTL)
def trajectory(run_id: int) -> pd.DataFrame:
    """One row per round, as `--report` prints it."""
    return report.trajectory(get_store(), run_id)


@st.cache_data(ttl=CACHE_TTL)
def split_summary(run_id: int) -> pd.DataFrame:
    """Per round and split: evasion, ASR@n and the guardrail metrics.

    Computed from the messages rather than read off the round documents, so it
    is available for a round that has been scored but not yet finished.
    """
    df = scored(messages_frame(run_id))
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (round_index, split), group in df.groupby(["round", "split"]):
        metrics = report.round_metrics(group.to_dict("records"))
        metrics.update({"round": round_index, "split": split})
        rows.append(metrics)

    frame = pd.DataFrame(rows).sort_values(["round", "split"])
    return frame.reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL)
def provenance(run_id: int) -> pd.DataFrame:
    """Checkpoint and dataset verification, per round."""
    return report.provenance(get_store(), run_id)


@st.cache_data(ttl=CACHE_TTL)
def progress(run_id: int) -> Dict:
    """What a live run has produced so far, per round and split."""
    store = get_store()
    config_ = run_config(run_id)
    n_samples = config_.get("n_samples") or 1
    prompts = (store.get_run(run_id) or {}).get("prompt_count") or 0
    expected = prompts * n_samples

    rounds = sorted(store.messages.distinct("round", {"run_id": run_id}))
    per_round = []
    for round_index in rounds:
        query = {"run_id": run_id, "round": round_index}
        per_round.append(
            {
                "round": round_index,
                "messages": store.messages.count_documents(query),
                "unscored": store.messages.count_documents(
                    {**query, "score": {"$exists": False}}
                ),
                "expected": expected,
            }
        )
    return {"expected_per_round": expected, "rounds": per_round}


@st.cache_data(ttl=CACHE_TTL)
def subjects_frame(run_id: int) -> pd.DataFrame:
    """The run's prompt specs, with which split each ended up in."""
    store = get_store()
    refs = store.run_subject_refs(run_id)
    held_from = len(refs) - (run_config(run_id).get("holdout_prompts") or 0)

    rows = []
    for prompt_id, spec in enumerate(store.run_prompts(run_id)):
        rows.append(
            {
                "prompt_id": prompt_id,
                "split": TRAIN_SPLIT if prompt_id < held_from else HOLDOUT_SPLIT,
                **spec,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL)
def messages_for_subject_text(subject: str, run_ids: Optional[List[int]] = None):
    """Every message written for one subject line, across runs.

    Subjects are content-addressed, so two runs over the same prompts.json share
    the document — which is what makes the decoding comparison a join rather
    than a guess.
    """
    store = get_store()
    frames = []
    for document in store.subjects.find({"subject": subject}, {"_id": 1}):
        for message in store.messages_for_subject(document["_id"]):
            if run_ids and message["run_id"] not in run_ids:
                continue
            frames.append(message)
    return pd.DataFrame(frames)
