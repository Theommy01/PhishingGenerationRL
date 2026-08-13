"""MongoDB persistence for the fine-tuning loop.

Three collections in one database:

    runs      one document per loop invocation, holding the config every round
              must share (prompts, n_samples, algorithm, ref_mode, threshold)
    rounds    one per round: which checkpoint it trained from and produced,
              the label balance it trained on, and its metrics
    messages  one per generated message, tagged with run/round/prompt/sample

Messages live in a single collection rather than one per checkpoint, because the
loop needs both slices cheaply: the cumulative training pool (`round <= N`) and
the round-scoped evaluation set (`round == N`).
"""

import hashlib
import json
import time
from typing import Any, Dict, Iterable, List, Optional

from pymongo import ASCENDING, MongoClient

DEFAULT_URI = "mongodb://localhost:27017/"
DEFAULT_DB = "phishnet_rl"


def prompts_fingerprint(prompts: List[Dict]) -> str:
    """Stable hash of the prompt specs.

    Rounds are only comparable when they were generated from identical prompts,
    so the fingerprint is stored on the run and checked on every append.
    """
    canonical = json.dumps(prompts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class LoopStore:
    """Thin wrapper over the three collections. No ODM, no schema migrations."""

    def __init__(self, db_name: str = DEFAULT_DB, uri: str = DEFAULT_URI, client=None):
        self.client = client or MongoClient(uri)
        self.db = self.client[db_name]
        self.runs = self.db["runs"]
        self.rounds = self.db["rounds"]
        self.messages = self.db["messages"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.runs.create_index([("run_id", ASCENDING)], unique=True)
        self.rounds.create_index(
            [("run_id", ASCENDING), ("round", ASCENDING)], unique=True
        )
        self.messages.create_index([("run_id", ASCENDING), ("round", ASCENDING)])
        self.messages.create_index([("run_id", ASCENDING), ("prompt_id", ASCENDING)])

    # -- runs ---------------------------------------------------------------

    def create_run(self, prompts: List[Dict], config: Dict[str, Any]) -> int:
        """Start a run. Returns its id (a unix timestamp, as phishnet uses)."""
        run_id = int(time.time())
        self.runs.insert_one(
            {
                "run_id": run_id,
                "created_at": run_id,
                "prompts_hash": prompts_fingerprint(prompts),
                "prompt_count": len(prompts),
                "config": config,
            }
        )
        return run_id

    def get_run(self, run_id: int) -> Optional[Dict]:
        return self.runs.find_one({"run_id": run_id}, {"_id": 0})

    def check_prompts(self, run_id: int, prompts: List[Dict]) -> None:
        """Raise if the prompts differ from the ones the run started with."""
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"no such run: {run_id}")
        actual = prompts_fingerprint(prompts)
        if actual != run["prompts_hash"]:
            raise ValueError(
                f"prompts changed since run {run_id} started "
                f"({run['prompts_hash']} -> {actual}); rounds would not be comparable"
            )

    # -- messages -----------------------------------------------------------

    def add_messages(self, run_id: int, round_index: int, records: Iterable[Dict]) -> int:
        """Store this round's generated + scored messages. Returns the count."""
        docs = []
        for record in records:
            doc = dict(record)
            doc["run_id"] = run_id
            doc["round"] = round_index
            docs.append(doc)

        if not docs:
            return 0
        self.messages.insert_many(docs)
        return len(docs)

    def get_messages(
        self,
        run_id: int,
        round_index: Optional[int] = None,
        max_round: Optional[int] = None,
    ) -> List[Dict]:
        """Messages for a run.

        `round_index` selects one round (evaluation); `max_round` selects every
        round up to and including it (the cumulative training pool). Passing
        neither returns everything.
        """
        query: Dict[str, Any] = {"run_id": run_id}
        if round_index is not None:
            query["round"] = round_index
        elif max_round is not None:
            query["round"] = {"$lte": max_round}
        return list(self.messages.find(query, {"_id": 0}).sort("round", ASCENDING))

    def training_pool(self, run_id: int, max_round: int) -> List[Dict]:
        """The cumulative KTO/BCO dataset: every message from rounds 0..max_round.

        Labels stay valid across rounds because ScamLLM is frozen, so nothing
        needs re-scoring.
        """
        return [
            {"prompt": m["prompt_text"], "completion": m["body"], "label": m["label"]}
            for m in self.get_messages(run_id, max_round=max_round)
        ]

    def label_counts(
        self, run_id: int, max_round: Optional[int] = None, round_index: Optional[int] = None
    ) -> Dict[str, int]:
        """Desirable/undesirable counts, for the KTO class weights."""
        messages = self.get_messages(run_id, round_index=round_index, max_round=max_round)
        desirable = sum(1 for m in messages if m["label"])
        return {
            "desirable": desirable,
            "undesirable": len(messages) - desirable,
            "total": len(messages),
        }

    def duplicate_count(self, run_id: int, max_round: Optional[int] = None) -> int:
        """Exact-duplicate bodies in the pool.

        Reusing the same prompts every round means a converging model starts
        repeating itself, which over-weights whatever it settles into.
        """
        bodies = [m["body"] for m in self.get_messages(run_id, max_round=max_round)]
        return len(bodies) - len(set(bodies))

    # -- rounds -------------------------------------------------------------

    def record_round(self, run_id: int, round_index: int, **fields) -> None:
        """Create or update a round document."""
        self.rounds.update_one(
            {"run_id": run_id, "round": round_index},
            {"$set": {**fields, "run_id": run_id, "round": round_index}},
            upsert=True,
        )

    def get_round(self, run_id: int, round_index: int) -> Optional[Dict]:
        return self.rounds.find_one(
            {"run_id": run_id, "round": round_index}, {"_id": 0}
        )

    def get_rounds(self, run_id: int) -> List[Dict]:
        return list(
            self.rounds.find({"run_id": run_id}, {"_id": 0}).sort("round", ASCENDING)
        )

    def latest_round(self, run_id: int) -> Optional[Dict]:
        rounds = self.get_rounds(run_id)
        return rounds[-1] if rounds else None

    # -- housekeeping -------------------------------------------------------

    def drop_run(self, run_id: int) -> None:
        """Delete a run and everything belonging to it."""
        self.messages.delete_many({"run_id": run_id})
        self.rounds.delete_many({"run_id": run_id})
        self.runs.delete_one({"run_id": run_id})

    def close(self) -> None:
        self.client.close()
