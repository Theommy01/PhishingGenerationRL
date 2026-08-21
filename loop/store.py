"""MongoDB persistence for the fine-tuning loop.

Five collections in one database:

    subjects  one document per prompt spec — the subject line plus the metadata
              prompts.json carries for it (category, generator, sentiment, urls,
              attachments). Content-addressed and never mutated.
    runs      one document per loop invocation, holding the config every round
              must share (n_samples, algorithm, ref_mode, threshold), the
              ordered subject refs the run generates from, and the prompt
              structure its messages were rendered with
    rounds    one per round: which checkpoint it trained from and produced,
              the dataset it trained on, and its metrics
    messages  one per generated message, tagged with run/round/sample, an
              `added_at` timestamp, and a DBRef to the subject it came from
    datasets  a named, point-in-time slice of `messages`: a query, an `as_of`
              cut-off, and the hash of the rows that came back

Messages live in a single collection rather than one per checkpoint, because the
loop needs both slices cheaply: the cumulative training pool (`round <= N`) and
the round-scoped evaluation set (`round == N`).

Three kinds of identity, deliberately kept apart
------------------------------------------------

**Subjects** are content-addressed on the whole spec (`spec_hash`), so the same
prompts.json reuses documents and flipping `urls` writes a new one rather than
editing the old. A message points at its subject with a DBRef instead of
copying `category`/`generator`, which could only drift out of step with it.

**Prompt structure** is the *shape* the model was shown — the field names and
markers, values stripped:

    subject:
    urls:
    attachments:
    sentiment:
    ->

That is what makes two messages comparable as model inputs; the values are
already pinned by the subject ref. A run records the structures its round-0
messages were rendered with, and every later round must stay within that set,
so a change to `generate_prompt` half way through a run is caught. Hashing the
specs (which the run's subject list already pins) could never catch that.

**Datasets** are a query plus an `as_of` cut-off, resolved against `added_at`.
Because `messages` is append-only, that pair names a stable slice however much
is generated afterwards. What makes it verifiable is `content_hash`, taken over
the materialised rows: generation is stochastic and unseeded, so no hash of the
inputs can make a run reproducible, but a hash of the output rows says exactly
which data a round trained on, and `verify_dataset` re-materialises the slice
and re-checks it later.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from bson import DBRef, ObjectId
from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

DEFAULT_URI = "mongodb://localhost:27017/"
DEFAULT_DB = "phishnet_rl"

SUBJECTS_COLLECTION = "subjects"
DATASETS_COLLECTION = "datasets"
CHECKPOINTS_COLLECTION = "checkpoints"

# The files that decide what a checkpoint generates. Optimizer, scheduler, RNG
# and trainer state are deliberately left out: two checkpoints differing only
# there produce identical text, so they are not part of its identity. The
# tokenizer files are in, because they change how a prompt is encoded.
CHECKPOINT_IDENTITY_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

# What a training row is made of: row field <- message field. Datasets store
# their own copy, so a dataset built under an older mapping still says what it
# meant.
TRAINING_FIELDS = {"prompt": "prompt_text", "completion": "body", "label": "label"}

# The prompt-spec fields a subject document carries, i.e. every field of a
# prompts.json entry. `subject` is the line itself; the rest is what
# generate_prompt renders around it, plus the two labels (category, generator)
# the analysis groups by.
SUBJECT_FIELDS = ("subject", "category", "generator", "sentiment", "urls", "attachments")


def subject_fingerprint(spec: Dict) -> str:
    """Stable hash of one prompt spec — the subject collection's identity.

    Hashes exactly SUBJECT_FIELDS, sorted, so it does not depend on key order
    and ignores anything else a prompts.json entry happens to carry.
    """
    missing = [field for field in SUBJECT_FIELDS if field not in spec]
    if missing:
        raise ValueError(f"prompt spec is missing {missing}: {spec!r}")

    canonical = json.dumps(
        {field: spec[field] for field in SUBJECT_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def subject_document(spec: Dict) -> Dict:
    """A prompts.json entry as a subject document, hash included."""
    doc = {field: spec[field] for field in SUBJECT_FIELDS}
    doc["spec_hash"] = subject_fingerprint(spec)
    return doc


# =============================================================================
# Time
#
# `added_at` on a message and `as_of` on a dataset are compared against each
# other, so they have to be the same kind of datetime. pymongo hands back naive
# datetimes whatever it was given (BSON dates carry no zone), so everything is
# normalised to naive UTC on the way in rather than trusting the client's
# tz_aware setting.
# =============================================================================


def utc_now() -> datetime:
    """Now, as the naive-UTC datetime that comes back out of MongoDB."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def as_utc(value: Union[datetime, int, float, str]) -> datetime:
    """Normalise an `as_of` to naive UTC.

    Accepts a datetime (aware or naive-UTC), a unix timestamp — which is what a
    `run_id` is — or an ISO-8601 string, so a cut-off can come off a CLI.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).replace(tzinfo=None)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return as_utc(parsed)
    raise TypeError(f"cannot read {value!r} as a timestamp")


# =============================================================================
# Prompt structure
# =============================================================================

# A prompt line is `field: value`. The field name is the structure; the value is
# the subject spec, which the message's subject DBRef already pins.
_FIELD_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_ -]{0,40}):")

# Lines that are neither `field:` nor a short marker like `->` are a value that
# wrapped onto its own line. They collapse to this, so a subject containing a
# newline cannot leak its text into the structure.
_VALUE_LINE = "..."


def prompt_structure(prompt_text: str) -> str:
    """The shape of a rendered prompt: field names and markers, values stripped.

        subject: URGENT: act now      subject:
        urls: True               ->   urls:
        ->                            ->

    Derived from the text actually sent to the model, not from the code that
    built it, so it cannot fall out of step with what was generated.
    """
    lines = []
    for line in prompt_text.strip().splitlines():
        stripped = line.strip()
        match = _FIELD_LINE.match(stripped)
        if match:
            lines.append(f"{match.group(1)}:")
        elif stripped and len(stripped) <= 4 and " " not in stripped:
            lines.append(stripped)  # a marker, e.g. ->
        elif stripped:
            lines.append(_VALUE_LINE)
    return "\n".join(lines)


def structure_fingerprint(structure: str) -> str:
    """Stable hash of a prompt structure."""
    return hashlib.sha256(structure.encode()).hexdigest()[:16]


# =============================================================================
# Dataset content
# =============================================================================


# =============================================================================
# Checkpoint identity
# =============================================================================


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(path: str) -> List[Dict[str, Any]]:
    """Per-file name, size and digest for a checkpoint's identity files."""
    entries = []
    for name in CHECKPOINT_IDENTITY_FILES:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            entries.append(
                {
                    "name": name,
                    "size": os.path.getsize(candidate),
                    "sha256": _file_digest(candidate),
                }
            )
    return entries


def checkpoint_fingerprint(files: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Stable hash of a checkpoint's weights, or None if there were no files.

    Covers file names and sizes as well as contents, so a checkpoint that is
    missing its tokenizer does not hash the same as one that has it.
    """
    if not files:
        return None
    canonical = "\n".join(
        f"{entry['name']}:{entry['size']}:{entry['sha256']}"
        for entry in sorted(files, key=lambda entry: entry["name"])
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def dataset_spec_fingerprint(
    query: Dict[str, Any], as_of: datetime, fields: Dict[str, str]
) -> str:
    """Stable hash of what a dataset *is*: its query, cut-off and projection.

    Matched on instead of the spec documents themselves, because MongoDB
    compares embedded documents by key order — `{run_id, round}` would not
    equal `{round, run_id}` — and a canonical hash does not care.
    """
    canonical = json.dumps(
        {"query": query, "as_of": as_of.isoformat(), "fields": fields},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def row_fingerprint(row: Dict) -> str:
    """Stable hash of one materialised row."""
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def dataset_fingerprint(rows: Sequence[Dict]) -> str:
    """Stable hash of a set of rows, independent of the order they came back in.

    Built from the sorted per-row hashes, because a `find` gives no ordering
    guarantee and the trainers shuffle anyway — so retrieval order is not part
    of what the dataset *is*. Duplicate rows are kept (they are duplicated
    training signal, and `duplicate_count` reports on them).
    """
    digest = hashlib.sha256()
    for row_hash in sorted(row_fingerprint(row) for row in rows):
        digest.update(row_hash.encode())
    return digest.hexdigest()[:16]


class LoopStore:
    """Thin wrapper over the five collections. No ODM, no schema migrations."""

    def __init__(self, db_name: str = DEFAULT_DB, uri: str = DEFAULT_URI, client=None):
        self.client = client or MongoClient(uri)
        self.db = self.client[db_name]
        self.subjects = self.db[SUBJECTS_COLLECTION]
        self.runs = self.db["runs"]
        self.rounds = self.db["rounds"]
        self.messages = self.db["messages"]
        self.datasets = self.db[DATASETS_COLLECTION]
        self.checkpoints = self.db[CHECKPOINTS_COLLECTION]
        self._digest_cache: Dict[tuple, List[Dict[str, Any]]] = {}
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.subjects.create_index([("spec_hash", ASCENDING)], unique=True)
        self.subjects.create_index([("category", ASCENDING), ("generator", ASCENDING)])
        self.runs.create_index([("run_id", ASCENDING)], unique=True)
        self.rounds.create_index(
            [("run_id", ASCENDING), ("round", ASCENDING)], unique=True
        )
        self.messages.create_index([("run_id", ASCENDING), ("round", ASCENDING)])
        self.messages.create_index([("run_id", ASCENDING), ("prompt_id", ASCENDING)])
        # DBRefs index on their $id subfield; this is the "every message ever
        # generated from this subject" query, across runs.
        self.messages.create_index([("subject.$id", ASCENDING)])
        # every dataset resolves through an added_at range
        self.messages.create_index([("added_at", ASCENDING)])
        self.datasets.create_index([("spec_hash", ASCENDING)], unique=True)
        self.datasets.create_index([("content_hash", ASCENDING)])
        self.datasets.create_index([("as_of", ASCENDING)])
        self.checkpoints.create_index([("key", ASCENDING)], unique=True)
        self.checkpoints.create_index([("weights_hash", ASCENDING)])
        # "which messages did this checkpoint generate", the inverse of the
        # provenance stamp on each message
        self.messages.create_index([("checkpoint.$id", ASCENDING)])

    # -- subjects -----------------------------------------------------------

    def upsert_subject(self, spec: Dict) -> ObjectId:
        """Insert a prompt spec if new, and return its subject id either way.

        Content-addressed on `spec_hash`, so re-running with the same
        prompts.json reuses the existing documents instead of duplicating them.
        """
        doc = subject_document(spec)
        result = self.subjects.find_one_and_update(
            {"spec_hash": doc["spec_hash"]},
            {"$setOnInsert": doc},
            upsert=True,
            return_document=True,
            projection={"_id": 1},
        )
        return result["_id"]

    def sync_subjects(self, prompts: List[Dict]) -> List[ObjectId]:
        """Upsert every spec and return their ids, in prompt order."""
        return [self.upsert_subject(spec) for spec in prompts]

    def subject_ref(self, subject_id: ObjectId) -> DBRef:
        return DBRef(SUBJECTS_COLLECTION, subject_id)

    def get_subject(self, ref) -> Optional[Dict]:
        """Dereference a subject DBRef (or a bare ObjectId)."""
        subject_id = ref.id if isinstance(ref, DBRef) else ref
        return self.subjects.find_one({"_id": subject_id})

    def find_subjects(self, **query) -> List[Dict]:
        """Subjects matching a plain field query, e.g. `category="Urgency"`."""
        return list(self.subjects.find(query))

    def _subject_map(self, refs: Iterable) -> Dict[ObjectId, Dict]:
        """Fetch the given subjects in one round trip, keyed by id."""
        ids = {ref.id if isinstance(ref, DBRef) else ref for ref in refs if ref}
        if not ids:
            return {}
        return {doc["_id"]: doc for doc in self.subjects.find({"_id": {"$in": list(ids)}})}

    # -- checkpoints ---------------------------------------------------------

    def _checkpoint_files(self, path: str) -> List[Dict[str, Any]]:
        """Digest a checkpoint's files, once per (path, size, mtime) per session.

        Hashing a LoRA adapter is ~0.3 s, which is nothing once a round but adds
        up if every batch re-hashes the same directory.
        """
        if not os.path.isdir(path):
            return []
        stat = os.stat(path)
        key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._digest_cache:
            self._digest_cache[key] = checkpoint_files(path)
        return self._digest_cache[key]

    def upsert_checkpoint(
        self, path: str, produced_by: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Content-address a checkpoint directory and return its document.

        Identity is the hash of the weights, not the path — the same adapter
        copied elsewhere is the same checkpoint, and a path reused for a
        different adapter is not. A directory that is not on disk (a stub, or a
        checkpoint since deleted) still gets a document, keyed by its path, so
        the messages it generated keep a provenance record either way.
        """
        files = self._checkpoint_files(path)
        weights_hash = checkpoint_fingerprint(files)

        base_model = None
        config_path = os.path.join(path, "adapter_config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path) as handle:
                    base_model = json.load(handle).get("base_model_name_or_path")
            except (OSError, json.JSONDecodeError):
                base_model = None

        document = {
            "key": weights_hash or f"path:{os.path.abspath(path)}",
            "weights_hash": weights_hash,
            "base_model": base_model,
            "files": files,
            "first_seen": utc_now(),
            "produced_by": produced_by,
        }

        return self.checkpoints.find_one_and_update(
            {"key": document["key"]},
            {
                "$setOnInsert": document,
                # location is not identity: record where it has been seen, and
                # keep `path` pointing at the most recent copy
                "$set": {"path": path},
                "$addToSet": {"paths": path},
            },
            upsert=True,
            return_document=True,
        )

    def checkpoint_ref(self, checkpoint) -> DBRef:
        checkpoint_id = checkpoint["_id"] if isinstance(checkpoint, dict) else checkpoint
        return DBRef(CHECKPOINTS_COLLECTION, checkpoint_id)

    def get_checkpoint(self, ref) -> Optional[Dict]:
        """Dereference a checkpoint DBRef (or a bare ObjectId)."""
        checkpoint_id = ref.id if isinstance(ref, DBRef) else ref
        return self.checkpoints.find_one({"_id": checkpoint_id})

    def resolve_checkpoint(self, checkpoint) -> Optional[Dict]:
        """Take a path, a document, a DBRef or an id, and return the document."""
        if checkpoint is None:
            return None
        if isinstance(checkpoint, str):
            return self.upsert_checkpoint(checkpoint)
        if isinstance(checkpoint, dict):
            return checkpoint
        return self.get_checkpoint(checkpoint)

    def checkpoint_for_message(self, message: Dict) -> Optional[Dict]:
        """The checkpoint that generated a given message."""
        return self.get_checkpoint(message["checkpoint"]) if message.get("checkpoint") else None

    def messages_for_checkpoint(self, checkpoint, run_id: Optional[int] = None) -> List[Dict]:
        """Every message a checkpoint generated, optionally within one run."""
        document = self.resolve_checkpoint(checkpoint)
        if document is None:
            return []
        query: Dict[str, Any] = {"checkpoint.$id": document["_id"]}
        if run_id is not None:
            query["run_id"] = run_id
        return list(
            self.messages.find(query, {"_id": 0}).sort(
                [("run_id", ASCENDING), ("round", ASCENDING)]
            )
        )

    def verify_checkpoint(self, checkpoint, path: Optional[str] = None) -> Dict[str, Any]:
        """Re-hash a checkpoint on disk and check it is the one that was used.

        Answers "was this message really generated by the adapter now sitting at
        that path". `path` overrides where to look, for a checkpoint that has
        been moved or restored from a copy.
        """
        document = self.resolve_checkpoint(checkpoint)
        if document is None:
            raise KeyError(f"no such checkpoint: {checkpoint}")

        location = path or document.get("path")
        present = bool(location) and os.path.isdir(location)
        actual = (
            checkpoint_fingerprint(self._checkpoint_files(location)) if present else None
        )

        return {
            "checkpoint_id": document["_id"],
            "path": location,
            "present": present,
            "ok": present and actual is not None and actual == document.get("weights_hash"),
            "recorded": document.get("weights_hash"),
            "actual": actual,
            "base_model": document.get("base_model"),
        }

    # -- runs ---------------------------------------------------------------

    def create_run(self, prompts: List[Dict], config: Dict[str, Any]) -> int:
        """Start a run. Returns its id (a unix timestamp, as phishnet uses).

        The prompt specs are upserted into `subjects` first; the run records the
        refs in prompt order, which is what `prompt_id` indexes into. That
        ordered list *is* the identity of the run's prompt set, so there is no
        separate hash of the specs. `prompt_structures` is filled in from the
        first messages stored, since the rendered shape is only known then.
        """
        subject_ids = self.sync_subjects(prompts)
        document = {
            "created_at": int(time.time()),
            "prompt_count": len(prompts),
            "subjects": [self.subject_ref(sid) for sid in subject_ids],
            "prompt_structures": None,
            "config": config,
        }

        # The id is a unix timestamp, as phishnet uses, so two runs started in
        # the same second would collide on the unique index. Step forward until
        # one is free rather than failing: the id stays sortable and readable,
        # and only the second run of a pair is off by a second.
        run_id = document["created_at"]
        while True:
            try:
                self.runs.insert_one({**document, "run_id": run_id})
                return run_id
            except DuplicateKeyError:
                run_id += 1

    def get_run(self, run_id: int) -> Optional[Dict]:
        return self.runs.find_one({"run_id": run_id}, {"_id": 0})

    def check_prompts(self, run_id: int, prompts: List[Dict]) -> None:
        """Raise if these specs are not the ones the run was created with.

        Compares against the run's subject documents directly — their
        `spec_hash` is the same fingerprint `upsert_subject` dedupes on — rather
        than against a second stored hash that could disagree with them.
        """
        refs = self.run_subject_refs(run_id)
        if len(prompts) != len(refs):
            raise ValueError(
                f"run {run_id} was created with {len(refs)} prompts, got "
                f"{len(prompts)}; rounds would not be comparable"
            )

        by_id = self._subject_map(refs)
        for prompt_id, (spec, ref) in enumerate(zip(prompts, refs)):
            expected = by_id[ref.id]["spec_hash"]
            actual = subject_fingerprint(spec)
            if actual != expected:
                raise ValueError(
                    f"prompt {prompt_id} changed since run {run_id} started "
                    f"({expected} -> {actual}: {spec.get('subject')!r}); "
                    "rounds would not be comparable"
                )

    def check_prompt_structure(self, run_id: int, structures: Iterable[str]) -> None:
        """Raise if a prompt shape appears that the run did not start with.

        The first structures seen are recorded on the run; everything after has
        to stay within that set. This is what catches `generate_prompt` changing
        under a half-finished run — a hash of the specs cannot see it, because
        the specs are unchanged and only their rendering moved.
        """
        seen = sorted(set(structures))
        if not seen:
            return

        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"no such run: {run_id}")

        known = run.get("prompt_structures")
        if not known:
            self.runs.update_one(
                {"run_id": run_id},
                {
                    "$set": {
                        "prompt_structures": [
                            {"hash": structure_fingerprint(s), "structure": s}
                            for s in seen
                        ]
                    }
                },
            )
            return

        known_structures = {entry["structure"] for entry in known}
        new = [s for s in seen if s not in known_structures]
        if new:
            raise ValueError(
                f"run {run_id} was generated with prompt structure(s) "
                f"{sorted(known_structures)!r}, but this round rendered "
                f"{new!r}; the messages are not comparable as model inputs"
            )

    def config_drift(self, run_id: int, config: Dict[str, Any]) -> Dict[str, tuple]:
        """Fields where `config` differs from what the run was created with.

        Returns {field: (run value, given value)}. Resuming builds a fresh
        runner from whatever flags were passed that time, so the run's config is
        its original intent, not a guarantee — this is what turns a silent
        divergence into a reported one.
        """
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"no such run: {run_id}")

        recorded = run.get("config") or {}
        return {
            field: (recorded[field], value)
            for field, value in config.items()
            if field in recorded and recorded[field] != value
        }

    def run_subject_refs(self, run_id: int) -> List[DBRef]:
        """The run's subject refs, indexed by `prompt_id`."""
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"no such run: {run_id}")
        return list(run.get("subjects") or [])

    def run_prompts(self, run_id: int) -> List[Dict]:
        """The run's prompt specs, in prompt order, straight from `subjects`.

        A resumed run generates from these rather than from prompts.json, so it
        keeps measuring what it started measuring even if the file has moved on.
        """
        refs = self.run_subject_refs(run_id)
        by_id = self._subject_map(refs)
        specs = []
        for prompt_id, ref in enumerate(refs):
            doc = by_id.get(ref.id)
            if doc is None:
                raise KeyError(
                    f"run {run_id} prompt {prompt_id} points at missing subject {ref.id}"
                )
            specs.append({field: doc[field] for field in SUBJECT_FIELDS})
        return specs

    # -- messages -----------------------------------------------------------

    def add_messages(
        self,
        run_id: int,
        round_index: int,
        records: Iterable[Dict],
        checkpoint=None,
        added_at: Optional[datetime] = None,
    ) -> int:
        """Store this round's generated + scored messages. Returns the count.

        Each record's `prompt_id` is resolved against the run's subject list and
        stored as a DBRef. The `category`/`generator` the generator carries on
        its in-memory records are dropped here: the subject document owns them,
        and a second copy on every message could only ever disagree with it.

        Every message is stamped with:

        `added_at`  one timestamp for the whole batch, so a round is a single
                    point in time and no `as_of` can cut a round in half
        `checkpoint` a DBRef to the checkpoint that generated it, plus
                    `checkpoint_hash`, the hash of that adapter's weights
        `prompt_structure_hash` the shape it was rendered with

        The two hashes are cached derivations of immutable things — this
        message's own `prompt_text`, and weights that are content-addressed —
        so unlike a copy of `category` they cannot go stale. Stamping the
        checkpoint here rather than leaving it on the round document means a
        message knows its own provenance even if the run dies before the round
        is recorded.
        """
        refs = self.run_subject_refs(run_id)
        stamp = as_utc(added_at) if added_at is not None else utc_now()

        checkpoint_doc = self.resolve_checkpoint(checkpoint)
        checkpoint_ref = self.checkpoint_ref(checkpoint_doc) if checkpoint_doc else None

        docs, structures = [], []
        for record in records:
            doc = dict(record)
            prompt_id = doc.get("prompt_id")
            if not isinstance(prompt_id, int) or not 0 <= prompt_id < len(refs):
                raise ValueError(
                    f"message has prompt_id {prompt_id!r}, which is not a prompt of "
                    f"run {run_id} (it has {len(refs)})"
                )
            for field in SUBJECT_FIELDS:
                doc.pop(field, None)
            doc["subject"] = refs[prompt_id]
            doc["run_id"] = run_id
            doc["round"] = round_index
            doc["added_at"] = stamp
            if checkpoint_ref is not None:
                doc["checkpoint"] = checkpoint_ref
                doc["checkpoint_hash"] = checkpoint_doc.get("weights_hash")

            structure = prompt_structure(doc.get("prompt_text", ""))
            doc["prompt_structure_hash"] = structure_fingerprint(structure)
            structures.append(structure)
            docs.append(doc)

        if not docs:
            return 0

        # checked before the insert, so a mismatched round does not land
        self.check_prompt_structure(run_id, structures)
        self.messages.insert_many(docs)
        return len(docs)

    def get_messages(
        self,
        run_id: int,
        round_index: Optional[int] = None,
        max_round: Optional[int] = None,
        with_subject: bool = True,
    ) -> List[Dict]:
        """Messages for a run.

        `round_index` selects one round (evaluation); `max_round` selects every
        round up to and including it (the cumulative training pool). Passing
        neither returns everything.

        `with_subject` joins each message's subject back in — `subject_text`
        plus the spec fields (`category`, `generator`, `sentiment`, `urls`,
        `attachments`) — which is the shape the analysis and the charts group by.
        The DBRef itself stays under `subject`. Pass False on the hot paths that
        only need bodies and labels, to skip the extra query.
        """
        query: Dict[str, Any] = {"run_id": run_id}
        if round_index is not None:
            query["round"] = round_index
        elif max_round is not None:
            query["round"] = {"$lte": max_round}

        messages = list(self.messages.find(query, {"_id": 0}).sort("round", ASCENDING))
        return self.attach_subjects(messages) if with_subject else messages

    def attach_subjects(self, messages: List[Dict]) -> List[Dict]:
        """Merge each message's subject spec into it, in one extra query."""
        by_id = self._subject_map(m.get("subject") for m in messages)

        for message in messages:
            ref = message.get("subject")
            doc = by_id.get(ref.id) if isinstance(ref, DBRef) else None
            if doc is None:
                continue
            # the subject line goes in as `subject_text`, because `subject` is
            # the ref — everything else keeps its prompts.json name
            message["subject_text"] = doc["subject"]
            for field in SUBJECT_FIELDS:
                if field != "subject":
                    message[field] = doc[field]

        return messages

    def messages_for_subject(self, subject, run_id: Optional[int] = None) -> List[Dict]:
        """Every message generated from one subject, optionally within a run.

        The cross-run query the subject collection exists to make possible: how
        this subject line has fared, over every run that used it.
        """
        subject_id = subject.id if isinstance(subject, DBRef) else subject
        query: Dict[str, Any] = {"subject.$id": subject_id}
        if run_id is not None:
            query["run_id"] = run_id
        return list(
            self.messages.find(query, {"_id": 0}).sort(
                [("run_id", ASCENDING), ("round", ASCENDING)]
            )
        )

    # -- datasets -----------------------------------------------------------

    def pool_query(self, run_id: int, max_round: int) -> Dict[str, Any]:
        """The query behind the cumulative training pool, as a dataset spec."""
        return {"run_id": run_id, "round": {"$lte": max_round}}

    def materialise(
        self,
        query: Dict[str, Any],
        as_of: Optional[Union[datetime, int, float, str]] = None,
        fields: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Resolve a dataset spec into rows, without storing anything.

        `query` is a plain filter over `messages`; `as_of` cuts it at a point in
        time against `added_at` (default: now). Because messages are only ever
        appended, that pair names the same rows however much is generated
        afterwards. `fields` maps row field <- message field.
        """
        fields = dict(fields or TRAINING_FIELDS)
        cutoff = as_utc(as_of) if as_of is not None else utc_now()
        filter_ = {**query, "added_at": {"$lte": cutoff}}

        rows, structures = [], set()
        for message in self.messages.find(filter_, {"_id": 0}):
            missing = [source for source in fields.values() if source not in message]
            if missing:
                raise ValueError(
                    f"message from run {message.get('run_id')} round "
                    f"{message.get('round')} has no {missing}; it was never scored"
                )
            rows.append({name: message[source] for name, source in fields.items()})
            if message.get("prompt_structure_hash"):
                structures.add(message["prompt_structure_hash"])

        return {
            "query": query,
            "as_of": cutoff,
            "fields": fields,
            "spec_hash": dataset_spec_fingerprint(query, cutoff, fields),
            "rows": rows,
            "count": len(rows),
            "content_hash": dataset_fingerprint(rows),
            "prompt_structures": sorted(structures),
        }

    def create_dataset(
        self,
        query: Dict[str, Any],
        as_of: Optional[Union[datetime, int, float, str]] = None,
        fields: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        export_path: Optional[str] = None,
        **extra,
    ) -> Dict[str, Any]:
        """Materialise a dataset, optionally export it, and record what it was.

        Re-creating the same spec (query + as_of + fields) returns the existing
        document when the rows still hash the same, and raises when they do not
        — a point-in-time slice that changed means history was rewritten
        underneath it, which is exactly what the content hash is for.

        Returns the stored document plus the materialised `rows`.
        """
        resolved = self.materialise(query, as_of, fields)
        rows = resolved.pop("rows")

        if export_path:
            with open(export_path, "w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, default=str) + "\n")

        existing = self.datasets.find_one({"spec_hash": resolved["spec_hash"]})
        if existing is not None:
            if existing["content_hash"] != resolved["content_hash"]:
                raise ValueError(
                    f"dataset {existing['_id']} ({existing.get('name')}) resolved to "
                    f"{resolved['count']} rows hashing {resolved['content_hash']}, but "
                    f"was recorded as {existing['count']} rows hashing "
                    f"{existing['content_hash']}; the messages under it have changed"
                )
            if export_path and existing.get("export_path") != export_path:
                self.datasets.update_one(
                    {"_id": existing["_id"]}, {"$set": {"export_path": export_path}}
                )
                existing["export_path"] = export_path
            return {**existing, "rows": rows}

        document = {
            **resolved,
            "name": name,
            "export_path": export_path,
            "created_at": utc_now(),
            **extra,
        }
        document["_id"] = self.datasets.insert_one(document).inserted_id
        return {**document, "rows": rows}

    def dataset_ref(self, dataset) -> DBRef:
        dataset_id = dataset["_id"] if isinstance(dataset, dict) else dataset
        return DBRef(DATASETS_COLLECTION, dataset_id)

    def get_dataset(self, ref) -> Optional[Dict]:
        """Dereference a dataset DBRef (or a bare ObjectId)."""
        dataset_id = ref.id if isinstance(ref, DBRef) else ref
        return self.datasets.find_one({"_id": dataset_id})

    def dataset_rows(self, ref) -> List[Dict]:
        """Re-materialise a stored dataset from its query and cut-off."""
        dataset = self.get_dataset(ref)
        if dataset is None:
            raise KeyError(f"no such dataset: {ref}")
        return self.materialise(
            dataset["query"], dataset["as_of"], dataset["fields"]
        )["rows"]

    def verify_dataset(self, ref) -> Dict[str, Any]:
        """Re-resolve a stored dataset and check it still hashes the same.

        The failsafe: `as_of` says the slice *should* be stable, and this is
        what proves it — a deleted, edited or back-dated message shows up as a
        hash mismatch rather than as a quietly different training set.
        """
        dataset = self.get_dataset(ref)
        if dataset is None:
            raise KeyError(f"no such dataset: {ref}")

        resolved = self.materialise(
            dataset["query"], dataset["as_of"], dataset["fields"]
        )
        return {
            "dataset_id": dataset["_id"],
            "name": dataset.get("name"),
            "ok": resolved["content_hash"] == dataset["content_hash"]
            and resolved["count"] == dataset["count"],
            "recorded": {
                "count": dataset["count"],
                "content_hash": dataset["content_hash"],
            },
            "actual": {
                "count": resolved["count"],
                "content_hash": resolved["content_hash"],
            },
        }

    def training_pool(self, run_id: int, max_round: int, as_of=None) -> List[Dict]:
        """The cumulative KTO/BCO dataset: every message from rounds 0..max_round.

        Labels stay valid across rounds because ScamLLM is frozen, so nothing
        needs re-scoring. Nothing is recorded — `create_dataset` is the path
        that pins a slice a round actually trained on.
        """
        return self.materialise(self.pool_query(run_id, max_round), as_of)["rows"]

    def label_counts(
        self, run_id: int, max_round: Optional[int] = None, round_index: Optional[int] = None
    ) -> Dict[str, int]:
        """Desirable/undesirable counts, for the KTO class weights."""
        messages = self.get_messages(
            run_id, round_index=round_index, max_round=max_round, with_subject=False
        )
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
        bodies = [
            m["body"]
            for m in self.get_messages(run_id, max_round=max_round, with_subject=False)
        ]
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
        """Delete a run and everything belonging to it.

        Subjects are left alone: they are shared with every other run that used
        the same prompt specs. `prune_subjects` clears out any that are then
        left unreferenced. Datasets scoped to this run go, since their query
        can no longer resolve to anything.
        """
        self.messages.delete_many({"run_id": run_id})
        self.rounds.delete_many({"run_id": run_id})
        self.datasets.delete_many({"query.run_id": run_id})
        self.runs.delete_one({"run_id": run_id})

    def prune_subjects(self) -> int:
        """Delete subjects no run refers to. Returns how many went."""
        referenced = {
            ref.id for run in self.runs.find({}, {"subjects": 1}) for ref in run.get("subjects") or []
        }
        return self.subjects.delete_many({"_id": {"$nin": list(referenced)}}).deleted_count

    def close(self) -> None:
        self.client.close()
