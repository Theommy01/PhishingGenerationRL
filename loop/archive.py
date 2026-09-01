"""Move runs between databases: export one to a file, import it somewhere else.

    python -m loop.archive export run-1786641452.tar.gz --run 1786641452
    python -m loop.archive inspect run-1786641452.tar.gz
    python -m loop.archive import run-1786641452.tar.gz

An archive is a gzipped JSONL file per collection plus a `manifest.json`, either
in a directory or wrapped in a `.tar.gz`. Documents are written as MongoDB
**canonical** Extended JSON, which is the part that matters: this schema is held
together by DBRefs — a message points at its subject and its checkpoint, a round
at its dataset — and `json.dumps(default=str)` would flatten those into strings,
landing a pile of documents on the far side that no longer point at each other.
Extended JSON round-trips DBRef, ObjectId and datetime exactly. Canonical rather
than relaxed because it keeps ints and doubles apart, and `dataset_fingerprint`
hashes them — under relaxed a dataset could arrive intact and still fail
`verify_dataset`. `bson.json_util.loads` reads a line back in one call.

This is the faithful, whole-provenance export. It is not the one to hand someone
who just wants rows to train on — that is `create_dataset(export_path=...)`,
which writes plain JSONL of prompt/completion/label.

What travels
------------

Naming a run pulls in everything it needs to stand up on its own: its rounds and
messages, the subjects and checkpoints they reference, and the datasets its
rounds trained on. Subjects and checkpoints are shared across runs, so exporting
two runs that use the same prompts.json carries one copy of each subject.

What does not travel is the adapter weights. A checkpoint document is a hash and
a file manifest, not the 168 MB it describes, so on the far side
`verify_checkpoint` reports `present: false` — correctly, since those weights are
not there. The hashes still transfer, so if the weights are shipped separately
the recipient can verify they are the ones that generated the messages.

Arriving without collisions
---------------------------

The import cannot simply insert what it reads. Subjects, checkpoints and
datasets are content-addressed, and the recipient may already hold the same
document under a different `_id` — from their own run over the same
prompts.json. So each is matched on its natural key (`spec_hash`, `key`,
`spec_hash`) first, and where a local one already exists the incoming `_id` is
mapped onto it and every DBRef pointing at it is rewritten. Nothing is ever
mutated to match the archive; the local document wins, which is what
content-addressing means.

`run_id` is a unix timestamp, so two researchers running the same afternoon can
collide on one. An existing `run_id` is skipped by default; `--replace` drops the
local run first and `--remap` imports under the next free id.

Re-importing the same archive twice is a no-op rather than a duplicate.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from bson import DBRef, ObjectId, json_util
from pymongo import UpdateOne

from loop.store import (
    CHECKPOINTS_COLLECTION,
    DATASETS_COLLECTION,
    SUBJECTS_COLLECTION,
    LoopStore,
    as_utc,
    dataset_spec_fingerprint,
    utc_now,
)

FORMAT = "phishnet-rl-archive"
FORMAT_VERSION = 1

MANIFEST_NAME = "manifest.json"

# Import order, not export order: a document is written only once everything it
# refers to is in place, so a half-finished import never leaves a dangling ref.
COLLECTIONS = (
    SUBJECTS_COLLECTION,
    CHECKPOINTS_COLLECTION,
    DATASETS_COLLECTION,
    "runs",
    "rounds",
    "messages",
)

# The field an entity is deduped on when it lands in a database that may already
# hold it. `runs`, `rounds` and `messages` are scoped to a run instead.
NATURAL_KEY = {
    SUBJECTS_COLLECTION: "spec_hash",
    CHECKPOINTS_COLLECTION: "key",
    DATASETS_COLLECTION: "spec_hash",
}

# Canonical Extended JSON: `{"$oid": ...}`, `{"$date": {"$numberLong": ...}}`,
# ints and doubles distinguishable. Verbose to read, exact to reload.
_JSON_OPTIONS = json_util.CANONICAL_JSON_OPTIONS


# =============================================================================
# Reading and writing the container
# =============================================================================


def _encode(documents: Iterable[Dict]) -> bytes:
    """One document per line, as canonical Extended JSON."""
    out = io.BytesIO()
    for document in documents:
        line = json_util.dumps(document, json_options=_JSON_OPTIONS)
        out.write(line.encode() + b"\n")
    return out.getvalue()


def _decode(payload: bytes) -> List[Dict]:
    return [
        json_util.loads(line, json_options=_JSON_OPTIONS)
        for line in payload.splitlines()
        if line.strip()
    ]


def _digest(payload: bytes) -> str:
    """Hash the *uncompressed* payload.

    gzip stamps an mtime into its header, so hashing the compressed bytes would
    give a different digest for the same documents exported twice.
    """
    return hashlib.sha256(payload).hexdigest()[:16]


def _is_tarball(path: str) -> bool:
    return path.endswith((".tar.gz", ".tgz"))


def _member_name(collection: str) -> str:
    return f"{collection}.jsonl.gz"


def _write(path: str, files: Dict[str, bytes]) -> None:
    """Write the archive, as a tarball or a directory of the same members."""
    if _is_tarball(path):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with tarfile.open(path, "w:gz") as tar:
            for name, payload in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mtime = int(time.time())
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(payload))
        return

    os.makedirs(path, exist_ok=True)
    for name, payload in files.items():
        with open(os.path.join(path, name), "wb") as handle:
            handle.write(payload)


def _read(path: str) -> Dict[str, bytes]:
    """Read every member of an archive, tarball or directory alike."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no archive at {path}")

    if os.path.isdir(path):
        return {
            name: open(os.path.join(path, name), "rb").read()
            for name in sorted(os.listdir(path))
            if name == MANIFEST_NAME or name.endswith(".jsonl.gz")
        }

    files = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # tarballs made elsewhere may carry a leading directory
            name = os.path.basename(member.name)
            if name == MANIFEST_NAME or name.endswith(".jsonl.gz"):
                extracted = tar.extractfile(member)
                if extracted is not None:
                    files[name] = extracted.read()
    return files


# =============================================================================
# Export
# =============================================================================


def _ref_ids(value: Any, collection: str, into: set) -> None:
    """Collect the ids of every DBRef into `collection`, however deeply nested."""
    if isinstance(value, DBRef):
        if value.collection == collection:
            into.add(value.id)
    elif isinstance(value, dict):
        for item in value.values():
            _ref_ids(item, collection, into)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _ref_ids(item, collection, into)


def collect(store: LoopStore, run_ids: Optional[Sequence[int]] = None) -> Dict[str, List[Dict]]:
    """Gather the documents that make the named runs self-contained.

    Passing no run ids takes the whole database. Documents come back with their
    `_id`s intact — the DBRefs between them are the point of the export, and an
    id that has to be remapped on arrival is remapped there.
    """
    if run_ids is None:
        runs = list(store.runs.find({}))
    else:
        runs = list(store.runs.find({"run_id": {"$in": list(run_ids)}}))
        missing = set(run_ids) - {run["run_id"] for run in runs}
        if missing:
            raise KeyError(f"no such run(s): {sorted(missing)}")

    ids = [run["run_id"] for run in runs]
    rounds = list(store.rounds.find({"run_id": {"$in": ids}}))
    messages = list(store.messages.find({"run_id": {"$in": ids}}))

    # Subjects and checkpoints are reached only through refs, so walk the
    # documents rather than assuming which fields hold them.
    subject_ids: set = set()
    checkpoint_ids: set = set()
    dataset_ids: set = set()
    for document in (*runs, *rounds, *messages):
        _ref_ids(document, SUBJECTS_COLLECTION, subject_ids)
        _ref_ids(document, CHECKPOINTS_COLLECTION, checkpoint_ids)
        _ref_ids(document, DATASETS_COLLECTION, dataset_ids)

    # A dataset scoped to one of these runs travels even if no round references
    # it: it names a slice of messages that are in the archive.
    datasets = list(
        store.datasets.find(
            {"$or": [{"_id": {"$in": list(dataset_ids)}}, {"query.run_id": {"$in": ids}}]}
        )
    )

    return {
        SUBJECTS_COLLECTION: list(store.subjects.find({"_id": {"$in": list(subject_ids)}})),
        CHECKPOINTS_COLLECTION: list(
            store.checkpoints.find({"_id": {"$in": list(checkpoint_ids)}})
        ),
        DATASETS_COLLECTION: datasets,
        "runs": runs,
        "rounds": rounds,
        "messages": messages,
    }


def export_archive(
    store: LoopStore,
    path: str,
    run_ids: Optional[Sequence[int]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Write the named runs to `path`. Returns the manifest."""
    documents = collect(store, run_ids)

    files, collections = {}, {}
    for collection in COLLECTIONS:
        payload = _encode(documents[collection])
        files[_member_name(collection)] = gzip.compress(payload)
        collections[collection] = {
            "count": len(documents[collection]),
            "sha256": _digest(payload),
        }

    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "exported_at": utc_now().isoformat(),
        "source_db": store.db.name,
        "runs": [
            {
                "run_id": run["run_id"],
                "prompt_count": run.get("prompt_count"),
                "config": run.get("config"),
                "rounds": sum(1 for r in documents["rounds"] if r["run_id"] == run["run_id"]),
                "messages": sum(
                    1 for m in documents["messages"] if m["run_id"] == run["run_id"]
                ),
            }
            for run in sorted(documents["runs"], key=lambda run: run["run_id"])
        ],
        "collections": collections,
        "note": note,
    }

    files[MANIFEST_NAME] = json.dumps(manifest, indent=2, default=str).encode() + b"\n"
    _write(path, files)
    return manifest


# =============================================================================
# Import
# =============================================================================


def read_archive(path: str) -> Dict[str, Any]:
    """Read and integrity-check an archive. Returns its manifest + documents."""
    files = _read(path)
    if MANIFEST_NAME not in files:
        raise ValueError(f"{path} has no {MANIFEST_NAME}; it is not an archive")

    manifest = json.loads(files[MANIFEST_NAME])
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} archive (says {manifest.get('format')!r})")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path} is format version {manifest.get('format_version')}, "
            f"and this reads version {FORMAT_VERSION}"
        )

    documents = {}
    for collection in COLLECTIONS:
        name = _member_name(collection)
        if name not in files:
            raise ValueError(f"{path} is missing {name}")
        payload = gzip.decompress(files[name])

        recorded = (manifest.get("collections") or {}).get(collection) or {}
        actual = _digest(payload)
        if recorded.get("sha256") and recorded["sha256"] != actual:
            raise ValueError(
                f"{name} hashes {actual}, but the manifest records "
                f"{recorded['sha256']}; the archive is damaged"
            )

        documents[collection] = _decode(payload)
        if recorded.get("count") is not None and recorded["count"] != len(documents[collection]):
            raise ValueError(
                f"{name} holds {len(documents[collection])} documents, but the "
                f"manifest records {recorded['count']}"
            )

    return {"manifest": manifest, "documents": documents}


def _remap_refs(value: Any, id_map: Dict[ObjectId, ObjectId]) -> Any:
    """Rebuild a document with every DBRef repointed through `id_map`."""
    if isinstance(value, DBRef):
        return DBRef(value.collection, id_map.get(value.id, value.id), value.database)
    if isinstance(value, dict):
        return {key: _remap_refs(item, id_map) for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_refs(item, id_map) for item in value]
    return value


def _remap_run_ids(value: Any, run_map: Dict[int, int]) -> Any:
    """Rewrite `run_id` wherever it appears, including inside a dataset query."""
    if isinstance(value, dict):
        return {
            key: run_map.get(item, item)
            if key == "run_id" and isinstance(item, int)
            else _remap_run_ids(item, run_map)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_remap_run_ids(item, run_map) for item in value]
    return value


def _free_run_id(store: LoopStore, run_id: int, claimed: Sequence[int] = ()) -> int:
    """The first id at or after `run_id` that is neither in the database nor taken.

    `claimed` is the ids the rest of this import is already going to use — an
    archive holding runs 100 and 101 imported into a database that has 100 must
    not remap 100 onto 101 and then collide with the archive's own 101.
    """
    taken = set(claimed)
    while run_id in taken or store.runs.find_one({"run_id": run_id}, {"_id": 1}) is not None:
        run_id += 1
    return run_id


def _place_entity(
    collection, document: Dict, key_field: str, id_map: Dict[ObjectId, ObjectId]
) -> str:
    """Land one content-addressed document, and record where its id ended up.

    Three cases: the natural key is already here (map onto the local document
    and change nothing — content-addressing means it *is* the same document);
    the id is free (insert as it came, so refs elsewhere in the archive need no
    rewriting); the id is taken by something else (insert under a fresh id and
    let the ref rewrite follow it).
    """
    incoming_id = document["_id"]
    existing = collection.find_one({key_field: document[key_field]}, {"_id": 1})
    if existing is not None:
        id_map[incoming_id] = existing["_id"]
        return "existing"

    if collection.find_one({"_id": incoming_id}, {"_id": 1}) is not None:
        document = {**document, "_id": ObjectId()}
        id_map[incoming_id] = document["_id"]

    collection.insert_one(document)
    return "inserted"


def import_archive(
    store: LoopStore, path: str, on_conflict: str = "skip", verify: bool = True
) -> Dict[str, Any]:
    """Load an archive into `store`.

    `on_conflict` decides what happens to a `run_id` this database already
    holds: `skip` leaves the local run alone, `replace` drops it first, `remap`
    imports under the next free id.

    Idempotent: importing the same archive twice inserts nothing the second
    time. Returns a summary, including a re-verification of every dataset that
    landed — which is what proves the messages under it arrived intact.
    """
    if on_conflict not in ("skip", "replace", "remap"):
        raise ValueError(f"unknown conflict policy: {on_conflict!r}")

    read = read_archive(path)
    documents = read["documents"]

    # -- decide what to do with each run, before anything is written ---------

    run_map: Dict[int, int] = {}
    skipped: List[int] = []
    replaced: List[int] = []
    remapped: Dict[int, int] = {}

    archived = [run["run_id"] for run in documents["runs"]]
    for run_id in archived:
        if store.runs.find_one({"run_id": run_id}, {"_id": 1}) is None:
            continue
        if on_conflict == "skip":
            skipped.append(run_id)
        elif on_conflict == "replace":
            store.drop_run(run_id)
            replaced.append(run_id)
        else:
            # the archive's own ids are claimed too, so remapping one run does
            # not land on another run in the same archive
            fresh = _free_run_id(store, run_id + 1, [*archived, *run_map.values()])
            run_map[run_id] = fresh
            remapped[run_id] = fresh

    wanted = set(archived) - set(skipped)
    if not wanted:
        return {
            "source_db": read["manifest"].get("source_db"),
            "runs": [],
            "skipped": skipped,
            "replaced": [],
            "remapped": {},
            "inserted": {collection: 0 for collection in COLLECTIONS},
            "datasets": [],
        }

    def in_scope(document: Dict) -> bool:
        return document.get("run_id") in wanted

    # -- entities first, so nothing is inserted with a ref that cannot resolve

    id_map: Dict[ObjectId, ObjectId] = {}
    inserted = {collection: 0 for collection in COLLECTIONS}

    for document in documents[SUBJECTS_COLLECTION]:
        if _place_entity(store.subjects, dict(document), "spec_hash", id_map) == "inserted":
            inserted[SUBJECTS_COLLECTION] += 1

    for document in documents[CHECKPOINTS_COLLECTION]:
        document = dict(document)
        existing = store.checkpoints.find_one({"key": document["key"]}, {"_id": 1})
        if existing is not None:
            id_map[document["_id"]] = existing["_id"]
            # The sender's paths are where *they* keep the adapter. Record them
            # as places it has been seen without touching the local `path`,
            # which is the copy `verify_checkpoint` can actually re-hash.
            paths = [p for p in (document.get("paths") or []) if p]
            if paths:
                store.checkpoints.update_one(
                    {"_id": existing["_id"]}, {"$addToSet": {"paths": {"$each": paths}}}
                )
            continue
        if _place_entity(store.checkpoints, document, "key", id_map) == "inserted":
            inserted[CHECKPOINTS_COLLECTION] += 1

    # -- datasets: entities, but their query names a run that may have moved --

    for document in documents[DATASETS_COLLECTION]:
        # A dataset scoped to a skipped run stays out: its query would resolve
        # against the *local* run of that id, which is not the slice it was
        # recorded against, and it would fail its own content hash.
        if document.get("query", {}).get("run_id") in skipped:
            continue
        document = _remap_refs(dict(document), id_map)
        if run_map:
            document = _remap_run_ids(document, run_map)
            # the spec hash covers the query, so a moved run_id changes it
            document["spec_hash"] = dataset_spec_fingerprint(
                document["query"], as_utc(document["as_of"]), document["fields"]
            )
        if _place_entity(store.datasets, document, "spec_hash", id_map) == "inserted":
            inserted[DATASETS_COLLECTION] += 1

    # -- runs, rounds, messages ----------------------------------------------

    def prepare(document: Dict) -> Dict:
        document = _remap_refs(dict(document), id_map)
        return _remap_run_ids(document, run_map) if run_map else document

    for document in documents["runs"]:
        if not in_scope(document):
            continue
        # skipped runs are out of `wanted`, replaced ones were dropped and
        # remapped ones now carry a free id, so this id is available
        document = prepare(document)
        document.pop("_id", None)
        store.runs.insert_one(document)
        inserted["runs"] += 1

    for document in documents["rounds"]:
        if not in_scope(document):
            continue
        document = prepare(document)
        document.pop("_id", None)
        result = store.rounds.update_one(
            {"run_id": document["run_id"], "round": document["round"]},
            {"$setOnInsert": document},
            upsert=True,
        )
        inserted["rounds"] += 1 if result.upserted_id else 0

    # Messages keep their `_id` under a plain import, so re-importing the same
    # archive is a no-op. A remapped run is a genuinely new run, so its messages
    # need new ids or the second import would collide with the first.
    operations = []
    for document in documents["messages"]:
        if not in_scope(document):
            continue
        document = prepare(document)
        if run_map:
            document["_id"] = ObjectId()
        operations.append(
            UpdateOne({"_id": document["_id"]}, {"$setOnInsert": document}, upsert=True)
        )
    if operations:
        result = store.messages.bulk_write(operations, ordered=False)
        inserted["messages"] = result.upserted_count

    # -- did it survive the trip? --------------------------------------------

    checks = []
    if verify:
        for document in documents[DATASETS_COLLECTION]:
            local_id = id_map.get(document["_id"], document["_id"])
            local = store.datasets.find_one({"_id": local_id})
            if local is None:
                continue
            try:
                checks.append(store.verify_dataset(local_id))
            except (KeyError, ValueError) as exc:
                checks.append({"dataset_id": local_id, "ok": False, "error": str(exc)})

    return {
        "source_db": read["manifest"].get("source_db"),
        "runs": sorted(run_map.get(run_id, run_id) for run_id in wanted),
        "skipped": skipped,
        "replaced": replaced,
        "remapped": remapped,
        "inserted": inserted,
        "datasets": checks,
    }


# =============================================================================
# CLI
# =============================================================================


def _describe(manifest: Dict[str, Any]) -> None:
    print(f"  format:      {manifest['format']} v{manifest['format_version']}")
    print(f"  exported:    {manifest['exported_at']} from {manifest['source_db']}")
    if manifest.get("note"):
        print(f"  note:        {manifest['note']}")
    print()
    for run in manifest.get("runs") or []:
        config = run.get("config") or {}
        detail = ", ".join(f"{k}={v}" for k, v in sorted(config.items()))
        print(
            f"  run {run['run_id']}  {run.get('rounds', 0)} rounds, "
            f"{run.get('messages', 0)} messages, {run.get('prompt_count')} prompts"
        )
        if detail:
            print(f"    {detail}")
    print()
    for collection, stats in (manifest.get("collections") or {}).items():
        print(f"  {collection:<12} {stats['count']:>7}  {stats['sha256']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db", default=None, help="database to read or write (default: phishnet_rl)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="write runs to an archive")
    export.add_argument(
        "path", help="destination: a .tar.gz to share, or a directory to browse"
    )
    export.add_argument(
        "--run",
        type=int,
        action="append",
        dest="runs",
        help="run id to export; repeat for several, omit for the whole database",
    )
    export.add_argument("--note", help="a line recorded in the manifest for the recipient")

    inspect = commands.add_parser(
        "inspect", help="show what an archive holds, without importing it"
    )
    inspect.add_argument("path")

    load = commands.add_parser("import", help="load an archive into the database")
    load.add_argument("path")
    load.add_argument(
        "--on-conflict",
        choices=("skip", "replace", "remap"),
        default="skip",
        help="what to do with a run id this database already holds "
        "(default: skip, leaving the local run untouched)",
    )
    load.add_argument(
        "--no-verify",
        action="store_true",
        help="skip re-materialising the imported datasets to check they still hash the same",
    )

    args = parser.parse_args(argv)
    store_args = {"db_name": args.db} if args.db else {}

    if args.command == "inspect":
        try:
            manifest = read_archive(args.path)["manifest"]
        except (OSError, ValueError) as exc:
            print(f"{exc}", file=sys.stderr)
            return 1
        print()
        _describe(manifest)
        return 0

    store = LoopStore(**store_args)

    if args.command == "export":
        try:
            manifest = export_archive(store, args.path, args.runs, note=args.note)
        except KeyError as exc:
            print(f"{exc}", file=sys.stderr)
            return 1
        if not manifest["runs"]:
            print(f"{store.db.name} holds no runs; nothing exported", file=sys.stderr)
            return 1
        print(f"\nwrote {args.path}\n")
        _describe(manifest)
        print(
            "\nThe adapter weights are not in here — a checkpoint travels as its "
            "hash and file list, so the recipient can verify weights you send "
            "separately, but `--verify` will report them missing until you do."
        )
        return 0

    try:
        summary = import_archive(
            store, args.path, on_conflict=args.on_conflict, verify=not args.no_verify
        )
    except (OSError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    print(f"\nfrom {summary['source_db']} into {store.db.name}\n")
    for collection in COLLECTIONS:
        print(f"  {collection:<12} {summary['inserted'][collection]:>7} inserted")

    if summary["runs"]:
        print(f"\n  imported run(s): {', '.join(str(r) for r in summary['runs'])}")
    for original, fresh in sorted(summary["remapped"].items()):
        print(f"  run {original} already existed here; imported as {fresh}")
    for run_id in summary["replaced"]:
        print(f"  run {run_id} replaced")

    status = 0
    if summary["skipped"]:
        print(
            f"\n  run(s) {', '.join(str(r) for r in summary['skipped'])} already exist "
            "here and were skipped. Use --on-conflict remap to import them "
            "alongside, or --on-conflict replace to overwrite them.",
            file=sys.stderr,
        )

    failed = [check for check in summary["datasets"] if not check.get("ok")]
    if failed:
        print(
            f"\nWARNING: {len(failed)} of {len(summary['datasets'])} imported dataset(s) "
            "no longer resolve to the rows they were recorded against — the messages "
            "under them did not arrive intact:",
            file=sys.stderr,
        )
        for check in failed:
            print(f"  {check.get('name') or check['dataset_id']}: {check}", file=sys.stderr)
        status = 2
    elif summary["datasets"]:
        print(f"\n  {len(summary['datasets'])} dataset(s) re-verified against the "
              "imported messages: all hash as recorded")

    return status


if __name__ == "__main__":
    raise SystemExit(main())
