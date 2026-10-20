"""Explicit, audited import of the failed v4 pilot; input files stay read-only."""
import json
from pathlib import Path
import sqlite3
from .metrics import suffix_metrics, prefix_length
from .store import Store, atomic_json, digest

OLD_REVISION = "bdcdfc3335e590fe1c1283273b2fd0cf3d16befe"
OLD_SOURCE = "ffb5fa2240879417663317b9c4e7a345656247882464396e7b1902ed991e2689"


def locate_unpacked(root):
    root = Path(root).resolve()
    candidates = []
    for manifest in root.rglob("run-manifest.json"):
        folder = manifest.parent.resolve()
        if not folder.is_relative_to(root):
            continue
        db = folder/"checkpoint.sqlite"
        if db.is_file() and db.resolve().is_relative_to(root):
            candidates.append(folder)
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one unpacked checkpoint beneath {root}; found {len(candidates)}")
    return candidates[0]


def import_checkpoint(root, output, config, revision):
    folder, output = locate_unpacked(root), Path(output)
    if (output/"run.sqlite").exists():
        raise ValueError("Import requires an output without a live database")
    identity = json.loads((folder/"run-manifest.json").read_text(encoding="utf-8"))
    if identity["config"] != config:
        raise ValueError("Checkpoint config differs from this run")
    db = folder/"checkpoint.sqlite"
    conn = sqlite3.connect(db.as_uri()+"?mode=ro&immutable=1", uri=True)
    try:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Checkpoint SQLite integrity failed")
        records = {(k, key): json.loads(v) for k, key, v in conn.execute("SELECT kind,key,payload FROM records")}
    finally:
        conn.close()
    if records.get(("meta", "identity")) != identity:
        raise ValueError("Checkpoint manifest/DB identity mismatch")
    current = identity["environment"]["commit"] == revision
    if not current and (identity["environment"]["commit"] != OLD_REVISION or identity["source_digest"] != OLD_SOURCE
                        or identity["protocol"] != "suffix-top1-v4-two-cohort-repairability" or identity["target_precision"] != "fp16"):
        raise ValueError("Only the audited v4 FP16 checkpoint or this exact runtime can be imported")
    if not current and (any(k in ("test2", "repair_plan", "event_complete") for k, key in records)
                        or not records.get(("meta", "references_complete"))):
        raise ValueError("Legacy migration requires completed references and no Test2 records")
    refs = {key: v for (kind,key), v in records.items() if kind == "reference"}
    for (kind,key), row in records.items():
        if kind != "test1":
            continue
        gold = refs[row["prompt_id"]]["tokens"][row["offset"]:row["offset"]+32]
        feedback, proposal = row["feedback"], row["draft"]["tokens"]
        accepted = prefix_length(proposal, feedback["top1"])
        if (row["reference"] != gold or row["metrics"] != suffix_metrics(proposal, gold, feedback["top1"])
                or accepted != row["metrics"]["accepted_prefix"]
                or (accepted < 32 and feedback["top1"][accepted] != gold[accepted])):
            raise ValueError("Imported Test1 consistency failed at "+key)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output/"importing.sqlite"
    if temporary.exists():
        raise ValueError("Incomplete previous import exists; preserve it and choose another output")
    store = Store(temporary)
    try:
        if current:
            for (kind,key), value in records.items():
                store.put(kind,key,value)
        else:
            for (kind,key), value in records.items():
                if kind in ("reference", "test1"):
                    store.put(kind,key,value)
                elif kind == "failure":
                    store.put("prior_failure",key,value)
            store.put("meta", "references_complete", True)
            store.put("meta", "migration_pending", identity)
            store.put("meta", "import_provenance", {"old_identity_digest": digest(identity),
                "old_revision": OLD_REVISION, "references": len(refs),
                "test1": sum(kind == "test1" for kind,key in records), "policy": "keep only audited passing rows; replay archived failure before continuation"})
    finally:
        store.close()
    temporary.replace(output/"run.sqlite")
    if current:
        atomic_json(output/"run-manifest.json", identity)
    else:
        atomic_json(output/"imported-run-manifest.json", identity)
    print("Imported checkpoint:", len(refs), "references;", sum(kind == "test1" for kind,key in records), "Test1 rows", flush=True)


def validate_identity(old, new):
    for key in ("config", "asset_digest", "target_precision", "tokenizer"):
        if old[key] != new[key]:
            raise ValueError("Migrated checkpoint differs: "+key)
    for key in ("cuda", "gpus", "target_device_map", "target_attention", "allow_fp16_reduced_precision_reduction"):
        if old["environment"].get(key) != new["environment"].get(key):
            raise ValueError("Migrated numerical environment differs: "+key)
    for key in ("torch", "transformers", "accelerate"):
        if old["environment"]["packages"].get(key) != new["environment"]["packages"].get(key):
            raise ValueError("Migrated package differs: "+key)


def replay_prior_failures(session, store, top_k):
    from .verification import checked_score, VerificationError
    evidence = []
    for key, payload in store.db.execute("SELECT key,payload FROM records WHERE kind='prior_failure' ORDER BY key").fetchall():
        row = json.loads(payload)
        dataset, index, offset, source = key.split(":")
        ref = store.get("reference", dataset+":"+index)
        try:
            score = checked_score(session, ref, int(offset), row["draft"], top_k)
        except VerificationError as error:
            store.put("failure", "migration:"+key, error.evidence)
            raise
        store.put("migration_replay", key, score)
        print("Migration replay passed:", key, score["verification_mode"], flush=True)
        evidence.append({"id": key, "mode": score["verification_mode"], "passed": True})
    return evidence
