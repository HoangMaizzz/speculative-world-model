"""Paired oracle shift-anchor B64 test on the completed guarded B64 cohort."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import random
from pathlib import Path
import statistics
import time
import zipfile

from .assets import ROOT, source_digest
from .long60 import clear_unused, load_bundle
from .long60_plan import decide, matching_runs, monotone_chain, until_eos
from .metrics import prefix_length
from .store import Store, atomic_json, digest

INPUT_SHA = "fb7e1ea13a0b6f6fce9f20d22c5270ab2c54e70cf83c59f7b95b31b44cacafca"
ARMS = ("fresh64", "positional64", "shifted64")


def input_zip(root):
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob("long64_checkpoint.zip"))
    matches = [p for p in paths if hashlib.sha256(p.read_bytes()).hexdigest() == INPUT_SHA]
    if len(matches) != 1:
        raise ValueError("Attach exactly one copy of the completed 200-prompt guarded long64_checkpoint.zip")
    return matches[0]


def make_plan(path):
    with zipfile.ZipFile(path) as archive:
        def rows(name):
            with archive.open(name + ".jsonl") as stream:
                return [json.loads(line) for line in stream]
        runtime = json.loads(archive.read("runtime.json"))
        status = json.loads(archive.read("execution_status.json"))
        prompts, refs, rounds = rows("prompts"), rows("references"), rows("rounds")
    eos = set(runtime["effective_eos_ids"])
    references = {r["id"]: r for r in refs}
    grouped = defaultdict(list)
    for row in rounds:
        grouped[row["prompt_id"]].append(row)
    if (status["status"] != "complete" or len(rounds) != 4050
            or Counter(p["dataset"] for p in prompts) != {"math500": 100, "gsm8k": 100}
            or len(references) != 200):
        raise ValueError("Unexpected guarded source cohort")
    plan = []
    for prompt in prompts:
        key, ids = prompt["id"], prompt["prompt_ids"]
        ref = references[key]
        gold = ref["tokens"]
        if ids != ref["prompt_ids"] or gold[:len(prompt["tokens"])] != prompt["tokens"]:
            raise ValueError(f"Source/reference identity differs for {key}")
        candidates = []
        for row in sorted(grouped[key], key=lambda r: r["round_index"]):
            d, offset = row["decision"], row["offset"]
            a = d["first_rejection"]
            if a is None or d["terminated"] or len(d["committed_tokens"]) != a+1:
                continue
            start = offset+a+1
            old = until_eos(row["proposal"][a+1:], eos, include=False)
            future = gold[start:start+64]
            if not old or not future or (len(future) != 64 and future[-1] not in eos):
                continue
            prefix = ids+gold[:offset]
            if (digest(prefix) != row["prefix_digest"] or gold[offset:start] != d["committed_tokens"]
                    or row["proposal"][:a] != gold[offset:start-1]
                    or row["proposal"][a] == gold[start-1]):
                raise ValueError(f"Invalid source rejection {row['id']}")
            content = until_eos(future, eos, include=False)
            runs = monotone_chain(matching_runs(old, content, radius=63, minimum=4))
            shifted = {j: old[i] for run in runs
                       for i, j in zip(range(run["source_start"], run["source_end"]),
                                       range(run["destination_start"], run["destination_end"]))}
            positional = {i: t for i, t in enumerate(old[:len(content)]) if t == content[i]}
            candidates.append({"source_id": row["id"], "start": start, "ids": ids,
                               "committed": gold[:start], "future": future, "old": old,
                               "runs": runs, "anchors": {"fresh64": [], "positional64": sorted(positional.items()),
                                                        "shifted64": sorted(shifted.items())}})
        # Frozen earliest eligible event, without selecting on anchor availability or arm results.
        plan.append({"prompt_id": key, "dataset": prompt["dataset"], "eligible_rejections": len(candidates),
                     "event": candidates[0] if candidates else None})
    return plan, runtime


def run_arm(bundle, event, arm, audit=False):
    from .extension import audited_draft
    prefix = event["ids"]+event["committed"]
    anchors = {int(i): int(t) for i, t in event["anchors"][arm]}
    if any(event["future"][i] != t for i, t in anchors.items()):
        raise AssertionError("Oracle anchor is not reference-correct")
    draft = (audited_draft(bundle.drafter, prefix, 64, anchors) if audit else
             asdict(bundle.drafter.draft(prefix, 64, anchors=anchors)))
    mutable = 64-len(anchors)
    positions = {step["position"] for step in draft["trace"]}
    if (len(draft["tokens"]) != 64 or len(draft["trace"]) != mutable
            or draft["stats"]["unmask_steps"] != mutable or draft["stats"]["unmask_forwards"] != mutable
            or positions != set(range(64))-anchors.keys()
            or any(draft["tokens"][i] != t for i, t in anchors.items())
            or any(step["mask_before"]-step["mask_after"] != 1
                   or step["absolute_position"] != len(prefix)+step["position"]
                   or step["token"] != draft["tokens"][step["position"]] for step in draft["trace"])):
        raise AssertionError("Hard anchors or native one-MASK schedule violated")
    proposal = until_eos(draft["tokens"], bundle.eos)
    bundle.verifier.begin_prompt(event["ids"], event["committed"])
    feedback = bundle.verifier.score(prefix, proposal)
    decision = decide(proposal, feedback["top1_with_bonus"], bundle.eos, 65)
    a = decision["accepted_draft"]
    expected = prefix_length(proposal, event["future"])
    future = event["future"]
    mutable_correct = sum(i not in anchors and t == reference
                          for i, (t, reference) in enumerate(zip(proposal, future)))
    accepted_eos = bool(a and proposal[a-1] in bundle.eos)
    return {"accepted": a, "reference_accepted": expected,
            "accepted_content": a-int(accepted_eos),
            "agreement": a == expected and (accepted_eos or a == len(future)
                                              or decision["committed_tokens"][-1] == future[a]),
            "logical_length": len(proposal), "anchor_count": len(anchors),
            "accepted_anchors": sum(i < a for i in anchors),
            "mutable_correct": mutable_correct, "mutable_reference_positions": len(future)-len(anchors),
            "accepted_eos": accepted_eos,
            "draft": draft, "verification": feedback, "decision": decision}


def summarize(plan, pairs):
    report = []
    for dataset in ("all", "math500", "gsm8k"):
        planned = [p for p in plan if dataset == "all" or p["dataset"] == dataset]
        for shifted_only in (False, True):
            subset = [p for p in planned if p["event"] is not None
                      and (not shifted_only or p["event"]["anchors"]["shifted64"])]
            completed = [pairs[p["prompt_id"]] for p in subset if p["prompt_id"] in pairs]
            row = {"dataset": dataset, "stratum": "has_shifted_anchor" if shifted_only else "all_eligible",
                   "cohort_prompts": len(planned), "planned_pairs": len(subset), "completed_pairs": len(completed),
                   "reference_shorter_than_64": sum(len(p["event"]["future"]) < 64 for p in subset)}
            if completed:
                row["mean_accepted"] = {arm: statistics.mean(r[arm]["accepted"] for r in completed) for arm in ARMS}
                row["mean_unmask_forwards"] = {
                    arm: statistics.mean(r[arm]["draft"]["stats"]["unmask_forwards"] for r in completed)
                    for arm in ARMS}
                row["mean_accepted_content"] = {
                    arm: statistics.mean(r[arm]["accepted_content"] for r in completed) for arm in ARMS}
                for base in ("fresh64", "positional64"):
                    delta = [r["shifted64"]["accepted"]-r[base]["accepted"] for r in completed]
                    rng = random.Random(42)
                    draws = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(2000))
                    row["shifted_minus_"+base] = {"mean": statistics.mean(delta), "median": statistics.median(delta),
                        "wins_ties_losses": [sum(x > 0 for x in delta), delta.count(0), sum(x < 0 for x in delta)],
                        "prompt_bootstrap_95": [draws[50], draws[1949]]}
            report.append(row)
    return report


def publish(output, store, plan, status):
    store.export("arm", output/"arms.jsonl")
    pairs = defaultdict(dict)
    for row in store.rows("arm"):
        if row["result"]["agreement"]:
            pairs[row["prompt_id"]][row["arm"]] = row["result"]
    complete = {k: v for k, v in pairs.items() if len(v) == len(ARMS)}
    atomic_json(output/"summary.json", summarize(plan, complete))
    atomic_json(output/"execution_status.json", {"status": status, "completed_pairs": len(complete)})
    store.backup(output/"checkpoint_snapshot.sqlite")
    temp = output/"shift64_checkpoint.tmp"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.iterdir()):
            if p.is_file() and (p.suffix in {".json", ".jsonl"} or p.name == "checkpoint_snapshot.sqlite"):
                archive.write(p, p.name)
    temp.replace(output/"shift64_checkpoint.zip")


def execute(input_root, output, revision, limit, resume=None):
    source = input_zip(input_root)
    plan, source_runtime = make_plan(source)
    if limit is not None:
        plan = [p for p in plan if p["dataset"] == "gsm8k"][:limit]+[p for p in plan if p["dataset"] == "math500"][:limit]
    # Interleave benchmarks so a partial run can report both.
    plan.sort(key=lambda p: (int(p["prompt_id"].split(":")[-1]), p["dataset"]))
    c = json.loads((ROOT/"configs/long64.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    identity = {"input_sha256": INPUT_SHA, "revision": revision, "source_digest": source_digest(),
                "config": c, "plan": digest(plan), "limit": limit}
    db = output/"checkpoint.sqlite"
    if resume:
        if db.exists():
            raise ValueError("Use local resume or an archive, not both")
        with zipfile.ZipFile(resume) as archive:
            if json.loads(archive.read("identity.json")) != identity:
                raise ValueError("Resume archive identity differs")
            db.write_bytes(archive.read("checkpoint_snapshot.sqlite"))
            (output/"runtime.json").write_bytes(archive.read("runtime.json"))
    store = Store(db, identity)
    atomic_json(output/"identity.json", identity)
    atomic_json(output/"plan.json", plan)
    status = "failed"
    started = time.monotonic()
    try:
        bundle = load_bundle(c, output)
        runtime = json.loads((output/"runtime.json").read_text())
        for key in ("models", "tokenizer", "effective_eos_ids", "target_device_map", "target_precision"):
            if runtime[key] != source_runtime[key]:
                raise ValueError(f"Source model/runtime mismatch: {key}")
        audited = False
        for p in plan:
            event = p["event"]
            if event is None:
                continue
            for arm in ARMS:
                key = p["prompt_id"]+"|"+arm
                existing = store.get("arm", key)
                if existing is not None:
                    if not existing["result"]["agreement"]:
                        raise AssertionError(f"Saved verifier mismatch: {key}")
                    continue
                if time.monotonic()-started >= c["soft_budget_seconds"]:
                    status = "budget_checkpoint"
                    return
                anchors = event["anchors"][arm]
                check = arm == "shifted64" and bool(anchors) and not audited
                result = run_arm(bundle, event, arm, audit=check)
                audited |= check
                store.put("arm", key, {"prompt_id": p["prompt_id"], "source_id": event["source_id"],
                                       "dataset": p["dataset"], "arm": arm, "result": result})
                print(f"{key} source={event['source_id']} anchors={len(anchors)} A={result['accepted']}", flush=True)
                if not result["agreement"]:
                    raise AssertionError(f"Canonical/reference acceptance differs: {key}")
                clear_unused()
            if (len(store.rows("arm")) // len(ARMS)) % 10 == 0:
                publish(output, store, plan, "running")
        status = "complete"
    except Exception as exc:
        atomic_json(output/"failure.json", {"type": type(exc).__name__, "message": str(exc),
                                           "evidence": getattr(exc, "evidence", None)})
        raise
    finally:
        try:
            publish(output, store, plan, status)
        finally:
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--limit", type=int, choices=range(1, 101))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    execute(args.input_root, args.output, args.revision, args.limit, args.resume)


if __name__ == "__main__":
    main()
