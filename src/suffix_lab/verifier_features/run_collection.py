"""Full collection with mandatory smoke, pilot milestone and immutable resume."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import traceback
from suffix_lab.assets import ROOT, source_digest
from suffix_lab.store import atomic_json, digest
from .checkpoint_loader import discover, PINNED_REVISION
from .prefix_reconstruction import reconstruct
from .offline_labels import prepare_labels, write_labels
from .target_model import Target
from .collect_old_features import collect as collect_old
from .collect_corrected_features import collect as collect_corrected
from .validate import validate_values
from .shards import Shards, restore
from .make_report import make_report


def select_order(events, labels, smoke_count):
    """Oracle-enriched smoke only; label fields never enter extraction objects."""
    ordered = sorted(events, key=lambda e: digest([42, e["id"]]))
    selected = []
    predicates = [lambda e: e["guard"]["numerical_disagreements"],
                  lambda e: not labels[e["id"]]["runs"],
                  lambda e: labels[e["id"]]["C0"],
                  lambda e: any(labels[e["id"]]["runs"][b["left"]]["delta"] != 0
                                and labels[e["id"]]["runs"][b["right"]]["delta"] != 0
                                for b in labels[e["id"]]["bridge_edges"]),
                  lambda e: len(e["proposal"]) < 64,
                  lambda e: e["accepted"] == 0,
                  lambda e: e["accepted"] >= 32,
                  lambda e: len(set(e["source"])) < len(e["source"])/2]
    for predicate in predicates:
        for ds in ("gsm8k", "math500"):
            candidate = next((e for e in ordered if e["dataset"] == ds and predicate(e) and e not in selected), None)
            if candidate is not None and len(selected) < smoke_count:
                selected.append(candidate)
    for e in ordered:
        if len(selected) >= smoke_count:
            break
        if e not in selected:
            selected.append(e)
    # Round-robin dataset / acceptance bins for the pilot and remaining full run.
    buckets = {}
    for e in ordered:
        if e not in selected:
            key = (e["dataset"], min(e["accepted"]//8, 7))
            buckets.setdefault(key, []).append(e)
    tail = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                tail.append(buckets[key].pop())
    return selected+tail


def event_nonzero_runs(label):
    return [r for r in label["runs"] if r["delta"] != 0]


def collect_one(target, event, config, hook_audit=False, eos=()):
    """Reference-free boundary: no label/reference argument or module reads."""
    import torch
    started = time.perf_counter()
    old, old_tensors, check, old_time = collect_old(target, event, config["top_k"], hook_audit)
    corrected, corrected_tensors, corrected_time = collect_corrected(
        target, event, config["top_k"], config["micro_probe_steps"], eos)
    tensors = {**old_tensors, **corrected_tensors}
    for layer in config["layers"]:
        vector = tensors[f"corrected_layer_{layer}"].float()
        comparison = torch.nn.functional.cosine_similarity(tensors[f"old_layer_{layer}"].float(), vector, dim=-1)
        tensors[f"cosine_pre_layer_{layer}"] = comparison[:-1].clone().contiguous()
        tensors[f"cosine_post_layer_{layer}"] = comparison[1:].clone().contiguous()
    validate_values(old, corrected, tensors, event, config["layers"], target.model.config.hidden_size)
    prompt_fold = int(digest(event["prompt_id"])[:8], 16) % 10
    values = {"id": event["id"], "prompt_id": event["prompt_id"], "dataset": event["dataset"],
              "prompt_split": "test" if prompt_fold == 0 else "validation" if prompt_fold == 1 else "train",
              "offset": event["offset"], "prefix_length": len(event["prefix"]),
              "prefix_digest": digest(event["prefix"]), "accepted": event["accepted"],
              "correction": event["correction"], "proposal_ids": event["proposal"],
              "correction_emitted_within_output_budget": event["correction_emitted_within_output_budget"],
              "source_ids": event["source"], "source_proposal_start": event["accepted"]+1,
              "source_absolute_start": len(event["prefix"])+event["accepted"]+1,
              "old": old, "corrected": corrected,
              "drafter": {k: event["draft"][k] for k in ("tokens", "confidence", "order", "trace")}}
    return {"event": values, "check": check, "timing": {"old": old_time, "corrected": corrected_time,
            "event_wall_ms_including_transfers_io_excluded": (time.perf_counter()-started)*1000}}, tensors


def validate_config(c):
    if (not c["layers"] or len(set(c["layers"])) != len(c["layers"])
            or any(type(i) is not int or not 0 <= i <= 27 for i in c["layers"])
            or c["top_k"] not in (16, 32) or c["micro_probe_steps"] not in (0, 1, 2, 3)
            or not 1 <= c["shard_events"] <= 64 or c["smoke_events"] < 1):
        raise ValueError("Invalid feature package")


def execute(cp, output, revision, config, *, resume_root=None, limit=None, budget_seconds=21600, target_factory=None):
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume_root:
        restore(resume_root, output)
    if (cp.manifest["code_revision"] != PINNED_REVISION or not cp.preflight["passed"]
            or cp.manifest["config"]["verification_mode"] != "canonical_guarded"
            or len(cp.prompts) != config["expected_prompts"] or len(cp.rounds) != config["expected_rounds"]
            or cp.identity != config["expected_input_identity"]):
        raise ValueError("Input checkpoint identity/count/mode differs from the audited completed run")
    print("[input] reconstructing all committed prefixes; no drafter inference", flush=True)
    eos = set(cp.runtime["effective_eos_ids"])
    events, excluded = reconstruct(cp.prompts, cp.rounds, eos)
    print("[labels] independent CPU reference audit + all alignments/bridge edges", flush=True)
    labels = prepare_labels(cp, events, eos)
    valid = [e for e in events if not labels[e["id"]]["reference_exhausted"]]
    excluded += [{"id": e["id"], "reason": "target_content_exhausted"} for e in events if e not in valid]
    if len(valid) != config["expected_events"]:
        raise ValueError("Eligible event count differs")
    events = select_order(valid, labels, config["smoke_events"])
    plan = {"all_event_ids": [e["id"] for e in events],
            "smoke_ids": [e["id"] for e in events[:config["smoke_events"]]],
            "pilot_ids": [e["id"] for e in events[:config["pilot_events"]]],
            "selection": "seed42; oracle-enriched smoke, dataset/acceptance-bin pilot; full includes all eligible",
            "excluded": excluded}
    completed_bytes = sum(p.stat().st_size for p in (output/"shards").glob("part-*.zip"))
    if shutil.disk_usage(output).free < max(0, config["minimum_free_disk_gib"]*2**30-completed_bytes):
        raise OSError("Insufficient working disk; need room for tensor shards and the final combined ZIP")
    requested = {"schema": config["schema_version"], "input_identity": cp.identity,
                 "code_revision": revision, "source_digest": source_digest(),
                 "config": config, "event_plan_digest": digest(plan)}
    manifest_path = output/"feature-manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous["request"] != requested:
        raise ValueError("Resume config/code/input differs; existing files preserved")
    atomic_json(output/"event-plan.json", plan)
    write_labels(output/"offline_labels.parquet", {e["id"]: labels[e["id"]] for e in events})
    target = (target_factory or (lambda: Target.load(cp, config["layers"])))()
    environment = target.environment()
    if previous and previous["environment"] != environment:
        raise ValueError("Resume numerical environment differs")
    identity = {"fingerprint": digest([requested, environment]), "schema": config["schema_version"]}
    manifest = {"request": requested, "environment": environment, "identity": identity,
                "input_hashes": cp.hashes, "parent_manifest": cp.manifest,
                "features": {"layer_ids": "zero-based decoder outputs before final norm",
                             "hidden_union": "slots i / i+1 = before predicting / after reading source[i]",
                             "M_off": "source i versus old-path prediction j; boundary column L included",
                             "q0": "next token after correction only", "rank_ties": "1 + count strictly greater logits",
                             "storage": "safetensors inside immutable ZIP shards; FP16 hidden, FP32 scores",
                             "reference_access": "only offline_labels and replay_checks; excluded from collect_one"}}
    atomic_json(manifest_path, manifest)
    shards = Shards(output, identity)
    if not set(shards.records).issubset(set(plan["all_event_ids"])):
        raise ValueError("Unexpected completed IDs")
    started, status = time.monotonic(), "failed"
    selected = events[:limit] if limit else events
    try:
        for index, event in enumerate(selected):
            if time.monotonic()-started >= budget_seconds:
                status = "budget_checkpoint"
                break
            if event["id"] not in shards.records:
                print(f"[collect {index+1}/{len(selected)}] {event['id']} a={event['accepted']} "
                      f"source={len(event['source'])}; old + corrected forwards", flush=True)
                record, tensors = collect_one(target, event, config,
                    hook_audit=index < config["smoke_events"], eos=eos)
                # Evaluation-only reference access starts AFTER extraction.
                expected = labels[event["id"]]["expected_corrected_next"]
                record["check"].update(corrected_next_reference=expected,
                    corrected_next_matches_reference=record["event"]["corrected"]["top1"][0] == expected,
                    quarantined=not record["check"]["old_top1_matches"])
                shards.add(record, tensors)
            milestone = index+1 in (config["smoke_events"], config["pilot_events"], len(selected))
            if len(shards.pending) >= config["shard_events"] or milestone:
                shards.flush()
                report = make_report(output, shards.records, len(events), "running", target.gpu_memory())
                remaining = report["estimated_remaining_inference_seconds"]
                print(f"[progress] saved={len(shards.records)}/{len(events)}; "
                      f"inference remaining~{remaining/60:.1f} min; packing/setup extra", flush=True)
                gc.collect()
                import torch
                for gpu in range(torch.cuda.device_count()):
                    with torch.cuda.device(gpu):
                        torch.cuda.empty_cache()
            if index+1 == config["smoke_events"]:
                smoke = [shards.records[k]["check"] for k in plan["smoke_ids"]]
                bad = sum(not r["old_top1_matches"] for r in smoke)
                if bad > config["smoke_max_old_replay_mismatches"]:
                    raise AssertionError("Smoke replay mismatch: evidence quarantined; full collection not started")
                atomic_json(output/"smoke-validation.json", {"passed": True, "events": len(smoke),
                    "old_replay_mismatches": bad, "hook_audits": len(smoke),
                    "note": "corrected cold/cached differences recorded separately"})
                print("[smoke PASS] continuing pilot then full with the same feature package", flush=True)
        else:
            status = "complete" if len(selected) == len(events) else "partial_requested"
    except BaseException:
        atomic_json(output/"failure.json", {"event_id": event["id"] if "event" in locals() else None,
                                           "traceback": traceback.format_exc()})
        raise
    finally:
        shards.flush()
        if status == "complete" and any(not r["check"]["old_top1_matches"] for r in shards.records.values()):
            status = "complete_with_quarantine"
        report = make_report(output, shards.records, len(events), status, target.gpu_memory())
        atomic_json(output/"execution_status.json", {**report,
                    "operational_seconds_this_session": time.monotonic()-started})
        try:
            print("[archive]", shards.archive(), flush=True)
        except OSError as exc:
            atomic_json(output/"archive_warning.json", {"error": str(exc), "shards_preserved": True})
            print("[archive warning]", exc, flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--resume-root")
    p.add_argument("--limit", type=int)
    p.add_argument("--budget-seconds", type=float, default=21600)
    p.add_argument("--micro-probe-steps", type=int, choices=(0, 1, 2, 3), default=0)
    p.add_argument("--top-k", type=int, choices=(16, 32), default=16)
    p.add_argument("--layers", type=int, nargs="+")
    args = p.parse_args()
    if args.limit is not None and args.limit < 20 or args.budget_seconds <= 0:
        p.error("limit >=20; budget must be positive")
    config = json.loads((ROOT/"configs/verifier_features.json").read_text())
    config.update(top_k=args.top_k, micro_probe_steps=args.micro_probe_steps)
    if args.layers is not None:
        config["layers"] = args.layers
    execute(discover(args.input_root), args.output, args.revision, config,
            resume_root=args.resume_root, limit=args.limit, budget_seconds=args.budget_seconds)


if __name__ == "__main__":
    main()
