"""Ten-event dLLM -> AR controlled redraft/extend diagnostic, no latency claims."""
from dataclasses import asdict
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import zipfile
from .assets import ROOT, TARGET, DLLM, source_digest
from .extension_plan import read_pilot, freeze_inputs, arms
from .metrics import prefix_length
from .store import Store, atomic_json, digest


def count_stats(stats):
    return {k: v for k, v in stats.items() if k.endswith("forwards") or k in
            ("input_tokens", "unmask_steps", "hard_keep_count")}


def add_counts(a, b):
    return {k: a.get(k, 0)+b.get(k, 0) for k in set(a)|set(b)}


def audited_draft(drafter, prefix, length, anchors):
    cached = drafter.draft(prefix, length, anchors=anchors)
    full = drafter.draft(prefix, length, anchors=anchors, use_prefix_cache=False)
    a, b = asdict(cached), asdict(full)
    for result in (a, b):
        if (len(result["tokens"]) != length or len(result["order"]) != length
                or any(result["tokens"][i] != t for i, t in anchors.items())
                or result["stats"]["unmask_steps"] != length-len(anchors)
                or len(result["trace"]) != length-len(anchors)
                or result["stats"]["unmask_forwards"] != length-len(anchors)
                or any(step["mask_before"]-step["mask_after"] != 1 for step in result["trace"])):
            raise AssertionError("Hard KEEP / one-forward-one-unmask contract failed")
    if a["tokens"] != b["tokens"]:
        error = AssertionError("Cached/full-prefix dLLM tokens differ; evidence preserved")
        error.evidence = {"prefix": prefix, "length": length, "anchors": sorted(anchors.items()),
                          "cached": a, "full_prefix": b}
        raise error
    a["stats"] = count_stats(a["stats"])
    a["cached_full_audit"] = {"passed": True, "full_prefix_counts": count_stats(b["stats"]),
                               "full_prefix_tokens": b["tokens"]}
    return a


def sequential_draft(drafter, event, first):
    """The completed L tokens become an immutable prefix; only E is generated."""
    prefix, length = event["prefix"], event["L"]
    extension = audited_draft(drafter, prefix+first["tokens"], event["H"]-length, {})
    order_offset = len(first["trace"])
    trace = [{**step, "stage": "redraft"} for step in first["trace"]]
    trace.extend({**step, "step": step["step"]+order_offset,
                  "position": step["position"]+length, "stage": "extend"}
                 for step in extension["trace"])
    return {"tokens": first["tokens"]+extension["tokens"],
            "confidence": first["confidence"]+extension["confidence"],
            "order": first["order"]+[i+order_offset for i in extension["order"]],
            "trace": trace, "stats": add_counts(first["stats"], extension["stats"]),
            "stage1_reused_from": first["id"], "new_drafter_counts": extension["stats"],
            "cached_full_audit": extension["cached_full_audit"]}


def arm_metrics(event, plan, tokens, trace):
    horizon, length = plan["horizon"], event["L"]
    if len(tokens) != horizon:
        raise AssertionError("Proposal horizon changed")
    anchors = dict(plan["anchors"])
    if any(tokens[i] != value for i, value in anchors.items()):
        raise AssertionError("KEEP token changed")
    accepted = prefix_length(tokens, event["reference"])
    mismatch = accepted if accepted < horizon else None
    category = None if mismatch is None else (
        "EXTEND" if mismatch >= length else "KEEP" if mismatch in anchors else "REDRAFT")
    old_pending, earlier_extension, visible_pairs = set(range(length))-set(anchors), 0, []
    for step in trace:
        pos = step["position"]
        if pos < length:
            old_pending.discard(pos)
        elif old_pending:
            earlier_extension += 1
            visible = [i for i in sorted(old_pending) if
                (len(event["prefix"])+i-1)//32 == (len(event["prefix"])+pos)//32]
            visible_pairs.append({"extension_position": pos, "unresolved_old_visible": visible})
    correct_keep = sum(value == event["reference"][i] for i, value in anchors.items())
    return {"accepted_prefix": accepted, "accepted_old": min(accepted, length),
            "accepted_extension": max(0, accepted-length),
            "first_mismatch_position_0based": mismatch, "first_mismatch_region": category,
            "ceiling": accepted == horizon, "oracle_correct_keep_count": correct_keep,
            "false_keep_count": len(anchors)-correct_keep,
            "extension_before_old_complete_count": earlier_extension,
            "extension_visibility": visible_pairs,
            "extension_visible_unresolved_old_pairs": sum(len(x["unresolved_old_visible"]) for x in visible_pairs)}


def run_matrix(inputs, config, store, drafter, verifier, checkpoint=lambda: None, event_done=lambda: None):
    """Injectable engines also exercise interruption/resume in independent tests."""
    from .verification import checked_score
    for event in inputs["events"]:
        cache = {}
        for plan in arms(event):
            key = event["event_id"]+"|"+plan["method"]
            signature = digest([event["event_id"], plan["schedule"], plan["horizon"], plan["anchors"]])
            existing = store.get("arm", key)
            if existing:
                cache[signature] = existing
                continue
            store.put("meta", "active_arm", {"event_id": event["event_id"], "method": plan["method"]})
            previous = cache.get(signature)
            if previous:
                measured = {k: previous[k] for k in ("tokens", "confidence", "order", "trace", "stats",
                    "cached_full_audit", "verification", "metrics")}
                measured["new_drafter_counts"] = {}
            else:
                if plan["schedule"] == "H_sequential":
                    first = store.get("arm", event["event_id"]+"|"+plan["strategy"]+"_L")
                    measured = sequential_draft(drafter, event, first)
                else:
                    measured = audited_draft(drafter, event["prefix"], plan["horizon"], dict(plan["anchors"]))
                    measured["new_drafter_counts"] = measured["stats"]
                if plan["method"] == "fresh_L":
                    prior = inputs["pilot_fresh_L"][event["event_id"]]
                    if measured["tokens"] != prior["tokens"]:
                        error = AssertionError("Fresh-L differs from saved pilot under the fixed configuration")
                        error.evidence = {"pilot": prior["tokens"], "new": measured["tokens"]}
                        raise error
                    measured["pilot_baseline_replay_equal"] = True
                ref = inputs["references"][event["prompt_id"]]
                feedback = checked_score(verifier, ref, event["reference_start"], measured["tokens"], config["top_k"])
                measured["verification"] = {k: feedback[k] for k in
                    ("top1", "verification_mode", "parallel_disagreement", "replay_forwards", "replay_prefix_tokens") if k in feedback}
                measured["metrics"] = arm_metrics(event, plan, measured["tokens"], measured["trace"])
                if prefix_length(measured["tokens"], feedback["top1"]) != measured["metrics"]["accepted_prefix"]:
                    raise AssertionError("Verifier acceptance disagrees with audited metrics")
                if plan["schedule"] == "H_sequential":
                    old_accepted = first["metrics"]["accepted_prefix"]
                    if (measured["tokens"][:event["L"]] != first["tokens"]
                            or (old_accepted < event["L"] and measured["metrics"]["accepted_prefix"] != old_accepted)):
                        raise AssertionError("Sequential changed first-stage tokens/accepted prefix")
            payload = {"id": key, "event_id": event["event_id"], "prompt_id": event["prompt_id"],
                       "dataset": event["dataset"], "cohort": event["cohort"], "source_drafter": "dllm_top1",
                       "redrafter": "dllm_top1", "verifier": TARGET, "suffix_length": event["L"],
                       "extension_length": plan["horizon"]-event["L"], "physical_phase": event["physical_phase"],
                       "old_endpoint_phase8": event["old_endpoint_phase8"],
                       "reference_token_ids": event["reference"][:plan["horizon"]],
                       **plan, **measured, "alias_of": previous["id"] if previous else None}
            store.put("arm", key, payload)
            cache[signature] = payload
            print("mini", event["event_id"], plan["method"],
                  "accepted=", payload["metrics"]["accepted_prefix"], "/", plan["horizon"],
                  "alias" if previous else "verified", flush=True)
            checkpoint()
        gc.collect()
        event_done()
    store.put("meta", "complete", True)
    store.put("meta", "active_arm", None)


def prepare_models():
    from huggingface_hub import snapshot_download
    lock = json.loads((ROOT/"source-lock.json").read_text(encoding="utf-8"))
    paths = {}
    for name in (TARGET, DLLM):
        print("Preparing", name, lock["models"][name], flush=True)
        paths[name] = snapshot_download(name, revision=lock["models"][name],
            ignore_patterns=["*.bin", "*.pt", "*.ot", "*.msgpack", "onnx/*"])
    for file, expected in lock["source_sha256"].items():
        if hashlib.sha256((Path(paths[DLLM])/file).read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned dLLM source changed: "+file)
    return paths, {name: lock["models"][name] for name in paths}


def pack_extension(output, store=None):
    output = Path(output)
    if store:
        store.backup(output/"checkpoint.sqlite")
        store.export("arm", output/"arms.jsonl")
    path, temp = output/"extension_checkpoint.zip", output/"extension_checkpoint.tmp"
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob("*")):
            if (p.is_file() and p not in (path, temp)
                    and not p.name.startswith("extension.sqlite")):
                z.write(p, p.relative_to(output).as_posix())
    temp.replace(path)
    return path


def execute(checkpoint_root, output, revision):
    from .extension_report import report
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/"configs/mini_extension.json").read_text())
    store = target = model = verifier = drafter = None
    started, status = time.monotonic(), "failed"
    try:
        print("[mini] Runtime started; importing checkpoint before model loading", flush=True)
        pilot = read_pilot(checkpoint_root)
        print("[mini] Selecting 10 events and freezing anchors", flush=True)
        inputs = freeze_inputs(pilot, config)
        del pilot
        if len(inputs["events"]) != config["events"]:
            raise ValueError("Frozen event count differs from the configured test")
        identity = {"protocol": "dllm-ar-anchored-extension-v1", "config": config,
                    "pilot_checkpoint_sha256": inputs["pilot_checkpoint_sha256"],
                    "plan_digest": inputs["plan_digest"], "code_revision": revision,
                    "source_digest": source_digest()}
        store = Store(output/"extension.sqlite", identity)
        atomic_json(output/"inputs.json", inputs)
        atomic_json(output/"extension-manifest.json", identity)
        if store.get("meta", "complete"):
            report(output, inputs, store)
            status = "complete"
            return
        print("Frozen mini-test:", len(inputs["events"]), "distinct prompts, dLLM source/redrafter only", flush=True)
        print("Skipping pilot references/Test1/Test2; replay Fresh-L for trace and baseline audit", flush=True)
        pack_extension(output, store)
        paths, revisions = prepare_models()
        from .runtime import configure_numerics, tokenizers, load_target, load_model, ARSession, DLLMDrafter, target_consistency, release
        from .experiment import environment
        from .migration import validate_identity
        import torch
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Requires GPU T4 x2")
        torch.manual_seed(config["seed"])
        configure_numerics(config)
        tokenizer, evidence = tokenizers(paths)
        target, _, _ = load_target(paths[TARGET], config)
        # No latency experiment here: avoid CUDA-stage timing instrumentation.
        target._suffix_exclude_peer_transfers = False
        verifier = ARSession(target)
        env = environment()
        if env["commit"] != revision:
            raise ValueError("Runtime Git revision differs from launcher")
        env.update(target_device_map={k: str(v) for k, v in target.hf_device_map.items()},
                   target_attention=config["target_attention"],
                   allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        prior = inputs["pilot_manifest"]
        validate_identity(prior, {**prior, "tokenizer": evidence, "environment": env})
        runtime_identity = {"environment": env, "tokenizer": evidence, "models": revisions}
        old_runtime = store.get("meta", "runtime")
        if old_runtime and digest(old_runtime) != digest(runtime_identity):
            raise ValueError("Mini-test resume numerical environment differs")
        store.put("meta", "runtime", runtime_identity)
        atomic_json(output/"runtime.json", runtime_identity)
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": "Compute 17 + 28. Explain briefly."}],
                                               tokenize=True, add_generation_prompt=True)
        audit = target_consistency(verifier, prefix)
        atomic_json(output/"preflight.json", {"target_audit": audit,
            "dllm_policy": "cached/full equality checked for every unique generated stage; strict one MASK per forward"})
        if not audit["passed"]:
            raise AssertionError("Fixed-FP16 target repeatability/parallel/causal preflight failed")
        model = load_model(paths[DLLM], "cuda:1")
        drafter = DLLMDrafter(model, config)
        run_matrix(inputs, config, store, drafter, verifier,
                   lambda: pack_extension(output, store), lambda: release(None))
        report(output, inputs, store, tokenizer)
        status = "complete"
    except BaseException as error:
        evidence = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                    "evidence": getattr(error, "evidence", None)}
        if store:
            evidence["active_arm"] = store.get("meta", "active_arm")
            store.put("failure_history", str(time.time_ns()), evidence)
        atomic_json(output/"failure.json", evidence)
        raise
    finally:
        try:
            # A rendering failure must not prevent packing the raw checkpoint.
            if store and status != "complete":
                try:
                    report(output, inputs, store)
                except Exception as error:
                    atomic_json(output/"report_failure.json", {"message": str(error)})
        finally:
            atomic_json(output/"execution_status.json", {"status": status, "code_revision": revision,
                "completed_arms": len(store.rows("arm")) if store else 0, "expected_arms": config["events"]*9,
                "operational_seconds": time.monotonic()-started,
                "latency_study": False, "verification_policy": "offline reference-audited actual model; no online SD claim"})
            archive = pack_extension(output, store)
            if store:
                store.close()
            drafter = verifier = model = target = None
            if "torch" in locals():
                from .runtime import release
                release(None)
            print(status.upper()+": mini-extension", "ZIP:", archive, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    execute(args.checkpoint_root, args.output, args.revision)


if __name__ == "__main__":
    main()
