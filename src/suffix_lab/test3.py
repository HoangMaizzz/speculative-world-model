"""Test 3: a new B24 draft, one correction, all-correct KEEP, refill to 23."""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time
import traceback
import zipfile
from .assets import ROOT, TARGET, DLLM, source_digest
from .extension import audited_draft, sequential_draft, count_stats, arm_metrics, prepare_models
from .extension_plan import read_pilot
from .metrics import prefix_length
from .store import Store, atomic_json, digest
from .test3_plan import context_pool, corrected_event, category, select_events, arm_plans, conditioning_metrics, visible_trace
from .verification import checked_score


class SelectionUnavailable(RuntimeError):
    pass


def feedback_record(feedback):
    return {k: feedback[k] for k in ("top1", "verification_mode", "parallel_disagreement",
            "replay_forwards", "replay_prefix_tokens") if k in feedback}


def single_draft(drafter, prefix, length):
    """Screening is cached generation at the ACTUAL requested horizon, not B32."""
    result = asdict(drafter.draft(prefix, length, anchors={}))
    if (len(result["tokens"]) != length or len(result["trace"]) != length
            or result["stats"]["unmask_steps"] != length
            or result["stats"]["unmask_forwards"] != length
            or any(s["mask_before"]-s["mask_after"] != 1 for s in result["trace"])):
        raise AssertionError("Screening violated strict one-token unmasking")
    result["stats"] = count_stats(result["stats"])
    return result


def availability(screens, config):
    rows = list(screens)
    eligible = [s for s in rows if s["status"] == "eligible"]
    return {"screened_states": len(rows), "eligible_states": len(eligible),
            "category_counts": dict(Counter(s["category"] for s in eligible)),
            "skip_reasons": dict(Counter(s["reason"] for s in rows if s["status"] != "eligible")),
            "required": config["category_quotas"], "required_gsm8k": config["gsm8k_events"],
            "distinct_eligible_prompts": len({s["event"]["prompt_id"] for s in eligible})}


def freeze_selection(store, screens, chosen, config):
    lookup = {s["event"]["event_id"]: s for s in screens if s["status"] == "eligible"}
    events = [{**lookup[key]["event"], "category": lookup[key]["category"],
               "screen_index": lookup[key]["screen_index"], "fresh23_screen": lookup[key]["fresh23"]}
              for key in chosen]
    if (len(events) != config["events"] or len({e["prompt_id"] for e in events}) != len(events)
            or Counter(e["category"] for e in events) != Counter(config["category_quotas"])
            or sum(e["dataset"] == "gsm8k" for e in events) != config["gsm8k_events"]):
        raise AssertionError("Selected cohort violates frozen diagnostic quotas")
    selection = {"event_ids": chosen, "plan_digest": digest(events),
                 "design": availability(screens, config),
                 "policy": "seeded screening; old correctness + Fresh only; first feasible disjoint-prompt quotas; no repair outcomes"}
    for event in events:
        store.put("event", event["event_id"], event)
    store.put("meta", "selection", selection)
    store.put("meta", "phase", "selected")
    return events


def screen(pool, config, store, drafter, verifier, checkpoint=lambda: None, event_done=lambda: None):
    previous = store.get("meta", "selection")
    if previous:
        events = [store.get("event", key) for key in previous["event_ids"]]
        if digest(events) != previous["plan_digest"]:
            raise ValueError("Frozen Test3 events differ from their selection digest")
        return events
    screens = {s["event_id"]: s for s in store.rows("screen")}
    chosen = select_events(screens.values(), config)
    if chosen:
        return freeze_selection(store, screens.values(), chosen, config)
    store.put("meta", "phase", "screening")
    for i, context in enumerate(pool["contexts"], 1):
        key = context["event_id"]
        if key in screens:
            continue
        store.put("meta", "active", {"phase": "screening", "event_id": key, "index": i})
        print(f"[test3 screen {i}/{len(pool['contexts'])}] {key}: native draft24", flush=True)
        ref = pool["references"][context["prompt_id"]]
        original_prefix = ref["prompt_ids"]+ref["tokens"][:context["offset"]]
        original = single_draft(drafter, original_prefix, config["window"])
        verification = checked_score(verifier, ref, context["offset"], original["tokens"], config["top_k"])
        event, reason = corrected_event(context, ref, original, feedback_record(verification), config)
        result = {"event_id": key, "screen_index": i, "original24": original,
                  "original24_verification": feedback_record(verification)}
        if event is None:
            result.update(status="skipped", reason=reason)
            print("[test3 screen] skip:", reason, flush=True)
        else:
            print(f"[test3 screen] a={event['accepted_before_rejection']} L={event['L']} E={event['extension_length']}; Fresh23", flush=True)
            fresh = single_draft(drafter, event["prefix"], event["H"])
            score = checked_score(verifier, ref, event["reference_start"], fresh["tokens"], config["top_k"])
            fresh["verification"] = feedback_record(score)
            event["fresh23_accepted"] = prefix_length(fresh["tokens"], event["reference"])
            label = category(event, fresh["tokens"], config)
            result.update(status="eligible", event=event, fresh23=fresh, category=label)
            print(f"[test3 screen] category={label}; old-correct={event['old_correct_count']}/{event['L']}; Fresh23 A={event['fresh23_accepted']}", flush=True)
        store.put("screen", key, result)
        screens[key] = result
        chosen = select_events(screens.values(), config)
        if chosen:
            events = freeze_selection(store, screens.values(), chosen, config)
            checkpoint(); event_done()
            print("[test3] Frozen 10 events before any oracle repair outcome", flush=True)
            return events
        store.put("meta", "availability", availability(screens.values(), config))
        checkpoint(); event_done()
    store.put("meta", "availability", availability(screens.values(), config))
    raise SelectionUnavailable("Screening exhausted without exact 4/3/2/1 groups and 5+5 benchmarks; inspect selection.json/screening.jsonl. No quotas were relaxed.")


def run_arms(pool, events, config, store, drafter, verifier, checkpoint=lambda: None, event_done=lambda: None):
    store.put("meta", "phase", "arms")
    for event in events:
        key, ref = event["event_id"], pool["references"][event["prompt_id"]]
        if not store.get("original_audit", key):
            print("[test3 audit] selected original24 + full corrected reference:", key, flush=True)
            original = audited_draft(drafter, event["old_context"], config["window"], {})
            if original["tokens"] != event["original_draft"]["tokens"]:
                error = AssertionError("Selected native B24 replay differs from screening; never substitute a B32 slice")
                error.evidence = {"screening": event["original_draft"], "replay": original}
                raise error
            # Passing ALL gold rows through the real causal target establishes
            # a greedy continuation from the exact corrected prefix. Saved
            # gold is audited, never substituted for model logits.
            gold_score = checked_score(verifier, ref, event["reference_start"], event["reference"], config["top_k"])
            if gold_score["top1"] != event["reference"]:
                raise AssertionError("Full corrected target continuation audit failed")
            store.put("original_audit", key, {"native24": original, "reference_audit": feedback_record(gold_score)})
        cache = {}
        for plan in arm_plans(event):
            arm_id = key+"|"+plan["method"]
            signature = digest(["sequential" if plan["schedule"] == "sequential" else "native",
                                plan["horizon"], plan["anchors"]])
            existing = store.get("arm", arm_id)
            if existing:
                cache[signature] = existing
                continue
            store.put("meta", "active", {"phase": "arms", "event_id": key, "method": plan["method"]})
            previous = cache.get(signature)
            if previous:
                measured = {k: previous[k] for k in ("tokens", "confidence", "order", "trace", "stats",
                    "cached_full_audit", "verification")}
                measured["new_drafter_counts"] = {}
            else:
                if plan["schedule"] == "sequential":
                    first = store.get("arm", key+"|O-L")
                    measured = sequential_draft(drafter, event, first)
                    if measured["tokens"][:event["L"]] != first["tokens"]:
                        raise AssertionError("Sequential modified completed old-horizon tokens")
                else:
                    measured = audited_draft(drafter, event["prefix"], plan["horizon"], dict(plan["anchors"]))
                    measured["new_drafter_counts"] = measured["stats"]
                if plan["method"] == "F-23" and measured["tokens"] != event["fresh23_screen"]["tokens"]:
                    error = AssertionError("Fresh23 replay differs from the frozen screening baseline")
                    error.evidence = {"screening": event["fresh23_screen"], "replay": measured}
                    raise error
                score = checked_score(verifier, ref, event["reference_start"], measured["tokens"], config["top_k"])
                measured["verification"] = feedback_record(score)
            if plan["oracle"] and any(t != event["reference"][i] for i, t in plan["anchors"]):
                raise AssertionError("Oracle retained an incorrect old token")
            measured["trace"] = visible_trace(event, measured["trace"], plan["anchors"])
            metrics = arm_metrics(event, plan, measured["tokens"], measured["trace"])
            if metrics["accepted_prefix"] != prefix_length(measured["tokens"], measured["verification"]["top1"]):
                raise AssertionError("Actual target acceptance disagrees with reference labels")
            baseline_name = "F-L" if plan["schedule"] == "short" else "F-23"
            baseline = measured if plan["method"] == baseline_name else store.get("arm", key+"|"+baseline_name)
            metrics.update(conditioning_metrics(event, measured["tokens"], baseline["tokens"], plan["anchors"]))
            metrics["delta_A_vs_fresh"] = metrics["accepted_prefix"]-prefix_length(baseline["tokens"], event["reference"])
            row = {"id": arm_id, "event_id": key, "prompt_id": event["prompt_id"], "dataset": event["dataset"],
                   "category": event["category"], "source_drafter": "dllm_top1", "redrafter": "dllm_top1", "verifier": TARGET,
                   "L": event["L"], "H": plan["horizon"], "extension_length": max(0, plan["horizon"]-event["L"]),
                   "reference_token_ids": event["reference"][:plan["horizon"]],
                   **plan, **measured, "metrics": metrics, "alias_of": previous["id"] if previous else None}
            if plan["schedule"] == "sequential":
                short = store.get("arm", key+"|O-L")
                if short["metrics"]["accepted_prefix"] < event["L"] and metrics["accepted_prefix"] != short["metrics"]["accepted_prefix"]:
                    raise AssertionError("Sequential extension changed acceptance inside old horizon")
            store.put("arm", arm_id, row)
            cache[signature] = row
            print(f"[test3 arm] {key} {plan['method']}: A={metrics['accepted_prefix']}/{plan['horizon']} delta={metrics['delta_A_vs_fresh']:+d}", flush=True)
            checkpoint()
        gc.collect(); event_done()
    if len(store.rows("arm")) != config["events"]*6:
        raise AssertionError("Test3 requires exactly 6 arms for each of the 10 events")
    store.put("meta", "complete", True)
    store.put("meta", "phase", "complete")
    store.put("meta", "active", None)


def pack(output, store=None):
    output = Path(output)
    if store:
        store.backup(output/"checkpoint.sqlite")
        for kind, file in (("screen", "screening.jsonl"), ("event", "events.jsonl"), ("arm", "arms.jsonl"), ("original_audit", "reference_audits.jsonl")):
            store.export(kind, output/file)
        atomic_json(output/"selection.json", store.get("meta", "selection") or store.get("meta", "availability") or {})
        atomic_json(output/"progress.json", {"phase": store.get("meta", "phase"), "active": store.get("meta", "active"),
            "screened": len(store.rows("screen")), "arms": len(store.rows("arm")), "complete": bool(store.get("meta", "complete"))})
    path, temp = output/"test3_checkpoint.zip", output/"test3_checkpoint.tmp"
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(output.rglob("*")):
            if file.is_file() and file not in (path, temp) and not file.name.startswith("test3.sqlite"):
                z.write(file, file.relative_to(output).as_posix())
    temp.replace(path)
    return path


def execute(checkpoint_root, output, revision):
    from .test3_report import report
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/"configs/test3_oracle_all.json").read_text())
    store = target = model = verifier = drafter = tokenizer = None
    started, status = time.monotonic(), "failed"
    try:
        print("[test3] Import pilot contexts/reference; original drafts will be regenerated at B=24", flush=True)
        pilot = read_pilot(checkpoint_root)
        pool = context_pool(pilot, config); del pilot
        identity = {"protocol": "test3-oracle-all-correct-refill-v1", "config": config,
                    "code_revision": revision, "source_digest": source_digest(),
                    "pilot_checkpoint_sha256": pool["pilot_checkpoint_sha256"], "pool_digest": pool["pool_digest"]}
        store = Store(output/"test3.sqlite", identity)
        atomic_json(output/"test3-manifest.json", identity)
        atomic_json(output/"pool.json", pool)
        if store.get("meta", "complete"):
            report(output, store, config)
            status = "complete"
            return
        print(f"[test3] Up to {len(pool['contexts'])} screening contexts; stop at first feasible 10-event cohort", flush=True)
        pack(output, store)
        paths, model_revisions = prepare_models()
        from .runtime import configure_numerics, tokenizers, load_target, load_model, ARSession, DLLMDrafter, target_consistency, release
        from .experiment import environment
        from .migration import validate_identity
        import torch
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Requires T4 x2")
        torch.manual_seed(config["seed"]); configure_numerics(config)
        tokenizer, evidence = tokenizers(paths)
        target, _, _ = load_target(paths[TARGET], config)
        target._suffix_exclude_peer_transfers = False
        verifier = ARSession(target)
        env = environment()
        if env["commit"] != revision:
            raise ValueError("Checked-out revision differs")
        env.update(target_device_map={k: str(v) for k, v in target.hf_device_map.items()}, target_attention=config["target_attention"],
                   allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        prior = pool["pilot_manifest"]
        validate_identity(prior, {**prior, "tokenizer": evidence, "environment": env})
        runtime_identity = {"environment": env, "tokenizer": evidence, "models": model_revisions}
        old_runtime = store.get("meta", "runtime")
        if old_runtime and digest(old_runtime) != digest(runtime_identity):
            raise ValueError("Test3 resume numerical environment differs")
        store.put("meta", "runtime", runtime_identity); atomic_json(output/"runtime.json", runtime_identity)
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": "Compute 17 + 28. Explain briefly."}], tokenize=True, add_generation_prompt=True)
        audit = target_consistency(verifier, prefix)
        atomic_json(output/"preflight.json", {"target_audit": audit,
            "policy": "selected B24 and all 60 arms/stages audited cached/full; all corrected gold rows checked by actual target"})
        if not audit["passed"]:
            raise AssertionError("Fixed-FP16 target preflight failed")
        model = load_model(paths[DLLM], "cuda:1"); drafter = DLLMDrafter(model, config)
        checkpoint = lambda: pack(output, store)
        event_done = lambda: release(None)
        events = screen(pool, config, store, drafter, verifier, checkpoint, event_done)
        run_arms(pool, events, config, store, drafter, verifier, checkpoint, event_done)
        report(output, store, config, tokenizer)
        status = "complete"
    except BaseException as error:
        status = "selection_shortfall" if isinstance(error, SelectionUnavailable) else "failed"
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                   "evidence": getattr(error, "evidence", None), "active": store.get("meta", "active") if store else None}
        atomic_json(output/"failure.json", failure)
        if store:
            store.put("failure_history", str(time.time_ns()), failure)
        raise
    finally:
        try:
            if store and status != "complete":
                try:
                    report(output, store, config)
                except Exception as error:
                    atomic_json(output/"report_failure.json", {"message": str(error)})
        finally:
            atomic_json(output/"execution_status.json", {"status": status, "code_revision": revision,
                "completed_arms": len(store.rows("arm")) if store else 0, "expected_arms": config["events"]*6,
                "screened_states": len(store.rows("screen")) if store else 0,
                "operational_seconds": time.monotonic()-started, "latency_study": False})
            archive = pack(output, store)
            if store:
                store.close()
            drafter = verifier = model = target = None
            if "torch" in locals():
                from .runtime import release
                release(None)
            print(status.upper()+": Test3", "ZIP:", archive, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    execute(args.checkpoint_root, args.output, args.revision)


if __name__ == "__main__":
    main()
