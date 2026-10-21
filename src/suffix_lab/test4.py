"""Test4 Phase1: frozen outcome-blind Main200, EOS-aware logical verification."""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sqlite3
import time
import traceback
import zipfile
from .assets import ROOT, TARGET, DLLM, source_digest
from .extension import audited_draft, count_stats, prepare_models
from .extension_plan import read_pilot
from .store import Store, atomic_json, digest, canonical
from .test3 import feedback_record
from .test3_plan import visible_trace
from .test4_plan import (through_eos, token_type, prompt_plan, choose_event, make_event,
                         plans, arm_fingerprint, eos_metrics, changes)
from .verification import checked_score, VerificationError


class BudgetReached(Exception):
    pass


class SelectionShortfall(RuntimeError):
    pass


def draft_native(drafter, prefix, length, anchors, audit=False):
    if audit:
        return audited_draft(drafter, prefix, length, anchors)
    result = asdict(drafter.draft(prefix, length, anchors=anchors))
    if (len(result["tokens"]) != length or len(result["order"]) != length
            or len(result["trace"]) != length-len(anchors)
            or any(result["tokens"][i] != v for i, v in anchors.items())
            or result["stats"]["unmask_forwards"] != length-len(anchors)
            or result["stats"]["unmask_steps"] != length-len(anchors)
            or any(t["mask_before"]-t["mask_after"] != 1 for t in result["trace"])):
        raise AssertionError("Native strict Top1/hard KEEP contract failed")
    result["stats"] = count_stats(result["stats"])
    result["cached_full_audit"] = {"sampled": False}
    return result


def verify_logical(verifier, ref, offset, tokens, eos_ids, top_k):
    logical = through_eos(tokens, eos_ids)
    gold = through_eos(ref["tokens"][offset:offset+len(logical)], eos_ids)
    candidate = logical[:len(gold)]
    if not candidate or (len(gold) < len(logical) and gold[-1] not in eos_ids):
        raise ValueError("Censored labels cannot be scored as an error or EOS")
    # The future target endpoint only caps evaluation, NEVER drafter capacity.
    return feedback_record(checked_score(verifier, ref, offset, candidate, top_k))


def continue_reference(verifier, prompt_ids, saved_tokens, required, eos_ids, context_limit):
    """Replay the ORIGINAL prompt and single-token canonical history before extending.

    No long-prefix re-prefill, gold substitution, or generation beyond real EOS.
    Existing committed tokens are checked against actual logits, including EOS.
    """
    import torch
    model, device = verifier.model, verifier.device
    saved = through_eos(saved_tokens, eos_ids)
    if saved != list(saved_tokens):
        raise ValueError("Saved reference contains tokens after EOS")
    capacity = max(0, context_limit-len(prompt_ids))
    limit = min(required, capacity)
    if len(saved) > capacity:
        raise ValueError("Saved reference exceeds target context capacity")
    if len(saved) >= required or (saved and saved[-1] in eos_ids):
        return {"tokens": saved, "stop_reason": "eos" if saved and saved[-1] in eos_ids else "token_budget",
                "reference_censored": False, "new_tokens": 0, "replay_forwards": 0}
    if not limit:
        return {"tokens": saved, "stop_reason": "context_limit", "reference_censored": True,
                "new_tokens": 0, "replay_forwards": 0}
    tokens, forwards = [], 0
    with torch.inference_mode():
        out = model(torch.tensor([prompt_ids], device=device), use_cache=True, logits_to_keep=1)
        forwards += 1
        cache, logits = out.past_key_values, out.logits[0, -1].float()
        for i in range(max(len(saved), limit)):
            actual = int(logits.argmax())
            if i < len(saved) and actual != saved[i]:
                raise VerificationError({"stage": "reference_extension_replay", "position": i,
                                         "saved": saved[i], "actual": actual})
            tokens.append(actual)
            if actual in eos_ids or i+1 >= max(len(saved), limit):
                break
            out = model(torch.tensor([[actual]], device=device), past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
            forwards += 1
            cache, logits = out.past_key_values, out.logits[0, -1].float()
    if not limit and not saved:
        tokens = []
    ended = bool(tokens and tokens[-1] in eos_ids)
    return {"tokens": tokens, "stop_reason": "eos" if ended else
            "context_limit" if capacity < required else "token_budget",
            "reference_censored": not ended and len(tokens) < required,
            "new_tokens": max(0, len(tokens)-len(saved)), "replay_forwards": forwards,
            "policy": "original_prompt_prefill_single_token_replay_then_extend"}


def load_prompt_plan(pilot, config):
    from datasets import load_dataset
    lock = json.loads((ROOT/"source-lock.json").read_text())
    primary = sorted([r for (kind, key), r in pilot["records"].items() if kind == "reference"],
                     key=lambda p: (p["dataset"], p["index"]))
    datasets = {}
    for ds in config["datasets"]:
        name = {"gsm8k": "openai/gsm8k", "math500": "HuggingFaceH4/MATH-500"}[ds]
        kw = {"name": "main"} if ds == "gsm8k" else {}
        print("[test4 datasets]", name, lock["datasets"][name], flush=True)
        data = load_dataset(name, split="test", revision=lock["datasets"][name], **kw)
        datasets[ds] = [{"id": f"{ds}:{i}", "index": i, "dataset": ds,
                          "question": r["question" if ds == "gsm8k" else "problem"],
                          "dataset_fingerprint": data._fingerprint} for i, r in enumerate(data)]
    # Never embed old B32 drafts, results or logits in the sampling plan.
    fields = ("id", "dataset", "index", "question", "dataset_fingerprint")
    primary = [{k: p[k] for k in fields} for p in primary]
    return prompt_plan(primary, datasets, config)


def screen_main(plan, config, store, drafter, verifier, eos_ids, models, reference_for,
                checkpoint=lambda: None, boundary=lambda: None):
    frozen = store.get("meta", "selection")
    if frozen:
        events = [store.get("event", k) for k in frozen["event_ids"]]
        if digest(events) != frozen["events_digest"]:
            raise ValueError("Frozen Main200 event digest differs")
        return events
    store.put("meta", "phase", "screening")
    wanted = config["events_per_dataset"]
    for ds in config["datasets"]:
        for rank, prompt in enumerate(plan[ds]):
            selected = [e for e in store.rows("event") if e["dataset"] == ds]
            if len(selected) >= wanted:
                break
            if store.get("prompt_done", prompt["id"]):
                continue
            boundary()
            print(f"[test4 prompt] {ds} selected={len(selected)}/{wanted}; {prompt['id']} {prompt['source']}", flush=True)
            ref = reference_for(prompt)
            candidates = []
            for offset in prompt["offsets"]:
                key = f"{prompt['id']}:{offset}:dllm24"
                previous = store.get("screen", key)
                if previous:
                    if previous["status"] == "eligible":
                        candidates.append(previous["event"])
                    continue
                boundary()
                store.put("meta", "active", {"phase": "screening", "event_id": key})
                gold = through_eos(ref["tokens"][:offset], eos_ids)
                if len(gold) < offset or (gold and gold[-1] in eos_ids):
                    row = {"event_id": key, "prompt_id": prompt["id"], "dataset": ds,
                           "status": "excluded", "reason": "offset_after_reference_end", "native_executed": False}
                elif offset >= len(ref["tokens"]):
                    row = {"event_id": key, "prompt_id": prompt["id"], "dataset": ds,
                           "status": "excluded", "reason": "reference_censored", "native_executed": False}
                else:
                    print(f"[test4 screen] {key}: native B24 + actual target verification", flush=True)
                    original = draft_native(drafter, ref["prompt_ids"]+ref["tokens"][:offset], 24, {})
                    row = {"event_id": key, "prompt_id": prompt["id"], "dataset": ds,
                           "native_executed": True, "original24": original}
                    if len(through_eos(original["tokens"], eos_ids)) < 24:
                        event, reason = None, "early_draft_eos"
                    elif len(ref["tokens"])-offset < 24 and ref["tokens"][-1] not in eos_ids:
                        event, reason = None, "reference_censored"
                    else:
                        score = verify_logical(verifier, ref, offset, original["tokens"], eos_ids, config["top_k"])
                        row["verification"] = score
                        from .metrics import prefix_length
                        row["accepted_before_rejection"] = prefix_length(original["tokens"],
                            through_eos(ref["tokens"][offset:offset+24], eos_ids))
                        event, reason = make_event(prompt, offset, ref, original, score, config, eos_ids, models)
                    row.update(status="eligible" if event else "excluded", reason=reason)
                    if event:
                        row["event"] = event
                        candidates.append(event)
                    print("[test4 screen]", key, row["status"], reason or "", flush=True)
                store.put("screen", key, row)
                checkpoint()
            chosen = choose_event(candidates, prompt["id"], config["seed"])
            decision = {"prompt_id": prompt["id"], "dataset": ds, "rank": rank,
                        "source": prompt["source"], "eligible_offsets": [e["offset"] for e in candidates],
                        "chosen_event": chosen["event_id"] if chosen else None}
            # One transaction: a selected event can never outlive its prompt decision.
            with store.db:
                if chosen:
                    store.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)",
                                     ("event", chosen["event_id"], canonical(chosen)))
                store.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)",
                                 ("prompt_done", prompt["id"], canonical(decision)))
            checkpoint()
            from .runtime import release
            release(None)
        if len([e for e in store.rows("event") if e["dataset"] == ds]) != wanted:
            raise SelectionShortfall(f"Exhausted frozen {ds} reserves without {wanted} eligible prompts")
    events = store.rows("event")
    if len(events) != wanted*len(config["datasets"]) or len({e["prompt_id"] for e in events}) != len(events):
        raise AssertionError("Main200 must have one event per distinct prompt")
    store.put("meta", "selection", {"event_ids": [e["event_id"] for e in events],
        "events_digest": digest(events), "policy": "uniform choice among ALL eligible fixed offsets per prompt; no Fresh/Oracle outcomes"})
    store.put("meta", "phase", "main")
    checkpoint()
    print(f"[test4] FROZEN {len(events)} events before any Main arm", flush=True)
    return events


def run_main(events, config, store, drafter, verifier, eos_ids, revision, models,
             checkpoint=lambda: None, boundary=lambda: None, tokenizer=None):
    for index, event in enumerate(events):
        reference = store.get("reference", event["prompt_id"])
        if not store.get("reference_audit", event["event_id"]):
            boundary()
            audit = verify_logical(verifier, reference, event["reference_start"], event["reference"], eos_ids, config["top_k"])
            if audit["top1"] != event["reference"]:
                raise AssertionError("Canonical continuation audit did not match actual target")
            original_audit = None
            if index % config["audit_every_events"] == 0:
                original_audit = draft_native(drafter, event["old_context"], 24, {}, True)
                if original_audit["tokens"] != event["original_draft"]["tokens"]:
                    raise AssertionError("Selected B24 cached/full audit differs from screening")
            store.put("reference_audit", event["event_id"], {"verification": audit, "original24": original_audit})
            checkpoint()
        for action in plans(event, eos_ids):
            arm_id = event["event_id"]+"|"+action["method"]
            fingerprint = arm_fingerprint(event, action, config, revision, models)
            existing = store.get("arm", arm_id)
            if existing:
                if existing["arm_fingerprint"] != fingerprint:
                    raise ValueError("Stored arm fingerprint differs")
                continue
            boundary()
            store.put("meta", "active", {"phase": "main", "event_id": event["event_id"], "method": action["method"]})
            cached = store.get("arm_cache", fingerprint)
            equivalent = store.get("arm", cached["id"]) if cached else None
            if equivalent:
                measured = equivalent["measured"]
            else:
                measured = draft_native(drafter, event["prefix"], 23, dict(action["anchors"]),
                                        index % config["audit_every_events"] == 0)
                measured["verification"] = verify_logical(verifier, reference, event["reference_start"],
                    measured["tokens"], eos_ids, config["top_k"])
                measured["trace"] = visible_trace(event, measured["trace"], action["anchors"])
            metrics = eos_metrics(measured["tokens"], event["reference"], eos_ids)
            n = metrics["evaluated_tokens"]
            from .metrics import prefix_length
            if prefix_length(measured["tokens"][:n], measured["verification"]["top1"]) != metrics["A_inclusive"]:
                raise AssertionError("Actual verifier acceptance differs from canonical EOS-aware labels")
            baseline = measured if action["method"] == "F-23" else store.get("arm", event["event_id"]+"|F-23")["measured"]
            metrics.update(changes(event, measured["tokens"], baseline["tokens"], action["anchors"], eos_ids))
            metrics["delta_A_vs_fresh"] = metrics["A_inclusive"]-eos_metrics(baseline["tokens"], event["reference"], eos_ids)["A_inclusive"]
            metrics["accepted_extension"] = max(0, metrics["A_inclusive"]-event["L"])
            cutoff = metrics["logical_proposal_length"]
            metrics["unmask_forwards_beyond_logical_eos"] = sum(t["position"] >= cutoff for t in measured["trace"])
            false_positions = [i for i, v in action["anchors"]
                               if i < len(event["reference"]) and v != event["reference"][i]]
            metrics["false_keep_count"] = len(false_positions)
            metrics["unscored_keep_after_target_end"] = sum(i >= len(event["reference"]) for i, _ in action["anchors"])
            metrics["first_false_keep_position"] = min(false_positions, default=None)
            if false_positions and metrics["A_inclusive"] > min(false_positions):
                raise AssertionError("Acceptance crossed an immutable false anchor")
            row = {"id": arm_id, "event_id": event["event_id"], "prompt_id": event["prompt_id"],
                   "dataset": event["dataset"], **action, "arm_fingerprint": fingerprint,
                   "measured": measured, "metrics": metrics,
                   "alias_of": equivalent["id"] if equivalent else None,
                   "new_drafter_counts": {} if equivalent else measured["stats"]}
            store.put("arm", arm_id, row)
            store.put("arm_cache", fingerprint, {"id": arm_id})
            print(f"[test4 main {index+1}/{len(events)}] {arm_id} A={metrics['A_inclusive']} content={metrics['A_content']} EOS={metrics['terminated_correctly']} delta={metrics['delta_A_vs_fresh']:+d}", flush=True)
            checkpoint()
        gc.collect()
        from .runtime import release
        release(None)
    if len(store.rows("arm")) != len(events)*3:
        raise AssertionError("Main requires exactly three completed arms/event")
    store.put("meta", "complete", True)
    store.put("meta", "phase", "complete")
    store.put("meta", "active", None)


def pack(output, store):
    output = Path(output)
    if store:
        store.backup(output/"checkpoint.sqlite")
        for key, filename in (("prompt_plan", "prompt_plan.json"), ("runtime", "runtime.json"),
                              ("preflight", "preflight.json"), ("anchor_tokens", "anchor_tokens.json")):
            value = store.get("meta", key)
            if value is not None:
                atomic_json(output/filename, value)
        for kind, file in (("screen", "screening.jsonl"), ("event", "events.jsonl"), ("arm", "arms.jsonl"),
                           ("reference", "references.jsonl"), ("prompt_done", "prompt_decisions.jsonl"),
                           ("reference_audit", "reference_audits.jsonl")):
            store.export(kind, output/file)
        atomic_json(output/"selection.json", store.get("meta", "selection") or {})
        atomic_json(output/"progress.json", {"phase": store.get("meta", "phase"),
            "active": store.get("meta", "active"), "prompts": len(store.rows("prompt_done")),
            "screened": len(store.rows("screen")), "events": len(store.rows("event")),
            "arms": len(store.rows("arm")), "complete": bool(store.get("meta", "complete"))})
    path, tmp = output/"test4_phase1_checkpoint.zip", output/"test4_phase1_checkpoint.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.iterdir()):
            if file.is_file() and file not in (path, tmp) and not file.name.startswith("test4.sqlite"):
                archive.write(file, file.name)
    tmp.replace(path)
    return path


def restore(root, output, identity):
    """Restore ONLY a matching Main checkpoint; never overwrite a live local run."""
    if not root:
        return
    output = Path(output)
    if (output/"test4.sqlite").exists():
        return
    root = Path(root)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="test4-resume-") as folder:
        if root.is_file():
            with zipfile.ZipFile(root) as z:
                if z.testzip() is not None:
                    raise ValueError("Resume ZIP CRC failed")
                manifest = json.loads(z.read("test4-manifest.json"))
                local = Path(folder)/"checkpoint.sqlite"
                local.write_bytes(z.read("checkpoint.sqlite"))
        else:
            found = list(root.rglob("test4-manifest.json"))
            if len(found) != 1:
                raise ValueError("Resume input must contain exactly one test4-manifest.json")
            manifest = json.loads(found[0].read_text())
            import shutil
            local = Path(folder)/"checkpoint.sqlite"
            shutil.copyfile(found[0].parent/"checkpoint.sqlite", local)
        if digest(manifest) != digest(identity):
            raise ValueError("Resume manifest differs from this Phase1/runtime/input")
        conn = sqlite3.connect(local.resolve().as_uri()+"?mode=ro&immutable=1", uri=True)
        try:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Resume SQLite integrity failed")
            saved = json.loads(conn.execute("SELECT payload FROM records WHERE kind='meta' AND key='identity'").fetchone()[0])
            if digest(saved) != digest(identity):
                raise ValueError("Resume database identity differs")
            dest = sqlite3.connect(str(output/"test4.sqlite"))
            try:
                conn.backup(dest)
            finally:
                dest.close()
        finally:
            conn.close()


def execute(checkpoint_root, output, revision, resume_root=None):
    from .test4_report import report
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT/"configs/test4_main.json").read_text())
    models = json.loads((ROOT/"source-lock.json").read_text())["models"]
    models = {k: models[k] for k in (TARGET, DLLM)}
    store = target = verifier = model = drafter = None
    started, status = time.monotonic(), "failed"
    try:
        pilot = read_pilot(checkpoint_root)
        identity = {"protocol": "test4-phase1-eos-main200-v1", "config": config,
                    "code_revision": revision, "source_digest": source_digest(),
                    "pilot_checkpoint_sha256": pilot["checkpoint_sha256"], "models": models}
        restore(resume_root, output, identity)
        store = Store(output/"test4.sqlite", identity)
        atomic_json(output/"test4-manifest.json", identity)
        if store.get("meta", "complete"):
            report(output, store, config); status = "complete"; return
        plan = store.get("meta", "prompt_plan")
        if plan is None:
            plan = load_prompt_plan(pilot, config)
            store.put("meta", "prompt_plan", plan)
        atomic_json(output/"prompt_plan.json", plan)
        pack(output, store)
        def boundary():
            if time.monotonic()-started >= config["soft_budget_seconds"]:
                raise BudgetReached("Soft six-hour session budget; resume saved atomic units")
        boundary()
        paths, _ = prepare_models()
        from .runtime import (configure_numerics, tokenizers, load_target, load_model,
                              ARSession, DLLMDrafter, target_consistency, release)
        from .experiment import environment
        from .migration import validate_identity
        import torch
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Requires GPU T4 x2")
        torch.manual_seed(config["seed"]); configure_numerics(config)
        tokenizer, evidence = tokenizers(paths)
        target, _, _ = load_target(paths[TARGET], config)
        # This is an acceptance diagnostic. Persist counts, not transfer-inclusive
        # model latency as a scientific metric. Operational budget uses real wall.
        target._suffix_exclude_peer_transfers = False
        verifier = ARSession(target)
        env = environment()
        if env["commit"] != revision:
            raise ValueError("Checked-out revision differs")
        env.update(target_device_map={k: str(v) for k, v in target.hf_device_map.items()},
            target_attention=config["target_attention"],
            allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        prior = pilot["manifest"]
        validate_identity(prior, {**prior, "tokenizer": evidence, "environment": env})
        model = load_model(paths[DLLM], "cuda:1"); drafter = DLLMDrafter(model, config)
        def ids(value):
            return [] if value is None else value if isinstance(value, (list, tuple)) else [value]
        stop_sources = {"tokenizer": ids(tokenizer.eos_token_id),
                        "target_generation": ids(getattr(getattr(target, "generation_config", None), "eos_token_id", None)),
                        "dllm_generation": ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None)),
                        "target_config": ids(getattr(target.config, "eos_token_id", None)),
                        "dllm_config": ids(getattr(model.config, "eos_token_id", None))}
        eos_ids = set(sum(stop_sources.values(), []))
        if not eos_ids or any(not isinstance(t, int) or t < 0 for t in eos_ids):
            raise ValueError("Invalid configured EOS IDs")
        runtime = {"environment": env, "tokenizer": evidence, "stop_sources": stop_sources,
                   "effective_eos_ids": sorted(eos_ids), "models": models}
        previous = store.get("meta", "runtime")
        if previous and digest(previous) != digest(runtime):
            raise ValueError("Resume numerical/runtime/EOS identity differs")
        store.put("meta", "runtime", runtime); atomic_json(output/"runtime.json", runtime)
        probe = tokenizer.apply_chat_template([{"role": "user", "content": "Compute 17 + 28. Explain briefly."}], tokenize=True, add_generation_prompt=True)
        audit = target_consistency(verifier, probe)
        atomic_json(output/"preflight.json", {"target": audit, "passed": False})
        if not audit["passed"]:
            raise AssertionError("Fixed FP16 target preflight failed")
        geometry = []
        for phase in (0, 1, 7, 8, 31):
            prefix = probe+[probe[-1]]*((phase-len(probe))%32)
            base = draft_native(drafter, prefix, 24, {}, True)
            anchors = {i: t for i, t in enumerate(base["tokens"]) if i%3 == 0 and t not in eos_ids}
            kept = draft_native(drafter, prefix, 23, {i: t for i, t in anchors.items() if i < 23}, True)
            geometry.append({"phase": phase, "passed": True, "original_stats": base["stats"], "keep_stats": kept["stats"]})
        atomic_json(output/"preflight.json", {"target": audit, "dllm_geometry": geometry, "passed": True,
            "EOS_policy": "unchanged native tensor; cut logical output before verification; no early-stop savings claim"})
        store.put("meta", "preflight", json.loads((output/"preflight.json").read_text()))
        references = {r["id"]: r for (kind, _), r in pilot["records"].items() if kind == "reference"}
        del pilot
        def reference_for(prompt):
            existing = store.get("reference", prompt["id"])
            if existing:
                return existing
            boundary()
            template = tokenizer.apply_chat_template([{"role": "user", "content": prompt["question"]}], tokenize=True, add_generation_prompt=True)
            old = references.get(prompt["id"])
            if old and old["prompt_ids"] != template:
                raise ValueError("Pilot prompt token IDs/template differ")
            store.put("meta", "active", {"phase": "reference", "prompt_id": prompt["id"]})
            print("[test4 reference]", prompt["id"], "canonical replay/extend up to", config["reference_tokens"], flush=True)
            continuation = continue_reference(verifier, template, old["tokens"] if old else [],
                config["reference_tokens"], eos_ids, target.config.max_position_embeddings)
            row = {**prompt, "prompt_ids": template, **continuation}
            store.put("reference", prompt["id"], row)
            pack(output, store)
            return row
        last_pack = [time.monotonic()]
        def checkpoint():
            # Every unit is committed synchronously to SQLite. Refresh the portable
            # snapshot at most two minutes apart, plus phase boundaries/finally.
            # This avoids repeatedly compressing the entire growing run per arm.
            if time.monotonic()-last_pack[0] >= 120:
                pack(output, store); last_pack[0] = time.monotonic()
        events = screen_main(plan, config, store, drafter, verifier, eos_ids, models, reference_for, checkpoint, boundary)
        pack(output, store)
        histogram, pieces = Counter(), {}
        for event in events:
            for _, token in event["oracle_anchors"]:
                piece = tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                group = token_type(piece, token, set(tokenizer.all_special_ids)|eos_ids)
                histogram[group] += 1
                pieces[str(token)] = {"raw_piece": piece, "repr": repr(piece), "type": group}
        anchor_record = {"histogram": dict(histogram), "pieces": pieces,
            "token_220": {"raw_piece": tokenizer.decode([220], clean_up_tokenization_spaces=False)}}
        atomic_json(output/"anchor_tokens.json", anchor_record)
        store.put("meta", "anchor_tokens", anchor_record)
        run_main(events, config, store, drafter, verifier, eos_ids, revision, models, checkpoint, boundary, tokenizer)
        report(output, store, config); status = "complete"
        (output/"failure.json").unlink(missing_ok=True)
    except BudgetReached as error:
        status = "budget_checkpoint"
        print("[test4]", str(error), flush=True)
    except BaseException as error:
        status = "selection_shortfall" if isinstance(error, SelectionShortfall) else "failed"
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                   "stop_reason": "runtime_error", "evidence": getattr(error, "evidence", None),
                   "active": store.get("meta", "active") if store else None}
        atomic_json(output/("failure.json" if store else "bootstrap_failure.json"), failure)
        if store:
            store.put("failure_history", str(time.time_ns()), failure)
        raise
    finally:
        # Wrong input/identity must not rewrite an existing run's portable ZIP,
        # status or matching manifest. The bootstrap error remains separate.
        if store is None:
            print("[test4] Initialization failed; existing checkpoint artifacts preserved", flush=True)
        else:
            finish(output, store, config, revision, started, status, report)
        drafter = verifier = model = target = None
        gc.collect()
        if "torch" in locals():
            release(None)


def finish(output, store, config, revision, started, status, report):
    """Finalize partial and complete runs, including a fresh portable snapshot."""
    try:
        if store and (status != "complete" or not (output/"report.md").exists()):
            try:
                report(output, store, config)
            except Exception as error:
                atomic_json(output/"report_failure.json", {"message": str(error)})
        atomic_json(output/"execution_status.json", {"status": status, "code_revision": revision,
            "completed_arms": len(store.rows("arm")) if store else 0,
            "expected_arms": config["events_per_dataset"]*len(config["datasets"])*3,
            "operational_seconds_this_session": time.monotonic()-started, "latency_study": False})
        archive = pack(output, store)
        print(status.upper()+": Test4 Phase1 ZIP:", archive, flush=True)
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--resume-root")
    args = parser.parse_args()
    execute(args.checkpoint_root, args.output, args.revision, args.resume_root)


if __name__ == "__main__":
    main()
