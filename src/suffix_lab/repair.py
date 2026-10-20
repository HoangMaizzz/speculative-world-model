"""Crossed, cold-replay repairs. Plans are frozen once, before either redrafter."""
import time
from .assets import AR, DLLM
from .metrics import redraft_metrics, prefix_length
from .plans import complete_plans
from .selection import Observation
from .store import atomic_json, digest


def run_repairs(runner):
    from .experiment import stratified_events, sampling_coverage, nonoracle_plans
    from .runtime import ARSession, DLLMDrafter, load_model, release, memory
    c, store = runner.config, runner.store
    if not store.get("meta", "test1_complete"):
        raise ValueError("Finish Test 1 before freezing the Test 2 sample")
    selected = store.get("meta", "test2_sample")
    if selected is None:
        selected = stratified_events(store.rows("test1"), c["test2_events"], c["seed"], c["test2_sources"])
        store.put("meta", "test2_sample", selected)
    atomic_json(runner.output/"sampling_coverage.json", sampling_coverage(store.rows("test1"), selected, c))
    # A single immutable plan (including bounded oracle subsets) for both models.
    for event in selected:
        if store.get("repair_plan", event):
            continue
        row = store.get("test1", event)
        r = row["metrics"]["first_rejection"]
        old, gold = row["draft"]["tokens"][r:], row["reference"][r:]
        obs = Observation(tuple(old), tuple(row["draft"]["confidence"][r:]),
                          tuple(row["feedback"]["probability"][r:]), tuple(row["feedback"]["top1"][r:]))
        seed = int(digest([c["seed"], event])[:8], 16)
        begin = time.perf_counter()
        nonoracle = nonoracle_plans(obs, c, seed)
        selector_ms = (time.perf_counter()-begin)*1000
        plans, conditioning = complete_plans(obs, gold, c, seed, nonoracle)
        store.put("repair_plan", event, {"event_id": event, "plans": plans,
            "selector_ms_event": selector_ms, "conditioning": conditioning,
            "offline_plan_wall_ms": (time.perf_counter()-begin)*1000})
    for redrafter in c["redrafters"]:
        if all(store.get("pair_complete", event+"|"+redrafter) for event in selected):
            continue
        name = DLLM if redrafter == "dllm_top1" else AR
        model = load_model(runner.manifest["paths"][name], "cuda:1")
        drafter = DLLMDrafter(model, c) if redrafter == "dllm_top1" else ARSession(model)
        try:
            refs = store.rows("reference")
            if refs:
                store.put("warmup", "test2_"+redrafter, drafter.draft(refs[0]["prompt_ids"], 2).stats)
            store.put("hardware", "test2_"+redrafter, memory())
            for event in selected:
                pair = event+"|"+redrafter
                if store.get("pair_complete", pair):
                    continue
                row = store.get("test1", event)
                ref = store.get("reference", row["prompt_id"])
                r = row["metrics"]["first_rejection"]
                prefix = ref["prompt_ids"]+ref["tokens"][:row["offset"]+r]
                old, gold = row["draft"]["tokens"][r:], row["reference"][r:]
                frozen = store.get("repair_plan", event)
                cached, completed = {}, []
                for plan in frozen["plans"]:
                    key = pair+"|"+plan["method"]
                    # Contents matter: equal positions with different values are different arms.
                    anchor_key = digest(plan["anchors"])
                    existing = store.get("test2", key)
                    if existing:
                        cached[anchor_key] = existing
                        completed.append(existing)
                        continue
                    anchors = dict(plan["anchors"])
                    previous = cached.get(anchor_key)
                    if previous is None:
                        begin = time.perf_counter()
                        result = drafter.draft(prefix, len(old), anchors=anchors)
                        feedback = runner.verifier.score(prefix, result.tokens, c["top_k"])
                        accepted = prefix_length(result.tokens, feedback["top1"])
                        expected = prefix_length(result.tokens, gold)
                        if accepted != expected or (accepted < len(old) and feedback["top1"][accepted] != gold[accepted]):
                            store.put("failure", key, {"reason": "repair verification/reference consistency",
                                "tokens": result.tokens, "reference": gold, "feedback": feedback})
                            raise AssertionError("Repair target/reference disagree: "+key)
                        measured = {"tokens": result.tokens, "stats": result.stats,
                            "verification": {k: feedback[k] for k in ("top1", "latency_ms", "wall_latency_ms", "latency_policy", "input_tokens") if k in feedback},
                            "verified_accepted_prefix": accepted,
                            "pipeline_wall_ms": (time.perf_counter()-begin)*1000}
                    else:
                        measured = {k: previous[k] for k in ("tokens", "stats", "verification", "verified_accepted_prefix", "pipeline_wall_ms")}
                    metrics = redraft_metrics(measured["tokens"], gold, old, sorted(anchors), anchors)
                    probe = plan.get("probe_horizon")
                    baseline = next((x for x in completed if x["method"] == "fresh_L"), None)
                    if probe is not None and redrafter == "ar_greedy" and measured["tokens"][:probe] != baseline["tokens"][:probe]:
                        store.put("failure", key, {"reason": "AR future anchor changed earlier predictions", "probe": probe})
                        raise AssertionError("AR causal negative control failed: "+key)
                    # Only within-physical-block future anchors are visible to dLLM probe positions.
                    visible = [[i, sum((len(prefix)+j)//32 == (len(prefix)+i-1)//32 for j in anchors if j > i)]
                               for i in range(probe or 0)]
                    payload = {"id": key, "event_id": event, "redrafter": redrafter,
                        "prompt_id": row["prompt_id"], "dataset": row["dataset"], "source": row["source"],
                        **plan, "oracle": plan["uses_reference"], "first_rejection": r, "horizon": len(old),
                        "progress_bin": row.get("progress_bin", "unknown"), "context_length": len(prefix),
                        "physical_phase": len(prefix)%32, "keep": sorted(anchors), "keep_ratio": len(anchors)/len(old),
                        **measured, "metrics": metrics, "mask_reused_from": previous["id"] if previous else None,
                        "selector_ms_event": frozen["selector_ms_event"],
                        "latency_policy": "independent cold replay; verifier stage compute and wall separate",
                        "probe_accepted_prefix": min(metrics["accepted_prefix"], probe) if probe else None,
                        "probe_gain": min(metrics["accepted_prefix"], probe)-min(baseline["metrics"]["accepted_prefix"], probe) if probe else None,
                        "future_anchors_visible_to_probe": visible}
                    store.put("test2", key, payload)
                    cached[anchor_key] = payload
                    completed.append(payload)
                    runner.checkpoint()
                oracles = [x for x in completed if x["arm_family"] == "oracle"]
                best = max(oracles, key=lambda x: (x["metrics"]["accepted_prefix"], x["metrics"]["full_block_accuracy"]))
                store.put("oracle_search", pair, {"event_id": event, "redrafter": redrafter,
                    "candidates": len(oracles), "best_id": best["id"], "bounded_not_global": True,
                    "acquisition_cost_excluded": True})
                store.put("pair_complete", pair, True)
                if all(store.get("pair_complete", event+"|"+d) for d in c["redrafters"]):
                    store.put("event_complete", event, True)
                print("test2", pair, "arms=", len(completed), flush=True)
        finally:
            del drafter, model
            release(None)
    store.put("meta", "test2_complete", True)
