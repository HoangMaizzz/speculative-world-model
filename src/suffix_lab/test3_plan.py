"""Test 3 contracts: new B24 proposals, all-correct anchors and diagnostic quotas."""
from collections import Counter
from functools import lru_cache
import random
from .metrics import prefix_length, runs
from .store import digest


def context_pool(pilot, config):
    """Only reuse canonical contexts; never slice or inspect old B32 proposals."""
    contexts, refs = [], {}
    for (kind, key), row in sorted(pilot["records"].items()):
        if kind != "test1" or row["source"] != "dllm_top1":
            continue
        ref = pilot["records"][("reference", row["prompt_id"])]
        if len(ref["tokens"])-row["offset"] < config["window"]+config["accepted_min"]:
            continue
        refs[ref["id"]] = ref
        contexts.append({"event_id": row["state_id"]+":dllm24", "state_id": row["state_id"],
            "pilot_test1_id": row["id"], "prompt_id": row["prompt_id"], "dataset": row["dataset"],
            "offset": row["offset"], "progress_bin": row.get("progress_bin", "unknown")})
    # Cover different prompts before screening additional offsets of one prompt.
    # Round-robin benchmarks; every shuffle is seeded, never outcome-driven.
    rng = random.Random(config["seed"])
    queues = {}
    for dataset in ("gsm8k", "math500"):
        groups = {}
        for row in contexts:
            if row["dataset"] == dataset:
                groups.setdefault(row["prompt_id"], []).append(row)
        prompts = sorted(groups)
        rng.shuffle(prompts)
        for rows in groups.values():
            rng.shuffle(rows)
        queue = []
        for i in range(max((len(rows) for rows in groups.values()), default=0)):
            queue.extend(groups[p][i] for p in prompts if i < len(groups[p]))
        queues[dataset] = queue
    ordered = []
    for i in range(max(map(len, queues.values()), default=0)):
        for dataset in ("gsm8k", "math500"):
            if i < len(queues[dataset]):
                ordered.append(queues[dataset][i])
    ordered = ordered[:config["screen_limit"]]
    return {"contexts": ordered, "references": refs, "pilot_manifest": pilot["manifest"],
            "pilot_checkpoint_sha256": pilot["checkpoint_sha256"], "pool_digest": digest(ordered)}


def corrected_event(context, ref, draft24, verification, config):
    """Correction belongs to context; H=B-1 and E=a, never fixed E=7/8."""
    proposal = draft24["tokens"]
    if len(proposal) != config["window"]:
        raise ValueError("Original proposal must be generated natively at B=24")
    offset = context["offset"]
    a = prefix_length(proposal, ref["tokens"][offset:offset+config["window"]])
    if not config["accepted_min"] <= a <= config["accepted_max"]:
        return None, "accepted_count_outside_3_15"
    start = offset+a+1
    horizon = config["next_horizon"]
    gold = ref["tokens"][start:start+horizon]
    if len(gold) != horizon:
        return None, "insufficient_reference_for_refill"
    if prefix_length(proposal, verification["top1"]) != a or verification["top1"][a] != ref["tokens"][offset+a]:
        raise AssertionError("First correction is not validated under the original context")
    old = proposal[a+1:]
    length, extension = len(old), horizon-len(old)
    if length+extension+1 != config["window"] or extension != a:
        raise AssertionError("Off-by-one in correction/old/extension counting")
    anchors = [[i, t] for i, t in enumerate(old) if t == gold[i]]
    prefix = ref["prompt_ids"]+ref["tokens"][:start]
    event = {**context, "original_window": config["window"], "original_draft": draft24,
        "question": ref.get("question", ""),
        "original_verification": verification, "old_context": ref["prompt_ids"]+ref["tokens"][:offset],
        "accepted_before_rejection": a, "first_rejection_1based": a+1,
        "correction_token": verification["top1"][a], "reference_start": start,
        "prefix": prefix, "old": old, "reference": gold, "L": length, "H": horizon,
        "extension_length": extension, "oracle_anchors": anchors,
        "oracle_keep_positions": [i for i, t in anchors], "old_correct_count": len(anchors),
        "old_correct_islands": runs([t == gold[i] for i, t in enumerate(old)]),
        "physical_phase": len(prefix)%32, "old_endpoint_phase8": (len(prefix)+length)%8}
    event["initial_oracle_joint_proposal"] = [dict(anchors).get(i, config["mask_id"]) for i in range(horizon)]
    return event, None


def category(event, fresh_tokens, config):
    a = prefix_length(fresh_tokens, event["reference"])
    keep, length = event["oracle_keep_positions"], event["L"]
    if a >= event["H"]-config["ceiling_remaining"]:
        return "fresh_ceiling"
    if len(keep) <= config["low_reuse_max"]:
        return "low_reuse"
    if a < length:
        if any(a <= i <= a+config["near_positions"] for i in keep):
            return "early_rescue"
        later = sum(i > a+config["near_positions"] for i in keep)
        if (later/len(keep) >= config["later_fraction_min"]
                and max(event["old_correct_islands"], default=0) >= config["later_min_island"]):
            return "later_anchor"
    return "other"


def select_events(screens, config):
    """Uses OLD correctness and Fresh only; no oracle repair outcomes available."""
    names = tuple(config["category_quotas"])
    eligible = [s for s in screens if s["status"] == "eligible" and s["category"] in names]
    counts = Counter(s["category"] for s in eligible)
    if any(counts[k] < config["category_quotas"][k] for k in names):
        return None
    groups = {}
    for s in sorted(eligible, key=lambda x: x["screen_index"]):
        groups.setdefault(s["event"]["prompt_id"], []).append(s)
    groups = list(groups.values())

    @lru_cache(None)
    def solve(i, remaining, gsm):
        n = sum(remaining)
        if n == 0:
            return () if gsm == 0 else None
        if i == len(groups) or len(groups)-i < n or not 0 <= gsm <= n:
            return None
        for s in groups[i]:
            j = names.index(s["category"])
            if remaining[j] == 0:
                continue
            next_counts = list(remaining); next_counts[j] -= 1
            tail = solve(i+1, tuple(next_counts), gsm-int(s["event"]["dataset"] == "gsm8k"))
            if tail is not None:
                return (s["event"]["event_id"],)+tail
        return solve(i+1, remaining, gsm)
    answer = solve(0, tuple(config["category_quotas"][k] for k in names), config["gsm8k_events"])
    return list(answer) if answer is not None else None


def arm_plans(event):
    oracle, keep_all = event["oracle_anchors"], list(enumerate(event["old"]))
    return [
        {"method": "F-23", "schedule": "joint", "horizon": event["H"], "anchors": [], "oracle": False},
        {"method": "F-L", "schedule": "short", "horizon": event["L"], "anchors": [], "oracle": False},
        {"method": "O-L", "schedule": "short", "horizon": event["L"], "anchors": oracle, "oracle": True},
        {"method": "O-Joint-23", "schedule": "joint", "horizon": event["H"], "anchors": oracle, "oracle": True},
        {"method": "O-Sequential-23", "schedule": "sequential", "horizon": event["H"], "anchors": oracle, "oracle": True},
        {"method": "KEEP-All-23", "schedule": "joint", "horizon": event["H"], "anchors": keep_all, "oracle": False}]


def conditioning_metrics(event, tokens, fresh, anchors):
    """Separate forced answers, actual mutable corrections and bridged matches."""
    gold, keep = event["reference"], dict(anchors)
    groups = {}
    for region in ("REDRAFT", "EXTEND"):
        positions = [i for i in range(len(tokens)) if i not in keep and
                     (i < event["L"]) == (region == "REDRAFT")]
        cells = Counter((fresh[i] == gold[i], tokens[i] == gold[i]) for i in positions)
        groups[region] = {"fresh_wrong_oracle_correct": cells[(False, True)],
            "fresh_correct_oracle_wrong": cells[(True, False)],
            "both_wrong": cells[(False, False)], "both_correct": cells[(True, True)],
            "changed_tokens": sum(tokens[i] != fresh[i] for i in positions)}
    before, after = prefix_length(fresh, gold), prefix_length(tokens, gold)
    attribution = Counter()
    for i in range(before, after):
        if i in keep:
            name = "direct_keep_corrected" if fresh[i] != gold[i] else "keep_already_correct"
        elif fresh[i] == gold[i]:
            name = "bridged_already_correct_nonkeep"
        else:
            name = "conditioning_redraft_corrected" if i < event["L"] else "conditioning_extension_corrected"
        attribution[name] += 1
    result = {k: attribution[k] for k in ("direct_keep_corrected", "keep_already_correct",
              "bridged_already_correct_nonkeep", "conditioning_redraft_corrected", "conditioning_extension_corrected")}
    if sum(result.values()) != max(0, after-before):
        raise AssertionError("Accepted gain attribution does not partition the added interval")
    return {"mutable_positionwise": groups, "accepted_gain_attribution": result,
            "positionwise_correct": sum(t == gold[i] for i, t in enumerate(tokens)),
            "positionwise_total": len(tokens)}


def visible_trace(event, trace, anchors):
    """Right-shift row/carry visibility, including newly unmasked extension."""
    known_extension, result = set(), []
    start = len(event["prefix"])
    keep = dict(anchors)
    for step in trace:
        i = step["position"]
        row = start+i-1
        end = (row//32+1)*32
        result.append({**step, "region": "EXTEND" if i >= event["L"] else "REDRAFT",
            "prediction_absolute_row": row, "native_visible_absolute_end_exclusive": end,
            "uses_previous_block_carry": step["prediction_row"] is None,
            "visible_future_keep_positions": [j for j in sorted(keep) if j > i and start+j < end],
            "visible_generated_future_extension": [j for j in sorted(known_extension) if j > i and start+j < end]})
        if i >= event["L"]:
            known_extension.add(i)
    return result
