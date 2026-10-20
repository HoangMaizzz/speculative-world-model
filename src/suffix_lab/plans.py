"""One immutable anchor plan shared by both redrafters at a rejection event."""
import random
from .selection import oracle_masks


def anchor_plan(method, anchors, family, uses_reference=False, **extra):
    return {"method": method, "anchors": [[i, int(t)] for i, t in sorted(anchors.items())],
            "arm_family": family, "uses_reference": uses_reference,
            "oracle_kind": "none", **extra}


def conditioning_plans(old, gold, config, seed):
    """Same future positions/count, different contents; offline interventions only."""
    n = len(old)
    if n < 2:
        return [], {"eligible": False, "reason": "horizon_has_no_future_position"}
    probe = min(config.get("conditioning_probe_tokens", 8), max(1, n//2))
    pool = list(range(probe, n))
    k = min(config.get("conditioning_anchor_count", 4), len(pool))
    positions = sorted(random.Random(seed).sample(pool, k))
    original = [old[i] for i in positions]
    correct = [gold[i] for i in positions]
    wrong = [next((t for t in old if t != gold[i] and t != config["mask_id"]),
                  0 if gold[i] != 0 else 1) for i in positions]
    shuffled = list(original)
    random.Random(seed+1).shuffle(shuffled)
    rng = random.Random(seed+2)
    randomized = [rng.choice(old) for _ in positions]
    plans = []
    for variant, values, oracle in (("old", original, False), ("correct", correct, True),
                                    ("wrong", wrong, True), ("shuffled", shuffled, False),
                                    ("random", randomized, False)):
        plans.append(anchor_plan("future_"+variant, dict(zip(positions, values)), "conditioning", oracle,
            conditioning_variant=variant, probe_horizon=probe,
            content_changed_from_old=values != original))
    return plans, {"eligible": True, "positions": positions, "anchor_count": k, "probe_horizon": probe,
                   "content_control": "same positions/count; correct/wrong contents use offline labels"}


def complete_plans(obs, gold, config, seed, nonoracle):
    plans = [anchor_plan(method, {i: obs.old_tokens[i] for i in keep}, "nonoracle")
             for method, keep in nonoracle]
    for i, keep in enumerate(oracle_masks(obs.old_tokens, gold, config["oracle_subset_limit"], seed)):
        plans.append(anchor_plan("oracle_A" if i == 0 else f"oracle_B#{i}",
                                 {j: obs.old_tokens[j] for j in keep}, "oracle", True,
                                 oracle_kind="old_token_keep" if i == 0 else "bounded_old_token_mask"))
    controls, metadata = [], []
    for repeat in config.get("conditioning_seeds", [0]):
        actual_seed = seed+1009*repeat
        group, evidence = conditioning_plans(obs.old_tokens, gold, config, actual_seed)
        for plan in group:
            plan["conditioning_repeat"] = repeat
            plan["conditioning_seed"] = actual_seed
            if repeat:
                plan["method"] += f"#seed{repeat}"
            plan["oracle_kind"] = ("future_reference_injection" if plan["conditioning_variant"] == "correct"
                                   else "future_wrong_label_control" if plan["conditioning_variant"] == "wrong"
                                   else "future_content_control")
        controls.extend(group)
        metadata.append({"repeat": repeat, "seed": actual_seed, **evidence})
    return plans+controls, metadata
