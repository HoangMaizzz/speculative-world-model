"""Exact positional labels; no decoded-text alignment or p/q acceptance."""
import statistics


def prefix_length(proposal, reference):
    k = 0
    for a, b in zip(proposal, reference):
        if a != b:
            break
        k += 1
    return k


def runs(bits):
    result, n = [], 0
    for value in list(bits) + [False]:
        if value:
            n += 1
        elif n:
            result.append(n)
            n = 0
    return result


def suffix_metrics(proposal, reference, old_top1=None):
    if len(reference) < len(proposal):
        raise ValueError("Insufficient reference; do not invent tail labels")
    if old_top1 is not None and len(old_top1) != len(proposal):
        raise ValueError("Old verifier horizon differs")
    n = len(proposal)
    k = prefix_length(proposal, reference)
    r = k + 1  # B+1 sentinel when no rejection
    matches = [a == b for a, b in zip(proposal, reference)]
    tail = matches[r:] if r <= n else []
    islands = runs(tail)
    rejoin = next((i for i, v in enumerate(tail) if not v), len(tail))
    old = None if old_top1 is None else [a == b for a, b in zip(proposal, old_top1)]
    return {"accepted_prefix": k, "first_rejection": r, "all_accepted": k == n,
            "suffix_length": len(tail), "reference_matches": matches, "old_matches": old,
            "suffix_match_count": sum(tail), "suffix_match_rate": sum(tail)/len(tail) if tail else None,
            "island_lengths": islands, "island_count": len(islands),
            "max_island_length": max(islands, default=0),
            "mean_island_length": statistics.mean(islands) if islands else 0.0,
            "direct_rejoin_length": rejoin,
            **{f"suffix_at_least_{j}": sum(tail) >= j for j in (1, 2, 4, 8)}}


def redraft_metrics(proposal, reference, old, keep, anchor_values=None):
    if len(proposal) != len(old) or len(reference) < len(proposal):
        raise ValueError("Mismatched redraft horizon")
    anchors = {i: old[i] for i in keep} if anchor_values is None else anchor_values
    if set(anchors) != set(keep):
        raise ValueError("Anchor positions differ from KEEP positions")
    if any(proposal[i] != anchors[i] for i in keep):
        raise AssertionError("Hard KEEP changed")
    matches = [x == y for x, y in zip(proposal, reference)]
    mutable = [i for i in range(len(proposal)) if i not in keep]
    def rate(indices):
        return sum(matches[i] for i in indices)/len(indices) if indices else None
    retained = [i for i in keep if anchors[i] == old[i]]
    wrong_keep = [i for i in sorted(keep) if anchors[i] != reference[i]]
    return {"accepted_prefix": prefix_length(proposal, reference),
            "full_block_accuracy": rate(list(range(len(proposal)))),
            "fully_correct": all(matches), "retained_precision": rate(retained),
            "anchor_precision": rate(keep), "anchor_count": len(keep),
            "anchor_correct_count": sum(anchors[i] == reference[i] for i in keep),
            "remasked_accuracy": rate(mutable), "retained_count": len(retained),
            "remasked_count": len(mutable), "false_keep_count": len(wrong_keep),
            "first_false_keep": wrong_keep[0] if wrong_keep else None,
            "recovered_old_tokens_in_prefix": sum(i in retained for i in range(prefix_length(proposal, reference)))}


def event_gains(arms):
    """Separate finite correct-KEEP search from guided/conditioning observations."""
    fresh = next(x for x in arms if x["method"] == "fresh_L")
    baseline = fresh["metrics"]["accepted_prefix"]
    family = lambda x: x.get("arm_family", "oracle" if x["oracle"] else "nonoracle")
    oracle = [x for x in arms if family(x) == "oracle"]
    nonoracle = [x for x in arms if family(x) == "nonoracle"]
    guided = [x for x in nonoracle if x["method"].startswith(("verifier_confidence", "drafter_confidence", "old_agreement"))]
    wrong = [x for x in arms if x["metrics"].get("anchor_count", x["metrics"].get("retained_count", 0)) > 0
             and x["metrics"]["false_keep_count"] == x["metrics"].get("anchor_count", x["metrics"].get("retained_count", 0))]
    def gain(rows):
        return max((x["metrics"]["accepted_prefix"]-baseline for x in rows), default=0)
    a = next((x for x in oracle if x["method"] == "oracle_A"), None)
    bounded, observed, selected = gain(oracle), gain(arms), gain(nonoracle)
    return {"oracle_A_gain": a["metrics"]["accepted_prefix"]-baseline if a else None,
            "bounded_oracle_gain": bounded, "best_nonoracle_gain": selected,
            "oracle_keep_gain": a["metrics"]["accepted_prefix"]-baseline if a else None,
            "bounded_old_mask_search_gain": bounded,
            "oracle_future_injection_gain": gain([x for x in arms if x.get("oracle_kind") == "future_reference_injection"]),
            "best_guided_gain": gain(guided), "wrong_anchor_gain": gain(wrong),
            "conditioning_gain": gain([x for x in arms if family(x) == "conditioning"]),
            "best_observed_gain": observed, "oracle_selector_gap": bounded > 0 and selected <= 0,
            "no_bounded_correct_keep_gain": bounded <= 0, "no_observed_gain": observed <= 0}



def benefit_metrics(row, baseline):
    """Paired costs, useful-token rates and a conservative false-KEEP diagnosis."""
    a, f = row["metrics"]["accepted_prefix"], baseline["metrics"]["accepted_prefix"]
    dt = row["stats"]["latency_ms"]-baseline["stats"]["latency_ms"]
    draft = row["stats"]["latency_ms"]
    verification = row.get("verification", {}).get("latency_ms")
    wall_verify = row.get("verification", {}).get("wall_latency_ms")
    first_false = row["metrics"]["first_false_keep"]
    loss = a < f
    capped = loss and first_false == a
    if a > f:
        outcome = "accepted_prefix_increased"
    elif a == f:
        outcome = "equal_prefix_cheaper_draft" if dt < 0 else "equal_prefix_no_draft_saving"
    elif capped:
        outcome = "loss_capped_at_false_keep"
    else:
        outcome = "prefix_loss_with_false_keep" if row["metrics"]["false_keep_count"] else "prefix_loss_without_false_keep"
    return {"outcome": outcome, "false_keep_caps_prefix": capped,
        "loss_with_false_keep": loss and row["metrics"]["false_keep_count"] > 0,
        "accepted_tokens_per_draft_ms": a/draft if draft > 0 else None,
        "accepted_tokens_per_draft_verify_compute_ms": a/(draft+verification) if verification is not None and draft+verification > 0 else None,
        "accepted_tokens_per_draft_verify_wall_ms": a/(draft+wall_verify) if wall_verify is not None and draft+wall_verify > 0 else None}
