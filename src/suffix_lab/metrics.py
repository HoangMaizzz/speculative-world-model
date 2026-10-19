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


def redraft_metrics(proposal, reference, old, keep):
    if len(proposal) != len(old) or len(reference) < len(proposal):
        raise ValueError("Mismatched redraft horizon")
    if any(proposal[i] != old[i] for i in keep):
        raise AssertionError("Hard KEEP changed")
    matches = [x == y for x, y in zip(proposal, reference)]
    mutable = [i for i in range(len(proposal)) if i not in keep]
    def rate(indices):
        return sum(matches[i] for i in indices)/len(indices) if indices else None
    wrong_keep = [i for i in sorted(keep) if old[i] != reference[i]]
    return {"accepted_prefix": prefix_length(proposal, reference),
            "full_block_accuracy": rate(list(range(len(proposal)))),
            "fully_correct": all(matches), "retained_precision": rate(keep),
            "remasked_accuracy": rate(mutable), "retained_count": len(keep),
            "remasked_count": len(mutable), "false_keep_count": len(wrong_keep),
            "first_false_keep": wrong_keep[0] if wrong_keep else None,
            "recovered_old_tokens_in_prefix": sum(i in keep for i in range(prefix_length(proposal, reference)))}

