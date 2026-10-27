"""Pure online commit rules and separate, post-hoc shifted-span diagnostics."""
from .store import digest


def until_eos(tokens, eos, include=True):
    for i, token in enumerate(tokens):
        if token in eos:
            return list(tokens[:i+int(include)])
    return list(tokens)


def decide(proposal, top1_with_bonus, eos, remaining):
    if not proposal or remaining < 1 or until_eos(proposal, eos) != list(proposal):
        raise ValueError("Expected a nonempty logical proposal and positive output budget")
    if len(top1_with_bonus) != len(proposal)+1:
        raise ValueError("Verifier must return shifted rows plus a real bonus row")
    a = next((i for i, t in enumerate(proposal) if t != top1_with_bonus[i]), len(proposal))
    rejected = a < len(proposal)
    terminal_draft = a == len(proposal) and proposal[-1] in eos
    extra = None if terminal_draft else top1_with_bonus[a]
    emitted = until_eos(list(proposal[:a])+([] if extra is None else [extra]), eos)[:remaining]
    if not emitted:
        raise AssertionError("Online round made no progress")
    return {"accepted_draft": a, "first_rejection": a if rejected else None,
            "correction": extra if rejected else None, "bonus": extra if not rejected else None,
            "committed_tokens": emitted, "terminated": emitted[-1] in eos}


def validate_draft(draft, length, prefix):
    if len(draft["tokens"]) != length or len(draft["trace"]) != length:
        raise AssertionError("Native requested length/trace changed")
    if draft["stats"]["unmask_forwards"] != length or draft["stats"]["unmask_steps"] != length:
        raise AssertionError("One token per unmask forward required")
    seen = set()
    for t in draft["trace"]:
        i = t["position"]
        if i in seen or not 0 <= i < length or t["mask_before"]-t["mask_after"] != 1:
            raise AssertionError("Invalid one-MASK trace")
        if t["absolute_position"] != len(prefix)+i or t["token"] != draft["tokens"][i]:
            raise AssertionError("Trace geometry/token mismatch")
        seen.add(i)


def matching_runs(old, reference, radius, minimum=4, include_zero=False):
    rows = []
    for delta in range(-radius, radius+1):
        if delta == 0 and not include_zero:
            continue
        start = None
        for i in range(len(old)+1):
            j = i+delta
            equal = i < len(old) and 0 <= j < len(reference) and old[i] == reference[j]
            if equal and start is None:
                start = i
            if start is not None and not equal:
                if i-start >= minimum:
                    rows.append({"source_start": start, "source_end": i,
                                 "destination_start": start+delta, "destination_end": i+delta,
                                 "length": i-start, "delta": delta})
                start = None
    return sorted(rows, key=lambda r: (-r["length"], abs(r["delta"]), r["destination_start"], r["source_start"], r["delta"]))


def monotone_chain(runs):
    """Best total length among maximal runs, not a global partial-run edit optimum."""
    ordered = sorted(runs, key=lambda r: (r["source_end"], r["destination_end"], r["source_start"], r["delta"]))
    def key(path):
        return (-sum(r["length"] for r in path), len(path), sum(abs(r["delta"]) for r in path),
                tuple((r["destination_start"], r["source_start"], r["delta"]) for r in path))
    paths = []
    for i, row in enumerate(ordered):
        choices = [[row]]
        for j in range(i):
            prev = ordered[j]
            if prev["source_end"] <= row["source_start"] and prev["destination_end"] <= row["destination_start"]:
                choices.append(paths[j]+[row])
        paths.append(min(choices, key=key))
    return min(paths, key=key) if paths else []


def diagnose(row, reference, eos, config):
    """Called only AFTER generation and independent canonical-output audit pass."""
    a = row["decision"]["first_rejection"]
    if a is None:
        return None
    old = until_eos(row["proposal"][a+1:], eos, False)
    start = row["offset"]+a+1
    raw = reference[start:start+config["diagnostic_horizon"]]
    gold = until_eos(raw, eos, False)
    terminal = row["decision"]["correction"] in eos
    if terminal:
        old, gold = [], []
    metrics = []
    chosen_runs = []
    for radius in config["shift_radii"]:
        runs = matching_runs(old, gold, radius, config["minimum_run"])
        chain = monotone_chain(runs)
        metrics.append({"radius": radius, "has_run": bool(runs),
                        "longest": runs[0]["length"] if runs else 0,
                        "blocks": len(chain), "covered_tokens": sum(r["length"] for r in chain)})
        if radius == max(config["shift_radii"]):
            chosen_runs, chosen_chain = runs, chain
    bridges = []
    for left, right in zip(chosen_chain, chosen_chain[1:]):
        src = old[left["source_end"]:right["source_start"]]
        dst = gold[left["destination_end"]:right["destination_start"]]
        assert len(dst)-len(src) == right["delta"]-left["delta"]
        bridges.append({"source_ids": src, "target_ids": dst, "source_length": len(src),
                        "target_length": len(dst), "delta_before": left["delta"], "delta_after": right["delta"],
                        "source_start": left["source_end"], "destination_start": left["destination_end"]})
    longest = chosen_runs[0] if chosen_runs else None
    c0 = not any(x == y for x, y in zip(old, gold))
    return {"id": row["id"], "prompt_id": row["prompt_id"], "dataset": row["dataset"],
            "offset": row["offset"], "reference_start": start, "old": old, "reference": gold,
            "terminal_correction": terminal, "empty_old": not old, "C0": c0,
            "right_reference_censored": len(raw) < config["diagnostic_horizon"] and not any(t in eos for t in raw),
            "radii": metrics, "runs": chosen_runs, "chain": chosen_chain, "bridges": bridges,
            "prefix_bridge": {"target_ids": gold[:chosen_chain[0]["destination_start"]],
                              "source_ids": old[:chosen_chain[0]["source_start"]]} if chosen_chain else None,
            "longest_right_censored": bool(longest and (longest["source_end"] == len(old) or longest["destination_end"] == len(gold))),
            "diagnostic_only": True}
