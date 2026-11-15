"""Exact-token oracle analysis of rejected paths from a native EAGLE-3 tree."""

from collections import Counter

HORIZONS = (10, 32, 64)
MINIMUM = 4


def first_difference(first, second):
    for index, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return index
    return None if len(first) == len(second) else min(len(first), len(second))


def rejected_suffixes(row, eos):
    """Use only tokens after each path's first rejected node, respecting EOS."""
    tokens, depths = row["tokens"], row["depths"]
    committed, next_seed = row["committed"], row["next_seed"]
    prefix = committed + [next_seed]
    branches = []
    for path_index, path in enumerate(row["paths"]):
        divergence = first_difference([tokens[node] for node in path[:len(prefix)]], prefix)
        if divergence is None or divergence == len(path):
            continue  # This leaf ends on the accepted prefix.
        if divergence == 0 or divergence >= len(prefix):
            raise AssertionError("Native verifier root/path alignment failed")
        wrong = path[divergence]
        suffix = []
        if tokens[wrong] not in eos:
            for node in path[divergence + 1:]:
                if tokens[node] in eos:
                    break
                suffix.append(node)
        if suffix:
            branches.append(dict(path=path_index, first_rejected=wrong, nodes=suffix,
                                 source_start=depths[suffix[0]] - len(prefix)))
    return branches


def find_runs(row, branches, future):
    """Never match across branches, even when identical tokens occur at sibling nodes."""
    from suffix_lab.long60_plan import matching_runs

    matches = {}
    for branch in branches:
        nodes = branch["nodes"]
        source = [row["tokens"][i] for i in nodes]
        runs = matching_runs(source, future, max(len(source), len(future)),
                             minimum=MINIMUM, include_zero=True)
        for run in runs:
            start, end = run["source_start"], run["source_end"]
            source_nodes = tuple(nodes[start:end])
            key = (source_nodes, run["destination_start"])
            if key not in matches:
                shift = branch["source_start"]
                matches[key] = dict(run, source_start=start + shift, source_end=end + shift,
                                    delta=run["delta"] - shift, source_nodes=list(source_nodes), paths=[])
            matches[key]["paths"].append(branch["path"])
    return list(matches.values())


def diagnose(spec, reference, eos):
    from suffix_lab.long60_plan import until_eos

    gold = reference["token_ids"]
    divergence = first_difference(spec["token_ids"], gold)
    shared_end = len(gold) if divergence is None else divergence
    results = {str(h): {"events": [], "attrition": Counter()} for h in HORIZONS}
    for index, row in enumerate(spec["rounds"]):
        correction = row["offset"] + len(row["committed"])
        start = correction + 1
        reason = None
        if row["committed"][-1] in eos or row["next_seed"] in eos:
            reason = "eos"
        elif correction >= len(gold):
            reason = "reference_ended_before_correction"
        elif start > shared_end or row["next_seed"] != gold[correction]:
            reason = "outside_greedy_parity"
        if reason:
            for result in results.values():
                result["attrition"][reason] += 1
            continue
        branches = rejected_suffixes(row, eos)
        for horizon in HORIZONS:
            result = results[str(horizon)]
            window = gold[start:start + horizon]
            future = until_eos(window, eos, include=False)
            if len(future) < MINIMUM:
                result["attrition"]["short_reference"] += 1
                continue
            result["events"].append(dict(round=index, corrected_offset=start, reference=future,
                                         right_censored=len(future) == len(window) < horizon,
                                         best_path=row["best_path"], branches=branches,
                                         runs=find_runs(row, branches, future)))
    return results


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_reuse(cases, horizon, selected_only):
    counts, attrition, lengths = Counter(), Counter(), Counter()
    unique_ref, unique_same, unique_shift = set(), set(), set()
    for case in cases:
        analysis = case["reuse"][str(horizon)]
        attrition.update(analysis["attrition"])
        prompt_has_shift = False
        for event in analysis["events"]:
            select = event["best_path"]
            branches = [b for b in event["branches"] if not selected_only or b["path"] == select]
            runs = [r for r in event["runs"] if not selected_only or select in r["paths"]]
            source = {node for branch in branches for node in branch["nodes"]}
            found = {node for run in runs for node in run["source_nodes"]}
            destinations, same, shifted = set(), set(), set()
            for run in runs:
                left, right = run["destination_start"], run["destination_end"]
                destinations.add((left, right))
                (shifted if run["delta"] else same).update(range(left, right))
            for left, right in destinations:
                lengths[right - left] += 1
            anchor_capable = any(len(b["nodes"]) >= MINIMUM for b in branches)
            counts.update(eligible_rounds=1, rounds_with_long_source=anchor_capable,
                          rounds_with_anchor=bool(runs), rounds_with_same=bool(same),
                          rounds_with_shift=bool(shifted), rounds_with_shift_only=bool(shifted) and not same,
                          censored_windows=event["right_censored"], reference_tokens=len(event["reference"]),
                          same_positions=len(same), shifted_positions=len(shifted),
                          either_positions=len(same | shifted), extra_shift_positions=len(shifted - same),
                          source_nodes=len(source), reusable_source_nodes=len(found))
            offset = event["corrected_offset"]
            unique_ref.update((case["id"], offset + i) for i in range(len(event["reference"])))
            unique_same.update((case["id"], offset + i) for i in same)
            unique_shift.update((case["id"], offset + i) for i in shifted)
            prompt_has_shift |= bool(shifted)
        counts["prompts_with_shift"] += prompt_has_shift
    for key in ("eligible_rounds", "rounds_with_long_source", "rounds_with_anchor", "rounds_with_same",
                "rounds_with_shift", "rounds_with_shift_only", "reference_tokens", "same_positions",
                "shifted_positions", "either_positions", "extra_shift_positions", "source_nodes",
                "reusable_source_nodes", "censored_windows", "prompts_with_shift"):
        counts.setdefault(key, 0)
    return dict(counts=counts, attrition=attrition, distinct_anchor_lengths=dict(sorted(lengths.items())),
                anchor_round_rate=ratio(counts["rounds_with_anchor"], counts["eligible_rounds"]),
                anchor_rate_given_long_source=ratio(counts["rounds_with_anchor"], counts["rounds_with_long_source"]),
                window_coverage=ratio(counts["either_positions"], counts["reference_tokens"]),
                extra_shift_coverage=ratio(counts["extra_shift_positions"], counts["reference_tokens"]),
                reusable_source_node_rate=ratio(counts["reusable_source_nodes"], counts["source_nodes"]),
                unique_reference_tokens=len(unique_ref), unique_covered_tokens=len(unique_same | unique_shift),
                unique_shift_extra_tokens=len(unique_shift - unique_same),
                unique_token_coverage=ratio(len(unique_same | unique_shift), len(unique_ref)))


def summarize(cases):
    result = {}
    for dataset in ("all", "math500", "gsm8k"):
        group = [c for c in cases if dataset == "all" or c["dataset"] == dataset]
        rounds = [r for c in group for r in c["spec"]["rounds"]]
        ar_ms = sum(c["ar"]["wall_ms"] for c in group)
        tree_ms = sum(c["spec"]["wall_ms"] for c in group)
        result[dataset] = dict(
            prompts=len(group), exact_greedy_prompts=sum(c["first_difference"] is None for c in group),
            verify_rounds=len(rounds), ar_wall_ms=ar_ms, tree_trace_wall_ms=tree_ms,
            accepted_draft_per_round=ratio(sum(r["accepted_draft"] for r in rounds), len(rounds)),
            committed_per_round=ratio(sum(len(r["committed"]) for r in rounds), len(rounds)),
            mean_verify_ms=ratio(sum(r["verify_ms"] for r in rounds), len(rounds)),
            max_submitted_draft_depth=max((max(r["depths"]) for r in rounds), default=None),
            reuse={str(h): {"whole_tree": summarize_reuse(group, h, False),
                            "selected_path": summarize_reuse(group, h, True)} for h in HORIZONS})
    return result
