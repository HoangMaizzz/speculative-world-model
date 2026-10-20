"""Disjoint representative and label-enriched cohorts with a frozen design."""
from collections import Counter, defaultdict
import random


def correct_bin(n):
    return "0" if n == 0 else ("1" if n == 1 else ("2-3" if n < 4 else "4+"))


def eligible_records(records, sources):
    return [r for r in records if r["source"] in sources
            and 1 <= r["metrics"]["first_rejection"] < len(r["draft"]["tokens"])]


def stratum(row):
    return row["source"], row["dataset"], row.get("progress_bin", "unknown")


def allocate(capacities, budget, seed):
    """Equal stratum quotas, capacity capped; seeded ties avoid lexical early bias."""
    order = sorted(capacities)
    random.Random(seed).shuffle(order)
    rank = {k: i for i, k in enumerate(order)}
    counts = dict.fromkeys(order, 0)
    source_used, benchmark_used, progress_used = defaultdict(int), defaultdict(int), defaultdict(int)
    for _ in range(min(budget, sum(capacities.values()))):
        available = [k for k in order if counts[k] < capacities[k]]
        key = min(available, key=lambda k: (counts[k] != 0, source_used[k[0]],
                  benchmark_used[k[:2]], progress_used[k[2]], counts[k], rank[k]))
        counts[key] += 1
        source_used[key[0]] += 1
        benchmark_used[key[:2]] += 1
        progress_used[key[2]] += 1
    return counts


def freeze_sample(records, config):
    rows = eligible_records(records, config["test2_sources"])
    total = config["test2_events"]
    targets = {"representative": total//2, "enriched": total-total//2}
    chosen, members, cohorts = set(), {}, {}
    for cohort in ("representative", "enriched"):
        pool = rows if cohort == "representative" else [r for r in rows if r["id"] not in chosen and r["metrics"]["suffix_match_count"] > 0]
        strata = defaultdict(list)
        for row in pool:
            strata[stratum(row)].append(row)
        seed = config["seed"]+(0 if cohort == "representative" else 100003)
        counts = allocate({k: len(v) for k, v in strata.items()}, targets[cohort], seed)
        selected = []
        rng = random.Random(seed+1)
        for key in sorted(strata):
            candidates = sorted(strata[key], key=lambda r: r["id"])
            rng.shuffle(candidates)
            if cohort == "enriched":
                candidates.sort(key=lambda r: (r["metrics"]["max_island_length"], r["metrics"]["suffix_match_count"]), reverse=True)
            n, population = counts[key], len(candidates)
            for row in candidates[:n]:
                event = row["id"]
                selected.append(event)
                chosen.add(event)
                members[event] = {"cohort": cohort, "sampling_stratum": list(key),
                    "inclusion_probability": n/population if cohort == "representative" else None,
                    "sampling_weight": population/n if cohort == "representative" else None}
        coverage = []
        for source in config["test2_sources"]:
            for dataset in config["datasets"]:
                for progress in ("early", "middle", "late"):
                    key = (source, dataset, progress)
                    coverage.append({"source": source, "dataset": dataset, "progress_bin": progress,
                        "available": len(strata.get(key, [])), "selected": counts.get(key, 0),
                        "empty_stratum": not strata.get(key)})
        cohorts[cohort] = {"requested": targets[cohort], "selected": len(selected), "shortfall": targets[cohort]-len(selected),
            "available": len(pool), "events": selected, "strata": coverage,
            "all_available_strata_sampled": bool(pool) and all(counts[k] > 0 for k in counts),
            "population_coverage": cohort == "representative" and bool(pool) and all(counts[k] > 0 for k in counts),
            "selection": "uniform without replacement within source/benchmark/progress strata" if cohort == "representative"
                         else "disjoint positive-old-correct diagnostic; balance strata then rank islands/count; no population weight"}
        selected_rows = [r for r in pool if r["id"] in set(selected)]
        for label, group in (("available_features", pool), ("selected_features", selected_rows)):
            cohorts[cohort][label] = {
                "source": dict(Counter(r["source"] for r in group)),
                "dataset": dict(Counter(r["dataset"] for r in group)),
                "progress": dict(Counter(r.get("progress_bin", "unknown") for r in group)),
                "rejection_bin": dict(Counter(str((r["metrics"]["first_rejection"]-1)//8) for r in group)),
                "correct_count_bin": dict(Counter(correct_bin(r["metrics"]["suffix_match_count"]) for r in group)),
                "max_island_bin": dict(Counter(correct_bin(r["metrics"]["max_island_length"]) for r in group))}
    return {"schema": "two-cohort-v1", "eligible": len(rows), "requested": total,
        "selected": len(chosen), "members": members, "cohorts": cohorts,
        "events": cohorts["representative"]["events"]+cohorts["enriched"]["events"],
        "population_scope": "finite eligible Test1 rejection events only; never pool enriched for population estimates"}
