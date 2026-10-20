"""Keep diagnostic enrichment separate from design-weighted population claims."""
from collections import defaultdict
import math
from .metrics import event_gains
from .store import atomic_json


def population_estimates(rows, design):
    """Horvitz-Thompson finite-population means, only for fully covered arms."""
    representative = design.get("cohorts", {}).get("representative", {})
    expected = set(representative.get("events", []))
    population = representative.get("available", 0)
    groups = defaultdict(list)
    for row in rows:
        if row["cohort"] == "representative":
            groups[row["redrafter"], row["method"]].append(row)
    output = []
    for (redrafter, method), part in sorted(groups.items()):
        complete = bool(population) and representative.get("population_coverage", False) and {r["event_id"] for r in part} == expected and len(part) == len(expected)
        complete = complete and all(r.get("sampling_weight") is not None and r["sampling_weight"] > 0 for r in part)
        if complete:
            complete = math.isclose(sum(r["sampling_weight"] for r in part), population, rel_tol=1e-9)
        result = {"redrafter": redrafter, "method": method, "sampled_events": len(part),
            "eligible_population_events": population, "estimable": complete,
            "estimator": "Horvitz-Thompson mean over finite eligible Test1 events; representative only"}
        if complete:
            for metric in ("gain", "accepted_prefix", "win", "tie", "loss", "latency_ms", "latency_delta_ms", "t_verify_compute_ms"):
                values = [r.get(metric) for r in part]
                result[metric] = sum(r["sampling_weight"]*v for r, v in zip(part, values))/population if all(v is not None and math.isfinite(v) for v in values) else None
            weighted_a = sum(r["sampling_weight"]*r["accepted_prefix"] for r in part)
            weighted_t = sum(r["sampling_weight"]*r["latency_ms"] for r in part)
            result["accepted_tokens_per_draft_ms_ratio_of_totals"] = weighted_a/weighted_t if weighted_t > 0 else None
        else:
            result["reason"] = "missing sampled outcomes or an eligible stratum has zero representative inclusion"
        output.append(result)
    return {"estimates": output, "uncertainty": "point estimates only; cohort bootstrap elsewhere is descriptive and not a design-based population CI",
            "scope": "conditional on these prompts/eligible rejection events, not all decoding or answer accuracy"}


def write_cohort_report(store, two, frame, destination):
    import pandas as pd
    design = store.get("meta", "test2_design") or {}
    members = [{"event_id": event, **member} for event, member in design.get("members", {}).items()]
    pd.DataFrame(members).to_csv(destination/"sampling_membership.csv", index=False)
    rows = frame.to_dict("records")
    atomic_json(destination/"representative_population_estimates.json", population_estimates(rows, design))
    benefit, efficiency, oracle = [], [], []
    if rows:
        for group, part in frame.groupby(["cohort", "source", "redrafter", "method", "oracle_kind"], dropna=False):
            keys = dict(zip(("cohort", "source", "redrafter", "method", "oracle_kind"), group))
            for outcome, count in part.outcome.value_counts().items():
                benefit.append({**keys, "outcome": outcome, "events": int(count), "fraction": float(count/len(part))})
            draft_ms = float(part.latency_ms.sum())
            verify_ms = float(part.t_verify_compute_ms.sum()) if part.t_verify_compute_ms.notna().all() else None
            verify_wall_ms = float(part.t_verify_wall_ms.sum()) if part.t_verify_wall_ms.notna().all() else None
            accepted = int(part.accepted_prefix.sum())
            efficiency.append({**keys, "events": len(part), "accepted_tokens": accepted,
                "mean_accepted_prefix": float(part.accepted_prefix.mean()), "mean_draft_wall_ms": draft_ms/len(part),
                "mean_verify_compute_ms": verify_ms/len(part) if verify_ms is not None else None,
                "mean_verify_wall_ms": verify_wall_ms/len(part) if verify_wall_ms is not None else None,
                "accepted_tokens_per_draft_ms": accepted/draft_ms if draft_ms else None,
                "accepted_tokens_per_draft_verify_compute_ms": accepted/(draft_ms+verify_ms) if verify_ms is not None and draft_ms+verify_ms > 0 else None,
                "accepted_tokens_per_draft_verify_wall_ms": accepted/(draft_ms+verify_wall_ms) if verify_wall_ms is not None and draft_ms+verify_wall_ms > 0 else None,
                "mean_delta_A": float(part.gain.mean()), "mean_delta_draft_ms": float(part.latency_delta_ms.mean()),
                "paired_arms_share_alias_measurements": True})
    pairs = defaultdict(list)
    for row in two:
        pairs[row["event_id"], row["redrafter"]].append(row)
    for (event, redrafter), part in sorted(pairs.items()):
        base = next((r for r in part if r["method"] == "fresh_L"), None)
        if base is None:
            continue
        oracle.append({"event_id": event, "redrafter": redrafter, "cohort": base["cohort"],
            "correct_count_bin": base.get("correct_count_bin"), "old_correct_count": base.get("old_correct_count"),
            "max_island_length": base.get("old_max_island_length"), "correct_old_positions": base.get("correct_old_positions"),
            "has_old_reuse_opportunity": (base.get("old_correct_count") or 0) > 0,
            **event_gains(part)})
    pd.DataFrame(benefit).to_csv(destination/"benefit_outcomes.csv", index=False)
    pd.DataFrame(efficiency).to_csv(destination/"efficiency.csv", index=False)
    atomic_json(destination/"oracle_comparison.json", oracle)
    return {"schema": design.get("schema", "legacy"), "eligible": design.get("eligible"),
        "groups": {name: {k: v for k, v in c.items() if k not in ("events", "strata")}
                   for name, c in design.get("cohorts", {}).items()},
        "population_estimates_use": "representative only with inverse inclusion weights; never pool enriched"}
