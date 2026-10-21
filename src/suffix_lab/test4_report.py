"""Main200 paired, prompt-clustered statistics; terminal strata stay explicit."""
from collections import Counter
from html import escape
import json
from pathlib import Path
import random
from statistics import mean, median
from .extension_report import csv_file
from .store import atomic_json, digest


def percentile(values, q):
    ordered = sorted(values)
    x = (len(ordered)-1)*q
    i = int(x)
    return ordered[i]+(ordered[min(i+1, len(ordered)-1)]-ordered[i])*(x-i)


def paired_stats(rows, samples=2000, seed=42):
    if not rows:
        return {"n": 0, "mean_delta_A": None, "median_delta_A": None,
                "mean_delta_ci95": None, "median_delta_ci95": None}
    if len({r["prompt_id"] for r in rows}) != len(rows):
        raise ValueError("Main bootstrap expects one paired event per prompt")
    values = [r["delta"] for r in rows]
    rng = random.Random(seed)
    means, medians = [], []
    for _ in range(samples):
        draw = rng.choices(values, k=len(values))
        means.append(mean(draw)); medians.append(median(draw))
    return {"n": len(rows), "mean_delta_A": mean(values), "median_delta_A": median(values),
        "mean_delta_ci95": [percentile(means, .025), percentile(means, .975)],
        "median_delta_ci95": [percentile(medians, .025), percentile(medians, .975)],
        "wins": sum(v > 0 for v in values), "ties": sum(v == 0 for v in values),
        "losses": sum(v < 0 for v in values),
        "mean_fresh": mean(r["fresh"] for r in rows), "mean_method": mean(r["accepted"] for r in rows),
        "mean_content": mean(r["A_content"] for r in rows),
        "full_23_rate": mean(r["full_23_accepted"] for r in rows),
        "terminated_correctly_rate": mean(r["terminated_correctly"] for r in rows),
        "mean_accepted_extension": mean(r["accepted_extension"] for r in rows)}


def report(output, store, config):
    output = Path(output)
    events, arms, screens = store.rows("event"), store.rows("arm"), store.rows("screen")
    lookup = {(a["event_id"], a["method"]): a for a in arms}
    paired, details = [], []
    names = ("F-23", "O-All-Joint-23", "KEEP-All-23")
    for event in events:
        rows = {m: lookup.get((event["event_id"], m)) for m in names}
        if not all(rows.values()):
            continue
        base = rows["F-23"]["metrics"]["A_inclusive"]
        for method in names:
            m = rows[method]["metrics"]
            paired.append({"event_id": event["event_id"], "prompt_id": event["prompt_id"],
                "dataset": event["dataset"], "method": method, "fresh": base,
                "accepted": m["A_inclusive"], "delta": m["A_inclusive"]-base,
                "reuse_applicable": event["reuse_applicable"],
                "terminal": event["target_terminal_within_horizon"],
                **{k: m[k] for k in ("A_content", "full_23_accepted", "terminated_correctly", "accepted_extension")}})
        details.append({"event_id": event["event_id"], "dataset": event["dataset"],
            "a": event["accepted_before_rejection"], "L": event["L"], "E": event["extension_length"],
            "correct_count": event["old_correct_count"], "terminal": event["target_terminal_within_horizon"],
            "physical_phase": event["physical_phase"], "old_endpoint_phase8": event["old_endpoint_phase8"],
            **{m: rows[m]["metrics"]["A_inclusive"] for m in names},
            "delta": rows[names[1]]["metrics"]["delta_A_vs_fresh"]})
    summaries = []
    for dataset in config["datasets"]:
        for stratum in ("all_eligible", "reuse_applicable", "nonterminal_full_horizon", "target_terminal"):
            for method in names:
                subset = [r for r in paired if r["dataset"] == dataset and r["method"] == method
                    and (stratum == "all_eligible" or stratum == "reuse_applicable" and r["reuse_applicable"]
                         or stratum == "nonterminal_full_horizon" and not r["terminal"]
                         or stratum == "target_terminal" and r["terminal"])]
                summaries.append({"dataset": dataset, "stratum": stratum, "method": method,
                    **paired_stats(subset, config["bootstrap_samples"], digest([config["seed"], dataset, stratum, method]))})
    exclusions = Counter(s.get("reason") for s in screens if s["status"] != "eligible")
    prompt_done = store.rows("prompt_done")
    attrition = {"prompts_examined": len(prompt_done), "prompts_with_screening": len({s["prompt_id"] for s in screens}),
        "prompt_exhaustions": sum(p["chosen_event"] is None for p in prompt_done),
        "native_drafts_screened": sum(s["native_executed"] for s in screens),
        "states_considered": len(screens), "eligible_states": sum(s["status"] == "eligible" for s in screens),
        "exclusions": dict(exclusions), "terminal_correction_count": exclusions["terminal_correction"],
        "early_draft_eos_count": exclusions["early_draft_eos"],
        "reference_censored_states": exclusions["reference_censored"],
        "selected_events": len(events), "complete_paired_events": len(details),
        "selected_by_benchmark": dict(Counter(e["dataset"] for e in events)),
        "a_histogram": dict(Counter(e["accepted_before_rejection"] for e in events)),
        "L_histogram": dict(Counter(e["L"] for e in events)),
        "E_histogram": dict(Counter(e["extension_length"] for e in events)),
        "correct_count_histogram": dict(Counter(e["old_correct_count"] for e in events)),
        "C_zero_events": sum(e["old_correct_count"] == 0 for e in events),
        "target_terminal_events": sum(e["target_terminal_within_horizon"] for e in events),
        "reference_stop_reasons": dict(Counter(r["stop_reason"] for r in store.rows("reference")))}
    result = {"complete": bool(store.get("meta", "complete")), "completed_arms": len(arms),
        "expected_arms": config["events_per_dataset"]*len(config["datasets"])*3,
        "unique_executed_arms": sum(a["alias_of"] is None for a in arms), "attrition": attrition,
        "summary": summaries, "paired_events": details,
        "anchor_tokens": store.get("meta", "anchor_tokens") or {},
        "interpretation": "Fixed reference-prefix offsets; eligible rejection events; offline Oracle diagnostic, not online throughput or selector precision threshold",
        "bootstrap": "percentile 95% paired prompt bootstrap; one Main event/prompt; benchmark and terminal strata kept separate",
        "EOS_policy": "full native tensor computed; logical proposals cut at configured EOS before actual verification; no post-EOS acceptance or early-stop savings"}
    atomic_json(output/"summary.json", result)
    atomic_json(output/"attrition.json", attrition)
    csv_file(output/"summary.csv", summaries)
    csv_file(output/"events.csv", details)
    csv_file(output/"paired.csv", paired)
    flat = [{"id": a["id"], "method": a["method"], "arm_fingerprint": a["arm_fingerprint"],
        "alias_of": a["alias_of"], **{k: v for k, v in a["metrics"].items() if not isinstance(v, (dict, list))}}
        for a in arms]
    csv_file(output/"arms.csv", flat)
    lines = ["# Test 4 — Phase 1 Main evaluation", "",
        f"Complete: **{result['complete']}**. Events {len(events)}; fully paired {len(details)}; arms {len(arms)}/{result['expected_arms']}.",
        "", "Offline dLLM → AR; FP16; strict one token/unmask. Oracle future labels are diagnostic. No latency/speedup claim.",
        "", "| Benchmark | Stratum | Arm | N | Mean A Fresh → arm | Mean ΔA [95% CI] | Median ΔA | W/T/L |",
        "|---|---|---|---:|---|---|---|---|"]
    for row in summaries:
        if row["n"]:
            lo, hi = row["mean_delta_ci95"]
            lines.append(f"| {row['dataset']} | {row['stratum']} | {row['method']} | {row['n']} | {row['mean_fresh']:.3f} → {row['mean_method']:.3f} | {row['mean_delta_A']:.3f} [{lo:.3f}, {hi:.3f}] | {row['median_delta_A']} | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines += ["", "## Sampling / attrition", "", "```json", json.dumps(attrition, ensure_ascii=False, indent=2), "```", "",
        "A_inclusive includes correctly reached EOS; A_content excludes it. Correction is committed context and is not counted. Terminal horizons have a shorter ceiling.",
        "KEEP-All never uses future target EOS to choose anchors; it excludes EOS/control terminators in the actual old suffix. Oracle has only old tokens matching valid pre-terminal reference positions.",
        "Full capacity is still computed by the unchanged native scheduler. Post-EOS native tensor tokens are not verified, committed or treated as recyclable suffix.",
        "Raw mutable-position changes and accepted gain attribution are in arms.jsonl; correct token pieces retain whitespace in anchor_tokens.json.",
        "If partial: statistics describe completed pairs only, not the intended final Main200 cohort. Screening/audits/reference replay are extra work beyond 600 logical arms."]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    html = ["<!doctype html><meta charset='utf-8'><title>Test4 Main cases</title><style>body{font:15px system-ui;margin:24px}pre{white-space:pre-wrap}details{margin:12px}</style>",
            "<h1>Test4 Main — token IDs and EOS-aware verification</h1>"]
    for row in sorted(details, key=lambda r: (-r["delta"], r["event_id"])):
        e = next(e for e in events if e["event_id"] == row["event_id"])
        view = {"event": row, "question": e["question"], "reference": e["reference"],
                "old": e["old"], "anchors": e["oracle_anchors"],
                "arms": {m: {"logical_tokens": lookup[(e["event_id"], m)]["metrics"]["logical_tokens"],
                              "metrics": lookup[(e["event_id"], m)]["metrics"]} for m in names}}
        html += ["<details><summary>"+escape(e["event_id"])+f" Δ={row['delta']}</summary><pre>"+
                 escape(json.dumps(view, ensure_ascii=False, indent=2))+"</pre></details>"]
    (output/"cases.html").write_text("\n".join(html), encoding="utf-8")
    if details:
        plots(output, details, events)
    return result


def plots(output, details, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ds, color in (("gsm8k", "#2864b4"), ("math500", "#e08030")):
        rows = [r for r in details if r["dataset"] == ds]
        if not rows:
            continue
        axes[0, 0].hist([r["delta"] for r in rows], bins=list(range(-24, 25)), alpha=.6, label=ds, color=color)
        counts = [sum(r["delta"] > 0 for r in rows), sum(r["delta"] == 0 for r in rows), sum(r["delta"] < 0 for r in rows)]
        offset = -.18 if ds == "gsm8k" else .18
        axes[0, 1].bar([i+offset for i in range(3)], counts, width=.35, label=ds, color=color)
    axes[0, 0].set(xlabel="Oracle − Fresh accepted tokens", ylabel="Events", title="Paired Main gain")
    axes[0, 0].legend()
    axes[0, 1].set(xticks=[0, 1, 2], xticklabels=["Win", "Tie", "Loss"], ylabel="Events", title="Oracle vs Fresh")
    axes[0, 1].legend()
    axes[1, 0].hist([e["old_correct_count"] for e in events], bins=list(range(25)), color="#409878")
    axes[1, 0].set(xlabel="Old correct non-EOS anchors", ylabel="Selected events")
    axes[1, 1].hist([e["accepted_before_rejection"] for e in events], bins=list(range(25)), color="#8054a0")
    axes[1, 1].set(xlabel="Accepted before original rejection (a)", ylabel="Selected events")
    fig.tight_layout(); fig.savefig(output/"main_distributions.png", dpi=160); plt.close(fig)
