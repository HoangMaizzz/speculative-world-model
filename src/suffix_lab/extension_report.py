"""Paired quality metrics and escaped token-level case visualizations."""
import csv
from html import escape
import json
from pathlib import Path
from .store import atomic_json


def paired_metrics(rows):
    lookup = {(r["event_id"], r["method"]): r for r in rows}
    paired = []
    for row in rows:
        baseline = lookup.get((row["event_id"], "fresh_"+row["schedule"]))
        short = lookup.get((row["event_id"], row["strategy"]+"_L"))
        if baseline is None:
            continue
        a, f = row["metrics"]["accepted_prefix"], baseline["metrics"]["accepted_prefix"]
        extension_gain = a-short["metrics"]["accepted_prefix"] if short and row["schedule"] != "L" else None
        paired.append({"event_id": row["event_id"], "method": row["method"],
            "baseline": baseline["method"], "delta_A_reuse": a-f,
            "delta_A_extension": extension_gain, "outcome": "win" if a > f else "loss" if a < f else "tie",
            "both_at_ceiling": row["metrics"]["ceiling"] and baseline["metrics"]["ceiling"]})
    return paired


def csv_file(path, records):
    fields = list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields or ["event_id"])
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def case_html(events, rows, paired, tokenizer=None):
    lookup = {(r["event_id"], r["method"]): r for r in rows}
    gains = {}
    for p in paired:
        if p["method"] in ("oracle_H_joint", "oracle_H_sequential"):
            gains.setdefault(p["event_id"], []).append(p["delta_A_reuse"])
    # Fixed case criterion: best oracle gain across the two schedules. Keep all
    # outcomes in CSV; this case selection is descriptive, not a success test.
    ranked = sorted(gains, key=lambda key: (-max(gains[key]), key))
    cases = [("best", e) for e in ranked[:3]] + [("worst", e) for e in ranked[-3:]]
    event_by_id = {e["event_id"]: e for e in events}
    def token(t):
        return tokenizer.decode([t]) if tokenizer is not None else str(t)
    html = ["<!doctype html><html lang='vi'><meta charset='utf-8'><title>Mini extension cases</title>",
        "<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse;margin:12px 0}"
        "td,th{border:1px solid #bbb;padding:6px;white-space:pre-wrap;min-width:24px}"
        ".KEEP{background:#c9edcc}.REDRAFT{background:#fff0b3}.EXTEND{background:#cce4ff}"
        ".reject{outline:3px solid #d22;outline-offset:-3px}.changed{border-bottom:4px solid #555}"
        ".scroll{overflow-x:auto}small{color:#444}</style>",
        "<h1>Anchors + redraft + extension: cases</h1>",
        "<p>KEEP: green; REDRAFT: yellow; EXTEND: blue. Red outline: first mismatch. "
        "Dark underline: old-position token changed. Hover shows token ID and unmask order. "
        "Cases ranked by maximum oracle gain vs schedule-matched Fresh-H; see CSV for all events.</p>"]
    for label, event_id in cases:
        e = event_by_id[event_id]
        html.append("<h2>"+escape(label+": "+event_id)+"</h2><div class='scroll'><table>")
        html.append("<tr><th>Arm / index</th>"+"".join(f"<th>{i}</th>" for i in range(e["H"]))+"</tr>")
        for name, tokens in (("OLD suffix", e["old"]), ("REFERENCE", e["reference"])):
            html.append("<tr><th>"+name+"</th>"+"".join("<td>"+escape(token(t))+"</td>" for t in tokens)+"</tr>")
        for method in (s+"_"+h for s in ("fresh", "oracle", "predicted") for h in ("L", "H_joint", "H_sequential")):
            row = lookup.get((event_id, method))
            if row is None:
                continue
            keep = dict(row["anchors"])
            html.append("<tr><th>"+escape(method)+f"<br>A={row['metrics']['accepted_prefix']}</th>")
            for i, t in enumerate(row["tokens"]):
                cls = ["EXTEND" if i >= e["L"] else "KEEP" if i in keep else "REDRAFT"]
                if i == row["metrics"]["first_mismatch_position_0based"]:
                    cls.append("reject")
                if i < e["L"] and t != e["old"][i]:
                    cls.append("changed")
                title = f"id={t}; index={i}; unmask order={row['order'][i]}"
                html.append("<td class='"+" ".join(cls)+"' title='"+escape(title, quote=True)+"'>"+escape(token(t))+"</td>")
            html.append("</tr>")
        html.append("</table></div>")
    html.append("</html>")
    return "\n".join(html)


def report(output, inputs, store, tokenizer=None):
    output = Path(output)
    rows = store.rows("arm")
    paired = paired_metrics(rows)
    pairing = {(x["event_id"], x["method"]): x for x in paired}
    detailed = []
    for row in rows:
        detailed.append({"event_id": row["event_id"], "method": row["method"], "dataset": row["dataset"],
            "cohort": row["cohort"], "source_drafter": row["source_drafter"], "redrafter": row["redrafter"],
            "suffix_length": row["suffix_length"], "extension_length": row["extension_length"],
            "keep_positions": [i for i, t in row["anchors"]], "anchors": row["anchors"],
            "proposal_token_ids": row["tokens"], "reference_token_ids": row["reference_token_ids"],
            "unmask_order": row["order"], "unmask_trace": row["trace"],
            "composed_drafter_counts": row["stats"], "new_drafter_counts": row["new_drafter_counts"],
            "cached_full_audit_counts": row["cached_full_audit"]["full_prefix_counts"],
            "stage1_reused_from": row.get("stage1_reused_from"), "alias_of": row["alias_of"],
            "verification_mode": row["verification"]["verification_mode"],
            **row["metrics"], **pairing.get((row["event_id"], row["method"]), {})})
    csv_file(output/"arms.csv", detailed)
    csv_file(output/"paired.csv", paired)
    event_rows = []
    for e in inputs["events"]:
        summary = {"event_id": e["event_id"], "dataset": e["dataset"], "cohort": e["cohort"],
                   "L": e["L"], "H": e["H"], "physical_phase": e["physical_phase"],
                   "old_endpoint_phase8": e["old_endpoint_phase8"]}
        for row in rows:
            if row["event_id"] == e["event_id"]:
                summary[row["method"]] = row["metrics"]["accepted_prefix"]
                p = pairing.get((e["event_id"], row["method"]))
                if p:
                    summary[row["method"]+"_delta"] = p["delta_A_reuse"]
        for strategy in ("oracle", "predicted"):
            names = [strategy+"_H_joint", strategy+"_H_sequential", "fresh_H_joint", "fresh_H_sequential"]
            if all(n in summary for n in names):
                a, b, c, d = (summary[n] for n in names)
                summary[strategy+"_schedule_interaction"] = (a-b)-(c-d)
        event_rows.append(summary)
    csv_file(output/"events.csv", event_rows)
    groups = {}
    for method in sorted({p["method"] for p in paired if not p["method"].startswith("fresh_")}):
        items = [p for p in paired if p["method"] == method]
        groups[method] = {"n": len(items), "wins": sum(p["outcome"] == "win" for p in items),
            "ties": sum(p["outcome"] == "tie" for p in items), "losses": sum(p["outcome"] == "loss" for p in items),
            "mean_delta_A": sum(p["delta_A_reuse"] for p in items)/len(items),
            "ceiling_ties": sum(p["both_at_ceiling"] for p in items)}
    result = {"complete": bool(store.get("meta", "complete")), "events": len(inputs["events"]),
        "logical_arms": len(rows), "unique_arms_executed": sum(r["alias_of"] is None for r in rows),
        "groups": groups, "latency_study": False,
        "interpretation": "10 purposive diagnostic events; no population efficacy, speedup, or online-SD claim"}
    atomic_json(output/"summary.json", result)
    (output/"cases.html").write_text(case_html(inputs["events"], rows, paired, tokenizer), encoding="utf-8")
    lines = ["# Mini-test dLLM -> AR: anchors + redraft + extension", "",
        f"Status: {'complete' if result['complete'] else 'partial'}; {len(rows)}/90 logical arms.", "",
        "10 distinct prompts: 5 GSM8K + 5 MATH, 6 enriched + 4 representative; old draft and redrafter are dLLM only.",
        "FP16 verifier on 2 GPUs, one dLLM on GPU1; one MASK unmasked per forward in earliest absolute subblock8.",
        "Reference/Test1/pilot event metadata reused. Fresh-L replayed for full trace and exact saved-baseline audit.",
        "Each unique generated stage passes cached/full-prefix equality; actual AR verifier rechecks every unique final proposal.",
        "", "| Arm | N | Win | Tie | Loss | Mean delta A | Ceiling ties |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for method, g in groups.items():
        lines.append(f"| {method} | {g['n']} | {g['wins']} | {g['ties']} | {g['losses']} | {g['mean_delta_A']:.2f} | {g['ceiling_ties']} |")
    lines.extend(["", "Comparisons use Fresh with the SAME horizon and schedule. Oracle keeps only existing correct old tokens; it is not a globally optimal mask.",
        "Predicted spans use only old verifier agreement, old-token probability >=0.8 and drafter confidence >=0.5; score is mean log geometric confidence.",
        "At most 2 spans of at least 2 tokens, total KEEP <=floor(L/4). Empty/identical masks are explicit aliases.",
        "Sequential reuses its completed short arm and locks ALL L tokens before generating E=8. Counts distinguish composed work, newly executed work, and audit work.",
        "Joint may affect old tokens only under actual block/subblock geometry; MASK padding already exists in short arms. See traces and visibility counts.",
        "", "No latency measurement or throughput claim. Reference-triggered canonical target replay is an OFFLINE diagnostic policy.",
        "All 10 outcomes: events.csv; all logical arms/traces: arms.csv and arms.jsonl; paired gains: paired.csv; best/worst token cases: cases.html."])
    (output/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return result
