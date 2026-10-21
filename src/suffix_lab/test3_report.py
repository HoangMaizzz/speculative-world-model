"""Test3 quality/headroom report; forced answers are not conditioning gains."""
from collections import Counter
from html import escape
import json
from pathlib import Path
from .extension_report import csv_file
from .store import atomic_json


def cases_html(events, arms, comparisons, tokenizer=None):
    lookup = {(r["event_id"], r["method"]): r for r in arms}
    ranked = sorted(comparisons, key=lambda e: (-e["delta_joint_vs_fresh"], e["event_id"]))
    best, worst = {e["event_id"] for e in ranked[:3]}, {e["event_id"] for e in ranked[-3:]}
    def tok(t):
        return tokenizer.decode([t]) if tokenizer is not None else str(t)
    html = ["<!doctype html><html lang='vi'><meta charset='utf-8'><title>Test3 Oracle All-Correct</title>",
        "<style>body{font:15px system-ui;margin:24px}.scroll{overflow-x:auto}table{border-collapse:collapse;margin:10px 0}"
        "td,th{border:1px solid #bbb;padding:6px;white-space:pre-wrap;min-width:24px}"
        ".KEEP{background:#c9edcc}.REDRAFT{background:#fff0b3}.EXTEND{background:#cce4ff}.ACCEPT{background:#ddd}"
        ".reject{outline:3px solid #d22;outline-offset:-3px}.changed{border-bottom:4px solid #555}"
        "details{margin:8px 0}small{color:#444}</style>",
        "<h1>Test 3 — Oracle-All-Correct Refill</h1>",
        "<p>Green KEEP; yellow REDRAFT; blue EXTEND; red outline first mismatch; dark underline changed old token. "
        "Correction C belongs to context and is excluded from accepted-draft counts. All indices below are 0-based.</p>"]
    for e in events:
        eid = e["event_id"]
        tag = (" BEST" if eid in best else "")+(" WORST" if eid in worst else "")
        html.append("<h2>"+escape(eid+" / "+e["category"]+tag)+"</h2><p>"+escape(e.get("question", ""))+"</p>")
        html.append(f"<p>a={e['accepted_before_rejection']}; r={e['first_rejection_1based']} (1-based); L={e['L']}; E={e['extension_length']}; H=23. "
                    +"Corrected prefix last token C="+escape(tok(e["correction_token"]))+f" (ID {e['correction_token']}). "
                    +f"All {len(e['oracle_anchors'])} old-correct tokens retained, without island/ratio cap.</p>")
        html.append("<div class='scroll'><table><tr><th>Original native D24</th>")
        for i, t in enumerate(e["original_draft"]["tokens"]):
            cls = "ACCEPT" if i < e["accepted_before_rejection"] else "reject" if i == e["accepted_before_rejection"] else "REDRAFT"
            html.append(f"<td class='{cls}' title='old index={i}; ID={t}'>"+escape(tok(t))+"</td>")
        html.append("</tr></table></div><div class='scroll'><table><tr><th>Next proposal index</th>"+"".join(f"<th>{i}</th>" for i in range(e["H"]))+"</tr>")
        for label, values in (("OLD suffix", e["old"]), ("Corrected reference", e["reference"]),
                              ("Oracle initial canvas", e["initial_oracle_joint_proposal"])):
            html.append("<tr><th>"+label+"</th>")
            for i, t in enumerate(values):
                mask = label == "Oracle initial canvas" and i not in dict(e["oracle_anchors"])
                cls = "EXTEND" if i >= e["L"] else "KEEP" if i in dict(e["oracle_anchors"]) else "REDRAFT"
                html.append(f"<td class='{cls}'>"+("MASK" if mask else escape(tok(t)))+"</td>")
            html.append("</tr>")
        for name in ("F-23", "O-Joint-23", "O-Sequential-23", "F-L", "O-L", "KEEP-All-23"):
            row = lookup.get((eid, name))
            if not row:
                continue
            keep = dict(row["anchors"])
            html.append("<tr><th>"+escape(name)+f"<br>A={row['metrics']['accepted_prefix']}</th>")
            for i, t in enumerate(row["tokens"]):
                classes = ["EXTEND" if i >= e["L"] else "KEEP" if i in keep else "REDRAFT"]
                if row["metrics"]["first_mismatch_position_0based"] == i:
                    classes.append("reject")
                if i < e["L"] and t != e["old"][i]:
                    classes.append("changed")
                title = f"ID={t}; index={i}; unmask order={row['order'][i]}"
                html.append("<td class='"+" ".join(classes)+"' title='"+escape(title, quote=True)+"'>"+escape(tok(t))+"</td>")
            html.append("</tr>")
        html.append("</table></div>")
        for name in ("O-Joint-23", "O-Sequential-23"):
            row = lookup.get((eid, name))
            if not row:
                continue
            html.append("<details><summary>"+escape(name)+" actual unmask/visibility trace</summary><div class='scroll'><table>"
                "<tr><th>Step</th><th>Position</th><th>Region</th><th>Absolute</th><th>Subblock</th><th>Prediction row</th><th>Carry</th><th>Visible future KEEP</th><th>Visible generated future EXT</th></tr>")
            for s in row["trace"]:
                values = [s["step"], s["position"], s["region"], s["absolute_position"], s["subblock"],
                          s["prediction_absolute_row"], s["uses_previous_block_carry"], s["visible_future_keep_positions"], s["visible_generated_future_extension"]]
                html.append("<tr>"+"".join("<td>"+escape(str(v))+"</td>" for v in values)+"</tr>")
            html.append("</table></div></details>")
    html.append("</html>")
    return "\n".join(html)


def report(output, store, config, tokenizer=None):
    output = Path(output)
    events, arms, screens = store.rows("event"), store.rows("arm"), store.rows("screen")
    lookup = {(r["event_id"], r["method"]): r for r in arms}
    comparisons, details = [], []
    for e in events:
        names = ("F-23", "O-Joint-23", "O-Sequential-23", "F-L", "O-L", "KEEP-All-23")
        rows = {name: lookup.get((e["event_id"], name)) for name in names}
        if not all(rows.values()):
            continue
        accepted = {name: row["metrics"]["accepted_prefix"] for name, row in rows.items()}
        m = rows["O-Joint-23"]["metrics"]
        comparisons.append({"event_id": e["event_id"], "dataset": e["dataset"], "category": e["category"],
            "a": e["accepted_before_rejection"], "L": e["L"], "E": e["extension_length"],
            "old_correct": e["old_correct_count"], **accepted,
            "delta_joint_vs_fresh": accepted["O-Joint-23"]-accepted["F-23"],
            "delta_joint_vs_sequential": accepted["O-Joint-23"]-accepted["O-Sequential-23"],
            "delta_old": accepted["O-L"]-accepted["F-L"],
            "extension_accepted": m["accepted_extension"], "both_at_ceiling": m["ceiling"] and rows["F-23"]["metrics"]["ceiling"],
            **m["accepted_gain_attribution"],
            "joint_sequential_identical_tokens": rows["O-Joint-23"]["tokens"] == rows["O-Sequential-23"]["tokens"]})
    for row in arms:
        details.append({"event_id": row["event_id"], "method": row["method"], "dataset": row["dataset"], "category": row["category"],
            "L": row["L"], "horizon": row["H"], "extension_length": row["extension_length"], "anchors": row["anchors"],
            "tokens": row["tokens"], "reference_token_ids": row["reference_token_ids"], "unmask_order": row["order"], "unmask_trace": row["trace"],
            "composed_counts": row["stats"], "new_drafter_counts": row["new_drafter_counts"],
            "audit_counts": row["cached_full_audit"]["full_prefix_counts"], "alias_of": row["alias_of"],
            "stage1_reused_from": row.get("stage1_reused_from"), "verification_mode": row["verification"]["verification_mode"], **row["metrics"]})
    summaries = []
    for name in ("F-23", "O-Joint-23", "O-Sequential-23", "F-L", "O-L", "KEEP-All-23"):
        rows = [r for r in arms if r["method"] == name]
        if not rows:
            continue
        n = len(rows)
        summaries.append({"method": name, "n": n, "mean_accepted": sum(r["metrics"]["accepted_prefix"] for r in rows)/n,
            "mean_positionwise_correct": sum(r["metrics"]["positionwise_correct"] for r in rows)/n,
            "mean_extension_accepted": sum(r["metrics"]["accepted_extension"] for r in rows)/n,
            "wins_vs_fresh": sum(r["metrics"]["delta_A_vs_fresh"] > 0 for r in rows),
            "ties_vs_fresh": sum(r["metrics"]["delta_A_vs_fresh"] == 0 for r in rows),
            "losses_vs_fresh": sum(r["metrics"]["delta_A_vs_fresh"] < 0 for r in rows),
            "mean_delta_A": sum(r["metrics"]["delta_A_vs_fresh"] for r in rows)/n,
            "ceiling_count": sum(r["metrics"]["ceiling"] for r in rows)})
    csv_file(output/"summary.csv", summaries)
    csv_file(output/"events.csv", comparisons)
    csv_file(output/"arms.csv", details)
    csv_file(output/"screening.csv", [{"event_id": s["event_id"], "screen_index": s["screen_index"], "status": s["status"],
        "category": s.get("category"), "reason": s.get("reason"),
        "fresh23_A": s.get("event", {}).get("fresh23_accepted"),
        "old_correct_count": s.get("event", {}).get("old_correct_count")} for s in screens])
    complete = bool(store.get("meta", "complete"))
    summary = {"complete": complete, "selected_events": len(events), "logical_arms": len(arms),
        "unique_executed_arms": sum(r["alias_of"] is None for r in arms), "screened_states": len(screens),
        "screen_categories": dict(Counter(s.get("category", "skipped") for s in screens)),
        "selection": store.get("meta", "selection") or store.get("meta", "availability"),
        "summary": summaries, "comparisons": comparisons, "latency_study": False,
        "interpretation": "Purpose-selected offline all-correct oracle headroom; no population, deployable-policy, or throughput claim"}
    atomic_json(output/"summary.json", summary)
    (output/"cases.html").write_text(cases_html(events, arms, comparisons, tokenizer), encoding="utf-8")
    lines = ["# Test 3 — Oracle-All-Correct Sliding-Window Refill", "",
        f"Status: {'complete' if complete else 'partial'}; {len(events)} events; {len(arms)}/{config['events']*6} logical arms; {len(screens)} screened states.",
        "", "Native original proposal B=24 regenerated; no B32 truncation. One correction enters context; next unverified horizon H=23; L=23-a, E=a, 3<=a<=15.",
        "All old-correct tokens are KEEP, including singleton islands, with no ratio/island budget. Other old positions and all extension positions start MASK.",
        "Strict one token/unmask forward, physical32/earliest absolute subblock8. Full future visibility is NOT assumed. Selected original B24 and each unique arm/stage pass cached/full audits.",
        "Corrected reference is reused from the canonical pilot trajectory and ALL 23 rows are audited against actual target predictions. Final proposals are independently reverified. Reference-triggered numerical replay remains offline diagnostic.",
        "", "| Arm | N | Mean A | Win | Tie | Loss | Mean positionwise correct |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['method']} | {s['n']} | {s['mean_accepted']:.2f} | {s['wins_vs_fresh']} | {s['ties_vs_fresh']} | {s['losses_vs_fresh']} | {s['mean_positionwise_correct']:.2f} |")
    lines.extend(["", "| Event | Group | Old correct/L | Fresh23 | Oracle Joint | Oracle Seq | Delta Joint | Extension accepted | Direct KEEP corrections | Mutable corrections in added prefix |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for e in comparisons:
        lines.append(f"| {e['event_id']} | {e['category']} | {e['old_correct']}/{e['L']} | {e['F-23']} | {e['O-Joint-23']} | {e['O-Sequential-23']} | {e['delta_joint_vs_fresh']:+d} | {e['extension_accepted']} | {e['direct_keep_corrected']} | {e['conditioning_redraft_corrected']+e['conditioning_extension_corrected']} |")
    lines.extend(["", "Primary contrast is O-Joint-23 minus F-23, same corrected prefix/horizon/joint schedule. O-Sequential minus F-23 also changes schedule; interpret it descriptively. F-L and O-L use each event's actual L, not fixed16.",
        "Selection: exact 4 early-rescue, 3 later-anchor, 2 low-reuse, 1 ceiling, 5+5 benchmarks/10 distinct prompts; no oracle repair outcomes used. Definitions and thresholds are frozen in config; no silent quota relaxation.",
        "Forced correct anchors can directly replace Fresh errors. Gain attribution separates that answer injection from mutable REDRAFT/EXTEND corrections and already-correct positions bridged into acceptance; no causal claim for every gained token.",
        "Every non-KEEP position has Fresh-wrong/Oracle-correct, harmful, both-wrong, both-correct and token-change counts, split old/extension; see arms.csv/jsonl.",
        "All 10 traces and three best/worst cases (rank by primary delta, stable event-ID ties): cases.html. Old24 -> rejection -> correction -> all-correct mask -> unmask order -> proposal -> target verification are retained.",
        "If quotas cannot be met, selection.json/screening.csv report the shortfall and this is not a completed 60-arm result. Screening work is additional to the 60 experimental arms, logged separately.",
        "No latency study. Diagnostic selection is not population sampling; positive results are offline oracle headroom only. Native padding and right-shift/carry boundaries constrain future conditioning."])
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary
