"""Descriptive post-hoc reports; all labels stay outside online decoding."""
from collections import Counter
import csv
import html
import json
from statistics import mean, median
from .store import atomic_json


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(output, store, c, tokenizer=None):
    rounds, prompts, diagnostics = store.rows("round"), store.rows("prompt"), store.rows("diagnostic")
    pieces_path = output/"token_pieces.json"
    pieces = json.loads(pieces_path.read_text(encoding="utf-8")) if pieces_path.exists() else {}
    decoded_path = output/"decoded_sequences.json"
    decoded = json.loads(decoded_path.read_text(encoding="utf-8")) if decoded_path.exists() else {}
    token_ids = {t for row in diagnostics for t in row["old"]+row["reference"]}
    if tokenizer is not None:
        for token in token_ids:
            if str(token) not in pieces:
                pieces[str(token)] = tokenizer.decode([token], skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False)
        atomic_json(pieces_path, pieces)
    def decode(ids):
        key = ",".join(map(str, ids))
        if key in decoded:
            return decoded[key]
        if tokenizer is not None:
            decoded[key] = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return decoded[key]
        return "".join(pieces.get(str(t), f"<ID:{t}>") for t in ids)
    completed = {p["id"] for p in prompts}
    audited_rounds = [r for r in rounds if r["prompt_id"] in completed]
    eligible = [d for d in diagnostics if not d["terminal_correction"] and d["old"] and d["reference"]]
    summaries = []
    for ds in c["datasets"]+["pooled"]:
        selected = [d for d in eligible if ds == "pooled" or d["dataset"] == ds]
        ps = [p for p in prompts if ds == "pooled" or p["dataset"] == ds]
        rs = [r for r in audited_rounds if ds == "pooled" or r["dataset"] == ds]
        for radius in c["shift_radii"]:
            values = [next(r for r in d["radii"] if r["radius"] == radius) for d in selected]
            lengths = [r["longest"] for r in values if r["has_run"]]
            summaries.append({"dataset": ds, "radius": radius, "audited_prompts": len(ps),
                "audited_rounds": len(rs), "eligible_rejections": len(selected),
                "events_with_run": sum(r["has_run"] for r in values),
                "events_with_2plus_blocks": sum(r["blocks"] >= 2 for r in values),
                "events_with_3plus_blocks": sum(r["blocks"] >= 3 for r in values),
                "mean_longest_given_hit": mean(lengths) if lengths else None,
                "median_longest_given_hit": median(lengths) if lengths else None,
                "total_chain_tokens": sum(r["covered_tokens"] for r in values),
                "old_tokens_available": sum(len(d["old"]) for d in selected),
                "mean_accepted_draft_per_round": mean(r["decision"]["accepted_draft"] for r in rs) if rs else None})
    bridge_rows, run_rows = [], []
    for d in eligible:
        for rank, r in enumerate(d["runs"]):
            run_rows.append({"event_id": d["id"], "prompt_id": d["prompt_id"], "dataset": d["dataset"],
                             "rank": rank, **r, "text": decode(d["old"][r["source_start"]:r["source_end"]])})
        for i, b in enumerate(d["bridges"]):
            bridge_rows.append({"event_id": d["id"], "prompt_id": d["prompt_id"], "dataset": d["dataset"],
                                "bridge_index": i, **b, "source_text": decode(b["source_ids"]),
                                "target_text": decode(b["target_ids"])})
    atomic_json(output/"summary.json", {"summary": summaries,
        "completed_prompts": len(prompts), "expected_prompts": c["prompts_per_dataset"]*len(c["datasets"]),
        "saved_rounds": len(rounds), "audited_rounds": len(audited_rounds),
        "unlabeled_rounds": len(rounds)-len(audited_rounds),
        "accepted_draft_histogram": dict(Counter(r["decision"]["accepted_draft"] for r in audited_rounds)),
        "emitted_output_tokens": sum(len(p["tokens"]) for p in prompts),
        "total_unmask_forwards_saved_rounds": sum(r["draft"]["stats"]["unmask_forwards"] for r in rounds),
        "terminal_corrections": sum(d["terminal_correction"] for d in diagnostics),
        "empty_old_suffixes": sum(d["empty_old"] for d in diagnostics),
        "reference_censored_events": sum(d["right_reference_censored"] for d in eligible),
        "best_run_right_censored_hits": sum(d["longest_right_censored"] for d in eligible),
        "bridge_target_length_histogram": dict(Counter(b["target_length"] for b in bridge_rows)),
        "bridge_offset_changes": sum(b["delta_before"] != b["delta_after"] for b in bridge_rows),
        "bridge_count": len(bridge_rows), "limits": [
            "Post-hoc exact matches are oracle candidates; no retrieved spans were committed.",
            "Rounds of one prompt are correlated; event rates are descriptive, not independent trials.",
            "Native physical32/active8, 60 one-MASK forwards even when a logical proposal ends early at EOS.",
            "Fixed draft60 may exceed final remaining output budget; accepted count includes un-emitted verified draft tokens.",
            "Chain optimizes only enumerated maximal runs, not every partial alignment.",
            "Operational wall includes transfers/setup/audits; no end-to-end speedup or task-accuracy claim."]})
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    hits = [d for d in eligible if d["runs"]]
    summary.update(best_shift_histogram=dict(Counter(d["runs"][0]["delta"] for d in hits)),
                   best_length_histogram=dict(Counter(d["runs"][0]["length"] for d in hits)),
                   chain_block_length_histogram=dict(Counter(r["length"] for d in eligible for r in d["chain"])),
                   eligible_C0=sum(d["C0"] for d in eligible),
                   C0_with_shifted_run=sum(d["C0"] for d in hits),
                   longest_ge8_ge16_ge24={str(n): sum(d["runs"][0]["length"] >= n for d in hits) for n in (8, 16, 24)})
    atomic_json(output/"summary.json", summary)
    write_csv(output/"summary.csv", list(summaries[0]), summaries)
    write_csv(output/"runs.csv", ["event_id", "prompt_id", "dataset", "rank", "source_start", "source_end",
              "destination_start", "destination_end", "length", "delta", "text"], run_rows)
    write_csv(output/"bridges.csv", ["event_id", "prompt_id", "dataset", "bridge_index", "source_length",
              "target_length", "delta_before", "delta_after", "source_start", "destination_start", "source_text", "target_text"], bridge_rows)
    write_csv(output/"prompts.csv", ["id", "dataset", "rounds", "terminated", "canonical_audit_passed"], prompts)
    lines = ["# Native draft60 → AR: online generation and offline shift diagnostics", "",
             f"Audited prompts: {len(prompts)}/{c['prompts_per_dataset']*len(c['datasets'])}. "
             f"Audited/saved rounds: {len(audited_rounds)}/{len(rounds)}.", "",
             "Draft length60; output cap256 by default; fixed FP16 target split18/10 layers on GPU0/GPU1; dLLM on GPU1.", "",
             "| Dataset | Radius | Eligible rejections | Any run≥4 | ≥2 blocks | ≥3 blocks | Mean longest when found |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        avg = "NA" if r["mean_longest_given_hit"] is None else f"{r['mean_longest_given_hit']:.2f}"
        lines.append(f"|{r['dataset']}|{r['radius']}|{r['eligible_rejections']}|{r['events_with_run']}|{r['events_with_2plus_blocks']}|{r['events_with_3plus_blocks']}|{avg}|")
    lines += ["", "All matches use actual independent target-only canonical continuation after generation. "
              "They do not enter online decisions and are not an implemented retrieval policy.", "",
              "Nonzero shifts; source/destination positions are relative to the post-correction suffix. "
              "Raw runs and monotone chains retain exact token IDs. EOS/post-EOS content is excluded from reuse labels.", "",
              "Read summary.json for censoring, terminal/empty suffix counts, acceptance histogram and interpretation limits. "
              "Read runs.csv, bridges.csv, diagnostics.jsonl and cases.html for content and geometry.", "",
              "No latency comparison: notebook/model setup, independent canonical audit and peer transfers remain in operational wall."]
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    cases = sorted(eligible, key=lambda d: (-len(d["chain"]), -sum(r["length"] for r in d["chain"]), d["id"]))[:40]
    page = ["<!doctype html><meta charset='utf-8'><title>Draft60 shifted spans</title>",
            "<style>body{font:15px system-ui;max-width:1100px;margin:30px auto}pre{white-space:pre-wrap;background:#f4f4f4;padding:12px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:5px}</style>",
            "<h1>Offline shifted-span candidates</h1><p>Oracle labels after independent canonical audit; not online retrieval success.</p>"]
    for d in cases:
        page += ["<h2>"+html.escape(d["id"])+"</h2>", "<p>Old suffix</p><pre>"+html.escape(decode(d["old"]))+"</pre>",
                 "<p>Target continuation</p><pre>"+html.escape(decode(d["reference"]))+"</pre>",
                 "<table><tr><th>Source</th><th>Destination</th><th>Shift</th><th>Length</th><th>Text</th></tr>"]
        for r in d["chain"]:
            page.append(f"<tr><td>{r['source_start']}:{r['source_end']}</td><td>{r['destination_start']}:{r['destination_end']}</td><td>{r['delta']}</td><td>{r['length']}</td><td>"+html.escape(decode(d["old"][r["source_start"]:r["source_end"]]))+"</td></tr>")
        page.append("</table>")
        if d["prefix_bridge"]:
            b = d["prefix_bridge"]
            page.append("<p>Bridge before first block: old → target</p><pre>"+
                        html.escape(repr(decode(b["source_ids"]))+" → "+repr(decode(b["target_ids"])))+"</pre>")
        for b in d["bridges"]:
            page.append(f"<p>Bridge: offset {b['delta_before']} → {b['delta_after']}; old → target</p><pre>"+
                        html.escape(repr(decode(b["source_ids"]))+" → "+repr(decode(b["target_ids"])))+"</pre>")
    (output/"cases.html").write_text("\n".join(page), encoding="utf-8")
    if tokenizer is not None:
        atomic_json(decoded_path, decoded)
