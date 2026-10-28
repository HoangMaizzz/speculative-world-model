"""Reports describe extraction coverage and audits, never detector performance."""
import csv
import json
from pathlib import Path
import statistics
from suffix_lab.store import atomic_json, canonical


def make_report(output, records, total, status, memory=None):
    output = Path(output)
    rows = sorted(records.values(), key=lambda r: r["event"]["id"])
    checks = [r["check"] for r in rows]
    mismatch = sum(not r["old_top1_matches"] for r in checks)
    corrected_mismatch = sum(r.get("corrected_next_matches_reference") is False for r in checks)
    samples = [r["timing"]["event_wall_ms_including_transfers_io_excluded"]/1000 for r in rows]
    summary = {"status": status, "completed_events": len(rows), "expected_events": total,
               "trusted_old_replays": len(rows)-mismatch, "quarantined_old_replays": mismatch,
               "corrected_cold_vs_canonical_disagreements": corrected_mismatch,
               "mean_event_seconds": statistics.mean(samples) if samples else None,
               "estimated_remaining_inference_seconds": statistics.mean(samples)*(total-len(rows)) if samples else None,
               "gpu_memory": memory, "training_performed": False, "speedup_measured": False,
               "timing_note": "CUDA model stage sum excludes peer dispatch; operational wall includes capture/transfers. No latency study."}
    atomic_json(output/"summary.json", summary)
    for name, values in (("events.jsonl", [{**r["event"], "tensors": {
            "shard": r["tensor_shard"], "member": r["tensor_file"], "sha256": r["tensor_sha256"]}} for r in rows]),
                         ("replay_checks.jsonl", checks)):
        temporary = output/(name+".tmp")
        temporary.write_text("".join(canonical(r)+"\n" for r in values), encoding="utf-8")
        temporary.replace(output/name)
    with (output/"timing.csv").open("w", encoding="utf-8", newline="") as f:
        names = ["id", "phase", "forward_count", "input_tokens", "compute_ms_excluding_peer_transfers",
                 "wall_ms_including_transfers_capture", "feature_reduce_wall_ms"]
        writer = csv.DictWriter(f, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            for phase in ("old", "corrected"):
                writer.writerow({"id": r["event"]["id"], "phase": phase, **r["timing"][phase]})
            for i, t in enumerate(r["event"]["corrected"]["micro_probe"]["timing"]):
                writer.writerow({"id": r["event"]["id"], "phase": f"probe_{i+2}", **t})
            for audit in ("hook_audit", "causal_tail_audit"):
                if r["check"].get(audit):
                    writer.writerow({"id": r["event"]["id"], "phase": audit, **r["check"][audit]["timing"]})
    text = ["# Verifier feature collection", "", f"Status: {status}",
            f"Events: {len(rows)}/{total}; quarantined old replays: {mismatch}.",
            f"Corrected cold-prefill / saved canonical next-token differences: {corrected_mismatch}.", "",
            "Input drafts/references reused. No drafter inference, redrafting, training or speculative trees.",
            "Online events/tensors and offline_labels.parquet are separate. Reference audits appear only in replay_checks.",
            "Old hidden union slots i / i+1 are pre-predict / post-read for source token i. Layer IDs are zero-based decoder outputs before final norm.",
            "M_off[i,j] uses the old proposal path, j=0..L with one extra boundary row. It is not corrected-context probability.",
            "q0 is immediate-next-token suitability only; repeated source token IDs have equal q0 values.",
            "TopK log probabilities and entropy use the full vocabulary. Draft rank = 1 + count(logit > draft logit).",
            "Cold corrected replay may differ numerically from cached canonical greedy; discrepancies are retained without substituting reference values.",
            "Every quarantined event retains raw features/checks; use old_top1_matches to select the matched replay subset.",
            "Prompt-level split IDs are frozen; smoke/pilot selection is enriched and is not a population estimate.",
            "CUDA compute timing excludes dispatch copies; wall/capture/packing costs remain separate. No inference of detector quality or speedup.", ""]
    if samples:
        text += [f"Mean measured extraction wall/event (excluding shard packing): {statistics.mean(samples):.3f}s.",
                 f"Projected remaining inference wall: {summary['estimated_remaining_inference_seconds']/60:.1f} min; setup/packing extra.", ""]
    for r in rows[:3]:
        e = r["event"]
        text += [f"Example {e['id']}: accepted={e['accepted']}, source={len(e['source_ids'])}, next={e['corrected']['top1'][0]}."]
    (output/"report.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return summary
