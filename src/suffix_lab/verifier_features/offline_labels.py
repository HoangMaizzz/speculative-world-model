"""Oracle labels and eligibility auditing, isolated from feature inference."""
from suffix_lab.long60_plan import matching_runs, until_eos


def label(event, reference, eos, horizon=63):
    start = event["offset"]+event["accepted"]+1
    raw = reference[start:start+horizon]
    target = until_eos(raw, eos, False)
    runs = matching_runs(event["source"], target, horizon, 4, include_zero=True)
    for i, r in enumerate(runs):
        r.update(run_id=i, source_endpoint=r["source_end"] == len(event["source"]),
                 target_endpoint=r["destination_end"] == len(target),
                 target_endpoint_kind="eos" if any(t in eos for t in raw) else "horizon")
    pairs = {str(k): [{"i": r["source_start"]+t, "j": r["destination_start"]+t}
                      for r in runs for t in range(r["length"]-k+1)] for k in (4, 8)}
    edges = [{"left": l["run_id"], "right": r["run_id"],
              "source_start": l["source_end"], "source_end": r["source_start"],
              "destination_start": l["destination_end"], "destination_end": r["destination_start"],
              "source_length": r["source_start"]-l["source_end"],
              "target_length": r["destination_start"]-l["destination_end"],
              "source_ids": event["source"][l["source_end"]:r["source_start"]],
              "target_ids": target[l["destination_end"]:r["destination_start"]]}
             for l in runs for r in runs
             if l["source_end"] <= r["source_start"] and l["destination_end"] <= r["destination_start"]]
    return {"id": event["id"], "prompt_id": event["prompt_id"], "dataset": event["dataset"],
            "reference_start": start, "reference_ids": target, "runs": runs,
            "matches4": pairs["4"], "matches8": pairs["8"], "bridge_edges": edges,
            "initial_bridge_candidates": [{"run_id": r["run_id"],
                "source_ids": event["source"][:r["source_start"]],
                "target_ids": target[:r["destination_start"]],
                "source_length": r["source_start"], "target_length": r["destination_start"]} for r in runs],
            "expected_corrected_next": raw[0] if raw else None,
            "reference_exhausted": not target,
            "right_reference_censored": len(raw) < horizon and not any(t in eos for t in raw),
            "source_endpoint_kind": "eos" if event.get("proposal", [None])[-1] in eos else "draft_capacity",
            "target_eos_visible": any(t in eos for t in raw),
            "C0": not any(a == b for a, b in zip(event["source"], target))}


def prepare_labels(cp, events, eos):
    refs = {r["id"]: r for r in cp.lines("references.jsonl")}
    for p in cp.prompts:
        r = refs[p["id"]]
        if p["prompt_ids"] != r["prompt_ids"] or p["tokens"] != r["tokens"][:len(p["tokens"])]:
            raise ValueError("Independent reference/output audit failed")
    for row in cp.rounds:
        gold = refs[row["prompt_id"]]["tokens"]
        n, offset = len(row["proposal"]), row["offset"]
        future = gold[offset:offset+n]
        a = next((i for i, (x, y) in enumerate(zip(row["proposal"], future)) if x != y), min(n, len(future)))
        if a != row["decision"]["accepted_draft"]:
            raise ValueError("Per-round acceptance differs from independent reference")
        if a < n and row["decision"]["correction"] != gold[offset+a]:
            raise ValueError("Verified correction differs from independent reference")
    labels = {e["id"]: label(e, refs[e["prompt_id"]]["tokens"], eos) for e in events}
    return labels


def write_labels(path, labels):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(list(labels.values())), str(path), compression="zstd")
