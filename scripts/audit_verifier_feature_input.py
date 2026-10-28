"""CPU-only audit and label reconstruction for the pinned B64 collection input."""
import argparse
from collections import Counter
import json
from pathlib import Path
from suffix_lab.assets import ROOT
from suffix_lab.store import atomic_json
from suffix_lab.verifier_features.checkpoint_loader import discover
from suffix_lab.verifier_features.prefix_reconstruction import reconstruct
from suffix_lab.verifier_features.offline_labels import prepare_labels, write_labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cp = discover(args.input_root)
    config = json.loads((ROOT/"configs/verifier_features.json").read_text())
    if cp.identity != config["expected_input_identity"]:
        raise ValueError("Input differs from pinned completed B64 run")
    eos = set(cp.runtime["effective_eos_ids"])
    events, excluded = reconstruct(cp.prompts, cp.rounds, eos)
    labels = prepare_labels(cp, events, eos)
    eligible = [e for e in events if not labels[e["id"]]["reference_exhausted"]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_labels(output/"offline_labels.parquet", {e["id"]: labels[e["id"]] for e in eligible})
    report = {"passed": len(cp.rounds) == 4050 and len(eligible) == 3847,
              "input_identity": cp.identity, "rounds_checked": len(cp.rounds), "prompts": len(cp.prompts),
              "eligible": len(eligible), "datasets": dict(Counter(e["dataset"] for e in eligible)),
              "suffix_tokens": sum(len(e["source"]) for e in eligible),
              "all_maximal_runs_including_zero": sum(len(labels[e["id"]]["runs"]) for e in eligible),
              "all_maximal_nonzero_runs": sum(r["delta"] != 0 for e in eligible for r in labels[e["id"]]["runs"]),
              "bridge_edges_between_compatible_maximal_runs": sum(len(labels[e["id"]]["bridge_edges"]) for e in eligible),
              "verified_correction_beyond_output_budget": sum(not e["correction_emitted_within_output_budget"] for e in eligible),
              "gpu_inference_performed": False}
    atomic_json(output/"input-audit.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise AssertionError("Input count audit failed")


if __name__ == "__main__":
    main()
