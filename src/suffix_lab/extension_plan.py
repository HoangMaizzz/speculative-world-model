"""Read-only pilot input, outcome-blind event selection and frozen span anchors."""
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import tempfile
from functools import lru_cache
from .metrics import suffix_metrics, prefix_length
from .migration import locate_unpacked
from .selection import Observation
from .store import digest

PILOT_REVISION = "fa330e5836db310172ced163162d511f6b2036b5"
PILOT_SOURCE = "e2177c1da830f29fbab58a659208494a4f05305170444bebb7eacee536675a83"


def read_pilot(root):
    print("[checkpoint] Locating unpacked pilot:", root, flush=True)
    folder = locate_unpacked(root)
    manifest = json.loads((folder/"run-manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("protocol") != "suffix-top1-v4-audited-canonical-replay"
            or manifest.get("source_digest") != PILOT_SOURCE
            or manifest.get("environment", {}).get("commit") != PILOT_REVISION
            or manifest.get("target_precision") != "fp16"):
        raise ValueError("Mini-test requires the completed FP16 recovery pilot fa330e58")
    expected = {"proposal_length": 32, "physical_block_size": 32, "subblock_size": 8,
                "mask_id": 151665, "target_precision": "fp16", "target_attention": "sdpa",
                "target_gpu0_layers": 18, "allow_nf4_fallback": False,
                "allow_fp16_reduced_precision_reduction": False}
    if any(manifest["config"].get(k) != value for k, value in expected.items()):
        raise ValueError("Pilot numerical/scheduler configuration differs")
    path = folder/"checkpoint.sqlite"
    size = path.stat().st_size
    print(f"[checkpoint] Staging {size/2**20:.1f} MiB to local temporary storage; input stays read-only", flush=True)
    fingerprint = hashlib.sha256()
    # Mounted Kaggle input can make SQLite's repeated page reads expensive.
    # Copy/hash in ONE sequential pass, then audit/query only the local copy.
    with tempfile.TemporaryDirectory(prefix="suffix-mini-input-") as temporary:
        local = Path(temporary)/"checkpoint.sqlite"
        copied, next_log = 0, 16*2**20
        with path.open("rb") as source, local.open("wb") as target:
            while block := source.read(4*2**20):
                target.write(block)
                fingerprint.update(block)
                copied += len(block)
                if copied >= next_log:
                    print(f"[checkpoint] Copied {copied/2**20:.0f}/{size/2**20:.0f} MiB", flush=True)
                    next_log += 16*2**20
        if copied != size:
            raise ValueError("Checkpoint size changed while copying")
        print("[checkpoint] Local SQLite integrity check", flush=True)
        conn = sqlite3.connect(local.resolve().as_uri()+"?mode=ro&immutable=1", uri=True)
        try:
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Pilot SQLite integrity failed")
            print("[checkpoint] Reading references, Test1, cohort metadata and dLLM Fresh-L baselines", flush=True)
            records = {}
            query = ("SELECT kind,key,payload FROM records WHERE "
                     "kind IN ('meta','reference','test1','pair_complete','failure') "
                     "OR (kind='test2' AND key LIKE '%|dllm_top1|fresh_L')")
            for i, (kind, key, value) in enumerate(conn.execute(query), 1):
                records[(kind, key)] = json.loads(value)
                if i % 500 == 0:
                    print(f"[checkpoint] Parsed {i} required records", flush=True)
        finally:
            conn.close()
    print(f"[checkpoint] Ready: {len(records)} records; unused pilot arms skipped", flush=True)
    if records.get(("meta", "identity")) != manifest:
        raise ValueError("Pilot manifest/database identity mismatch")
    if not all(records.get(("meta", key)) for key in
               ("references_complete", "test1_complete", "test2_complete")):
        raise ValueError("Use a completed pilot checkpoint")
    if any(kind == "failure" for kind, key in records):
        raise ValueError("Pilot contains unresolved failure records")
    design = records[("meta", "test2_design")]
    return {"manifest": manifest, "records": records, "design": design,
            "checkpoint_sha256": fingerprint.hexdigest()}


def validate_event(row, ref):
    offset = row["offset"]
    proposal, feedback = row["draft"]["tokens"], row["feedback"]
    gold = ref["tokens"][offset:offset+32]
    if (len(proposal) != 32 or len(gold) != 32 or row["reference"] != gold
            or row["metrics"] != suffix_metrics(proposal, gold, feedback["top1"])):
        raise ValueError("Corrupt pilot event: "+row["id"])
    k = prefix_length(proposal, gold)
    if (prefix_length(proposal, feedback["top1"]) != k
            or (k < 32 and feedback["top1"][k] != gold[k])):
        raise ValueError("Pilot correction/reference disagreement: "+row["id"])
    for feature in (row["draft"]["confidence"], feedback["probability"]):
        if len(feature) != 32 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in feature):
            raise ValueError("Invalid pilot confidence")


def choose_events(candidates, config):
    """Seeded feasibility selection on prompt groups; not a population sample."""
    ordered = sorted(candidates, key=lambda x: x["event_id"])
    random.Random(config["seed"]).shuffle(ordered)
    groups = {}
    for row in ordered:
        groups.setdefault(row["prompt_id"], []).append(row)
    groups = list(groups.values())

    @lru_cache(None)
    def solve(i, count, gsm, enriched):
        if count == 0:
            return () if gsm == enriched == 0 else None
        if (i == len(groups) or len(groups)-i < count
                or not 0 <= gsm <= count or not 0 <= enriched <= count):
            return None
        for row in groups[i]:
            tail = solve(i+1, count-1, gsm-int(row["dataset"] == "gsm8k"),
                         enriched-int(row["cohort"] == "enriched"))
            if tail is not None:
                return (row["event_id"],)+tail
        return solve(i+1, count, gsm, enriched)

    selected = solve(0, config["events"], config["gsm8k_events"], config["enriched_events"])
    if selected is None:
        raise ValueError("Insufficient distinct prompts for frozen mini-test quotas; no silent relaxation")
    lookup = {x["event_id"]: x for x in candidates}
    return [lookup[key] for key in selected]


def spans(mask):
    start = None
    for i, yes in enumerate(list(mask)+[False]):
        if yes and start is None:
            start = i
        elif not yes and start is not None:
            yield start, i
            start = None


def span_positions(mask, scores, config, *, oracle=False):
    budget = math.floor(len(mask)*config["keep_fraction"])
    candidates = []
    for start, stop in spans(mask):
        for left in range(start, stop):
            for size in range(config["min_span"], min(budget, stop-left)+1):
                positions = tuple(range(left, left+size))
                score = sum(scores[i] for i in positions)/size
                rank = (-size, left) if oracle else (-score, -size, left)
                candidates.append((rank, positions))
    selected, chosen_spans = set(), []
    for _, positions in sorted(candidates):
        if len(chosen_spans) == config["max_spans"]:
            break
        if len(selected)+len(positions) > budget:
            continue
        # Adjacent fragments form one span; prefer selecting the actual span.
        if any(i in selected or i-1 in selected or i+1 in selected for i in positions):
            continue
        selected.update(positions)
        chosen_spans.append(list(positions))
    return sorted(selected)


def predicted_positions(obs, config):
    """No corrected reference argument or access. Old feedback is heuristic only."""
    n = len(obs.old_tokens)
    if any(len(x) != n for x in (obs.draft_confidence, obs.verifier_probability, obs.verifier_top1)):
        raise ValueError("Selector feature lengths differ")
    mask = [t == v and pv >= config["predicted_verifier_min"]
            and pd >= config["predicted_drafter_min"] for t, v, pv, pd in
            zip(obs.old_tokens, obs.verifier_top1, obs.verifier_probability, obs.draft_confidence)]
    scores = [0.5*(math.log(max(pv, 1e-30))+math.log(max(pd, 1e-30)))
              for pv, pd in zip(obs.verifier_probability, obs.draft_confidence)]
    return span_positions(mask, scores, config)


def event_plan(row, ref, cohort, config):
    r, offset = row["metrics"]["first_rejection"], row["offset"]
    old = row["draft"]["tokens"][r:]
    start = offset+r
    gold = ref["tokens"][start:start+len(old)+config["extension"]]
    obs = Observation(tuple(old), tuple(row["draft"]["confidence"][r:]),
                      tuple(row["feedback"]["probability"][r:]), tuple(row["feedback"]["top1"][r:]))
    predicted = predicted_positions(obs, config)
    oracle = span_positions([a == b for a, b in zip(old, gold)], [0.]*len(old), config, oracle=True)
    prefix = ref["prompt_ids"]+ref["tokens"][:start]
    return {"event_id": row["id"], "prompt_id": row["prompt_id"], "dataset": row["dataset"],
            "source": row["source"], "cohort": cohort, "first_rejection_1based": r,
            "offset": offset, "reference_start": start, "prefix": prefix, "old": old,
            "reference": gold, "L": len(old), "H": len(gold),
            "anchors": {"fresh": [], "oracle": [[i, old[i]] for i in oracle],
                        "predicted": [[i, old[i]] for i in predicted]},
            "physical_phase": len(prefix)%32, "old_endpoint_phase8": (len(prefix)+len(old))%8,
            "single_physical_block": len(prefix)%32+len(gold) <= 32}


def freeze_inputs(pilot, config):
    if config["source"] != "dllm_top1" or config["redrafter"] != "dllm_top1":
        raise ValueError("This mini-test is dLLM -> AR only")
    records, candidates = pilot["records"], []
    for event, member in sorted(pilot["design"]["members"].items()):
        row = records[("test1", event)]
        if row["source"] != "dllm_top1":
            continue
        ref = records[("reference", row["prompt_id"])]
        validate_event(row, ref)
        r = row["metrics"]["first_rejection"]
        if not config["rejection_min"] <= r <= config["rejection_max"]:
            continue
        end = row["offset"]+32+config["extension"]
        if end > len(ref["tokens"]):
            continue
        eos = pilot["manifest"]["tokenizer"]["eos_token_id"]
        if eos in ref["tokens"][:end-1]:
            continue
        if member["cohort"] not in ("enriched", "representative"):
            raise ValueError("Unknown pilot cohort")
        candidates.append(event_plan(row, ref, member["cohort"], config))
    # Selection reads event metadata only; neither the anchor masks nor new/old
    # repair outcomes are passed to the quota solver.
    selected_meta = choose_events([{k: x[k] for k in
        ("event_id", "prompt_id", "dataset", "cohort")} for x in candidates], config)
    lookup = {x["event_id"]: x for x in candidates}
    selected = [lookup[x["event_id"]] for x in selected_meta]
    refs, baselines = {}, {}
    for x in selected:
        ref = records[("reference", x["prompt_id"])]
        refs[ref["id"]] = ref
        key = x["event_id"]+"|dllm_top1|fresh_L"
        baseline = records[("test2", key)]
        if (baseline["tokens"] is None or baseline["anchors"] or baseline["horizon"] != x["L"]
                or baseline["verified_accepted_prefix"] != prefix_length(baseline["tokens"], x["reference"])
                or not records.get(("pair_complete", x["event_id"]+"|dllm_top1"))):
            raise ValueError("Pilot Fresh-L audit failed")
        baselines[x["event_id"]] = baseline
    return {"schema": "dllm-ar-extension-input-v1", "pilot_manifest": pilot["manifest"],
            "pilot_checkpoint_sha256": pilot["checkpoint_sha256"], "eligible_events": len(candidates),
            "selection_policy": "seeded distinct-prompt quota feasibility; no repair outcomes used; diagnostic only",
            "events": selected, "references": refs, "pilot_fresh_L": baselines,
            "plan_digest": digest(selected)}


def arms(event):
    result = []
    for strategy in ("fresh", "oracle", "predicted"):
        for schedule, horizon in (("L", event["L"]), ("H_joint", event["H"]), ("H_sequential", event["H"])):
            result.append({"method": strategy+"_"+schedule, "strategy": strategy,
                           "schedule": schedule, "horizon": horizon, "anchors": event["anchors"][strategy]})
    return result
