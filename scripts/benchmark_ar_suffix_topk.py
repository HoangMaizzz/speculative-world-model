"""Replay saved greedy AR-AR proposals and inspect verifier logits after rejection.

The post-rejection logits are teacher-forced on the OLD rejected proposal.
They are not logits under the corrected continuation.
"""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

K_VALUES = (1, 2, 4, 8, 16, 32)


def read_trace(path):
    if path.is_dir():
        read = lambda name: (path / name).read_bytes()
    else:
        with ZipFile(path) as archive:
            data = {name: archive.read(name) for name in ("summary.json", "cases.jsonl", "prompts.json")}
        read = data.__getitem__
    summary = json.loads(read("summary.json"))
    cases = [json.loads(line) for line in read("cases.jsonl").splitlines() if line.strip()]
    prompts = {p["id"]: p for p in json.loads(read("prompts.json"))}
    if len(prompts) != len(cases) or {c["id"] for c in cases} != set(prompts):
        raise ValueError("Trace case/prompt IDs do not match")
    return summary, cases, prompts


def source_anywhere_positions(suffix, future, minimum=4):
    """Source positions participating in an exact run anywhere in future."""
    positive = set()
    for i in range(len(suffix)):
        for j in range(len(future)):
            if suffix[i] != future[j]:
                continue
            length = 0
            while i + length < len(suffix) and j + length < len(future) and suffix[i + length] == future[j + length]:
                length += 1
            if length >= minimum:
                positive.update(range(i, i + length))
    return positive


def bucket(distance):
    if distance == 0:
        return "reject"
    if distance == 1:
        return "after_1"
    if distance <= 3:
        return "after_2_3"
    if distance <= 7:
        return "after_4_7"
    if distance <= 15:
        return "after_8_15"
    return "after_16_plus"


def tally(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], "all")].append(row)
        groups[("all", "all")].append(row)
        groups[("all", row["distance_bucket"])].append(row)
        if row["distance_after_reject"]:
            groups[("all", "reusable" if row["source_reusable_anywhere4"] else "other")].append(row)
    result = []
    for (dataset, group), items in sorted(groups.items()):
        record = {"dataset": dataset, "group": group, "positions": len(items),
                  "reference_available": sum(x["reference_token"] is not None for x in items),
                  "old_equals_reference": sum(x["old_equals_reference"] is True for x in items)}
        for kind in ("reference", "old"):
            for k in K_VALUES:
                selected = [x for x in items if x[f"{kind}_rank_le32"] is not None]
                record[f"{kind}_top{k}_count"] = sum(x[f"{kind}_rank_le32"] <= k for x in selected)
                record[f"{kind}_top{k}_denominator"] = sum(x[f"{kind}_token"] is not None for x in items)
        result.append(record)
    return result


def replay_case(model, prompt, case, max_k, eos):
    import torch
    from suffix_lab.long60_plan import decide

    run = case["arms"]["fresh"]
    reference = run["token_ids"]
    rounds = run["rounds"]
    prefix = prompt["prompt_tokens"]
    device = model.device
    cache = None
    prefill = model(torch.tensor([prefix], device=device), past_key_values=None,
                    use_cache=True, logits_to_keep=1)
    cache = prefill.past_key_values
    next_token = int(prefill.logits[0, -1].argmax())
    pending = []
    records = []
    for round_index, saved in enumerate(rounds):
        proposal = saved["proposal"]
        offset = saved["offset"]
        if offset != sum(len(r["decision"]["committed_tokens"]) for r in rounds[:round_index]):
            raise AssertionError("Saved offsets are not contiguous")
        base = cache.get_seq_length()
        input_ids = pending + proposal
        output = model(torch.tensor([input_ids], device=device), past_key_values=cache,
                       use_cache=True, logits_to_keep=len(input_ids))
        cache = output.past_key_values
        logits = output.logits[0]
        if pending:
            rows = logits[len(pending) - 1:len(pending) - 1 + len(proposal)]
            bonus = logits[-1]
        else:
            rows = torch.cat((prefill.logits[0, -1:], logits[:-1]), dim=0) if round_index == 0 else torch.cat((last_next[None], logits[:-1]), dim=0)
            bonus = logits[-1]
        if len(rows) != len(proposal):
            raise AssertionError("Verifier row alignment failed")
        top1 = rows.argmax(-1).tolist()
        top1_bonus = [*top1, int(bonus.argmax())]
        decision = decide(proposal, top1_bonus, eos, len(reference) - offset)
        for key in ("accepted_draft", "first_rejection", "correction", "bonus", "committed_tokens"):
            if decision[key] != saved["decision"][key]:
                raise AssertionError(f"Replay mismatch {case['id']} round={round_index} field={key}: {decision[key]} != {saved['decision'][key]}")
        reject = decision["first_rejection"]
        if reject is not None:
            suffix = proposal[reject + 1:]
            future = reference[offset:offset + 64]
            reusable = source_anywhere_positions(suffix, future)
            positions = list(range(reject, len(proposal)))
            selected = rows[positions].float()
            winners = selected.argmax(-1, keepdim=True)
            alternatives = selected.clone().scatter_(1, winners, -torch.inf)
            other_ids = alternatives.topk(max_k - 1, dim=-1).indices
            ranked_ids = torch.cat((winners, other_ids), dim=-1)
            top_ids = ranked_ids.tolist()
            top_values = selected.gather(1, ranked_ids)
            normalizer = selected.logsumexp(-1)
            for local, position in enumerate(positions):
                target_position = offset + position
                ref = reference[target_position] if target_position < len(reference) else None
                old = proposal[position]
                ids = top_ids[local]
                ranks = {token: (ids.index(token) + 1 if token in ids else None)
                         for token in (old, ref) if token is not None}
                records.append(dict(case_id=case["id"], dataset=case["dataset"], round=round_index,
                                    offset=offset, proposal_position=position, first_rejection=reject,
                                    distance_after_reject=position - reject,
                                    distance_bucket=bucket(position - reject),
                                    old_token=old, reference_token=ref,
                                    old_equals_reference=(old == ref if ref is not None else None),
                                    source_reusable_anywhere4=(position > reject and position - reject - 1 in reusable),
                                    old_rank_le32=ranks[old],
                                    reference_rank_le32=ranks.get(ref) if ref is not None else None,
                                    old_probability=float((selected[local, old] - normalizer[local]).exp()),
                                    reference_probability=(float((selected[local, ref] - normalizer[local]).exp()) if ref is not None else None),
                                    topk_ids=ids, topk_probabilities=(top_values[local] - normalizer[local]).exp().tolist()))
        accepted = min(decision["accepted_draft"], len(decision["committed_tokens"]))
        cache.crop(base + len(pending) + accepted)
        pending = decision["committed_tokens"][accepted:]
        last_next = bonus
        if decision["terminated"]:
            break
    if sum(len(r["decision"]["committed_tokens"]) for r in rounds) != len(reference):
        raise AssertionError("Saved output length mismatch")
    return records


def run(args):
    import torch
    from huggingface_hub import snapshot_download
    from suffix_lab.assets import TARGET
    from suffix_lab.runtime import configure_numerics, load_target, sync_model

    trace = args.trace.resolve()
    summary, cases, prompts = read_trace(trace)
    manifest = summary["manifest"]
    if manifest["precision"] != "fp16" or manifest["temperature"] != 0:
        raise ValueError("Expected the saved greedy FP16 AR-AR trace")
    expected = manifest["models"][TARGET]
    config_path = Path(__file__).resolve().parents[1] / "configs/long64.json"
    config = json.loads(config_path.read_text())
    configure_numerics(config)
    path = snapshot_download(TARGET, revision=expected,
                             allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
    model, _, _ = load_target(path, config)
    eos = set(manifest["eos"])
    if args.limit:
        cases = cases[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    completed_path = args.output / "completed.jsonl"
    done = {}
    if completed_path.exists():
        for line in completed_path.read_text().splitlines():
            item = json.loads(line)
            done[item["case_id"]] = item
    started = perf_counter()
    with torch.inference_mode(), completed_path.open("a", encoding="utf-8") as out:
        for index, case in enumerate(cases):
            if case["id"] in done:
                continue
            records = replay_case(model, prompts[case["id"]], case, args.max_k, eos)
            sync_model(model)
            item = {"case_id": case["id"], "dataset": case["dataset"], "rows": records}
            out.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            out.flush()
            done[case["id"]] = item
            print(f"{index + 1}/{len(cases)} {case['id']} rows={len(records)} elapsed={perf_counter()-started:.1f}s", flush=True)
    rows = [row for case in cases for row in done[case["id"]]["rows"]]
    result = {"protocol": "ar-suffix-old-context-topk-v1", "trace_sha256": sha256(trace.read_bytes()).hexdigest() if trace.is_file() else None,
              "model": TARGET, "model_revision": expected, "cases": len(cases), "max_k": args.max_k,
              "definition": "Teacher-forced OLD proposal. Reference token at same output index; reusable source is any exact run >=4 within next 64 reference tokens.",
              "groups": tally(rows)}
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with ZipFile(args.output / "ar_suffix_topk_results.zip", "w", ZIP_DEFLATED) as archive:
        archive.write(args.output / "summary.json", "summary.json")
        archive.write(completed_path, "completed.jsonl")
    print(json.dumps({"cases": len(cases), "positions": len(rows), "seconds": round(perf_counter()-started, 1),
                      "zip": str(args.output / "ar_suffix_topk_results.zip")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=32, choices=(32,))
    parser.add_argument("--limit", type=int, default=0)
    run(parser.parse_args())

