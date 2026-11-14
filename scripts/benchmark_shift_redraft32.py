"""Paired 100+100 one-step redrafting benchmark on the saved draft32 run."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import runpy
from statistics import mean, median
import subprocess
import sys
from time import perf_counter
from types import MethodType
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile


ARMS = ("fresh", "oracle_shift", "model_shift_oracle_remask")
CASE_HASH = "88edbcffcac6fafae5efa58c229ce9a1fedbaf5ea43ba2e18f9c921438e9c2dd"
SUMMARY_HASH = "3a6a19411fc2be29ee0226732ac5eb5281847b89a0e9e9ebcf275d736d2ac540"
SEED = 42
BASE_REVISION = "de9f4ba9cf19da772a3c842c05234d6df8edd95d"
REPOSITORY = "https://github.com/HoangMaizzz/speculative-world-model.git"


def input_members(path):
    names = ("cases.jsonl", "summary.json")
    if path.is_dir():
        return [(path / name).read_bytes() for name in names]
    with ZipFile(path) as archive:
        return [archive.read(name) for name in names]


def find_input(root):
    paths = [root]
    if root.is_dir():
        paths += sorted(root.rglob("dllm_ar32_results*.zip"))
        paths += sorted({p.parent for p in root.rglob("cases.jsonl")})
    for path in paths:
        try:
            cases, summary = input_members(path)
        except (OSError, KeyError, BadZipFile):
            continue
        if sha256(cases).hexdigest() == CASE_HASH and sha256(summary).hexdigest() == SUMMARY_HASH:
            return path, [json.loads(line) for line in cases.splitlines()], json.loads(summary)
    raise FileNotFoundError(f"Expected the saved dllm_ar32_results.zip or its extracted files under {root}")


def select_cases(rows, eos, limit, length):
    from suffix_lab.long60_plan import until_eos

    selected = []
    for row in rows:
        ar = row["ar"]["token_ids"]
        spec = row["spec"]["token_ids"]
        for index, rnd in enumerate(row["spec"]["rounds"]):
            accepted, verified = rnd["accepted"], rnd["verified"]
            offset = rnd["offset"]
            start = offset + accepted + 1
            if accepted == verified or rnd["correction"] in eos or start >= min(limit, len(ar)):
                continue
            if ar[:start] != spec[:start] or ar[start - 1] != rnd["correction"]:
                continue
            old = until_eos(rnd["draft_ids"][accepted + 1:verified], eos, False)
            gold = ar[start:start + length + 1]
            future = until_eos(gold[:length], eos, False)
            if min(len(old), len(future)) < 4:
                continue
            prompt = row["prompt_tokens"]
            selected.append(dict(id=row["id"], dataset=row["dataset"], round=index, offset=offset,
                                 prompt=prompt, prefix=prompt + ar[:start], source_prefix=prompt + ar[:offset],
                                 source_proposal=rnd["draft_ids"][:verified], source_start=accepted + 1,
                                 old=old, gold=gold, future=future, remaining=limit - start,
                                 correction=rnd["correction"]))
            break
    if len({c["id"] for c in selected}) != len(rows) or len(selected) != len(rows):
        raise ValueError("Expected one eligible shared-prefix rejection for every saved prompt")
    return selected


def source_features(target, case):
    from suffix_lab.runtime import ARSession

    stats = ARSession(target).score(case["source_prefix"], case["source_proposal"], top_k=2)
    start = case["source_start"]
    stop = start + len(case["old"])
    return {name: stats[name][start:stop] for name in
            ("probability", "top1_probability", "logit_margin", "entropy", "top1")}


def candidate_rows(case, feedback):
    old = case["old"]
    candidates, features = [], []
    # Destinations depend on the canvas and output cap, never on the reference EOS.
    horizon = min(32, case["remaining"])
    for length in range(4, min(8, len(old), horizon) + 1):
        for i in range(len(old) - length + 1):
            span = old[i:i + length]
            q = feedback["probability"][i:i + length]
            m = feedback["logit_margin"][i:i + length]
            e = feedback["entropy"][i:i + length]
            w = feedback["top1_probability"][i:i + length]
            common = [length, i, len(old), case["source_start"], case["offset"],
                      len(case["prefix"]) % 32, case["dataset"] == "math500",
                      len(set(span)) / length, span[0] == case["correction"],
                      sum(x == y for x, y in zip(span, feedback["top1"][i:i + length])) / length,
                      min(q), mean(q), q[0], q[-1], mean(w), mean(m), mean(e)]
            for j in range(horizon - length + 1):
                if i == j:
                    continue
                candidates.append((i, j, length))
                features.append(common + [j, j - i, abs(j - i), j + length,
                                          sum(x == y for x, y in zip(span, old[j:j + length])) / length])
    return candidates, features


def labels(case, candidates):
    old, future = case["old"], case["future"]
    return [2 * sum(a == b for a, b in zip(old[i:i + n], future[j:j + n])) - n
            for i, j, n in candidates]


def predicted_anchors(cases, records):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    folds = {}
    for dataset in ("math500", "gsm8k"):
        keys = sorted(c["id"] for c in cases if c["dataset"] == dataset)
        random.Random(SEED).shuffle(keys)
        folds.update((key, i % 5) for i, key in enumerate(keys))
    tables = {}
    build_ms = {}
    for c in cases:
        start = perf_counter()
        candidates, features = candidate_rows(c, records[c["id"]])
        tables[c["id"]] = candidates, np.asarray(features, dtype=np.float32)
        build_ms[c["id"]] = 1000 * (perf_counter() - start)
    predictions = {}
    with threadpool_limits(limits=4):
        for fold in range(5):
            train = [c for c in cases if folds[c["id"]] != fold]
            x = np.concatenate([tables[c["id"]][1] for c in train])
            y = np.concatenate([labels(c, tables[c["id"]][0]) for c in train])
            model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, min_samples_leaf=50,
                                                  early_stopping=False, random_state=SEED).fit(x, y)
            for c in cases:
                key = c["id"]
                if folds[key] != fold:
                    continue
                candidates, features = tables[key]
                start = perf_counter()
                scores = model.predict(features)
                best = min(range(len(scores)), key=lambda k: (-float(scores[k]), candidates[k][1],
                           abs(candidates[k][1] - candidates[k][0]), candidates[k]))
                predictions[key] = dict(fold=fold, candidate_count=len(scores), span=candidates[best],
                                        score=float(scores[best]), candidate_build_ms=build_ms[key],
                                        predict_rank_ms=1000 * (perf_counter() - start))
            print(f"Fit placement fold {fold + 1}/5 on {len(train)} prompts and {len(y)} candidates", flush=True)
    return predictions


def keep_plans(case, prediction):
    from suffix_lab.long60_plan import matching_runs, monotone_chain

    old, future = case["old"], case["future"]
    spans = monotone_chain(matching_runs(old, future, 31, minimum=4))
    oracle = {r["destination_start"] + d: old[r["source_start"] + d]
              for r in spans for d in range(r["length"])}
    placed = {}
    if prediction["span"] is not None:
        i, j, n = prediction["span"]
        placed = dict(enumerate(old[i:i + n], start=j))
    remasked = [j for j, token in placed.items() if j >= len(future) or token != future[j]]
    return dict(zip(ARMS, ({}, oracle, {j: token for j, token in placed.items() if j not in remasked}),
                    strict=True)), spans, placed, remasked


def window_attention(inner):
    native_mask = inner.eval_mask

    def eval_mask(self, length, block_size, cache_length):
        if cache_length or length <= 32:
            raise ValueError("Expected a full prefix plus a 32-position draft with no draft KV cache")
        mask = native_mask(length, block_size, cache_length)
        boundary = length - 32
        # Prefix queries cannot see draft tokens except the predecessor row used to predict slot 0.
        mask[:boundary - 1, boundary:] = False
        mask[boundary - 1:, :] = True
        return mask

    inner.eval_mask = MethodType(eval_mask, inner)


def draft_once(model, prefix, anchors, block_size, mask_id):
    import torch

    canvas = [anchors.get(j, mask_id) for j in range(32)]
    ids = torch.tensor([prefix + canvas], device=model.device)
    output = model(ids, block_size=block_size, use_cache=False, use_block_cache=False, logits_to_keep=33)
    logits = output.logits[0, :32]
    if logits.shape[0] != 32:
        raise RuntimeError("Expected 32 shifted logit rows in one drafter call")
    logits[:, mask_id] = -torch.inf
    proposed = logits.argmax(-1).tolist()
    return [anchors.get(j, token) for j, token in enumerate(proposed)]


def reference_decision(proposal, gold, eos, remaining):
    from suffix_lab.long60_plan import decide

    required = min(len(proposal) + 1, remaining)
    if len(gold) < required and (not gold or gold[-1] not in eos):
        raise ValueError("Archived greedy reference has insufficient lookahead")
    # After EOS or the output cap, any rows needed only to satisfy decide's shape are unused.
    top1 = (gold + [min(eos)] * (len(proposal) + 1))[:len(proposal) + 1]
    decision = decide(proposal, top1, eos, remaining)
    if decision["committed_tokens"] != gold[:len(decision["committed_tokens"])]:
        raise ValueError("Archived greedy reference is inconsistent")
    return decision


def run_case(case, prediction, drafter, target, eos, block_size, mask_id):
    from benchmark_dllm_ar32 import CachedTarget
    from suffix_lab.long60_plan import decide, matching_runs, until_eos
    from suffix_lab.runtime import sync, sync_model

    plans, spans, placed, remasked = keep_plans(case, prediction)
    prefix = case["prefix"]
    start = perf_counter()
    verifier = CachedTarget(target, prefix)
    sync_model(target)
    prefix_ms = 1000 * (perf_counter() - start)
    order, results = list(ARMS), {}
    random.Random(case["id"]).shuffle(order)
    for name in order:
        keep = plans[name]
        sync(drafter.device)
        start = perf_counter()
        tokens = draft_once(drafter, prefix, keep, block_size, mask_id)
        sync(drafter.device)
        draft_ms = 1000 * (perf_counter() - start)
        proposal = until_eos(tokens[:case["remaining"]], eos)
        start = perf_counter()
        top1 = [verifier.next_token] + verifier.forward(proposal)
        verifier.cache.crop(len(prefix))
        if verifier.cache.get_seq_length() != len(prefix):
            raise RuntimeError("Verifier cache did not return to the shared prefix")
        sync_model(target)
        verify_ms = 1000 * (perf_counter() - start)
        decision = decide(proposal, top1, eos, case["remaining"])
        gold = case["gold"]
        canonical = reference_decision(proposal, gold, eos, case["remaining"])
        accepted = canonical["accepted_draft"]
        first_anchor = min(keep) if keep else None
        results[name] = dict(draft=tokens, verified_length=len(proposal), keep=keep,
                             remasked=remasked if name == ARMS[2] else [],
                             remasked_correct=sum(j < len(case["future"]) and tokens[j] == case["future"][j]
                                                  for j in remasked) if name == ARMS[2] else 0,
                             draft_forwards=1, verify_forwards=1, draft_ms=draft_ms, verify_ms=verify_ms,
                             accepted=accepted, committed=len(canonical["committed_tokens"]),
                             native_accepted=decision["accepted_draft"],
                             native_commit_exact=decision["committed_tokens"] == canonical["committed_tokens"],
                             anchors_accepted=sum(j < accepted for j in keep),
                             bridge_tokens=first_anchor,
                             bridge_accepted=first_anchor is not None and accepted >= first_anchor,
                             passed_last_anchor=bool(keep) and max(keep) < accepted)
    exact_destinations = sorted({j for span in matching_runs(case["old"], case["future"], 31)
                                 for j in range(span["destination_start"], span["destination_end"])})
    return dict(id=case["id"], dataset=case["dataset"], source_round=case["round"],
                old_suffix=case["old"], gold=case["gold"], prefix_ms=prefix_ms, model=prediction,
                model_placed=placed, oracle_spans=spans, oracle_exact_destinations=exact_destinations,
                arms=results)


def summarize(rows):
    groups = {"all": rows, "math500": [r for r in rows if r["dataset"] == "math500"],
              "gsm8k": [r for r in rows if r["dataset"] == "gsm8k"],
              "has_oracle_shift": [r for r in rows if r["oracle_spans"]],
              "has_model_placement": [r for r in rows if r["model_placed"]]}
    out = {}
    for name, subset in groups.items():
        if not subset:
            continue
        out[name] = {"cases": len(subset), "mean_shared_prefix_ms": mean(r["prefix_ms"] for r in subset)}
        for arm in ARMS:
            a = [r["arms"][arm] for r in subset]
            deltas = [r["arms"][arm]["accepted"] - r["arms"]["fresh"]["accepted"] for r in subset]
            ms = sum(r["draft_ms"] + r["verify_ms"] for r in a)
            out[name][arm] = dict(mean_accepted=mean(r["accepted"] for r in a),
                                  mean_native_accepted=mean(r["native_accepted"] for r in a),
                                  mean_committed=mean(r["committed"] for r in a),
                                  mean_delta_vs_fresh=mean(deltas),
                                  wins_ties_losses=[sum(x > 0 for x in deltas), deltas.count(0),
                                                    sum(x < 0 for x in deltas)],
                                  mean_draft_ms=mean(r["draft_ms"] for r in a),
                                  median_draft_ms=median(r["draft_ms"] for r in a),
                                  mean_verify_ms=mean(r["verify_ms"] for r in a),
                                  ratio_draft_verify=sum(r["draft_ms"] for r in a) / sum(r["verify_ms"] for r in a),
                                  measured_ms_per_canonical_committed=ms / sum(r["committed"] for r in a),
                                  draft_forwards=sum(r["draft_forwards"] for r in a),
                                  verify_forwards=sum(r["verify_forwards"] for r in a),
                                  cases_with_keep=sum(bool(r["keep"]) for r in a),
                                  kept_tokens=sum(len(r["keep"]) for r in a),
                                  kept_tokens_accepted=sum(r["anchors_accepted"] for r in a),
                                  cases_passed_bridge=sum(r["bridge_accepted"] for r in a),
                                  cases_passed_last_anchor=sum(r["passed_last_anchor"] for r in a),
                                  remasked=sum(len(r["remasked"]) for r in a),
                                  remasked_correct=sum(r["remasked_correct"] for r in a),
                                  native_commit_mismatches=sum(not r["native_commit_exact"] for r in a))
        total_placed = sum(len(r["model_placed"]) for r in subset)
        correct = out[name][ARMS[2]]["kept_tokens"]
        covered = sum(len(set(map(int, r["arms"][ARMS[2]]["keep"])) & set(r["oracle_exact_destinations"]))
                      for r in subset)
        exact_covered = sum(len(set(map(int, r["model_placed"])) & set(r["oracle_exact_destinations"]))
                            for r in subset if not r["arms"][ARMS[2]]["remasked"])
        oracle_tokens = sum(len(r["oracle_exact_destinations"]) for r in subset)
        out[name]["model_pre_remask"] = dict(placed_tokens=total_placed, correct_tokens=correct,
                                            token_precision=correct / total_placed if total_placed else None,
                                            whole_span_exact=sum(bool(r["model_placed"]) and
                                                                 not r["arms"][ARMS[2]]["remasked"] for r in subset),
                                            oracle_token_positions=oracle_tokens, oracle_positions_covered=covered,
                                            oracle_position_coverage=covered / oracle_tokens if oracle_tokens else None,
                                            oracle_positions_covered_by_wholly_exact_spans=exact_covered,
                                            whole_span_oracle_position_coverage=exact_covered / oracle_tokens
                                            if oracle_tokens else None,
                                            mean_candidate_build_ms=mean(r["model"]["candidate_build_ms"] for r in subset),
                                            mean_predict_rank_ms=mean(r["model"]["predict_rank_ms"] for r in subset))
    return out


def run(args):
    import torch
    from suffix_lab.assets import DLLM, ROOT, TARGET
    from suffix_lab.extension import prepare_models
    from suffix_lab.runtime import configure_numerics, load_model, load_target
    from suffix_lab.store import Store, atomic_json

    path, rows, archived = find_input(args.input)
    original = archived["manifest"]
    eos = set(original["eos_ids"])
    if Counter(r["dataset"] for r in rows) != {"math500": 100, "gsm8k": 100}:
        raise ValueError("Expected the completed 100+100 benchmark")
    cases = select_cases(rows, eos, original["max_new_tokens"], original["draft_length"])
    print(f"Loaded {len(cases)} fixed rejection cases from {path}", flush=True)
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Select Kaggle GPU T4 x2")
    config = json.loads((ROOT / "configs/long64.json").read_text(encoding="utf-8"))
    configure_numerics(config)
    paths, revisions = prepare_models()
    if revisions != original["models"] or config["mask_id"] != original["mask_id"]:
        raise ValueError("Model revisions or MASK differ from the source run")
    target, _, _ = load_target(paths[TARGET], config)
    target._suffix_exclude_peer_transfers = False
    drafter = load_model(paths[DLLM], "cuda:1")
    window_attention(drafter.model)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol="paired-one-step-shift-redraft32", cases_hash=CASE_HASH, models=revisions,
                    script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), torch=torch.__version__,
                    cuda=torch.version.cuda, gpus=[torch.cuda.get_device_name(i) for i in range(2)],
                    draft_length=32, steps=1, attention="all 32 prediction rows see full draft canvas",
                    model="5-fold held-out HGB, highest expected correct minus wrong, one placement per case",
                    timing="paired draft/verify GPU wall; prefix and offline oracle/model costs excluded")
    store = Store(output / "checkpoint.sqlite", manifest)
    try:
        with torch.inference_mode():
            records = {}
            for index, case in enumerate(cases, 1):
                key = case["id"]
                item = store.get("features", key)
                if item is None:
                    start = perf_counter()
                    item = dict(features=source_features(target, case), ms=1000 * (perf_counter() - start))
                    store.put("features", key, item)
                records[key] = item["features"]
                if index % 25 == 0:
                    print(f"Captured verifier features {index}/{len(cases)}", flush=True)
            predictions = predicted_anchors(cases, records)
            pending = [c for c in cases if store.get("case", c["id"]) is None]
            if pending:
                run_case(pending[0], predictions[pending[0]["id"]], drafter, target, eos,
                         config["physical_block_size"], config["mask_id"])
            for index, c in enumerate(pending, len(cases) - len(pending) + 1):
                result = run_case(c, predictions[c["id"]], drafter, target, eos,
                                  config["physical_block_size"], config["mask_id"])
                store.put("case", c["id"], result)
                print(f"{index}/{len(cases)} {c['id']} | accept " + " / ".join(
                    str(result["arms"][name]["accepted"]) for name in ARMS), flush=True)
    finally:
        finished = store.rows("case")
        features = store.rows("features")
        summary = dict(manifest=manifest, expected=len(cases), completed=len(finished),
                       mean_offline_feature_replay_ms=mean(r["ms"] for r in features) if features else None,
                       metrics=summarize(finished))
        atomic_json(output / "summary.json", summary)
        store.export("case", output / "cases.jsonl")
        store.close()
        temporary = output / "results.tmp"
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for name in ("summary.json", "cases.jsonl", "checkpoint.sqlite"):
                archive.write(output / name, name)
        temporary.replace(output / "shift_redraft32_results.zip")
    print(json.dumps(summary["metrics"], indent=2), flush=True)


def launch_kaggle(args):
    work = Path("/kaggle/working")
    repo = work / f"shift-redraft32-src-{BASE_REVISION[:12]}"
    if not (repo / ".git").is_dir():
        subprocess.run(["git", "init", str(repo)], check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    if head.stdout.strip() != BASE_REVISION:
        subprocess.run(["git", "fetch", "--depth", "1", REPOSITORY, BASE_REVISION], cwd=repo, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo, check=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != BASE_REVISION:
        raise RuntimeError("Unexpected benchmark dependency revision")
    setup = runpy.run_path(str(repo / "scripts/kaggle_single_cell.py"))
    stream = runpy.run_path(str(repo / "scripts/kaggle_extension_cell.py"))["run_streamed"]
    setup["check_gpus"]()
    venv = Path("/tmp/shift-redraft32-venv")
    log = work / "shift_redraft32_setup.log"
    python = setup["ensure_environment"](venv, log)
    stamp = venv / "installed-revision"
    if not stamp.exists() or stamp.read_text() != BASE_REVISION:
        install = setup["setup_command"]
        install([str(python), "-m", "pip", "install", "torch==2.9.0", "--index-url",
                 "https://download.pytorch.org/whl/cu126"], log)
        install([str(python), "-m", "pip", "install", "-r", str(repo / "requirements-kaggle.txt")], log)
        install([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], log)
        stamp.write_text(BASE_REVISION)
    env = setup["child_environment"](os.environ)
    env.setdefault("HF_HOME", "/tmp/suffix-hf-cache")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Place the launcher beside the baseline script so its CachedTarget can be reused.
    entry = repo / "scripts/benchmark_shift_redraft32.py"
    entry.write_bytes(Path(__file__).read_bytes())
    code = stream([str(python), "-u", str(entry), "--input", str(args.input.resolve()),
                   "--output", str(args.output.resolve())], cwd=repo, env=env,
                  log_path=work / "shift_redraft32_run.log")
    if code:
        raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kaggle", action="store_true")
    args = parser.parse_args()
    if args.kaggle:
        args.input, _, _ = find_input(args.input)
        launch_kaggle(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
