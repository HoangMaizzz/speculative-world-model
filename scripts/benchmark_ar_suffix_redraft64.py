"""Replay AR-AR64 rejections with three paired one-forward Fast-dLLM redrafts."""

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
from time import perf_counter
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from benchmark_shift_redraft32 import CachedDrafter, candidate_rows, evaluate_arms, source_features


LENGTH = 64
SEED = 42
ARMS = ("fresh", "oracle_shift", "predicted_shift")
THRESHOLDS = (.15, .25, .35, .45, .55, .65, .75, .85, .95, 1.01)
ARCHIVE = "ar_suffix_redraft64_results.zip"
INPUT_HASHES = {
    "cases.jsonl": "c93237e432344f8c1d0246021714cd1152db88a41d392b9fa9f01ad83f4754da",
    "summary.json": "89c8461209df639bca6d3c8e4f98ca6731c64135733864340c2c51f6afa7f1db",
    "prompts.json": "7ef8d45c124f9b205ca69dfd4b4fabbfc32ec096a995d5136fb00aa727be1248",
}


def find_input(root):
    paths = [root]
    if root.is_dir():
        paths.extend(sorted(root.rglob("ar_ar_reuse_results*.zip")))
        paths.extend(sorted({p.parent for p in root.rglob("cases.jsonl")}))
    for path in dict.fromkeys(paths):
        try:
            if path.is_dir():
                blobs = {n: (path / n).read_bytes() for n in INPUT_HASHES}
            else:
                with ZipFile(path) as archive:
                    blobs = {n: archive.read(n) for n in INPUT_HASHES}
        except (OSError, BadZipFile, KeyError):
            continue
        if all(sha256(blobs[n]).hexdigest() == h for n, h in INPUT_HASHES.items()):
            return path, list(map(json.loads, blobs["cases.jsonl"].splitlines())), json.loads(
                blobs["prompts.json"]), json.loads(blobs["summary.json"])
    raise FileNotFoundError(f"Attach the original ar_ar_reuse_results.zip or its extracted files under {root}")


def select_cases(rows, prompts, archive):
    from suffix_lab.long60_plan import until_eos

    manifest = archive["manifest"]
    eos, limit = set(manifest["eos_ids"]), manifest["max_new_tokens"]
    cases, skipped = [], []
    prompt_ids = {r["id"]: r["prompt_tokens"] for r in prompts}
    for row in rows:
        eligible = row["reuse"]["64"]["events"]
        if not eligible:
            skipped.append(dict(id=row["id"], dataset=row["dataset"], reason="no eligible rejection"))
            continue
        event = eligible[0]
        index, start = event["round"], event["corrected_offset"]
        step = row["spec"]["rounds"][index]
        prefix, offset, old = prompt_ids[row["id"]], step["offset"], event["old"]
        source_start = step["decision"]["first_rejection"] + 1
        ar = row["ar"]["token_ids"]
        proposal = step["proposal"]
        gold = ar[start:start + LENGTH + 1]
        future = until_eos(gold[:LENGTH], eos, include=False)
        if (start != offset + source_start or row["spec"]["token_ids"][:start] != ar[:start]
                or ar[start - 1] != step["decision"]["correction"]
                or old != until_eos(proposal[source_start:], eos, include=False)
                or future != event["reference"] or min(len(old), len(future)) < 4):
            raise ValueError(f"Invalid archived rejection near {row['id']} at {start}")
        cases.append(dict(id=row["id"], dataset=row["dataset"], round=index, offset=offset,
                          prefix=prefix + ar[:start], source_prefix=prefix + ar[:offset],
                          source_proposal=proposal, source_start=source_start,
                          correction=step["decision"]["correction"], old=old, gold=gold,
                          future=future, remaining=limit - start))
    return cases, skipped


def fold_map(cases):
    out = {}
    for dataset in ("math500", "gsm8k"):
        ids = sorted(r["id"] for r in cases if r["dataset"] == dataset)
        random.Random(SEED).shuffle(ids)
        out.update((key, i % 5) for i, key in enumerate(ids))
    return out


def exact_labels(case, candidates):
    old, future = case["old"], case["future"]
    return [old[i:i + n] == future[j:j + n] for i, j, n in candidates]


def select_spans(candidates, scores, order, threshold):
    """Keep disjoint spans with a consistent left-to-right source/target order."""
    chosen = []
    for index in order:
        score = float(scores[index])
        if score < threshold:
            break
        i, j, n = candidates[index]
        if all((i + n <= a and j + n <= b) or (a + k <= i and b + k <= j) for a, b, k in chosen):
            chosen.append((i, j, n))
    return sorted(chosen, key=lambda span: span[1])


def anchors_from_spans(old, spans):
    return {j + d: old[i + d] for i, j, n in spans for d in range(n)}


def correctness(anchors, future):
    right = sum(j < len(future) and future[j] == token for j, token in anchors.items())
    return right, len(anchors) - right


def predictions(cases, feedback):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    tables, build_ms = {}, {}
    for case in cases:
        tick = perf_counter()
        candidates, features = candidate_rows(case, feedback[case["id"]], horizon_limit=LENGTH)
        tables[case["id"]] = candidates, np.asarray(features, dtype=np.float32)
        build_ms[case["id"]] = 1000 * (perf_counter() - tick)
    folds = fold_map(cases)
    output, history = {}, []
    rng = np.random.default_rng(SEED)
    with threadpool_limits(limits=4):
        for test in range(5):
            val = (test + 1) % 5
            train_x, train_y = [], []
            for case in cases:
                key = case["id"]
                if folds[key] in (test, val):
                    continue
                cand, x = tables[key]
                labels = np.asarray(exact_labels(case, cand), dtype=bool)
                pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
                sample = np.concatenate((pos, rng.choice(neg, min(512, len(neg)), replace=False)))
                train_x.append(x[sample])
                train_y.append(labels[sample])
            y = np.concatenate(train_y)
            if len(set(y)) != 2:
                raise ValueError("Fold requires exact placements and wrong placements")
            model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, min_samples_leaf=25,
                                                   early_stopping=False, random_state=SEED).fit(
                                                       np.concatenate(train_x), y)
            validation = {t: [0, 0] for t in THRESHOLDS}
            for case in cases:
                key = case["id"]
                if folds[key] != val:
                    continue
                cand, x = tables[key]
                scores = model.predict_proba(x)[:, 1]
                order = np.argsort(-scores, kind="stable")
                for threshold, counts in validation.items():
                    spans = select_spans(cand, scores, order, threshold)
                    right, wrong = correctness(anchors_from_spans(case["old"], spans), case["future"])
                    counts[0] += right
                    counts[1] += wrong
            # A wrong hard anchor can stop the entire accepted prefix.
            threshold = max(THRESHOLDS, key=lambda t: (validation[t][0] - 2 * validation[t][1], t))
            history.append(dict(test=test, validation=val, train_prompts=len(train_x),
                                positive_train_candidates=int(y.sum()), training_candidates=len(y),
                                threshold=threshold, validation_right_wrong=validation))
            for case in cases:
                key = case["id"]
                if folds[key] != test:
                    continue
                cand, x = tables[key]
                tick = perf_counter()
                scores = model.predict_proba(x)[:, 1]
                spans = select_spans(cand, scores, np.argsort(-scores, kind="stable"), threshold)
                output[key] = dict(fold=test, spans=spans, threshold=threshold, candidate_count=len(cand),
                                   candidate_ms=build_ms[key], score_and_select_ms=1000 * (perf_counter() - tick))
            print(f"Fold {test + 1}/5 | train {len(train_x)} | held-out threshold {threshold:.2f}", flush=True)
    return output, history


def paired_case(case, prediction, drafter, target, eos):
    from suffix_lab.long60_plan import matching_runs, monotone_chain

    runs = matching_runs(case["old"], case["future"], LENGTH - 1)
    chain = monotone_chain(runs)
    oracle = {r["destination_start"] + d: case["old"][r["source_start"] + d]
              for r in chain for d in range(r["length"])}
    predicted = anchors_from_spans(case["old"], prediction["spans"])
    result = evaluate_arms(case, dict(zip(ARMS, ({}, oracle, predicted), strict=True)), drafter, target, eos)
    labels = exact_labels(case, prediction["spans"])
    right, wrong = correctness(predicted, case["future"])
    oracle_positions = {j for run in runs for j in range(run["destination_start"], run["destination_end"])}
    return dict(result, prediction=prediction, oracle_spans=chain, oracle_candidate_positions=len(oracle_positions),
                predicted_correct=right, predicted_wrong=wrong, predicted_exact_spans=sum(labels),
                predicted_exact_covered=sum(n for (i, j, n), ok in zip(
                    prediction["spans"], labels) if ok))


def summarize(rows):
    output = {}
    for name in ("all", "math500", "gsm8k", "oracle_available", "prediction_kept"):
        subset = [r for r in rows if name == "all" or r["dataset"] == name
                  or name == "oracle_available" and r["oracle_spans"]
                  or name == "prediction_kept" and r["prediction"]["spans"]]
        if not subset:
            continue
        arms = {}
        for arm in ARMS:
            metrics = [r["arms"][arm] for r in subset]
            deltas = [r["arms"][arm]["accepted"] - r["arms"]["fresh"]["accepted"] for r in subset]
            seconds = sum(m["draft_ms"] + m["verify_ms"] for m in metrics)
            committed = sum(m["committed"] for m in metrics)
            selection = (sum(r["prediction"]["candidate_ms"] + r["prediction"]["score_and_select_ms"]
                             for r in subset) if arm == "predicted_shift" else 0)
            arms[arm] = dict(accepted=mean(m["accepted"] for m in metrics),
                             native_accepted=mean(m["native_accepted"] for m in metrics),
                             native_commit_mismatch=sum(not m["native_commit_exact"] for m in metrics),
                             delta_accept=mean(deltas),
                             wins_ties_losses=[sum((d > 0) - (d < 0) == s for d in deltas) for s in (1, 0, -1)],
                             draft_ms=mean(m["draft_ms"] for m in metrics),
                             median_draft_ms=median(m["draft_ms"] for m in metrics),
                             verify_ms=mean(m["verify_ms"] for m in metrics),
                             mean_verified_length=mean(m["verified_length"] for m in metrics),
                             exact_64_submitted=sum(m["verified_length"] == LENGTH for m in metrics),
                             selector_ms=selection / len(metrics),
                             measured_ms_per_committed=seconds / committed,
                             measured_ms_per_committed_with_selector=(seconds + selection) / committed,
                             kept_tokens=sum(len(m["keep"]) for m in metrics),
                             past_last_anchor=sum(m["passed_last_anchor"] for m in metrics))
        correct, wrong = sum(r["predicted_correct"] for r in subset), sum(r["predicted_wrong"] for r in subset)
        nspans = sum(len(r["prediction"]["spans"]) for r in subset)
        exact_spans = sum(r["predicted_exact_spans"] for r in subset)
        oracle_positions = sum(r["oracle_candidate_positions"] for r in subset)
        coverage = sum(r["predicted_exact_covered"] for r in subset)
        output[name] = dict(cases=len(subset), arms=arms, predicted_correct=correct, predicted_wrong=wrong,
                            predicted_token_precision=correct / (correct + wrong) if correct + wrong else None,
                            predicted_spans=nspans, exact_spans=exact_spans,
                            whole_span_precision=exact_spans / nspans if nspans else None,
                            oracle_positions=oracle_positions, whole_span_covered=coverage,
                            whole_span_coverage=coverage / oracle_positions if oracle_positions else None,
                            mean_draft_prefill_ms=mean(r["draft_prefix_ms"] for r in subset),
                            mean_verify_prefill_ms=mean(r["prefix_ms"] for r in subset))
    return output


def save(store, output, manifest, cases, skipped):
    from suffix_lab.store import atomic_json

    finished = store.rows("case")
    features = store.rows("features")
    atomic_json(output / "summary.json", dict(
        status="complete" if len(finished) == len(cases) else "partial",
        expected=len(cases), completed=len(finished), skipped=skipped, manifest=manifest,
        mean_offline_feature_replay_ms=mean(r["ms"] for r in features) if features else None,
        metrics=summarize(finished)))
    store.export("case", output / "cases.jsonl")
    snapshot = output / "checkpoint.snapshot.sqlite"
    store.backup(snapshot)
    temporary = output / "results.tmp"
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        archive.write(snapshot, "checkpoint.sqlite")
        for name in ("summary.json", "cases.jsonl", "cache_audit.json", "folds.json"):
            if (output / name).is_file():
                archive.write(output / name, name)
    temporary.replace(output / ARCHIVE)
    snapshot.unlink()


def run(args):
    import torch
    import transformers
    from suffix_lab.assets import DLLM, ROOT, TARGET, source_digest
    from suffix_lab.extension import prepare_models
    from suffix_lab.runtime import configure_numerics, load_model, load_target, tokenizers
    from suffix_lab.store import Store, atomic_json

    path, rows, prompts, archived = find_input(args.input)
    cases, skipped = select_cases(rows, prompts, archived)
    print(f"{path}: {len(cases)} first rejections; skipped {len(skipped)} archived prompts", flush=True)
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Select GPU T4 x2")
    config = json.loads((ROOT / "configs/long64.json").read_text(encoding="utf-8"))
    configure_numerics(config)
    paths, revisions = prepare_models()
    _, vocab = tokenizers(paths)
    if (revisions[TARGET] != archived["manifest"]["models"][TARGET]
            or vocab["shared_vocab_sha256"] != archived["manifest"]["tokenizer_sha256"]):
        raise ValueError("Verifier or tokenizer differs from the archived run")
    target, _, _ = load_target(paths[TARGET], config)
    drafter = CachedDrafter(load_model(paths[DLLM], "cuda:1"), config["physical_block_size"],
                           config["mask_id"], draft_length=LENGTH)
    eos = set(archived["manifest"]["eos_ids"])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol="ar-suffix-one-step-dllm-redraft64", git_revision=args.revision,
                    input_hashes=INPUT_HASHES, source_digest=source_digest(), models=revisions,
                    scripts={name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in (Path(__file__).name, "benchmark_shift_redraft32.py")},
                    draft_length=LENGTH, unmask_forwards=1, full_canvas_attention=True,
                    model="HGB exact span placement; 3 train folds, 1 validation fold, 1 held-out fold",
                    thresholds=list(THRESHOLDS), validation_utility="right tokens - 2 * wrong tokens",
                    torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                    gpus=[torch.cuda.get_device_name(i) for i in range(2)])
    store = Store(output / "checkpoint.sqlite", manifest)
    try:
        with torch.inference_mode():
            audit_cases = {}
            for case in cases:
                audit_cases.setdefault(len(case["prefix"]) % config["physical_block_size"], case)
            atomic_json(output / "cache_audit.json", drafter.audit(list(audit_cases.values())[::6]))
            feedback = {}
            for index, case in enumerate(cases, 1):
                key = case["id"]
                record = store.get("features", key)
                if record is None:
                    tick = perf_counter()
                    record = dict(features=source_features(target, case), ms=1000 * (perf_counter() - tick))
                    store.put("features", key, record)
                feedback[key] = record["features"]
                if index % 25 == 0:
                    print(f"Verifier feedback {index}/{len(cases)}", flush=True)
            predicted, history = predictions(cases, feedback)
            atomic_json(output / "folds.json", history)
            pending = [r for r in cases if store.get("case", r["id"]) is None]
            if pending:
                paired_case(pending[0], predicted[pending[0]["id"]], drafter, target, eos)
            for index, case in enumerate(pending, len(cases) - len(pending) + 1):
                result = paired_case(case, predicted[case["id"]], drafter, target, eos)
                store.put("case", case["id"], result)
                print(f"{index}/{len(cases)} {case['id']} | accept " + " / ".join(
                    str(result["arms"][arm]["accepted"]) for arm in ARMS), flush=True)
                if index % 25 == 0:
                    save(store, output, manifest, cases, skipped)
    finally:
        try:
            save(store, output, manifest, cases, skipped)
        finally:
            store.close()
    print("Result ZIP:", output / ARCHIVE, flush=True)


def launch(args):
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != args.revision:
        raise ValueError("Notebook and checkout revisions differ")
    source, *_ = find_input(args.input)
    setup = runpy.run_path(str(repo / "scripts/kaggle_single_cell.py"))
    stream = runpy.run_path(str(repo / "scripts/kaggle_extension_cell.py"))["run_streamed"]
    setup["check_gpus"]()
    log = Path("/kaggle/working/ar_suffix_redraft64_setup.log")
    venv = Path("/tmp/ar-suffix-redraft64-venv")
    python = setup["ensure_environment"](venv, log)
    stamp = venv / "installed-revision"
    if not stamp.exists() or stamp.read_text() != head:
        install = setup["setup_command"]
        install([str(python), "-m", "pip", "install", "torch==2.9.0", "--index-url",
                 "https://download.pytorch.org/whl/cu126"], log)
        install([str(python), "-m", "pip", "install", "-r", str(repo / "requirements-kaggle.txt")], log)
        install([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], log)
        stamp.write_text(head)
    env = setup["child_environment"](os.environ)
    env.setdefault("HF_HOME", "/tmp/suffix-hf-cache")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    code = stream([str(python), "-u", str(Path(__file__).resolve()), "--revision", head,
                   "--input", str(source.resolve()), "--output", str(args.output.resolve())],
                  cwd=repo, env=env, log_path="/kaggle/working/ar_suffix_redraft64_run.log")
    if code:
        raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="local")
    parser.add_argument("--kaggle", action="store_true")
    args = parser.parse_args()
    (launch if args.kaggle else run)(args)


if __name__ == "__main__":
    main()
