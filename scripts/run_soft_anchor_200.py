"""Fresh-question Soft Anchor evaluation with one preselected rejection per prompt."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import random
import sqlite3
from statistics import mean
import sys
import time
from zipfile import ZipFile, ZIP_DEFLATED

import run_soft_anchor_pilot as pilot
from suffix_lab.long60_plan import decide, until_eos
from suffix_lab.store import Store, atomic_json, digest


ROOT = Path(__file__).resolve().parents[1]
ASSET_SHA = "1b7e9c4e1ab4dd995ba310c262adefae1e4b54eff54d85f5bd849829857a92e9"
DOMAINS = {"math500": ("HuggingFaceH4/MATH-500", "problem"), "gsm8k": ("openai/gsm8k", "question")}
ASSET_MEMBERS = ("selected_joint_model.pt", "projection.npz", "training_sources.json",
                 "scripts/benchmark_savh_spans.py", "scripts/benchmark_learned_anchors.py",
                 "scripts/benchmark_token_link_anchors.py")
THRESHOLD = 0.575
SEARCH_ROUNDS = 6
OUTPUT_CAP = 256


def install_assets(path, output):
    if sha256(path.read_bytes()).hexdigest() != ASSET_SHA:
        raise ValueError("Input ZIP checksum differs from the frozen head package")
    with ZipFile(path) as archive:
        for name in ASSET_MEMBERS:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    sys.path.insert(0, str(output / "scripts"))
    return json.loads((output / "training_sources.json").read_text(encoding="utf8"))


def question_hash(text):
    return sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def sample_domain(domain, questions, old, count):
    excluded_ids = {item["id"] for item in old}
    excluded_hashes = {item["hash"] for item in old}
    seen, pool = set(), []
    for index, question in enumerate(questions):
        key = f"{domain}:{index}"
        qhash = question_hash(question)
        if key in excluded_ids or qhash in excluded_hashes or qhash in seen:
            continue
        seen.add(qhash)
        pool.append(dict(id=key, dataset=domain, index=index, question=question, question_sha256=qhash))
    if len(pool) < count:
        raise ValueError(f"Insufficient unseen questions in {domain}: {len(pool)}")
    return random.Random(f"soft-anchor-200-v1/{domain}").sample(pool, count)


def make_plan(assets, mode):
    from datasets import load_dataset
    lock = json.loads((ROOT / "source-lock.json").read_text(encoding="utf8"))
    groups = {}
    for ds, (name, key) in DOMAINS.items():
        options = {"name": "main"} if ds == "gsm8k" else {}
        data = load_dataset(name, split="test", revision=lock["datasets"][name], **options)
        groups[ds] = sample_domain(ds, data[key], assets["prompts"], 100)
    count = 2 if mode == "smoke" else 100
    return [groups[ds][i] for i in range(count) for ds in DOMAINS]


def legal_targets(source_start, length, prefix_len, remaining):
    return [j for j in range(1, 8) if j != source_start and j + length <= remaining
            and prefix_len % 32 + j + length <= 32]


def collect_source(question, bundle):
    """Choose the first structurally eligible reject, without future labels or head scores."""
    prompt = bundle.tokenizer.apply_chat_template([{"role": "user", "content": question["question"]}],
                                                   tokenize=True, add_generation_prompt=True)
    verifier = bundle.verifier
    verifier.begin_prompt(prompt, [])
    committed = []
    for round_index in range(SEARCH_ROUNDS):
        offset = len(committed)
        prefix = prompt + committed
        if len(prefix) + 64 > 4096:
            return None, "context_limit", round_index
        draft = bundle.drafter.draft(prefix, 64, anchors={})
        if draft.stats["unmask_forwards"] != 64:
            raise AssertionError("Fresh B64 must fill one mask per forward")
        proposal = until_eos(draft.tokens, bundle.eos)
        feedback = verifier.score(prefix, proposal)
        decision = decide(proposal, feedback["top1_with_bonus"], bundle.eos, OUTPUT_CAP - offset)
        committed.extend(decision["committed_tokens"])
        if decision["terminated"] or len(committed) == OUTPUT_CAP:
            return None, "ended_before_eligible_reject", round_index + 1
        rejected = decision["first_rejection"]
        if rejected is None:
            continue
        source = until_eos(proposal[rejected + 1:], bundle.eos, False)
        remaining = OUTPUT_CAP - len(committed)
        if len(source) < 8 or not legal_targets(0, 4, len(prompt) + len(committed), remaining):
            continue
        if verifier.cursor.prefix != prompt + committed:
            raise AssertionError("Corrected prefix cache and commit differ")
        return dict(prompt=prompt, committed=committed, prefix=prefix, proposal=proposal,
                    draft=draft, source=source, source_start=rejected + 1, offset=offset,
                    remaining=remaining, parallel=feedback["parallel_feedback"]["top1_with_bonus"]), None, round_index + 1
    return None, "no_eligible_reject_in_first_six_rounds", SEARCH_ROUNDS


def make_features(source, logits, hidden, projection, dataset):
    """Match the exact SAVH feature layout and train-only layer-27 projection."""
    import numpy as np
    from suffix_lab.verifier_features.replay_alignment import draft_features
    values, _ = draft_features(logits, source["proposal"], 16)
    if values["top1"] != source["parallel"]:
        return None
    tokens, start = source["source"], source["source_start"]
    n = len(tokens)
    stop = start + n
    vec = hidden.numpy().astype("float32")
    if vec.shape != (n, 3584):
        raise ValueError("Wrong post-read layer-27 tensor shape")
    norms = np.linalg.norm(vec, axis=1)
    normal = vec / np.maximum(norms[:, None], 1e-8)
    pcs = normal @ projection["components"].T - projection["mean"] @ projection["components"].T
    draft = source["draft"]
    scalar = np.column_stack([np.arange(n)/64, np.full(n,n/64), np.full(n,start/64),
                              np.full(n,source["offset"]/256), np.full(n,dataset == "math500"),
                              draft.confidence[start:stop], np.asarray(draft.order[start:stop])/64,
                              *[values[key][start:stop] for key in ("verifier_logp_draft", "verifier_greedy_match",
                                  "verifier_gap_to_top1", "entropy", "logit_margin", "top1_probability")],
                              np.log1p(values["verifier_rank_draft"][start:stop]),
                              [tokens.count(t)/n for t in tokens], np.log1p(norms)]).astype("float32")
    features = np.column_stack((scalar, pcs)).astype("float32")
    if features.shape != (n, 48) or not np.isfinite(features).all():
        raise ValueError("Wrong SAVH input feature shape or values")
    return features


def capture(source, target, projection, dataset):
    start = len(source["prefix"]) + source["source_start"]
    positions = list(range(start, start + len(source["source"])))
    logits, hidden, _, _ = target.forward(source["prefix"] + source["proposal"], len(source["proposal"]) + 1, positions)
    return make_features(source, logits, hidden["layer_27"], projection, dataset)


def reference(cursor, length, eos):
    tokens = []
    for i in range(length):
        token, _ = cursor.prediction()
        tokens.append(token)
        if token in eos or i + 1 == length:
            break
        cursor.advance(token)
    return tokens


def choose_anchor(source, head, gold):
    """Freeze the highest score source first, then use oracle placement for that span."""
    prefix_len = len(source["prompt"]) + len(source["committed"])
    for span in sorted(head["anchors"], key=lambda item: (-item["score"], item["start"], item["end"])):
        left, end = span["start"], span["end"]
        length = end - left
        if not 4 <= length <= 8:
            continue
        targets = legal_targets(left, length, prefix_len, min(len(gold), source["remaining"]))
        if not targets:
            continue
        tokens = source["source"][left:end]
        choices = []
        for j in targets:
            errors = [i for i, t in enumerate(tokens) if t != gold[j + i]]
            choices.append((len(errors), j, errors))
        _, j, errors = min(choices)
        rank = sorted(range(length), key=lambda i: (head["token_probabilities"][left+i], i))
        return dict(source_start=left, destination=j, tokens=tokens, errors=errors, rank=rank, score=span["score"])
    return None


def prepare_case(question, bundle, target, projection, assets):
    from benchmark_token_link_anchors import infer
    from suffix_lab.runtime import sync_model
    begin = time.perf_counter()
    source, reason, rounds = collect_source(question, bundle)
    search_ms = (time.perf_counter() - begin)*1000
    if source is None:
        return None, dict(state=reason, source_rounds=rounds, search_ms=search_ms)
    begin = time.perf_counter()
    features = capture(source, target, projection, question["dataset"])
    head = infer(assets / "selected_joint_model.pt", features, THRESHOLD) if features is not None else None
    feature_ms = (time.perf_counter() - begin)*1000
    begin = time.perf_counter()
    gold = reference(bundle.verifier.cursor, min(65, source["remaining"]), bundle.eos)
    sync_model(target.model)
    reference_ms = (time.perf_counter() - begin)*1000
    anchor = choose_anchor(source, head, gold) if head is not None else None
    distance = len(anchor["errors"]) if anchor else None
    cohort = {None: "representative", 0: "exact", 1: "one_error"}.get(distance, "other")
    if distance == 2 and len(anchor["tokens"]) >= 6:
        cohort = "two_errors"
    case = {key: source[key] for key in ("prompt", "committed", "remaining")}
    case.update(id=f"{question['id']}:{rounds-1:05d}", dataset=question["dataset"], prompt_id=question["id"],
                gold=gold, anchor=anchor, cohort=cohort)
    status = dict(state="prepared", source_rounds=rounds, search_ms=search_ms, feature_ms=feature_ms,
                  reference_ms=reference_ms, distance=distance,
                  head_replay_passed=features is not None, anchor_count=len(head["anchors"]) if head else None)
    return case, status


def policy_summary(store):
    """Score head-mask with fresh fallback on every completed, eligible question."""
    cases = store.rows("case")
    result = {}
    for domain in ("all", *DOMAINS):
        group = [case for case in cases if domain == "all" or case["dataset"] == domain]
        deltas = []
        for case in group:
            arms = {arm["arm"]: arm["metrics"]["accepted"] for arm in case["arms"]}
            deltas.append(arms.get("head1", arms["fresh64"]) - arms["fresh64"])
        if deltas:
            result[domain] = dict(paired_questions=len(deltas), mean_delta_vs_fresh=mean(deltas),
                                  wins=sum(d > 0 for d in deltas), ties=sum(d == 0 for d in deltas),
                                  losses=sum(d < 0 for d in deltas),
                                  note="Oracle placement, frozen head mask, fresh fallback if no anchor.")
    return result


def save(output, store, status):
    rows = pilot.summarize(store)
    status = {**status, "prompt_states": dict(Counter(row["state"] for row in store.rows("prompt"))),
              "verified_cases": len(store.rows("case")), "quarantined": len(store.rows("failure"))}
    atomic_json(output / "execution_status.json", status)
    atomic_json(output / "summary.json", {"by_cohort": rows, "policy": policy_summary(store)})
    if rows:
        with (output / "summary.csv").open("w", encoding="utf8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for kind, name in (("case", "cases.jsonl"), ("failure", "numerical_failures.jsonl"), ("prompt", "prompts.jsonl")):
        store.export(kind, output / name)
    store.backup(output / "checkpoint.sqlite")
    path = output / "soft_anchor_200_results.zip"
    temp = path.with_suffix(".tmp")
    with ZipFile(temp, "w", ZIP_DEFLATED) as archive:
        for file in output.iterdir():
            if file.is_file() and file not in (path, temp) and not file.name.startswith("live.sqlite"):
                archive.write(file, file.name)
    temp.replace(path)


def restore(path, output, identity):
    live = output / "live.sqlite"
    if live.exists():
        raise ValueError("Resume ZIP is for a new output folder. Existing live SQLite is preserved.")
    temp = output / "resume.sqlite.tmp"
    with ZipFile(path) as archive:
        temp.write_bytes(archive.read("checkpoint.sqlite"))
        with sqlite3.connect(temp.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            row = db.execute("SELECT payload FROM records WHERE kind='meta' AND key='identity'").fetchone()
            valid = row and digest(json.loads(row[0])) == digest(identity) and db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        db.close()
        if not valid:
            raise ValueError("Resume ZIP plan/revision or SQLite integrity differs")
        if "runtime.json" in archive.namelist():
            atomic_json(output / "runtime.json", json.loads(archive.read("runtime.json")))
    temp.replace(live)


def run(args):
    import numpy as np
    import torch
    from suffix_lab.long60 import load_bundle
    from suffix_lab.verifier_features.target_model import Target
    torch.set_num_threads(4)
    begin = time.monotonic()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "head_assets"
    source_info = install_assets(args.input, assets)
    questions = make_plan(source_info, args.mode)
    config = json.loads((ROOT / "configs/long64.json").read_text(encoding="utf8"))
    identity = dict(protocol="soft-anchor-200-v1", revision=args.revision, asset_sha=ASSET_SHA,
                    mode=args.mode, questions_sha=digest(questions), config=config,
                    search_rounds=SEARCH_ROUNDS, threshold=THRESHOLD)
    if args.resume:
        restore(args.resume, output, identity)
    store = Store(output / "live.sqlite", identity)
    atomic_json(output / "plan.json", dict(identity=identity, questions=questions))
    status = dict(status="failed", planned=len(questions), revision=args.revision)
    try:
        pending = [q for q in questions if store.get("prompt", q["id"]) is None]
        if pending:
            bundle = load_bundle(config, output)
            runtime = json.loads((output / "runtime.json").read_text(encoding="utf8"))
            if (runtime["models"] != source_info["models"] or set(bundle.eos) != set(source_info["eos"])
                    or runtime["tokenizer"]["shared_vocab_sha256"] != source_info["tokenizer_sha256"]):
                raise ValueError("Model, EOS or tokenizer differs from the trained head data")
            target = Target(bundle.verifier.parallel.model, [27])
            with np.load(assets / "projection.npz", allow_pickle=False) as file:
                projection = {key: file[key] for key in ("mean", "components")}
            for question in pending:
                if time.monotonic() - begin >= args.budget_seconds:
                    status["status"] = "budget_checkpoint"
                    break
                print(f"[{len(store.rows('prompt'))+1}/{len(questions)}] {question['id']}", flush=True)
                completed = []
                case, progress = prepare_case(question, bundle, target, projection, assets)
                if case is not None:
                    arms = list(pilot.actions(case).items())
                    random.Random("soft-anchor-200-order:"+case["id"]).shuffle(arms)
                    try:
                        for arm, masks in arms:
                            row = pilot.execute_arm(case, arm, masks, bundle)
                            completed.append(row)
                            print(f"  {arm}: accepted={row['metrics']['accepted']}", flush=True)
                    except pilot.ReferenceMismatch as error:
                        progress["state"] = "quarantined"
                        store.put("failure", question["id"], dict(case=case, evidence=error.evidence, partial_arms=completed))
                    else:
                        progress["state"] = "paired" if case["anchor"] else "fresh_only"
                        store.put("case", question["id"], {**case, "arms": completed})
                store.put("prompt", question["id"], dict(id=question["id"], dataset=question["dataset"], **progress))
                if len(store.rows("prompt")) % 5 == 0:
                    save(output, store, {**status, "status": "running"})
        if status["status"] != "budget_checkpoint":
            status["status"] = "complete"
    except BaseException as error:
        status["error"] = str(error)
        raise
    finally:
        save(output, store, status)
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--budget-seconds", type=int, default=18000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
