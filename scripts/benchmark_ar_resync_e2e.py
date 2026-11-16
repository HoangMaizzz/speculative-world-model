"""Paired end-to-end greedy AR-AR64 decoding with two rejected-suffix COPY policies."""

import argparse
from collections import Counter
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark_ar_ar_reuse import CachedDrafter, CachedTarget, DATASETS, first_difference
from benchmark_causal_resync import FEATURES, before_eos, candidates, lcp
from train_ar_resync_gain import predict

ARMS = ("fresh", "checked_copy", "direct_copy")
ARCHIVE = "ar_resync_e2e_results.zip"
DRAFT_LENGTH = 64
MAX_PROBE = 16
CHECKED_LENGTH = 4
EVAL_SEED = 2718


class CopyPolicy:
    def __init__(self, arm, old, context, head):
        self.arm = arm
        self.old = old
        self.context = context
        self.head = head
        self.idf = {int(k): v for k, v in head["idf"].items()} if arm == "direct_copy" else {}

    def select(self, proposal, budget):
        t = len(proposal)
        if not self.old or t > MAX_PROBE or t + CHECKED_LENGTH > budget:
            return None
        if self.arm == "checked_copy":
            if t < CHECKED_LENGTH:
                return None
            choices = []
            for i, token in enumerate(self.old[:-CHECKED_LENGTH]):
                if token == proposal[-1]:
                    k = lcp(proposal[-8:][::-1], self.old[max(0, i - 7):i + 1][::-1])
                    if k >= CHECKED_LENGTH:
                        choices.append((-k, abs(t - 1 - i), i))
            if not choices:
                return None
            neg_k, _, i = min(choices)
            return dict(source=i + 1, tail=self.old[i + 1:i + 1 + CHECKED_LENGTH], match=-neg_k)

        import numpy as np

        event = dict(self.context, old=self.old, observed=proposal, budget=budget)
        best, best_score = None, self.head["threshold"]
        for action, features in candidates(event, self.idf, self.head["unknown"], last_only=True):
            _, i, length, k, _, _ = action
            score = predict(self.head["trees"], self.head["intercept"], np.asarray(features, dtype=np.float32))
            if score >= self.head["threshold"] and (best is None or score > best_score):
                best_score = score
                best = dict(source=i + 1, tail=self.old[i + 1:i + 1 + length], match=k, score=score)
        return best


class CopyDrafter(CachedDrafter):
    def propose_copy(self, length, eos, policy, *, checked):
        self.prefix_length = self.cache.get_seq_length() + len(self.pending)
        forwards = int(bool(self.pending))
        token = self.forward(self.pending)[-1] if self.pending else self.next_token
        proposal, action = [token], None
        search_ms = block_ms = 0.0
        while len(proposal) < length and proposal[-1] not in eos:
            t = len(proposal)
            picked = None
            if action is None and policy.old and t <= MAX_PROBE:
                started = perf_counter()
                picked = policy.select(proposal, length)
                search_ms += (perf_counter() - started) * 1000
            if picked is None:
                proposal.append(self.forward(proposal[-1:])[-1])
                forwards += 1
                continue

            tail = picked["tail"]
            action = dict(picked, destination=t, kept=0, block_forward=0, first_token_skip=False)
            has_room = t + len(tail) < length
            block = tail if has_room else tail[:-1]
            if checked:
                next_id = self.forward(proposal[-1:])[-1]
                forwards += 1
                if next_id != tail[0]:
                    action["first_token_skip"] = True
                    proposal.append(next_id)
                    continue
                started = perf_counter()
                top1 = [next_id] + self.forward(block)
                kept = lcp(tail, top1)
                valid_cache = min(self.prefix_length + t + kept, self.cache.get_seq_length())
                if valid_cache < self.cache.get_seq_length():
                    self.cache.crop(valid_cache)
                proposal.extend(tail[:kept])
                if len(proposal) < length:
                    proposal.append(top1[kept])
                action["kept"] = kept
            else:
                started = perf_counter()
                # Update KV under the corrected prefix without checking COPY against q.
                top1 = self.forward(proposal[-1:] + block)
                proposal.extend(tail)
                if has_room:
                    proposal.append(top1[-1])
                action["kept"] = len(tail)
            block_ms += (perf_counter() - started) * 1000
            action["block_forward"] = 1
            forwards += 1

        if self.cache.get_seq_length() != self.prefix_length + len(proposal) - 1:
            raise AssertionError("Drafter must leave exactly its final proposal token pending")
        return proposal, forwards, action, search_ms, block_ms


def decode(target, drafter, prompt, eos, limit, arm, head, sync):
    sync(drafter)
    sync(target)
    started = perf_counter()
    draft_state = CopyDrafter(drafter, prompt)
    sync(drafter)
    middle = perf_counter()
    target_state = CachedTarget(target, prompt)
    sync(target)
    after_prefill = perf_counter()
    tokens, rounds = [], []
    old, context = [], {}
    while len(tokens) < limit and (not tokens or tokens[-1] not in eos):
        offset = len(tokens)
        remaining = limit - offset
        budget = min(DRAFT_LENGTH, remaining)
        tick = perf_counter()
        action, search_ms, block_ms = None, 0.0, 0.0
        if arm == "fresh" or not old:
            proposal, forwards = draft_state.propose(budget, eos)
        else:
            policy = CopyPolicy(arm, old, context, head)
            proposal, forwards, action, search_ms, block_ms = draft_state.propose_copy(
                budget, eos, policy, checked=arm == "checked_copy")
        sync(drafter)
        draft_ms = (perf_counter() - tick) * 1000
        tick = perf_counter()
        decision = target_state.verify(proposal, eos, remaining)
        sync(target)
        verify_ms = (perf_counter() - tick) * 1000
        tick = perf_counter()
        draft_state.commit(decision)
        cache_ms = (perf_counter() - tick) * 1000
        tokens.extend(decision["committed_tokens"])
        if any(state.cache.get_seq_length() + len(state.pending) != len(prompt) + len(tokens)
               for state in (draft_state, target_state)):
            raise AssertionError("Committed prefix and cached state differ in length")
        reject = decision["first_rejection"]
        old = before_eos(proposal[reject + 1:], eos) if reject is not None and arm != "fresh" else []
        context = dict(offset=len(tokens), prior_accepted=decision["accepted_draft"],
                       corrected=decision["correction"])
        if action is not None:
            action["accepted_by_target"] = max(0, min(action["kept"], decision["accepted_draft"]
                                                     - action["destination"]))
        rounds.append(dict(offset=offset, proposal=proposal, decision=decision, copy=action,
                           draft_forwards=forwards, draft_ms=draft_ms, verify_ms=verify_ms,
                           cache_ms=cache_ms, search_ms=search_ms, block_ms=block_ms))
    return dict(token_ids=tokens, rounds=rounds, wall_ms=(perf_counter() - started) * 1000,
                draft_prefill_ms=(middle - started) * 1000,
                verify_prefill_ms=(after_prefill - middle) * 1000)


def flatten(case, arm):
    run = case["arms"][arm]
    rounds = run["rounds"]
    copies = [r["copy"] for r in rounds if r["copy"] is not None]
    return dict(id=case["id"], dataset=case["dataset"], arm=arm, order=case["order"].index(arm),
                output_tokens=len(run["token_ids"]), wall_ms=run["wall_ms"],
                prefill_ms=run["draft_prefill_ms"] + run["verify_prefill_ms"],
                draft_ms=sum(r["draft_ms"] for r in rounds), verify_ms=sum(r["verify_ms"] for r in rounds),
                search_ms=sum(r["search_ms"] for r in rounds), block_ms=sum(r["block_ms"] for r in rounds),
                cache_ms=sum(r["cache_ms"] for r in rounds), verify_forwards=len(rounds),
                draft_forwards=sum(r["draft_forwards"] for r in rounds),
                accepted=sum(r["decision"]["accepted_draft"] for r in rounds),
                exact_output=run["first_output_diff"] is None,
                exact_proposals=run["first_proposal_diff_round"] is None,
                copy_choices=len(copies), copy_block_forwards=sum(a["block_forward"] for a in copies),
                copy_requested=sum(len(a["tail"]) for a in copies), copy_kept=sum(a["kept"] for a in copies),
                copy_target_accepted=sum(a["accepted_by_target"] for a in copies))


def summarize(cases):
    from statistics import median
    import numpy as np

    rows = [flatten(case, arm) for case in cases for arm in ARMS]
    groups = {}
    for dataset in ("all", *DATASETS):
        fresh = {r["id"]: r for r in rows if r["arm"] == "fresh"
                 and (dataset == "all" or r["dataset"] == dataset)}
        groups[dataset] = {}
        for arm in ARMS:
            selected = [r for r in rows if r["arm"] == arm and r["id"] in fresh]
            if not selected:
                continue
            exact = [r for r in selected if r["exact_output"]]
            n = len(selected)
            totals = {k: sum(r[k] for r in selected) for k in
                      ("wall_ms", "prefill_ms", "draft_ms", "verify_ms", "search_ms", "block_ms", "cache_ms",
                       "verify_forwards", "draft_forwards", "accepted", "output_tokens", "copy_choices",
                       "copy_block_forwards", "copy_requested", "copy_kept", "copy_target_accepted")}
            speed_delta = [r["wall_ms"] - fresh[r["id"]]["wall_ms"] for r in selected]
            counts = Counter("win" if d < 0 else "loss" if d > 0 else "tie" for d in speed_delta)
            accept_delta = [r["accepted"] - fresh[r["id"]]["accepted"] for r in selected]
            accept_counts = Counter("win" if d > 0 else "loss" if d < 0 else "tie" for d in accept_delta)
            verify_delta = [r["verify_forwards"] - fresh[r["id"]]["verify_forwards"] for r in selected]
            bootstrap = np.random.default_rng(EVAL_SEED).integers(0, n, size=(2000, n))
            ratios = (np.array([fresh[r["id"]]["wall_ms"] for r in selected])[bootstrap].sum(axis=1)
                      / np.array([r["wall_ms"] for r in selected])[bootstrap].sum(axis=1))
            groups[dataset][arm] = dict(
                prompts=n, exact_output=len(exact), exact_proposals=sum(r["exact_proposals"] for r in selected),
                totals=totals, mean_wall_ms=totals["wall_ms"] / n,
                median_wall_ms=median(r["wall_ms"] for r in selected),
                mean_paired_delta_wall_ms=sum(speed_delta) / n, paired_wall_wtl=dict(counts),
                mean_paired_delta_accepted=sum(accept_delta) / n, paired_accepted_wtl=dict(accept_counts),
                mean_paired_delta_verify_forwards=sum(verify_delta) / n,
                tokens_per_second=1000 * totals["output_tokens"] / totals["wall_ms"],
                accepted_per_verify=totals["accepted"] / totals["verify_forwards"],
                speedup_vs_fresh=sum(fresh[r["id"]]["wall_ms"] for r in selected) / totals["wall_ms"],
                speedup_vs_fresh_prompt_bootstrap_ci95=np.quantile(ratios, [.025, .975]).tolist(),
                speedup_vs_fresh_exact_output=(sum(fresh[r["id"]]["wall_ms"] for r in exact)
                                               / sum(r["wall_ms"] for r in exact)) if exact else None)
    return groups, rows


def compare_outputs(arms):
    from itertools import zip_longest

    baseline = arms["fresh"]
    for run in arms.values():
        run["first_output_diff"] = first_difference(baseline["token_ids"], run["token_ids"])
        run["first_proposal_diff_round"] = next(
            (i for i, (a, b) in enumerate(zip_longest(baseline["rounds"], run["rounds"]))
             if a is None or b is None or (a["offset"], a["proposal"]) != (b["offset"], b["proposal"])), None)


def export(store, output, manifest, expected, status):
    from suffix_lab.store import atomic_json

    cases = store.rows("case")
    metrics, rows = summarize(cases)
    atomic_json(output / "summary.json", dict(status=status, completed=len(cases), expected=expected,
                                              manifest=manifest, metrics=metrics))
    store.export("case", output / "cases.jsonl")
    snapshot = output / "checkpoint.snapshot.sqlite"
    store.backup(snapshot)
    files = ["summary.json", "cases.jsonl", "prompts.json", "preflight.json"]
    if rows:
        with (output / "paired.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files.append("paired.csv")
    temporary = output / (ARCHIVE + ".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as packaged:
        packaged.write(snapshot, "checkpoint.sqlite")
        for name in files:
            path = output / name
            if path.is_file():
                packaged.write(path, name)
    temporary.replace(output / ARCHIVE)
    snapshot.unlink()


def prepare_evaluation(tokenizer, lock, count, excluded, limit, context_limit):
    from datasets import load_dataset

    datasets = {"math500": ("HuggingFaceH4/MATH-500", "problem", {}),
                "gsm8k": ("openai/gsm8k", "question", {"name": "main"})}
    groups = {}
    for ds, (name, field, options) in datasets.items():
        data = load_dataset(name, split="test", revision=lock["datasets"][name], **options)
        banned = {int(item.split(":")[1]) for item in excluded if item.startswith(ds + ":")}
        banned_text = {data[i][field].strip() for i in banned}
        indices = list(range(len(data)))
        random.Random(EVAL_SEED).shuffle(indices)
        group = []
        for i in indices:
            question = data[i][field].strip()
            if question in banned_text or i in banned:
                continue
            ids = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                                tokenize=True, add_generation_prompt=True)
            if len(ids) + limit > context_limit:
                continue
            group.append(dict(id=f"{ds}:{i}", dataset=ds, index=i, question=question, prompt_tokens=ids))
            banned_text.add(question)
            if len(group) == count:
                break
        if len(group) != count:
            raise ValueError(f"Insufficient disjoint questions for {ds}")
        groups[ds] = group
    return [groups[ds][i] for i in range(count) for ds in DATASETS]


def preflight(target, drafter, prompt, eos, head, sync):
    warm = {arm: decode(target, drafter, prompt, eos, 12, arm, head, sync) for arm in ARMS}
    compare_outputs(warm)
    baseline = CopyDrafter(drafter, prompt)
    proposal, count = baseline.propose(16, set())
    if any(x in eos for x in proposal[:8]):
        return dict(warm=warm, forced_block="skipped_early_eos")
    checked = CopyDrafter(drafter, prompt)
    policy = CopyPolicy("checked_copy", proposal[:8], {}, head)
    copied, calls, action, _, _ = checked.propose_copy(16, set(), policy, checked=True)
    sync(drafter)
    if not action or not action["block_forward"]:
        raise AssertionError("Preflight did not exercise the checked COPY block")
    return dict(warm=warm, forced_block=dict(first_diff=first_difference(proposal, copied),
                                            action=action, fresh_forwards=count, copy_forwards=calls))


def restore(source, output):
    import shutil

    destination = output / "checkpoint.sqlite"
    if source.is_dir():
        source = source / "checkpoint.sqlite"
    if destination.exists():
        raise ValueError("Resume into a new output directory or omit --resume to reuse its own checkpoint")
    temporary = destination.with_suffix(".restore")
    try:
        if source.suffix.lower() == ".zip":
            with ZipFile(source) as archive, archive.open("checkpoint.sqlite") as saved, temporary.open("wb") as out:
                shutil.copyfileobj(saved, out)
        else:
            shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def run(args):
    from itertools import permutations
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from suffix_lab.assets import AR, ROOT, TARGET, source_digest
    from suffix_lab.runtime import configure_numerics, load_model, load_target, sync_model, tokenizers
    from suffix_lab.store import Store, atomic_json, digest

    if torch.cuda.device_count() < 2:
        raise RuntimeError("Select GPU T4 x2 on Kaggle")
    lock = json.loads((ROOT / "source-lock.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/long64.json").read_text(encoding="utf-8"))
    head_bytes = args.head.read_bytes()
    head = json.loads(gzip.decompress(head_bytes))
    if tuple(head["features"]) != FEATURES:
        raise ValueError("Feature version mismatch")
    configure_numerics(config)
    revisions = {name: lock["models"][name] for name in (TARGET, AR)}
    paths = {name: snapshot_download(name, revision=rev,
                                     allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
             for name, rev in revisions.items()}
    tokenizer, identity = tokenizers(paths)
    prompts = prepare_evaluation(tokenizer, lock, args.count, head["excluded_prompt_ids"],
                                 args.max_new_tokens, config["context_limit"])
    target, _, _ = load_target(paths[TARGET], config)
    drafter = load_model(paths[AR], "cuda:1")
    eos = set()
    for obj in (tokenizer, target.generation_config, target.config, drafter.generation_config, drafter.config):
        ids = obj.eos_token_id
        if ids is not None:
            eos.update(ids if isinstance(ids, (list, tuple)) else [ids])
    if not eos or drafter.config.vocab_size > target.config.vocab_size:
        raise ValueError("Invalid EOS or incompatible vocabularies")
    manifest = dict(protocol="ar-resync-e2e-v1", revision=args.revision, source_digest=source_digest(),
                    runner_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    head_sha256=sha256(head_bytes).hexdigest(), direct_threshold=head["threshold"],
                    head_training_trace=head["trace_sha256"], models=revisions, datasets=lock["datasets"],
                    prompt_digest=digest(prompts), prompts_per_dataset=args.count, seed=EVAL_SEED,
                    draft_length=DRAFT_LENGTH, max_new_tokens=args.max_new_tokens, max_probe=MAX_PROBE,
                    checked_length=CHECKED_LENGTH, checked_k=CHECKED_LENGTH, eos=sorted(eos),
                    precision="fp16", temperature=0, tokenizer=identity["shared_vocab_sha256"],
                    target_device_map={k: str(v) for k, v in target.hf_device_map.items()},
                    drafter_device=str(drafter.device), torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.version.cuda, gpus=[torch.cuda.get_device_name(i) for i in range(2)])
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        restore(args.resume, output)
    store = Store(output / "checkpoint.sqlite", manifest)
    status = "partial"
    try:
        atomic_json(output / "prompts.json", prompts)
        with torch.inference_mode():
            warm = tokenizer.apply_chat_template([{"role": "user", "content": "Count from 1 to 20."}],
                                                  tokenize=True, add_generation_prompt=True)
            atomic_json(output / "preflight.json", preflight(target, drafter, warm, eos, head, sync_model))
            started = perf_counter()
            since_last_export = started
            orders = list(permutations(ARMS))
            for i, item in enumerate(prompts):
                if store.get("case", item["id"]) is not None:
                    continue
                if perf_counter() - started >= args.budget_seconds:
                    status = "budget_checkpoint"
                    break
                # Cycle through all six orderings equally, separately within each dataset.
                order = orders[(i // len(DATASETS)) % len(orders)]
                arms = {arm: decode(target, drafter, item["prompt_tokens"], eos, args.max_new_tokens,
                                    arm, head, sync_model) for arm in order}
                compare_outputs(arms)
                store.put("case", item["id"], dict(id=item["id"], dataset=item["dataset"],
                                                  order=order, arms=arms))
                print(f"{i + 1}/{len(prompts)} {item['id']} " + " | ".join(
                    f"{arm}: {arms[arm]['wall_ms'] / 1000:.2f}s, {len(arms[arm]['rounds'])} verifies, "
                    f"out-diff={arms[arm]['first_output_diff']}" for arm in ARMS), flush=True)
                if perf_counter() - since_last_export >= 180:
                    export(store, output, manifest, len(prompts), "partial")
                    since_last_export = perf_counter()
            else:
                status = "complete"
    finally:
        try:
            export(store, output, manifest, len(prompts), status)
        finally:
            store.close()
    print(status, output / ARCHIVE, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True, help="Head produced by train_ar_resync_gain.py")
    parser.add_argument("--revision", default="local")
    parser.add_argument("--count", type=int, default=100, help="Questions per dataset")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--budget-seconds", type=int, default=28800)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not 1 <= args.count <= 100 or args.max_new_tokens < 1 or args.budget_seconds < 1:
        parser.error("Use 1–100 questions per dataset and positive limits")
    run(args)
