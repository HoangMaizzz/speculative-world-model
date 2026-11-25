"""Four-way paired live benchmark for retrieval drafting and aligned recovery."""
import argparse
from collections import Counter, deque
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from random import Random
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, DynamicCache
from threadpoolctl import threadpool_limits

from retrieval_alignment_core import CachedTarget, Index, LiveTarget, decode, generate, ids_tensor, longest_prefix, select
from suffix_lab.long60_plan import until_eos
from suffix_lab.runtime import load_target


ARMS = ("A", "B", "C", "D")
HASHES = {
    "retrieval_alignment_data.json": "5c8835603b8242166c1549d72fd5850c181feb9e391abe3bee189147150a16b4",
    "selectors.joblib": "56a67f0404aeb3303b6c501502fab96415c3fa40e6a7298f1dca24cfaf14b452",
}


def inputs(location):
    root = Path(location)
    if root.is_file():
        locations = [root]
    else:
        folders = [p.parent for p in root.rglob("retrieval_alignment_data.json")]
        locations = folders + list(root.rglob("retrieval_alignment_kaggle_input.zip"))
    for candidate in locations:
        if candidate.is_dir():
            if not all((candidate / name).is_file() for name in HASHES):
                continue
            files = {name: (candidate / name).read_bytes() for name in HASHES}
        else:
            with ZipFile(candidate) as z:
                files = {name: z.read(name) for name in HASHES}
        if all(sha256(files[name]).hexdigest() == digest for name, digest in HASHES.items()):
            return json.loads(files["retrieval_alignment_data.json"]), joblib.load(BytesIO(files["selectors.joblib"]))["models"]["Q+raw+alignedP"]
    raise ValueError(f"Attach retrieval_alignment_kaggle_input.zip or its extracted contents under {root}")


def clock():
    for gpu in range(torch.cuda.device_count()):
        torch.cuda.synchronize(gpu)
    return perf_counter()


def rollout(q, p, prompt, eos, limit, arm, corpus, selector):
    retrieve = arm != "A"
    aligned = arm in ("C", "D")
    start = clock()
    prefix = list(prompt)
    qkv = q.model(ids_tensor(q, prefix[:-1]), use_cache=True).past_key_values
    v = LiveTarget(p, prefix) if aligned else CachedTarget(p, prefix)
    history, bank = Index(), deque(maxlen=4)
    if retrieve:
        history.append([], prompt)
    old = None
    steps, tokens, counts = [], [], Counter()
    tick = clock()
    counts["prefill_ms"] = 1000 * (tick - start)
    while len(tokens) < limit and (not tokens or tokens[-1] not in eos):
        remaining = limit - len(tokens)
        missing = prefix[qkv.get_seq_length():]
        assert missing
        if len(missing) > 1:
            q.model(ids_tensor(q, missing[:-1]), past_key_values=qkv, use_cache=True)
            counts["q_fill_calls"] += 1
        sources = [(history, 1.), (corpus, .75)] if retrieve else []
        sources.extend((s, 1.5 * .75 ** age) for age, s in enumerate(reversed(bank)))
        ready = aligned and old is not None
        draft, rows = decode(q, qkv, prefix[-4:], min(remaining, 16), eos, sources,
                             16 if retrieve else 0, counts, collect=ready)
        assert qkv.get_seq_length() == len(prefix) + len(draft) - 1
        drafted = clock()
        counts["draft_ms"] += 1000 * (drafted - tick)
        choice = select(rows, selector, prefix, draft, old, len(tokens), eos, True, False) if ready else None
        ranked = clock()
        counts["rank_ms"] += 1000 * (ranked - drafted)
        pos, branch, fork = 0, [], None
        if choice is not None:
            pos, root, _ = choice
            branch = [root]
            if arm == "D" and remaining > pos + 1:
                fork = DynamicCache.from_legacy_cache(qkv.to_legacy_cache())
                fork.crop(len(prefix) + pos)
                tail, _ = generate(q, fork, root, min(3, remaining-pos-1), eos)
                branch.extend(tail)
        forked = clock()
        counts["regrow_ms"] += 1000 * (forked - ranked)
        if aligned:
            emitted, accepted, matched, gain, old = v.check(prefix, draft, branch, pos, eos, remaining, True)
        else:
            decision = v.verify(draft, eos, remaining)
            emitted = decision["committed_tokens"]
            accepted = min(decision["accepted_draft"], len(emitted))
            matched, gain = 0, 0
        verified = clock()
        counts["verify_ms"] += 1000 * (verified - forked)
        if matched:
            if fork is None:
                qkv.crop(len(prefix) + pos)
            else:
                qkv = fork
        qkv.crop(min(qkv.get_seq_length(), len(prefix) + accepted))
        reject = longest_prefix(draft, emitted)
        if retrieve and reject < min(len(draft), len(emitted)):
            suffix = until_eos(draft[reject+1:], eos, include=False)
            if suffix:
                index = Index()
                index.append(prefix[-4:] + draft[:reject+1], suffix)
                bank.append(index)
        if retrieve:
            history.append(prefix[-4:], emitted)
        steps.append(dict(offset=len(tokens), draft=draft, choice=choice, branch=branch,
                          accepted=accepted, matched=matched, extra=gain, emitted=emitted))
        prefix.extend(emitted)
        tokens.extend(emitted)
        assert v.cache.get_seq_length() + len(v.pending) == len(prefix)
        counts.update(rounds=1, accepted=accepted, eligible=int(ready), branches=int(bool(branch)),
                      branch_hits=int(matched > 0), branch_nodes=len(branch), extra=gain,
                      regrow_calls=max(0, len(branch)-1))
        tick = clock()
        counts["commit_ms"] += 1000 * (tick - verified)
    counts["wall_ms"] = 1000 * (clock() - start)
    return dict(tokens=tokens, counts=dict(counts), rounds=steps)


def summarize(rows):
    stats = {}
    for name in ARMS:
        cnt = Counter()
        for row in rows:
            cnt.update(row["runs"][name]["counts"])
        tokens = sum(len(row["runs"][name]["tokens"]) for row in rows)
        stats[name] = dict(counts=dict(cnt), tokens=tokens, tokens_per_s=tokens*1000/cnt["wall_ms"],
                           committed_per_round=tokens/cnt["rounds"], exact_vs_A=sum(
                               r["runs"][name]["tokens"] == r["runs"]["A"]["tokens"] for r in rows))
    pairs = {}
    for base, method in (("A","B"), ("B","C"), ("B","D"), ("C","D")):
        key = f"{method}_vs_{base}"
        valid = [r for r in rows if r["runs"][method]["tokens"] == r["runs"][base]["tokens"]]
        pairs[key] = dict(exact=len(valid), total=len(rows))
        if not valid:
            continue
        def times(r):
            return [r["runs"][arm]["counts"]["wall_ms"] for arm in (base, method)]
        times_ms = np.array([times(r) for r in valid])
        rng = np.random.default_rng(925)
        draws = np.zeros((2000, 2))
        for dataset in sorted({r["dataset"] for r in valid}):
            group = np.array([times(r) for r in valid if r["dataset"] == dataset])
            draws += group[rng.integers(len(group), size=(2000,len(group)))].sum(axis=1)
        ratio = times_ms.sum(axis=0)
        pairs[key].update(speedup=ratio[0]/ratio[1], faster_prompts=int((times_ms[:,0]>times_ms[:,1]).sum()),
                          paired_prompt_ci95=np.quantile(draws[:,0]/draws[:,1],[.025,.975]).tolist())
    return dict(arms=stats, comparisons=pairs, exact_A_B_round_drafts=sum(
        [s["draft"] for s in r["runs"]["A"]["rounds"]] == [s["draft"] for s in r["runs"]["B"]["rounds"]]
        for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, choices=range(1,26), default=25)
    parser.add_argument("--max-new", type=int, choices=range(1,257), default=256)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    data, selector = inputs(args.input)
    assert len(data["corpus"]) == 100 and len(data["prompts"]) == 50
    assert not ({r["id"] for r in data["corpus"]} & {r["id"] for r in data["prompts"]})
    corpus = Index()
    for row in data["corpus"]:
        corpus.append([], row["token_ids"])
    # Prime the immutable corpus cache outside measured arms for every order.
    for context in tuple(corpus.table):
        corpus.hits(context)
    per_domain = []
    for domain in ("math500", "gsm8k"):
        pool = sorted((r for r in data["prompts"] if r["dataset"] == domain), key=lambda r:r["id"])
        per_domain.append(Random("abcd/"+domain).sample(pool, args.per_domain))
    planned = [r for pair in zip(*per_domain, strict=True) for r in pair]
    precision = dict(torch_dtype=torch.float16, attn_implementation="sdpa")
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Select Kaggle GPU T4 x2")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    paths = {key:snapshot_download(key, revision=val, allow_patterns=["config.json","generation_config.json","*.safetensors","*.safetensors.index.json"])
             for key,val in data["models"].items() if key in ("Qwen/Qwen2.5-1.5B-Instruct","Qwen/Qwen2.5-7B-Instruct")}
    q = AutoModelForCausalLM.from_pretrained(paths["Qwen/Qwen2.5-1.5B-Instruct"], device_map={"":"cuda:1"}, **precision).eval()
    p, _, _ = load_target(paths["Qwen/Qwen2.5-7B-Instruct"], {
        "target_precision": "fp16", "target_gpu0_layers": 18, "exclude_inter_gpu_transfer_time": False,
    })
    result = dict(revision=args.revision, models=data["models"], placement=p.hf_device_map,
                  devices=[torch.cuda.get_device_name(i) for i in range(2)], torch=str(torch.__version__),
                  threshold=selector["threshold"], max_new=args.max_new, planned=[r["id"] for r in planned], rows=[])
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out/"results.json"
    if checkpoint.exists():
        previous = json.loads(checkpoint.read_bytes())
        if any(previous[k] != v for k,v in result.items() if k not in ("rows",)):
            raise ValueError("Existing checkpoint belongs to a different configuration")
        result = previous
    def save():
        tmp = out/"results.tmp"
        tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
        tmp.replace(checkpoint)
        with ZipFile(out/"bundle.tmp", "w", ZIP_DEFLATED) as bundle:
            bundle.write(checkpoint, "results.json")
        (out/"bundle.tmp").replace(out/"retrieval_alignment_results.zip")
    eos = set(data["eos"])
    with torch.inference_mode(), threadpool_limits(limits=2):
        for arm in ARMS:
            rollout(q, p, planned[0]["prompt_tokens"], eos, 32, arm, corpus, selector)
        for i, row in enumerate(planned):
            if i < len(result["rows"]):
                assert result["rows"][i]["prompt"] == row["id"]
                continue
            order = list(ARMS)
            Random("abcd/order/"+row["id"]).shuffle(order)
            runs = {}
            for arm in order:
                runs[arm] = rollout(q,p,row["prompt_tokens"],eos,args.max_new,arm,corpus,selector)
                print(f"{i+1}/{len(planned)} {row['id']} {arm} {runs[arm]['counts']['wall_ms']/1000:.2f}s",flush=True)
            result["rows"].append(dict(prompt=row["id"],dataset=row["dataset"],order=order,runs=runs))
            result["summary"] = summarize(result["rows"])
            save()
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
