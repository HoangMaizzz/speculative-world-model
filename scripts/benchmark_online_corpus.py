"""Live paired comparison of a frozen corpus and per-arm streaming corpus updates."""
import argparse
from collections import Counter, deque
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from random import Random
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

ARMS = ("static100", "online_empty", "online_seed16")
ARCHIVE = "online_corpus_results.zip"
DATA = "retrieval_alignment_data.json"
DATA_SHA256 = "5c8835603b8242166c1549d72fd5850c181feb9e391abe3bee189147150a16b4"


def load_data(path):
    path = Path(path)
    candidates = [path] if path.is_file() else sorted(path.rglob(DATA)) + sorted(
        path.rglob("retrieval_alignment_kaggle_input.zip"))
    for candidate in candidates:
        if candidate.suffix == ".zip":
            with ZipFile(candidate) as z:
                if DATA not in z.namelist():
                    continue
                raw = z.read(DATA)
        else:
            raw = candidate.read_bytes()
        if sha256(raw).hexdigest() == DATA_SHA256:
            return json.loads(raw)
    raise ValueError(f"Mount datatest with the pinned {DATA} or retrieval_alignment_kaggle_input.zip under {path}")


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


def make_banks(index_factory, corpus):
    banks = {arm: index_factory() for arm in ARMS}
    seen = Counter()
    for record in corpus:
        banks["static100"].append([], record["token_ids"])
        domain = record["id"].split(":")[0]
        if seen[domain] < 8:
            banks["online_seed16"].append([], record["token_ids"])
            seen[domain] += 1
    if seen != {"math500": 8, "gsm8k": 8}:
        raise ValueError("Seed corpus must supply eight examples from each dataset")
    for index in banks.values():
        for key in index.table:
            index.hits(key)
    return banks


def check_row(record, signature, position, prompt):
    if record["signature"] != signature or (record["position"], record["prompt"]) != (position, prompt["id"]):
        raise ValueError(f"Checkpoint belongs to another run or order at position {position}")
    if set(record["runs"]) != set(ARMS):
        raise ValueError(f"Incomplete arms in checkpoint at position {position}")
    return record


def restore(path, out, manifest):
    """Read only the numbered row files specified by the bundle."""
    if not path:
        return
    with ZipFile(path) as z:
        if json.loads(z.read("manifest.json")) != manifest:
            raise ValueError("Resume ZIP does not match the revision, prompts, hardware, or settings")
        summary = json.loads(z.read("summary.json"))
        total = summary["completed"]
        if not 0 <= total <= len(manifest["prompts"]):
            raise ValueError("Invalid completed count in resume ZIP")
        signature = sha256(canonical(manifest)).hexdigest()
        records = [
            check_row(json.loads(z.read(f"rows/{i:03d}.json")), signature, i, manifest["prompts"][i])
            for i in range(total)
        ]
    for record in records:
        target = out / "rows" / f'{record["position"]:03d}.json'
        for result in record["runs"].values():
            result.pop("coverage", None)
        if target.exists():
            if json.loads(target.read_bytes()) != record:
                raise ValueError(f"Resume would overwrite a different local row: {target}")
        else:
            atomic(target, record)


def execute(out, manifest, banks, infer, checkpoint):
    signature = sha256(canonical(manifest)).hexdigest()
    row_dir = out / "rows"
    saved = sorted(path.name for path in row_dir.glob("*.json"))
    if saved != [f"{i:03d}.json" for i in range(len(saved))] or len(saved) > len(manifest["prompts"]):
        raise ValueError("Checkpoint rows must be a contiguous prefix of the prompt plan")
    rows = []
    for i, prompt in enumerate(manifest["prompts"]):
        path = row_dir / f"{i:03d}.json"
        if path.is_file():
            row = check_row(json.loads(path.read_bytes()), signature, i, prompt)
            for arm in ARMS[1:]:
                banks[arm].append([], row["runs"][arm]["tokens"])
        else:
            pid = prompt["id"]
            order = list(ARMS)
            Random("online-corpus-v1/" + pid).shuffle(order)
            row = dict(position=i, prompt=pid, dataset=prompt["dataset"], signature=signature, order=order, runs={})
            for arm in order:
                bank = banks[arm]
                result = infer(prompt["prompt_tokens"], bank)
                tokens, counts = result["tokens"], result["counts"]
                if not 0 < len(tokens) <= manifest["max_new"] or counts["rounds"] < 1:
                    raise ValueError(f"Incomplete decode for {pid} {arm}")
                counts["decode_ms"] = counts["wall_ms"]
                start = perf_counter()
                if arm != "static100":
                    bank.append([], tokens)
                counts["index_update_ms"] = 1000 * (perf_counter() - start) if arm != "static100" else 0.0
                counts["wall_ms"] += counts["index_update_ms"]
                row["runs"][arm] = result
                print(f"{i+1}/{len(manifest['prompts'])} {pid} {arm}: "
                      f"{counts['wall_ms']/1000:.2f}s, Q={counts['draft_forwards']}, "
                      f"P={counts['rounds']}", flush=True)
            atomic(path, row)
        rows.append(row)
        if len(rows) == 1 or len(rows) % 10 == 0 or len(rows) == len(manifest["prompts"]):
            checkpoint(rows)
    return rows


def vocabulary(index):
    return set(token for ctx, counts in index.table.items() if len(ctx) == 1 for token in counts)


def audit_trace(index_factory, tree, prompt, steps, bank, bank_vocab, eos, limit, draft_length, width):
    """Rebuild the packed Q paths outside timed inference."""
    prefix = tuple(prompt)
    history = index_factory()
    history.append([], prefix)
    history_vocab = vocabulary(history)
    suffixes = deque(maxlen=4)
    stats = Counter()
    for step in steps:
        draft, emitted = tuple(step["draft"]), tuple(step["emitted"])
        sources = [(history, 1.0), (bank, .75)]
        sources.extend((index, 1.5 * .75**age) for age, (index, _) in enumerate(reversed(suffixes)))
        suffix_vocab = set().union(*(tokens for _, tokens in suffixes))
        horizon = min(draft_length, limit - step["offset"])
        used = set()
        start = 0
        while start < len(draft):
            paths = tree((prefix + draft[:start])[-4:], sources, width, horizon - start - 1)
            end = start
            while end < len(draft) and draft[start:end+1] in paths:
                end += 1
            used.update(range(start, end))
            stats.update(draft_forwards=1, tree_nodes=len(paths))
            start = end + 1
        accepted = step["accepted"]
        for i, token in enumerate(emitted):
            if token in eos:
                break
            ctx = (prefix + emitted[:i])[-4:]
            stats.update(output_positions=1, dictionary_covered=token in bank_vocab or token in history_vocab
                         or token in suffix_vocab, shortlisted=any(token in dict(ix.hits(ctx)[1]) for ix, _ in sources),
                         used_by_tree=i < accepted and i in used)
        if accepted < min(len(draft), len(emitted)):
            tail = draft[accepted+1:]
            tail = tail[:next((i for i, token in enumerate(tail) if token in eos), len(tail))]
            if tail:
                index = index_factory()
                index.append((prefix + draft[:accepted+1])[-4:], tail)
                suffixes.append((index, set(tail)))
        history.append(prefix[-4:], emitted)
        history_vocab.update(emitted)
        prefix += emitted
    return stats


def add_coverage(rows, manifest, corpus, index_factory, tree):
    banks = make_banks(index_factory, corpus)
    vocabs = {arm: vocabulary(bank) for arm, bank in banks.items()}
    eos = set(manifest["eos"])
    for row, prompt in zip(rows, manifest["prompts"], strict=True):
        for arm in ARMS:
            result = row["runs"][arm]
            coverage = audit_trace(index_factory, tree, prompt["prompt_tokens"], result["rounds"],
                                   banks[arm], vocabs[arm], eos, manifest["max_new"], manifest["draft_length"],
                                   manifest["tree_width"])
            for key in ("draft_forwards", "tree_nodes"):
                if coverage[key] != result["counts"][key]:
                    raise ValueError(f"Tree replay failed for {row['prompt']} {arm}: {key}")
            result["coverage"] = {key: coverage[key] for key in
                                  ("output_positions", "dictionary_covered", "shortlisted", "used_by_tree")}
            if arm != "static100":
                banks[arm].append([], result["tokens"])
                vocabs[arm].update(result["tokens"][1:])


def describe(rows, paired):
    groups = {}
    for arm in ARMS:
        counts, coverage = Counter(), Counter()
        tokens, exact = 0, 0
        for row in rows:
            value = row["runs"][arm]
            counts.update(value["counts"])
            coverage.update(value.get("coverage", {}))
            tokens += len(value["tokens"])
            exact += value["tokens"] == row["runs"]["static100"]["tokens"]
        report = dict(counts=counts, tokens=tokens, tokens_per_second=1000 * tokens / counts["wall_ms"],
                      accepted_per_verify=counts["accepted"] / counts["rounds"],
                      token_id_equal_to_static=exact,
                      q_total_forwards=counts["draft_forwards"] + counts["q_fill_calls"])
        if coverage["output_positions"]:
            report["coverage"] = dict(coverage, dictionary_fraction=coverage["dictionary_covered"]/coverage["output_positions"],
                                      shortlisted_fraction=coverage["shortlisted"]/coverage["output_positions"],
                                      used_by_tree_fraction=coverage["used_by_tree"]/coverage["output_positions"])
        if arm != "static100":
            same = [r for r in rows if r["runs"][arm]["tokens"] == r["runs"]["static100"]["tokens"]]
            report["speedup_vs_static_same_output"] = paired(same, "static100", arm)
        groups[arm] = report
    return groups


def write_bundle(out, manifest, rows, paired):
    size = len(manifest["prompts"])
    groups = {"all": rows}
    for domain in ("math500", "gsm8k"):
        groups[domain] = [row for row in rows if row["dataset"] == domain]
    for quarter in range(4):
        groups[f"quarter_{quarter+1}"] = [
            row for row in rows if quarter * size <= 4 * row["position"] < (quarter+1) * size
        ]
    summary = dict(completed=len(rows), planned=size, complete=len(rows) == size,
                   coverage_complete=all("coverage" in r["runs"]["static100"] for r in rows),
                   groups={name: describe(group, paired) for name, group in groups.items() if group})
    buffer = StringIO()
    writer = csv.writer(buffer)
    metrics = ("wall_ms", "decode_ms", "index_update_ms", "draft_ms", "verify_ms", "draft_forwards",
               "q_fill_calls", "rounds", "accepted", "tree_nodes")
    writer.writerow(["position", "prompt", "dataset", "arm", "tokens", "equal_to_static", *metrics])
    for row in rows:
        for arm in row["order"]:
            result = row["runs"][arm]
            writer.writerow([row["position"], row["prompt"], row["dataset"], arm, len(result["tokens"]),
                             result["tokens"] == row["runs"]["static100"]["tokens"],
                             *(result["counts"].get(metric, 0) for metric in metrics)])
    tmp = out / "online_corpus.tmp"
    with ZipFile(tmp, "w", ZIP_DEFLATED, compresslevel=3) as z:
        z.writestr("manifest.json", canonical(manifest))
        z.writestr("summary.json", json.dumps(summary, indent=2))
        z.writestr("per_prompt.csv", buffer.getvalue())
        for row in rows:
            z.writestr(f'rows/{row["position"]:03d}.json', canonical(row))
    tmp.replace(out / ARCHIVE)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--per-domain", type=int, choices=range(1, 101), default=100)
    ap.add_argument("--max-new", type=int, choices=range(1, 257), default=256)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    data = load_data(args.input)

    import torch
    from huggingface_hub import snapshot_download
    from threadpoolctl import threadpool_limits
    from benchmark_retrieval_ouro import plan
    from consensus_tree_core import DRAFT, WIDTH, run
    from retrieval_alignment_core import Index, tree
    from retrieval_ouro_report import paired
    from suffix_lab.assets import ROOT, TARGET, AR
    from suffix_lab.runtime import load_model, load_target, tokenizers

    if torch.cuda.device_count() != 2 or not all("T4" in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("Select Kaggle GPU T4 x2")
    lock = json.loads((ROOT / "source-lock.json").read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    paths = {name: snapshot_download(name, revision=lock["models"][name],
                                     allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
             for name in (AR, TARGET)}
    tokenizer, identity = tokenizers(paths)
    excluded = {record["id"] for record in data["corpus"] + data["prompts"]}
    prompts = plan(tokenizer, lock["datasets"], args.per_domain, excluded)
    q = load_model(paths[AR], "cuda:1")
    p, _, _ = load_target(paths[TARGET], dict(target_precision="fp16", target_gpu0_layers=18,
                                            exclude_inter_gpu_transfer_time=False))
    eos = set(data["eos"])
    manifest = dict(revision=args.revision, prompts=prompts, max_new=args.max_new, arms=ARMS,
                    eos=sorted(eos), draft_length=DRAFT, tree_width=WIDTH, corpus_sha256=DATA_SHA256,
                    initial_docs=dict(static100=100, online_empty=0, online_seed16=16),
                    models={name: lock["models"][name] for name in paths}, datasets=lock["datasets"],
                    shared_vocab=identity["shared_vocab_sha256"], placement=p.hf_device_map,
                    dtype="float16", devices=[torch.cuda.get_device_name(i) for i in range(2)],
                    torch=str(torch.__version__), cuda=torch.version.cuda)
    # Normalize tuples and integer dictionary keys through the same JSON representation used on disk.
    manifest = json.loads(canonical(manifest))
    out = args.output
    (out / "rows").mkdir(parents=True, exist_ok=True)
    meta = out / "manifest.json"
    if meta.exists() and json.loads(meta.read_bytes()) != manifest:
        raise ValueError("Existing checkpoint has a different configuration")
    restore(args.resume, out, manifest)
    atomic(meta, manifest)
    banks = make_banks(Index, data["corpus"])

    def infer(ids, bank):
        return run(q, p, ids, bank, eos, args.max_new, "base")

    def checkpoint(rows):
        result = write_bundle(out, manifest, rows, paired)
        print(f'Checkpoint {result["completed"]}/{result["planned"]}: {out / ARCHIVE}', flush=True)

    with torch.inference_mode(), threadpool_limits(limits=2):
        warmup = tokenizer.apply_chat_template([{"role": "user", "content": "What is 5 + 7?"}],
                                              tokenize=True, add_generation_prompt=True)
        run(q, p, warmup, Index(), eos, 16, "base")
        rows = execute(out, manifest, banks, infer, checkpoint)
    print("Auditing tree coverage outside inference timing", flush=True)
    add_coverage(rows, manifest, data["corpus"], Index, tree)
    checkpoint(rows)
    print(out / ARCHIVE, flush=True)


if __name__ == "__main__":
    main()
