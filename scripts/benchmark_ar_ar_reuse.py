"""Trace rejected Qwen2.5-1.5B drafts verified by Qwen2.5-7B, then measure exact reuse."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
from statistics import median
import subprocess
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

from benchmark_dllm_ar32 import CachedTarget, decode_ar, first_difference


DATASETS = ("math500", "gsm8k")
HORIZONS = (10, 32)
MINIMUM = 4
RADII = (1, 4, 8, 16)
ARCHIVE = "ar_ar_reuse_results.zip"


class CachedDrafter(CachedTarget):
    """Reuse committed KV, deferring the extra token and any unconsumed draft token."""

    def propose(self, length, eos):
        self.prefix_length = self.cache.get_seq_length() + len(self.pending)
        forwards = int(bool(self.pending))
        token = self.forward(self.pending)[-1] if self.pending else self.next_token
        proposal = [token]
        while len(proposal) < length and proposal[-1] not in eos:
            proposal.append(self.forward(proposal[-1:])[-1])
            forwards += 1
        return proposal, forwards

    def commit(self, decision):
        committed = decision["committed_tokens"]
        accepted_end = self.prefix_length + min(decision["accepted_draft"], len(committed))
        keep = min(self.cache.get_seq_length(), accepted_end)
        if keep < self.prefix_length:
            raise AssertionError("Drafter cache fell behind committed prefix")
        self.cache.crop(keep)
        self.pending = committed[keep - self.prefix_length:]
        if self.cache.get_seq_length() + len(self.pending) != self.prefix_length + len(committed):
            raise AssertionError("Drafter cache does not represent the committed prefix")


def decode_spec(target, drafter, prompt, eos, limit, draft_length, sync):
    sync(target)
    sync(drafter)
    start = perf_counter()
    draft_state = CachedDrafter(drafter, prompt)
    sync(drafter)
    split = perf_counter()
    target_state = CachedTarget(target, prompt)
    sync(target)
    draft_prefill_ms = 1000 * (split - start)
    verify_prefill_ms = 1000 * (perf_counter() - split)
    tokens, rounds = [], []
    while len(tokens) < limit and (not tokens or tokens[-1] not in eos):
        remaining = limit - len(tokens)
        tick = perf_counter()
        proposal, forwards = draft_state.propose(min(draft_length, remaining), eos)
        sync(drafter)
        draft_ms = 1000 * (perf_counter() - tick)
        tick = perf_counter()
        decision = target_state.verify(proposal, eos, remaining)
        sync(target)
        verify_ms = 1000 * (perf_counter() - tick)
        tick = perf_counter()
        draft_state.commit(decision)
        draft_cache_ms = 1000 * (perf_counter() - tick)
        offset = len(tokens)
        tokens.extend(decision["committed_tokens"])
        if draft_state.cache.get_seq_length() + len(draft_state.pending) != len(prompt) + len(tokens):
            raise AssertionError("Drafter cache/token boundary changed")
        if target_state.cache.get_seq_length() + len(target_state.pending) != len(prompt) + len(tokens):
            raise AssertionError("Verifier cache/token boundary changed")
        rounds.append(dict(offset=offset, proposal=proposal, decision=decision, draft_forwards=forwards,
                           draft_ms=draft_ms, verify_ms=verify_ms, draft_cache_ms=draft_cache_ms))
    return dict(token_ids=tokens, rounds=rounds, draft_prefill_ms=draft_prefill_ms,
                verify_prefill_ms=verify_prefill_ms, wall_ms=1000 * (perf_counter() - start))


def positions(runs):
    return {j for run in runs for j in range(run["destination_start"], run["destination_end"])}


def diagnose(spec, gold, eos, horizon):
    """Run after decoding. Source and destination start after the verifier correction."""
    from suffix_lab.long60_plan import matching_runs, until_eos

    divergence = first_difference(spec["token_ids"], gold)
    shared_end = len(gold) if divergence is None else divergence
    events, attrition = [], Counter()
    rounds = spec["rounds"]
    for index, row in enumerate(rounds):
        decision = row["decision"]
        rejection = decision["first_rejection"]
        if rejection is None:
            attrition["no_rejection"] += 1
            continue
        start = row["offset"] + len(decision["committed_tokens"])
        old = until_eos(row["proposal"][rejection + 1:], eos, include=False)
        raw_gold = gold[start:start + horizon]
        future = until_eos(raw_gold, eos, include=False)
        if decision["terminated"]:
            attrition["eos_correction"] += 1
        elif start > shared_end:
            attrition["after_ar_divergence"] += 1
        elif len(old) < MINIMUM:
            attrition["short_rejected_suffix"] += 1
        elif len(future) < MINIMUM:
            attrition["short_reference"] += 1
        else:
            following = rounds[index + 1] if index + 1 < len(rounds) else None
            next_proposal = []
            next_accepted = next_committed = 0
            if following:
                if following["offset"] != start:
                    raise AssertionError("Next draft is not on the corrected prefix")
                next_proposal = following["proposal"]
                next_committed = min(shared_end - start, len(following["decision"]["committed_tokens"]))
                next_accepted = min(following["decision"]["accepted_draft"], next_committed)
            runs = matching_runs(old, future, max(len(old), len(future)) - 1,
                                 minimum=MINIMUM, include_zero=True)
            for run in runs:
                left, right = run["destination_start"], run["destination_end"]
                run.update(nonconstant=len(set(future[left:right])) > 1,
                           next_reaches=right <= len(next_proposal),
                           next_reproposes=next_proposal[left:right] == future[left:right],
                           next_accepts=right <= next_accepted,
                           next_commits=right <= next_committed)
            events.append(dict(round=index, corrected_offset=start, old=old, reference=future, runs=runs,
                               reference_right_censored=len(raw_gold) < horizon and len(raw_gold) == len(future)))
    return dict(events=events, attrition=dict(attrition))


def reuse_summary(rows, horizon):
    counts = Counter()
    lengths, displacements = Counter(), Counter()
    destinations = {name: set() for name in ("reference", "same", "shift")}
    for row in rows:
        source = row["reuse"][str(horizon)]
        for key, value in source["attrition"].items():
            counts[key] += value
        counts["prompts_with_shift"] += any(
            r["delta"] for e in source["events"] for r in e["runs"])
        for event in source["events"]:
            runs = event["runs"]
            same = [r for r in runs if r["delta"] == 0]
            shift = [r for r in runs if r["delta"] != 0]
            a, b = positions(same), positions(shift)
            counts.update(eligible_rounds=1, reference_tokens=len(event["reference"]),
                          rejected_tokens=len(event["old"]), reference_right_censored=event["reference_right_censored"],
                          same_rounds=bool(same), shift_rounds=bool(shift),
                          shift_only_rounds=bool(shift) and not same, either_rounds=bool(runs),
                          same_positions=len(a), shift_positions=len(b), union_positions=len(a | b),
                          shift_extra_positions=len(b - a))
            for name, values in (("reference", range(len(event["reference"]))), ("same", a), ("shift", b)):
                destinations[name].update((row["id"], event["corrected_offset"] + p) for p in values)
            if shift:
                lengths[max(r["length"] for r in shift)] += 1
                displacements.update(r["delta"] for r in shift)
                remaining = [r for r in shift if r["nonconstant"] and not r["next_commits"]]
                counts.update(next_reaches=any(r["next_reaches"] for r in shift),
                              next_reproposes=any(r["next_reproposes"] for r in shift),
                              next_accepts=any(r["next_accepts"] for r in shift),
                              next_commits=any(r["next_commits"] for r in shift),
                              nonconstant_shift=any(r["nonconstant"] for r in shift),
                              unaccepted_nonconstant_at_start=any(r["destination_start"] == 0 for r in remaining),
                              unaccepted_nonconstant_at_first_4=any(r["destination_start"] <= 3 for r in remaining))
                for radius in RADII:
                    counts[f"shift_within_{radius}"] += any(abs(r["delta"]) <= radius for r in shift)
    unique = {key: len(value) for key, value in destinations.items()}
    unique["union"] = len(destinations["same"] | destinations["shift"])
    unique["shift_extra"] = len(destinations["shift"] - destinations["same"])
    denominator = counts["reference_tokens"]
    return dict(counts=dict(counts), distinct_prompt_positions=unique,
                longest_shift_histogram=dict(sorted(lengths.items())),
                shift_span_displacement_histogram=dict(sorted(displacements.items())),
                shift_rate=(counts["shift_rounds"] / counts["eligible_rounds"] if counts["eligible_rounds"] else None),
                token_coverage={key: counts[key + "_positions"] / denominator if denominator else None
                                for key in ("same", "shift", "union", "shift_extra")})


def summarize(rows):
    report = {}
    for group in ("all", *DATASETS):
        selected = [r for r in rows if group == "all" or r["dataset"] == group]
        if not selected:
            continue
        rounds = [step for r in selected for step in r["spec"]["rounds"]]
        exact = [r for r in selected if r["first_difference"] is None]
        drafted = sum(len(r["proposal"]) for r in rounds)
        accepted = sum(r["decision"]["accepted_draft"] for r in rounds)
        forwards = sum(r["draft_forwards"] for r in rounds)
        emitted = sum(len(r["spec"]["token_ids"]) for r in selected)
        report[group] = dict(
            questions=len(selected), exact_vs_ar=len(exact), rounds=len(rounds),
            draft_forwards=forwards, verify_forwards=len(rounds),
            draft_tokens=drafted, accepted_draft_tokens=accepted, accepted_per_round=accepted / len(rounds),
            acceptance_rate=accepted / drafted, emitted_per_verify=emitted / len(rounds),
            accepted_per_drafter_forward=accepted / forwards if forwards else None,
            mean_draft_ms=sum(r["draft_ms"] for r in rounds) / len(rounds),
            median_draft_ms=median(r["draft_ms"] for r in rounds),
            mean_verify_ms=sum(r["verify_ms"] for r in rounds) / len(rounds),
            median_verify_ms=median(r["verify_ms"] for r in rounds),
            ar_tokens_per_second=1000 * sum(len(r["ar"]["token_ids"]) for r in selected)
            / sum(r["ar"]["wall_ms"] for r in selected),
            spec_tokens_per_second=1000 * emitted / sum(r["spec"]["wall_ms"] for r in selected),
            speedup_vs_ar_exact_subset=(sum(r["ar"]["wall_ms"] for r in exact)
                                       / sum(r["spec"]["wall_ms"] for r in exact)) if exact else None,
            reuse={str(h): reuse_summary(selected, h) for h in HORIZONS},
        )
    return report


def export(store, output, manifest, expected):
    from suffix_lab.store import atomic_json

    rows = store.rows("case")
    payload = dict(status="complete" if len(rows) == expected else "partial",
                   completed=len(rows), expected=expected, manifest=manifest, metrics=summarize(rows))
    atomic_json(output / "summary.json", payload)
    store.export("case", output / "cases.jsonl")
    snapshot = output / "checkpoint.snapshot.sqlite"
    store.backup(snapshot)
    temporary = output / (ARCHIVE + ".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        archive.write(snapshot, "checkpoint.sqlite")
        for name in ("summary.json", "cases.jsonl", "prompts.json", "preflight.json"):
            path = output / name
            if path.exists():
                archive.write(path, name)
    temporary.replace(output / ARCHIVE)
    snapshot.unlink()


def run(args):
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from suffix_lab.assets import AR, ROOT, TARGET, source_digest
    from suffix_lab.long60 import prepare_prompts
    from suffix_lab.runtime import configure_numerics, load_model, load_target, sync_model, tokenizers
    from suffix_lab.store import Store, atomic_json, digest

    if torch.cuda.device_count() < 2:
        raise RuntimeError("Select GPU T4 x2 on Kaggle")
    lock = json.loads((ROOT / "source-lock.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/long64.json").read_text(encoding="utf-8"))
    config.update(datasets=list(DATASETS), prompts_per_dataset=args.count)
    configure_numerics(config)
    revisions = {name: lock["models"][name] for name in (TARGET, AR)}
    paths = {name: snapshot_download(name, revision=revision,
                                     allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
             for name, revision in revisions.items()}
    tokenizer, identity = tokenizers(paths)
    prompts = prepare_prompts(config)
    prompt_plan = []
    for item in prompts:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": item["question"]}],
                                            tokenize=True, add_generation_prompt=True)
        if len(ids) + args.max_new_tokens > config["context_limit"]:
            raise ValueError(f"{item['id']} exceeds the context budget")
        prompt_plan.append(dict(item, prompt_tokens=ids))
    target, _, _ = load_target(paths[TARGET], config)
    drafter = load_model(paths[AR], "cuda:1")
    eos = set()
    for obj in (tokenizer, target.generation_config, target.config, drafter.generation_config, drafter.config):
        ids = obj.eos_token_id
        if ids is not None:
            eos.update(ids if isinstance(ids, (list, tuple)) else [ids])
    if not eos or drafter.config.vocab_size > target.config.vocab_size:
        raise ValueError("Invalid EOS or incompatible vocabularies")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol="qwen-ar-ar-rejected-suffix", git_revision=args.revision,
                    source_digest=source_digest(), runner_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    models=revisions, dataset_revisions=lock["datasets"], prompts_sha256=digest(prompt_plan),
                    seed=config["seed"], draft_length=args.draft_length, max_new_tokens=args.max_new_tokens,
                    horizon_tokens=list(HORIZONS), minimum_exact_span=MINIMUM, eos_ids=sorted(eos),
                    tokenizer_sha256=identity["shared_vocab_sha256"], precision="fp16", temperature=0,
                    target_device_map={k: str(v) for k, v in target.hf_device_map.items()},
                    drafter_device=str(drafter.device), torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.version.cuda, gpus=[torch.cuda.get_device_name(i) for i in range(2)])
    store = Store(output / "checkpoint.sqlite", manifest)
    try:
        atomic_json(output / "prompts.json", prompt_plan)
        with torch.inference_mode():
            warm = tokenizer.apply_chat_template([{"role": "user", "content": "What is 5 + 7?"}],
                                                  tokenize=True, add_generation_prompt=True)
            warm_ar = decode_ar(target, warm, eos, 12)
            warm_spec = decode_spec(target, drafter, warm, eos, 12, min(4, args.draft_length), sync_model)
            atomic_json(output / "preflight.json", dict(ar=warm_ar, spec=warm_spec,
                        first_difference=first_difference(warm_ar["token_ids"], warm_spec["token_ids"])))
            for i, item in enumerate(prompt_plan):
                if store.get("case", item["id"]) is not None:
                    continue
                ids = item["prompt_tokens"]
                if (i // len(DATASETS) + i) % 2:
                    spec = decode_spec(target, drafter, ids, eos, args.max_new_tokens, args.draft_length, sync_model)
                    ar = decode_ar(target, ids, eos, args.max_new_tokens)
                else:
                    ar = decode_ar(target, ids, eos, args.max_new_tokens)
                    spec = decode_spec(target, drafter, ids, eos, args.max_new_tokens, args.draft_length, sync_model)
                diff = first_difference(ar["token_ids"], spec["token_ids"])
                reuse = {str(h): diagnose(spec, ar["token_ids"], eos, h) for h in HORIZONS}
                store.put("case", item["id"], dict(id=item["id"], dataset=item["dataset"], ar=ar, spec=spec,
                                                  first_difference=diff, reuse=reuse))
                print(f"{i+1}/{len(prompt_plan)} {item['id']} | {len(spec['rounds'])} rounds | "
                      f"AR {ar['wall_ms']/1000:.2f}s | SD {spec['wall_ms']/1000:.2f}s | first diff {diff}", flush=True)
                if (i + 1) % 20 == 0:
                    export(store, output, manifest, len(prompt_plan))
    finally:
        try:
            export(store, output, manifest, len(prompt_plan))
        finally:
            store.close()
    print("Result ZIP:", output / ARCHIVE, flush=True)


def launch_kaggle(args):
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != args.revision:
        raise RuntimeError("Unexpected repository revision")
    work = Path("/kaggle/working")
    setup = runpy.run_path(str(repo / "scripts/kaggle_single_cell.py"))
    stream = runpy.run_path(str(repo / "scripts/kaggle_extension_cell.py"))["run_streamed"]
    setup["check_gpus"]()
    venv, log = Path("/tmp/ar-ar-reuse-venv"), work / "ar_ar_reuse_setup.log"
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
    command = [str(python), "-u", str(Path(__file__).resolve()), "--output", str(args.output.resolve()),
               "--revision", head, "--count", str(args.count), "--draft-length", str(args.draft_length),
               "--max-new-tokens", str(args.max_new_tokens)]
    code = stream(command, cwd=repo, env=env, log_path=work / "ar_ar_reuse_run.log")
    if code:
        raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="local")
    parser.add_argument("--count", type=int, default=100, help="Questions per dataset")
    parser.add_argument("--draft-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--kaggle", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.count <= 100 or not 5 <= args.draft_length <= 64 or args.max_new_tokens < 1:
        parser.error("Use 1–100 questions per dataset, draft length 5–64 and a positive generation limit")
    (launch_kaggle if args.kaggle else run)(args)


if __name__ == "__main__":
    main()
