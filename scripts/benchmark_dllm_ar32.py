"""Benchmark Fast-dLLM one-step draft32 + Qwen greedy verification against Qwen AR."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
from statistics import median
import subprocess
import sys
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile


BASE_REVISION = "59d4148e69cae6d850f0ac009dcaf6e5f9ec2752"
REPOSITORY = "https://github.com/HoangMaizzz/speculative-world-model.git"
DRAFT_LENGTH = 32
DATASETS = ("math500", "gsm8k")


def draft_once(model, prefix, block_size, mask_id):
    import torch

    # Keep the checkpoint's native attention and score all 32 positions in one call.
    # This model's logit row at position i predicts the token at position i + 1.
    canvas = prefix + [mask_id] * DRAFT_LENGTH
    ids = torch.tensor([canvas], device=model.device)
    output = model(ids, block_size=block_size, use_cache=False, use_block_cache=False,
                   logits_to_keep=DRAFT_LENGTH + 1)
    logits = output.logits[0, :DRAFT_LENGTH]
    if logits.shape[0] != DRAFT_LENGTH:
        raise RuntimeError("Expected one logit row for each of the 32 draft positions")
    logits[:, mask_id] = -torch.inf
    return logits.argmax(dim=-1).tolist()


class CachedTarget:
    """Keep the accepted Qwen KV prefix and process the correction on the next call."""

    def __init__(self, model, prompt):
        self.model = model
        self.cache = None
        self.next_token = self.forward(prompt)[-1]
        self.pending = []

    def forward(self, tokens):
        import torch

        ids = torch.tensor([tokens], device=self.model.device)
        result = self.model(ids, past_key_values=self.cache, use_cache=True,
                            logits_to_keep=len(tokens) if self.cache is not None else 1)
        self.cache = result.past_key_values
        return result.logits[0].argmax(dim=-1).tolist()

    def verify(self, proposal, eos, remaining):
        from suffix_lab.long60_plan import decide

        cached = self.cache.get_seq_length()
        top1 = self.forward(self.pending + proposal)
        if not self.pending:
            top1.insert(0, self.next_token)
        decision = decide(proposal, top1, eos, remaining)
        committed = decision["committed_tokens"]
        accepted = min(decision["accepted_draft"], len(committed))
        self.cache.crop(cached + len(self.pending) + accepted)
        self.pending = committed[accepted:]
        return decision


def decode_ar(model, prompt, eos, limit):
    start = perf_counter()
    target = CachedTarget(model, prompt)
    tokens = [target.next_token]
    while len(tokens) < limit and tokens[-1] not in eos:
        tokens.extend(target.forward(tokens[-1:]))
    return dict(token_ids=tokens, wall_ms=1000 * (perf_counter() - start), forwards=len(tokens))


def decode_spec(model, draft, prompt, eos, limit):
    from suffix_lab.long60_plan import until_eos

    start = perf_counter()
    target = CachedTarget(model, prompt)
    prefill_ms = 1000 * (perf_counter() - start)
    tokens, rounds = [], []
    while len(tokens) < limit and (not tokens or tokens[-1] not in eos):
        draft_start = perf_counter()
        draft_ids = draft(prompt + tokens)
        draft_ms = 1000 * (perf_counter() - draft_start)
        if len(draft_ids) != DRAFT_LENGTH:
            raise RuntimeError("Expected exactly 32 draft tokens")
        remaining = limit - len(tokens)
        proposal = until_eos(draft_ids[:remaining], eos)
        verify_start = perf_counter()
        decision = target.verify(proposal, eos, remaining)
        verify_ms = 1000 * (perf_counter() - verify_start)
        rounds.append(dict(offset=len(tokens), draft_ids=draft_ids, verified=len(proposal),
                           accepted=decision["accepted_draft"], correction=decision["correction"],
                           bonus=decision["bonus"], draft_ms=draft_ms, verify_ms=verify_ms))
        tokens.extend(decision["committed_tokens"])
    return dict(token_ids=tokens, prefill_ms=prefill_ms, wall_ms=1000 * (perf_counter() - start),
                draft_forwards=len(rounds), verify_forwards=len(rounds), rounds=rounds)


def first_difference(a, b):
    for i, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def summarize(rows):
    summary = {}
    for group in ("all", *DATASETS):
        selected = [r for r in rows if group == "all" or r["dataset"] == group]
        if not selected:
            continue
        matched = [r for r in selected if r["first_difference"] is None]
        steps = [step for row in selected for step in row["spec"]["rounds"]]
        count = len(steps)
        draft_ms = sum(s["draft_ms"] for s in steps)
        verify_ms = sum(s["verify_ms"] for s in steps)
        total_tokens = sum(len(r["spec"]["token_ids"]) for r in selected)
        summary[group] = dict(
            questions=len(selected), exact_vs_ar=len(matched), draft_forwards=count, verify_forwards=count,
            spec_target_forwards_including_prefill=count + len(selected),
            ar_forwards=sum(r["ar"]["forwards"] for r in selected),
            verified_draft_tokens=sum(s["verified"] for s in steps),
            mean_accepted_draft_tokens=sum(s["accepted"] for s in steps) / count,
            mean_emitted_tokens=total_tokens / count,
            mean_draft_ms=draft_ms / count, median_draft_ms=median(s["draft_ms"] for s in steps),
            mean_verify_ms=verify_ms / count, median_verify_ms=median(s["verify_ms"] for s in steps),
            draft_to_verify_time_ratio=draft_ms / verify_ms,
            spec_tokens_per_second=1000 * total_tokens / sum(r["spec"]["wall_ms"] for r in selected),
            ar_tokens_per_second=1000 * sum(len(r["ar"]["token_ids"]) for r in selected)
            / sum(r["ar"]["wall_ms"] for r in selected),
            speedup_vs_ar_exact_subset=(sum(r["ar"]["wall_ms"] for r in matched)
                                       / sum(r["spec"]["wall_ms"] for r in matched)) if matched else None,
        )
    return summary


def run(args):
    import torch
    import transformers
    from suffix_lab.assets import DLLM, ROOT, TARGET, source_digest
    from suffix_lab.extension import prepare_models
    from suffix_lab.long60 import prepare_prompts
    from suffix_lab.runtime import configure_numerics, load_model, load_target, tokenizers
    from suffix_lab.store import Store, atomic_json, digest

    if torch.cuda.device_count() < 2:
        raise RuntimeError("Select GPU T4 x2 on Kaggle for the paired FP16 setup")
    config = json.loads((ROOT / "configs/long64.json").read_text(encoding="utf-8"))
    config.update(datasets=list(DATASETS), prompts_per_dataset=args.count)
    configure_numerics(config)
    prompts = prepare_prompts(config)
    paths, revisions = prepare_models()
    tokenizer, tokenizer_evidence = tokenizers(paths)
    target, _, _ = load_target(paths[TARGET], config)
    drafter = load_model(paths[DLLM], "cuda:1")
    eos = set()
    for obj in (tokenizer, target.config, target.generation_config, drafter.config, drafter.generation_config):
        ids = obj.eos_token_id
        if ids is not None:
            eos.update(ids if isinstance(ids, (list, tuple)) else [ids])
    block_size, mask_id = config["physical_block_size"], config["mask_id"]
    if not eos or mask_id in eos or drafter.config.vocab_size > target.config.vocab_size:
        raise ValueError("Invalid EOS/MASK configuration or incompatible model vocabulary sizes")

    def draft(prefix):
        return draft_once(drafter, prefix, block_size, mask_id)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol="fresh-dllm32-one-step-qwen-ar", base_revision=BASE_REVISION,
                    source_digest=source_digest(), runner_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                    models=revisions, prompt_sha256=digest(prompts), seed=config["seed"],
                    count_per_dataset=args.count, max_new_tokens=args.max_new_tokens,
                    draft_length=DRAFT_LENGTH, unmask_steps_per_draft=1, physical_block_size=block_size,
                    draft_cache="full prefix per draft", target_cache="cached accepted prefix",
                    tokenizer=tokenizer_evidence["shared_vocab_sha256"],
                    target_device_map={k: str(v) for k, v in target.hf_device_map.items()},
                    mask_id=mask_id, eos_ids=sorted(eos), precision="fp16", torch=torch.__version__,
                    transformers=transformers.__version__, cuda=torch.version.cuda,
                    gpus=[torch.cuda.get_device_name(i) for i in range(2)])
    store = Store(output / "checkpoint.sqlite", manifest)
    try:
        with torch.inference_mode():
            warmup = tokenizer.apply_chat_template([{"role": "user", "content": "What is 5 + 7?"}],
                                                   tokenize=True, add_generation_prompt=True)
            decode_ar(target, warmup, eos, 8)
            CachedTarget(target, warmup).verify(draft(warmup), set(), DRAFT_LENGTH)
            for i, item in enumerate(prompts):
                if store.get("case", item["id"]) is not None:
                    continue
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": item["question"]}],
                                                       tokenize=True, add_generation_prompt=True)
                if len(prompt) + args.max_new_tokens + DRAFT_LENGTH + block_size - 1 > config["context_limit"]:
                    raise ValueError(f"{item['id']} exceeds the experiment context budget")
                if (i // len(DATASETS) + i) % 2:
                    spec = decode_spec(target, draft, prompt, eos, args.max_new_tokens)
                    ar = decode_ar(target, prompt, eos, args.max_new_tokens)
                else:
                    ar = decode_ar(target, prompt, eos, args.max_new_tokens)
                    spec = decode_spec(target, draft, prompt, eos, args.max_new_tokens)
                difference = first_difference(ar["token_ids"], spec["token_ids"])
                store.put("case", item["id"], dict(id=item["id"], dataset=item["dataset"],
                                                  prompt_tokens=prompt, ar=ar, spec=spec,
                                                  first_difference=difference))
                print(f"{i + 1}/{len(prompts)} {item['id']} | draft/verify rounds {spec['verify_forwards']} | "
                      f"AR {ar['wall_ms'] / 1000:.2f}s | spec {spec['wall_ms'] / 1000:.2f}s | "
                      f"first difference {difference}", flush=True)
    finally:
        rows = store.rows("case")
        result = dict(manifest=manifest, completed=len(rows), expected=len(prompts), metrics=summarize(rows))
        atomic_json(output / "summary.json", result)
        store.export("case", output / "cases.jsonl")
        store.close()
        temporary = output / "results.tmp"
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for name in ("summary.json", "cases.jsonl", "checkpoint.sqlite"):
                archive.write(output / name, name)
        temporary.replace(output / "dllm_ar32_results.zip")
    print(json.dumps(result["metrics"], indent=2), flush=True)


def launch_kaggle(args):
    work = Path("/kaggle/working")
    repo = work / f"dllm-ar32-src-{BASE_REVISION[:12]}"
    if not (repo / ".git").is_dir():
        subprocess.run(["git", "init", str(repo)], check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    if head.stdout.strip() != BASE_REVISION:
        subprocess.run(["git", "fetch", "--depth", "1", REPOSITORY, BASE_REVISION], cwd=repo, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo, check=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != BASE_REVISION:
        raise RuntimeError("Unexpected repository revision")
    setup = runpy.run_path(str(repo / "scripts/kaggle_single_cell.py"))
    stream = runpy.run_path(str(repo / "scripts/kaggle_extension_cell.py"))["run_streamed"]
    setup["check_gpus"]()
    venv = Path("/tmp/dllm-ar32-venv")
    log = work / "dllm_ar32_setup.log"
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
    code = stream([str(python), "-u", str(Path(__file__).resolve()), "--output", str(args.output.resolve()),
                   "--count", str(args.count), "--max-new-tokens", str(args.max_new_tokens)],
                  cwd=repo, env=env, log_path=work / "dllm_ar32_run.log")
    if code:
        raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100, help="Questions per dataset")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--kaggle", action="store_true", help="Set up the pinned Kaggle environment first")
    args = parser.parse_args()
    if not 1 <= args.count <= 100 or args.max_new_tokens < 1:
        parser.error("Use 1–100 questions per dataset and a positive generation limit")
    (launch_kaggle if args.kaggle else run)(args)


if __name__ == "__main__":
    main()
