"""Run the pinned native EAGLE-3 tree and measure exact reuse of rejected tokens."""

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

from eagle3_qwen_runtime import AUX_LAYERS, EAGLE_REVISION, HEAD_MODEL, HEAD_REVISION, load_model
from eagle3_reuse_metrics import HORIZONS, MINIMUM, diagnose, first_difference, summarize


REPOSITORY = "https://github.com/SafeAILab/EAGLE.git"
DATASETS = ("math500", "gsm8k")
CONTEXT = 2048
ARCHIVE = "eagle3_reuse_results.zip"


def paths_for_prompts(tokenizer, config):
    from suffix_lab.long60 import prepare_prompts

    prompts = []
    for item in prepare_prompts(config):
        ids = tokenizer.apply_chat_template([{"role": "user", "content": item["question"]}],
                                            tokenize=True, add_generation_prompt=True)
        if len(ids) + config["max_new_tokens"] > CONTEXT:
            raise ValueError(f"{item['id']} exceeds the context budget")
        prompts.append(dict(item, prompt_tokens=ids))
    return prompts


def export(store, output, manifest, expected, status):
    from suffix_lab.store import atomic_json

    rows = store.rows("case")
    summary = dict(status=status, completed=len(rows), expected=expected,
                   manifest=manifest, metrics=summarize(rows), failure=store.get("meta", "failure"))
    atomic_json(output / "summary.json", summary)
    store.export("case", output / "cases.jsonl")
    snapshot = output / "checkpoint.snapshot.sqlite"
    store.backup(snapshot)
    temporary = output / (ARCHIVE + ".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        archive.write(snapshot, "checkpoint.sqlite")
        for name in ("summary.json", "cases.jsonl", "prompts.json", "preflight.json"):
            path = output / name
            if path.is_file():
                archive.write(path, name)
    temporary.replace(output / ARCHIVE)
    snapshot.unlink()


def preflight(model, tokenizer, prompts, eos):
    from suffix_lab.runtime import memory

    warm = tokenizer.apply_chat_template([{"role": "user", "content": "What is 5 + 7?"}],
                                        tokenize=True, add_generation_prompt=True)
    candidates = [dict(id="synthetic", prompt_tokens=warm), *prompts[:2]]
    checks = []
    for case in candidates:
        prompt = case["prompt_tokens"]
        ar = model.decode_reference(prompt, eos, 16)
        tree = model.decode_tree(prompt, eos, 16)
        checks.append(dict(id=case["id"], ar=ar, spec=tree,
                           first_difference=first_difference(ar["token_ids"], tree["token_ids"])))
    return dict(checks=checks, memory=memory())


def run(args):
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from suffix_lab.assets import ROOT, TARGET, source_digest
    from suffix_lab.runtime import configure_numerics, tokenizers
    from suffix_lab.store import Store, atomic_json, digest

    started = perf_counter()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Select GPU T4 x2 on Kaggle")
    lock = json.loads((ROOT / "source-lock.json").read_text(encoding="utf-8"))
    config = json.loads((ROOT / "configs/long64.json").read_text(encoding="utf-8"))
    config.update(datasets=list(DATASETS), prompts_per_dataset=args.count, max_new_tokens=args.max_new_tokens)
    configure_numerics(config)
    revisions = {TARGET: lock["models"][TARGET], HEAD_MODEL: HEAD_REVISION}
    paths = {name: Path(snapshot_download(name, revision=revision,
                                          allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"]))
             for name, revision in revisions.items()}
    tokenizer, identity = tokenizers({TARGET: str(paths[TARGET])})
    prompts = paths_for_prompts(tokenizer, config)
    model = load_model(paths[TARGET], paths[HEAD_MODEL], CONTEXT + args.nodes, args.nodes, args.depth, args.top_k)
    eos = set()
    for obj in (tokenizer, model.base_model.generation_config, model.base_model.config):
        ids = obj.eos_token_id
        if ids is not None:
            eos.update(ids if isinstance(ids, (list, tuple)) else [ids])
    if not eos:
        raise ValueError("Missing target EOS IDs")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol="eagle3-tree-reuse-v1", git_revision=args.revision, source_digest=source_digest(),
                    eagle_upstream_revision=EAGLE_REVISION, models=revisions, dataset_revisions=lock["datasets"],
                    prompts_sha256=digest(prompts), seed=config["seed"], context_limit=CONTEXT,
                    tree_budget_including_verifier_root=args.nodes, draft_expansion_depth=args.depth,
                    top_k=args.top_k, max_new_tokens=args.max_new_tokens, horizons=list(HORIZONS),
                    minimum_exact_span=MINIMUM, eos_ids=sorted(eos), auxiliary_post_layers=list(AUX_LAYERS),
                    tokenizer_sha256=identity["shared_vocab_sha256"], precision="fp16", temperature=0,
                    target_attention="eager", target_device_map=model.base_model.hf_device_map,
                    drafter_device=str(model.head_device), torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.version.cuda, gpus=[torch.cuda.get_device_name(i) for i in range(2)])
    store = Store(output / "checkpoint.sqlite", manifest)
    status = "failed"
    current_id = "preflight"
    try:
        store.put("meta", "failure", None)
        atomic_json(output / "prompts.json", prompts)
        with torch.inference_mode():
            check = preflight(model, tokenizer, prompts, eos)
            atomic_json(output / "preflight.json", check)
            differences = {r["id"]: r["first_difference"] for r in check["checks"]}
            print(f"Greedy preflight first differences (None = exact): {differences}", flush=True)
            if all(value is not None for value in differences.values()):
                raise RuntimeError("All greedy preflight cases diverged; inspect preflight.json before the full run")
            completed = len(store.rows("case"))
            status = "partial"
            for i, item in enumerate(prompts):
                current_id = item["id"]
                if store.get("case", current_id) is not None:
                    continue
                if perf_counter() - started >= args.budget_seconds:
                    print(f"Time budget reached after {completed}/{len(prompts)} cases", flush=True)
                    break
                ids = item["prompt_tokens"]
                if (i + i // len(DATASETS)) % 2:
                    ar = model.decode_reference(ids, eos, args.max_new_tokens)
                    spec = model.decode_tree(ids, eos, args.max_new_tokens)
                else:
                    spec = model.decode_tree(ids, eos, args.max_new_tokens)
                    ar = model.decode_reference(ids, eos, args.max_new_tokens)
                diff = first_difference(ar["token_ids"], spec["token_ids"])
                store.put("case", current_id, dict(id=current_id, dataset=item["dataset"], ar=ar, spec=spec,
                                                   first_difference=diff, reuse=diagnose(spec, ar, eos)))
                completed += 1
                print(f"{completed}/{len(prompts)} {current_id} | {len(spec['rounds'])} tree rounds | "
                      f"AR {ar['wall_ms']/1000:.2f}s | EAGLE {spec['wall_ms']/1000:.2f}s | first diff {diff}", flush=True)
                if completed % 20 == 0 and completed < len(prompts):
                    export(store, output, manifest, len(prompts), status)
            status = "complete" if completed == len(prompts) else "partial"
    except BaseException as error:
        status = "failed"
        store.put("meta", "failure", dict(id=current_id, type=type(error).__name__, message=str(error)))
        raise
    finally:
        try:
            export(store, output, manifest, len(prompts), status)
        finally:
            store.close()
    print(f"{status}: {output / ARCHIVE}", flush=True)


def fetch_eagle(path):
    commit = EAGLE_REVISION
    if not (path / ".git").is_dir():
        path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=path, capture_output=True, text=True)
    if head.returncode:
        subprocess.run(["git", "fetch", "--depth", "1", REPOSITORY, commit], cwd=path, check=True)
        subprocess.run(["git", "checkout", "-q", "--detach", commit], cwd=path, check=True)
    elif head.stdout.strip() != commit:
        raise RuntimeError(f"Different pinned upstream revision in {path}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True).strip():
        raise RuntimeError(f"Tracked upstream files are modified in {path}")


def launch_kaggle(args):
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != args.revision:
        raise RuntimeError("Unexpected benchmark Git revision")
    work = Path("/kaggle/working")
    setup = runpy.run_path(str(repo / "scripts/kaggle_single_cell.py"))
    stream = runpy.run_path(str(repo / "scripts/kaggle_extension_cell.py"))["run_streamed"]
    setup["check_gpus"]()
    upstream = Path("/kaggle/temp") / f"eagle-{EAGLE_REVISION[:12]}"
    fetch_eagle(upstream)
    venv, log = Path("/tmp/eagle3-reuse-venv"), work / "eagle3_reuse_setup.log"
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
    env["PYTHONPATH"] = str(upstream)
    env.setdefault("HF_HOME", "/tmp/suffix-hf-cache")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    command = [str(python), "-u", str(Path(__file__).resolve()), "--output", str(args.output.resolve()),
               "--revision", head, "--count", str(args.count), "--nodes", str(args.nodes),
               "--depth", str(args.depth), "--top-k", str(args.top_k),
               "--max-new-tokens", str(args.max_new_tokens), "--budget-seconds", str(args.budget_seconds)]
    code = stream(command, cwd=repo, env=env, log_path=work / "eagle3_reuse_run.log")
    if code:
        sys.exit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="local")
    parser.add_argument("--count", type=int, default=100, help="Questions per dataset")
    parser.add_argument("--nodes", type=int, default=32, help="Tree node budget including the verifier root")
    parser.add_argument("--depth", type=int, default=7, help="Upstream EAGLE-3 expansion iterations")
    parser.add_argument("--top-k", type=int, default=4, help="Width of the upstream draft search frontier")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--budget-seconds", type=int, default=21600)
    parser.add_argument("--kaggle", action="store_true")
    args = parser.parse_args()
    if (not 1 <= args.count <= 100 or not 0 < args.max_new_tokens < CONTEXT or args.budget_seconds < 1
            or not 1 <= args.top_k <= 16 or not 1 <= args.depth <= 32
            or not 2 <= args.nodes <= min(128, 1 + args.top_k + args.depth * args.top_k**2)):
        parser.error("Invalid question count, decode budget, or EAGLE-3 tree geometry")
    (launch_kaggle if args.kaggle else run)(args)


if __name__ == "__main__":
    main()
