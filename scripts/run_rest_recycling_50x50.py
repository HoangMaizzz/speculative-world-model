"""Run native REST and Token Recycling against the same greedy Vicuna reference."""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time


REPOS = {
    "REST": ("FasterDecoding/REST", "154d0f0ae45de5d8a9cbd1c696f6ad5da10904bf"),
    "TokenRecycling": ("Luowaterbi/TokenRecycling", "1b4c05cc642d111f12c61d4a99950d737121d9fe"),
}
MODEL = "lmsys/vicuna-7b-v1.3"
MODEL_REVISION = "236eeeab96f0dc2e463f2bebb7bb49809279c6d6"
MATH_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
SHAREGPT_FILE = "ShareGPT_2023.05.04v0_Wasteland_Edition.json"
SHAREGPT_SHA256 = "701b6b1beea876183b768b2ac530ba0b7f32a5e3755885c4e36b92302aa4c6bd"
WHEEL = "draftretriever-0.1.0-cp39-cp39-manylinux_2_34_x86_64.whl"
METHODS = ("ar", "rest", "recycling")
LIMITS = {"math500": 1024, "humaneval": 512}
WARMUPS = {
    "math500": "A box contains 19 blue beads and 23 red beads. How many beads are there in all? Explain.",
    "humaneval": 'def triple_or_zero(n: int) -> int:\n    """Return three times n for positive n, otherwise zero."""\n',
}
PINNED = ("transformers==4.37.1", "accelerate==0.24.1", "huggingface-hub==0.30.2",
          "numpy==1.26.1", "sentencepiece==0.2.0", "protobuf==3.20.3", "tqdm==4.66.1")


def checkout(work, key, env):
    name, revision = REPOS[key]
    repo = work / f"{key}-{revision[:12]}"
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", str(repo)], check=True, env=env)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, env=env, text=True,
                          capture_output=True).stdout.strip()
    if head != revision:
        subprocess.run(["git", "fetch", "--depth", "1", f"https://github.com/{name}.git", revision],
                       cwd=repo, check=True, env=env)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=repo, check=True, env=env)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, env=env, text=True).strip() != revision:
        raise RuntimeError(f"Unexpected revision in {repo}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, env=env).strip():
        raise RuntimeError(f"Tracked upstream files were changed in {repo}")
    return repo


def launch(args):
    import platform

    libc, version = platform.libc_ver()
    if sys.platform != "linux" or platform.machine() != "x86_64" or libc != "glibc" or (
            tuple(map(int, version.split(".")[:2])) < (2, 34)):
        raise RuntimeError("The pinned Rust wheel needs Linux x86-64 with glibc >= 2.34")
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", RAYON_NUM_THREADS="6", HF_HOME=str(args.work / "hf"))
    env.pop("PYTHONPATH", None)
    rest = checkout(args.work, "REST", env)
    recycling = checkout(args.work, "TokenRecycling", env)
    uv = [sys.executable, "-m", "uv"]
    venv = args.work / "venv"
    python = venv / "bin" / "python"
    if not python.is_file():
        subprocess.run([*uv, "venv", "--python", "3.9", str(venv)], check=True, env=env)
    if not (venv / ".ready").is_file():
        install = [*uv, "pip", "install", "--python", str(python)]
        subprocess.run([*install, "torch==2.3.1", "--index-url", "https://download.pytorch.org/whl/cu121"],
                       check=True, env=env)
        subprocess.run([*install, *PINNED, str(rest / "DraftRetriever" / "wheels" / WHEEL)],
                       check=True, env=env)
        (venv / ".ready").touch()
    command = [str(python), "-u", str(Path(__file__).resolve()), "--worker", str(rest), str(recycling),
               "--work", str(args.work), "--output", str(args.output)]
    if args.smoke:
        command.append("--smoke")
    subprocess.run(command, env=env, check=True)


def json_file(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_plan(math_path, human_path, smoke):
    math_rows = [json.loads(line) for line in math_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with gzip.open(human_path, "rt", encoding="utf-8") as file:
        human_rows = [json.loads(line) for line in file if line.strip()]
    if len(math_rows) != 500 or len(human_rows) != 164:
        raise ValueError(f"Unexpected dataset sizes: {len(math_rows)}, {len(human_rows)}")
    # Take the first 50 of the existing seed-42 MATH-500 sample order.
    quota = 2 if smoke else 50
    plan = []
    for dataset, rows in (("math500", math_rows), ("humaneval", human_rows)):
        indices = random.Random(42).sample(range(len(rows)), 100 if dataset == "math500" else 50)
        for index in indices[:quota]:
            row = rows[index]
            if dataset == "math500":
                prompt = "Solve the following math problem step by step. Give the final answer in \\boxed{}.\n\n"
                prompt += row["problem"] + "\n\nSolution:"
                item = {"id": f"math500:{index}", "source_id": row["unique_id"], "prompt": prompt}
            else:
                item = {"id": row["task_id"], "source_id": row["task_id"], "prompt": row["prompt"]}
            plan.append({"dataset": dataset, "index": index, "limit": 64 if smoke else LIMITS[dataset], **item})
    return plan


def author_function(path, name, globals_):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function, = (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    if name in ("update_inference_inputs", "recycling_forward"):
        # The native code packs every layer in one tensor. Apply its same gather
        # and copy per device, keeping the native accepted indices and lengths.
        loop = next(node for node in function.body if isinstance(node, ast.For)) if name == "recycling_forward" else function
        for i, node in enumerate(loop.body):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "tgt":
                break
        else:
            raise ValueError(f"Native KV compaction was not found in {path}")
        original = loop.body[i:i + 4]
        if ([type(node) for node in original] != [ast.Assign, ast.Assign, ast.Expr, ast.Expr]
                or [node.targets[0].id for node in original[:2]] != ["tgt", "dst"]
                or original[2].value.func.attr != "copy_" or original[3].value.func.attr != "fill_"):
            raise ValueError(f"Unexpected native KV compaction in {path}")
        indices = "verify_input_ids.size(1) + index_path" if name == "recycling_forward" else "select_indices"
        offset = "verify_input_ids.size(1)" if name == "recycling_forward" else "prev_input_len"
        call = ast.parse(f"compact_cache(past_key_values_data, current_length_data, {indices}, {offset})").body[0]
        loop.body[i:i + 4] = [ast.copy_location(call, original[0])]
        ast.fix_missing_locations(function)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), globals_)
    return globals_[name]


def device_map(visible_gpus):
    if visible_gpus == 1:
        return {"": "cuda:0"}
    mapping = {"model.embed_tokens": "cuda:0", "model.norm": "cuda:0", "lm_head": "cuda:0"}
    mapping.update({f"model.layers.{i}": f"cuda:{i // 16}" for i in range(32)})
    return mapping


def allocate_cache(torch, model, cache_type, capacity):
    config = model.config
    devices = {}
    for i, layer in enumerate(model.model.layers):
        devices.setdefault(next(layer.parameters()).device, []).append(i)
    lengths = torch.zeros(config.num_hidden_layers * 2, dtype=torch.long, device="cpu")
    caches = [None] * config.num_hidden_layers
    blocks = []
    for device, layers in devices.items():
        block = torch.zeros(
            (2 * len(layers), 1, config.num_key_value_heads, capacity, config.hidden_size // config.num_attention_heads),
            device=device, dtype=model.dtype,
        )
        blocks.append(block)
        for row, layer in enumerate(layers):
            caches[layer] = [cache_type(block[2 * row + j], lengths[2 * layer + j]) for j in range(2)]
    return caches, blocks, lengths


def compact_cache(blocks, lengths, indices, offset):
    for block in blocks:
        selected = block.index_select(-2, indices.to(block.device))
        block[..., offset:offset + indices.numel(), :].copy_(selected, non_blocking=True)
    lengths.fill_(offset + indices.numel())


def rest_index(work, tokenizer, sharegpt_path, draftretriever):
    with sharegpt_path.open("rb") as file:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: file.read(4 << 20), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != SHAREGPT_SHA256:
        raise ValueError("ShareGPT source checksum differs from the pinned file")
    path = work / f"sharegpt-{MODEL_REVISION[:12]}-{SHAREGPT_SHA256[:12]}.idx"
    if not path.is_file():
        from tqdm import tqdm

        with tempfile.TemporaryDirectory(dir=work) as directory:
            pending = Path(directory) / "sharegpt.idx"
            writer = draftretriever.Writer(str(pending), max_chunk_len=512 * 1024 * 1024,
                                          vocab_size=tokenizer.vocab_size + len(tokenizer.get_added_vocab()))
            with sharegpt_path.open(encoding="utf-8") as file:
                corpus = json.load(file)
            for chat in tqdm(corpus, desc="Building ShareGPT index"):
                for message in chat["conversations"]:
                    writer.add_entry(tokenizer.encode(message["value"]))
            writer.finalize()
            del writer
            if not pending.stat().st_size:
                raise RuntimeError("Empty REST datastore")
            pending.replace(path)
    return path


def comparable_ids(raw, eos, limit):
    for position, token in enumerate(raw[:limit]):
        if token == eos:
            return raw[:position + 1]
    return raw[:limit]


def first_difference(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right)) if len(left) != len(right) else None


def draft_counts(method, native_lengths, comparable_length):
    if method == "ar":
        return 0
    # REST's first native trace already removes the first verifier root.
    committed_before, accepted = 0, 0
    for i, native_length in enumerate(native_lengths):
        committed = native_length + int(method == "rest" and i == 0) - int(method == "recycling" and i == 0)
        inside_cap = min(committed, max(0, comparable_length - committed_before))
        roots = int(method == "rest" or i > 0)
        accepted += max(0, inside_cap - roots)
        committed_before += committed
    return accepted


def decode(torch, tokenizer, forward, method, prompt, limit, devices):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    for device in devices:
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    output, native_count, lengths = forward(ids, limit)
    for device in devices:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    raw = output[0, ids.shape[1]:].tolist()
    eos = tokenizer.eos_token_id
    # Upstream Recycling counts a verifier-predicted EOS in this case without appending it.
    implicit_eos = method == "recycling" and int(native_count) == len(raw) + 1
    restored = raw + [eos] if implicit_eos else raw
    tokens = comparable_ids(restored, eos, limit)
    return {"seconds": elapsed, "token_ids": tokens, "native_new_tokens": int(native_count),
            "native_output_length": len(raw), "implicit_eos_restored": implicit_eos,
            "discarded_tail": len(restored) - len(tokens), "steps": len(lengths),
            "accepted_draft_tokens": draft_counts(method, lengths, len(tokens)),
            "native_accept_lengths": lengths, "text": tokenizer.decode(tokens, skip_special_tokens=True)}


def measure(torch, tokenizer, forward, method, plan, output, devices, reset=None):
    rows = []
    with (output / f"{method}.jsonl").open("w", encoding="utf-8") as file:
        for i, item in enumerate(plan):
            dataset = item["dataset"]
            if i == 0 or dataset != plan[i - 1]["dataset"]:
                if reset:
                    reset()
                decode(torch, tokenizer, forward, method, WARMUPS[dataset], 32, devices)
                if reset:
                    reset()
            try:
                result = decode(torch, tokenizer, forward, method, item["prompt"], item["limit"], devices)
            except torch.cuda.OutOfMemoryError as exc:
                raise torch.cuda.OutOfMemoryError(f"{method} {item['id']} ({i + 1}/{len(plan)}): {exc}") from exc
            row = {**{k: item[k] for k in ("dataset", "id", "index", "limit")}, "method": method, **result}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            rows.append(row)
            print(f"{method} {i + 1}/{len(plan)} {item['id']} {row['seconds']:.2f}s "
                  f"{len(row['token_ids'])} tokens, {row['steps']} forwards", flush=True)
    return rows


def summary(results):
    reference = {r["id"]: r for r in results["ar"]}
    report, differences = [], []
    for dataset in ("math500", "humaneval", "all"):
        for method, rows in results.items():
            group = [r for r in rows if dataset in (r["dataset"], "all")]
            exact = []
            for row in group:
                ref = reference[row["id"]]
                mismatch = first_difference(ref["token_ids"], row["token_ids"])
                if mismatch is None:
                    exact.append(row)
                elif dataset != "all":
                    differences.append({"dataset": dataset, "id": row["id"], "method": method, "position": mismatch,
                                        "ar_token": next(iter(ref["token_ids"][mismatch:mismatch + 1]), None),
                                        "method_token": next(iter(row["token_ids"][mismatch:mismatch + 1]), None)})
            seconds = sum(r["seconds"] for r in group)
            steps = sum(r["steps"] for r in group)
            tokens = sum(len(r["token_ids"]) for r in group)
            accepted = sum(r["accepted_draft_tokens"] for r in group)
            ratio = sum(reference[r["id"]]["seconds"] for r in exact) / sum(r["seconds"] for r in exact) if exact else None
            report.append({"dataset": dataset, "method": method, "questions": len(group), "exact_vs_ar": len(exact),
                           "new_tokens": tokens, "seconds": seconds, "tokens_per_second": tokens / seconds,
                           "decode_forwards": steps, "emitted_per_decode_forward": tokens / steps,
                           "accepted_draft_tokens": accepted, "mean_accepted_draft": accepted / steps,
                           "discarded_tail_tokens": sum(r["discarded_tail"] for r in group),
                           "speedup_vs_ar": ratio if len(exact) == len(group) else None,
                           "speedup_vs_ar_exact_subset": ratio})
    return {"metrics": report, "first_differences": differences}


def probe_cuda(torch):
    runtime = f"torch={torch.__version__}, torch CUDA={torch.version.cuda}, " \
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
    print(f"CUDA preflight: {runtime}", flush=True)
    try:
        torch.cuda.init()
    except (AssertionError, RuntimeError) as exc:
        raise RuntimeError(
            "PyTorch could not initialize CUDA in the benchmark worker. "
            "On Kaggle select a GPU accelerator for this run, including Save & Run All. "
            f"{runtime}. CUDA error: {exc}"
        ) from exc
    visible = torch.cuda.device_count()
    if visible < 1:
        raise RuntimeError(f"No CUDA GPU is visible to the benchmark worker. {runtime}")
    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_properties(0)
    print(f"CUDA preflight passed: visible GPUs={visible}; primary cuda:0 ({gpu.name}, "
          f"{gpu.total_memory / 2**30:.1f} GiB)", flush=True)
    return gpu, visible


def worker(args):
    import gc
    import torch

    gpu, visible_gpus = probe_cuda(torch)
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import draftretriever

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    rest, recycling = map(Path, args.worker)
    sys.path[:0] = [str(rest), str(recycling)]
    from rest.model import utils as rest_utils
    from rest.model.rest_model import RestModel
    from rest.model.kv_cache import KVCache as RestCache
    from model.recycling.modeling_llama_kv import LlamaForCausalLM
    from model.recycling.kv_cache import KVCache as RecyclingCache
    from model.recycling.tree_template_ import choose_tree_template

    math_path = Path(hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset",
                                     revision=MATH_REVISION))
    human_path = rest / "human_eval" / "HumanEval.jsonl.gz"
    plan = load_plan(math_path, human_path, args.smoke)
    output = Path(tempfile.mkdtemp(prefix="smoke-" if args.smoke else "full-", dir=args.output))
    print(f"Result files: {output}", flush=True)
    json_file(output / "prompts.json", plan)
    model_path = snapshot_download(MODEL, revision=MODEL_REVISION, allow_patterns=["*.json", "tokenizer.model",
                                                                                 "pytorch_model-*.bin"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    for item in plan:
        item["prompt_tokens"] = len(tokenizer.encode(item["prompt"]))
        if item["prompt_tokens"] + item["limit"] + 82 > 2048:
            raise ValueError(f"{item['id']} exceeds the shared Vicuna context window")
    json_file(output / "prompts.json", plan)
    placement = device_map(visible_gpus)
    devices = tuple(sorted(set(placement.values())))
    # At most one extra committed tree plus the next verifier input. Allocate
    # by the actual benchmark window instead of the checkpoint's max context.
    capacity = max(item["prompt_tokens"] + item["limit"] for item in plan) + 82
    capacity = max(capacity, max(len(tokenizer.encode(p)) for p in WARMUPS.values()) + 32 + 82)
    print(f"Benchmark uses {', '.join(devices)}; native tree KV capacity={capacity} tokens", flush=True)
    sharegpt = Path(hf_hub_download("Aeala/ShareGPT_Vicuna_unfiltered", SHAREGPT_FILE, repo_type="dataset"))
    index_path = rest_index(args.work, tokenizer, sharegpt, draftretriever)
    json_file(output / "manifest.json", {
        "mode": "smoke" if args.smoke else "full", "repos": REPOS, "model": MODEL, "model_revision": MODEL_REVISION,
        "math_revision": MATH_REVISION, "human_eval_sha256": hashlib.sha256(human_path.read_bytes()).hexdigest(),
        "sharegpt_sha256": SHAREGPT_SHA256, "index_bytes": index_path.stat().st_size, "seed": 42, "dtype": "float16",
        "gpu": gpu.name, "gpu_vram_bytes": gpu.total_memory, "visible_gpus": visible_gpus, "devices": devices,
        "device_map": placement, "native_cache_capacity": capacity,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "temperature": 0, "warmup_prompts": WARMUPS, "warmup_token_budget": 32,
        "recycling_state": "zero after warmup and across datasets, retained within each dataset",
        "native_stop_policy": "native calls keep their stop logic; recycling gets one extra iteration for the prompt root",
        "rest": {"nodes": 64, "suffix_lengths": [16, 2], "max_continuation": 10},
        "recycling": {"top_k": 8, "tree": "2.2.2", "draft_nodes": 80},
    })
    options = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True, "device_map": placement,
               "local_files_only": True}

    def run_arm(method):
        reset = None
        if method == "ar":
            model = AutoModelForCausalLM.from_pretrained(model_path, **options).eval()

            def forward(ids, limit):
                output_ids = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                            max_new_tokens=limit, eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.eos_token_id)
                count = output_ids.shape[1] - ids.shape[1]
                return output_ids, count, [1] * count
        elif method == "rest":
            model = RestModel.from_pretrained(model_path, **options).eval()
            datastore = draftretriever.Reader(str(index_path))
            update = author_function(rest / "rest" / "model" / "utils.py", "update_inference_inputs", {
                **vars(rest_utils), "compact_cache": compact_cache,
            })
            author_forward = author_function(rest / "llm_judge" / "gen_model_answer_rest.py", "rest_forward", {
                **vars(rest_utils), "torch": torch, "time": time, "update_inference_inputs": update,
                "initialize_past_key_values": lambda model: allocate_cache(torch, model, RestCache, capacity),
            })

            def forward(ids, limit):
                output_ids, count, _, lengths, _ = author_forward(
                    ids, model, tokenizer, limit, 0.0, 0.0, datastore, 64, list(range(16, 1, -1))
                )
                return output_ids, count, lengths
        else:
            from types import SimpleNamespace

            model = LlamaForCausalLM.from_pretrained(model_path, **options).eval()
            adjacency = torch.zeros((model.config.vocab_size, 8), device="cuda:0", dtype=torch.long)
            reset = adjacency.zero_
            author_forward = author_function(recycling / "evaluation" / "inference_recycling.py", "recycling_forward", {
                "torch": torch, "initialize_past_key_values": lambda model: allocate_cache(
                    torch, model, RecyclingCache, capacity
                ),
                "choose_tree_template": choose_tree_template, "adj_matrix": adjacency, "compact_cache": compact_cache,
            })

            def forward(ids, limit):
                # The first upstream iteration can emit only the final prompt token.
                output_ids, count, _, lengths = author_forward(
                    SimpleNamespace(input_ids=ids), model, tokenizer, limit + 1
                )
                return output_ids, count, lengths

        with torch.inference_mode():
            return measure(torch, tokenizer, forward, method, plan, output, devices, reset)

    results = {}
    try:
        for method in METHODS:
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            results[method] = run_arm(method)
            gc.collect()
        json_file(output / "summary.json", {"status": "complete", **summary(results)})
        print(f"Completed: {output / 'summary.json'}", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        json_file(output / "execution_status.json", {
            "status": "partial_cuda_oom", "completed_methods": list(results), "failed_method": method, "error": str(exc)
        })
        print(f"INCOMPLETE: CUDA ran out of memory. Partial results preserved. {exc}", flush=True)
    finally:
        archive = shutil.make_archive(str(output), "zip", root_dir=output)
        print(f"Results archive: {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run 2 questions per dataset at a 64-token limit")
    parser.add_argument("--work", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--output", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    (worker if args.worker else launch)(args)


if __name__ == "__main__":
    main()
