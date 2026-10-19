"""Pinned public assets and reproducible prompt selection; never store credentials."""
import hashlib
import json
from pathlib import Path
import random
from .store import atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]
TARGET = "Qwen/Qwen2.5-7B-Instruct"
AR = "Qwen/Qwen2.5-1.5B-Instruct"
DLLM = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"


def source_digest():
    files = sorted((ROOT/"src").rglob("*.py")) + [ROOT/"requirements-kaggle.txt", ROOT/"source-lock.json"]
    return digest({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def prepare(config, directory):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "source-lock.json").read_text())
    paths = {}
    for name, revision in lock["models"].items():
        print("Preparing", name, revision[:12], flush=True)
        paths[name] = snapshot_download(name, revision=revision,
            ignore_patterns=["*.bin", "*.pt", "*.ot", "*.msgpack", "onnx/*"])
    for filename, expected in lock["source_sha256"].items():
        actual = hashlib.sha256((Path(paths[DLLM]) / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Pinned remote source hash differs: {filename}")
    prompts = []
    for benchmark in config["datasets"]:
        name = {"gsm8k": "openai/gsm8k", "math500": "HuggingFaceH4/MATH-500"}[benchmark]
        kwargs = {"name": "main"} if benchmark == "gsm8k" else {}
        dataset = load_dataset(name, split="test", revision=lock["datasets"][name], **kwargs)
        indices = random.Random(config["seed"]).sample(range(len(dataset)), config["prompts_per_dataset"])
        for i in indices:
            row = dataset[i]
            prompts.append({"id": f"{benchmark}:{i}", "dataset": benchmark, "index": i,
                            "question": row["question" if benchmark == "gsm8k" else "problem"],
                            "dataset_fingerprint": dataset._fingerprint})
    manifest = {"lock": lock, "paths": paths, "prompts": prompts,
                "selection": {k: config[k] for k in ("datasets", "seed", "prompts_per_dataset")}}
    manifest["content_digest"] = digest({k: v for k, v in manifest.items() if k != "paths"})
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def read_assets(directory, config):
    manifest = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    if manifest["selection"] != {k: config[k] for k in ("datasets", "seed", "prompts_per_dataset")}:
        raise ValueError("Asset selection differs; run prepare with this configuration")
    for path in manifest["paths"].values():
        if not Path(path).is_dir():
            raise FileNotFoundError("Model cache moved; rerun prepare (resume identity uses revisions, not paths)")
    return manifest
