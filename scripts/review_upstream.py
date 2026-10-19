"""Fetch pinned public source only; --refresh-lock explicitly resolves newer revisions."""
import argparse
import hashlib
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "upstream-review"
MODELS = ["Efficient-Large-Model/Fast_dLLM_v2_1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
DATASETS = ["openai/gsm8k", "HuggingFaceH4/MATH-500"]


def get(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-lock", action="store_true")
    args = parser.parse_args()
    lock = json.loads((ROOT/"source-lock.json").read_text())
    if args.refresh_lock:
        for family, names in (("models", MODELS), ("datasets", DATASETS)):
            for name in names:
                lock[family][name] = json.loads(get(f"https://huggingface.co/api/{family}/{name}"))["sha"]
    OUT.mkdir(exist_ok=True)
    for filename in ("modeling.py", "configuration.py", "config.json"):
        revision = lock["models"][MODELS[0]]
        content = get(f"https://huggingface.co/{MODELS[0]}/resolve/{revision}/{filename}")
        value = hashlib.sha256(content).hexdigest()
        if args.refresh_lock:
            lock["source_sha256"][filename] = value
        elif value != lock["source_sha256"][filename]:
            raise RuntimeError(f"Pinned source hash differs: {filename}")
        (OUT/filename).write_bytes(content)
    if args.refresh_lock:
        (ROOT/"source-lock.json").write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")
    print("Pinned public source verified; no weights downloaded.")


if __name__ == "__main__":
    main()
