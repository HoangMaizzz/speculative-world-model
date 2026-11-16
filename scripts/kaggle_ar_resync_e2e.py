"""Set up and run the AR resynchronization benchmark in a Kaggle notebook."""

import argparse
import os
from pathlib import Path
import runpy
import subprocess


def find_trace(path):
    if path.is_file() and path.suffix.lower() == ".zip":
        return path
    if path.is_dir():
        if (path / "cases.jsonl").is_file() and (path / "summary.json").is_file():
            return path
        extracted = {p.parent for p in path.rglob("cases.jsonl") if (p.parent / "summary.json").is_file()}
        zipped = set(path.rglob("ar_ar_reuse_results.zip"))
        if len(extracted) == 1:
            return next(iter(extracted))
        if not extracted and len(zipped) == 1:
            return next(iter(zipped))
    raise FileNotFoundError(f"Provide the original AR-AR64 ZIP or the folder with cases.jsonl and summary.json: {path}")


def launch(revision, trace, output, count, resume=None):
    trace = find_trace(trace.resolve())
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if revision != head:
        raise ValueError("Checkout does not match pinned benchmark revision")
    setup = runpy.run_path(str(repo / "scripts/kaggle_single_cell.py"))
    stream = runpy.run_path(str(repo / "scripts/kaggle_extension_cell.py"))["run_streamed"]
    setup["check_gpus"]()
    log_dir = Path("/kaggle/working")
    venv, log = Path("/tmp/ar-resync-e2e-venv"), log_dir / "ar_resync_setup.log"
    python = setup["ensure_environment"](venv, log)
    stamp = venv / "installed-revision"
    if not stamp.exists() or stamp.read_text() != revision:
        install = setup["setup_command"]
        install([str(python), "-m", "pip", "install", "torch==2.9.0", "--index-url",
                 "https://download.pytorch.org/whl/cu126"], log)
        install([str(python), "-m", "pip", "install", "-r", str(repo / "requirements-kaggle.txt")], log)
        install([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], log)
        stamp.write_text(revision)
    env = setup["child_environment"](os.environ)
    env.setdefault("HF_HOME", "/tmp/suffix-hf-cache")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("OMP_NUM_THREADS", "2")
    head = Path("/tmp/ar_resync_head.json.gz")
    result = stream([str(python), "-u", str(repo / "scripts/train_ar_resync_gain.py"), str(trace), str(head)],
                    cwd=repo, env=env, log_path=log_dir / "ar_resync_train.log")
    if result:
        raise SystemExit(result)
    command = [str(python), "-u", str(repo / "scripts/benchmark_ar_resync_e2e.py"),
               "--revision", revision, "--head", str(head), "--output", str(output), "--count", str(count)]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    result = stream(command, cwd=repo, env=env, log_path=log_dir / "ar_resync_run.log")
    if result:
        raise SystemExit(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    launch(args.revision, args.trace, args.output, args.count, args.resume)
