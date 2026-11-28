"""Launch the online-corpus benchmark from a pinned Kaggle checkout."""
import argparse
import os
from pathlib import Path
import subprocess

from benchmark_online_corpus import ARCHIVE, load_data
from kaggle_extension_cell import run_streamed
from kaggle_single_cell import check_gpus, child_environment, ensure_environment, setup_command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="/kaggle/input/datasets/ainzkhail/datatest")
    ap.add_argument("--per-domain", type=int, choices=range(1, 101), default=100)
    ap.add_argument("--max-new", type=int, choices=range(1, 257), default=256)
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()
    load_data(args.input)
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    check_gpus()
    repo = Path(__file__).resolve().parents[1]
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    out = Path("/kaggle/working") / f"online_corpus_{args.per_domain}x2_{args.max_new}_{rev[:10]}"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "setup.log"
    venv = Path("/tmp/retrieval-alignment-venv")
    python = ensure_environment(venv, log)
    marker = venv / "installed-revision"
    if not marker.is_file() or marker.read_text() != rev:
        setup_command([str(python), "-m", "pip", "install", "torch==2.9.0",
                       "--index-url", "https://download.pytorch.org/whl/cu126"], log)
        setup_command([str(python), "-m", "pip", "install", "-r", str(repo/"requirements-kaggle.txt")], log)
        setup_command([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], log)
        marker.write_text(rev)
    env = child_environment(os.environ)
    env.setdefault("HF_HOME", "/tmp/retrieval-alignment-hf")
    env["OMP_NUM_THREADS"] = "2"
    cmd = [str(python), "-u", str(repo/"scripts/benchmark_online_corpus.py"), "--input", args.input,
           "--output", str(out), "--revision", rev, "--per-domain", str(args.per_domain),
           "--max-new", str(args.max_new)]
    if args.resume:
        cmd.extend(["--resume", str(args.resume)])
    code = run_streamed(cmd, cwd=repo, env=env, log_path=out/"run.log")
    if code:
        raise SystemExit(code)
    print(out / ARCHIVE, flush=True)


if __name__ == "__main__":
    main()
