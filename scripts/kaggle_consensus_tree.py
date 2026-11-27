"""Launch the paired consensus tree benchmark from a pinned Kaggle checkout."""
import argparse
import os
from pathlib import Path
import subprocess

from kaggle_extension_cell import run_streamed
from kaggle_single_cell import check_gpus, child_environment, ensure_environment, setup_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--per-domain", type=int, default=50)
    parser.add_argument("--max-new", type=int, default=256)
    args = parser.parse_args()
    check_gpus()
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git","rev-parse","HEAD"], cwd=repo, text=True).strip()
    parent = Path("/kaggle/working")
    out = parent / f"consensus_tree_{args.per_domain}x2_{args.max_new}_{revision[:10]}"
    out.mkdir(parents=True, exist_ok=True)
    log = out/"setup.log"
    venv = Path("/tmp/retrieval-alignment-venv")
    python = ensure_environment(venv, log)
    marker = venv/"installed-revision"
    if not marker.exists() or marker.read_text() != revision:
        setup_command([str(python),"-m","pip","install","torch==2.9.0","--index-url","https://download.pytorch.org/whl/cu126"], log)
        setup_command([str(python),"-m","pip","install","-r",str(repo/"requirements-kaggle.txt")], log)
        setup_command([str(python),"-m","pip","install","--no-deps","-e",str(repo)], log)
        marker.write_text(revision)
    env = child_environment(os.environ)
    env.setdefault("HF_HOME", "/tmp/retrieval-alignment-hf")
    env["OMP_NUM_THREADS"] = "2"
    status = run_streamed([str(python), "-u", str(repo/"scripts/benchmark_consensus_tree.py"),
                          "--input",args.input,"--per-domain",str(args.per_domain),"--max-new",str(args.max_new),
                          "--revision",revision,"--output",str(out)], cwd=repo, env=env, log_path=out/"run.log")
    if status:
        raise SystemExit(status)
    print(out/"consensus_tree_results.zip", flush=True)


if __name__ == "__main__":
    main()
