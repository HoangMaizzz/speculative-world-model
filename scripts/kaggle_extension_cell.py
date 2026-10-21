"""Standalone Kaggle setup for the pinned dLLM -> AR mini-extension runtime."""
import os
from pathlib import Path
import runpy
import subprocess
import sys


def launch(revision, checkpoint_root):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected an immutable lowercase Git commit SHA")
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Kaggle Settings: GPU T4 x2 and Internet ON")
    if not Path(checkpoint_root).is_dir():
        raise FileNotFoundError("Unpacked checkpoint input is missing: "+str(checkpoint_root))
    os.environ.update(HF_HOME="/tmp/suffix-hf-cache", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_TELEMETRY="1", MPLCONFIGDIR="/tmp/suffix-mpl", PYTHONUNBUFFERED="1")
    work = Path("/kaggle/working")
    repo = work/("extension-src-"+revision[:12])
    url = "https://github.com/HoangMaizzz/speculative-world-model.git"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_TRACE="0", GIT_CURL_VERBOSE="0")
    if not repo.exists():
        subprocess.run(["git", "clone", "--branch", "codex/suffix-top1-math-gsm8k", "--single-branch", url, str(repo)], env=env, check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=repo, env=env, check=True)
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if remote != url or head != revision:
        raise RuntimeError("Existing source differs; it was preserved. Use the matching revision.")
    # Reuse the tested ensurepip-free installer; do not invoke its pilot main.
    installer = runpy.run_path(str(repo/"scripts/kaggle_single_cell.py"), run_name="mini_installer_helpers")
    installer["check_gpus"]()
    log = work/"extension_setup.log"
    venv = Path("/tmp/suffix-venv")
    python = installer["ensure_environment"](venv, log)
    stamp = venv/"suffix-installed-commit"
    if not stamp.exists() or stamp.read_text() != revision:
        setup = installer["setup_command"]
        setup([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
               "torch==2.9.0", "--index-url", "https://download.pytorch.org/whl/cu126"], log)
        setup([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
               "-r", str(repo/"requirements-kaggle.txt")], log)
        setup([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], log)
        stamp.write_text(revision)
    output = work/"extension_mini"
    manifest = output/"extension-manifest.json"
    if manifest.exists():
        import json
        if json.loads(manifest.read_text())["code_revision"] != revision:
            output = work/("extension_mini_"+revision[:12])
    print("dLLM -> AR only; strict 1 unmask/forward. Input:", checkpoint_root, "Output:", output, flush=True)
    child_env = installer["child_environment"]()
    child_env.pop("SUFFIX_RESUME_ROOT", None)
    result = subprocess.run([str(python), "-u", "-m", "suffix_lab.extension", "--checkpoint-root", str(checkpoint_root),
                            "--output", str(output), "--revision", revision], cwd=repo, env=child_env)
    from IPython.display import display, Markdown, FileLink
    if (output/"REPORT.md").exists():
        display(Markdown((output/"REPORT.md").read_text()))
    archive = output/"extension_checkpoint.zip"
    if archive.exists():
        print("ZIP:", archive, flush=True)
        display(FileLink(str(archive)))
    if result.returncode:
        raise RuntimeError("Mini-extension failed; inspect the actual error above. Partial ZIP was preserved when possible.")


if __name__ == "__main__":
    launch(globals()["SUFFIX_EXTENSION_REVISION"], globals().get("SUFFIX_EXTENSION_INPUT", "/kaggle/input/datasets/ainzkhail/checkpt"))
