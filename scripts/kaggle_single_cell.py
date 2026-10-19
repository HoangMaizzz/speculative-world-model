"""Public, revision-pinned Kaggle launcher. No notebook variables are required."""
from pathlib import Path
import json
import os
import subprocess
import sys

URL = "https://github.com/HoangMaizzz/speculative-world-model.git"
BRANCH = "codex/suffix-top1-math-gsm8k"


def check_gpus():
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Kaggle Settings: select GPU T4 x2 before running this cell.")
    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU{index}: {props.name}, {props.total_memory/2**30:.1f} GiB", flush=True)
        if props.total_memory < 14*2**30:
            raise RuntimeError("Each GPU needs approximately 16 GB VRAM.")


def launch(mode, revision):
    if mode not in ("smoke", "pilot") or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected smoke/pilot and a full lowercase commit SHA")
    check_gpus()
    os.environ.update(HF_HOME="/tmp/suffix-hf-cache", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_TELEMETRY="1", MPLCONFIGDIR="/tmp/suffix-mpl", PYTHONUNBUFFERED="1")
    # Public weights do not require this optional secret.
    if not os.environ.get("HF_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
            if token:
                os.environ["HF_TOKEN"] = token
            del token
        except Exception:
            pass
    work = Path("/kaggle/working")
    repo = work / ("suffix-src-"+revision[:12])
    git_env = os.environ.copy()
    git_env.update(GIT_TERMINAL_PROMPT="0", GIT_TRACE="0", GIT_CURL_VERBOSE="0")
    if not repo.exists():
        subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", URL, str(repo)],
                       env=git_env, check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=repo, env=git_env, check=True)
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if remote != URL or head != revision:
        raise RuntimeError("Existing checkout differs; it was preserved. Use the matching launcher revision.")
    print("Experiment source:", head, "mode:", mode, flush=True)
    venv = Path("/tmp/suffix-venv")
    python = venv / "bin/python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    stamp = venv / "suffix-installed-commit"
    if not stamp.exists() or stamp.read_text() != revision:
        subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
                        "torch==2.9.0", "--index-url", "https://download.pytorch.org/whl/cu126"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
                        "-r", str(repo/"requirements-kaggle.txt")], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], check=True)
        stamp.write_text(revision)
    subprocess.run([str(python), "-c", "import torch; assert torch.cuda.device_count() >= 2; print(torch.__version__, torch.version.cuda)"], check=True)
    # Distinct output roots prevent a new code revision from mixing older results.
    base = work / ("suffix_smoke" if mode == "smoke" else "suffix_pilot")
    existing = base / "run-manifest.json"
    if existing.exists() and json.loads(existing.read_text()).get("environment", {}).get("commit") != revision:
        base = work / (base.name+"_"+revision[:12])
        print("Older output preserved. This revision uses:", base, flush=True)
    assets = Path("/tmp/suffix-assets") / (mode+"-"+revision[:12])
    result = subprocess.run([str(python), "-u", str(repo/"scripts/kaggle_run.py"),
                             "--mode", mode, "--revision", revision,
                             "--assets", str(assets), "--output", str(base)], cwd=repo)
    try:
        from IPython.display import display, FileLink, Markdown, Image
        report = base / "analysis/REPORT.md"
        if report.exists():
            display(Markdown(report.read_text()))
            for figure in sorted((base/"analysis").glob("*.png")):
                display(Image(filename=str(figure)))
        archive = base / "suffix_checkpoint.zip"
        if archive.exists():
            print("Checkpoint ZIP:", archive, flush=True)
            display(FileLink(str(archive)))
    finally:
        if result.returncode:
            raise RuntimeError(f"{mode} failed (exit {result.returncode}); inspect logs above. Available checkpoint files were preserved.")


if __name__ == "__main__":
    launch(globals()["SUFFIX_MODE"], globals()["SUFFIX_CODE_REVISION"])
