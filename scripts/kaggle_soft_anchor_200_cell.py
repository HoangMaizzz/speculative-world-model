"""Pinned one-cell launcher for the fresh 100+100 Soft Anchor evaluation."""

from hashlib import sha256
import os
from pathlib import Path
import runpy
import subprocess


URL = "https://github.com/HoangMaizzz/speculative-world-model.git"
BRANCH = "codex/soft-anchor-200"
ASSET_NAME = "soft_anchor_200_input.zip"
ASSET_SHA = "1b7e9c4e1ab4dd995ba310c262adefae1e4b54eff54d85f5bd849829857a92e9"


def find_input(hint=None):
    root = Path(hint) if hint else Path("/kaggle/input")
    candidates = [root] if root.is_file() else sorted(root.rglob(ASSET_NAME))
    for path in candidates:
        if path.is_file() and path.stat().st_size <= 2_000_000 and sha256(path.read_bytes()).hexdigest() == ASSET_SHA:
            return path
    raise FileNotFoundError(f"Attach the original {ASSET_NAME}. Searched {root}.")


def launch(revision, input_path=None, mode="full", resume=None):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision) or mode not in {"full", "smoke"}:
        raise ValueError("Expected immutable Git SHA and mode full or smoke")
    assets = find_input(input_path)
    print("Verified input:", assets, flush=True)
    work = Path("/kaggle/working")
    repo = work / f"soft-anchor-200-src-{revision[:12]}"
    output = work / f"soft_anchor_200_{mode}_{revision[:12]}"
    os.environ.update(HF_HOME="/tmp/suffix-hf-cache", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_TELEMETRY="1", PYTHONUNBUFFERED="1")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_TRACE="0", GIT_CURL_VERBOSE="0")
    if not repo.exists():
        subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", URL, str(repo)], env=env, check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=repo, env=env, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip()
    if head != revision or origin != URL:
        raise RuntimeError("Existing checkout does not match the pinned source")
    helpers = runpy.run_path(str(repo / "scripts/kaggle_single_cell.py"), run_name="soft_200_installer")
    stream = runpy.run_path(str(repo / "scripts/kaggle_extension_cell.py"), run_name="soft_200_stream")["run_streamed"]
    helpers["check_gpus"]()
    venv = Path("/tmp/soft-anchor-venv")
    log = work / "soft_anchor_200_setup.log"
    python = helpers["ensure_environment"](venv, log)
    stamp = venv / "soft-anchor-installed-revision"
    if not stamp.exists() or stamp.read_text() != revision:
        setup = helpers["setup_command"]
        setup([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off", "torch==2.9.0",
               "--index-url", "https://download.pytorch.org/whl/cu126"], log)
        setup([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off", "-r",
               str(repo / "requirements-kaggle.txt")], log)
        setup([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], log)
        stamp.write_text(revision)
    command = [str(python), "-u", str(repo / "scripts/run_soft_anchor_200.py"), "--input", str(assets),
               "--output", str(output), "--revision", revision, "--mode", mode]
    if resume:
        command.extend(["--resume", str(resume)])
    print("Output:", output, flush=True)
    code = stream(command, cwd=repo, env=helpers["child_environment"](), log_path=work / f"soft_anchor_200_{mode}.log")
    status = output / "execution_status.json"
    archive = output / "soft_anchor_200_results.zip"
    if status.is_file():
        print(status.read_text())
    if archive.is_file():
        from IPython.display import FileLink, display
        print("Results ZIP:", archive, flush=True)
        display(FileLink(str(archive)))
    if code:
        raise RuntimeError(f"Evaluation exited with code {code}. See the log and saved ZIP.")


if __name__ == "__main__":
    launch(globals()["SOFT_200_REVISION"], globals().get("SOFT_200_INPUT"),
           globals().get("SOFT_200_MODE", "full"), globals().get("SOFT_200_RESUME"))
