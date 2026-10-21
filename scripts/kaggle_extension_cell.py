"""Standalone Kaggle setup for the pinned dLLM -> AR mini-extension runtime."""
import os
from pathlib import Path
import runpy
import subprocess
import sys
import queue
import threading
import time


def run_streamed(args, *, cwd, env, log_path, heartbeat_seconds=15):
    """Explicitly relay child stdout/stderr into Jupyter and preserve a log."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as log, subprocess.Popen(
            args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        messages = queue.Queue()
        def reader():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        last_line = "waiting for Python child startup"
        try:
            while True:
                try:
                    line = messages.get(timeout=heartbeat_seconds)
                except queue.Empty:
                    line = f"[launcher] Child running ({time.monotonic()-started:.0f}s). Last output: {last_line}\n"
                    print(line, end="", flush=True)
                    log.write(line); log.flush()
                    continue
                if line is None:
                    break
                print(line, end="", flush=True)
                log.write(line); log.flush()
                last_line = line.strip()[-240:] or last_line
            thread.join()
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            raise


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
    run_log = work/"extension_run.log"
    print("Live output and diagnostic log:", run_log, flush=True)
    code = run_streamed([str(python), "-u", "-m", "suffix_lab.extension", "--checkpoint-root", str(checkpoint_root),
                         "--output", str(output), "--revision", revision],
                        cwd=repo, env=child_env, log_path=run_log)
    from IPython.display import display, Markdown, FileLink
    if (output/"REPORT.md").exists():
        display(Markdown((output/"REPORT.md").read_text()))
    archive = output/"extension_checkpoint.zip"
    if archive.exists():
        print("ZIP:", archive, flush=True)
        display(FileLink(str(archive)))
    if code:
        raise RuntimeError("Mini-extension failed; inspect the actual error above. Partial ZIP was preserved when possible.")


if __name__ == "__main__":
    launch(globals()["SUFFIX_EXTENSION_REVISION"], globals().get("SUFFIX_EXTENSION_INPUT", "/kaggle/input/datasets/ainzkhail/checkpt"))
