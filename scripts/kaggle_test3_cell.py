"""Single-cell installer/streaming launcher for Test3 Oracle-All-Correct."""
import json
import os
from pathlib import Path
import runpy
import subprocess


def launch(revision, checkpoint_root):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected immutable lowercase Git revision")
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Select GPU T4 x2 and Internet ON")
    if not Path(checkpoint_root).is_dir():
        raise FileNotFoundError("Unpacked pilot checkpoint missing: "+str(checkpoint_root))
    os.environ.update(HF_HOME="/tmp/suffix-hf-cache", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_TELEMETRY="1", MPLCONFIGDIR="/tmp/suffix-mpl", PYTHONUNBUFFERED="1")
    work = Path("/kaggle/working")
    repo = work/("test3-src-"+revision[:12])
    url = "https://github.com/HoangMaizzz/speculative-world-model.git"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_TRACE="0", GIT_CURL_VERBOSE="0")
    if not repo.exists():
        subprocess.run(["git", "clone", "--branch", "codex/suffix-top1-math-gsm8k", "--single-branch", url, str(repo)], env=env, check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=repo, env=env, check=True)
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if remote != url or head != revision:
        raise RuntimeError("Existing checkout differs; it was preserved")
    installer = runpy.run_path(str(repo/"scripts/kaggle_single_cell.py"), run_name="test3_installer_helpers")
    stream = runpy.run_path(str(repo/"scripts/kaggle_extension_cell.py"), run_name="test3_stream_helpers")["run_streamed"]
    installer["check_gpus"]()
    venv, setup_log = Path("/tmp/suffix-venv"), work/"test3_setup.log"
    python = installer["ensure_environment"](venv, setup_log)
    stamp = venv/"suffix-installed-commit"
    if not stamp.exists() or stamp.read_text() != revision:
        setup = installer["setup_command"]
        setup([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
               "torch==2.9.0", "--index-url", "https://download.pytorch.org/whl/cu126"], setup_log)
        setup([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
               "-r", str(repo/"requirements-kaggle.txt")], setup_log)
        setup([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], setup_log)
        stamp.write_text(revision)
    output = work/"test3_oracle_all"
    manifest = output/"test3-manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["code_revision"] != revision:
        output = work/("test3_oracle_all_"+revision[:12])
    child_env = installer["child_environment"]()
    child_env.pop("SUFFIX_RESUME_ROOT", None)
    print("Test3 dLLM -> AR; B24 -> correction+23; ALL old-correct KEEP; 1 token/unmask forward", flush=True)
    print("Input:", checkpoint_root, "Output:", output, flush=True)
    print("Stage1: native B24/Fresh23 screening; Stage2: 10 frozen events x6 arms", flush=True)
    code = stream([str(python), "-u", "-m", "suffix_lab.test3", "--checkpoint-root", str(checkpoint_root),
                   "--output", str(output), "--revision", revision], cwd=repo, env=child_env, log_path=work/"test3_run.log")
    from IPython.display import display, Markdown, FileLink
    if (output/"report.md").exists():
        display(Markdown((output/"report.md").read_text()))
    archive = output/"test3_checkpoint.zip"
    if archive.exists():
        print("ZIP:", archive, flush=True)
        display(FileLink(str(archive)))
    if code:
        raise RuntimeError("Test3 did not complete. See the actual error, screening shortfall or audit evidence above; partial ZIP is preserved when possible.")


if __name__ == "__main__":
    launch(globals()["SUFFIX_TEST3_REVISION"], globals().get("SUFFIX_TEST3_INPUT", "/kaggle/input/datasets/ainzkhail/checkpt"))
