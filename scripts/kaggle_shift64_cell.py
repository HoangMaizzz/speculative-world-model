"""One-cell oracle shifted-anchor B64 replay from the completed guarded B64 ZIP."""
import json
import os
from pathlib import Path
import runpy
import subprocess


def launch(revision, input_root="/kaggle/input", resume_root=None, limit=None):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected immutable Git commit")
    if not Path(input_root).exists():
        raise FileNotFoundError("Attach the completed long64_checkpoint.zip as Kaggle input first")
    os.environ.update(HF_HOME="/tmp/suffix-hf-cache", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_TELEMETRY="1", PYTHONUNBUFFERED="1", MPLCONFIGDIR="/tmp/suffix-mpl")
    work = Path("/kaggle/working")
    repo = work/("shift64-src-"+revision[:12])
    url = "https://github.com/HoangMaizzz/speculative-world-model.git"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_TRACE="0", GIT_CURL_VERBOSE="0")
    if not repo.exists():
        subprocess.run(["git", "clone", "--single-branch", "--branch", "codex/shifted-anchor-b64", url, str(repo)], env=env, check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=repo, env=env, check=True)
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != revision
            or subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip() != url):
        raise ValueError("Existing source differs; preserved")
    helpers = runpy.run_path(str(repo/"scripts/kaggle_single_cell.py"), run_name="feature_install_helpers")
    stream = runpy.run_path(str(repo/"scripts/kaggle_extension_cell.py"), run_name="feature_stream_helpers")["run_streamed"]
    helpers["check_gpus"]()
    venv, setup_log = Path("/tmp/suffix-venv"), work/"shift64_setup.log"
    python = helpers["ensure_environment"](venv, setup_log)
    stamp = venv/"suffix-installed-commit"
    if not stamp.exists() or stamp.read_text() != revision:
        setup = helpers["setup_command"]
        setup([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
               "torch==2.9.0", "--index-url", "https://download.pytorch.org/whl/cu126"], setup_log)
        setup([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
               "-r", str(repo/"requirements-kaggle.txt")], setup_log)
        setup([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], setup_log)
        stamp.write_text(revision)
    child_env = helpers["child_environment"]()
    output = work/("shift64_"+revision[:12]+(f"_smoke{limit}" if limit else ""))
    args = [str(python), "-u", "-m", "suffix_lab.shift64", "--input-root", str(input_root),
            "--output", str(output), "--revision", revision]
    if resume_root:
        args += ["--resume", str(resume_root)]
    if limit is not None:
        args += ["--limit", str(limit)]
    print("[shift64] input:", input_root, "output:", output, flush=True)
    code = stream(args, cwd=repo, env=child_env, log_path=work/"shift64_run.log")
    from IPython.display import display, FileLink
    for name in ("summary.json", "execution_status.json"):
        if (output/name).is_file():
            print((output/name).read_text(), flush=True)
    archive = output/"shift64_checkpoint.zip"
    if archive.is_file():
        display(FileLink(str(archive)))
    if code:
        raise RuntimeError("Shift64 failed. Inspect the run log and any checkpoint evidence.")


if __name__ == "__main__":
    launch(globals()["SHIFT64_REVISION"], globals().get("SHIFT64_INPUT_ROOT", "/kaggle/input"),
           globals().get("SHIFT64_RESUME"), globals().get("SHIFT64_LIMIT"))
