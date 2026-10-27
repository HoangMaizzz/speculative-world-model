"""One-cell pinned installer for native draft64 SD; no previous ZIP is required."""
import json
import os
from pathlib import Path
import runpy
import subprocess


def launch(revision, count=100, max_new_tokens=256, resume_root=None):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected immutable lowercase Git SHA")
    if not 1 <= count <= 100 or not 1 <= max_new_tokens <= 1024:
        raise ValueError("Invalid count/output cap")
    os.environ.update(HF_HOME="/tmp/suffix-hf-cache", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_TELEMETRY="1", MPLCONFIGDIR="/tmp/suffix-mpl", PYTHONUNBUFFERED="1")
    work = Path("/kaggle/working")
    repo = work/("long64-src-"+revision[:12])
    url = "https://github.com/HoangMaizzz/speculative-world-model.git"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_TRACE="0", GIT_CURL_VERBOSE="0")
    if not repo.exists():
        subprocess.run(["git", "clone", "--branch", "codex/suffix-top1-math-gsm8k", "--single-branch", url, str(repo)], env=env, check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=repo, env=env, check=True)
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if remote != url or head != revision:
        raise RuntimeError("Existing checkout differs; preserved without modification")
    helpers = runpy.run_path(str(repo/"scripts/kaggle_single_cell.py"), run_name="long64_installer")
    stream = runpy.run_path(str(repo/"scripts/kaggle_extension_cell.py"), run_name="long64_stream")["run_streamed"]
    helpers["check_gpus"]()
    venv, setup_log = Path("/tmp/suffix-venv"), work/"long64_setup.log"
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
    print("[long64] CPU correctness/resume tests before pretrained models", flush=True)
    code = stream([str(python), "-u", "-m", "unittest", "discover", "-s", str(repo/"tests"),
                   "-p", "test_long6[04].py", "-q"], cwd=repo, env=child_env, log_path=work/"long64_checks.log")
    if code:
        raise RuntimeError("B64 CPU checks failed; pretrained experiment was not started")
    output = work/"long64"
    manifest = output/"long64-manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if (old["code_revision"] != revision or old["config"]["prompts_per_dataset"] != count
                or old["config"]["max_new_tokens"] != max_new_tokens):
            output = work/f"long64_{revision[:12]}_{count}_{max_new_tokens}"
    print(f"[long64] {count} MATH + {count} GSM8K; draft64; strict1MASK/forward; FP16 two GPUs", flush=True)
    print("[long64] Output:", output, "resume:", resume_root or "local checkpoint / fresh", flush=True)
    args = [str(python), "-u", "-m", "suffix_lab.long64", "--output", str(output), "--revision", revision,
            "--count", str(count), "--max-new-tokens", str(max_new_tokens)]
    if resume_root:
        args += ["--resume-root", str(resume_root)]
    code = stream(args, cwd=repo, env=child_env, log_path=work/"long64_run.log")
    from IPython.display import display, FileLink, Markdown
    if (output/"report.md").exists():
        display(Markdown((output/"report.md").read_text()))
    archive = output/"long64_checkpoint.zip"
    if archive.exists():
        print("ZIP:", archive, flush=True)
        display(FileLink(str(archive)))
    if (output/"execution_status.json").exists():
        status = json.loads((output/"execution_status.json").read_text())
        print(json.dumps(status, indent=2), flush=True)
        if status["status"] == "budget_checkpoint":
            print("Soft six-hour budget: upload this B64 ZIP for resume, or rerun the same cell in this session.", flush=True)
    if code:
        raise RuntimeError("B64 failed; raw evidence and available ZIP are preserved; inspect logs above")


if __name__ == "__main__":
    launch(globals()["LONG64_REVISION"], globals().get("LONG64_COUNT", 100),
           globals().get("LONG64_MAX_NEW_TOKENS", 256), globals().get("LONG64_RESUME"))
