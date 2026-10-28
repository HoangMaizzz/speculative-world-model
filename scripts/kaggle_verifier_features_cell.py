"""One-cell full verifier feature collection from the completed guarded B64 ZIP."""
import json
import os
from pathlib import Path
import runpy
import subprocess


def launch(revision, input_root="/kaggle/input", resume_root=None, limit=None, micro_probe_steps=0):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected immutable Git commit")
    if not Path(input_root).exists():
        raise FileNotFoundError("Attach the completed long64_checkpoint.zip as Kaggle input first")
    os.environ.update(HF_HOME="/tmp/suffix-hf-cache", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_DISABLE_TELEMETRY="1", PYTHONUNBUFFERED="1", MPLCONFIGDIR="/tmp/suffix-mpl")
    work = Path("/kaggle/working")
    repo = work/("verifier-features-src-"+revision[:12])
    url = "https://github.com/HoangMaizzz/speculative-world-model.git"
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_TRACE="0", GIT_CURL_VERBOSE="0")
    if not repo.exists():
        subprocess.run(["git", "clone", "--single-branch", "--branch", "codex/suffix-top1-math-gsm8k", url, str(repo)], env=env, check=True)
        subprocess.run(["git", "checkout", "--detach", revision], cwd=repo, env=env, check=True)
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != revision
            or subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip() != url):
        raise ValueError("Existing source differs; preserved")
    helpers = runpy.run_path(str(repo/"scripts/kaggle_single_cell.py"), run_name="feature_install_helpers")
    stream = runpy.run_path(str(repo/"scripts/kaggle_extension_cell.py"), run_name="feature_stream_helpers")["run_streamed"]
    helpers["check_gpus"]()
    venv, setup_log = Path("/tmp/suffix-venv"), work/"verifier_features_setup.log"
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
    code = stream([str(python), "-u", "-m", "unittest", "discover", "-s", str(repo/"tests"),
                   "-p", "test_verifier_features*.py", "-q"], cwd=repo, env=child_env,
                   log_path=work/"verifier_features_checks.log")
    if code:
        raise RuntimeError("CPU collector checks failed; pretrained collection not started")
    output = work/("verifier_features_"+revision[:12]+(f"_probe{micro_probe_steps}" if micro_probe_steps else ""))
    args = [str(python), "-u", "-m", "suffix_lab.verifier_features.run_collection",
            "--input-root", str(input_root), "--output", str(output), "--revision", revision,
            "--micro-probe-steps", str(micro_probe_steps)]
    if resume_root:
        args += ["--resume-root", str(resume_root)]
    if limit is not None:
        args += ["--limit", str(limit)]
    print("[features] Full 3847 events; mandatory smoke20 -> pilot500 milestone -> remaining events", flush=True)
    print("[features] Qwen FP16 split across 2 GPUs; no drafter; probes:", micro_probe_steps, flush=True)
    print("[features] Input:", input_root, "Output:", output, flush=True)
    code = stream(args, cwd=repo, env=child_env, log_path=work/"verifier_features_run.log")
    from IPython.display import display, FileLink, Markdown
    if (output/"report.md").exists():
        display(Markdown((output/"report.md").read_text()))
    archive = output/"verifier_features_checkpoint.zip"
    if archive.exists():
        print("ZIP:", archive, flush=True)
        display(FileLink(str(archive)))
    else:
        print("Available shard ZIPs:", output/"shards", flush=True)
        for part in sorted((output/"shards").glob("part-*.zip")):
            display(FileLink(str(part)))
    if (output/"execution_status.json").exists():
        print((output/"execution_status.json").read_text(), flush=True)
    if code:
        raise RuntimeError("Feature collection failed; completed shards and audit evidence were preserved")


if __name__ == "__main__":
    launch(globals()["FEATURE_REVISION"], globals().get("FEATURE_INPUT_ROOT", "/kaggle/input"),
           globals().get("FEATURE_RESUME_ROOT"), globals().get("FEATURE_LIMIT"),
           globals().get("FEATURE_MICRO_PROBE_STEPS", 0))
