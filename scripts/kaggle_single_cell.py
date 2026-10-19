"""Public, revision-pinned Kaggle launcher. No notebook variables are required."""
from pathlib import Path
import json
import os
import subprocess
import sys

URL = "https://github.com/HoangMaizzz/speculative-world-model.git"
BRANCH = "codex/suffix-top1-math-gsm8k"


def child_environment(source=None):
    """Keep credentials/CUDA settings, excluding notebook Python startup hooks."""
    env = dict(os.environ if source is None else source)
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    # The venv exports PNGs; the notebook renders the resulting files separately.
    env["MPLBACKEND"] = "Agg"
    return env


def setup_command(args, log_path):
    """Print and preserve setup output, including the real venv/pip failure."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Setup:", " ".join(map(str, args)), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ "+" ".join(map(str, args))+"\n")
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1,
                              env=child_environment()) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f"Setup failed (exit {code}). Full log: {log_path}. See the actual error printed above.")


def ensure_environment(venv, log_path, run_setup=setup_command, probe=subprocess.run, base_python=sys.executable):
    """Repair partial venvs without invoking ensurepip or changing system packages."""
    venv = Path(venv)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists() or not (venv/"pyvenv.cfg").exists():
        # Python 3.13 in some Kaggle images has venv but omits ensurepip.
        run_setup([base_python, "-m", "venv", "--without-pip", str(venv)], log_path)
    if probe([str(python), "-c", "import sys; print(sys.version)"],
             capture_output=True, text=True, env=child_environment()).returncode:
        raise RuntimeError(f"Virtual environment interpreter is not usable: {python}; no files were deleted.")
    if probe([str(python), "-m", "pip", "--version"], capture_output=True, text=True,
             env=child_environment()).returncode:
        host = probe([base_python, "-m", "pip", "--version"], capture_output=True, text=True,
                     env=child_environment())
        if host.returncode:
            raise RuntimeError("Notebook Python has no pip. Capture this output and the Kaggle image/Python version; no system packages were changed.")
        # pip --python explicitly supports environments where pip is absent.
        run_setup([base_python, "-m", "pip", "--python", str(python), "install",
                   "--no-cache-dir", "pip==25.2"], log_path)
    run_setup([str(python), "-m", "pip", "--version"], log_path)
    return python


def check_gpus():
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Kaggle Settings: select GPU T4 x2 before running this cell.")
    for index in range(2):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU{index}: {props.name}, {props.total_memory/2**30:.1f} GiB", flush=True)
        if props.total_memory < 14*2**30:
            raise RuntimeError("Each GPU needs approximately 16 GB VRAM.")


def output_root(work, mode, revision):
    base = Path(work) / ("suffix_smoke" if mode == "smoke" else "suffix_pilot")
    manifest, status = base/"run-manifest.json", base/"execution_status.json"
    previous = None
    if manifest.exists():
        previous = json.loads(manifest.read_text()).get("environment", {}).get("commit")
    elif status.exists():
        # A failed preflight has no run-manifest yet, but must also be preserved.
        previous = json.loads(status.read_text()).get("code_revision")
    if previous is not None and previous != revision:
        base = base.with_name(base.name+"_"+revision[:12])
        print("Older output preserved. This revision uses:", base, flush=True)
    return base


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
    setup_log = work / ("suffix_setup_"+mode+".log")
    print("Notebook Python:", sys.executable, sys.version, "Setup log:", setup_log, flush=True)
    python = ensure_environment(venv, setup_log)
    stamp = venv / "suffix-installed-commit"
    if not stamp.exists() or stamp.read_text() != revision:
        setup_command([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
                       "torch==2.9.0", "--index-url", "https://download.pytorch.org/whl/cu126"], setup_log)
        setup_command([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off",
                       "-r", str(repo/"requirements-kaggle.txt")], setup_log)
        setup_command([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], setup_log)
        stamp.write_text(revision)
    subprocess.run([str(python), "-c", "import torch; assert torch.cuda.device_count() >= 2; print(torch.__version__, torch.version.cuda)"],
                   check=True, env=child_environment())
    # Distinct output roots prevent a new code revision from mixing older results.
    base = output_root(work, mode, revision)
    assets = Path("/tmp/suffix-assets") / (mode+"-"+revision[:12])
    result = subprocess.run([str(python), "-u", str(repo/"scripts/kaggle_run.py"),
                             "--mode", mode, "--revision", revision,
                             "--assets", str(assets), "--output", str(base)], cwd=repo,
                            env=child_environment())
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
