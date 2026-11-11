"""One-cell launcher for the frozen Soft Anchor pilot on Kaggle T4 x2."""

from hashlib import sha256
import os
from pathlib import Path
import runpy
import subprocess
from zipfile import BadZipFile, ZipFile


URL = "https://github.com/HoangMaizzz/speculative-world-model.git"
BRANCH = "codex/soft-anchor-pilot"
PLAN_MEMBER = "soft_anchor_plan.json"
PLAN_ZIP = "soft_anchor_pilot_input.zip"
PLAN_SHA256 = "470499c62213b56a427dcbb51041c7368c21286d487abba7885f650d2fce7acf"


def input_matches(path):
    try:
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                if archive.getinfo(PLAN_MEMBER).file_size > 2_000_000:
                    return False
                raw = archive.read(PLAN_MEMBER)
        else:
            if path.stat().st_size > 2_000_000:
                return False
            raw = path.read_bytes()
    except (BadZipFile, KeyError, OSError):
        return False
    return sha256(raw).hexdigest() == PLAN_SHA256


def find_input(hint=None):
    if hint:
        path = Path(hint)
        if path.is_file() and input_matches(path):
            return path
        if not path.is_dir():
            raise ValueError(f"Input path missing or fails checksum: {path}")
        root = path
    else:
        root = Path("/kaggle/input")
    for name in (PLAN_ZIP, PLAN_MEMBER):
        for path in sorted(root.rglob(name)):
            if path.is_file() and input_matches(path):
                return path
    raise FileNotFoundError(f"Attach {PLAN_ZIP} or its extracted {PLAN_MEMBER}. Searched {root}.")


def launch(revision, input_path=None, mode="pilot"):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected an immutable lowercase Git SHA")
    if mode not in {"pilot", "smoke"}:
        raise ValueError("MODE must be pilot or smoke")
    plan = find_input(input_path)
    print("Verified input:", plan, flush=True)
    work = Path("/kaggle/working")
    repo = work / f"soft-anchor-src-{revision[:12]}"
    output = work / f"soft_anchor_{mode}_{revision[:12]}"
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
    helpers = runpy.run_path(str(repo / "scripts/kaggle_single_cell.py"), run_name="soft_anchor_installer")
    stream = runpy.run_path(str(repo / "scripts/kaggle_extension_cell.py"), run_name="soft_anchor_stream")["run_streamed"]
    helpers["check_gpus"]()
    venv = Path("/tmp/soft-anchor-venv")
    setup_log = work / "soft_anchor_setup.log"
    python = helpers["ensure_environment"](venv, setup_log)
    stamp = venv / "soft-anchor-installed-revision"
    if not stamp.exists() or stamp.read_text() != revision:
        setup = helpers["setup_command"]
        setup([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off", "torch==2.9.0",
               "--index-url", "https://download.pytorch.org/whl/cu126"], setup_log)
        setup([str(python), "-m", "pip", "install", "--no-cache-dir", "--progress-bar", "off", "-r",
               str(repo / "requirements-kaggle.txt")], setup_log)
        setup([str(python), "-m", "pip", "install", "--no-deps", "-e", str(repo)], setup_log)
        stamp.write_text(revision)
    command = [str(python), "-u", str(repo / "scripts/run_soft_anchor_pilot.py"), "--input", str(plan),
               "--output", str(output), "--revision", revision]
    if mode == "smoke":
        command.append("--smoke")
    print("Output:", output, flush=True)
    code = stream(command, cwd=repo, env=helpers["child_environment"](), log_path=work / f"soft_anchor_{mode}.log")
    archive = output / "soft_anchor_pilot_results.zip"
    if (output / "execution_status.json").is_file():
        print((output / "execution_status.json").read_text())
    if archive.is_file():
        from IPython.display import FileLink, display
        print("Results ZIP:", archive, flush=True)
        display(FileLink(str(archive)))
    if code:
        raise RuntimeError(f"Pilot exited with code {code}. See the run log and any saved results ZIP.")


if __name__ == "__main__":
    launch(globals()["SOFT_ANCHOR_REVISION"], globals().get("SOFT_ANCHOR_INPUT"), globals().get("SOFT_ANCHOR_MODE", "pilot"))
