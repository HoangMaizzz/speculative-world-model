"""One source for the copyable cells and the downloadable Kaggle notebook."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
CELLS = [
'''# 1. GPU check (Kaggle Settings: Internet ON, GPU T4 x2)
import os, sys, subprocess, json
from pathlib import Path
import torch
assert torch.cuda.device_count() >= 2, "Select GPU T4 x2 in Kaggle Settings before running."
for i in range(2):
    props = torch.cuda.get_device_properties(i)
    print(i, props.name, round(props.total_memory / 2**30, 1), "GiB")
    assert props.total_memory >= 14 * 2**30, "Each GPU needs approximately 16 GB VRAM."
WORK = Path("/kaggle/working")
REPO = WORK / "speculative-world-model"
BRANCH = "codex/suffix-top1-math-gsm8k"
URL = "https://github.com/HoangMaizzz/speculative-world-model.git"
os.environ.update(HF_HOME="/tmp/suffix-hf-cache", TOKENIZERS_PARALLELISM="false",
                  HF_HUB_DISABLE_TELEMETRY="1", MPLCONFIGDIR="/tmp/suffix-mpl")
''',
'''# 2. Clone public repo without a token; optional Secret fallback for private repos
from kaggle_secrets import UserSecretsClient
import base64
secrets = UserSecretsClient()
try:
    hf_token = secrets.get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
except Exception:
    pass  # All three model repositories are public.
if not REPO.exists():
    git_env = os.environ.copy()
    git_env.update(GIT_TERMINAL_PROMPT="0", GIT_TRACE="0", GIT_CURL_VERBOSE="0")
    clone_args = ["git", "clone", "--branch", BRANCH, "--single-branch", URL, str(REPO)]
    result = subprocess.run(clone_args, env=git_env, capture_output=True, text=True)
    if result.returncode:
        try:
            token = secrets.get_secret("GITHUB_TOKEN")
        except Exception:
            raise RuntimeError("Clone failed. Check Kaggle Internet and repository visibility; a private repo needs Kaggle Secret GITHUB_TOKEN.") from None
        git_env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
            GIT_CONFIG_VALUE_0="AUTHORIZATION: basic " + base64.b64encode(("x-access-token:"+token).encode()).decode())
        try:
            result = subprocess.run(clone_args, env=git_env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("Private clone failed. Check token repository access and Kaggle Internet.")
        finally:
            del token
    del git_env
assert subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=REPO, text=True).strip() == URL
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
print("Experiment commit:", COMMIT)
os.chdir(REPO)
''',
'''# 3. Isolated inference environment; avoids Kaggle's preloaded package conflicts
VENV = Path("/tmp/suffix-venv")
if not (VENV / "bin/python").exists():
    subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
PY = str(VENV / "bin/python")
STAMP = VENV / "suffix-installed-commit"
if not STAMP.exists() or STAMP.read_text() != COMMIT:
    subprocess.run([PY, "-m", "pip", "install", "--no-cache-dir", "torch==2.9.0",
                    "--index-url", "https://download.pytorch.org/whl/cu126"], check=True)
    subprocess.run([PY, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements-kaggle.txt"], check=True)
    subprocess.run([PY, "-m", "pip", "install", "--no-deps", "-e", "."], check=True)
    STAMP.write_text(COMMIT)
subprocess.run([PY, "-c", "import torch; assert torch.cuda.device_count() >= 2; print(torch.__version__, torch.version.cuda)"], check=True)
def lab(command, *, smoke=False, stage=None, allow_partial=False):
    name = "smoke" if smoke else "pilot"
    args = [PY, "-m", "suffix_lab.cli", command,
            "--assets", "/tmp/suffix-assets/"+name, "--output", str(WORK / ("suffix_"+name))]
    if smoke:
        args.append("--smoke")
    if stage:
        args += ["--stage", stage]
    result = subprocess.run(args, cwd=REPO)
    if result.returncode and not (allow_partial and result.returncode == 75):
        raise RuntimeError(f"{command} failed (exit {result.returncode}). Inspect the error above; checkpoints are retained.")
    return result.returncode
''',
'''# 4. Prepare pinned models + fixed 100 MATH-500 / 100 GSM8K prompts; optional resume
lab("prepare", smoke=True)
lab("prepare")
PILOT = WORK / "suffix_pilot"
archives = sorted(Path("/kaggle/input").rglob("suffix_checkpoint.zip"))
if not (PILOT / "run.sqlite").exists() and archives:
    # Only auto-select a checkpoint whose saved identity matches this exact pilot config.
    import zipfile
    expected = json.loads((REPO / "configs/math_gsm8k.json").read_text())
    matches = []
    for path in archives:
        with zipfile.ZipFile(path) as z:
            if "run-manifest.json" in z.namelist():
                m = json.loads(z.read("run-manifest.json"))
                if m.get("config") == expected and m.get("environment", {}).get("commit") == COMMIT:
                    matches.append(path)
    if len(matches) > 1:
        raise RuntimeError("Multiple matching checkpoints attached: keep only the desired checkpoint dataset and rerun.")
    if matches:
        subprocess.run([PY, "-m", "suffix_lab.cli", "restore", "--archive", str(matches[0]),
                        "--output", str(PILOT)], check=True)
        print("Restored", matches[0].name)
    else:
        raise RuntimeError("Attached checkpoints do not match this pilot commit/config; attach the matching checkpoint or remove them.")
''',
'''# 5. CPU contracts + tiny random official model checks, pretrained GPU audits, 5+5 smoke
manifest = json.loads(Path("/tmp/suffix-assets/pilot/manifest.json").read_text())
test_env = os.environ.copy()
test_env["SUFFIX_UPSTREAM_DIR"] = manifest["paths"]["Efficient-Large-Model/Fast_dLLM_v2_1.5B"]
subprocess.run([PY, "-m", "unittest", "discover", "-s", "tests", "-v"], env=test_env, check=True)
lab("preflight", smoke=True)
try:
    lab("run", smoke=True)
finally:
    lab("analyze", smoke=True)
    lab("pack", smoke=True)
smoke_status = json.loads((WORK / "suffix_smoke/status.json").read_text())
assert smoke_status["test2_complete"], "Finish or resume smoke before starting pilot."
''',
'''# 6. Main pilot: 100 MATH-500 + 100 GSM8K; resume by rerunning this cell
assert json.loads((WORK / "suffix_smoke/status.json").read_text())["test2_complete"]
# Reuse saved precision after restoring a run; Runner refuses any identity mismatch.
if not (PILOT / "preflight.json").exists():
    lab("preflight")
try:
    code = lab("run", allow_partial=True)
finally:
    lab("analyze")
    lab("pack")
print("Pilot paused at the soft wall budget; rerun cell 6 to resume." if code == 75 else "Pilot complete.")
''',
'''# 7. Inspect report, figures and download a resumable checkpoint
from IPython.display import display, Markdown, FileLink, Image
report = PILOT / "analysis/REPORT.md"
if report.exists():
    display(Markdown(report.read_text()))
    for figure in sorted((PILOT / "analysis").glob("*.png")):
        display(Image(filename=str(figure)))
display(FileLink(str(PILOT / "suffix_checkpoint.zip")))
display(FileLink(str(WORK / "suffix_smoke/suffix_checkpoint.zip")))
# Save Version -> output files, or download the ZIP. For a new session, attach it
# as a private Kaggle Dataset; cell 4 restores it. /tmp model cache is not archived.
'''
]


# Each quickstart cell can restore the setup namespace after a kernel restart.
# Only the three reviewed setup cells are loaded from this immutable revision.
SETUP_BOOTSTRAP = '''import json, os, subprocess
from pathlib import Path
import urllib.request

if not (callable(globals().get("lab")) and
        all(name in globals() for name in ("PY", "REPO", "WORK", "COMMIT"))):
    setup_url = (
        "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
        "bc04842857cb5d739f89c9c9bc5acfbf5bd8254b/notebooks/math_gsm8k.ipynb"
    )
    with urllib.request.urlopen(setup_url, timeout=60) as response:
        setup_notebook = json.load(response)
    setup_cells = [c for c in setup_notebook["cells"] if c["cell_type"] == "code"]
    for index, cell in enumerate(setup_cells[:3], start=1):
        print(f"Setup {index}/3", flush=True)
        exec(compile("".join(cell["source"]), f"setup-cell-{index}", "exec"), globals())
assert callable(globals().get("lab")), "Setup did not define lab; inspect the preceding error."
'''

SINGLE_CELL_REVISION = "c78cd1ba7da5f52b078a7537460e08f949746c39"
SINGLE_CELL_LAUNCHER_REVISION = "274762b1cb9e5d8058bced1d7672152748c0a9df"
SINGLE_CELL_SHA256 = "8797900ba4561981d8fe6968f797706a1e963e9caee6ea5ddad293f6f79c1ac8"


def single_cell(mode):
    if mode not in ("smoke", "pilot"):
        raise ValueError(mode)
    title = "SMOKE 5 MATH + 5 GSM8K" if mode == "smoke" else "PILOT 100 MATH + 100 GSM8K (Save Version)"
    return f'''# {title}: this cell includes the entire setup and execution.
import hashlib
import urllib.request
import torch

if torch.cuda.device_count() < 2:
    raise RuntimeError("Select GPU T4 x2 and enable Internet in Kaggle Settings.")

_suffix_revision = "{SINGLE_CELL_REVISION}"
_suffix_launcher_revision = "{SINGLE_CELL_LAUNCHER_REVISION}"
_suffix_url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + _suffix_launcher_revision + "/scripts/kaggle_single_cell.py"
)
with urllib.request.urlopen(_suffix_url, timeout=60) as response:
    _suffix_source = response.read()
if hashlib.sha256(_suffix_source).hexdigest() != "{SINGLE_CELL_SHA256}":
    raise RuntimeError("Launcher checksum differs; execution stopped.")
_suffix_scope = {{"__name__": "__main__", "__file__": _suffix_url,
                 "SUFFIX_MODE": "{mode}", "SUFFIX_CODE_REVISION": _suffix_revision}}
exec(compile(_suffix_source, "kaggle-single-cell", "exec"), _suffix_scope)
'''


QUICK_CELLS = [single_cell("smoke"), single_cell("pilot")]


def build():
    cells = [textwrap.dedent(source).lstrip() for source in CELLS]
    destination = ROOT / "notebooks"
    destination.mkdir(exist_ok=True)
    intro = ("# Kaggle: suffix diagnostics, 100 MATH-500 + 100 GSM8K\n\n"
             "Enable Internet and GPU T4 x2. Public repositories clone without a GitHub token. "
             "Only if the repo is private, add and enable Kaggle Secret `GITHUB_TOKEN` with read access. `HF_TOKEN` is optional. Run cells in order. "
             "The first smoke uses 5 prompts per dataset, then the pilot uses exactly 100 per dataset. "
             "Downloading models needs disk space in /tmp. A 6-hour soft run budget can require multiple sessions.\n")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "gpu"},
        "cells": [{"cell_type": "markdown", "id": "introduction", "metadata": {}, "source": intro.splitlines(True)}] +
                 [{"cell_type": "code", "id": f"cell-{i+1}", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": source.splitlines(True)} for i, source in enumerate(cells)]}
    (destination/"math_gsm8k.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    (ROOT/"KAGGLE_CELLS.md").write_text(intro+"\n"+"\n\n".join("```python\n"+source+"```" for source in cells)+"\n", encoding="utf-8")
    quick_intro = ("# Independent single-cell Kaggle runs\n\n"
                   "Enable Internet and GPU T4 x2; no GitHub token is required. Each cell works in a fresh kernel. "
                   "Protocol v3: source AR/dLLM x redrafter AR/dLLM, frozen future-anchor controls. Smoke runs5+5 with up to3 prefixes and256-token references. Pilot runs100+100 with its own correctness checks; it never requires smoke globals/status. "
                   "For Save Version, put ONLY the pilot cell in its own notebook, then Save & Run All. "
                   "Separate one-code-cell files: math_gsm8k_smoke.ipynb and math_gsm8k_pilot.ipynb. "
                   "Launcher/source revision and SHA256 are pinned. Both runs export ZIP and report complete/partial/failed. "
                   "To resume a new session, attach exactly one matching ZIP as input and run the same cell; it restores automatically.\n")
    quick_notebook = {**notebook, "cells": [
        {"cell_type": "markdown", "id": "quickstart", "metadata": {}, "source": quick_intro.splitlines(True)}] +
        [{"cell_type": "code", "id": f"quick-{i+1}", "metadata": {}, "execution_count": None,
          "outputs": [], "source": source.splitlines(True)} for i, source in enumerate(QUICK_CELLS)]}
    (destination/"math_gsm8k_two_cells.ipynb").write_text(json.dumps(quick_notebook, indent=2), encoding="utf-8")
    (ROOT/"KAGGLE_TWO_CELLS.md").write_text(quick_intro+"\n"+"\n\n".join(
        "```python\n"+source+"```" for source in QUICK_CELLS)+"\n", encoding="utf-8")
    for mode, source in zip(("smoke", "pilot"), QUICK_CELLS):
        standalone = {**notebook, "cells": [
            {"cell_type": "markdown", "id": mode+"-intro", "metadata": {}, "source": quick_intro.splitlines(True)},
            {"cell_type": "code", "id": mode+"-run", "metadata": {}, "execution_count": None,
             "outputs": [], "source": source.splitlines(True)}]}
        (destination/f"math_gsm8k_{mode}.ipynb").write_text(json.dumps(standalone, indent=2), encoding="utf-8")
    return cells


if __name__ == "__main__":
    build()
