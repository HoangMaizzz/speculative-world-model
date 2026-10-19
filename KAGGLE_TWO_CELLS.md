# Kaggle two-cell quickstart

Enable Internet and GPU T4 x2. Cell1 includes setup, correctness checks and smoke5+5. Cell2 runs100 MATH-500 +100 GSM8K after smoke; no earlier setup cell is needed. Public repo: no GitHub token. Setup uses the three reviewed cells at an immutable commit. Same-session resume: rerun the relevant cell. For new-session ZIP restore use math_gsm8k.ipynb.

```python
import json, os, subprocess
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

# Smoke: setup automatically, prepare 5+5, audit, then run both studies.
lab("prepare", smoke=True)
# 5. CPU contracts + tiny random official model checks, pretrained GPU audits, 5+5 smoke
manifest = json.loads(Path("/tmp/suffix-assets/smoke/manifest.json").read_text())
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

from IPython.display import display, FileLink
display(FileLink(str(WORK / "suffix_smoke/suffix_checkpoint.zip")))
```

```python
import json, os, subprocess
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

# Pilot: run after smoke; rerun this cell to resume in the same session.
PILOT = WORK / "suffix_pilot"
assert (WORK / "suffix_smoke/status.json").exists(), "Run the smoke cell first; for a new-session checkpoint restore, use the full 7-cell notebook."
assert json.loads((WORK / "suffix_smoke/status.json").read_text())["test2_complete"], "Finish smoke before preparing the pilot."
lab("prepare")
# 6. Main pilot: 100 MATH-500 + 100 GSM8K; resume by rerunning this cell
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

from IPython.display import display, Markdown, FileLink
display(Markdown((PILOT / "analysis/REPORT.md").read_text()))
display(FileLink(str(PILOT / "suffix_checkpoint.zip")))
```
