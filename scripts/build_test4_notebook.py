"""Pinned standalone Test4 Phase1 code cell: no prior notebook state or secrets."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "6bc16fd80d946a0b099dda3a44f199a90a61a70e"
LAUNCHER_SHA256 = "c2f989a4df4b0a0fa14f439e4a13a51e4648db2880a0781b5865c1b24a7e5800"


def cell():
    return f'''# TEST 4 PHASE 1: Main200 EOS-aware, three arms, strict 1 token/unmask
import hashlib
import urllib.request
from pathlib import Path
import torch

checkpoint = Path("/kaggle/input/datasets/ainzkhail/checkpt")
if not checkpoint.is_dir():
    raise RuntimeError(f"Chưa tìm thấy pilot checkpoint đã unzip: {{checkpoint}}")
if torch.cuda.device_count() < 2:
    raise RuntimeError("Bật GPU T4 x2 và Internet trong Kaggle Settings.")

resume = None  # Optional: Phase1 ZIP or unpacked folder uploaded as Kaggle input
revision = "{REVISION}"
url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + revision + "/scripts/kaggle_test4_cell.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read()
if hashlib.sha256(source).hexdigest() != "{LAUNCHER_SHA256}":
    raise RuntimeError("Launcher checksum không khớp.")
scope = {{
    "__name__": "__main__",
    "__file__": url,
    "SUFFIX_TEST4_REVISION": revision,
    "SUFFIX_TEST4_INPUT": str(checkpoint),
    "SUFFIX_TEST4_RESUME": resume,
}}
exec(compile(source, "kaggle-test4", "exec"), scope)
'''


def build():
    description = """# Test 4 — Phase 1 Main200

GPU T4 x2, Internet ON. Completed original FP16 pilot unpacked at `/kaggle/input/datasets/ainzkhail/checkpt` (same input used for Test3; not the Test3 ZIP). One code cell, Save Version → Run All, no GitHub token.

Only dLLM1.5B → Qwen7B AR verification, FP16, physical32/subblock8, one token/unmask forward. Freeze 100 GSM8K +100 MATH events, one per prompt, from five fixed offsets and seeded reserves, without Fresh/Oracle outcome selection. Rebuild actual B24, no a3..15 filter. Main has600 logical arms: Fresh23, Oracle-All-Joint23 and KEEP-All23.

EOS-aware logical evaluation with unchanged native scheduler; reference extension by original-prompt single-token canonical replay. Target-terminal events retained and reported separately. No post-EOS acceptance, no online speedup claim.

CPU EOS/resume checks precede pretrained GPU preflight. Every state/arm committed to SQLite; ZIP refreshed at safe boundaries, approximately two minutes apart. Six-hour child soft budget saves partial progress; completion in one session is not guaranteed. To resume in a new session, attach the Phase1 ZIP/unpacked folder and set `resume` in the cell to its Kaggle input path. Keep the original pilot input attached.

ZIP: `/kaggle/working/test4_phase1/test4_phase1_checkpoint.zip` (revision suffix if older output exists). Includes report, paired benchmark/terminal strata with95% prompt bootstrap intervals, attrition/EOS records, anchor token types, plots, raw IDs/traces and SQLite. Live output plus15-second silence heartbeat; logs in `/kaggle/working/test4_run.log` and `test4_checks.log`.

Protocol: `docs/TEST4_PHASE1.vi.md`. Phase2+ stress tests are not run. Local checks do not establish pretrained Test4 efficacy.
"""
    source = cell()
    notebook = {"cells": [
        {"cell_type": "markdown", "id": "test4-description", "metadata": {}, "source": description.splitlines(True)},
        {"cell_type": "code", "id": "test4-run", "metadata": {}, "execution_count": None, "outputs": [], "source": source.splitlines(True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}
    (ROOT/"notebooks/test4_phase1.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (ROOT/"docs/KAGGLE_TEST4.md").write_text(description+"\nRuntime: `"+REVISION+"`.\n\n```python\n"+source+"```\n", encoding="utf-8")


if __name__ == "__main__":
    build()
