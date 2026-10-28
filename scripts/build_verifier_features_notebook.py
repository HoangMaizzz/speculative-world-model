"""Generate an immutable one-code-cell notebook and matching copyable docs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(revision, directory=ROOT):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected immutable commit")
    directory = Path(directory)
    checksum = hashlib.sha256((ROOT/"scripts/kaggle_verifier_features_cell.py").read_text(encoding="utf-8").encode()).hexdigest()
    cell = f'''import hashlib
import urllib.request

INPUT_ROOT = "/kaggle/input"  # attach the completed long64_checkpoint.zip (ZIP or unpacked)
RESUME_ROOT = None           # optional: previous feature ZIP / unpacked output folder
REVISION = "{revision}"
EXPECTED_SHA256 = "{checksum}"
url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + REVISION + "/scripts/kaggle_verifier_features_cell.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read()
if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("Launcher checksum mismatch")
scope = {{
    "__name__": "__main__",
    "FEATURE_REVISION": REVISION,
    "FEATURE_INPUT_ROOT": INPUT_ROOT,
    "FEATURE_RESUME_ROOT": RESUME_ROOT,
    "FEATURE_LIMIT": None,
    "FEATURE_MICRO_PROBE_STEPS": 0,
}}
exec(compile(source, url, "exec"), scope)
'''
    notebook = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [{"cell_type": "code", "id": "verifier-features-full", "execution_count": None,
                   "metadata": {}, "outputs": [], "source": cell.splitlines(keepends=True)}]}
    (directory/"notebooks").mkdir(parents=True, exist_ok=True)
    (directory/"docs").mkdir(parents=True, exist_ok=True)
    (directory/"notebooks/verifier_features_full.ipynb").write_text(json.dumps(notebook, indent=2)+"\n", encoding="utf-8")
    text = ("# Kaggle: full verifier features B64\n\n"
            "Settings: GPU T4 x2, Internet ON. Add Input: completed guarded long64_checkpoint.zip (200 prompts/4050 rounds). "
            "Auto-discovery supports ZIP or unpacked files below /kaggle/input. No GitHub token needed.\n\n"
            "One independent cell for Save Version: setup + CPU checks + GPU smoke20 + pilot500 milestone + full3847. "
            "Qwen only, FP16 two GPUs, top16/layers7,14,21,27, probes OFF.\n\n"
            f"Runtime `{revision}`; launcher SHA256 `{checksum}`.\n\n"
            "```python\n"+cell+"```\n\n"
            f"Output: `/kaggle/working/verifier_features_{revision[:12]}/verifier_features_checkpoint.zip`. "
            "Expect several GiB; partial shards retained on budget/error. Same cell resumes local shards. "
            "Across sessions attach prior output as input and set RESUME_ROOT to that ZIP/unpacked folder; keep original B64 input too.\n\n"
            "Soft budget6h after model load; stage compute excludes peer dispatch and operational wall is recorded separately. "
            "Smoke replay mismatch stops full with preserved evidence. See VERIFIER_FEATURE_COLLECTION.vi.md for schema/limits.\n")
    (directory/"docs/KAGGLE_VERIFIER_FEATURES.md").write_text(text, encoding="utf-8")
    return cell


if __name__ == "__main__":
    import sys
    build(sys.argv[1])
