# Kaggle · SAVH learned anchor selection

Use the completed `savh_ca813c7c0101_full.zip` or the verified extracted members under `/kaggle/input/datasets/ainzkhail/savh_data`. The runner also searches `/kaggle/input`. Enable Internet and select a GPU, or set `DEVICE` to `cpu`. HGB trains on CPU and A4 uses the selected device.

The scalar pilot compares A0 Fixed‑4, A1 whole-span HGB, A2 conditional chain, A3 whole-span HGB with prompt-group out-of-fold extension scores, and A4 duration MLP. Full uses three A4 seeds. It compares seed floors 0.50 and 0.40 and optional shortlist caps of 4 and 8. Retention uses calibrated exact-span confidence, permits empty outputs, and removes contained source intervals. Validation chooses thresholds across the 55–75% precision grid. The test curves are descriptive and the locked points report actual test precision. The optional raw-feature `M_off` ablation is a separate experiment.

```python
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import urllib.request

INPUT = Path("/kaggle/input/datasets/ainzkhail/savh_data")
DEVICE = "cuda"
REV = "587300bb97f4aab1c8bcf81e905b40777ae23bda"
FILES = {
    "benchmark_savh_spans.py": "6f898a67e3f6443c1640fa1a85a3204fa7ff02144d8ba7b8b97e85018090f56c",
    "benchmark_learned_anchors.py": "1f26f6279c755546d41875fcb0881f3831dd9a4d1e830b6a320f673555266dfc",
}

work = Path("/kaggle/working")
src = work / "learned_anchor_runner"
src.mkdir(parents=True, exist_ok=True)
for name, expected in FILES.items():
    url = f"https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/{REV}/scripts/{name}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"Runner SHA-256 mismatch: {name}")
    (src / name).write_bytes(data)

cmd = [sys.executable, "-u", str(src / "benchmark_learned_anchors.py"),
       "--input", str(INPUT), "--search-root", "/kaggle/input", "--device", DEVICE]
smoke = work / "savh_learned_anchor_smoke"
full = work / "savh_learned_anchor_compare"
subprocess.run(cmd + ["--output", str(smoke), "--smoke"], check=True)
subprocess.run(cmd + ["--output", str(full)], check=True)
print("Kết quả:", shutil.make_archive(str(full), "zip", full))
```

`summary.json` contains completion status, oracle counts, candidate pool coverage, locked operating points, and paired prompt bootstrap for A3 versus A1. `precision_coverage.csv` and `rank_by_length.csv` contain the threshold and rank sweeps. `cases.jsonl` records selected intervals at a locked operating point or the 0.50 diagnostic fallback. `manifest.json` stores the split and grids. Timings combine separately measured stages with scores computed over the dense candidate table. They provide a cached-feature diagnostic before drafter integration.
