# Mini-test dLLM -> AR: redraft + extension

GPU T4 x2, Internet ON. Add the unpacked pilot dataset at `/kaggle/input/datasets/ainzkhail/checkpt`.
Run the single code cell, or Save Version with Run All. No GitHub token needed.

10 dLLM-source rejection events: 5 MATH +5 GSM8K, 6 enriched +4 representative. Only dLLM redrafting and AR verification; no AR drafter. Nine arms/event, fixed E=8, strict one token per unmask forward in earliest absolute subblock8. Oracle and predicted anchors are frozen before inference.

Reuse pilot references/Test1/event metadata. Replay only selected Fresh-L baselines for unmask traces and exact pilot comparison. Audit cached/full-prefix equality and reverify every unique final proposal. FP16 target across both GPUs; dLLM on GPU1. No latency or speedup claim.

Output: `/kaggle/working/extension_mini/extension_checkpoint.zip`; includes CSVs, per-arm traces, report, and token-level `cases.html`. Partial output is preserved on runtime failure. Full protocol: `docs/MINI_EXTENSION.vi.md`.

Runtime: `4f8359329e32cf66daafbd7d7fe961956f938d87`.

```python
# MINI-TEST: dLLM -> AR, 10 events, 90 arms, 1 token/unmask forward
import hashlib
import urllib.request
from pathlib import Path
import torch

checkpoint = Path("/kaggle/input/datasets/ainzkhail/checkpt")
if not checkpoint.is_dir():
    raise RuntimeError(f"Chưa tìm thấy checkpoint đã unzip: {checkpoint}")
if torch.cuda.device_count() < 2:
    raise RuntimeError("Bật GPU T4 x2 và Internet trong Kaggle Settings.")

revision = "4f8359329e32cf66daafbd7d7fe961956f938d87"
url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + revision + "/scripts/kaggle_extension_cell.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read()
if hashlib.sha256(source).hexdigest() != "a71679034010ee9755969b38bc803b8c3b4804b36fd27b382ee12f7df5c04cb6":
    raise RuntimeError("Launcher checksum không khớp.")
scope = {
    "__name__": "__main__",
    "__file__": url,
    "SUFFIX_EXTENSION_REVISION": revision,
    "SUFFIX_EXTENSION_INPUT": str(checkpoint),
}
exec(compile(source, "kaggle-mini-extension", "exec"), scope)
```
