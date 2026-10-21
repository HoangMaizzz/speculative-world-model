# Test 3 — Oracle-All-Correct Refill

GPU T4 x2, Internet ON. Unpacked completed pilot at `/kaggle/input/datasets/ainzkhail/checkpt`. Run the one code cell or Save Version with Run All. No GitHub token required.

Only Fast-dLLM1.5B -> Qwen7B AR verification, FP16. Regenerate original B24, accept a=3..15, commit correction into context, then compare H23 unverified tokens (L=23-a, E=a). Keep ALL old-correct tokens, including singleton islands; no25% or two-island limit. Strict one token per unmask forward in native earliest absolute subblock8/physical32.

Screening comes first: actual B24 and Fresh23 generation on seeded pilot contexts until 10 distinct prompts meet4early-rescue/3later-anchor/2low-reuse/1ceiling and5GSM8K+5MATH. Screening may examine more than10states and is additional to the60experimental arms. No repair outcomes used for sample selection; unmet quotas are reported explicitly.

Then six arms/event: F23, Oracle Joint23, Oracle Sequential23, F-L, Oracle-L, KEEP-All23. Cached/full checks, independent actual-target verification, and direct-KEEP versus mutable-conditioning diagnostics are mandatory. No latency/speedup claim.

Output `/kaggle/working/test3_oracle_all/test3_checkpoint.zip` (revision suffix if an older output exists). Includes events/arms JSONL, summary.csv, report.md, all10token views in cases.html, screening/selection and per-state/arm checkpoints. Live stage logs and15-second silence heartbeats; full log `/kaggle/working/test3_run.log`.

Protocol: `docs/TEST3_ORACLE_ALL.vi.md`. Pretrained GPU Test3 results are not established by local mocked/tiny-model checks.

Runtime: `adad6015518e3d69160781105d45cbdcb68c7bc2`.

```python
# TEST 3: Oracle-All-Correct, B24 -> correction + 23, strict 1 token/unmask
import hashlib
import urllib.request
from pathlib import Path
import torch

checkpoint = Path("/kaggle/input/datasets/ainzkhail/checkpt")
if not checkpoint.is_dir():
    raise RuntimeError(f"Chưa tìm thấy pilot checkpoint đã unzip: {checkpoint}")
if torch.cuda.device_count() < 2:
    raise RuntimeError("Bật GPU T4 x2 và Internet trong Kaggle Settings.")

revision = "adad6015518e3d69160781105d45cbdcb68c7bc2"
url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + revision + "/scripts/kaggle_test3_cell.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read()
if hashlib.sha256(source).hexdigest() != "fd4079129041de342ea8709f040995b6d4976c1c44c493e8613b5734e86b535d":
    raise RuntimeError("Launcher checksum không khớp.")
scope = {
    "__name__": "__main__",
    "__file__": url,
    "SUFFIX_TEST3_REVISION": revision,
    "SUFFIX_TEST3_INPUT": str(checkpoint),
}
exec(compile(source, "kaggle-test3", "exec"), scope)
```
