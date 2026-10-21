# Test 4 — Phase 1 Main200

GPU T4 x2, Internet ON. Completed original FP16 pilot unpacked at `/kaggle/input/datasets/ainzkhail/checkpt` (same input used for Test3; not the Test3 ZIP). One code cell, Save Version → Run All, no GitHub token.

Only dLLM1.5B → Qwen7B AR verification, FP16, physical32/subblock8, one token/unmask forward. Freeze 100 GSM8K +100 MATH events, one per prompt, from five fixed offsets and seeded reserves, without Fresh/Oracle outcome selection. Rebuild actual B24, no a3..15 filter. Main has600 logical arms: Fresh23, Oracle-All-Joint23 and KEEP-All23.

EOS-aware logical evaluation with unchanged native scheduler; reference extension by original-prompt single-token canonical replay. Target-terminal events retained and reported separately. No post-EOS acceptance, no online speedup claim.

CPU EOS/resume checks precede pretrained GPU preflight. Every state/arm committed to SQLite; ZIP refreshed at safe boundaries, approximately two minutes apart. Six-hour child soft budget saves partial progress; completion in one session is not guaranteed. To resume in a new session, attach the Phase1 ZIP/unpacked folder and set `resume` in the cell to its Kaggle input path. Keep the original pilot input attached.

ZIP: `/kaggle/working/test4_phase1/test4_phase1_checkpoint.zip` (revision suffix if older output exists). Includes report, paired benchmark/terminal strata with95% prompt bootstrap intervals, attrition/EOS records, anchor token types, plots, raw IDs/traces and SQLite. Live output plus15-second silence heartbeat; logs in `/kaggle/working/test4_run.log` and `test4_checks.log`.

Protocol: `docs/TEST4_PHASE1.vi.md`. Phase2+ stress tests are not run. Local checks do not establish pretrained Test4 efficacy.

Runtime: `6bc16fd80d946a0b099dda3a44f199a90a61a70e`.

```python
# TEST 4 PHASE 1: Main200 EOS-aware, three arms, strict 1 token/unmask
import hashlib
import urllib.request
from pathlib import Path
import torch

checkpoint = Path("/kaggle/input/datasets/ainzkhail/checkpt")
if not checkpoint.is_dir():
    raise RuntimeError(f"Chưa tìm thấy pilot checkpoint đã unzip: {checkpoint}")
if torch.cuda.device_count() < 2:
    raise RuntimeError("Bật GPU T4 x2 và Internet trong Kaggle Settings.")

resume = None  # Optional: Phase1 ZIP or unpacked folder uploaded as Kaggle input
revision = "6bc16fd80d946a0b099dda3a44f199a90a61a70e"
url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + revision + "/scripts/kaggle_test4_cell.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read()
if hashlib.sha256(source).hexdigest() != "c2f989a4df4b0a0fa14f439e4a13a51e4648db2880a0781b5865c1b24a7e5800":
    raise RuntimeError("Launcher checksum không khớp.")
scope = {
    "__name__": "__main__",
    "__file__": url,
    "SUFFIX_TEST4_REVISION": revision,
    "SUFFIX_TEST4_INPUT": str(checkpoint),
    "SUFFIX_TEST4_RESUME": resume,
}
exec(compile(source, "kaggle-test4", "exec"), scope)
```
