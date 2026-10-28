# Kaggle: full verifier features B64

Settings: GPU T4 x2, Internet ON. Add Input: completed guarded long64_checkpoint.zip (200 prompts/4050 rounds). Auto-discovery supports ZIP or unpacked files below /kaggle/input. No GitHub token needed.

One independent cell for Save Version: setup + CPU checks + GPU smoke20 + pilot500 milestone + full3847. Qwen only, FP16 two GPUs, top16/layers7,14,21,27, probes OFF.

Runtime `07ccc836489ce771d0152e468f030226bed6ec90`; launcher SHA256 `7ccfdef9476ba2327caab1450c38b2c97cb5a1365813c4f0482b2122cca3df5d`.

```python
import hashlib
import urllib.request

INPUT_ROOT = "/kaggle/input"  # attach the completed long64_checkpoint.zip (ZIP or unpacked)
RESUME_ROOT = None           # optional: previous feature ZIP / unpacked output folder
REVISION = "07ccc836489ce771d0152e468f030226bed6ec90"
EXPECTED_SHA256 = "7ccfdef9476ba2327caab1450c38b2c97cb5a1365813c4f0482b2122cca3df5d"
url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + REVISION + "/scripts/kaggle_verifier_features_cell.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read()
if hashlib.sha256(source).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("Launcher checksum mismatch")
scope = {
    "__name__": "__main__",
    "FEATURE_REVISION": REVISION,
    "FEATURE_INPUT_ROOT": INPUT_ROOT,
    "FEATURE_RESUME_ROOT": RESUME_ROOT,
    "FEATURE_LIMIT": None,
    "FEATURE_MICRO_PROBE_STEPS": 0,
}
exec(compile(source, url, "exec"), scope)
```

Output: `/kaggle/working/verifier_features_07ccc836489c/verifier_features_checkpoint.zip`. Expect several GiB; partial shards retained on budget/error. Same cell resumes local shards. Across sessions attach prior output as input and set RESUME_ROOT to that ZIP/unpacked folder; keep original B64 input too.

Soft budget6h after model load; stage compute excludes peer dispatch and operational wall is recorded separately. Smoke replay mismatch stops full with preserved evidence. See VERIFIER_FEATURE_COLLECTION.vi.md for schema/limits.
