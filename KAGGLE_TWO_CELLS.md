# Independent single-cell Kaggle runs

Enable Internet and GPU T4 x2; no GitHub token is required. Each cell works in a fresh kernel. Smoke runs5+5. Pilot runs100+100 with its own correctness checks; it never requires smoke globals/status. For Save Version, put ONLY the pilot cell in its own notebook, then Save & Run All. Separate one-code-cell files: math_gsm8k_smoke.ipynb and math_gsm8k_pilot.ipynb. Launcher/source revision and SHA256 are pinned. Both runs export ZIP and report complete/partial/failed. To resume a new session, attach exactly one matching ZIP as input and run the same cell; it restores automatically.

```python
# SMOKE 5 MATH + 5 GSM8K: this cell includes the entire setup and execution.
import hashlib
import urllib.request
import torch

if torch.cuda.device_count() < 2:
    raise RuntimeError("Select GPU T4 x2 and enable Internet in Kaggle Settings.")

_suffix_revision = "c4a25fafcdd66884c33ef7251f14a6bc6874a388"
_suffix_url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + _suffix_revision + "/scripts/kaggle_single_cell.py"
)
with urllib.request.urlopen(_suffix_url, timeout=60) as response:
    _suffix_source = response.read()
if hashlib.sha256(_suffix_source).hexdigest() != "d09babf75c46c32a6871c800a470be82b9593748968880a41e4a83b65f9b702e":
    raise RuntimeError("Launcher checksum differs; execution stopped.")
_suffix_scope = {"__name__": "__main__", "__file__": _suffix_url,
                 "SUFFIX_MODE": "smoke", "SUFFIX_CODE_REVISION": _suffix_revision}
exec(compile(_suffix_source, "kaggle-single-cell", "exec"), _suffix_scope)
```

```python
# PILOT 100 MATH + 100 GSM8K (Save Version): this cell includes the entire setup and execution.
import hashlib
import urllib.request
import torch

if torch.cuda.device_count() < 2:
    raise RuntimeError("Select GPU T4 x2 and enable Internet in Kaggle Settings.")

_suffix_revision = "c4a25fafcdd66884c33ef7251f14a6bc6874a388"
_suffix_url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + _suffix_revision + "/scripts/kaggle_single_cell.py"
)
with urllib.request.urlopen(_suffix_url, timeout=60) as response:
    _suffix_source = response.read()
if hashlib.sha256(_suffix_source).hexdigest() != "d09babf75c46c32a6871c800a470be82b9593748968880a41e4a83b65f9b702e":
    raise RuntimeError("Launcher checksum differs; execution stopped.")
_suffix_scope = {"__name__": "__main__", "__file__": _suffix_url,
                 "SUFFIX_MODE": "pilot", "SUFFIX_CODE_REVISION": _suffix_revision}
exec(compile(_suffix_source, "kaggle-single-cell", "exec"), _suffix_scope)
```
