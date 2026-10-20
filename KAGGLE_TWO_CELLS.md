# Independent single-cell Kaggle runs

Enable Internet and GPU T4 x2; no GitHub token is required. Each cell works in a fresh kernel. Protocol v3: source AR/dLLM x redrafter AR/dLLM, frozen future-anchor controls. Smoke runs5+5 with up to3 prefixes and256-token references. Pilot runs100+100 with its own correctness checks; it never requires smoke globals/status. For Save Version, put ONLY the pilot cell in its own notebook, then Save & Run All. Separate one-code-cell files: math_gsm8k_smoke.ipynb and math_gsm8k_pilot.ipynb. Launcher/source revision and SHA256 are pinned. Both runs export ZIP and report complete/partial/failed. To resume a new session, attach exactly one matching ZIP as input and run the same cell; it restores automatically.

```python
# SMOKE 5 MATH + 5 GSM8K: this cell includes the entire setup and execution.
import hashlib
import urllib.request
import torch

if torch.cuda.device_count() < 2:
    raise RuntimeError("Select GPU T4 x2 and enable Internet in Kaggle Settings.")

_suffix_revision = "c78cd1ba7da5f52b078a7537460e08f949746c39"
_suffix_launcher_revision = "274762b1cb9e5d8058bced1d7672152748c0a9df"
_suffix_url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + _suffix_launcher_revision + "/scripts/kaggle_single_cell.py"
)
with urllib.request.urlopen(_suffix_url, timeout=60) as response:
    _suffix_source = response.read()
if hashlib.sha256(_suffix_source).hexdigest() != "8797900ba4561981d8fe6968f797706a1e963e9caee6ea5ddad293f6f79c1ac8":
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

_suffix_revision = "c78cd1ba7da5f52b078a7537460e08f949746c39"
_suffix_launcher_revision = "274762b1cb9e5d8058bced1d7672152748c0a9df"
_suffix_url = (
    "https://raw.githubusercontent.com/HoangMaizzz/speculative-world-model/"
    + _suffix_launcher_revision + "/scripts/kaggle_single_cell.py"
)
with urllib.request.urlopen(_suffix_url, timeout=60) as response:
    _suffix_source = response.read()
if hashlib.sha256(_suffix_source).hexdigest() != "8797900ba4561981d8fe6968f797706a1e963e9caee6ea5ddad293f6f79c1ac8":
    raise RuntimeError("Launcher checksum differs; execution stopped.")
_suffix_scope = {"__name__": "__main__", "__file__": _suffix_url,
                 "SUFFIX_MODE": "pilot", "SUFFIX_CODE_REVISION": _suffix_revision}
exec(compile(_suffix_source, "kaggle-single-cell", "exec"), _suffix_scope)
```
