# Resume an unpacked pilot checkpoint

The input is read-only. Legacy audited v4 references/passing proposals are imported and the archived failure is replayed before continuation. New-runtime checkpoints can also be resumed from the same unpacked layout.

```python
# RESUME PILOT 100 MATH + 100 GSM8K: this cell includes the entire setup and execution.
import os
from pathlib import Path
import hashlib
import urllib.request
import torch

if torch.cuda.device_count() < 2:
    raise RuntimeError("Select GPU T4 x2 and enable Internet in Kaggle Settings.")

checkpoint_root = Path("/kaggle/input/datasets/ainzkhail/checkpoint")
if not checkpoint_root.is_dir():
    raise RuntimeError("Attach the unpacked checkpoint dataset at the configured path.")
os.environ["SUFFIX_RESUME_ROOT"] = str(checkpoint_root)

_suffix_revision = "fa330e5836db310172ced163162d511f6b2036b5"
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
