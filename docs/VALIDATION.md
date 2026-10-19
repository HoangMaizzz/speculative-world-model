# Validation record

Development host: Windows, Python3.12.14, PyTorch2.9.0+cpu, Transformers4.53.3, einops0.8.1. Analysis checked with pandas2.2.3, matplotlib3.10.1, scikit-learn1.6.1 (local transitive NumPy2.5.4 / HF Hub0.36.2; Kaggle pins NumPy2.1.3 / HF Hub0.35.3). No pretrained model weights loaded locally.

The local unittest suite completed **30 tests, no skips**. It checks:

- Independent hand-computed first-rejection/suffix/shift examples, empty suffix/EOS horizon bounds, islands and false-anchor caps.
- One MASK filled per step, maximum confidence restricted to earliest absolute active subblock, tie order, hard KEEP, independent branches and no output from padded positions.
- Non-oracle interface isolation, deterministic selectors, matched KEEP counts, bounded correct-only oracle masks and stratified frozen sampling.
- Committed SQLite resume, precision identity mismatch rejection, consistent hot backup/ZIP restoration and unsafe ZIP path rejection.
- A complete synthetic teacher pipeline through paired Test1, Test2, rerun/resume and analysis; the fixture is not evidence of model quality.
- Tiny random Qwen2: incremental greedy matches repeated full-prefix argmax and parallel shifted verification.
- Tiny random model using **the exact hash-pinned Fast-dLLM remote source**: cached vs complete-prefix results at prefix lengths1/7/8/31/32/33/63/64/65, 35-token horizon across block boundaries, partial/full hard KEEP. No pretrained quality inference follows from random weights.
- Notebook cells match the copyable Markdown, compile as Python, and the main config is exactly100 MATH-500 +100 GSM8K.
- Empty partial outputs and populated synthetic outputs both produce all10 plots and summaries.

The full optional checks need torch/transformers/einops plus analysis dependencies. Without them, optional tests explicitly skip; passing the dependency-free subset is not the same validation result. To fetch only pinned official source for the optional tiny-model test:

```bash
python scripts/review_upstream.py
python -m unittest discover -s tests -v
```

`--refresh-lock` is an explicit new-source review operation, not part of running this frozen experiment. Kaggle cells instead point the test at the already downloaded pinned model source.

**Not yet validated:** actual Qwen7B INT8/NF4 output equality, real Fast-dLLM FP16 cached/full equality, quantization/kernel behavior on two T4 GPUs, pretrained suffix quality, runtime/VRAM for the200-prompt pilot, or online speedup. Notebook preflight must establish real-model consistency before running; it stops rather than treating numerical inconsistencies as acceptance. Per-proposal causal/reference consistency is checked again throughout Test1.

Source revisions and hashes are in `source-lock.json`. Run identity includes actual precision, full package inventory, Python/CUDA/driver/GPU, code commit/source hashes, exact config, tokenizer evidence and asset selection. Preflight cannot change an established run's precision. All timings include synchronization; model warmup occurs outside measured experiment arms and is logged.

Subsequent quickstart fix: added a two-cell notebook and Markdown whose smoke and pilot cells initialize their own setup namespace. Two new regression checks passed: missing `lab` in an empty namespace loads exactly the three setup cells and reuses them on repeat execution, and generated quickstart cells match/compile. The setup test mocks the download and setup bodies; it does not claim Kaggle installation or GPU execution. No inference protocol changed.

2026-10-20 independent Save Version update: smoke and pilot each have a separate notebook with exactly one code cell. Pilot performs setup/tests/preflight itself and has no smoke-file/kernel-global dependency. Runtime is pinned to commitc4a25faf with a checked launcher SHA256. Five mocked orchestration tests cover a clean pilot session, smoke flags, partial ZIP cleanup, failed-preflight stopping/cleanup and exact checkpoint selection. Artifact tests also execute each entry cell in its own empty namespace with mocked network/GPU, verify one code cell per notebook, and check the launcher digest. These establish control flow, not a real Kaggle GPU pass. The inference schedule/configuration is unchanged.

The complete updated local suite passed **40 tests, no skips**, including optional tiny random official-model CPU audits and analysis. The frozen runtime commitc4a25faf contains37 tests; the three additional release-artifact checks verify the subsequently generated standalone cells/notebooks. Real pretrained T4 execution remains unvalidated locally.

Kaggle user report: `/usr/bin/python3 -m venv /tmp/suffix-venv` failed in a Python3.13 session before model loading. The supplied traceback omits the underlying venv stderr, so missing ensurepip is a likely cause, not an established diagnosis. Launcher8fab80c now avoids ensurepip altogether, repairs an existing interpreter with missing pip, and streams/persists full setup output. Four regression checks cover fresh/no-pip creation, partial environment repair, healthy reuse and actual failed-child log preservation. Experiment runtime remains c4a25faf; only installer/entry-cell pins change.

2026-10-20 dependency correction: the subsequent Kaggle log establishes a separate packaging error: `datasets==4.2.0` requires `pyarrow>=21.0.0`, while the original requirements pinned19.0.1. Requirements now pin `pyarrow==21.0.0`. A pip dry-run resolved all12 direct requirements together with `torch==2.9.0`, selecting Linux cp313 binary wheels (no packages installed). This was a cross-target resolver check from Windows/Python3.12; pip still evaluates some platform markers using the host, so it does not establish a complete Linux installation, CUDA compatibility or GPU inference. Entry cells must pin a new runtime commit containing this requirements change; updating only the launcher would still clone the old conflicting requirements. The experimental protocol and model/dataset revisions are unchanged.

The corrected runtime is `723200ace9c58041a84925a1d04852c82fda0020`. The complete local suite passed **44 tests, no skips**, including the optional tiny random official-model CPU audits. Standalone cells now select this runtime and retain the independently pinned installer8fab80c. Neither the resolver dry-run nor these tests establish a pretrained Kaggle GPU pass.
