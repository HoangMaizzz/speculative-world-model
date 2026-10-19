# Speculative World Model — Suffix Reusability Pilot

This repository contains a **greedy, no-fine-tuning, fixed-32-token** pilot to compare AR–AR and dLLM–AR rejected-suffix reuse.

- **Verifier:** `Qwen/Qwen2.5-7B-Instruct` on `cuda:0`, 8-bit (Kaggle T4).
- **AR drafter:** `Qwen/Qwen2.5-1.5B-Instruct` on `cuda:1`, FP16.
- **Diffusion drafter:** `Efficient-Large-Model/Fast_dLLM_v2_1.5B` on `cuda:1`, FP16. Native `threshold=1.0, small_block_size=8, block_size=32, temperature=0`, selecting exactly one remaining max-confidence token per unmask forward.
- **Test 1:** paired target-reference prefixes, first-rejection position, old-prefix verifier logits, exact positional suffix agreement, matched islands, direct rejoin length.
- **Test 2:** offline rollback to the verifier-corrected prefix, then hard-KEEP selective dLLM infilling vs fresh/random/drafter-guided/verifier-guided/oracle KEEP.
- **Scope:** headroom / mechanism study only. No online speedup claim. No target-reference labels used by non-oracle selectors.

See `KAGGLE_CELLS.md` for cells; `suffix_pilot.py` for the runner.

**Important:** Quantizing the verifier changes its greedy reference relative to FP16. All arms share the *same* INT8 verifier in this pilot. On a T4×2, GPU memories are not pooled; we **place** independent models across GPUs, not DDP-shard Qwen 7B.

Outputs: `test1.jsonl`, `test2.jsonl`, `summary.json`, `run_config.json`. Do not commit model weights or benchmark outputs.

## Run
```bash
python suffix_pilot.py --dataset gsm8k --num-prompts 3 --max-test2-events 2 --output /kaggle/working/suffix_smoke
```

See the Kaggle cells for initial environment setup, model credentials, and a larger pilot.
