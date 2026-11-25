# Paired retrieval and recovery benchmark

The full run uses the 50 existing held-out prompts (25 MATH-500, 25 GSM8K), up to 256 output tokens each, trunk length 16, retrieval width 16. The static corpus is the 7B output on 100 training prompts. Each prompt/arm starts with empty suffix history. The selector and its validation median threshold are frozen. Prompt order is deterministic, interleaved by domain. Arm order is randomized per prompt. Each arm warms up before measurement.

| Arm | Main proposal | Recovery proposal |
|---|---|---|
| A | Greedy 1.5B | None |
| B | Greedy 1.5B accelerated by corpus, history and up to four rejected suffixes | None |
| C | Same retrieval mechanism | One selector-chosen root |
| D | Same retrieval mechanism | Root plus up to three serial 1.5B continuation tokens |

The selector considers ranks 2 through 4 on the actual greedy path taken through the 1.5B tree. It retrieves up to three aligned old contexts plus the unshifted position and reads the old verifier top-32 rows. A matched branch can earn a target bonus. Recovery feedback is available after a main rejection without a winning branch, following the prior live pilot. A verifier forward checks the entire main proposal and side branch together. The cache commits only the verified path.

Kaggle requires Internet, T4 x2 and the supplied 138 KB input archive, mounted directly or as extracted files. Input file hashes are checked before loading the classifier. The model revisions are pinned in the input. The drafter is on GPU 1; 18 verifier blocks run on GPU 0 and the remaining blocks on GPU 1. No CPU offload. Wall time includes prefill, drafting, transfers, selection, regrowth, verification and cache maintenance. Downloads, warmup, corpus initialization and file writes are outside timing.

The output contains every round, exact-output counts, accepted proposal tokens, committed tokens per verifier call, stage times and paired speed ratios B/A, C/B, D/B, D/C. The speed ratio is baseline total time divided by method total time on matching outputs. The 95% interval uses prompt resampling within dataset. Inspect exact counts before interpreting speed. The result ZIP is refreshed atomically after each complete prompt. A rerun in the same working directory resumes from the last completed prompt.
