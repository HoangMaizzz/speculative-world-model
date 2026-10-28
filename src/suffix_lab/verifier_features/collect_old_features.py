"""The original parallel proposal path, including all off-path rows and bonus."""
import time
from .replay_alignment import draft_features


def collect(target, event, top_k=16, audit_hooks=False):
    import torch
    p, a, source = len(event["prefix"]), event["accepted"], event["source"]
    n, length = len(event["proposal"]), len(source)
    # Union of prediction/pre-read and post-read positions: L+1, not 2L.
    positions = list(range(p+a, p+a+length+1))
    logits, hidden, _, timing = target.forward(event["prefix"]+event["proposal"], n+1, positions)
    if logits.shape[0] != n+1:
        raise AssertionError("Prediction rows must include bonus")
    begin = time.perf_counter()
    values, logp = draft_features(logits, event["proposal"], top_k)
    expected = event["parallel"]["top1_with_bonus"]
    differences = [{"position": i, "stored": x, "replayed": y}
                   for i, (x, y) in enumerate(zip(expected, values["top1"])) if x != y]
    check = {"id": event["id"], "old_top1_matches": not differences, "old_top1_differences": differences,
             "old_margin_max_abs_difference": max(abs(x-y) for x, y in zip(
                 event["parallel"]["logit_margin"], values["logit_margin"])),
             "stored_guard_disagreements": event["guard"]["numerical_disagreements"],
             "hook_audit": None}
    # Columns j=0..L predict old S[j], plus a boundary prediction at j=L.
    # That boundary is the bonus row only when S reaches the final proposal ID.
    src = torch.tensor(source, device=logp.device)
    matrix = logp[a+1:a+length+2].index_select(1, src).T.contiguous().cpu()
    if matrix.shape != (length, length+1):
        raise AssertionError("Off-path alignment shape")
    tensors = {"M_off": matrix, "M_off_valid": torch.ones_like(matrix, dtype=torch.bool),
               "source_ids": torch.tensor(source, dtype=torch.int64),
               "hidden_absolute_positions": torch.tensor(positions, dtype=torch.int64)}
    tensors.update({"old_"+k: v for k, v in hidden.items()})
    timing["feature_reduce_wall_ms"] = (time.perf_counter()-begin)*1000
    if audit_hooks:
        plain, _, _, audit_time = target.forward(event["prefix"]+event["proposal"], n+1)
        check["hook_audit"] = {"top1_equal": torch.equal(plain.argmax(-1), logits.argmax(-1)),
                               "max_logit_abs_difference": float((plain-logits).abs().max()),
                               "timing": audit_time}
        if not check["hook_audit"]["top1_equal"]:
            raise AssertionError("Capture hooks changed greedy decisions")
        changed = list(event["proposal"])
        changed[-1] = (changed[-1]+1) % logits.shape[-1]
        causal, _, _, causal_time = target.forward(event["prefix"]+changed, n+1)
        check["causal_tail_audit"] = {"top1_equal_before_last_read": torch.equal(
            causal[:-1].argmax(-1), logits[:-1].argmax(-1)), "timing": causal_time}
        if not check["causal_tail_audit"]["top1_equal_before_last_read"]:
            raise AssertionError("Future proposal token changed earlier causal predictions")
    return values, tensors, check, timing
