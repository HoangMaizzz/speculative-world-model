"""Structural/probability checks with explicit FP16 capture storage tolerances."""
import math


def validate_values(old, corrected, tensors, event, layers, hidden_size):
    import torch
    n, length = len(event["proposal"]), len(event["source"])
    for data in (old, corrected):
        rows = len(data["top1"])
        for key in ("entropy", "logit_margin", "top1_probability", "topk_ids", "topk_logp"):
            if len(data[key]) != rows:
                raise AssertionError("Distribution row count")
        if any(not math.isfinite(x) or x < -1e-5 for x in data["entropy"]+data["logit_margin"]):
            raise AssertionError("Nonfinite/negative entropy or margin")
        if any(not 0 < x <= 1.00001 for x in data["top1_probability"]):
            raise AssertionError("Invalid top1 probability")
        for winner, ids, logp in zip(data["top1"], data["topk_ids"], data["topk_logp"]):
            if winner != ids[0] or len(set(ids)) != len(ids) or any(not math.isfinite(p) or p > 1e-5 for p in logp):
                raise AssertionError("Invalid full-vocabulary TopK log probabilities")
            if sum(math.exp(p) for p in logp) > 1.00001:
                raise AssertionError("TopK mass exceeds full vocabulary mass")
    if len(old["top1"]) != n+1 or len(old["verifier_logp_draft"]) != n:
        raise AssertionError("Prediction indexing")
    if any(x < 1 for x in old["verifier_rank_draft"]) or any(x > 1e-5 for x in old["verifier_logp_draft"]):
        raise AssertionError("Invalid draft rank/log probability")
    if any(not math.isfinite(x) for x in old["verifier_logp_draft"]+old["verifier_gap_to_top1"]):
        raise AssertionError("Nonfinite draft token features")
    for key, value in tensors.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise AssertionError(f"Invalid tensor {key}")
    for i in layers:
        if tensors[f"old_layer_{i}"].shape != (length+1, hidden_size):
            raise AssertionError("Sparse old hidden shape")
        if tensors[f"corrected_layer_{i}"].shape != (1, hidden_size):
            raise AssertionError("Correction hidden shape")
    if tensors["q0_logp_source"].shape != (length,) or tensors["M_off"].shape != (length, length+1):
        raise AssertionError("Alignment tensor shape")
    if not torch.equal(tensors["source_ids"], torch.tensor(event["source"])):
        raise AssertionError("Source ID mapping")
