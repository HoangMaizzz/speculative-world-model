"""One real corrected prefix; probes use actual model predictions only."""
import time
from .replay_alignment import distributions


def collect(target, event, top_k=16, micro_probe_steps=0, eos=()):
    import torch
    logits, hidden, cache, timing = target.forward(event["corrected_prefix"], 1,
        [len(event["corrected_prefix"])-1], final_norm=True, use_cache=micro_probe_steps > 1)
    begin = time.perf_counter()
    values, logp = distributions(logits, top_k)
    source = torch.tensor(event["source"], device=logp.device)
    tensors = {"q0_logp_source": logp[0].index_select(0, source).cpu().contiguous()}
    tensors.update({"corrected_"+k: v for k, v in hidden.items()})
    timing["feature_reduce_wall_ms"] = (time.perf_counter()-begin)*1000
    probes, probe_times = [], []
    for step in range(micro_probe_steps):
        token = int(logits[0].argmax())
        probes.append(token)
        if token in eos or step == micro_probe_steps-1:
            break
        logits, _, cache, t = target.forward([token], 1, use_cache=True, cache=cache)
        probe_times.append(t)
    matches = [i for i in range(len(event["source"])-len(probes)+1)
               if probes and event["source"][i:i+len(probes)] == probes]
    values["micro_probe"] = {"tokens": probes, "source_offsets": matches,
                              "occurrences_by_probe_length": [
                                  {"length": k, "source_offsets": [i for i in range(len(event["source"])-k+1)
                                    if event["source"][i:i+k] == probes[:k]]} for k in range(1, len(probes)+1)],
                              "competing_matches": len(matches),
                              "continuations": [{"source_start": i, "tail_start": i+len(probes),
                                                 "tail_ids": event["source"][i+len(probes):]} for i in matches],
                              "additional_forwards": len(probe_times), "timing": probe_times}
    return values, tensors, timing
