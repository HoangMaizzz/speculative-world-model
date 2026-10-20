"""Reference-audited offline verification with an actual model replay on mismatch."""
from .metrics import prefix_length


class VerificationError(AssertionError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__("Canonical target/reference consistency failed; inspect saved evidence")


def consistent(proposal, feedback, gold):
    k = prefix_length(proposal, feedback["top1"])
    return k == prefix_length(proposal, gold) and (k == len(proposal) or feedback["top1"][k] == gold[k])


def checked_score(session, ref, offset, proposal, top_k=8):
    """Never substitute gold logits/tokens. Gold only audits independent model outputs.

    This mismatch trigger uses saved reference labels and is an OFFLINE diagnostic
    policy, not a deployable online verification algorithm or speedup benchmark.
    """
    prefix = ref["prompt_ids"]+ref["tokens"][:offset]
    gold = ref["tokens"][offset:offset+len(proposal)]
    if not proposal or len(gold) != len(proposal):
        raise ValueError("Verification requires a nonempty proposal with complete reference labels")
    parallel = session.score(prefix, proposal, top_k)
    if consistent(proposal, parallel, gold):
        parallel["verification_mode"] = "parallel_reference_audit_passed"
        return parallel
    try:
        replay = session.score_replay(ref["prompt_ids"], ref["tokens"][:offset], proposal, top_k)
    except VerificationError as error:
        raise VerificationError({"parallel": parallel, "replay_failure": error.evidence}) from error
    if not consistent(proposal, replay, gold):
        raise VerificationError({"parallel": parallel, "replay": replay, "reference": gold, "proposal": proposal})
    result = dict(replay)
    result["verification_mode"] = "canonical_replay_after_parallel_disagreement"
    result["parallel_disagreement"] = parallel
    result["replay_timing"] = {k: replay[k] for k in ("latency_ms", "wall_latency_ms", "input_tokens", "timed_stage_calls") if k in replay}
    # Operational offline cost includes BOTH attempts and canonical prefix replay.
    for key in ("latency_ms", "wall_latency_ms", "input_tokens", "timed_stage_calls"):
        if key in parallel and key in replay:
            result[key] = parallel[key]+replay[key]
    if "compute_by_device_ms" in replay:
        result["compute_by_device_ms"] = {d: replay["compute_by_device_ms"][d]+parallel["compute_by_device_ms"][d]
                                         for d in replay["compute_by_device_ms"]}
    return result
