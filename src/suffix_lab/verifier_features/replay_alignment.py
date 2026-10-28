"""Full-vocabulary reductions in FP32; no truncated-softmax probabilities."""
def distributions(logits, top_k):
    import torch
    z = logits.float()
    logp = z.log_softmax(-1)
    top1 = z.argmax(-1, keepdim=True)
    others = z.clone().scatter(1, top1, -torch.inf).topk(top_k-1, dim=-1).indices
    ids = torch.cat([top1, others], -1)
    ordered = z.gather(1, ids)
    return {"top1": top1[:, 0].tolist(), "topk_ids": ids.tolist(),
            "topk_logp": logp.gather(1, ids).tolist(),
            "top1_probability": logp.gather(1, top1).exp()[:, 0].tolist(),
            "entropy": (-(logp.exp()*logp).sum(-1)).tolist(),
            "logit_margin": (ordered[:, 0]-ordered[:, 1]).tolist()}, logp


def draft_features(logits, tokens, top_k):
    import torch
    z = logits.float()
    result, logp = distributions(z, top_k)
    ids = torch.tensor(tokens, device=z.device)[:, None]
    chosen = z[:-1].gather(1, ids)
    result.update(verifier_logp_draft=logp[:-1].gather(1, ids)[:, 0].tolist(),
                  verifier_rank_draft=(1+(z[:-1] > chosen).sum(-1)).tolist(),
                  verifier_tied_with_top1=(z[:-1].amax(-1) == chosen[:, 0]).tolist(),
                  verifier_greedy_match=(z[:-1].argmax(-1) == ids[:, 0]).tolist(),
                  verifier_gap_to_top1=(z[:-1].amax(-1)-chosen[:, 0]).tolist())
    return result, logp
