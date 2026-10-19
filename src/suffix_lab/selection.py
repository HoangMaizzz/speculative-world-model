"""Non-oracle selectors cannot receive target-reference tokens."""
from dataclasses import dataclass
import random
import math


@dataclass(frozen=True)
class Observation:
    old_tokens: tuple[int, ...]
    draft_confidence: tuple[float, ...]
    verifier_probability: tuple[float, ...]
    verifier_top1: tuple[int, ...]


def select_keep(obs, method, ratio, seed=42):
    n = len(obs.old_tokens)
    if any(len(feature) != n for feature in (obs.draft_confidence, obs.verifier_probability, obs.verifier_top1)):
        raise ValueError("Selector feature lengths differ")
    if not 0 <= ratio <= 1:
        raise ValueError("KEEP ratio must be within [0,1]")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in obs.draft_confidence + obs.verifier_probability):
        raise ValueError("Selector confidence is invalid")
    k = min(n, max(0, int(math.floor(n * ratio + 0.5))))
    if method == "random":
        return sorted(random.Random(seed).sample(range(n), k))
    if method == "drafter_confidence":
        scores = obs.draft_confidence
    elif method == "verifier_confidence":
        scores = obs.verifier_probability
    elif method == "old_agreement_ranked":
        scores = [(t == v, p) for t, v, p in zip(obs.old_tokens, obs.verifier_top1, obs.verifier_probability)]
    elif method == "old_agreement":
        return [i for i, (t, v) in enumerate(zip(obs.old_tokens, obs.verifier_top1)) if t == v]
    else:
        raise ValueError(method)
    return sorted(sorted(range(n), key=lambda i: (scores[i], -i), reverse=True)[:k])


def oracle_masks(old, reference, limit=8, seed=42):
    correct = [i for i, (x, y) in enumerate(zip(old, reference)) if x == y]
    masks = [correct, []]
    rng = random.Random(seed)
    for ratio in (0.25, 0.5, 0.75):
        k = int(len(correct) * ratio)
        masks.extend([correct[:k], sorted(rng.sample(correct, k))])
    result = []
    for mask in masks:
        if mask not in result:
            result.append(mask)
    return result[:max(1, limit)]

