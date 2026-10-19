"""Pure scheduling contracts, shared by real inference and independent tests."""
from dataclasses import dataclass, field
from typing import Protocol
import math


@dataclass
class Draft:
    tokens: list[int]
    confidence: list[float | None]
    order: list[int | None]
    trace: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class Provider(Protocol):
    def predict(self, canvas: list[int], block_start: int,
                positions: list[int]) -> dict[int, tuple[int, float]]: ...
    def stats(self) -> dict: ...


def prediction_row(position: int, block_start: int) -> int | None:
    """None means carry the preceding physical block's last prediction row.

    Inside a block raw[j-1] predicts token j. Never duplicate raw[0] to
    predict a boundary token: upstream supplies that token with a seed.
    """
    return None if position == block_start else position - block_start - 1


def strict_top1(prefix, length, anchors, provider, mask_id=151665,
                block_size=32, subblock_size=8):
    """One mutable token per call, max confidence within earliest active subblock.

    Hard anchors are provisional. MASK padding outside the requested output is
    context only and is never silently emitted. Lowest position breaks ties.
    """
    if not prefix or length < 0 or block_size % subblock_size:
        raise ValueError("Invalid prefix/length/block geometry")
    if mask_id in prefix:
        raise ValueError("Committed prefix must not contain MASK")
    if any(not 0 <= i < length or token == mask_id for i, token in anchors.items()):
        raise ValueError("Invalid hard anchor")
    start, end = len(prefix), len(prefix) + length
    padded = ((end + block_size - 1) // block_size) * block_size
    canvas = list(prefix) + [mask_id] * (padded - start)
    for i, token in anchors.items():
        canvas[start + i] = token
    confidence, order, trace = [None] * length, [None] * length, []
    pending = {start + i for i in range(length) if i not in anchors}
    while pending:
        first = min(pending)
        active = sorted(p for p in pending if p // subblock_size == first // subblock_size)
        block_start = first // block_size * block_size
        predictions = provider.predict(canvas, block_start, active)
        if set(predictions) != set(active):
            raise RuntimeError("Provider returned the wrong prediction positions")
        if any(not math.isfinite(predictions[p][1]) or not 0 <= predictions[p][1] <= 1
               or predictions[p][0] == mask_id for p in active):
            raise RuntimeError("Invalid probability or MASK predicted as output")
        chosen = max(active, key=lambda p: (predictions[p][1], -p))
        token, prob = predictions[chosen]
        before = sum(canvas[p] == mask_id for p in range(start, end))
        canvas[chosen] = int(token)
        pending.remove(chosen)
        after = sum(canvas[p] == mask_id for p in range(start, end))
        if before - after != 1:
            raise AssertionError("A step must fill exactly one MASK")
        i = chosen - start
        confidence[i], order[i] = float(prob), len(trace)
        trace.append({"step": len(trace), "position": i, "absolute_position": chosen,
                      "block_start": block_start, "subblock": chosen // subblock_size,
                      "prediction_row": prediction_row(chosen, block_start),
                      "confidence": float(prob), "token": int(token),
                      "mask_before": before, "mask_after": after})
        if canvas[:start] != list(prefix) or any(canvas[start+i] != t for i, t in anchors.items()):
            raise AssertionError("Prefix/KEEP mutated")
    result = canvas[start:end]
    if mask_id in result or len(trace) != length - len(anchors):
        raise AssertionError("Incomplete proposal or incorrect step count")
    return Draft(result, confidence, order, trace,
                 {**provider.stats(), "unmask_steps": len(trace), "hard_keep_count": len(anchors)})

