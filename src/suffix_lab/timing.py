"""CUDA compute-stage timing inside Accelerate's dispatch/copy hooks."""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps


class StageTimer:
    """Sum non-nested Qwen stages on their own GPUs, excluding dispatch copies.

    Accelerate1.10.1 calls module._old_forward after pre_forward (input transfer)
    and before post_forward (output transfer). Instrument that inner callable.
    This is stage compute time, not wall time or a measurement of link latency.
    """
    def __init__(self, model):
        self.active, self.records = False, []
        self.stages = []
        for name, module in model.named_modules():
            parts = name.split(".")
            selected = name in ("model.embed_tokens", "model.rotary_emb", "model.norm", "lm_head")
            selected |= len(parts) == 3 and parts[:2] == ["model", "layers"] and parts[2].isdigit()
            if not selected:
                continue
            devices = {tensor.device for tensor in module.parameters()}
            devices.update(t.device for t in module.buffers())
            if len(devices) != 1 or next(iter(devices)).type != "cuda":
                raise RuntimeError(f"Compute timing requires one CUDA device per stage: {name}")
            device = next(iter(devices))
            attribute = "_old_forward" if hasattr(module, "_old_forward") else "forward"
            original = getattr(module, attribute)
            setattr(module, attribute, self.wrap(original, name, device))
            self.stages.append(name)
        if not self.stages:
            raise RuntimeError("No Qwen compute stages found; refusing to label wall time as compute time")

    def wrap(self, original, name, device):
        @wraps(original)
        def forward(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            import torch
            with torch.cuda.device(device):
                stream = torch.cuda.current_stream(device)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(stream)
                try:
                    return original(*args, **kwargs)
                finally:
                    end.record(stream)
                    self.records.append((name, str(device), start, end))
        return forward

    @contextmanager
    def measure(self):
        if self.active:
            raise RuntimeError("Nested timing would double count compute")
        self.records = []
        self.active = True
        try:
            yield
        except BaseException:
            self.records = []
            raise
        finally:
            self.active = False

    def report(self, wall_ms):
        # Caller synchronizes every model device before asking for elapsed time.
        totals = defaultdict(float)
        for name, device, start, end in self.records:
            totals[device] += float(start.elapsed_time(end))
        if not self.records:
            raise RuntimeError("No CUDA events captured; compute-only timing is unavailable")
        report = {"latency_ms": sum(totals.values()), "wall_latency_ms": wall_ms,
                  "compute_by_device_ms": dict(totals), "timed_stage_calls": len(self.records),
                  "latency_policy": "cuda_stage_events_excluding_dispatch_copies",
                  "timed_stages": self.stages}
        self.records = []
        return report
