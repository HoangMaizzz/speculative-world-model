"""Matched Qwen-only FP16 sharding, sparse capture and separated wall/compute."""
import importlib.metadata
import platform
import time
from contextlib import nullcontext
from suffix_lab.assets import TARGET
from suffix_lab.runtime import configure_numerics, load_target, sync_model, memory
from .hidden_capture import HiddenCapture


class Target:
    def __init__(self, model, layers, timer=None):
        self.model, self.layers, self.timer = model, layers, timer

    @classmethod
    def load(cls, cp, layers):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from suffix_lab.timing import StageTimer
        if torch.cuda.device_count() != 2:
            raise RuntimeError("Select Kaggle GPU T4 x2")
        if [torch.cuda.get_device_name(i) for i in range(2)] != cp.runtime["environment"]["gpus"]:
            raise ValueError("Matched replay requires the same GPU types as the source checkpoint (two T4s)")
        for package in ("torch", "transformers", "accelerate"):
            current = importlib.metadata.version(package)
            expected = cp.runtime["environment"]["packages"][package]
            if current != expected:
                raise ValueError(f"Matched replay requires {package}={expected}; got {current}")
        configure_numerics(cp.manifest["config"])
        revision = cp.manifest["models"][TARGET]
        print(f"[model] Qwen only, revision={revision}; FP16 split18/10, no drafter", flush=True)
        path = snapshot_download(TARGET, revision=revision, ignore_patterns=["*.bin", "*.pt", "*.ot", "onnx/*"])
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        import hashlib
        import json
        vocab_hash = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
        if vocab_hash != cp.runtime["tokenizer"]["shared_vocab_sha256"]:
            raise ValueError("Tokenizer identity differs")
        model, _, _ = load_target(path, cp.manifest["config"])
        if model.config.num_hidden_layers != 28 or model.config.hidden_size != 3584:
            raise ValueError("Unexpected Qwen architecture")
        model._suffix_exclude_peer_transfers = False
        return cls(model, layers, StageTimer(model))

    def environment(self):
        import torch
        import subprocess
        try:
            drivers = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                              text=True, stderr=subprocess.DEVNULL).splitlines()
        except (OSError, subprocess.CalledProcessError):
            drivers = None
        return {"python": platform.python_version(), "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "transformers": importlib.metadata.version("transformers"),
                "accelerate": importlib.metadata.version("accelerate"),
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "driver": drivers, "attention": self.model.config._attn_implementation,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
                "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                "device_map": {k: str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()},
                "hidden_size": self.model.config.hidden_size}

    def forward(self, ids, keep, positions=None, *, final_norm=False, use_cache=False, cache=None):
        import torch
        sync_model(self.model)
        started = time.perf_counter()
        capture = HiddenCapture(self.model, self.layers, positions, final_norm) if positions is not None else None
        with torch.inference_mode(), capture if capture else nullcontext():
            with self.timer.measure() if self.timer else nullcontext():
                out = self.model(torch.tensor([ids], device=self.model.device), use_cache=use_cache,
                                 past_key_values=cache, logits_to_keep=keep)
            sync_model(self.model)
        wall = (time.perf_counter()-started)*1000
        timing = {"forward_count": 1, "input_tokens": len(ids), "wall_ms_including_transfers_capture": wall,
                  "compute_ms_excluding_peer_transfers": None}
        if self.timer:
            report = self.timer.report(wall)
            timing.update(compute_ms_excluding_peer_transfers=report["latency_ms"],
                          compute_by_device_ms=report["compute_by_device_ms"],
                          compute_timing_policy=report["latency_policy"])
        return out.logits[0].float(), (capture.values if capture else {}), out.past_key_values, timing

    def gpu_memory(self):
        import torch
        return [{**m, "peak_allocated_gib": torch.cuda.max_memory_allocated(m["device"])/2**30,
                 "peak_reserved_gib": torch.cuda.max_memory_reserved(m["device"])/2**30} for m in memory()]
