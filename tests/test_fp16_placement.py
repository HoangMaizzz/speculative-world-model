"""Model placement contracts; mocked loading is not a two-GPU execution test."""
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import nullcontext

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(AVAILABLE, "Optional torch/transformers dependencies absent")
class FP16PlacementTests(unittest.TestCase):
    def test_release_clears_allocator_caches_on_both_gpus(self):
        from suffix_lab import runtime
        with patch("torch.cuda.device_count", return_value=2), \
             patch("torch.cuda.device", side_effect=lambda *a: nullcontext()) as device, \
             patch("torch.cuda.empty_cache") as empty:
            runtime.release(None)
        self.assertEqual([call.args[0] for call in device.call_args_list], [0, 1])
        self.assertEqual(empty.call_count, 2)
    def config(self):
        return {"target_precision": "fp16", "target_gpu0_layers": 18, "target_attention": "sdpa"}

    def test_verifier_uses_explicit_contiguous_layers_on_both_gpus(self):
        from suffix_lab import runtime
        model = SimpleNamespace()
        architecture = SimpleNamespace(model_type="qwen2", num_hidden_layers=28)
        with patch("transformers.AutoConfig.from_pretrained", return_value=architecture), \
             patch.object(runtime, "load_model", return_value=model) as load:
            loaded, precision, fallback = runtime.load_target("fixture", self.config())
        self.assertIs(loaded, model)
        self.assertEqual(precision, "fp16")
        self.assertIsNone(fallback)
        self.assertEqual(load.call_args.args, ("fixture", "cuda:0", "fp16"))
        mapping = load.call_args.kwargs["device_map"]
        self.assertEqual(mapping["model.embed_tokens"], 0)
        self.assertEqual(mapping["model.layers.17"], 0)
        self.assertEqual(mapping["model.layers.18"], 1)
        self.assertEqual(mapping["lm_head"], 1)
        self.assertEqual(set(mapping.values()), {0, 1})
        self.assertEqual(len([k for k in mapping if k.startswith("model.layers.")]), 28)

    def test_quantized_resume_is_rejected_before_loading(self):
        from suffix_lab import runtime
        with patch.object(runtime, "load_model") as load:
            for precision in ("int8", "nf4"):
                with self.assertRaisesRegex(ValueError, "fresh FP16"):
                    runtime.load_target("fixture", self.config(), required=precision)
            load.assert_not_called()

    def test_loader_passes_fp16_without_quantization_or_cpu_offload(self):
        import torch
        from suffix_lab import runtime
        model = SimpleNamespace(hf_device_map={"layer": 1})
        model.eval = lambda: model
        with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load:
            runtime.load_model("fixture", "cuda:1")
        self.assertEqual(load.call_args.kwargs["torch_dtype"], torch.float16)
        self.assertEqual(load.call_args.kwargs["device_map"], {"": "cuda:1"})
        self.assertNotIn("quantization_config", load.call_args.kwargs)
        model.hf_device_map = {"layer": "cpu"}
        with patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "offload"):
                runtime.load_model("fixture", "cuda:1")

    def test_sharded_timing_synchronizes_both_devices(self):
        from suffix_lab import runtime
        model = SimpleNamespace(hf_device_map={"emb": 0, "block": 1, "head": 1})
        with patch.object(runtime, "sync") as sync:
            runtime.sync_model(model)
        self.assertEqual([call.args[0] for call in sync.call_args_list], ["cuda:0", "cuda:1"])

    def test_failed_fp16_audit_does_not_reload_another_precision(self):
        from suffix_lab import runtime
        model = SimpleNamespace(device="cpu")
        with patch.object(runtime, "load_target", return_value=(model, "fp16", None)) as load, \
             patch.object(runtime, "target_consistency", return_value={"passed": False}), \
             patch.object(runtime, "release"):
            with self.assertRaises(runtime.TargetConsistencyError):
                runtime.checked_target("fixture", self.config(), [1, 2])
        self.assertEqual(load.call_count, 1)


if __name__ == "__main__":
    unittest.main()
