"""Tiny random CPU models check mechanics only, never pretrained GPU performance."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(AVAILABLE, "Optional local CPU model dependencies missing")
class Long60ModelTests(unittest.TestCase):
    def test_real_qwen_parallel_bonus_and_causal_future(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from suffix_lab.runtime import ARSession
        from suffix_lab.long60 import ParallelVerifier
        from suffix_lab.long60_plan import decide
        torch.set_num_threads(1)
        torch.manual_seed(123)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        config._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(config).eval()
        prefix = [1, 2, 3, 4, 5]
        gold = ARSession(model).draft(prefix, 61).tokens
        scorer = ParallelVerifier(model)
        result = scorer.score(prefix, gold[:60])
        self.assertEqual(result["top1_with_bonus"], gold)
        self.assertEqual(decide(gold[:60], result["top1_with_bonus"], set(), 100)["committed_tokens"], gold)
        changed = list(gold[:60])
        changed[-1] = (changed[-1]+1) % 64
        feedback = scorer.score(prefix, changed)
        self.assertEqual(feedback["top1_with_bonus"][:60], gold[:60])
        self.assertEqual(decide(changed, feedback["top1_with_bonus"], set(), 100)["first_rejection"], 59)

    def test_official_diffusion_sixty_cached_full(self):
        import torch
        from importlib import import_module
        from suffix_lab.runtime import DLLMDrafter
        from suffix_lab.long60_plan import validate_draft
        from dataclasses import asdict
        directory = Path(os.environ.get("SUFFIX_UPSTREAM_DIR", "upstream-review")).resolve()
        if not (directory/"modeling.py").exists():
            self.skipTest("Pinned official model source is absent")
        torch.set_num_threads(1)
        torch.manual_seed(123)
        package = types.ModuleType("suffix_long60_upstream")
        package.__path__ = [str(directory)]
        sys.modules[package.__name__] = package
        configuration = import_module(package.__name__+".configuration")
        modeling = import_module(package.__name__+".modeling")
        c = configuration.Fast_dLLM_QwenConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=512, bd_size=32, eos_token_id=63)
        c._attn_implementation = "sdpa"
        drafter = DLLMDrafter(modeling.Fast_dLLM_QwenForCausalLM(c).eval(),
                             {"physical_block_size": 32, "subblock_size": 8, "mask_id": 62})
        for phase in (0, 1, 7, 8, 31):
            prefix = [1+i % 50 for i in range(32+phase)]
            cached = asdict(drafter.draft(prefix, 60))
            full = asdict(drafter.draft(prefix, 60, use_prefix_cache=False))
            self.assertEqual(cached["tokens"], full["tokens"])
            validate_draft(cached, 60, prefix)
            self.assertEqual(cached["stats"]["unmask_forwards"], 60)


if __name__ == "__main__":
    unittest.main()
