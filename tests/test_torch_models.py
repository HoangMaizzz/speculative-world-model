"""Optional CPU audits against tiny *random* official models, never quality tests.

Set SUFFIX_UPSTREAM_DIR to the hash-checked modeling/configuration snapshots
downloaded by scripts/review_upstream.py. Missing deps are explicitly skipped.
"""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(AVAILABLE, "Optional torch/transformers dependencies are absent")
class TinyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(123)

    def test_qwen_incremental_parallel_shift_and_greedy(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from suffix_lab.runtime import ARSession
        c = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512, eos_token_id=63)
        c._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(c).eval()
        session = ARSession(model)
        prefix = [1, 2, 3, 4, 5]
        a = session.draft(prefix, 12)
        self.assertEqual(a.tokens, session.draft(prefix, 12).tokens)
        self.assertEqual(a.tokens, session.score(prefix, a.tokens)["top1"])
        with torch.inference_mode():
            expected = []
            for i in range(12):
                ids = torch.tensor([prefix+a.tokens[:i]])
                expected.append(int(model(ids, use_cache=False).logits[0, -1].argmax()))
        self.assertEqual(expected, a.tokens)

    def test_official_diffusion_cache_full_and_anchors(self):
        import torch
        from suffix_lab.runtime import DLLMDrafter
        directory = Path(os.environ.get("SUFFIX_UPSTREAM_DIR", "upstream-review")).resolve()
        if not (directory/"modeling.py").exists():
            self.skipTest("Pinned upstream snapshots absent; run scripts/review_upstream.py")
        package = types.ModuleType("suffix_upstream_test")
        package.__path__ = [str(directory)]
        sys.modules[package.__name__] = package
        from importlib import import_module
        configuration = import_module("suffix_upstream_test.configuration")
        modeling = import_module("suffix_upstream_test.modeling")
        c = configuration.Fast_dLLM_QwenConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512, bd_size=32, eos_token_id=63)
        c._attn_implementation = "sdpa"
        model = modeling.Fast_dLLM_QwenForCausalLM(c).eval()
        config = {"physical_block_size": 32, "subblock_size": 8, "mask_id": 62}
        drafter = DLLMDrafter(model, config)
        for length in (1, 7, 8, 31, 32, 33, 63, 64, 65):
            with self.subTest(prefix_length=length):
                prefix = [1+i % 50 for i in range(length)]
                a = drafter.draft(prefix, 35)
                b = drafter.draft(prefix, 35, use_prefix_cache=False)
                self.assertEqual(a.tokens, b.tokens)
                self.assertEqual(a.stats["unmask_steps"], 35)
                self.assertEqual(a.stats["unmask_forwards"], 35)
                for anchor_set in ({0: 4, 9: 7, 34: 8}, {i: 3 for i in range(35)}):
                    x = drafter.draft(prefix, 35, anchor_set)
                    y = drafter.draft(prefix, 35, anchor_set, use_prefix_cache=False)
                    self.assertEqual(x.tokens, y.tokens)
                    self.assertTrue(all(x.tokens[i] == t for i, t in anchor_set.items()))
                    self.assertEqual(x.stats["unmask_steps"], 35-len(anchor_set))


if __name__ == "__main__":
    unittest.main()
