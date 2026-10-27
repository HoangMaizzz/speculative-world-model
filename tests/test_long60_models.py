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
    def test_guarded_actual_target_kv_survives_deliberately_wrong_parallel_rows(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from suffix_lab.runtime import ARSession
        from suffix_lab.long60 import ParallelVerifier
        from suffix_lab.canonical_guard import GuardedVerifier
        from suffix_lab.long60_plan import decide
        torch.set_num_threads(1)
        torch.manual_seed(123)
        c = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        c._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(c).eval()
        class WrongParallel(ParallelVerifier):
            def score(self, prefix, proposal):
                f = super().score(prefix, proposal)
                f["top1_with_bonus"] = [(t+1) % 64 for t in f["top1_with_bonus"]]
                return f
        guard = GuardedVerifier(WrongParallel(model), set())
        prompt, output = [1, 2, 3, 4, 5], []
        canonical = ARSession(model).draft(prompt, 140).tokens
        guard.begin_prompt(prompt, [])
        for length in (64, 64):
            proposal = canonical[len(output):len(output)+length]
            f = guard.score(prompt+output, proposal)
            commit = decide(proposal, f["top1_with_bonus"], set(), 200)["committed_tokens"]
            output.extend(commit)
            self.assertTrue(f["canonical_guard"]["numerical_disagreements"])
        self.assertEqual(output, canonical[:130])
        guard.begin_prompt(prompt, output[:70])
        f = guard.score(prompt+output[:70], canonical[70:134])
        self.assertEqual(decide(canonical[70:134], f["top1_with_bonus"], set(), 200)["committed_tokens"], canonical[70:135])

    def test_guarded_cursor_rejects_noncanonical_resume_prefix(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from suffix_lab.canonical_guard import CanonicalCursor
        torch.set_num_threads(1)
        c = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        model = Qwen2ForCausalLM(c).eval()
        cursor = CanonicalCursor(model, set())
        cursor.begin([1, 2, 3], [])
        token, _ = cursor.prediction()
        with self.assertRaisesRegex(AssertionError, "not canonical"):
            cursor.begin([1, 2, 3], [(token+1) % 64])

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
        scorer = ParallelVerifier(model)
        for length in (60, 64):
            gold = ARSession(model).draft(prefix, length+1).tokens
            result = scorer.score(prefix, gold[:length])
            self.assertEqual(result["top1_with_bonus"], gold)
            self.assertEqual(decide(gold[:length], result["top1_with_bonus"], set(), 100)["committed_tokens"], gold)
            changed = list(gold[:length])
            changed[-1] = (changed[-1]+1) % 64
            feedback = scorer.score(prefix, changed)
            self.assertEqual(feedback["top1_with_bonus"][:length], gold[:length])
            self.assertEqual(decide(changed, feedback["top1_with_bonus"], set(), 100)["first_rejection"], length-1)

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
            for length in (60, 64):
                cached = asdict(drafter.draft(prefix, length))
                full = asdict(drafter.draft(prefix, length, use_prefix_cache=False))
                self.assertEqual(cached["tokens"], full["tokens"])
                validate_draft(cached, length, prefix)
                self.assertEqual(cached["stats"]["unmask_forwards"], length)


if __name__ == "__main__":
    unittest.main()
