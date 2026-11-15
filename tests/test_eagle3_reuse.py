"""Regression checks for tree geometry, exact reuse, and the upstream integration path."""

import importlib.util
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eagle3_qwen_runtime import EagleQwen, check_vocab_map, trace_parents
from eagle3_reuse_metrics import diagnose, summarize_reuse


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.gold = [9, 20, 21, 22, 4, 5, 6, 7, 30, 31]
        self.row = dict(offset=0, tokens=[9, 20, 21, 99, 4, 5, 6, 7, 98, 4, 5, 6, 7, 97],
                        depths=[0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 5],
                        paths=[[0, 1, 2, 3, 4, 5, 6, 7], [0, 8, 9, 10, 11, 12], [0, 8, 9, 10, 11, 13]],
                        committed=[9, 20, 21], next_seed=22, best_path=0)

    def test_tree_topology_and_bad_trace(self):
        self.assertEqual(trace_parents(self.row["paths"], self.row["depths"]),
                         [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 11])
        with self.assertRaisesRegex(ValueError, "Missing"):
            trace_parents(self.row["paths"][:2], self.row["depths"])
        with self.assertRaisesRegex(ValueError, "multiple parents"):
            trace_parents([[0, 1, 3], [0, 2, 3]], [0, 1, 1, 2])

    def test_coverage_counts_exact_spans_and_deduplicates_destinations(self):
        result = diagnose(dict(token_ids=self.gold, rounds=[self.row]), dict(token_ids=self.gold), {0})
        event, = result["10"]["events"]
        self.assertEqual(event["corrected_offset"], 4)
        self.assertEqual([(r["source_start"], r["delta"], r["paths"]) for r in event["runs"]],
                         [(0, 0, [0]), (-2, 2, [1])])
        cases = [dict(id="math500:0", reuse=result)]
        counts = summarize_reuse(cases, 10, False)["counts"]
        self.assertEqual((counts["either_positions"], counts["extra_shift_positions"], counts["reference_tokens"]),
                         (4, 0, 6))
        self.assertEqual(counts["reusable_source_nodes"], 8)
        self.assertEqual(summarize_reuse(cases, 10, True)["counts"]["shifted_positions"], 0)

    def test_rejects_unaligned_correction_and_stops_at_eos(self):
        ar = dict(token_ids=self.gold)
        mismatch = dict(self.row, next_seed=96)
        result = diagnose(dict(token_ids=self.gold, rounds=[mismatch]), ar, {0})
        self.assertEqual(result["10"]["attrition"]["outside_greedy_parity"], 1)
        result = diagnose(dict(token_ids=self.gold, rounds=[self.row]), ar, {6})
        self.assertEqual(result["10"]["attrition"]["short_reference"], 1)

    def test_vocabulary_mapping_matches_exported_offset_convention(self):
        check_vocab_map([0, 2, 3], [True, False, False, True, False, True], 3, 6)
        with self.assertRaisesRegex(ValueError, "disagree"):
            check_vocab_map([0, 3, 5], [True, False, False, True, False, True], 3, 6)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("eagle"),
                     "Needs torch/Transformers and the pinned upstream EAGLE source")
class UpstreamIntegration(unittest.TestCase):
    def test_real_upstream_small_qwen_tree_and_kv(self):
        import torch
        from eagle.model.cnets import Model
        from eagle.model.configs import EConfig
        from eagle.model.modeling_qwen2_kv import Qwen2ForCausalLM
        from transformers import Qwen2Config

        torch.manual_seed(42)
        cfg = dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_attention_heads=4,
                   num_key_value_heads=2, max_position_embeddings=96)
        target_config = Qwen2Config(**cfg, num_hidden_layers=6)
        target_config._attn_implementation = "eager"
        target = Qwen2ForCausalLM(target_config).eval()
        head = Model(EConfig(**cfg, num_hidden_layers=1, draft_vocab_size=4),
                     total_tokens=12, depth=3, top_k=4, load_emb=False).eval()
        head.embed_tokens.weight = target.model.embed_tokens.weight
        head.init_tree()
        runner = EagleQwen(target, head, (0, 2, 4), 96)
        with torch.inference_mode():
            random_ar = runner.decode_reference([1, 2, 3], {31}, 8)
            random_tree = runner.decode_tree([1, 2, 3], {31}, 8)
            target.lm_head.weight.zero_()
            ar = runner.decode_reference([1, 3, 4], {31}, 20)
            tree = runner.decode_tree([1, 3, 4], {31}, 20)
            changed_prompt = runner.decode_tree([3, 5], {0}, 20)
        self.assertEqual(random_ar["token_ids"], random_tree["token_ids"])
        self.assertEqual(ar["token_ids"], tree["token_ids"])
        self.assertEqual(changed_prompt["token_ids"], [0])
        self.assertGreater(len(tree["rounds"]), 1)
        self.assertTrue(any(r["accepted_draft"] for r in tree["rounds"]))
        self.assertTrue(all(len(r["tokens"]) == 12 for r in tree["rounds"]))


if __name__ == "__main__":
    unittest.main()
