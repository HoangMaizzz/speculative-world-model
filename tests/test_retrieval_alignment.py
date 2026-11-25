"""Regression checks for winning-path logits and end-to-end branch KV rollback."""
from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_retrieval_alignment as bench
from retrieval_alignment_core import Index, decode, generate, ids_tensor


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(153)
        torch.set_num_threads(1)
        config = Qwen2Config(vocab_size=67, hidden_size=32, intermediate_size=48,
                             num_hidden_layers=2, num_attention_heads=4,num_key_value_heads=2)
        self.q = Qwen2ForCausalLM(config).eval()
        self.p = Qwen2ForCausalLM(config).eval()
        self.prompt = [5,7,9,11,13]

    @torch.inference_mode()
    def test_winning_logits(self):
        kv = self.q.model(ids_tensor(self.q, self.prompt[:-1]), use_cache=True).past_key_values
        expected, logits = generate(self.q, kv, self.prompt[-1], 16, set(), 16)
        index = Index()
        index.append(self.prompt, expected)
        kv = self.q.model(ids_tensor(self.q,self.prompt[:-1]), use_cache=True).past_key_values
        counts = Counter()
        got, rows = decode(self.q, kv, self.prompt, 16, set(), [(index,1.)], 16, counts, True)
        self.assertEqual(expected, got)
        self.assertEqual(kv.get_seq_length(), len(self.prompt)+15)
        self.assertLess(counts["draft_forwards"], 16)
        torch.testing.assert_close(torch.stack(rows), torch.stack(logits), rtol=1e-4, atol=2e-6)

    @torch.inference_mode()
    def test_hit_and_rollback(self):
        corpus = Index()
        empty = set()
        kv = self.p.model(ids_tensor(self.p,self.prompt[:-1]),use_cache=True).past_key_values
        gold, _ = generate(self.p, kv, self.prompt[-1], 36, empty)
        def force_hit(rows, selector, prefix, draft, old, off, *args):
            for j,(proposed,correct) in enumerate(zip(draft,gold[off:])):
                if proposed != correct:
                    return j, correct, 1.
            return None
        with patch.object(bench, "select", side_effect=force_hit):
            for arm in bench.ARMS:
                r = bench.rollout(self.q,self.p,self.prompt,empty,len(gold),arm,corpus,None)
                self.assertEqual(r["tokens"], gold)
                if arm in ("C","D"):
                    self.assertGreater(r["counts"]["branch_hits"],0)
                    if arm == "C":
                        self.assertEqual(r["counts"]["regrow_calls"],0)


if __name__ == "__main__":
    unittest.main()
