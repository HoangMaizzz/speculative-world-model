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
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
from contextlib import redirect_stdout
import io

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
        replay = session.score_replay(prefix, a.tokens[:5], a.tokens[5:])
        self.assertEqual(replay["top1"], a.tokens[5:])
        self.assertEqual(replay["replay_forwards"], 12)

    def test_constrained_ar_forces_into_kv_and_preserves_causal_negative_control(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from suffix_lab.runtime import ARSession, constrained_ar_consistency
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512, eos_token_id=63)
        config._attn_implementation = "sdpa"
        session = ARSession(Qwen2ForCausalLM(config).eval())
        self.assertTrue(constrained_ar_consistency(session, [1, 2, 3, 4, 5], 12)["passed"])
        anchors = {0: 17, 3: 23, 7: 11}
        result = session.draft([1, 2, 3], 8, anchors=anchors)
        expected = []
        with torch.inference_mode():
            for i in range(8):
                logits = session.model(torch.tensor([[1, 2, 3]+expected]), use_cache=False).logits[0, -1].float()
                token = anchors[i] if i in anchors else int(logits.argmax())
                expected.append(token)
                self.assertAlmostEqual(result.confidence[i], float(logits.softmax(-1)[token]), places=6)
        self.assertEqual(result.tokens, expected)
        self.assertEqual(result.stats["total_forwards"], 8)
        self.assertEqual(result.stats["unmask_steps"], 5)
        self.assertEqual([result.order[i] for i in anchors], [None]*3)
        with self.assertRaisesRegex(ValueError, "fixed horizon"):
            session.draft([1, 2], 8, {63}, anchors=anchors)

    def test_tied_logits_use_same_greedy_token_in_draft_and_score(self):
        import torch
        from suffix_lab.runtime import ARSession, target_consistency
        class TiedModel:
            device = torch.device("cpu")
            def __call__(self, ids, **kwargs):
                n = min(ids.shape[1], kwargs.get("logits_to_keep", ids.shape[1]))
                return SimpleNamespace(logits=torch.zeros(1, n, 1000), past_key_values=None)
        session = ARSession(TiedModel())
        draft = session.draft([10, 11, 12], 4)
        self.assertEqual(draft.tokens, [0]*4)
        for k in (2, 8, 16):
            feedback = session.score([10, 11, 12], draft.tokens, k)
            self.assertEqual(feedback["top1"], [0]*4)
            self.assertEqual(feedback["logit_margin"], [0.0]*4)
            for ids in feedback["topk_ids"]:
                self.assertEqual(ids[0], 0)
                self.assertEqual(len(set(ids)), k)
        self.assertTrue(target_consistency(session, [10, 11], 4)["passed"])

    def test_target_consistency_distinguishes_repeatability_and_parallel_errors(self):
        from suffix_lab.runtime import target_consistency
        from suffix_lab.core import Draft
        class FaultySession:
            def __init__(self, repeatable):
                self.calls, self.repeatable = 0, repeatable
            def draft(self, prefix, length):
                self.calls += 1
                tokens = [3, 4] if self.repeatable or self.calls == 1 else [3, 8]
                return Draft(tokens, [], [])
            def score(self, prefix, proposal, top_k=8):
                return {"top1": [3, 9] if len(proposal) == 2 else [4],
                        "logit_margin": [1.0, 0.125] if len(proposal) == 2 else [0.25]}
        for repeatable in (True, False):
            report = target_consistency(FaultySession(repeatable), [1, 2], 2)
            self.assertFalse(report["passed"])
            self.assertEqual(report["repeatable"], repeatable)
            self.assertFalse(report["parallel_shifted_equal"])
            row = report["first_mismatches"][0]
            self.assertEqual(row["position"], 1)
            self.assertEqual(row["cached_token"], 4)
            self.assertEqual(row["parallel_token"], 9)
            self.assertEqual(row["full_prefix_token"], 4)

    def test_future_tail_cannot_change_earlier_verifier_predictions(self):
        from suffix_lab.runtime import target_consistency
        from suffix_lab.core import Draft
        class TailDependentSession:
            def draft(self, prefix, length):
                return Draft([3, 4], [], [])
            def score(self, prefix, proposal, top_k=8):
                return {"top1": [3, 4] if proposal == [3, 4] else ([9, 4] if len(proposal) == 2 else [3]),
                        "logit_margin": [0.5]*len(proposal)}
        report = target_consistency(TailDependentSession(), [1, 2], 2)
        self.assertTrue(report["repeatable"])
        self.assertTrue(report["parallel_shifted_equal"])
        self.assertFalse(report["causal_tail_equal"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["first_mismatches"][0]["tail_changed_token"], 9)

    def test_preflight_failure_records_evidence_and_invalidates_stale_pass(self):
        from suffix_lab import runtime
        fake_tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2, 3])
        audit = {"passed": False, "repeatable": True, "parallel_shifted_equal": False,
                 "first_mismatches": [{"position": 1, "cached_token": 4, "parallel_token": 9}]}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output/"preflight.json").write_text('{"passed": true}')
            with patch("torch.cuda.device_count", return_value=2), \
                 patch.object(runtime, "tokenizers", return_value=(fake_tokenizer, {})), \
                 patch.object(runtime, "load_target", return_value=(SimpleNamespace(device="cpu"), "int8", None)), \
                 patch.object(runtime, "release"), \
                 patch.object(runtime, "target_consistency", return_value=audit):
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(AssertionError, "preflight_failure.json"):
                    runtime.preflight({"paths": {runtime.TARGET: "fixture"}, "content_digest": "fixture"},
                                      {"seed": 42}, output)
            report = json.loads((output/"preflight_failure.json").read_text())
            self.assertEqual(report["target_audit"], audit)
            self.assertEqual(report["stage"], "target_consistency")
            self.assertFalse(json.loads((output/"preflight.json").read_text())["passed"])

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

        # Exercise the NEW joint/sequential runner on the official tiny random
        # architecture. This establishes mechanics only, never pretrained quality.
        from suffix_lab.extension import audited_draft, sequential_draft
        event = {"prefix": [1+i % 50 for i in range(31)], "L": 13, "H": 21}
        first = audited_draft(drafter, event["prefix"], 13, {2: 4, 3: 6})
        first["id"] = "tiny-short"
        sequential = sequential_draft(drafter, event, first)
        joint = audited_draft(drafter, event["prefix"], 21, {2: 4, 3: 6})
        self.assertEqual(sequential["tokens"][:13], first["tokens"])
        for result in (sequential, joint):
            self.assertEqual(result["stats"]["unmask_steps"], 19)
            self.assertEqual(len(result["trace"]), 19)
            self.assertEqual(result["tokens"][2:4], [4, 6])


if __name__ == "__main__":
    unittest.main()
