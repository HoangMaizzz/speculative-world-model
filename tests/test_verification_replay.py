"""A shape-dependent model reproduces disagreement; no gold-token substitution."""
import importlib.util
import unittest
from types import SimpleNamespace
from suffix_lab.verification import checked_score, VerificationError


class ReplayPolicyTests(unittest.TestCase):
    def test_passing_parallel_does_not_invoke_replay(self):
        class Session:
            def score(self, prefix, proposal, top_k):
                return {"top1": [3, 4], "latency_ms": 2.}
            def score_replay(self, *args):
                raise AssertionError("unexpected replay")
        result = checked_score(Session(), {"prompt_ids": [1], "tokens": [3, 4]}, 0, [3, 9])
        self.assertEqual(result["verification_mode"], "parallel_reference_audit_passed")

    def test_model_replay_overrides_disagreement_and_both_attempt_costs_are_kept(self):
        class Session:
            def score(self, *args):
                return {"top1": [3, 8], "latency_ms": 2., "wall_latency_ms": 3., "input_tokens": 4,
                        "timed_stage_calls": 32, "compute_by_device_ms": {"cuda:0": 1., "cuda:1": 1.}}
            def score_replay(self, *args):
                return {"top1": [3, 4], "latency_ms": 10., "wall_latency_ms": 11., "input_tokens": 7,
                        "timed_stage_calls": 128, "compute_by_device_ms": {"cuda:0": 6., "cuda:1": 4.}}
        result = checked_score(Session(), {"prompt_ids": [1], "tokens": [3, 4]}, 0, [3, 8])
        self.assertEqual(result["top1"], [3, 4])
        self.assertEqual(result["parallel_disagreement"]["top1"], [3, 8])
        self.assertEqual(result["latency_ms"], 12.)
        self.assertEqual(result["wall_latency_ms"], 14.)
        self.assertEqual(result["timed_stage_calls"], 160)
        self.assertEqual(sum(result["compute_by_device_ms"].values()), 12.)

    def test_wrong_replay_is_a_hard_failure_not_replaced_with_reference(self):
        class Session:
            def score(self, *args):
                return {"top1": [8, 8]}
            def score_replay(self, *args):
                return {"top1": [9, 9]}
        with self.assertRaises(VerificationError) as error:
            checked_score(Session(), {"prompt_ids": [1], "tokens": [3, 4]}, 0, [3, 8])
        self.assertEqual(error.exception.evidence["replay"]["top1"], [9, 9])


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch absent")
class NativeReplayTests(unittest.TestCase):
    def test_original_prefill_and_tokenwise_committed_prefix_are_replayed(self):
        import torch
        from suffix_lab.runtime import ARSession
        class ShapeSensitiveModel:
            device = torch.device("cpu")
            def __init__(self):
                self.calls = []
            def __call__(self, ids, past_key_values=None, use_cache=False, **kwargs):
                values = ids[0].tolist()
                self.calls.append((values, use_cache))
                history = list(past_key_values or [])
                logits = []
                for token in values:
                    history.append(token)
                    winner = (sum(history)+1)%32
                    if not use_cache:  # controlled drift only in the parallel path
                        winner = (winner+1)%32
                    row = torch.full((32,), -10.)
                    row[winner] = 10.
                    logits.append(row)
                keep = kwargs.get("logits_to_keep", len(logits))
                return SimpleNamespace(logits=torch.stack(logits[-keep:])[None], past_key_values=history)
        model = ShapeSensitiveModel()
        session = ARSession(model)
        prompt = [1, 2]
        reference = session.draft(prompt, 8).tokens
        proposal = list(reference[3:7])
        proposal[-1] = (proposal[-1]+5)%32
        model.calls.clear()
        result = checked_score(session, {"prompt_ids": prompt, "tokens": reference}, 3, proposal)
        self.assertEqual(result["top1"][:3], reference[3:6])
        self.assertEqual(result["verification_mode"], "canonical_replay_after_parallel_disagreement")
        cached = [ids for ids, cache in model.calls if cache]
        self.assertEqual(cached[0], prompt)
        self.assertTrue(all(len(ids) == 1 for ids in cached[1:]))
        self.assertEqual(result["replay_forwards"], 7)
        corrupted = reference[:]
        corrupted[1] = (corrupted[1]+1)%32
        with self.assertRaises(VerificationError) as error:
            session.score_replay(prompt, corrupted[:3], proposal)
        self.assertEqual(error.exception.evidence["stage"], "committed_prefix_replay")


if __name__ == "__main__":
    unittest.main()
