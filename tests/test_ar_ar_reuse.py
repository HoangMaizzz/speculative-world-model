"""Regression checks for cached rollback and exact suffix alignment."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_ar_ar_reuse import CachedTarget, decode_spec, diagnose, reuse_summary


class FakeCache:
    def __init__(self):
        self.tokens = []

    def get_seq_length(self):
        return len(self.tokens)

    def crop(self, length):
        del self.tokens[length:]


def fake_forward(state, tokens):
    if state.cache is None:
        state.cache = FakeCache()
    predictions = []
    for token in tokens:
        state.cache.tokens.append(token)
        predictions.append(state.model(tuple(state.cache.tokens)))
    return predictions


class ARReuseTests(unittest.TestCase):
    def test_cached_native_rejection_followed_by_full_acceptance(self):
        prompt = [99]

        def target(ids):
            return len(ids)

        def drafter(ids):
            return 70 if ids == (99, 1) else len(ids)

        with patch.object(CachedTarget, "forward", fake_forward):
            result = decode_spec(target, drafter, prompt, set(), 9, 4, lambda model: None)
        self.assertEqual(result["token_ids"], list(range(1, 10)))
        first, second, last = result["rounds"]
        self.assertEqual(first["proposal"], [1, 70, 3, 4])
        self.assertEqual(first["decision"]["correction"], 2)
        self.assertEqual(second["proposal"], [3, 4, 5, 6])
        self.assertEqual(second["decision"]["bonus"], 7)
        self.assertEqual(last["proposal"], [8, 9])
        self.assertEqual([r["draft_forwards"] for r in result["rounds"]], [3, 4, 2])

    def test_terminal_draft_and_terminal_correction(self):
        for draft, anticipated in ((lambda ids: len(ids), [1, 2]), (lambda ids: 9, [1, 2])):
            with patch.object(CachedTarget, "forward", fake_forward):
                result = decode_spec(len, draft, [99], {2}, 20, 5, lambda model: None)
            self.assertEqual(result["token_ids"], anticipated)
            self.assertTrue(result["rounds"][-1]["decision"]["terminated"])

    def test_single_64_token_proposal_with_one_verification(self):
        with patch.object(CachedTarget, "forward", fake_forward):
            result = decode_spec(len, len, [99], set(), 65, 64, lambda model: None)
        step, = result["rounds"]
        self.assertEqual(step["proposal"], list(range(1, 65)))
        self.assertEqual(step["draft_forwards"], 63)  # The prompt prefill supplies the first prediction.
        self.assertEqual(step["decision"]["bonus"], 65)

    def test_exact_shift_union_and_next_round(self):
        gold = [90, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        first = dict(offset=0, proposal=[99, 8, 1, 2, 3, 4, 6, 7, 8, 9],
                     decision=dict(first_rejection=0, committed_tokens=[90], terminated=False))
        following = dict(offset=1, proposal=[1, 2, 3, 4, 88],
                         decision=dict(first_rejection=4, committed_tokens=[1, 2, 3, 4, 5],
                                       accepted_draft=4, terminated=False))
        spec = dict(token_ids=gold, rounds=[first, following])
        audit = diagnose(spec, gold, set(), 10)
        event, = audit["events"]
        same, shift = sorted(event["runs"], key=lambda r: abs(r["delta"]))
        self.assertEqual((same["delta"], same["length"]), (0, 4))
        self.assertEqual((shift["delta"], shift["length"], shift["destination_start"]), (-1, 4, 0))
        self.assertTrue(shift["next_accepts"])
        self.assertTrue(shift["next_reproposes"])
        report = reuse_summary([dict(id="q", reuse={"10": audit})], 10)
        self.assertEqual(report["counts"]["union_positions"], 8)
        self.assertEqual(report["counts"]["shift_extra_positions"], 4)
        self.assertEqual(report["token_coverage"]["union"], 0.8)
        mismatch = diagnose(dict(spec, token_ids=[98] + gold[1:]), gold, set(), 10)
        self.assertEqual(mismatch["attrition"]["after_ar_divergence"], 2)
        self.assertFalse(mismatch["events"])

    def test_64_token_reference_finds_a_span_beyond_32(self):
        future = list(range(100, 164))
        gold = [90] + future
        proposal = [91] + [777] * 4 + future[42:47] + [777] * 54
        step = dict(offset=0, proposal=proposal,
                    decision=dict(first_rejection=0, committed_tokens=[90], terminated=False))
        spec = dict(token_ids=gold, rounds=[step])
        self.assertFalse(diagnose(spec, gold, set(), 32)["events"][0]["runs"])
        run, = diagnose(spec, gold, set(), 64)["events"][0]["runs"]
        self.assertEqual((run["destination_start"], run["length"], run["delta"]), (42, 5, 38))


if __name__ == "__main__":
    unittest.main()
