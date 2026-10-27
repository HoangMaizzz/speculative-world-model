"""Online guard correctness with adversarial parallel feedback; no future labels."""
from types import SimpleNamespace
import unittest
from suffix_lab.canonical_guard import guard_feedback
from suffix_lab.long60_plan import decide


class Cursor:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.current = next(self.tokens)
        self.forwards = self.input_tokens = 0
        self.advanced = []
    def prediction(self):
        return self.current, .01
    def advance(self, token):
        assert token == self.current
        self.advanced.append(token)
        self.forwards += 1
        self.input_tokens += 1
        self.current = next(self.tokens, 999)


class CanonicalGuardTests(unittest.TestCase):
    def test_false_parallel_accept_is_corrected_online(self):
        parallel = {"top1_with_bonus": [1, 9, 3, 4], "logit_margin": [.1]*4, "forwards": 1}
        cursor = Cursor([1, 2, 3, 4])
        f = guard_feedback(parallel, [1, 9, 3], cursor, set())
        self.assertEqual(decide([1, 9, 3], f["top1_with_bonus"], set(), 20)["committed_tokens"], [1, 2])
        self.assertEqual(cursor.advanced, [1, 2])
        self.assertEqual(f["parallel_feedback"]["top1_with_bonus"], [1, 9, 3, 4])
        self.assertEqual(f["canonical_guard"]["numerical_disagreements"][0]["position"], 1)

    def test_false_parallel_reject_does_not_reduce_canonical_acceptance(self):
        parallel = {"top1_with_bonus": [9, 9, 9, 9], "forwards": 1}
        f = guard_feedback(parallel, [1, 2, 3], Cursor([1, 2, 3, 4, 5]), set())
        self.assertEqual(decide([1, 2, 3], f["top1_with_bonus"], set(), 20)["committed_tokens"], [1, 2, 3, 4])
        self.assertEqual(f["canonical_guard"]["checked_rows"], 4)

    def test_correction_eos_never_advances_into_tail(self):
        cursor = Cursor([1, 99, 3, 4])
        f = guard_feedback({"top1_with_bonus": [1, 2, 3, 4]}, [1, 2, 3], cursor, {99})
        self.assertEqual(cursor.advanced, [1, 99])
        self.assertTrue(decide([1, 2, 3], f["top1_with_bonus"], {99}, 20)["terminated"])
        self.assertEqual(f["canonical_guard"]["checked_rows"], 2)

    def test_accepted_eos_does_not_require_bonus(self):
        cursor = Cursor([1, 99, 77])
        f = guard_feedback({"top1_with_bonus": [1, 99, 77]}, [1, 99], cursor, {99})
        self.assertEqual(f["canonical_guard"]["checked_rows"], 2)
        self.assertEqual(decide([1, 99], f["top1_with_bonus"], {99}, 20)["committed_tokens"], [1, 99])

    def test_real_extra_forwards_are_counted(self):
        f = guard_feedback({"top1_with_bonus": [1, 2, 3], "forwards": 1, "input_tokens": 80},
                           [1, 2], Cursor([1, 2, 3, 4]), set())
        self.assertEqual(f["forwards"], 4)
        self.assertEqual(f["input_tokens"], 83)


if __name__ == "__main__":
    unittest.main()
