import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_ar_suffix_redraft64 import ARMS, CachedDrafter, exact_labels, paired_case, select_spans


class ARSuffixRedraftTests(unittest.TestCase):
    def test_selection_allows_multiple_compatible_spans_and_abstention(self):
        candidates = [(0, 1, 4), (3, 4, 4), (5, 8, 4), (8, 0, 4)]
        scores = [.9, .8, .7, .6]
        self.assertEqual(select_spans(candidates, scores, range(4), .65), [candidates[0], candidates[2]])
        self.assertEqual(select_spans(candidates, scores, range(4), .95), [])

    def test_wrong_predicted_tokens_reach_redrafter_unchanged(self):
        case = dict(old=[0, 1, 2, 3, 4, 5, 6, 7], future=[1, 2, 3, 4, 9, 9, 9, 9])
        chosen = [(0, 3, 4)]
        self.assertEqual(exact_labels(case, [(1, 0, 4), *chosen, (1, 6, 4)]), [True, False, False])
        with patch("benchmark_ar_suffix_redraft64.evaluate_arms", return_value={}) as eval_arms:
            result = paired_case(case, {"spans": chosen}, None, None, set())
        plans = eval_arms.call_args.args[1]
        self.assertEqual(plans[ARMS[0]], {})
        self.assertEqual(plans[ARMS[1]], {i: i + 1 for i in range(4)})
        self.assertEqual(plans[ARMS[2]], {3 + i: i for i in range(4)})
        self.assertEqual((result["predicted_correct"], result["predicted_wrong"]), (0, 4))

    def test_anchor_attention_mask_spans_the_64_canvas(self):
        class Mask:
            def __init__(self, n, m):
                self.rows = [[False] * m for _ in range(n)]

            def __setitem__(self, item, value):
                rows, cols = item
                for row in self.rows[rows]:
                    for col in range(*cols.indices(len(row))):
                        row[col] = value

        drafter = object.__new__(CachedDrafter)
        drafter.redrafting = True
        drafter.draft_length = 64
        drafter.native_mask = lambda length, block, cache: Mask(length, length + cache)
        result = drafter.eval_mask(None, 69, 32, 64).rows
        self.assertFalse(any(row[-1] for row in result[:4]))
        self.assertTrue(all(all(row) for row in result[4:]))
