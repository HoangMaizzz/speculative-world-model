import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_savh_spans as bench


class SpanBenchmarkTest(unittest.TestCase):
    def test_exact_oracle_uses_a_shared_nonzero_shift(self):
        source = [1, 2, 3, 4, 5]
        reference = [0, 1, 2, 3, 4, 0, 2, 3, 4, 5]
        rows = np.array([[0, 4], [1, 5], [0, 5]])
        self.assertEqual(bench.match_labels(bench.longest(source, reference), rows).tolist(), [True, True, False])
        self.assertEqual(bench.longest(source, source)[0], 0)

    def test_append_matches_dense_intervals_containing_seed(self):
        seeds = np.array([[0, 4], [4, 8], [9, 13]])
        expected = {(a, b) for a, b in bench.dense_rows(13, 8).tolist()
                    if b - a > 4 and any(a <= i and j <= b for i, j in seeds)}
        self.assertEqual(set(map(tuple, bench.append_rows(seeds, 13, 8))), expected)

    def test_packing_and_threshold(self):
        rows = np.array([[0, 8], [4, 8], [8, 16], [16, 20]])
        scores = np.array([.91, .90, .80, .75])
        self.assertEqual(bench.pack(rows, scores).tolist(), [0, 2])
        points = list(bench.curve([(.91, True, 8, 0, "gsm8k:1"),
                                  (.80, False, 4, 1, "math500:2")], 2, 16))
        self.assertEqual(points[-1]["span_precision"], .5)
        self.assertAlmostEqual(points[-1]["token_precision"], 2 / 3)
        self.assertEqual(points[-1]["token_recall"], .5)

    def test_shared_head_scores_an_interval_the_same_in_either_batch(self):
        x = torch.rand(1, 11, 15)
        dense = torch.as_tensor(bench.dense_rows(11, 8))[None]
        indices = [1, 7, 11, 19]
        with torch.inference_mode():
            for kind in ("mlp", "conv", "attention", "biaffine"):
                head = bench.SpanHead(15, kind, 8).eval()
                enc = head.encode(x, torch.tensor([11]))
                self.assertTrue(torch.allclose(head.score_rows(enc, dense)[:, indices],
                                               head.score_rows(enc, dense[:, indices]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
