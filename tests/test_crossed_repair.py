"""Independent expectations for content controls, causal forcing and gain labels."""
import json
from pathlib import Path
import unittest
from suffix_lab.assets import ROOT
from suffix_lab.metrics import redraft_metrics, event_gains
from suffix_lab.plans import conditioning_plans
from suffix_lab.experiment import offsets, progress_bin, sampling_coverage, stratified_events


class CrossedRepairTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/"configs/math_gsm8k.json").read_text())

    def test_injected_correct_content_is_not_recovered_old_token(self):
        m = redraft_metrics([1, 2, 3, 4], [1, 2, 3, 4], [1, 9, 3, 8], [1, 2, 3], {1: 2, 2: 3, 3: 4})
        self.assertEqual(m["anchor_count"], 3)
        self.assertEqual(m["anchor_precision"], 1)
        self.assertEqual(m["retained_count"], 1)
        self.assertEqual(m["recovered_old_tokens_in_prefix"], 1)

    def test_future_content_controls_share_positions_and_count(self):
        old, gold = list(range(20)), [9]*20
        plans, meta = conditioning_plans(old, gold, self.config(), 42)
        self.assertEqual(len(plans), 5)
        self.assertEqual(meta["anchor_count"], 4)
        positions = [[i for i, t in p["anchors"]] for p in plans]
        self.assertTrue(all(p == positions[0] for p in positions))
        self.assertTrue(all(i >= meta["probe_horizon"] for i in positions[0]))
        by_name = {p["method"]: p for p in plans}
        self.assertTrue(all(t != gold[i] for i, t in by_name["future_wrong"]["anchors"]))
        self.assertTrue(all(t == gold[i] for i, t in by_name["future_correct"]["anchors"]))
        self.assertEqual(sorted(t for i, t in by_name["future_old"]["anchors"]), sorted(t for i, t in by_name["future_shuffled"]["anchors"]))
        self.assertEqual(plans, conditioning_plans(old, gold, self.config(), 42)[0])
        self.assertFalse(conditioning_plans([1], [1], self.config(), 42)[1]["eligible"])

    def test_guided_wrong_anchor_gain_is_not_no_observed_headroom(self):
        def arm(name, accepted, oracle=False, **metrics):
            return {"method": name, "oracle": oracle, "metrics": {"accepted_prefix": accepted,
                "anchor_count": 0, "false_keep_count": 0, **metrics}}
        rows = [arm("fresh_L", 2), arm("oracle_A", 2, True),
                arm("verifier_confidence@0.5", 7, anchor_count=7, false_keep_count=7)]
        gains = event_gains(rows)
        self.assertTrue(gains["no_bounded_correct_keep_gain"])
        self.assertFalse(gains["no_observed_gain"])
        self.assertEqual(gains["best_nonoracle_gain"], 5)
        self.assertEqual(gains["wrong_anchor_gain"], 5)
        self.assertFalse(gains["oracle_selector_gap"])

    def test_progress_sampling_has_three_regions_and_alignment(self):
        for count in (3, 5):
            chosen = offsets(45, 256, 32, count)
            self.assertEqual({progress_bin(x, 256) for x in chosen}, {"early", "middle", "late"})
            self.assertTrue(any((x+45)%32 == 0 for x in chosen))
        self.assertEqual(offsets(45, 33, 32, 5), [0])

    def test_sampling_reports_empty_strata_and_redistributes(self):
        rows = [{"id": str(i), "source": s, "dataset": "gsm8k", "progress_bin": "early",
                 "metrics": {"first_rejection": 1}, "draft": {"tokens": [1]*32}}
                for i, s in enumerate(["dllm_top1"]+["ar_greedy"]*5)]
        selected = stratified_events(rows, 4)
        report = sampling_coverage(rows, selected, self.config())
        self.assertEqual(report["eligible"], 6)
        self.assertEqual(report["selected"], 4)
        self.assertEqual(sum(x["selected"] for x in report["strata"]), 4)
        self.assertTrue(any(x["empty_stratum"] for x in report["strata"]))


if __name__ == "__main__":
    unittest.main()
