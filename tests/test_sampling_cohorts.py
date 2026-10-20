"""Sampling probabilities, leakage boundaries and paired scientific reporting."""
from copy import deepcopy
import unittest
from suffix_lab.sampling import freeze_sample, correct_bin
from suffix_lab.cohort_report import population_estimates
from suffix_lab.metrics import benefit_metrics, event_gains
from suffix_lab.plans import complete_plans
from suffix_lab.selection import Observation


class CohortTests(unittest.TestCase):
    def fixture(self):
        config = {"seed": 42, "test2_events": 24, "test2_sources": ["dllm_top1", "ar_greedy"],
                  "datasets": ["math500", "gsm8k"]}
        rows = [{"id": f"{s}:{d}:{p}:{i}", "source": s, "dataset": d, "progress_bin": p,
                 "draft": {"tokens": [1]*32}, "metrics": {"first_rejection": 2,
                    "suffix_match_count": k, "max_island_length": k}}
                for s in config["test2_sources"] for d in config["datasets"]
                for p in ("early", "middle", "late") for i, k in enumerate((0, 1, 2, 5))]
        return config, rows

    def test_cohorts_are_disjoint_and_cover_progress_without_gold_in_representative(self):
        config, rows = self.fixture()
        design = freeze_sample(rows, config)
        rep, enriched = (design["cohorts"][name]["events"] for name in ("representative", "enriched"))
        self.assertEqual((len(rep), len(enriched)), (12, 12))
        self.assertFalse(set(rep) & set(enriched))
        self.assertEqual(design, freeze_sample(list(reversed(rows)), config))
        for cohort in (rep, enriched):
            self.assertEqual({r["progress_bin"] for r in rows if r["id"] in cohort}, {"early", "middle", "late"})
        self.assertTrue(all(r["metrics"]["suffix_match_count"] > 0 for r in rows if r["id"] in enriched))
        modified = deepcopy(rows)
        for r in modified:
            r["metrics"].update(suffix_match_count=0, max_island_length=0)
        self.assertEqual(rep, freeze_sample(modified, config)["cohorts"]["representative"]["events"])
        weights = [design["members"][e]["sampling_weight"] for e in rep]
        self.assertEqual(sum(weights), len(rows))
        self.assertTrue(all(design["members"][e]["sampling_weight"] is None for e in enriched))

    def test_enriched_prioritizes_islands_within_balanced_strata(self):
        config, rows = self.fixture()
        design = freeze_sample(rows, config)
        rep = set(design["cohorts"]["representative"]["events"])
        enriched = set(design["cohorts"]["enriched"]["events"])
        for row in rows:
            if row["id"] in enriched:
                remaining = [r for r in rows if r["id"] not in rep and all(r[k] == row[k] for k in ("source", "dataset", "progress_bin"))]
                self.assertEqual(row["metrics"]["max_island_length"], max(r["metrics"]["max_island_length"] for r in remaining))

    def test_shortfalls_never_fill_enriched_with_zero_correct_events(self):
        config, rows = self.fixture()
        for row in rows:
            row["metrics"].update(suffix_match_count=0, max_island_length=0)
        design = freeze_sample(rows, config)
        self.assertEqual(design["cohorts"]["enriched"]["selected"], 0)
        self.assertEqual(design["cohorts"]["enriched"]["shortfall"], 12)

    def test_uncovered_representative_stratum_disables_population_claim(self):
        config, rows = self.fixture()
        config["test2_events"] = 2
        design = freeze_sample(rows, config)
        self.assertFalse(design["cohorts"]["representative"]["population_coverage"])

    def test_weighted_population_mean_and_enriched_exclusion(self):
        design = {"cohorts": {"representative": {"events": ["a", "b"], "available": 10, "population_coverage": True}}}
        rows = [{"cohort": "representative", "event_id": e, "redrafter": "ar_greedy", "method": "fresh_L",
                 "sampling_weight": w, "accepted_prefix": a, "gain": a, "win": 1, "tie": 0, "loss": 0,
                 "latency_ms": 10., "latency_delta_ms": 0., "t_verify_compute_ms": 1.}
                for e, w, a in (("a", 2., 2), ("b", 8., 8))]
        rows.append({**rows[0], "cohort": "enriched", "event_id": "c", "gain": 999})
        result = population_estimates(rows, design)["estimates"][0]
        self.assertTrue(result["estimable"])
        self.assertAlmostEqual(result["gain"], 6.8)
        self.assertAlmostEqual(result["accepted_tokens_per_draft_ms_ratio_of_totals"], .68)
        self.assertFalse(population_estimates(rows[1:], design)["estimates"][0]["estimable"])

    def test_cost_outcomes_distinguish_false_caps_and_equal_quality_saving(self):
        baseline = {"metrics": {"accepted_prefix": 5}, "stats": {"latency_ms": 10.}}
        row = {"metrics": {"accepted_prefix": 5, "first_false_keep": None, "false_keep_count": 0},
               "stats": {"latency_ms": 8.}, "verification": {"latency_ms": 2., "wall_latency_ms": 3.}}
        metrics = benefit_metrics(row, baseline)
        self.assertEqual(metrics["outcome"], "equal_prefix_cheaper_draft")
        self.assertEqual(metrics["accepted_tokens_per_draft_ms"], .625)
        self.assertEqual(metrics["accepted_tokens_per_draft_verify_compute_ms"], .5)
        row["metrics"].update(accepted_prefix=2, first_false_keep=2, false_keep_count=1)
        self.assertEqual(benefit_metrics(row, baseline)["outcome"], "loss_capped_at_false_keep")

    def test_three_oracles_and_multiseed_content_controls_remain_separate(self):
        obs = Observation(tuple(range(20)), (.5,)*20, (.5,)*20, tuple(range(20)))
        config = {"oracle_subset_limit": 4, "mask_id": 99, "conditioning_seeds": [0, 1, 2]}
        plans, meta = complete_plans(obs, list(range(20)), config, 42, [("fresh_L", [])])
        controls = [p for p in plans if p["arm_family"] == "conditioning"]
        self.assertEqual(len(controls), 15)
        self.assertEqual(len({p["method"] for p in controls}), 15)
        self.assertEqual(len(meta), 3)
        for repeat in range(3):
            group = [p for p in controls if p["conditioning_repeat"] == repeat]
            self.assertEqual(len({tuple(i for i,t in p["anchors"]) for p in group}), 1)
        def arm(name, a, kind, family):
            return {"method": name, "oracle": True, "oracle_kind": kind, "arm_family": family,
                    "metrics": {"accepted_prefix": a, "false_keep_count": 0, "anchor_count": 0}}
        arms = [arm("fresh_L", 1, "none", "nonoracle"), arm("oracle_A", 1, "old_token_keep", "oracle"),
                arm("future_correct", 9, "future_reference_injection", "conditioning")]
        gains = event_gains(arms)
        self.assertEqual(gains["bounded_old_mask_search_gain"], 0)
        self.assertEqual(gains["oracle_future_injection_gain"], 8)
        self.assertTrue(gains["no_bounded_correct_keep_gain"])

    def test_correct_count_bins(self):
        self.assertEqual([correct_bin(n) for n in (0, 1, 2, 3, 4, 20)], ["0", "1", "2-3", "2-3", "4+", "4+"])


if __name__ == "__main__":
    unittest.main()
