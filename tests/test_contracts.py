"""Independent hand-computed expectations; no pretrained weights required."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from suffix_lab.core import strict_top1, prediction_row
from suffix_lab.metrics import suffix_metrics, redraft_metrics, prefix_length
from suffix_lab.selection import Observation, select_keep, oracle_masks
from suffix_lab.store import Store, digest
from suffix_lab.experiment import offsets, stratified_events, nonoracle_plans, pack, restore


class FakeProvider:
    def __init__(self, mask=99):
        self.calls = []
        self.mask = mask

    def predict(self, canvas, block_start, positions):
        self.calls.append((list(canvas), block_start, list(positions)))
        # Highest absolute position wins *within* the active subblock.
        return {p: (10+p % 50, .1+p/1000) for p in positions}

    def stats(self):
        return {"total_forwards": len(self.calls)}


class SchedulerTests(unittest.TestCase):
    def test_one_update_per_step_and_active_window(self):
        provider = FakeProvider()
        result = strict_top1([1, 2, 3], 10, {}, provider, 99, 8, 4)
        self.assertEqual([x["absolute_position"] for x in result.trace], [3, 7, 6, 5, 4, 11, 10, 9, 8, 12])
        self.assertEqual(result.tokens, list(range(13, 23)))
        for trace in result.trace:
            self.assertEqual(trace["mask_before"]-trace["mask_after"], 1)
        for canvas, block, positions in provider.calls:
            self.assertEqual(canvas[:3], [1, 2, 3])
            self.assertEqual(len({p//4 for p in positions}), 1)
            self.assertEqual(len({p//8 for p in positions}), 1)

    def test_hard_keep_and_independent_arms(self):
        anchors = {0: 67, 5: 68}
        a = strict_top1([1]*7, 12, anchors, FakeProvider(), 99, 8, 4)
        b = strict_top1([1]*7, 12, {}, FakeProvider(), 99, 8, 4)
        c = strict_top1([1]*7, 12, anchors, FakeProvider(), 99, 8, 4)
        self.assertEqual(a.tokens, c.tokens)
        self.assertNotEqual(a.tokens, b.tokens)
        self.assertEqual([a.tokens[i] for i in anchors], [67, 68])
        self.assertEqual(a.stats["unmask_steps"], 10)

    def test_all_keep_zero_forward(self):
        p = FakeProvider()
        result = strict_top1([1]*32, 3, {0: 2, 1: 3, 2: 4}, p, 99)
        self.assertEqual(result.tokens, [2, 3, 4])
        self.assertEqual(p.calls, [])

    def test_partial_final_block_does_not_emit_padding(self):
        p = FakeProvider()
        result = strict_top1([1]*31, 2, {}, p, 99)
        self.assertEqual(len(result.tokens), 2)
        self.assertEqual(result.stats["unmask_steps"], 2)
        self.assertTrue(all(max(positions) <= 32 for _, _, positions in p.calls))

    def test_tie_break_lowest_position(self):
        p = FakeProvider()
        p.predict = lambda canvas, start, positions: {i: (3, .5) for i in positions}
        result = strict_top1([1], 4, {}, p, 99)
        self.assertEqual(result.order, [0, 1, 2, 3])

    def test_shift(self):
        self.assertIsNone(prediction_row(64, 64))
        self.assertEqual(prediction_row(65, 64), 0)
        self.assertEqual(prediction_row(95, 64), 30)

    def test_invalid_provider(self):
        p = FakeProvider()
        p.predict = lambda canvas, start, positions: {i: (99, .8) for i in positions}
        with self.assertRaises(RuntimeError):
            strict_top1([1], 3, {}, p, 99)

    def test_invalid_anchor(self):
        with self.assertRaises(ValueError):
            strict_top1([1], 3, {3: 2}, FakeProvider(), 99)
        with self.assertRaises(ValueError):
            strict_top1([99], 3, {}, FakeProvider(), 99)


class MetricTests(unittest.TestCase):
    def test_reject_index_and_suffix_excludes_correction(self):
        m = suffix_metrics([1, 9, 3, 4, 9, 6], [1, 2, 3, 4, 5, 6], [1, 2, 9, 4, 9, 9])
        self.assertEqual(m["first_rejection"], 2)
        self.assertEqual(m["accepted_prefix"], 1)
        self.assertEqual(m["suffix_length"], 4)
        self.assertEqual(m["suffix_match_count"], 3)
        self.assertEqual(m["island_lengths"], [2, 1])
        self.assertEqual(m["direct_rejoin_length"], 2)
        self.assertNotEqual(m["old_matches"], m["reference_matches"])

    def test_all_accepted_and_last_reject_have_no_tail(self):
        for proposal in ([1, 2], [1, 8]):
            m = suffix_metrics(proposal, [1, 2])
            self.assertEqual(m["suffix_length"], 0)
            self.assertIsNone(m["suffix_match_rate"])
        self.assertEqual(suffix_metrics([1, 2], [1, 2])["first_rejection"], 3)

    def test_reference_short_not_silently_zipped(self):
        with self.assertRaises(ValueError):
            suffix_metrics([1, 2], [1])
        with self.assertRaises(ValueError):
            suffix_metrics([1, 2], [1, 2], [1])

    def test_false_keep_caps_prefix(self):
        result = redraft_metrics([1, 7, 3, 4], [1, 2, 3, 4], [1, 7, 8, 4], [0, 1, 3])
        self.assertEqual(result["accepted_prefix"], 1)
        self.assertEqual(result["false_keep_count"], 1)
        self.assertEqual(result["recovered_old_tokens_in_prefix"], 1)
        self.assertEqual(result["remasked_accuracy"], 1)
        self.assertAlmostEqual(result["retained_precision"], 2/3)

    def test_keep_mutation_detected(self):
        with self.assertRaises(AssertionError):
            redraft_metrics([2], [2], [1], [0])

    def test_corrected_context_slice(self):
        # r=2: accepted 10, correction20, retained suffix starts at index2.
        draft, gold, r = [10, 99, 30, 40], [10, 20, 30, 41], 2
        corrected = [7] + gold[:r]
        self.assertEqual(corrected, [7, 10, 20])
        self.assertEqual(draft[r:], [30, 40])
        self.assertEqual(len(draft[r:]), 4-r)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.obs = Observation((5, 6, 7, 8), (.8, .9, .2, .1), (.1, .2, .8, .7), (5, 9, 7, 0))

    def test_selectors_and_matched_counts(self):
        self.assertEqual(select_keep(self.obs, "drafter_confidence", .5), [0, 1])
        self.assertEqual(select_keep(self.obs, "verifier_confidence", .5), [2, 3])
        self.assertEqual(select_keep(self.obs, "old_agreement", 0), [0, 2])
        self.assertEqual(select_keep(self.obs, "old_agreement_ranked", .5), [0, 2])
        self.assertEqual(len(select_keep(self.obs, "random", .5)), 2)

    def test_determinism_and_no_future_labels_in_nonoracle_interface(self):
        config = {"keep_ratios": [.25, .5, .75], "random_repeats": 3}
        before = nonoracle_plans(self.obs, config, 42)
        oracle_masks(self.obs.old_tokens, [5, 0, 7, 0])
        after = nonoracle_plans(self.obs, config, 42)
        self.assertEqual(before, after)
        self.assertNotIn("reference", Observation.__dataclass_fields__)
        self.assertEqual(self.obs.old_tokens, (5, 6, 7, 8))

    def test_oracle_only_correct_and_bounded(self):
        patterns = oracle_masks([1, 8, 3, 9], [1, 2, 3, 4], 5)
        self.assertEqual(patterns[0], [0, 2])
        self.assertLessEqual(len(patterns), 5)
        self.assertTrue(all(set(p) <= {0, 2} for p in patterns))

    def test_bad_features_rejected(self):
        with self.assertRaises(ValueError):
            select_keep(Observation((1,), (), (.5,), (1,)), "random", .5)


class CheckpointTests(unittest.TestCase):
    def test_resume_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"run.sqlite"
            a = Store(path, {"precision": "int8"})
            a.put("test1", "state", {"tokens": [1, 2]})
            a.close()
            b = Store(path, {"precision": "int8"})
            self.assertEqual(b.get("test1", "state")["tokens"], [1, 2])
            b.close()
            with self.assertRaises(ValueError):
                Store(path, {"precision": "nf4"})

    def test_hot_backup_pack_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, destination = Path(tmp)/"source", Path(tmp)/"restored"
            source.mkdir()
            store = Store(source/"run.sqlite", {"v": 1})
            store.put("test2", "arm", {"keep": [1]})
            archive = pack(source)
            store.close()
            restore(archive, destination)
            recovered = Store(destination/"run.sqlite", {"v": 1})
            self.assertEqual(recovered.get("test2", "arm"), {"keep": [1]})
            recovered.close()

    def test_restore_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"evil.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("../escape", "x")
            with self.assertRaises(ValueError):
                restore(path, Path(tmp)/"restored")


class SamplingTests(unittest.TestCase):
    def test_offsets_valid_eos_bounds_and_phase(self):
        result = offsets(45, 256, 32, 5)
        self.assertEqual(len(result), 5)
        self.assertIn(0, result)
        self.assertIn(19, result)
        self.assertTrue(all(x+33 <= 256 for x in result))
        self.assertEqual(offsets(30, 32), [])

    def test_stratified_sample_frozen_unique_and_dllm_priority(self):
        rows = [{"id": f"{s}:{d}:{i}", "source": s, "dataset": d,
                 "metrics": {"first_rejection": i+1}, "draft": {"tokens": [1]*32}}
                for s in ("dllm_top1", "ar_greedy") for d in ("math500", "gsm8k") for i in range(32)]
        result = stratified_events(rows, 30)
        self.assertEqual(result, stratified_events(list(reversed(rows)), 30))
        self.assertEqual(len(set(result)), 30)
        self.assertEqual(sum(x.startswith("dllm") for x in result), 20)
        self.assertTrue(all(not x.endswith(":31") for x in result))


if __name__ == "__main__":
    unittest.main()
