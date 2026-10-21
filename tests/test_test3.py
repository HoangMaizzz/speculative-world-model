"""Test3 independent counting, all-correct masks, screening and six-arm mechanics."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from suffix_lab.assets import ROOT
from suffix_lab.core import strict_top1
from suffix_lab.metrics import prefix_length
from suffix_lab.store import Store
from suffix_lab.test3 import screen, run_arms, pack, execute, SelectionUnavailable
from suffix_lab.test3_plan import context_pool, corrected_event, category, select_events, conditioning_metrics, visible_trace, arm_plans
from suffix_lab.test3_report import report, cases_html


def config():
    return json.loads((ROOT/"configs/test3_oracle_all.json").read_text())


def one_config():
    return {**config(), "events": 1, "gsm8k_events": 1, "category_quotas": {"early_rescue": 1}}


def pool():
    ref = {"id": "gsm8k:1", "prompt_ids": [49], "tokens": list(range(1, 65)), "question": "<script>question"}
    context = {"event_id": "gsm8k:1:0:dllm24", "state_id": "gsm8k:1:0", "pilot_test1_id": "old32",
               "prompt_id": "gsm8k:1", "dataset": "gsm8k", "offset": 0}
    return {"contexts": [context], "references": {ref["id"]: ref}}


class Drafter:
    """Unconditional token i at absolute i, with controlled original/Fresh errors."""
    def __init__(self, profiles=None):
        self.calls, self.profiles = [], profiles or {}
    def draft(self, prefix, length, anchors=None, use_prefix_cache=True):
        anchors = anchors or {}
        self.calls.append((list(prefix), length, dict(anchors), use_prefix_cache))
        kind = self.profiles.get(prefix[0], "early_rescue")
        keep = {"early_rescue": set(range(0, 16, 2)), "later_anchor": {7, 8, 9},
                "low_reuse": {5}, "fresh_ceiling": set(range(16))}[kind]
        class Provider:
            n = 0
            def predict(self, canvas, block, positions):
                self.n += 1
                result = {}
                for p in positions:
                    token = p
                    if len(prefix) == 1 and length == 24:
                        if p == 8 or (9 <= p < 25 and p-9 not in keep):
                            token = 100  # Original a=7; eight singleton correct anchors.
                    elif len(prefix) == 9 and not anchors and p == 9 and kind != "fresh_ceiling":
                        token = 101  # Fresh fails at first post-correction token.
                    result[p] = (token, 0.9)
                return result
            def stats(self):
                return {"unmask_forwards": self.n, "total_forwards": self.n, "input_tokens": self.n*32}
        return strict_top1(prefix, length, anchors, Provider(), mask_id=151665)


class Verifier:
    def __init__(self):
        self.calls = []
    def score(self, prefix, proposal, top_k):
        self.calls.append((list(prefix), list(proposal)))
        return {"top1": list(range(len(prefix), len(prefix)+len(proposal)))}


class Test3Tests(unittest.TestCase):
    def test_pool_uses_only_metadata_not_B32_tokens_and_covers_prompts_first(self):
        records = {}
        for j in range(12):
            ds, pid = ("gsm8k" if j%2 else "math500"), str(j)
            records[("reference", pid)] = {"id": pid, "tokens": list(range(100)), "prompt_ids": [99]}
            for offset in (0, 10):
                key = pid+":"+str(offset)
                # Deliberately no draft or reference slice in old Test1 record.
                records[("test1", key)] = {"id": key, "source": "dllm_top1", "prompt_id": pid,
                    "state_id": key, "dataset": ds, "offset": offset}
        records[("test1", "ar")] = {"source": "ar_greedy"}
        pilot = {"records": records, "manifest": {}, "checkpoint_sha256": "fixture"}
        a = context_pool(pilot, config()); b = context_pool(pilot, config())
        self.assertEqual(a, b)
        self.assertEqual(len(a["contexts"]), 24)
        self.assertEqual(len({c["prompt_id"] for c in a["contexts"][:12]}), 12)
        self.assertEqual([c["dataset"] for c in a["contexts"][:4]], ["gsm8k", "math500"]*2)

    def test_B24_correction_counting_and_KEEP_all_singletons_no_budget(self):
        p, d, v = pool(), Drafter(), Verifier()
        c, ref = p["contexts"][0], p["references"]["gsm8k:1"]
        from dataclasses import asdict
        original = asdict(d.draft([49], 24))
        e, reason = corrected_event(c, ref, original, v.score([49], original["tokens"], 8), config())
        self.assertIsNone(reason)
        self.assertEqual((e["accepted_before_rejection"], e["L"], e["extension_length"], e["H"]), (7, 16, 7, 23))
        self.assertEqual(e["prefix"], [49]+list(range(1, 9)))
        self.assertEqual(e["correction_token"], 8)
        self.assertEqual(e["old"], original["tokens"][8:])
        self.assertEqual(e["oracle_keep_positions"], list(range(0, 16, 2)))
        self.assertEqual(len(e["old_correct_islands"]), 8)
        self.assertGreater(e["old_correct_count"], e["L"]//4)
        self.assertTrue(all(e["initial_oracle_joint_proposal"][i] == 151665 for i in range(16, 23)))
        for a in (3, 15):
            proposal = list(range(1, 25)); proposal[a] = 100
            x, reason = corrected_event(c, ref, {"tokens": proposal}, {"top1": list(range(1, 25))}, config())
            self.assertEqual((x["L"], x["extension_length"]), (23-a, a))
        with self.assertRaisesRegex(ValueError, "B=24"):
            corrected_event(c, ref, {"tokens": [1]*32}, {}, config())

    def test_insufficient_gold_skipped_and_correction_verified(self):
        p, d, v = pool(), Drafter(), Verifier()
        from dataclasses import asdict
        original = asdict(d.draft([49], 24)); feedback = v.score([49], original["tokens"], 8)
        ref = deepcopy(p["references"]["gsm8k:1"]); ref["tokens"] = ref["tokens"][:30]
        e, reason = corrected_event(p["contexts"][0], ref, original, feedback, config())
        self.assertIsNone(e)
        self.assertIn("insufficient", reason)
        feedback["top1"][7] = 55
        with self.assertRaisesRegex(AssertionError, "correction"):
            corrected_event(p["contexts"][0], p["references"]["gsm8k:1"], original, feedback, config())

    def test_categories_are_disjoint_and_later_has_no_near_anchor(self):
        gold = list(range(1, 24))
        def event(keep):
            from suffix_lab.metrics import runs
            return {"reference": gold, "H": 23, "L": 16, "oracle_keep_positions": keep,
                    "old_correct_islands": runs([i in keep for i in range(16)])}
        def fresh(f):
            values = gold[:]
            if f < 23: values[f] = 100
            return values
        self.assertEqual(category(event([3, 4]), fresh(3), config()), "early_rescue")
        self.assertEqual(category(event([7, 8, 9]), fresh(2), config()), "later_anchor")
        self.assertEqual(category(event([7, 9]), fresh(2), config()), "later_anchor")
        self.assertEqual(category(event([3]), fresh(3), config()), "low_reuse")
        self.assertEqual(category(event([]), fresh(23), config()), "fresh_ceiling")
        self.assertEqual(category(event([0, 1]), fresh(10), config()), "other")

    def test_exact_category_benchmark_and_distinct_prompt_quotas(self):
        screens = []
        categories = ["early_rescue"]*8+["later_anchor"]*6+["low_reuse"]*4+["fresh_ceiling"]*2
        for i, name in enumerate(categories):
            screens.append({"status": "eligible", "screen_index": i, "category": name,
                "event": {"event_id": str(i), "prompt_id": str(i//2), "dataset": "gsm8k" if i%2 else "math500"}})
        selected = select_events(screens, config())
        self.assertEqual(len(selected), 10)
        picked = [s for s in screens if s["event"]["event_id"] in selected]
        from collections import Counter
        self.assertEqual(Counter(s["category"] for s in picked), Counter(config()["category_quotas"]))
        self.assertEqual(len({s["event"]["prompt_id"] for s in picked}), 10)
        self.assertEqual(sum(s["event"]["dataset"] == "gsm8k" for s in picked), 5)
        self.assertIsNone(select_events(screens[:-2], config()))

    def test_actual_screen_regenerates_24_and_23_then_freezes_before_oracle(self):
        p, d, v = pool(), Drafter(), Verifier()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            store = Store(Path(tmp)/"test3.sqlite")
            events = screen(p, one_config(), store, d, v)
            self.assertEqual([c[1] for c in d.calls], [24, 23])
            self.assertFalse(store.rows("arm"))
            self.assertEqual(events[0]["category"], "early_rescue")
            self.assertEqual(store.get("meta", "phase"), "selected")
            before = len(d.calls)
            self.assertEqual(screen(p, one_config(), store, d, v), events)
            self.assertEqual(len(d.calls), before)
            store.close()

    def test_six_arms_all_correct_keeps_and_negative_control(self):
        p, d, v = pool(), Drafter(), Verifier()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            store = Store(Path(tmp)/"test3.sqlite")
            events = screen(p, one_config(), store, d, v)
            run_arms(p, events, one_config(), store, d, v)
            rows = {r["method"]: r for r in store.rows("arm")}
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows["F-23"]["metrics"]["accepted_prefix"], 0)
            self.assertEqual(rows["O-Joint-23"]["metrics"]["accepted_prefix"], 23)
            self.assertEqual(rows["O-L"]["metrics"]["accepted_prefix"], 16)
            self.assertEqual(rows["KEEP-All-23"]["metrics"]["accepted_prefix"], 1)
            self.assertEqual(rows["KEEP-All-23"]["metrics"]["first_mismatch_region"], "KEEP")
            self.assertEqual(rows["O-Sequential-23"]["tokens"][:16], rows["O-L"]["tokens"])
            for row in rows.values():
                self.assertEqual(len(row["trace"]), row["horizon"]-len(row["anchors"]))
                self.assertTrue(all(s["mask_before"]-s["mask_after"] == 1 for s in row["trace"]))
                self.assertTrue(all(row["tokens"][i] == t for i, t in row["anchors"]))
                self.assertEqual(row["metrics"]["positionwise_correct"], sum(a == b for a, b in zip(row["tokens"], row["reference_token_ids"])))
                self.assertTrue(row["cached_full_audit"]["passed"])
            self.assertTrue(store.get("meta", "complete"))
            self.assertTrue(all(call[1] != 32 for call in d.calls))
            self.assertIsNotNone(store.get("original_audit", events[0]["event_id"]))
            summary = report(tmp, store, one_config())
            self.assertTrue(summary["complete"])
            html = cases_html(events, list(rows.values()), summary["comparisons"],
                              type("Tokenizer", (), {"decode": lambda self, ids: "<script>"})())
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            archive = pack(tmp, store)
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                for name in ("events.jsonl", "arms.jsonl", "summary.csv", "report.md", "checkpoint.sqlite", "screening.jsonl"):
                    self.assertIn(name, z.namelist())
                self.assertNotIn("test3.sqlite", z.namelist())
            store.close()

    def test_complete_orchestration_screens_true_groups_runs_60_and_loads_no_ar_drafter(self):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            self.skipTest("Optional torch absent")
        from types import SimpleNamespace
        from suffix_lab.assets import TARGET, DLLM
        from suffix_lab.store import digest
        profiles = ["early_rescue"]*4+["later_anchor"]*3+["low_reuse"]*2+["fresh_ceiling"]
        p = {"contexts": [], "references": {}, "pool_digest": "fixture", "pilot_checkpoint_sha256": "fixture"}
        mapping = {}
        for i, kind in enumerate(profiles):
            dataset = "gsm8k" if i < 5 else "math500"
            pid = dataset+":"+str(i)
            c = {"event_id": pid+":0:dllm24", "state_id": pid+":0", "pilot_test1_id": pid+":old32",
                 "prompt_id": pid, "dataset": dataset, "offset": 0}
            p["contexts"].append(c)
            p["references"][pid] = {"id": pid, "tokens": list(range(1, 65)), "prompt_ids": [1000+i]}
            mapping[1000+i] = kind
        env = {"commit": "a"*40, "cuda": "12.6", "gpus": ["T4", "T4"],
               "target_device_map": {"layer": "0"}, "target_attention": "sdpa",
               "allow_fp16_reduced_precision_reduction": False,
               "packages": {"torch": "fixture", "transformers": "fixture", "accelerate": "fixture"}}
        p["pilot_manifest"] = {"config": {}, "asset_digest": "fixture", "target_precision": "fp16",
                               "tokenizer": {}, "environment": env}
        d = Drafter(mapping)
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [49], decode=lambda ids: str(ids[0]))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch("suffix_lab.test3.read_pilot", return_value={}), \
             patch("suffix_lab.test3.context_pool", return_value=p), \
             patch("suffix_lab.test3.prepare_models", return_value=({TARGET: "target", DLLM: "dllm"}, {})), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("suffix_lab.runtime.tokenizers", return_value=(tokenizer, {})), \
             patch("suffix_lab.runtime.load_target", return_value=(SimpleNamespace(hf_device_map={"layer": 0}), "fp16", None)), \
             patch("suffix_lab.runtime.load_model", return_value=object()) as load, \
             patch("suffix_lab.runtime.ARSession", return_value=Verifier()), \
             patch("suffix_lab.runtime.DLLMDrafter", return_value=d), \
             patch("suffix_lab.runtime.target_consistency", return_value={"passed": True}), \
             patch("suffix_lab.runtime.release"), patch("suffix_lab.experiment.environment", return_value=env):
            execute("input", tmp, "a"*40)
            load.assert_called_once_with("dllm", "cuda:1")
            with zipfile.ZipFile(Path(tmp)/"test3_checkpoint.zip") as z:
                status = json.loads(z.read("execution_status.json"))
                self.assertEqual(status["status"], "complete")
                self.assertEqual(status["completed_arms"], 60)
                self.assertEqual(status["screened_states"], 10)
                self.assertEqual(len(z.read("arms.jsonl").splitlines()), 60)
                selected = [json.loads(l) for l in z.read("events.jsonl").splitlines()]
                from collections import Counter
                self.assertEqual(Counter(e["category"] for e in selected), Counter(config()["category_quotas"]))
                self.assertEqual(sum(e["dataset"] == "gsm8k" for e in selected), 5)
            self.assertTrue(all(call[1] != 32 for call in d.calls))

    def test_interruption_resume_and_frozen_selection_does_not_use_repair_results(self):
        p = pool()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            store = Store(Path(tmp)/"test3.sqlite")
            events = screen(p, one_config(), store, Drafter(), Verifier())
            frozen = store.get("meta", "selection")
            def interrupt():
                if len(store.rows("arm")) == 3:
                    raise RuntimeError("interrupted")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_arms(p, events, one_config(), store, Drafter(), Verifier(), interrupt)
            old = store.rows("arm")
            d, v = Drafter(), Verifier()
            run_arms(p, events, one_config(), store, d, v)
            for row in old:
                self.assertEqual(store.get("arm", row["id"]), row)
            self.assertEqual(store.get("meta", "selection"), frozen)
            before = len(d.calls)
            run_arms(p, events, one_config(), store, d, v)
            self.assertEqual(len(d.calls), before)
            store.close()

    def test_shortfall_is_explicit_and_packs_screening_without_fake_arms(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            store = Store(Path(tmp)/"test3.sqlite")
            with self.assertRaisesRegex(SelectionUnavailable, "No quotas were relaxed"):
                screen(pool(), config(), store, Drafter(), Verifier())
            self.assertFalse(store.rows("arm"))
            self.assertFalse(store.get("meta", "complete"))
            report(tmp, store, config()); path = pack(tmp, store)
            with zipfile.ZipFile(path) as z:
                self.assertEqual(json.loads(z.read("selection.json"))["screened_states"], 1)
            store.close()

    def test_direct_keep_gain_is_not_mislabeled_as_conditioning(self):
        e = {"reference": [1, 2, 3, 4, 5, 6, 7], "L": 5}
        fresh, actual = [9, 2, 3, 4, 8, 6, 8], [1, 2, 3, 4, 5, 6, 7]
        m = conditioning_metrics(e, actual, fresh, [[0, 1], [3, 4]])
        a = m["accepted_gain_attribution"]
        self.assertEqual(a["direct_keep_corrected"], 1)
        self.assertEqual(a["conditioning_redraft_corrected"], 1)
        self.assertEqual(a["conditioning_extension_corrected"], 1)
        self.assertEqual(sum(a.values()), 7)
        self.assertEqual(m["mutable_positionwise"]["REDRAFT"]["fresh_wrong_oracle_correct"], 1)
        self.assertEqual(m["mutable_positionwise"]["EXTEND"]["fresh_wrong_oracle_correct"], 1)

    def test_visibility_uses_shifted_prediction_block_and_committed_extension(self):
        e = {"prefix": [9]*31, "L": 2}
        trace = [{"step": 0, "position": 0, "absolute_position": 31, "prediction_row": 30},
                 {"step": 1, "position": 1, "absolute_position": 32, "prediction_row": None}]
        rows = visible_trace(e, trace, [[2, 99]])
        self.assertEqual(rows[0]["visible_future_keep_positions"], [])
        self.assertEqual(rows[1]["visible_future_keep_positions"], [])
        self.assertTrue(rows[1]["uses_previous_block_carry"])
        e = {"prefix": [9]*9, "L": 8}
        trace = [{"step": 0, "position": 8, "absolute_position": 17, "prediction_row": 16},
                 {"step": 1, "position": 7, "absolute_position": 16, "prediction_row": 15}]
        rows = visible_trace(e, trace, [])
        self.assertEqual(rows[1]["visible_generated_future_extension"], [8])

    def test_input_error_still_exports_failed_status_zip(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch("suffix_lab.test3.read_pilot", side_effect=ValueError("bad input")):
            with self.assertRaisesRegex(ValueError, "bad input"):
                execute("bad", tmp, "a"*40)
            with zipfile.ZipFile(Path(tmp)/"test3_checkpoint.zip") as z:
                self.assertEqual(json.loads(z.read("execution_status.json"))["status"], "failed")


if __name__ == "__main__":
    unittest.main()
