"""Independent controlled experiments: quotas, no leakage, stage semantics/resume."""
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from contextlib import redirect_stdout
import unittest
import zipfile
from unittest.mock import patch
from suffix_lab.assets import ROOT
from suffix_lab.core import strict_top1
from suffix_lab.extension import audited_draft, run_matrix, arm_metrics, pack_extension, execute
from suffix_lab.extension_plan import (choose_events, event_plan, predicted_positions, span_positions,
                                      validate_event, read_pilot, freeze_inputs, PILOT_REVISION, PILOT_SOURCE)
from suffix_lab.extension_report import report, paired_metrics, case_html
from suffix_lab.metrics import suffix_metrics
from suffix_lab.selection import Observation
from suffix_lab.store import Store, atomic_json


def config():
    return json.loads((ROOT/"configs/mini_extension.json").read_text())


class ToyDrafter:
    def __init__(self, base=9, drift=False):
        self.base, self.calls, self.drift = base, [], drift
    def draft(self, prefix, length, anchors=None, use_prefix_cache=True):
        self.calls.append((list(prefix), length, dict(anchors or {}), use_prefix_cache))
        base = self.base
        class Provider:
            n = 0
            def predict(self, canvas, block, positions):
                self.n += 1
                return {p: (p-base+1, 0.9) for p in positions}
            def stats(self):
                return {"unmask_forwards": self.n, "total_forwards": self.n, "input_tokens": self.n*32}
        result = strict_top1(prefix, length, anchors or {}, Provider(), mask_id=62)
        if self.drift and not use_prefix_cache:
            i = next(i for i in range(length) if i not in (anchors or {}))
            result.tokens[i] += 100
        return result


class ToyVerifier:
    def __init__(self):
        self.calls = 0
    def score(self, prefix, proposal, top_k):
        self.calls += 1
        return {"top1": list(range(1, len(proposal)+1))}


def mini_inputs():
    e = {"event_id": "gsm8k:1:0:dllm_top1", "prompt_id": "gsm8k:1", "dataset": "gsm8k",
         "source": "dllm_top1", "cohort": "representative", "first_rejection_1based": 24,
         "reference_start": 8, "prefix": [50]+[40]*8, "old": [1, 2, 3, 99, 5, 6, 7, 8],
         "reference": list(range(1, 17)), "L": 8, "H": 16,
         "anchors": {"fresh": [], "oracle": [[1, 2], [2, 3]], "predicted": []},
         "physical_phase": 9, "old_endpoint_phase8": 1}
    return {"events": [e], "references": {"gsm8k:1": {"prompt_ids": [50], "tokens": [40]*8+list(range(1, 17))}},
            "pilot_fresh_L": {e["event_id"]: {"tokens": list(range(1, 9))}}}


class ExtensionTests(unittest.TestCase):
    def test_quota_selection_unique_prompt_deterministic_and_outcome_blind(self):
        candidates = [{"event_id": str(i)+":"+str(j), "prompt_id": str(i),
                       "dataset": "gsm8k" if i%2 else "math500",
                       "cohort": "enriched" if i%3 else "representative"}
                      for i in range(30) for j in range(2)]
        a = choose_events(candidates, config())
        b = choose_events(list(reversed(candidates)), config())
        self.assertEqual(a, b)
        self.assertEqual(len({x["prompt_id"] for x in a}), 10)
        self.assertEqual(sum(x["dataset"] == "gsm8k" for x in a), 5)
        self.assertEqual(sum(x["cohort"] == "enriched" for x in a), 6)
        with self.assertRaisesRegex(ValueError, "no silent relaxation"):
            choose_events(candidates[:4], config())

    def test_predicted_has_no_reference_argument_and_obeys_span_budget(self):
        import inspect
        self.assertEqual(list(inspect.signature(predicted_positions).parameters), ["obs", "config"])
        obs = Observation(tuple(range(24)), (0.95,)*24, (0.95,)*24, tuple(range(24)))
        picked = predicted_positions(obs, config())
        self.assertLessEqual(len(picked), 6)
        self.assertGreaterEqual(len(picked), 2)
        bad = Observation(obs.old_tokens, obs.draft_confidence, (0.1,)*24, obs.verifier_top1)
        self.assertEqual(predicted_positions(bad, config()), [])
        # Eight-token suffix has budget TWO, never the four-KEEP illustration.
        self.assertEqual(span_positions([True]*8, [0]*8, config(), oracle=True), [0, 1])
        self.assertEqual(span_positions([True, False]*4, [0]*8, config(), oracle=True), [])

    def test_readonly_import_rejects_incomplete_identity_and_keeps_input_bytes(self):
        import hashlib
        pilot_config = {**config(), "proposal_length": 32, "allow_nf4_fallback": False}
        identity = {"protocol": "suffix-top1-v4-audited-canonical-replay", "source_digest": PILOT_SOURCE,
                    "environment": {"commit": PILOT_REVISION}, "target_precision": "fp16", "config": pilot_config}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"nested"; root.mkdir()
            path = root/"checkpoint.sqlite"
            store = Store(path, identity)
            for key in ("references_complete", "test1_complete", "test2_complete"):
                store.put("meta", key, True)
            store.put("meta", "test2_design", {"members": {}})
            store.close()
            atomic_json(root/"run-manifest.json", identity)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = read_pilot(Path(tmp))
            self.assertEqual(loaded["checkpoint_sha256"], before)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            bad = deepcopy(identity); bad["target_precision"] = "int8"
            atomic_json(root/"run-manifest.json", bad)
            with self.assertRaisesRegex(ValueError, "FP16"):
                read_pilot(Path(tmp))

    def test_freeze_uses_only_dllm_and_requires_complete_extension_reference(self):
        records, members = {}, {}
        for j in range(16):
            dataset = "gsm8k" if j%2 else "math500"
            prompt, event = f"{dataset}:{j}", f"{dataset}:{j}:0:dllm_top1"
            ref = {"id": prompt, "tokens": list(range(64)), "prompt_ids": [99]}
            draft = list(range(32)); draft[7] = 77
            row = {"id": event, "prompt_id": prompt, "source": "dllm_top1", "dataset": dataset, "offset": 0,
                   "draft": {"tokens": draft, "confidence": [0.9]*32}, "reference": list(range(32)),
                   "feedback": {"top1": list(range(32)), "probability": [0.9]*32}}
            row["metrics"] = suffix_metrics(draft, row["reference"], row["feedback"]["top1"])
            records[("reference", prompt)], records[("test1", event)] = ref, row
            records[("test2", event+"|dllm_top1|fresh_L")] = {"tokens": list(range(8, 32)), "anchors": [],
                                                          "horizon": 24, "verified_accepted_prefix": 24}
            records[("pair_complete", event+"|dllm_top1")] = True
            members[event] = {"cohort": "enriched" if j < 8 else "representative"}
        # A corrupted AR row is never inspected or selected by this protocol.
        records[("test1", "AR-only")] = {"source": "ar_greedy"}
        members["AR-only"] = {"cohort": "enriched"}
        # End requires offset+40; do not pad/truncate missing target labels.
        records[("reference", "math500:0")]["tokens"] = list(range(39))
        pilot = {"records": records, "design": {"members": members},
                 "manifest": {"tokenizer": {"eos_token_id": 151645}}, "checkpoint_sha256": "fixture"}
        frozen = freeze_inputs(pilot, config())
        self.assertEqual(frozen["eligible_events"], 15)
        self.assertEqual(len(frozen["events"]), 10)
        self.assertTrue(all(e["source"] == "dllm_top1" for e in frozen["events"]))
        self.assertTrue(all(e["prompt_id"] != "math500:0" for e in frozen["events"]))
        self.assertTrue(all(e["H"] == e["L"]+8 for e in frozen["events"]))

    def test_complete_orchestration_packs_90_arms_and_never_loads_ar_drafter(self):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            self.skipTest("Optional torch absent")
        from types import SimpleNamespace
        inputs = mini_inputs()
        template = deepcopy(inputs["events"][0]); ref = deepcopy(inputs["references"][template["prompt_id"]])
        inputs["events"], inputs["references"], inputs["pilot_fresh_L"] = [], {}, {}
        for j in range(10):
            e = deepcopy(template)
            e["dataset"] = "gsm8k" if j < 5 else "math500"
            e["cohort"] = "enriched" if j < 6 else "representative"
            e["prompt_id"], e["event_id"] = f"{e['dataset']}:{j}", f"event:{j}"
            inputs["events"].append(e)
            inputs["references"][e["prompt_id"]] = ref
            inputs["pilot_fresh_L"][e["event_id"]] = {"tokens": list(range(1, 9))}
        env = {"commit": "a"*40, "cuda": "12.6", "gpus": ["T4", "T4"],
               "target_device_map": {"layer": "0"}, "target_attention": "sdpa",
               "allow_fp16_reduced_precision_reduction": False,
               "packages": {"torch": "fixture", "transformers": "fixture", "accelerate": "fixture"}}
        inputs.update(plan_digest="fixture", pilot_checkpoint_sha256="fixture",
            pilot_manifest={"config": {}, "asset_digest": "fixture", "target_precision": "fp16",
                            "tokenizer": {}, "environment": env})
        from suffix_lab.assets import TARGET, DLLM
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [50, 40], decode=lambda ids: str(ids[0]))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch("suffix_lab.extension.read_pilot", return_value={}), \
             patch("suffix_lab.extension.freeze_inputs", return_value=inputs), \
             patch("suffix_lab.extension.prepare_models", return_value=({TARGET: "target", DLLM: "dllm"}, {})), \
             patch("torch.cuda.device_count", return_value=2), \
             patch("suffix_lab.runtime.tokenizers", return_value=(tokenizer, {})), \
             patch("suffix_lab.runtime.load_target", return_value=(SimpleNamespace(hf_device_map={"layer": 0}), "fp16", None)), \
             patch("suffix_lab.runtime.load_model", return_value=object()) as load, \
             patch("suffix_lab.runtime.ARSession", return_value=ToyVerifier()), \
             patch("suffix_lab.runtime.DLLMDrafter", return_value=ToyDrafter()), \
             patch("suffix_lab.runtime.target_consistency", return_value={"passed": True}), \
             patch("suffix_lab.runtime.release"), patch("suffix_lab.experiment.environment", return_value=env):
            execute("input", tmp, "a"*40)
            load.assert_called_once_with("dllm", "cuda:1")
            with zipfile.ZipFile(Path(tmp)/"extension_checkpoint.zip") as z:
                status = json.loads(z.read("execution_status.json"))
                self.assertEqual(status["status"], "complete")
                self.assertEqual(status["completed_arms"], 90)
                self.assertEqual(len(z.read("arms.jsonl").splitlines()), 90)

    def test_native_subblock_order_limits_extension_backwards_influence(self):
        def generate(prefix_len):
            class Provider:
                def predict(self, canvas, block, positions):
                    return {p: (p-prefix_len+1, 0.99 if p >= prefix_len+8 else 0.5) for p in positions}
                def stats(self): return {}
            return strict_top1([50]*prefix_len, 16, {}, Provider(), mask_id=62)
        e = mini_inputs()["events"][0]
        # Endpoint 16 is aligned: all old MASKs must finish before extension.
        aligned = generate(8)
        m = arm_metrics({**e, "prefix": [50]*8}, {"horizon": 16, "anchors": []}, aligned.tokens, aligned.trace)
        self.assertEqual(m["extension_before_old_complete_count"], 0)
        # Endpoint17 shares a subblock: extension can win confidence selection.
        joint = generate(9)
        m = arm_metrics(e, {"horizon": 16, "anchors": []}, joint.tokens, joint.trace)
        self.assertGreater(m["extension_before_old_complete_count"], 0)
        self.assertGreater(m["extension_visible_unresolved_old_pairs"], 0)
        # A boundary token uses previous-block carry, even if unmasked later.
        boundary = generate(25)
        m = arm_metrics({**e, "prefix": [50]*25}, {"horizon": 16, "anchors": []}, boundary.tokens, boundary.trace)
        self.assertGreater(m["extension_before_old_complete_count"], 0)
        self.assertEqual(m["extension_visible_unresolved_old_pairs"], 0)

    def test_correction_excluded_and_extended_gold_from_full_reference(self):
        ref = {"tokens": list(range(80)), "prompt_ids": [99]}
        proposal = list(range(32)); proposal[12] = 77
        row = {"id": "e", "prompt_id": "p", "source": "dllm_top1", "dataset": "math500", "offset": 0,
               "draft": {"tokens": proposal, "confidence": [0.9]*32}, "reference": list(range(32)),
               "feedback": {"top1": list(range(32)), "probability": [0.9]*32}}
        row["metrics"] = suffix_metrics(proposal, row["reference"], row["feedback"]["top1"])
        validate_event(row, ref)
        e = event_plan(row, ref, "enriched", config())
        self.assertEqual(e["L"], 19)
        self.assertEqual(e["prefix"], [99]+list(range(13)))
        self.assertEqual(e["old"], proposal[13:])
        self.assertEqual(e["reference"], list(range(13, 40)))
        corrupted = deepcopy(row); corrupted["feedback"]["top1"][12] = 66
        corrupted["metrics"] = suffix_metrics(proposal, row["reference"], corrupted["feedback"]["top1"])
        with self.assertRaisesRegex(ValueError, "correction"):
            validate_event(corrupted, ref)

    def test_all_arms_one_forward_per_unmask_sequential_locks_stage1(self):
        inputs, drafter, verifier = mini_inputs(), ToyDrafter(), ToyVerifier()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            store = Store(Path(tmp)/"extension.sqlite")
            run_matrix(inputs, config(), store, drafter, verifier)
            rows = store.rows("arm")
            self.assertEqual(len(rows), 9)
            self.assertEqual(sum(r["alias_of"] is not None for r in rows), 3)
            self.assertEqual(verifier.calls, 6)
            by_name = {r["method"]: r for r in rows}
            for strategy in ("fresh", "oracle", "predicted"):
                short, seq = by_name[strategy+"_L"], by_name[strategy+"_H_sequential"]
                self.assertEqual(seq["tokens"][:8], short["tokens"])
                self.assertEqual(seq["stats"]["unmask_steps"], 16-len(seq["anchors"]))
                self.assertEqual(len(seq["trace"]), seq["stats"]["unmask_steps"])
                self.assertTrue(all(s["mask_before"]-s["mask_after"] == 1 for s in seq["trace"]))
            # Every unique stage is run cached and without cache. Extension is
            # generated only after all L completed tokens enter its prefix.
            extension_calls = [c for c in drafter.calls if len(c[0]) == 17]
            self.assertEqual(len(extension_calls), 4)
            self.assertTrue(all(c[0][-8:] == list(range(1, 9)) for c in extension_calls))
            store.close()

    def test_wrong_keep_caps_sequential_acceptance_before_extension(self):
        inputs = mini_inputs(); inputs["events"][0]["anchors"]["predicted"] = [[3, 99]]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            store = Store(Path(tmp)/"extension.sqlite")
            run_matrix(inputs, config(), store, ToyDrafter(), ToyVerifier())
            key = inputs["events"][0]["event_id"]+"|predicted_H_sequential"
            row = store.get("arm", key)
            self.assertEqual(row["metrics"]["accepted_prefix"], 3)
            self.assertEqual(row["metrics"]["accepted_extension"], 0)
            self.assertEqual(row["metrics"]["first_mismatch_region"], "KEEP")
            store.close()

    def test_interrupted_run_resumes_without_redrafting_completed_arms(self):
        inputs = mini_inputs()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            store = Store(Path(tmp)/"extension.sqlite")
            def interrupt():
                if len(store.rows("arm")) == 3:
                    raise RuntimeError("interruption")
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run_matrix(inputs, config(), store, ToyDrafter(), ToyVerifier(), interrupt)
            first = store.rows("arm")
            self.assertIsNone(store.get("meta", "complete"))
            drafter = ToyDrafter(); verifier = ToyVerifier()
            run_matrix(inputs, config(), store, drafter, verifier)
            self.assertTrue(store.get("meta", "complete"))
            self.assertEqual(verifier.calls, 3)
            for row in first:
                self.assertEqual(store.get("arm", row["id"]), row)
            # Final rerun is a no-op.
            run_matrix(inputs, config(), store, drafter, verifier)
            self.assertEqual(verifier.calls, 3)
            store.close()

    def test_cache_disagreement_preserves_evidence_and_pilot_baseline_is_audited(self):
        with self.assertRaisesRegex(AssertionError, "Cached/full") as ctx:
            audited_draft(ToyDrafter(drift=True), [50]+[40]*8, 8, {})
        self.assertIn("cached", ctx.exception.evidence)
        inputs = mini_inputs(); inputs["pilot_fresh_L"][inputs["events"][0]["event_id"]]["tokens"][0] = 99
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            store = Store(Path(tmp)/"extension.sqlite")
            with self.assertRaisesRegex(AssertionError, "saved pilot"):
                run_matrix(inputs, config(), store, ToyDrafter(), ToyVerifier())
            self.assertFalse(store.rows("arm"))
            store.close()

    def test_metrics_schedule_matched_ceiling_and_extension_classification(self):
        e = mini_inputs()["events"][0]
        p = {"horizon": 16, "anchors": []}
        m = arm_metrics(e, p, list(range(1, 17)), [])
        self.assertTrue(m["ceiling"])
        self.assertIsNone(m["first_mismatch_position_0based"])
        self.assertEqual(m["accepted_extension"], 8)
        tokens = list(range(1, 17)); tokens[10] = 99
        self.assertEqual(arm_metrics(e, p, tokens, [])["first_mismatch_region"], "EXTEND")
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            store = Store(Path(tmp)/"extension.sqlite")
            inputs = mini_inputs()
            run_matrix(inputs, config(), store, ToyDrafter(), ToyVerifier())
            paired = paired_metrics(store.rows("arm"))
            seq = next(x for x in paired if x["method"] == "oracle_H_sequential")
            self.assertEqual(seq["baseline"], "fresh_H_sequential")
            summary = report(tmp, inputs, store)
            self.assertTrue(summary["complete"])
            self.assertFalse(summary["latency_study"])
            for file in ("arms.csv", "paired.csv", "events.csv", "cases.html", "REPORT.md"):
                self.assertTrue((Path(tmp)/file).is_file())
            class Tokenizer:
                def decode(self, ids): return "<script>&bad"
            html = case_html(inputs["events"], store.rows("arm"), paired, Tokenizer())
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            archive = pack_extension(tmp, store)
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertIn("checkpoint.sqlite", z.namelist())
                self.assertNotIn("extension.sqlite", z.namelist())
            store.close()

    def test_failure_before_model_setup_still_exports_status_zip(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch("suffix_lab.extension.read_pilot", side_effect=ValueError("input invalid")):
            with self.assertRaisesRegex(ValueError, "input invalid"):
                execute("not-a-real-input", tmp, "a"*40)
            with zipfile.ZipFile(Path(tmp)/"extension_checkpoint.zip") as z:
                self.assertEqual(json.loads(z.read("execution_status.json"))["status"], "failed")
                self.assertIn("input invalid", json.loads(z.read("failure.json"))["message"])


if __name__ == "__main__":
    unittest.main()
