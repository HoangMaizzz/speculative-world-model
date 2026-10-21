"""Main200 EOS/selection/reference replay, exact action identity and interruption tests."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
from suffix_lab.assets import ROOT
from suffix_lab.core import strict_top1
from suffix_lab.store import Store
from suffix_lab.test4 import (draft_native, verify_logical, continue_reference, screen_main,
                             run_main, pack, restore, execute, SelectionShortfall, BudgetReached)
from suffix_lab.test4_plan import (through_eos, token_type, prompt_plan, choose_event,
                                 make_event, plans, arm_fingerprint, eos_metrics, changes)
from suffix_lab.test4_report import paired_stats, report
from suffix_lab.verification import VerificationError

EOS = {999}
MODELS = {"target": "pinned", "drafter": "pinned"}


def config():
    c = json.loads((ROOT/"configs/test4_main.json").read_text())
    c.update(events_per_dataset=1, offsets=[0, 24], bootstrap_samples=30)
    return c


def prompt(ds="gsm8k", index=0):
    return {"id": f"{ds}:{index}", "index": index, "dataset": ds,
            "question": "<script>escaped</script>", "source": "primary", "offsets": [0, 24]}


def reference(p=None):
    p = p or prompt()
    return {**p, "prompt_ids": [1000], "tokens": list(range(1, 271)),
            "stop_reason": "token_budget", "reference_censored": False}


class Drafter:
    def __init__(self):
        self.calls = []
    def draft(self, prefix, length, anchors=None, use_prefix_cache=True):
        anchors = anchors or {}
        self.calls.append((tuple(prefix), length, tuple(anchors.items()), use_prefix_cache))
        class Provider:
            n = 0
            def predict(self, canvas, block, positions):
                self.n += 1
                return {p: (777 if p == len(prefix)+7 and length == 24 else
                            778 if p == len(prefix)+1 and not anchors and length == 23 else p, .9)
                        for p in positions}
            def stats(self):
                return {"unmask_forwards": self.n, "total_forwards": self.n, "input_tokens": self.n*32}
        return strict_top1(prefix, length, anchors, Provider(), mask_id=151665)


class Verifier:
    def __init__(self):
        self.calls = []
    def score(self, prefix, candidate, top_k):
        self.calls.append((list(prefix), list(candidate)))
        return {"top1": list(range(len(prefix), len(prefix)+len(candidate)))}


def event_at(a=7, gold=None, old_eos=None):
    ref = reference()
    if gold is not None:
        ref["tokens"] = gold
    tokens = list(range(1, 25))
    tokens[a] = 777
    if old_eos is not None:
        tokens[old_eos] = 999
    return make_event(prompt(), 0, ref, {"tokens": tokens},
                      {"top1": ref["tokens"][:24]}, config(), EOS, MODELS)


class Test4Tests(unittest.TestCase):
    def test_endpoint_rejections_zero_and_twenty_three(self):
        for a in (0, 1, 22, 23):
            e, reason = event_at(a)
            self.assertIsNone(reason)
            self.assertEqual((e["L"], e["extension_length"], e["H"]), (23-a, a, 23))
            self.assertEqual(e["prefix"][-1], a+1)
        e, _ = event_at(23)
        self.assertEqual(plans(e, EOS)[0]["anchors"], plans(e, EOS)[1]["anchors"])

    def test_all_accepted_terminal_correction_early_draft_EOS(self):
        e, _ = event_at()
        original = {"tokens": list(range(1, 25))}
        self.assertEqual(make_event(prompt(), 0, reference(), original,
            {"top1": original["tokens"]}, config(), EOS, MODELS)[1], "all_24_accepted")
        self.assertEqual(event_at(7, list(range(1, 8))+[999])[1], "terminal_correction")
        self.assertEqual(event_at(7, old_eos=11)[1], "early_draft_eos")
        e, reason = event_at(7, old_eos=23)
        self.assertIsNone(reason)
        self.assertNotIn(999, [v for _, v in plans(e, EOS)[2]["anchors"]])

    def test_terminal_reference_retained_no_eos_or_after_eos_oracle_anchors(self):
        gold = list(range(1, 13))+[999]
        e, reason = event_at(7, gold)
        self.assertIsNone(reason)
        self.assertTrue(e["target_terminal_within_horizon"])
        self.assertEqual(e["reference"], [9, 10, 11, 12, 999])
        self.assertEqual(e["oracle_keep_positions"], [0, 1, 2, 3])
        self.assertTrue(all(i < 4 and t != 999 for i, t in e["oracle_anchors"]))
        # KEEP-All cannot know the future target endpoint; it still uses actual
        # non-EOS old tokens, while evaluation caps at target's EOS.
        self.assertGreater(len(plans(e, EOS)[2]["anchors"]), 4)

    def test_censored_reference_never_becomes_acceptance_zero(self):
        self.assertEqual(event_at(7, list(range(1, 30)))[1], "reference_censored")
        with self.assertRaisesRegex(ValueError, "Censored"):
            verify_logical(Verifier(), {"prompt_ids": [1000], "tokens": [1, 2]}, 0, [1, 2, 3], EOS, 8)

    def test_eos_acceptance_and_content_counts_ignore_post_end(self):
        m = eos_metrics([1, 2, 999, 6, 7], [1, 2, 999], EOS)
        self.assertEqual((m["A_inclusive"], m["A_content"], m["I_EOS"]), (3, 2, 1))
        self.assertTrue(m["terminated_correctly"])
        self.assertEqual(m["tokens_beyond_logical_eos"], 2)
        for candidate, expected in (([1, 999, 3], 1), ([1, 8, 999], 1), ([1, 2, 3, 4], 2)):
            m = eos_metrics(candidate, [1, 2, 999], EOS)
            self.assertEqual(m["A_inclusive"], expected)
            self.assertEqual(m["I_EOS"], 0)
        self.assertEqual(through_eos([1, 999, 999], EOS), [1, 999])

    def test_verifier_receives_no_tokens_after_either_endpoint(self):
        verifier = Verifier()
        # Reference ending token3 is a configured stop for this test.
        verify_logical(verifier, {"prompt_ids": [1000], "tokens": [1, 2, 3]}, 0, [1, 2, 3, 8, 9], {3}, 8)
        self.assertEqual(verifier.calls[-1][1], [1, 2, 3])
        verify_logical(verifier, reference(), 0, [1, 2, 3, 8, 9], {3}, 8)
        self.assertEqual(verifier.calls[-1][1], [1, 2, 3])

    def test_uniform_choice_independent_of_outcomes_and_list_order(self):
        rows = [{"offset": i, "old_correct_count": 0} for i in (0, 56, 111, 167, 223)]
        result = choose_event(rows, "gsm8k:0", 42)["offset"]
        for r in rows:
            r.update(fresh_acceptance=100, oracle_gain=-50, old_correct_count=23)
        self.assertEqual(choose_event(list(reversed(rows)), "gsm8k:0", 42)["offset"], result)
        selected = {choose_event(rows, f"gsm8k:{i}", 42)["offset"] for i in range(200)}
        self.assertEqual(selected, {0, 56, 111, 167, 223})

    def test_primary_and_reserve_frozen_no_overlaps_or_question_drift(self):
        c = config()
        ds = {d: [prompt(d, i) for i in range(4)] for d in c["datasets"]}
        primary = [ds[d][0] for d in c["datasets"]]
        p = prompt_plan(primary, ds, c)
        self.assertEqual(p, prompt_plan(primary, ds, c))
        for d in c["datasets"]:
            self.assertEqual(p[d][0]["source"], "primary")
            self.assertEqual(len({r["id"] for r in p[d]}), 4)
        bad = deepcopy(primary); bad[0]["question"] = "changed"
        with self.assertRaisesRegex(ValueError, "question differ"):
            prompt_plan(bad, ds, c)

    def test_action_identity_includes_horizon_values_schedule_and_revision(self):
        e, _ = event_at()
        c, p = config(), plans(e, EOS)[0]
        f = arm_fingerprint(e, p, c, "a"*40, MODELS)
        for variant in ({**p, "horizon": 16}, {**p, "anchors": [[1, 5]]}, {**p, "schedule": "sequential"}):
            self.assertNotEqual(f, arm_fingerprint(e, variant, c, "a"*40, MODELS))
        self.assertNotEqual(f, arm_fingerprint(e, p, c, "b"*40, MODELS))
        # Same actual action aliases despite a different human-readable arm name.
        self.assertEqual(f, arm_fingerprint(e, {**p, "method": "alias"}, c, "a"*40, MODELS))

    def test_raw_token_types_preserve_spaces_and_priority(self):
        for piece, expected in ((" ", "whitespace_only"), ("\n  ", "newline_indentation"),
                                (" +", "lexical_mixed"), ("...", "punctuation"),
                                (" 123", "numeric"), (" hello", "lexical_mixed")):
            self.assertEqual(token_type(piece, 220, EOS), expected)
        self.assertEqual(token_type(" ", 999, EOS), "special_control")

    def test_native_eos_does_not_abort_unresolved_prefix_masks(self):
        class EarlyEOS(Drafter):
            def draft(self, prefix, length, anchors=None, use_prefix_cache=True):
                class Provider:
                    n = 0
                    def predict(self, canvas, block, positions):
                        self.n += 1
                        return {p: (999 if p == len(prefix)+2 else p, .99 if p == len(prefix)+2 else .5) for p in positions}
                    def stats(self):
                        return {"unmask_forwards": self.n}
                return strict_top1(prefix, length, anchors or {}, Provider())
        r = draft_native(EarlyEOS(), [1000], 6, {})
        self.assertEqual(r["trace"][0]["position"], 2)
        self.assertEqual(len(r["trace"]), 6)
        self.assertEqual(through_eos(r["tokens"], EOS), [1, 2, 999])

    def test_screen_then_main_interruption_resume_pack_restore_and_report(self):
        c, drafter, verifier = config(), Drafter(), Verifier()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); identity = {"fixture": True}
            store = Store(root/"test4.sqlite", identity)
            plan = {d: [prompt(d)] for d in c["datasets"]}
            def get_reference(p):
                ref = reference(p); store.put("reference", p["id"], ref); return ref
            events = screen_main(plan, c, store, drafter, verifier, EOS, MODELS, get_reference)
            self.assertEqual(len(events), 2)
            self.assertEqual(len(store.rows("screen")), 4)
            self.assertEqual({call[1] for call in drafter.calls}, {24})
            self.assertEqual(len(store.rows("arm")), 0)
            old_calls = len(drafter.calls)
            self.assertEqual(events, screen_main(plan, c, store, drafter, verifier, EOS, MODELS, get_reference))
            self.assertEqual(old_calls, len(drafter.calls))
            def stop():
                if len(store.rows("arm")) == 2:
                    raise BudgetReached()
            with patch("suffix_lab.runtime.release"), self.assertRaises(BudgetReached):
                run_main(events, c, store, drafter, verifier, EOS, "a"*40, MODELS, checkpoint=stop)
            self.assertEqual(len(store.rows("arm")), 2)
            saved = deepcopy(store.rows("arm"))
            with patch("suffix_lab.runtime.release"):
                run_main(events, c, store, drafter, verifier, EOS, "a"*40, MODELS)
            self.assertEqual(len(store.rows("arm")), 6)
            for row in saved:
                self.assertEqual(store.get("arm", row["id"]), row)
            with patch("suffix_lab.test4_report.plots"):
                summary = report(root, store, c)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["attrition"]["native_drafts_screened"], 4)
            self.assertIn("&lt;script&gt;", (root/"cases.html").read_text())
            (root/"test4-manifest.json").write_text(json.dumps(identity))
            archive = pack(root, store)
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip()); self.assertIn("checkpoint.sqlite", z.namelist())
            destination = root/"restored"; destination.mkdir()
            restore(archive, destination, identity)
            restored = Store(destination/"test4.sqlite", identity)
            self.assertEqual(store.rows("arm"), restored.rows("arm"))
            restored.close()
            bad = root/"bad"; bad.mkdir()
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                restore(archive, bad, {"fixture": False})
            self.assertFalse((bad/"test4.sqlite").exists())
            store.close()

    def test_sampling_exhaustion_explicit_not_quota_relaxation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/"test.sqlite")
            with self.assertRaises(SelectionShortfall):
                screen_main({d: [] for d in config()["datasets"]}, config(), store,
                            Drafter(), Verifier(), EOS, MODELS, lambda _: None)
            store.close()

    def test_bootstrap_pairs_not_mask_observations(self):
        row = {"prompt_id": "p", "delta": 4, "fresh": 2, "accepted": 6, "A_content": 6,
               "full_23_accepted": False, "terminated_correctly": False, "accepted_extension": 0}
        result = paired_stats([row], samples=30)
        self.assertEqual(result["mean_delta_ci95"], [4, 4])
        with self.assertRaisesRegex(ValueError, "one paired event"):
            paired_stats([row, row], samples=30)
        self.assertEqual(paired_stats([])["n"], 0)

    def test_reference_single_token_replay_preserves_history_and_eos(self):
        import torch
        class Model:
            def __init__(self, end=5):
                self.calls, self.end = [], end
            def __call__(self, ids, past_key_values=None, **kwargs):
                self.calls.append(ids.tolist()[0])
                step = 0 if past_key_values is None else past_key_values
                token = 999 if step == self.end else step+1
                logits = torch.full((1, 1, 1001), -10.)
                logits[0, 0, token] = 10.
                return SimpleNamespace(logits=logits, past_key_values=step+1)
        model = Model()
        session = SimpleNamespace(model=model, device="cpu")
        result = continue_reference(session, [1000, 1000], [1, 2, 3], 20, EOS, 100)
        self.assertEqual(result["tokens"], [1, 2, 3, 4, 5, 999])
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(model.calls, [[1000, 1000], [1], [2], [3], [4], [5]])
        with self.assertRaises(VerificationError):
            continue_reference(session, [1000], [1, 8], 20, EOS, 100)
        result = continue_reference(session, [1000], [], 20, EOS, 3)
        self.assertEqual(result["stop_reason"], "context_limit")
        self.assertTrue(result["reference_censored"])
        result = continue_reference(session, [1000], [], 20, EOS, 1)
        self.assertEqual(result["replay_forwards"], 0)
        with self.assertRaisesRegex(ValueError, "after EOS"):
            continue_reference(session, [1000], [1, 999, 2], 20, EOS, 100)

    def test_execute_full_lifecycle_complete_then_model_free_resume(self):
        from suffix_lab import test4, runtime
        c = config()
        refs = {("reference", d+":0"): reference(prompt(d)) for d in c["datasets"]}
        pilot = {"records": refs, "manifest": {}, "checkpoint_sha256": "fixture"}
        fake_tokenizer = SimpleNamespace(eos_token_id=999, all_special_ids=[999],
            apply_chat_template=lambda *a, **kw: [1000], decode=lambda ids, **kw: " " if ids == [220] else str(ids[0]))
        fake_target = SimpleNamespace(hf_device_map={"layers": 0}, config=SimpleNamespace(max_position_embeddings=500, eos_token_id=999),
                                      generation_config=SimpleNamespace(eos_token_id=999))
        fake_model = SimpleNamespace(config=SimpleNamespace(eos_token_id=999), generation_config=SimpleNamespace(eos_token_id=999))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp); (root/"configs").mkdir()
            (root/"configs/test4_main.json").write_text(json.dumps(c))
            from suffix_lab.assets import TARGET, DLLM
            (root/"source-lock.json").write_text(json.dumps({"models": {TARGET: "target", DLLM: "dllm"}}))
            output = root/"output"
            verifier = Verifier(); verifier.model = fake_target; verifier.device = "cpu"
            with patch.object(test4, "ROOT", root), patch.object(test4, "read_pilot", return_value=pilot), \
                 patch.object(test4, "load_prompt_plan", return_value={d: [prompt(d)] for d in c["datasets"]}), \
                 patch.object(test4, "prepare_models", return_value=({TARGET: "target", DLLM: "dllm"}, {})) as prepare, \
                 patch.object(runtime, "tokenizers", return_value=(fake_tokenizer, {})), \
                 patch.object(runtime, "load_target", return_value=(fake_target, "fp16", None)), \
                 patch.object(runtime, "load_model", return_value=fake_model), \
                 patch.object(runtime, "ARSession", return_value=verifier), \
                 patch.object(runtime, "DLLMDrafter", return_value=Drafter()), \
                 patch.object(runtime, "target_consistency", return_value={"passed": True}), \
                 patch.object(runtime, "release"), patch("torch.cuda.device_count", return_value=2), \
                 patch("suffix_lab.migration.validate_identity"), \
                 patch("suffix_lab.experiment.environment", return_value={"commit": "a"*40}), \
                 patch("suffix_lab.test4_report.plots"):
                execute("pilot-input", output, "a"*40)
                status = json.loads((output/"execution_status.json").read_text())
                self.assertEqual((status["status"], status["completed_arms"], status["expected_arms"]), ("complete", 6, 6))
                with zipfile.ZipFile(output/"test4_phase1_checkpoint.zip") as z:
                    for name in ("runtime.json", "preflight.json", "anchor_tokens.json", "prompt_plan.json", "summary.json"):
                        self.assertIn(name, z.namelist())
                execute("pilot-input", output, "a"*40)
                self.assertEqual(prepare.call_count, 1)

    def test_bad_initialization_preserves_existing_checkpoint_and_status(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            output = Path(tmp)
            files = ("test4_phase1_checkpoint.zip", "test4-manifest.json", "execution_status.json")
            for name in files:
                (output/name).write_bytes(b"existing-artifact")
            with patch("suffix_lab.test4.read_pilot", side_effect=ValueError("bad input")):
                with self.assertRaisesRegex(ValueError, "bad input"):
                    execute("bad", output, "a"*40)
            for name in files:
                self.assertEqual((output/name).read_bytes(), b"existing-artifact")
            self.assertTrue((output/"bootstrap_failure.json").exists())


if __name__ == "__main__":
    unittest.main()
