"""Independent commit, B60 geometry, offline alignment, lifecycle and ZIP resume checks."""
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
from suffix_lab.long60 import (execute, generate_prompt, audit_and_diagnose, pack,
                               restore, validate_config, numerical_identity, BudgetReached)
from suffix_lab.long60_plan import decide, diagnose, matching_runs, monotone_chain, validate_draft
from suffix_lab.store import Store

EOS = {999999}


class Drafter:
    def __init__(self, clock=None):
        self.calls, self.clock = [], clock
    def draft(self, prefix, length, anchors=None):
        self.calls.append((list(prefix), length))
        if self.clock is not None:
            self.clock[0] += 1
        class Provider:
            n = 0
            def predict(self, canvas, block, positions):
                self.n += 1
                return {p: (p+10000 if p == len(prefix)+4 else p, .9) for p in positions}
            def stats(self):
                return {"unmask_forwards": self.n, "total_forwards": self.n, "input_tokens": self.n*32}
        return strict_top1(prefix, length, anchors or {}, Provider())


class Tokenizer:
    def apply_chat_template(self, messages, **kw):
        return [1000]
    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


def bundle(clock=None):
    def canonical(p, n):
        return {"tokens": list(range(len(p), len(p)+n)), "stats": {"total_forwards": n}}
    return SimpleNamespace(tokenizer=Tokenizer(), drafter=Drafter(clock), eos=EOS,
        canonical=canonical,
        verifier=SimpleNamespace(score=lambda p, t: {"top1_with_bonus": list(range(len(p), len(p)+len(t)+1)), "forwards": 1}))


def config():
    c = json.loads((ROOT/"configs/long60.json").read_text())
    c.update(prompts_per_dataset=1, max_new_tokens=13)
    return c


def prompt(ds="gsm8k"):
    return {"id": ds+":0", "dataset": ds, "question": "<script>question</script>", "index": 0}


class Long60Tests(unittest.TestCase):
    def test_resume_numerical_fingerprint_excludes_driver_and_memory_but_keeps_gpu_packages(self):
        r = {"environment": {"python": "3.13", "cuda": "12.6", "gpus": ["T4", "T4"],
             "packages": {"torch": "2.9.0", "transformers": "4.53.3"}, "driver": ["old"]},
             "tokenizer": {"shared_vocab_sha256": "abc"}, "target_device_map": {"x": "0"},
             "target_precision": "fp16", "effective_eos_ids": [999]}
        changed = deepcopy(r)
        changed["environment"]["driver"] = ["new"]
        self.assertEqual(numerical_identity(r), numerical_identity(changed))
        changed["environment"]["gpus"] = ["A100", "A100"]
        self.assertNotEqual(numerical_identity(r), numerical_identity(changed))

    def test_all_accepted_uses_real_bonus(self):
        row = decide([1, 2, 3], [1, 2, 3, 4], EOS, 10)
        self.assertEqual(row["committed_tokens"], [1, 2, 3, 4])
        self.assertEqual((row["first_rejection"], row["bonus"]), (None, 4))

    def test_first_and_last_rejection(self):
        self.assertEqual(decide([9, 2], [1, 2, 3], EOS, 10)["committed_tokens"], [1])
        self.assertEqual(decide([1, 9], [1, 2, 3], EOS, 10)["committed_tokens"], [1, 2])

    def test_eos_draft_correction_bonus_and_cap(self):
        eos = next(iter(EOS))
        self.assertEqual(decide([1, eos], [1, eos, 77], EOS, 10)["committed_tokens"], [1, eos])
        self.assertEqual(decide([1, 9], [1, eos, 77], EOS, 10)["committed_tokens"], [1, eos])
        self.assertTrue(decide([1, 2], [1, 2, eos], EOS, 10)["terminated"])
        self.assertEqual(decide([1, 2, 3], [1, 2, 3, 4], EOS, 2)["committed_tokens"], [1, 2])
        with self.assertRaises(ValueError):
            decide([eos, 2], [eos, 2, 3], EOS, 10)
        with self.assertRaises(ValueError):
            decide([1, 2], [1, 2], EOS, 10)

    def test_actual_sixty_updates_across_native_blocks(self):
        d = Drafter()
        for phase in (0, 1, 7, 8, 31):
            ids = [1000]*(32+phase)
            result = asdict(d.draft(ids, 60))
            validate_draft(result, 60, ids)
            self.assertEqual(result["stats"]["unmask_forwards"], 60)
            self.assertGreaterEqual(len({t["block_start"] for t in result["trace"]}), 2)

    def test_shift_runs_and_bridge_change_offset(self):
        old = [10, 11, 12, 13, 20, 21, 22, 23]
        gold = [99, 10, 11, 12, 13, 88, 20, 21, 22, 23]
        runs = matching_runs(old, gold, 8)
        chain = monotone_chain(runs)
        self.assertEqual([(r["length"], r["delta"]) for r in chain], [(4, 1), (4, 2)])
        self.assertEqual(sum(r["length"] for r in chain), 8)
        row = {"id": "x", "prompt_id": "p", "dataset": "gsm8k", "offset": 0,
               "proposal": [77]+old, "decision": {"first_rejection": 0, "correction": 99}}
        d = diagnose(row, [99]+gold, EOS, config())
        self.assertEqual((d["bridges"][0]["source_length"], d["bridges"][0]["target_ids"]), (0, [88]))

    def test_chain_never_overlaps_or_reverses(self):
        rs = matching_runs([1]*20, [1]*20, 8)
        chain = monotone_chain(rs)
        for a, b in zip(chain, chain[1:]):
            self.assertLessEqual(a["source_end"], b["source_start"])
            self.assertLessEqual(a["destination_end"], b["destination_start"])
        self.assertLessEqual(sum(r["length"] for r in chain), 20)
        self.assertTrue(all(r["delta"] != 0 for r in rs))

    def test_nonzero_radius_and_endpoint_censoring(self):
        old = list(range(10, 20))
        self.assertFalse(matching_runs(old, [999]*12+old, 8))
        self.assertEqual(matching_runs(old, [999]*12+old, 16)[0]["delta"], 12)
        row = {"id": "x", "prompt_id": "p", "dataset": "gsm8k", "offset": 0,
               "proposal": [77]+old, "decision": {"first_rejection": 0, "correction": 2}}
        d = diagnose(row, [2, 999]+old, EOS, config())
        self.assertTrue(d["longest_right_censored"])
        self.assertTrue(d["right_reference_censored"])

    def test_terminal_and_empty_suffix_not_reused(self):
        row = {"id": "x", "prompt_id": "p", "dataset": "gsm8k", "offset": 0,
               "proposal": [77, 1, 2, 3, 4], "decision": {"first_rejection": 0, "correction": next(iter(EOS))}}
        d = diagnose(row, [next(iter(EOS))], EOS, config())
        self.assertEqual((d["old"], d["reference"], d["chain"]), ([], [], []))
        row["decision"]["first_rejection"] = None
        self.assertIsNone(diagnose(row, [], EOS, config()))

    def test_round_resume_uses_prefix_not_future_labels(self):
        c, b = config(), bundle()
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder)/"x.sqlite")
            def boundary():
                if store.rows("round"):
                    raise BudgetReached()
            with self.assertRaises(BudgetReached), redirect_stdout(io.StringIO()):
                generate_prompt(prompt(), [1000], c, store, b, boundary, lambda: None)
            self.assertEqual(len(store.rows("round")), 1)
            self.assertEqual(store.rows("reference"), [])
            self.assertEqual(store.rows("diagnostic"), [])
            with redirect_stdout(io.StringIO()):
                emitted, rounds = generate_prompt(prompt(), [1000], c, store, b, lambda: None, lambda: None)
                audit_and_diagnose(prompt(), [1000], emitted, rounds, c, store, b)
            self.assertEqual(emitted, list(range(1, 14)))
            self.assertEqual([n for _, n in b.drafter.calls], [60]*3)
            self.assertTrue(store.get("prompt", "gsm8k:0")["canonical_audit_passed"])
            store.close()

    def test_full_cpu_lifecycle_report_and_zip(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/"out"
            with patch("suffix_lab.long60.prepare_prompts", return_value=[prompt(), prompt("math500")]), \
                 patch("suffix_lab.long60.load_bundle", return_value=bundle()), \
                 patch("suffix_lab.long60.clear_unused"), redirect_stdout(io.StringIO()):
                self.assertEqual(execute(out, "a"*40, config()), "complete")
                self.assertEqual(execute(out, "a"*40, config()), "complete")
            summary = json.loads((out/"summary.json").read_text())
            self.assertEqual((summary["completed_prompts"], summary["audited_rounds"]), (2, 6))
            with zipfile.ZipFile(out/"long60_checkpoint.zip") as z:
                self.assertIsNone(z.testzip())
                self.assertIn("checkpoint.sqlite", z.namelist())
                self.assertIn("bridges.csv", z.namelist())

    def test_soft_budget_zip_restore_finishes_remaining_rounds(self):
        clock, c = [0], config()
        c.update(datasets=["gsm8k"], soft_budget_seconds=1.5)
        with tempfile.TemporaryDirectory() as folder:
            out, resumed = Path(folder)/"first", Path(folder)/"second"
            with patch("suffix_lab.long60.prepare_prompts", return_value=[prompt()]), \
                 patch("suffix_lab.long60.load_bundle", side_effect=lambda *a: bundle(clock)), \
                 patch("suffix_lab.long60.time.monotonic", side_effect=lambda: clock[0]), \
                 patch("suffix_lab.long60.clear_unused"), redirect_stdout(io.StringIO()):
                self.assertEqual(execute(out, "a"*40, c), "budget_checkpoint")
                self.assertEqual(json.loads((out/"execution_status.json").read_text())["saved_rounds"], 2)
                clock[0] = 0
                self.assertEqual(execute(resumed, "a"*40, c, out/"long60_checkpoint.zip"), "complete")
                summary = json.loads((resumed/"summary.json").read_text())
                self.assertEqual(summary["saved_rounds"], 3)

    def test_canonical_failure_preserves_raw_and_does_not_label(self):
        b = bundle()
        b.canonical = lambda p, n: {"tokens": [888]*n, "stats": {}}
        c = config()
        c["datasets"] = ["gsm8k"]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with patch("suffix_lab.long60.prepare_prompts", return_value=[prompt()]), \
                 patch("suffix_lab.long60.load_bundle", return_value=b), \
                 patch("suffix_lab.long60.clear_unused"), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(AssertionError, "canonical greedy"):
                    execute(out, "a"*40, c)
            self.assertTrue((out/"failure.json").exists())
            self.assertEqual((out/"diagnostics.jsonl").read_text(), "")
            with zipfile.ZipFile(out/"long60_checkpoint.zip") as z:
                self.assertIn("failure.json", z.namelist())

    def test_configuration_and_resume_identity_reject_changes(self):
        c = config()
        validate_config(c)
        for key, value in (("draft_length", 32), ("target_precision", "int8"), ("reference_lookahead", 60)):
            changed = deepcopy(c)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_config(changed)
        with tempfile.TemporaryDirectory() as folder:
            out, other = Path(folder)/"a", Path(folder)/"b"
            out.mkdir()
            other.mkdir()
            store = Store(out/"long60.sqlite", {"x": 1})
            pack(out, store)
            store.close()
            with self.assertRaisesRegex(ValueError, "differ"):
                restore(out/"long60_checkpoint.zip", other, {"x": 2})
            self.assertFalse((other/"long60.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
