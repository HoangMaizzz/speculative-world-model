"""B64 is ordinary fresh drafting; protects geometry, artifacts and B60 isolation."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from suffix_lab.assets import ROOT
from suffix_lab.long60 import execute, generate_prompt, validate_config, restore, pack
from suffix_lab.long60_plan import diagnose, matching_runs
from suffix_lab.store import Store
from test_long60 import bundle, prompt


def config():
    c = json.loads((ROOT/"configs/long64.json").read_text())
    c.update(prompts_per_dataset=1, max_new_tokens=13)
    return c


class Long64Tests(unittest.TestCase):
    def test_capacity_horizon_and_lookahead(self):
        c = config()
        validate_config(c)
        self.assertEqual((c["draft_length"], c["diagnostic_horizon"], c["shift_radii"][-1]), (64, 63, 63))
        self.assertEqual((c["physical_block_size"], c["subblock_size"]), (32, 8))
        self.assertGreaterEqual(c["reference_lookahead"], 127)
        self.assertEqual(c["draft_length"] % c["subblock_size"], 0)
        self.assertEqual(json.loads((ROOT/"configs/long60.json").read_text())["draft_length"], 60)

    def test_every_round_is_new_sixty_four_masks_without_anchors(self):
        c, b = config(), bundle()
        original = b.drafter.draft
        anchors_seen = []
        def record(*args, **kw):
            anchors_seen.append(kw.get("anchors"))
            return original(*args, **kw)
        b.drafter.draft = record
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder)/"state.sqlite")
            with redirect_stdout(io.StringIO()):
                emitted, rows = generate_prompt(prompt(), [1000], c, store, b, lambda: None, lambda: None)
            self.assertEqual(emitted, list(range(1, 14)))
            self.assertEqual([length for _, length in b.drafter.calls], [64, 64, 64])
            self.assertEqual(anchors_seen, [{}, {}, {}])
            self.assertEqual([len(r["draft"]["trace"]) for r in rows], [64]*3)
            self.assertEqual(store.rows("reference"), [])
            store.close()

    def test_diagnostic_uses_sixty_three_future_positions(self):
        old = list(range(100, 163))
        row = {"id": "x", "prompt_id": "p", "dataset": "gsm8k", "offset": 0,
               "proposal": [77]+old, "decision": {"first_rejection": 0, "correction": 2}}
        d = diagnose(row, [2, 3, 4, 5, 6]+old, {999999}, config())
        self.assertEqual((len(d["old"]), len(d["reference"])), (63, 63))
        self.assertEqual(d["runs"][0]["length"], 59)
        self.assertEqual(d["runs"][0]["destination_end"], 63)
        gold = [999]*59+[1, 2, 3, 4]
        self.assertFalse(matching_runs([1, 2, 3, 4], gold, 32))
        self.assertEqual(matching_runs([1, 2, 3, 4], gold, 63)[0]["delta"], 59)

    def test_full_lifecycle_creates_only_b64_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with patch("suffix_lab.long60.prepare_prompts", return_value=[prompt(), prompt("math500")]), \
                 patch("suffix_lab.long60.load_bundle", return_value=bundle()), \
                 patch("suffix_lab.long60.clear_unused"), redirect_stdout(io.StringIO()):
                self.assertEqual(execute(out, "b"*40, config()), "complete")
            self.assertTrue((out/"long64.sqlite").exists())
            self.assertFalse((out/"long60.sqlite").exists())
            self.assertFalse((out/"long60_checkpoint.zip").exists())
            manifest = json.loads((out/"long64-manifest.json").read_text())
            self.assertEqual(manifest["protocol"], "online-dllm64-ar-shift-diagnostics-v1")
            self.assertIn("draft64", (out/"report.md").read_text(encoding="utf-8"))
            with zipfile.ZipFile(out/"long64_checkpoint.zip") as z:
                self.assertIsNone(z.testzip())
                self.assertNotIn("long64.sqlite", z.namelist())
                self.assertIn("checkpoint.sqlite", z.namelist())

    def test_b64_budget_checkpoint_restore_finishes(self):
        clock, c = [0], config()
        c.update(datasets=["gsm8k"], soft_budget_seconds=1.5)
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder)/"first", Path(folder)/"second"
            with patch("suffix_lab.long60.prepare_prompts", return_value=[prompt()]), \
                 patch("suffix_lab.long60.load_bundle", side_effect=lambda *a: bundle(clock)), \
                 patch("suffix_lab.long60.time.monotonic", side_effect=lambda: clock[0]), \
                 patch("suffix_lab.long60.clear_unused"), redirect_stdout(io.StringIO()):
                self.assertEqual(execute(first, "b"*40, c), "budget_checkpoint")
                clock[0] = 0
                self.assertEqual(execute(second, "b"*40, c, first/"long64_checkpoint.zip"), "complete")
            status = json.loads((second/"execution_status.json").read_text())
            self.assertEqual((status["completed_prompts"], status["saved_rounds"]), (1, 3))

    def test_b60_zip_is_not_a_b64_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = Path(folder)/"old", Path(folder)/"new"
            old.mkdir()
            new.mkdir()
            store = Store(old/"long60.sqlite", {"config": {"draft_length": 60}})
            pack(old, store)
            store.close()
            with self.assertRaisesRegex(ValueError, "differ"):
                restore(old/"long60_checkpoint.zip", new, {"config": {"draft_length": 64}})
            self.assertFalse((new/"long64.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
