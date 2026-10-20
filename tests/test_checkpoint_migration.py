"""Readonly unpacked-input import, narrow legacy compatibility and idempotent resume."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy
from suffix_lab.migration import import_checkpoint, locate_unpacked, validate_identity, replay_prior_failures, OLD_REVISION, OLD_SOURCE
from suffix_lab.store import Store, atomic_json
from suffix_lab.metrics import suffix_metrics


class MigrationTests(unittest.TestCase):
    def test_archived_failure_is_replayed_and_its_model_outputs_are_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, config, _, ref, row = self.fixture(root/"input")
            import_checkpoint(root/"input", root/"output", config, "a"*40)
            store = Store(root/"output"/"run.sqlite")
            class Session:
                def score(self, prefix, proposal, top_k):
                    return {"top1": proposal, "latency_ms": 1.}
                def score_replay(self, prompt, committed, proposal, top_k):
                    return {"top1": list(range(32)), "latency_ms": 10.}
            audit = replay_prior_failures(Session(), store, 8)
            self.assertTrue(audit[0]["passed"])
            self.assertEqual(store.rows("migration_replay")[0]["top1"], list(range(32)))
            self.assertEqual(len(store.rows("prior_failure")), 1)
            self.assertFalse(store.rows("failure"))
            store.close()

    def fixture(self, root, revision=OLD_REVISION, test2=False):
        folder = root/"nested"/"checkpoint"
        folder.mkdir(parents=True)
        config = {"proposal_length": 32}
        identity = {"config": config, "protocol": "suffix-top1-v4-two-cohort-repairability",
            "source_digest": OLD_SOURCE, "asset_digest": "fixture", "target_precision": "fp16", "tokenizer": {},
            "environment": {"commit": revision, "cuda": "12.6", "gpus": ["T4", "T4"],
                "target_device_map": {"layer": 0}, "target_attention": "sdpa",
                "allow_fp16_reduced_precision_reduction": False,
                "packages": {"torch": "2.9.0", "transformers": "4.53.3", "accelerate": "1.10.1"}}}
        ref = {"id": "gsm8k:1", "tokens": list(range(40)), "prompt_ids": [90], "offsets": [0]}
        gold = list(range(32))
        proposal = gold[:]
        proposal[5] = 99
        row = {"id": "gsm8k:1:0:dllm_top1", "prompt_id": "gsm8k:1", "offset": 0,
               "reference": gold, "draft": {"tokens": proposal}, "feedback": {"top1": gold},
               "metrics": suffix_metrics(proposal, gold, gold)}
        store = Store(folder/"checkpoint.sqlite", identity)
        store.put("reference", ref["id"], ref)
        store.put("test1", row["id"], row)
        store.put("failure", "gsm8k:1:0:dllm_top1", {"draft": proposal, "reference": gold})
        store.put("meta", "references_complete", True)
        if test2:
            store.put("test2", "arm", {})
        store.close()
        atomic_json(folder/"run-manifest.json", identity)
        return folder, config, identity, ref, row

    def test_import_preserves_passing_rows_and_input_and_archives_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder, config, identity, ref, row = self.fixture(root/"input")
            before = hashlib.sha256((folder/"checkpoint.sqlite").read_bytes()).hexdigest()
            output = root/"output"
            import_checkpoint(root/"input", output, config, "a"*40)
            self.assertEqual(before, hashlib.sha256((folder/"checkpoint.sqlite").read_bytes()).hexdigest())
            store = Store(output/"run.sqlite")
            self.assertEqual(store.get("reference", ref["id"]), ref)
            self.assertEqual(store.get("test1", row["id"]), row)
            self.assertEqual(store.get("meta", "migration_pending"), identity)
            self.assertIsNone(store.get("meta", "identity"))
            self.assertFalse(store.rows("failure"))
            self.assertEqual(len(store.rows("prior_failure")), 1)
            store.close()
            with self.assertRaisesRegex(ValueError, "live database"):
                import_checkpoint(root/"input", output, config, "a"*40)

    def test_wrong_config_unreviewed_revision_or_existing_test2_are_rejected(self):
        for kind in ("config", "revision", "test2"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                folder, config, *_ = self.fixture(root/"input", revision="b"*40 if kind == "revision" else OLD_REVISION,
                                                  test2=kind == "test2")
                if kind == "config":
                    config = {"proposal_length": 16}
                with self.assertRaises(ValueError):
                    import_checkpoint(root/"input", root/"output", config, "a"*40)
                self.assertFalse((root/"output"/"run.sqlite").exists())

    def test_current_revision_unpacked_restore_preserves_full_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder, config, identity, *_ = self.fixture(root/"input", revision="a"*40, test2=True)
            import_checkpoint(root/"input", root/"output", config, "a"*40)
            store = Store(root/"output"/"run.sqlite", identity)
            self.assertEqual(store.rows("test2"), [{}])
            self.assertIsNone(store.get("meta", "migration_pending"))
            store.close()

    def test_ambiguous_unpacked_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root/"one")
            self.fixture(root/"two")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                locate_unpacked(root)

    def test_precision_assets_and_numerical_versions_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            _, _, old, *_ = self.fixture(Path(temp)/"input")
            new = deepcopy(old)
            new["environment"]["commit"] = "a"*40
            validate_identity(old, new)
            for key in ("asset_digest", "target_precision", "config"):
                changed = deepcopy(new)
                changed[key] = "different"
                with self.assertRaises(ValueError):
                    validate_identity(old, changed)
            changed = deepcopy(new)
            changed["environment"]["packages"]["torch"] = "different"
            with self.assertRaises(ValueError):
                validate_identity(old, changed)


if __name__ == "__main__":
    unittest.main()
