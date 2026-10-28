"""Collection contracts with actual tiny CPU Qwen; not pretrained GPU validation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from suffix_lab.store import digest
from suffix_lab.verifier_features.checkpoint_loader import Checkpoint, PINNED_REVISION
from suffix_lab.verifier_features.offline_labels import label
from suffix_lab.verifier_features.prefix_reconstruction import reconstruct
from suffix_lab.verifier_features.shards import safe_name, Shards, restore


class LabelContracts(unittest.TestCase):
    def event(self):
        return {"id": "r", "prompt_id": "p", "dataset": "gsm8k", "offset": 0,
                "accepted": 0, "source": [1, 2, 3, 4, 1, 2, 3, 4]}

    def test_all_ambiguous_matches_and_zero_shift_are_retained(self):
        v = label(self.event(), [90, 1, 2, 3, 4, 1, 2, 3, 4, 99], {99})
        self.assertEqual({(x["i"], x["j"]) for x in v["matches4"]},
                         {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 4), (4, 0)})
        self.assertEqual(v["matches8"], [{"i": 0, "j": 0}])
        self.assertTrue(any(r["delta"] == 0 for r in v["runs"]))
        self.assertTrue(any(r["delta"] != 0 for r in v["runs"]))
        self.assertTrue(v["target_eos_visible"])
        self.assertEqual(len(v["runs"]), 3)

    def test_bridge_coordinates_and_endpoints(self):
        e = self.event()
        e["source"] = [1, 2, 3, 4, 50, 5, 6, 7, 8]
        v = label(e, [90, 1, 2, 3, 4, 80, 81, 5, 6, 7, 8, 99], {99})
        edge = next(x for x in v["bridge_edges"] if x["source_length"] == 1)
        self.assertEqual(edge["source_ids"], [50])
        self.assertEqual(edge["target_ids"], [80, 81])
        self.assertTrue(any(r["target_endpoint"] for r in v["runs"]))

    def test_no_future_content_is_not_negative_alignment(self):
        v = label(self.event(), [90, 99], {99})
        self.assertTrue(v["reference_exhausted"])
        self.assertEqual(v["runs"], [])

    def test_repeated_ids_preserve_multiple_source_offsets(self):
        v = label(self.event(), [90, 1, 2, 3, 4, 99], {99})
        self.assertEqual({x["i"] for x in v["matches4"]}, {0, 4})

    def test_resume_path_rejects_traversal_and_windows_paths(self):
        for path in ("../x", "/x", "C:/x", "a\\..\\x"):
            self.assertFalse(safe_name(path))
        self.assertTrue(safe_name("shards/part-00000.zip"))

    def test_bad_archive_is_not_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root/"bad.zip", "w") as z:
                z.writestr("feature-manifest.json", "{}")
                z.writestr("../escape", "bad")
            with self.assertRaises(ValueError):
                restore(root/"bad.zip", root/"output")
            self.assertFalse((root/"escape").exists())


AVAILABLE = importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")


@unittest.skipUnless(AVAILABLE, "Optional local CPU Qwen dependencies missing")
class CollectorModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from suffix_lab.runtime import ARSession
        from suffix_lab.long60 import ParallelVerifier
        from suffix_lab.long60_plan import decide
        from suffix_lab.verifier_features.target_model import Target
        torch.set_num_threads(1)
        torch.manual_seed(311)
        cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        cfg._attn_implementation = "sdpa"
        cls.model = Qwen2ForCausalLM(cfg).eval()
        cls.target = Target(cls.model, [0, 1])
        cls.target.environment = lambda: {"test": "tiny-cpu-qwen", "hidden_size": 32}
        cls.target.gpu_memory = lambda: []
        prefix, output, rounds = [1, 2, 3], [], []
        gold = ARSession(cls.model).draft(prefix, 128).tokens
        for idx in range(2):
            x = prefix+output
            proposal = [10+i % 20 for i in range(64)]
            proposal[:2] = gold[len(output):len(output)+2]
            proposal[2] = (gold[len(output)+2]+1) % 64
            f = ParallelVerifier(cls.model).score(x, proposal)
            trace = [{"position": i, "absolute_position": len(x)+i, "token": token,
                      "mask_before": 64-i, "mask_after": 63-i, "step": i,
                      "subblock": (len(x)+i)//8} for i, token in enumerate(proposal)]
            draft = {"tokens": proposal, "confidence": [.5]*64, "order": list(range(64)), "trace": trace,
                     "stats": {"unmask_forwards": 64, "unmask_steps": 64}}
            d = decide(proposal, f["top1_with_bonus"], set(), 6-len(output))
            rounds.append({"id": f"gsm8k:0:{idx:05d}", "prompt_id": "gsm8k:0", "dataset": "gsm8k",
                           "round_index": idx, "offset": len(output), "prefix_digest": digest(x),
                           "proposal": proposal, "draft": draft, "decision": d,
                           "verification": {**f, "parallel_feedback": f,
                                            "canonical_guard": {"numerical_disagreements": []}}})
            output += d["committed_tokens"]
        cls.prompts = [{"id": "gsm8k:0", "dataset": "gsm8k", "prompt_ids": prefix, "tokens": output}]
        cls.rounds = rounds
        cls.reference = {"id": "gsm8k:0", "prompt_ids": prefix, "tokens": gold}
        cls.events, _ = reconstruct(cls.prompts, rounds, set(), 6)
        cls.config = {"schema_version": 1, "layers": [0, 1], "top_k": 16, "micro_probe_steps": 0,
                      "shard_events": 1, "smoke_events": 1, "pilot_events": 1,
                      "smoke_max_old_replay_mismatches": 0, "expected_prompts": 1, "expected_rounds": 2,
                      "expected_events": 2, "minimum_free_disk_gib": 0}

    def checkpoint(self, root):
        content = {"long64-manifest.json": {"code_revision": PINNED_REVISION,
                    "config": {"verification_mode": "canonical_guarded"}},
                   "preflight.json": {"passed": True}, "runtime.json": {"effective_eos_ids": []}}
        for name, v in content.items():
            (root/name).write_text(json.dumps(v))
        for name, rows in (("prompts.jsonl", self.prompts), ("rounds.jsonl", self.rounds),
                           ("references.jsonl", [self.reference])):
            (root/name).write_text("".join(json.dumps(r)+"\n" for r in rows))
        return Checkpoint(root)

    def test_exact_rank_ties_and_full_vocab_entropy(self):
        import torch
        from suffix_lab.verifier_features.replay_alignment import draft_features
        z = torch.zeros(3, 32)
        z[0, 7] = 3
        v, lp = draft_features(z, [1, 3], 16)
        self.assertEqual(v["verifier_rank_draft"], [2, 1])
        self.assertEqual(v["top1"][:2], [7, 0])
        self.assertAlmostEqual(v["entropy"][1], __import__('math').log(32), places=5)
        self.assertLess(sum(__import__('math').exp(p) for p in v["topk_logp"][1]), .51)

    def test_hidden_union_and_old_logits_causal_indexing(self):
        import torch
        from suffix_lab.verifier_features.collect_old_features import collect
        e = self.events[0]
        v, tensors, check, _ = collect(self.target, e, audit_hooks=True)
        p, a = len(e["prefix"]), e["accepted"]
        with torch.inference_mode():
            out = self.model(torch.tensor([e["prefix"]+e["proposal"]]), output_hidden_states=True)
        # hidden_states[1] is output decoder block0; block1 output precedes final norm.
        self.assertTrue(torch.equal(tensors["old_layer_0"], out.hidden_states[1][0, p+a:p+64].half()))
        lp = out.logits[0].float().log_softmax(-1)
        self.assertAlmostEqual(tensors["M_off"][0, 0].item(), lp[p+a, e["source"][0]].item(), places=5)
        self.assertAlmostEqual(v["verifier_logp_draft"][a], lp[p+a-1, e["proposal"][a]].item(), places=5)
        self.assertTrue(check["old_top1_matches"])
        self.assertTrue(check["hook_audit"]["top1_equal"])

    def test_q0_and_probe_are_real_conditioned_model_outputs(self):
        import torch
        from suffix_lab.verifier_features.collect_corrected_features import collect
        from suffix_lab.runtime import ARSession
        e = self.events[0]
        v, tensors, _ = collect(self.target, e, micro_probe_steps=3)
        actual = ARSession(self.model).draft(e["corrected_prefix"], 3).tokens
        self.assertEqual(v["micro_probe"]["tokens"], actual)
        self.assertEqual(v["micro_probe"]["additional_forwards"], 2)
        with torch.inference_mode():
            out = self.model(torch.tensor([e["corrected_prefix"]]))
        expected = out.logits[0, -1].float().log_softmax(-1)[torch.tensor(e["source"])]
        self.assertTrue(torch.allclose(tensors["q0_logp_source"], expected))
        self.assertEqual(tensors["corrected_final_norm"].shape, (1, 32))

    def test_one_token_probe_reuses_corrected_logits(self):
        from suffix_lab.verifier_features.collect_corrected_features import collect
        v, _, _ = collect(self.target, self.events[0], micro_probe_steps=1)
        self.assertEqual(v["micro_probe"]["additional_forwards"], 0)

    def test_future_labels_cannot_change_online_features(self):
        import torch
        from suffix_lab.verifier_features.run_collection import collect_one
        e = copy.deepcopy(self.events[0])
        a, ta = collect_one(self.target, e, self.config)
        first = label(e, self.reference["tokens"], set())
        second = label(e, [40]*len(self.reference["tokens"]), set())
        self.assertNotEqual(first, second)
        b, tb = collect_one(self.target, e, self.config)
        self.assertEqual(a["event"], b["event"])
        self.assertTrue(all(torch.equal(ta[k], tb[k]) for k in ta))

    def test_corrupt_prefix_is_rejected(self):
        rounds = copy.deepcopy(self.rounds)
        rounds[1]["offset"] += 1
        with self.assertRaises(ValueError):
            reconstruct(self.prompts, rounds, set(), 6)

    def test_native_eos_tail_is_not_replayed(self):
        from suffix_lab.verifier_features.collect_old_features import collect
        e = copy.deepcopy(self.events[0])
        e["proposal"] = e["proposal"][:10]
        from suffix_lab.long60 import ParallelVerifier
        e["parallel"] = ParallelVerifier(self.model).score(e["prefix"], e["proposal"])
        e["source"] = e["proposal"][e["accepted"]+1:-1]  # final token treated as EOS
        v, t, _, _ = collect(self.target, e)
        self.assertEqual(len(v["top1"]), 11)
        self.assertEqual(t["M_off"].shape[1], len(e["source"])+1)

    def test_replay_mismatch_is_preserved(self):
        from suffix_lab.verifier_features.collect_old_features import collect
        e = copy.deepcopy(self.events[0])
        e["parallel"]["top1_with_bonus"][0] = (e["parallel"]["top1_with_bonus"][0]+1) % 64
        _, _, check, _ = collect(self.target, e)
        self.assertFalse(check["old_top1_matches"])
        self.assertEqual(check["old_top1_differences"][0]["position"], 0)

    def test_actual_collection_archive_restore_and_skip_completed(self):
        from suffix_lab.verifier_features.run_collection import execute
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"source"; source.mkdir()
            cp = self.checkpoint(source)
            config = {**self.config, "expected_input_identity": cp.identity}
            output = root/"out"
            r = execute(cp, output, "test-rev", config, limit=1, target_factory=lambda: self.target)
            self.assertEqual(r["status"], "partial_requested")
            first = (output/"shards/part-00000.zip").read_bytes()
            # Restore a portable ZIP to a different output, then finish the second event.
            other = root/"resumed"
            r = execute(cp, other, "test-rev", config, resume_root=output/"verifier_features_checkpoint.zip",
                        target_factory=lambda: self.target)
            self.assertEqual(r["status"], "complete")
            self.assertEqual(r["completed_events"], 2)
            self.assertEqual(first, (other/"shards/part-00000.zip").read_bytes())
            from suffix_lab.verifier_features.shards import load_event_tensors
            row = json.loads((other/"events.jsonl").read_text().splitlines()[0])
            self.assertIn("q0_logp_source", load_event_tensors(other, row))
            with zipfile.ZipFile(other/"verifier_features_checkpoint.zip") as z:
                self.assertIsNone(z.testzip())
                self.assertIn("offline_labels.parquet", z.namelist())
            r = execute(cp, other, "test-rev", config, target_factory=lambda: self.target)
            self.assertEqual(r["completed_events"], 2)
            with self.assertRaises(ValueError):
                execute(cp, other, "different-rev", config, target_factory=lambda: self.target)

    def test_corrupted_tensor_shard_is_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            s = Shards(tmp, {"a": 1})
            s.add({"event": {"id": "r"}}, {"x": torch.ones(1)})
            s.flush()
            part = Path(tmp)/"shards/part-00000.zip"
            with zipfile.ZipFile(part) as z:
                data = {n: z.read(n) for n in z.namelist()}
            name = next(n for n in data if n.endswith("safetensors"))
            data[name] = b"corrupted"
            with zipfile.ZipFile(part, "w") as z:
                for n, value in data.items():
                    z.writestr(n, value)
            with self.assertRaisesRegex(ValueError, "Corrupted"):
                Shards(tmp, {"a": 1})

    def test_smoke_failure_preserves_partial_archive(self):
        from suffix_lab.verifier_features.run_collection import execute, collect_one
        def bad(*args, **kwargs):
            r, tensors = collect_one(*args, **kwargs)
            r["check"]["old_top1_matches"] = False
            return r, tensors
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"source"; source.mkdir()
            cp = self.checkpoint(source)
            cfg = {**self.config, "expected_input_identity": cp.identity}
            out = Path(tmp)/"out"
            with patch("suffix_lab.verifier_features.run_collection.collect_one", side_effect=bad):
                with self.assertRaisesRegex(AssertionError, "Smoke replay mismatch"):
                    execute(cp, out, "test-rev", cfg, target_factory=lambda: self.target)
            self.assertTrue((out/"verifier_features_checkpoint.zip").exists())
            status = json.loads((out/"execution_status.json").read_text())
            self.assertEqual(status["completed_events"], 1)
            self.assertEqual(status["status"], "failed")

    def test_budget_checkpoint_then_resume(self):
        from suffix_lab.verifier_features.run_collection import execute
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/"source"; source.mkdir()
            cp = self.checkpoint(source)
            cfg = {**self.config, "expected_input_identity": cp.identity}
            out = Path(tmp)/"out"
            with patch("suffix_lab.verifier_features.run_collection.time.monotonic", side_effect=[0., 1., 2.]):
                r = execute(cp, out, "test-rev", cfg, budget_seconds=.5, target_factory=lambda: self.target)
            self.assertEqual(r["status"], "budget_checkpoint")
            self.assertEqual(r["completed_events"], 0)
            r = execute(cp, out, "test-rev", cfg, target_factory=lambda: self.target)
            self.assertEqual(r["status"], "complete")


if __name__ == "__main__":
    unittest.main()
