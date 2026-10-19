"""Exercise the complete offline branch/checkpoint pipeline with a known teacher."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import time
import unittest
import importlib.util
import weakref
from unittest.mock import patch
from suffix_lab.assets import ROOT, AR, DLLM
from suffix_lab.core import Draft
from suffix_lab.experiment import Runner
from suffix_lab.store import Store


class FakeDrafter:
    def __init__(self, model, config=None):
        self.model = model

    def draft(self, prefix, length, anchors=None):
        anchors = anchors or {}
        tokens = [10+len(prefix)-2+i for i in range(length)]
        if anchors:
            for i, token in anchors.items():
                tokens[i] = token
        elif length == 32:  # Test 1 has controlled first rejections.
            tokens[5 if self.model == AR else 9] = 0
        n = length-len(anchors)
        return Draft(tokens, [.5]*length, list(range(length)), stats={"unmask_steps": n,
            "total_forwards": n+1, "latency_ms": float(n+2), "mutable_latency_ms": float(n)})


class FakeVerifier:
    def score(self, prefix, proposal, top_k=8):
        n = len(proposal)
        return {"top1": [10+len(prefix)-2+i for i in range(n)], "probability": [.6]*n}


class PipelineTests(unittest.TestCase):
    def test_paired_proposals_repair_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Runner.__new__(Runner)
            runner.config = json.loads((ROOT/"configs/math_gsm8k.json").read_text())
            runner.config.update(test2_events=2, random_repeats=1)
            runner.output = Path(tmp)
            runner.manifest = {"paths": {AR: AR, DLLM: DLLM}}
            runner.deadline = time.monotonic()+100
            runner.completed_units = 0
            runner.verifier = FakeVerifier()
            runner.store = Store(Path(tmp)/"run.sqlite", {"fixture": True})
            runner.store.put("reference", "gsm8k:0", {"id": "gsm8k:0", "dataset": "gsm8k",
                "prompt_ids": [1, 2], "tokens": list(range(10, 90)), "offsets": [0, 16]})
            runner.store.put("meta", "references_complete", True)
            loaded, weak_models = [], []
            class ModelFixture:
                def __init__(self, name):
                    self.name = name
                def __eq__(self, other):
                    return self.name == other
            def load_one(name, device):
                self.assertTrue(all(ref() is None for ref in weak_models), "Previous drafter still resident")
                model = ModelFixture(name)
                weak_models.append(weakref.ref(model))
                loaded.append(name)
                return model
            with patch("suffix_lab.runtime.load_model", side_effect=load_one), \
                 patch("suffix_lab.runtime.ARSession", FakeDrafter), \
                 patch("suffix_lab.runtime.DLLMDrafter", FakeDrafter), \
                 patch("suffix_lab.runtime.release"), patch("suffix_lab.runtime.memory", return_value=[]):
                runner.test1()
                self.assertEqual(len(runner.store.rows("test1")), 4)
                runner.test2()
                self.assertEqual(loaded[:3], [DLLM, AR, DLLM])
                self.assertTrue(all(ref() is None for ref in weak_models))
                old = runner.store.rows("test2")
                self.assertEqual(len(runner.store.rows("event_complete")), 2)
                for row in old:
                    self.assertEqual(row["horizon"], 32-row["first_rejection"])
                    self.assertEqual(row["stats"]["unmask_steps"], row["horizon"]-len(row["keep"]))
                runner.test1()
                runner.test2()
                self.assertEqual(old, runner.store.rows("test2"))
            runner.store.close()
            if importlib.util.find_spec("matplotlib"):
                from suffix_lab.analysis import analyze
                report = analyze(tmp, 20)
                self.assertEqual(report["paired_states"], 2)
                self.assertEqual(report["complete_test2_events"], 2)
                self.assertEqual(len(list((Path(tmp)/"analysis").glob("*.png"))), 10)


if __name__ == "__main__":
    unittest.main()
