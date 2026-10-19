"""Event boundaries are tested with simulated copies, not actual GPU timings."""
from contextlib import nullcontext
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from suffix_lab.timing import StageTimer


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch absent")
class ComputeTimingTests(unittest.TestCase):
    def fixture(self, log):
        import torch
        class Stage:
            def __init__(self, device="cuda:1"):
                self.device = torch.device(device)
            def parameters(self):
                return iter([SimpleNamespace(device=self.device)])
            def buffers(self):
                return iter([])
            def _old_forward(self, value):
                log.append("compute")
                return value+1
            def forward(self, value):
                log.append("copy_in")
                result = self._old_forward(value)
                log.append("copy_out")
                return result
        stage = Stage()
        model = SimpleNamespace(named_modules=lambda: [("model.layers.18", stage)])
        counter = iter(range(100))
        class Event:
            def __init__(self, **kwargs):
                self.number = next(counter)
            def record(self, stream):
                log.append("start" if self.number % 2 == 0 else "end")
            def elapsed_time(self, end):
                return 2.5
        return model, stage, Event

    def test_nonoverlapping_stage_times_are_summed_across_devices(self):
        log = []
        _, stage1, event = self.fixture(log)
        stage0 = type(stage1)("cuda:0")
        model = SimpleNamespace(named_modules=lambda: [("model.layers.0", stage0), ("model.layers.18", stage1)])
        timer = StageTimer(model)
        with patch("torch.cuda.device", side_effect=lambda *a: nullcontext()), \
             patch("torch.cuda.current_stream", return_value="fixture"), \
             patch("torch.cuda.Event", event):
            with timer.measure():
                stage1.forward(stage0.forward(0))
        report = timer.report(20.0)
        self.assertEqual(report["latency_ms"], 5.0)
        self.assertEqual(report["compute_by_device_ms"], {"cuda:0": 2.5, "cuda:1": 2.5})

    def test_copy_hooks_are_outside_compute_event_boundaries(self):
        log = []
        model, stage, event = self.fixture(log)
        timer = StageTimer(model)
        with patch("torch.cuda.device", side_effect=lambda *a: nullcontext()), \
             patch("torch.cuda.current_stream", return_value="fixture"), \
             patch("torch.cuda.Event", event):
            with timer.measure():
                self.assertEqual(stage.forward(3), 4)
        self.assertEqual(log, ["copy_in", "start", "compute", "end", "copy_out"])
        result = timer.report(12.0)
        self.assertEqual(result["latency_ms"], 2.5)
        self.assertEqual(result["wall_latency_ms"], 12.0)
        self.assertEqual(result["compute_by_device_ms"], {"cuda:1": 2.5})
        self.assertEqual(timer.records, [])
        self.assertFalse(timer.active)

    def test_inactive_timer_adds_no_events(self):
        log = []
        model, stage, _ = self.fixture(log)
        timer = StageTimer(model)
        stage.forward(3)
        self.assertEqual(log, ["copy_in", "compute", "copy_out"])
        self.assertEqual(timer.records, [])

    def test_missing_events_never_fall_back_to_wall_time(self):
        log = []
        model, _, _ = self.fixture(log)
        timer = StageTimer(model)
        with self.assertRaisesRegex(RuntimeError, "No CUDA events"):
            timer.report(9.0)
        with self.assertRaisesRegex(RuntimeError, "Nested"):
            with timer.measure():
                with timer.measure():
                    pass
        self.assertFalse(timer.active)


if __name__ == "__main__":
    unittest.main()
