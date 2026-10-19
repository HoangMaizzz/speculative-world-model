"""Fresh-session and Save Version regressions without downloading or using GPUs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from suffix_lab.assets import ROOT


def runner_module():
    spec = importlib.util.spec_from_file_location("kaggle_run_test", ROOT/"scripts/kaggle_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StandaloneTests(unittest.TestCase):
    def fake_run(self, assets, output, calls, fail=None, partial=False):
        def invoke(args, **kwargs):
            command = args[4] if "suffix_lab.cli" in args else "unit_checks"
            calls.append((command, list(args)))
            if command == "prepare":
                assets.mkdir(parents=True, exist_ok=True)
                (assets/"manifest.json").write_text(json.dumps({"paths": {
                    "Efficient-Large-Model/Fast_dLLM_v2_1.5B": "fixture_source"}}))
            if command == "run":
                (output/"run.sqlite").touch()
            code = 1 if command == fail else (75 if command == "run" and partial else 0)
            return SimpleNamespace(returncode=code)
        return invoke

    def test_pilot_fresh_session_needs_no_smoke_globals_or_files(self):
        module = runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            assets, output = Path(tmp)/"assets", Path(tmp)/"pilot"
            calls = []
            result = module.execute("pilot", "a"*40, assets, output, Path(tmp)/"input",
                                    self.fake_run(assets, output, calls))
            self.assertEqual(result["status"], "complete")
            self.assertEqual([c for c, _ in calls], ["prepare", "unit_checks", "preflight", "run", "analyze", "pack"])
            self.assertTrue(all("--smoke" not in args for _, args in calls))
            self.assertFalse((Path(tmp)/"suffix_smoke").exists())

    def test_smoke_has_its_own_flag_and_output(self):
        module = runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            assets, output = Path(tmp)/"assets", Path(tmp)/"smoke"
            calls = []
            module.execute("smoke", "a"*40, assets, output, Path(tmp)/"input",
                           self.fake_run(assets, output, calls))
            for command, args in calls:
                if command != "unit_checks":
                    self.assertIn("--smoke", args)

    def test_partial_is_exported_without_claiming_complete(self):
        module = runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            assets, output = Path(tmp)/"assets", Path(tmp)/"pilot"
            calls = []
            result = module.execute("pilot", "a"*40, assets, output, Path(tmp)/"input",
                                    self.fake_run(assets, output, calls, partial=True))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["experiment_exit_code"], 75)
            self.assertEqual(calls[-1][0], "pack")

    def test_failed_preflight_still_packs_and_does_not_run(self):
        module = runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            assets, output = Path(tmp)/"assets", Path(tmp)/"pilot"
            calls = []
            with self.assertRaises(subprocess.CalledProcessError):
                module.execute("pilot", "a"*40, assets, output, Path(tmp)/"input",
                               self.fake_run(assets, output, calls, fail="preflight"))
            self.assertNotIn("run", [c for c, _ in calls])
            self.assertEqual(calls[-1][0], "pack")
            self.assertEqual(json.loads((output/"execution_status.json").read_text())["status"], "failed")

    def test_checkpoint_selection_matches_config_and_code(self):
        module = runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def checkpoint(folder, config, revision):
                path = root/folder/"suffix_checkpoint.zip"
                path.parent.mkdir()
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr("run-manifest.json", json.dumps({"config": config, "environment": {"commit": revision}}))
                return path
            checkpoint("smoke", {"n": 5}, "a"*40)
            checkpoint("old", {"n": 100}, "b"*40)
            expected = checkpoint("pilot", {"n": 100}, "a"*40)
            self.assertEqual(module.matching_checkpoint(root, {"n": 100}, "a"*40), expected)
            checkpoint("duplicate", {"n": 100}, "a"*40)
            with self.assertRaises(ValueError):
                module.matching_checkpoint(root, {"n": 100}, "a"*40)


if __name__ == "__main__":
    unittest.main()
