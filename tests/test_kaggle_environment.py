"""Regression cases for missing ensurepip and a half-created virtualenv."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from suffix_lab.assets import ROOT


def launcher_module():
    spec = importlib.util.spec_from_file_location("kaggle_launcher_test", ROOT/"scripts/kaggle_single_cell.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EnvironmentTests(unittest.TestCase):
    def test_child_environment_preserves_gpu_settings_without_notebook_hooks(self):
        module = launcher_module()
        source = {"PYTHONPATH": "/notebook/hooks", "PYTHONHOME": "/notebook/python",
                  "MPLBACKEND": "module://matplotlib_inline.backend_inline",
                  "CUDA_VISIBLE_DEVICES": "0,1", "HF_HOME": "/cache", "PATH": "fixture"}
        env = module.child_environment(source)
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertEqual(env["MPLBACKEND"], "Agg")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(env["HF_HOME"], "/cache")
        self.assertEqual(env["PATH"], "fixture")
        self.assertEqual(source["PYTHONPATH"], "/notebook/hooks")

    def test_setup_child_does_not_load_injected_sitecustomize(self):
        import sys
        module = launcher_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"sitecustomize.py").write_text("raise RuntimeError('fixture leaked startup hook')")
            log = root/"setup.log"
            with patch.dict(os.environ, {"PYTHONPATH": tmp, "PYTHONHOME": str(root/"invalid-home"),
                    "MPLBACKEND": "module://matplotlib_inline.backend_inline"}):
                module.setup_command([sys.executable, "-c",
                    "import os; assert 'PYTHONPATH' not in os.environ; "
                    "assert 'PYTHONHOME' not in os.environ; "
                    "assert os.environ['MPLBACKEND']=='Agg'; print('clean child startup')"], log)
            output = log.read_text()
            self.assertIn("clean child startup", output)
            self.assertNotIn("fixture leaked startup hook", output)

    def test_new_environment_does_not_invoke_ensurepip(self):
        module = launcher_module()
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp)/"venv"
            calls = []
            def run_setup(args, log):
                calls.append(args)
            answers = iter([0, 1, 0])  # interpreter works, no venv pip, host pip works
            def probe(args, **kwargs):
                return SimpleNamespace(returncode=next(answers))
            python = module.ensure_environment(venv, Path(tmp)/"setup.log", run_setup, probe, "base-python")
            self.assertIn("--without-pip", calls[0])
            self.assertEqual(calls[1][1:5], ["-m", "pip", "--python", str(python)])
            self.assertIn("pip==25.2", calls[1])
            self.assertEqual(calls[-1][-3:], ["-m", "pip", "--version"])

    def test_partial_environment_with_python_but_no_pip_is_repaired(self):
        module = launcher_module()
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp)/"venv"
            python = venv/("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.touch()
            (venv/"pyvenv.cfg").write_text("home = fixture")
            calls = []
            answers = iter([0, 1, 0])
            result = module.ensure_environment(venv, Path(tmp)/"setup.log",
                lambda args, log: calls.append(args),
                lambda args, **kwargs: SimpleNamespace(returncode=next(answers)), "base-python")
            self.assertEqual(result, python)
            self.assertTrue(all("venv" not in args for args in calls))
            self.assertIn("--python", calls[0])
            self.assertTrue((venv/"pyvenv.cfg").exists())

    def test_healthy_environment_is_reused(self):
        module = launcher_module()
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp)/"venv"
            python = venv/("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.touch()
            (venv/"pyvenv.cfg").write_text("home = fixture")
            calls = []
            module.ensure_environment(venv, Path(tmp)/"setup.log", lambda args, log: calls.append(args),
                                      lambda args, **kwargs: SimpleNamespace(returncode=0), "base-python")
            self.assertEqual(calls, [[str(python), "-m", "pip", "--version"]])

    def test_setup_error_preserves_actual_output_in_log(self):
        import sys
        module = launcher_module()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/"setup.log"
            with self.assertRaisesRegex(RuntimeError, "Full log"):
                module.setup_command([sys.executable, "-c", "print('fixture ensurepip unavailable'); raise SystemExit(1)"], log)
            self.assertIn("fixture ensurepip unavailable", log.read_text())


if __name__ == "__main__":
    unittest.main()
