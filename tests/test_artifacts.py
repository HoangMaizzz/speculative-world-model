import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import io
import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from suffix_lab.assets import ROOT
from suffix_lab.store import Store


class ArtifactTests(unittest.TestCase):
    def builder(self):
        spec = importlib.util.spec_from_file_location("build_notebook", ROOT/"scripts/build_notebook.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_quickstart_bootstrap_restores_missing_lab_once(self):
        module = self.builder()
        sources = ["setup_order = [1]\n", "setup_order.append(2)\nWORK=REPO=COMMIT='fixture'\n",
                   "setup_order.append(3)\nPY='python'\ndef lab(*args, **kwargs): return 123\n",
                   "raise AssertionError('must not execute experiments during setup')\n"]
        notebook = {"cells": [{"cell_type": "code", "source": [s]} for s in sources]}
        namespace = {}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(notebook).encode())) as fetch:
            exec(compile(module.SETUP_BOOTSTRAP, "bootstrap", "exec"), namespace)
            self.assertEqual(namespace["lab"]("prepare"), 123)
            self.assertEqual(namespace["setup_order"], [1, 2, 3])
            exec(compile(module.SETUP_BOOTSTRAP, "bootstrap", "exec"), namespace)
            fetch.assert_called_once()
            self.assertEqual(namespace["setup_order"], [1, 2, 3])

    def test_quick_notebook_matches_copyable_cells_and_compiles(self):
        module = self.builder()
        notebook = json.loads((ROOT/"notebooks/math_gsm8k_two_cells.ipynb").read_text())
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(cells, module.QUICK_CELLS)
        markdown = (ROOT/"KAGGLE_TWO_CELLS.md").read_text()
        for i, cell in enumerate(cells):
            compile(cell, f"quick-cell-{i}", "exec")
            self.assertIn(cell, markdown)
            self.assertIn("torch.cuda.device_count() < 2", cell)

    def test_each_standalone_file_has_exactly_one_code_cell(self):
        module = self.builder()
        for mode in ("smoke", "pilot"):
            notebook = json.loads((ROOT/f"notebooks/math_gsm8k_{mode}.ipynb").read_text())
            cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
            self.assertEqual(cells, [module.single_cell(mode)])
            self.assertNotIn('lab(', cells[0])
            self.assertNotIn('suffix_smoke/status.json', cells[0])

    def test_standalone_cells_execute_in_separate_empty_namespaces(self):
        module = self.builder()
        payload = b"observed_mode = SUFFIX_MODE\nobserved_revision = SUFFIX_CODE_REVISION\n"
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
        for mode in ("smoke", "pilot"):
            namespace = {}
            with patch.dict("sys.modules", {"torch": fake_torch}), \
                 patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as download, \
                 patch("hashlib.sha256", return_value=SimpleNamespace(hexdigest=lambda: module.SINGLE_CELL_SHA256)):
                exec(compile(module.single_cell(mode), mode, "exec"), namespace)
            self.assertEqual(namespace["_suffix_scope"]["observed_mode"], mode)
            self.assertEqual(namespace["_suffix_scope"]["observed_revision"], module.SINGLE_CELL_REVISION)
            self.assertIn(module.SINGLE_CELL_LAUNCHER_REVISION, download.call_args.args[0])

    def test_standalone_checksum_matches_the_reviewed_launcher(self):
        module = self.builder()
        raw = (ROOT/"scripts/kaggle_single_cell.py").read_text(encoding="utf-8").encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), module.SINGLE_CELL_SHA256)

    def test_notebook_and_copyable_cells_match_and_compile(self):
        spec = importlib.util.spec_from_file_location("build_notebook", ROOT/"scripts/build_notebook.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        notebook = json.loads((ROOT/"notebooks/math_gsm8k.ipynb").read_text())
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(code, [source.lstrip() for source in module.CELLS])
        markdown = (ROOT/"KAGGLE_CELLS.md").read_text()
        for i, source in enumerate(code):
            compile(source, f"kaggle-cell-{i+1}", "exec")
            self.assertIn(source, markdown)
        self.assertIn("torch.cuda.device_count() >= 2", code[0])

    def test_current_pilot_is_only_math_and_gsm8k(self):
        config = json.loads((ROOT/"configs/math_gsm8k.json").read_text())
        self.assertEqual(config["datasets"], ["math500", "gsm8k"])
        self.assertEqual(config["prompts_per_dataset"], 100)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["target_precision"], "fp16")
        self.assertEqual(config["target_gpu0_layers"], 18)
        self.assertFalse(config["allow_nf4_fallback"])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional analysis dependencies absent")
    def test_analysis_first_import_ignores_unavailable_notebook_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, MPLBACKEND="module://suffix_missing_inline_fixture")
            code = (
                "import sys; from pathlib import Path; from suffix_lab.analysis import analyze; "
                "report=analyze(sys.argv[1],20); "
                "assert report['test1_proposals']==0; "
                "assert len(list((Path(sys.argv[1])/'analysis').glob('*.png')))==10"
            )
            result = subprocess.run([sys.executable, "-c", code, tmp], cwd=ROOT, env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Optional analysis dependencies absent")
    def test_empty_partial_analysis_writes_all_ten_plots(self):
        from suffix_lab.analysis import analyze
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/"run.sqlite")
            store.close()
            report = analyze(tmp, 20)
            self.assertEqual(report["test1_proposals"], 0)
            self.assertFalse(report["status"]["test1"])
            self.assertEqual(len(list((Path(tmp)/"analysis").glob("*.png"))), 10)

    def test_every_python_source_compiles(self):
        for folder in ("src", "tests", "scripts"):
            for path in (ROOT/folder).rglob("*.py"):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
