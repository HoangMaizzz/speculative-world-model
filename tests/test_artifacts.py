import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from suffix_lab.assets import ROOT
from suffix_lab.store import Store


class ArtifactTests(unittest.TestCase):
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
