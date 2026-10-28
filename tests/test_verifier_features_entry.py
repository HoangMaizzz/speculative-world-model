"""Standalone notebook and launcher contracts without network or model loading."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FeatureEntryTests(unittest.TestCase):
    def test_generated_notebook_has_one_self_contained_pinned_cell(self):
        spec = importlib.util.spec_from_file_location("feature_builder", ROOT/"scripts/build_verifier_features_notebook.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            cell = builder.build("a"*40, tmp)
            nb = json.loads((Path(tmp)/"notebooks/verifier_features_full.ipynb").read_text())
            self.assertEqual(len(nb["cells"]), 1)
            self.assertEqual("".join(nb["cells"][0]["source"]), cell)
            ast.parse(cell)
            self.assertIn('"FEATURE_LIMIT": None', cell)
            self.assertIn('"FEATURE_MICRO_PROBE_STEPS": 0', cell)
            self.assertIn('RESUME_ROOT = None', cell)
            self.assertIn('EXPECTED_SHA256', cell)
            self.assertIn('REVISION = "'+'a'*40+'"', cell)

    def test_launcher_uses_pinned_installer_and_full_collector(self):
        text = (ROOT/"scripts/kaggle_verifier_features_cell.py").read_text()
        ast.parse(text)
        self.assertIn('"--detach", revision', text)
        self.assertIn('suffix_lab.verifier_features.run_collection', text)
        self.assertIn('ensure_environment', text)
        self.assertIn('test_verifier_features*.py', text)
        self.assertIn('run_streamed', text)
        self.assertNotIn('lab("prepare"', text)


if __name__ == "__main__":
    unittest.main()
