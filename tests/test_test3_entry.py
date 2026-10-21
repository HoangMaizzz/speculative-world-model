"""Standalone Test3 cell/notebook pin, source checksum and namespace contract."""
import hashlib
import io
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from suffix_lab.assets import ROOT


class Test3EntryTests(unittest.TestCase):
    def test_notebook_and_docs_have_one_matching_code_cell_and_valid_hash(self):
        b = runpy.run_path(str(ROOT/"scripts/build_test3_notebook.py"))
        source = b["cell"]()
        notebook = json.loads((ROOT/"notebooks/test3_oracle_all.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"], [source])
        self.assertIn(source, (ROOT/"docs/KAGGLE_TEST3.md").read_text(encoding="utf-8"))
        compile(source, "test3-cell", "exec")
        actual = hashlib.sha256((ROOT/"scripts/kaggle_test3_cell.py").read_text(encoding="utf-8").encode()).hexdigest()
        self.assertEqual(actual, b["LAUNCHER_SHA256"])

    def test_cell_is_independent_and_passes_test3_input_and_immutable_revision(self):
        b = runpy.run_path(str(ROOT/"scripts/build_test3_notebook.py"))
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
        payload = b"seen_revision=SUFFIX_TEST3_REVISION\nseen_input=SUFFIX_TEST3_INPUT\n"
        namespace = {}
        with patch.dict("sys.modules", {"torch": fake_torch}), patch("pathlib.Path.is_dir", return_value=True), \
             patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as fetch, \
             patch("hashlib.sha256", return_value=SimpleNamespace(hexdigest=lambda: b["LAUNCHER_SHA256"])):
            exec(compile(b["cell"](), "test3-cell", "exec"), namespace)
        self.assertEqual(namespace["scope"]["seen_revision"], b["REVISION"])
        self.assertEqual(Path(namespace["scope"]["seen_input"]), Path("/kaggle/input/datasets/ainzkhail/checkpt"))
        self.assertIn(b["REVISION"], fetch.call_args.args[0])

    def test_corrupt_launcher_never_executes(self):
        b = runpy.run_path(str(ROOT/"scripts/build_test3_notebook.py"))
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
        with patch.dict("sys.modules", {"torch": fake_torch}), patch("pathlib.Path.is_dir", return_value=True), \
             patch("urllib.request.urlopen", return_value=io.BytesIO(b"raise AssertionError('must not execute')")):
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                exec(compile(b["cell"](), "test3-cell", "exec"), {})


if __name__ == "__main__":
    unittest.main()
