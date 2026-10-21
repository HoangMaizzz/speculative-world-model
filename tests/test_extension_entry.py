"""The published copyable cell and notebook need no prior notebook globals."""
import hashlib
import io
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from suffix_lab.assets import ROOT


class ExtensionEntryTests(unittest.TestCase):
    def builder(self):
        return runpy.run_path(str(ROOT/"scripts/build_extension_notebook.py"))

    def test_one_cell_matches_notebook_docs_and_launcher_hash(self):
        module = self.builder()
        source = module["cell"]()
        notebook = json.loads((ROOT/"notebooks/mini_extension.ipynb").read_text(encoding="utf-8"))
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(cells, [source])
        self.assertIn(source, (ROOT/"docs/KAGGLE_MINI_EXTENSION.md").read_text(encoding="utf-8"))
        compile(source, "mini-extension-cell", "exec")
        raw = (ROOT/"scripts/kaggle_extension_cell.py").read_text(encoding="utf-8").encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), module["LAUNCHER_SHA256"])

    def test_standalone_cell_executes_with_correct_input_and_revision(self):
        module = self.builder()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
        payload = b"observed_revision=SUFFIX_EXTENSION_REVISION\nobserved_input=SUFFIX_EXTENSION_INPUT\n"
        namespace = {}
        with patch.dict("sys.modules", {"torch": fake_torch}), patch("pathlib.Path.is_dir", return_value=True), \
             patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as fetch, \
             patch("hashlib.sha256", return_value=SimpleNamespace(hexdigest=lambda: module["LAUNCHER_SHA256"])):
            exec(compile(module["cell"](), "cell", "exec"), namespace)
        self.assertEqual(namespace["scope"]["observed_revision"], module["REVISION"])
        self.assertEqual(Path(namespace["scope"]["observed_input"]), Path("/kaggle/input/datasets/ainzkhail/checkpt"))
        self.assertIn(module["REVISION"], fetch.call_args.args[0])

    def test_checksum_mismatch_refuses_execution(self):
        module = self.builder()
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
        with patch.dict("sys.modules", {"torch": fake_torch}), patch("pathlib.Path.is_dir", return_value=True), \
             patch("urllib.request.urlopen", return_value=io.BytesIO(b"raise AssertionError('unsafe')")):
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                exec(compile(module["cell"](), "cell", "exec"), {})


if __name__ == "__main__":
    unittest.main()
