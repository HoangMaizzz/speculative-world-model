"""Actual child-process output relay, heartbeat and nonzero-exit evidence."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from suffix_lab.assets import ROOT


class ExtensionStreamTests(unittest.TestCase):
    def test_stdout_stderr_and_failure_code_reach_notebook_and_log(self):
        run = runpy.run_path(str(ROOT/"scripts/kaggle_extension_cell.py"))["run_streamed"]
        source = "import sys; print('child progress',flush=True); print('child error',file=sys.stderr,flush=True); sys.exit(7)"
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as captured:
            path = Path(tmp)/"run.log"
            code = run([sys.executable, "-u", "-c", source], cwd=ROOT, env=os.environ.copy(), log_path=path)
            text = captured.getvalue()
            self.assertEqual(code, 7)
            self.assertIn("child progress", text)
            self.assertIn("child error", text)
            self.assertIn("child progress", path.read_text(encoding="utf-8"))
            self.assertIn("child error", path.read_text(encoding="utf-8"))

    def test_silent_child_emits_heartbeat_before_completion(self):
        run = runpy.run_path(str(ROOT/"scripts/kaggle_extension_cell.py"))["run_streamed"]
        source = "import time; time.sleep(0.15); print('finished',flush=True)"
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as captured:
            code = run([sys.executable, "-u", "-c", source], cwd=ROOT, env=os.environ.copy(),
                       log_path=Path(tmp)/"run.log", heartbeat_seconds=0.02)
            text = captured.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("[launcher] Child running", text)
            self.assertIn("finished", text)
            self.assertLess(text.index("Child running"), text.index("finished"))


if __name__ == "__main__":
    unittest.main()
