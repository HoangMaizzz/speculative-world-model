import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_ar_resync_e2e import ARMS, CopyDrafter, compare_outputs, decode, export, restore
from suffix_lab.store import Store


class FakeCache:
    def __init__(self):
        self.tokens = []

    def get_seq_length(self):
        return len(self.tokens)

    def crop(self, length):
        del self.tokens[length:]


class FakeDrafter(CopyDrafter):
    def __init__(self, prompt):
        self.cache = FakeCache()
        self.next_token = self.forward(prompt)[-1]
        self.pending = []

    def forward(self, tokens):
        result = []
        for token in tokens:
            self.cache.tokens.append(token)
            result.append((sum(self.cache.tokens) + len(self.cache.tokens) * 3) % 97)
        return result


class FixedPolicy:
    old = [1]

    def __init__(self, bridge, tail):
        self.bridge = bridge
        self.tail = tail

    def select(self, proposal, length):
        if len(proposal) == self.bridge:
            return dict(source=4, tail=self.tail, match=4)
        return None


class ResyncTests(unittest.TestCase):
    def test_checked_crop_fallback_and_budget(self):
        for length in (8, 14):
            fresh, baseline = FakeDrafter([40]).propose(length, set())
            for mismatch in range(5):
                with self.subTest(length=length, mismatch=mismatch):
                    tail = fresh[4:8]
                    if mismatch < 4:
                        tail[mismatch] = (tail[mismatch] + 1) % 97
                    state = FakeDrafter([40])
                    proposal, count, action, *_ = state.propose_copy(
                        length, set(), FixedPolicy(4, tail), checked=True)
                    self.assertEqual(proposal, fresh)
                    self.assertEqual(state.cache.tokens, [40] + fresh[:-1])
                    self.assertEqual(action["kept"], mismatch)
                    expected = max(0, mismatch - 1) - int(length == 8 and mismatch == 4)
                    self.assertEqual(baseline - count, expected)

    def test_direct_constructs_kv_from_new_context_and_commits_correction(self):
        prefix = [40, 17]
        fresh, _ = FakeDrafter(prefix).propose(12, set())
        tail = [17, 64, 3, 55]
        for length in (8, 12):
            state = FakeDrafter(prefix)
            proposal, _, action, *_ = state.propose_copy(
                length, set(), FixedPolicy(4, tail), checked=False)
            expected = fresh[:4] + tail
            extra, _ = FakeDrafter(prefix + expected).propose(length - 8, set()) if length > 8 else ([], 0)
            self.assertEqual(proposal, expected + extra)
            self.assertEqual(state.cache.tokens, prefix + proposal[:-1])
            self.assertEqual(action["kept"], 4)
            decision = dict(committed_tokens=proposal[:5] + [31], accepted_draft=5)
            state.commit(decision)
            self.assertEqual(state.pending, [31])
            resumed, _ = state.propose(8, set())
            canonical, _ = FakeDrafter(prefix + decision["committed_tokens"]).propose(8, set())
            self.assertEqual(resumed, canonical)

    def test_checked_can_stop_on_eos_inside_failed_block(self):
        prompt = [40]
        fresh, _ = FakeDrafter(prompt).propose(14, set())
        eos = fresh[6]
        self.assertNotIn(eos, fresh[:6])
        state = FakeDrafter(prompt)
        tail = fresh[4:8]
        tail[2] = (tail[2] + 1) % 97
        out, _, action, *_ = state.propose_copy(14, {eos}, FixedPolicy(4, tail), checked=True)
        self.assertEqual(out, fresh[:7])
        self.assertEqual(action["kept"], 2)

    def test_three_arm_output_and_restore(self):
        from unittest.mock import patch

        with patch("benchmark_ar_resync_e2e.CopyDrafter", side_effect=lambda model, p: FakeDrafter(p)), \
             patch("benchmark_ar_resync_e2e.CachedTarget", side_effect=lambda model, p: FakeDrafter(p)):
            from benchmark_dllm_ar32 import CachedTarget
            FakeDrafter.verify = CachedTarget.verify
            arms = {arm: decode(None, None, [40], set(), 20, arm, {}, lambda m: None) for arm in ARMS}
        compare_outputs(arms)
        self.assertTrue(all(a["first_output_diff"] is None for a in arms.values()))
        self.assertTrue(all(a["first_proposal_diff_round"] is None for a in arms.values()))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            identity = {"test": 1}
            store = Store(output / "checkpoint.sqlite", identity)
            store.put("case", "math500:1", dict(id="math500:1", dataset="math500", arms=arms, order=ARMS))
            export(store, output, identity, 1, "complete")
            store.close()
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["metrics"]["all"]["checked_copy"]["exact_output"], 1)
            resumed = output / "resumed"
            resumed.mkdir()
            restore(output / "ar_resync_e2e_results.zip", resumed)
            reopened = Store(resumed / "checkpoint.sqlite", identity)
            self.assertEqual(len(reopened.rows("case")), 1)
            reopened.close()


if __name__ == "__main__":
    unittest.main()
