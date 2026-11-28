import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_online_corpus import ARMS, ARCHIVE, execute, make_banks, restore, write_bundle


class StubIndex:
    def __init__(self):
        self.table = {}
        self.docs = []

    def append(self, prefix, tokens):
        assert not prefix
        self.docs.append(tuple(tokens))


class OnlineCorpusTest(unittest.TestCase):
    def test_resume_reconstructs_each_arm_from_its_own_outputs(self):
        corpus = [dict(id=f"{dataset}:{i}", token_ids=[1000+i])
                  for dataset in ("math500", "gsm8k") for i in range(50)]
        manifest = dict(max_new=1, prompts=[
            dict(id="math500:100", dataset="math500", prompt_tokens=[1]),
            dict(id="gsm8k:100", dataset="gsm8k", prompt_tokens=[2]),
        ])
        outputs = {arm: [number] for arm, number in zip(ARMS, (10, 20, 30), strict=True)}

        def make_infer(banks, expected_before):
            def infer(prompt, bank):
                arm = next(arm for arm in ARMS if banks[arm] is bank)
                self.assertEqual(len(bank.docs), expected_before[arm])
                if prompt == [2] and arm != "static100":
                    self.assertEqual(bank.docs[-1], tuple(outputs[arm]))
                return dict(tokens=outputs[arm][:], rounds=[], counts=dict(
                    wall_ms=1.0, rounds=1, draft_forwards=1, accepted=1))
            return infer

        def paired(rows, _, __):
            return {"n": len(rows)}

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "first"
            (out / "rows").mkdir(parents=True)
            banks = make_banks(StubIndex, corpus)
            # Stop after the first checkpoint, as a new Kaggle session would.
            def interrupted(rows):
                for run in rows[0]["runs"].values():
                    run["coverage"] = {"output_positions": 1}
                write_bundle(out, manifest, rows, paired)
                raise InterruptedError("simulated session exit")

            with self.assertRaises(InterruptedError):
                execute(out, manifest, banks,
                        make_infer(banks, dict(zip(ARMS, (100, 0, 16), strict=True))), interrupted)
            resumed = Path(tmp) / "resumed"
            (resumed / "rows").mkdir(parents=True)
            restore(out / ARCHIVE, out, manifest)  # Audited ZIP also matches the existing raw checkpoint.
            restore(out / ARCHIVE, resumed, manifest)
            banks = make_banks(StubIndex, corpus)
            rows = execute(resumed, manifest, banks,
                           make_infer(banks, dict(zip(ARMS, (100, 1, 17), strict=True))), lambda _: None)
            self.assertEqual(len(rows), 2)
            self.assertEqual([len(banks[a].docs) for a in ARMS], [100, 2, 18])
            self.assertEqual([r["runs"]["online_empty"]["tokens"] for r in rows], [[20], [20]])


if __name__ == "__main__":
    unittest.main()
