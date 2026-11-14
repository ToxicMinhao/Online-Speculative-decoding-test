import json
from pathlib import Path
import tempfile
import time
import unittest

from experiments.threatened_repair_replay.work import Ledger, WorkLimit, read_ledger


class WorkTests(unittest.TestCase):
    def test_scopes_restore_but_physical_work_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = Ledger(Path(folder)/"work.jsonl", limits={"optimizer_step":2})
            for state in ("U", "T"):
                with ledger.scope(category="disposable", state=state, window=8):
                    with ledger.operation("optimizer_step"):
                        ledger.tick("layer_forwards", 5)
                self.assertEqual(ledger.context, {})
            with self.assertRaises(WorkLimit):
                with ledger.operation("optimizer_step"):
                    self.fail("Work ceiling admitted another optimizer application")
            ledger.close()
            receipt=read_ledger(ledger.path)
            self.assertEqual(receipt["completed"], {"optimizer_step":2})
            self.assertEqual(receipt["physical"], {"layer_forwards":10})

    def test_failed_response_is_saved_before_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=Ledger(Path(folder)/"work.jsonl")
            with self.assertRaisesRegex(ValueError,"fixture failure"):
                with ledger.operation("response_forward", category="qualification"):
                    ledger.tick("vocabulary_head_calls",3)
                    raise ValueError("fixture failure")
            receipt=read_ledger(ledger.path)
            self.assertEqual(receipt["failed"], {"response_forward":1})
            self.assertEqual(receipt["physical"], {"vocabulary_head_calls":3})
            ledger.close()

    def test_crashed_start_remains_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"work.jsonl"
            p.write_text(json.dumps(dict(phase="started",serial=1,kind="response_backward"))+"\n")
            self.assertEqual(read_ledger(p)["unknown"], {"response_backward":1})

    def test_deadline_admits_no_more_work(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger=Ledger(Path(folder)/"work.jsonl",deadline=time.monotonic()-1)
            with self.assertRaises(WorkLimit):
                with ledger.operation("optimizer_step"):
                    self.fail("Expired budget")
            self.assertEqual(dict(ledger.attempts),{})
            ledger.close()


if __name__ == "__main__":
    unittest.main()
