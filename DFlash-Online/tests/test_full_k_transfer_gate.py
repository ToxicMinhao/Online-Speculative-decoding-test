"""The transfer barrier cannot begin final serving without explicit clearance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.storage import write_json
from experiments.full_k_gsm8k import transfer_gate as G


class TransferGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        (self.root / "campaign_001").mkdir()
        self.marker = self.root / "TRANSFER_CLEARANCE.json"
        self.plan = {"binding": "scientific-binding", "execution_source": {
            "commit": "a" * 40, "dirty": False}}
        self.clearance = dict(schema="full_k_transfer_clearance_v1", status="cleared",
            run_root=str(self.root), binding=self.plan["binding"],
            execution_source=deepcopy(self.plan["execution_source"]),
            copy_process_stopped=True, transport_processes_stopped=True,
            no_transfers_until_campaign_exit=True)
        self.environment = patch.dict(G.os.environ, {G.ENVIRONMENT_VARIABLE: str(self.marker)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def receipt(self):
        return json.loads((self.root / "campaign_001/TRANSFER_CLEARANCE_WAIT.json").read_text())

    def test_missing_marker_waits_until_atomic_valid_clearance(self):
        def complete_copy(seconds):
            self.assertEqual(seconds, 10)
            self.assertEqual(self.receipt()["status"], "waiting")
            write_json(self.marker, self.clearance)

        with patch.object(G.time, "sleep", side_effect=complete_copy) as sleep:
            result = G.wait_for_transfer_clearance(self.root, self.plan)
        sleep.assert_called_once()
        self.assertEqual(result["status"], "cleared")
        self.assertEqual(result["model_work"], 0)
        self.assertGreaterEqual(result["wait_wall_s"], 0)
        self.assertEqual(result, self.receipt())
        self.assertEqual(len(result["marker_sha256"]), 64)

    def test_valid_marker_passes_without_polling(self):
        write_json(self.marker, self.clearance)
        with patch.object(G.time, "sleep") as sleep:
            result = G.wait_for_transfer_clearance(self.root, self.plan)
        sleep.assert_not_called()
        self.assertEqual(result["clearance"], self.clearance)

    def test_wrong_binding_source_or_quiescence_fails_closed(self):
        cases = {
            "binding": "another-run", "execution_source": {"commit": "b" * 40, "dirty": False},
            "run_root": str(self.root / "other"), "status": "copying", "schema": "other",
            "copy_process_stopped": False, "transport_processes_stopped": False,
            "no_transfers_until_campaign_exit": 1,
        }
        for key, value in cases.items():
            with self.subTest(field=key):
                write_json(self.marker, {**self.clearance, key: value})
                with patch.object(G.time, "sleep") as sleep:
                    with self.assertRaises(ValueError):
                        G.wait_for_transfer_clearance(self.root, self.plan)
                sleep.assert_not_called()
                self.assertEqual(self.receipt()["status"], "failed")

    def test_malformed_marker_fails_closed(self):
        self.marker.write_text("{unfinished")
        with self.assertRaises(json.JSONDecodeError):
            G.wait_for_transfer_clearance(self.root, self.plan)
        self.assertEqual(self.receipt()["status"], "failed")

    def test_outside_or_symlink_marker_is_rejected(self):
        for value in (str(self.root.parent / "outside.json"), "relative.json", ""):
            with self.subTest(path=value), patch.dict(G.os.environ, {G.ENVIRONMENT_VARIABLE: value}):
                with self.assertRaisesRegex(ValueError, "inside this run root"):
                    G.wait_for_transfer_clearance(self.root, self.plan)
        target = self.root / "actual.json"
        write_json(target, self.clearance)
        self.marker.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "inside this run root"):
            G.wait_for_transfer_clearance(self.root, self.plan)

    def test_deadline_interruption_is_recorded_without_clearance(self):
        with patch.object(G.time, "sleep", side_effect=SystemExit(143)):
            with self.assertRaises(SystemExit):
                G.wait_for_transfer_clearance(self.root, self.plan)
        self.assertEqual(self.receipt()["status"], "failed")
        self.assertIn("143", self.receipt()["error"])

    def test_unconfigured_campaign_does_not_wait(self):
        with patch.dict(G.os.environ, {}, clear=True), patch.object(G.time, "sleep") as sleep:
            result = G.wait_for_transfer_clearance(self.root, self.plan)
        sleep.assert_not_called()
        self.assertEqual(result, {"status": "not_configured", "wait_wall_s": 0.0})
        self.assertFalse((self.root / "campaign_001/TRANSFER_CLEARANCE_WAIT.json").exists())


if __name__ == "__main__":
    unittest.main()
