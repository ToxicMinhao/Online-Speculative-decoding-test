"""Check storage overlap and irreversible-retirement prerequisites."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.osd_persistent.retention import GIB, admit_storage, checked_retire
from online_feedback.storage import write_json


class PersistentRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "campaign"
        self.root.mkdir()
        self.draft = self.base / "released"
        self.draft.mkdir()
        (self.draft / "model.safetensors").write_bytes(b"w" * 10)

    def blob(self, relative, size=10):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    def fixture(self):
        self.event = self.root / "events/event_002"
        self.event.mkdir(parents=True)
        self.state = self.blob("trainer/training_state.pt", 30)
        self.current = self.blob("trainer/events/event_002/export/model.safetensors").parent
        self.previous = self.blob("trainer/events/event_001/export/model.safetensors").parent
        write_json(self.previous / "config.json", {"retained": True})
        write_json(self.current / "config.json", {"retained": True})
        self.capture = self.blob("native/attempt_001/request_000001.pt")
        self.feature = self.blob("events/event_002/loop/features/response_001.pt")
        self.terminal = self.blob("events/event_002/loop/features/response_001.terminal.pt")
        capture = {"prompt_ids": [1], "response_ids": [3, 4], "feature_path": str(self.capture)}
        terminal = {"tensor_path": str(self.terminal), "input_ids": [1, 3, 4], "positions": [2]}
        self.prepared = {"record_id": "response_001", "feature_path": str(self.feature),
                         "terminal_feature": terminal, "status": "prepared"}
        self.records = [{"record_id": "response_001", "prompt_ids": [1], "response_ids": [3, 4],
                         "checkpoint_id": str(self.previous), "metadata": {"capture": capture}}]
        self.training = {"event_id": "event_002", "status": "completed",
                         "checkpoint_path": str(self.current), "full_state_path": str(self.state)}
        self.receipt = {"event_id": "event_002", "preparation": [self.prepared],
                        "training": self.training, "activation": {"status": "passed"}}
        self.buffer_path = self.event / "loop/feedback_buffer/state.json"
        self.buffer = {"records": self.records, "events": [{"event_id": "event_002",
                       "status": "committed", "record_ids": ["response_001"],
                       "checkpoint_id": str(self.current), "receipt": self.receipt}]}
        self.marker = {"status": "passed", "all_equal": True, "event_id": "event_002",
                       "checkpoint_path": str(self.current)}
        write_json(self.capture.with_suffix(".json"), capture)
        write_json(self.feature.with_suffix(".json"), self.prepared)
        write_json(self.terminal.with_suffix(".json"), terminal)
        write_json(self.previous.parent / "RESULT.json", {
            "status": "completed", "checkpoint_path": str(self.previous)})
        self.persist_fixture()

    def persist_fixture(self):
        write_json(self.current.parent / "RESULT.json", self.training)
        write_json(self.event / "loop/events/event_002.json", self.receipt)
        write_json(self.buffer_path, self.buffer)
        write_json(self.event / "PUBLICATION_CHECK.json", self.marker)

    def retire(self):
        return checked_retire(self.event, self.records, self.receipt, self.previous, self.root)

    def assert_nothing_retired(self):
        for path in (self.state, self.current / "model.safetensors", self.previous / "model.safetensors",
                     self.capture, self.feature, self.terminal):
            self.assertTrue(path.is_file(), str(path))
        self.assertFalse((self.event / "RETIREMENT_PLAN.json").exists())

    def test_budget_accounts_for_peak_and_deducts_existing_overlap_only_once(self):
        with patch("experiments.osd_persistent.retention.shutil.disk_usage",
                   return_value=SimpleNamespace(free=10**15)):
            fresh = admit_storage(self.root, self.draft, max_sequence_length=2048)
            state = self.blob("trainer/training_state.pt", 30)
            export = self.blob("trainer/events/event_001/export/model.safetensors").parent
            later = admit_storage(self.root, self.draft, max_sequence_length=2048,
                                  current_state=state, current_export=export)
        features = 2 * 128 * 2048 * 4096 * 6 * 2 + 128 * 2048 * 12
        self.assertEqual(fresh["feature_tensor_bound_bytes"], features)
        self.assertEqual(fresh["tensor_peak_bytes"], features + 2 * 30 + 2 * 10)
        self.assertEqual(fresh["required_additional_and_reserve_bytes"] -
                         later["required_additional_and_reserve_bytes"], 40)
        self.assertEqual(later["reserve_bytes"], 8 * GIB)
        self.assertEqual(later["compact_cumulative_allowance_bytes"], 8 * GIB)

    def test_budget_fails_below_reserve_and_records_admission(self):
        receipt = self.root / "admission.json"
        with patch("experiments.osd_persistent.retention.shutil.disk_usage",
                   return_value=SimpleNamespace(free=8 * GIB)):
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                admit_storage(self.root, self.draft, max_sequence_length=2048, receipt_path=receipt)
        self.assertFalse(json.loads(receipt.read_text())["admitted"])

    def test_compact_evidence_gets_its_own_cumulative_bound(self):
        (self.root / "evidence.log").write_text("twelve bytes")
        with patch("experiments.osd_persistent.retention.shutil.disk_usage",
                   return_value=SimpleNamespace(free=10**15)):
            with self.assertRaisesRegex(ValueError, "cumulative"):
                admit_storage(self.root, self.draft, max_sequence_length=10, compact_allowance_bytes=5)

    def test_only_obsolete_tensors_are_removed_after_plan(self):
        self.fixture()
        original_unlink = Path.unlink
        def observe_unlink(path, *args, **kwargs):
            if path in {self.capture, self.feature, self.terminal, self.previous / "model.safetensors"}:
                plan = json.loads((self.event / "RETIREMENT_PLAN.json").read_text())
                self.assertEqual(plan["status"], "planned")
                self.assertTrue(plan["compact_evidence"])
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", observe_unlink):
            result = self.retire()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["removed_bytes"], 40)
        for path in (self.state, self.current / "model.safetensors", self.previous / "config.json",
                     self.capture.with_suffix(".json"), self.feature.with_suffix(".json"), self.buffer_path):
            self.assertTrue(path.is_file(), str(path))
        self.assertFalse(self.capture.exists())
        self.assertFalse((self.previous / "model.safetensors").exists())
        with self.assertRaisesRegex(ValueError, "already attempted"):
            self.retire()

    def test_failed_buffer_blocks_every_deletion(self):
        self.fixture()
        self.buffer["events"][0]["status"] = "failed"
        self.persist_fixture()
        with self.assertRaisesRegex(ValueError, "committed buffer"):
            self.retire()
        self.assert_nothing_retired()

    def test_publication_marker_must_bind_current_checkpoint(self):
        self.fixture()
        self.marker["checkpoint_path"] = str(self.previous)
        self.persist_fixture()
        with self.assertRaisesRegex(ValueError, "publication check"):
            self.retire()
        self.assert_nothing_retired()

    def test_current_export_is_protected_even_when_passed_as_previous(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, "Current export"):
            checked_retire(self.event, self.records, self.training, self.current, self.root)
        self.assert_nothing_retired()

    def test_outside_previous_export_is_protected(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError, "escapes campaign"):
            checked_retire(self.event, self.records, self.training, self.draft, self.root)
        self.assert_nothing_retired()
        self.assertTrue((self.draft / "model.safetensors").is_file())

    def test_unrelated_run_checkpoint_is_not_retired(self):
        self.fixture()
        unrelated = self.blob("trainer/events/unrelated/export/model.safetensors").parent
        write_json(unrelated.parent / "RESULT.json", {
            "status": "completed", "checkpoint_path": str(unrelated)})
        with self.assertRaisesRegex(ValueError, "served this event"):
            checked_retire(self.event, self.records, self.training, unrelated, self.root)
        self.assert_nothing_retired()
        self.assertTrue((unrelated / "model.safetensors").is_file())

    def test_symlink_capture_cannot_retire_an_external_file(self):
        self.fixture()
        outside = self.base / "outside.pt"
        outside.write_bytes(b"unique")
        self.capture.unlink()
        self.capture.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.retire()
        self.assertEqual(outside.read_bytes(), b"unique")
        self.assertTrue(self.feature.is_file())
        self.assertFalse((self.event / "RETIREMENT_PLAN.json").exists())

    def test_malformed_compact_json_blocks_retirement(self):
        self.fixture()
        (self.event / "comparison.json").write_text("{incomplete")
        with self.assertRaises(json.JSONDecodeError):
            self.retire()
        self.assert_nothing_retired()

    def test_partial_retirement_failure_is_explicit_and_preserves_remaining_files(self):
        self.fixture()
        original_unlink = Path.unlink
        def fail_second(path, *args, **kwargs):
            if path == self.feature:
                raise OSError("simulated disk error")
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", fail_second):
            with self.assertRaisesRegex(OSError, "simulated"):
                self.retire()
        result = json.loads((self.event / "RETIREMENT.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual([item["path"] for item in result["removed"]], [str(self.capture)])
        self.assertTrue(self.feature.is_file())
        self.assertTrue(self.state.is_file())


if __name__ == "__main__":
    unittest.main()
