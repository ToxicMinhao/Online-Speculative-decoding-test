"""Buffer transition tests with exact token records and no model execution."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from graduate.buffer import BufferFullError, FeedbackBuffer, FeedbackError
from graduate.feedback import ResponseRecord


def response(index, *, session="conversation", record_id=None):
    return ResponseRecord(
        record_id=record_id or f"response_{index}", session_id=session,
        request_index=index, checkpoint_id="released_draft", purpose="serving",
        prompt_ids=(10, 20), response_ids=(30, 40, 2), finish_reason="eos",
        stop_token_ids=(2,), max_new_tokens=8, metadata={"native": {"token": 30}},
    )


class FeedbackBufferTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="graduate-buffer-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.buffer = FeedbackBuffer(self.root, capacity=3)

    def add_responses(self, count):
        for index in range(count):
            self.buffer.add(response(index))

    def test_exact_records_keep_arrival_order_and_seal_the_earliest(self):
        first = response(7, session="first")
        second = response(1, session="second")
        self.assertEqual(self.buffer.add(first), first)
        self.buffer.add(second)
        self.assertEqual(self.buffer.pending(), [first, second])
        self.assertEqual(self.buffer.records([first.record_id, second.record_id]), [first, second])
        event = self.buffer.seal_event("update_1", 1)
        self.assertEqual(event["record_ids"], [first.record_id])
        self.assertEqual(event["status"], "sealed")
        self.assertEqual(self.buffer.pending(), [second])
        self.assertEqual(self.buffer.records([first.record_id])[0].input_ids, (10, 20, 30, 40, 2))

    def test_duplicate_and_nonchronological_records_are_rejected(self):
        self.add_responses(2)
        rejected = [response(3, record_id="response_0"), response(1, record_id="new"),
                    response(0, record_id="older")]
        for record in rejected:
            with self.subTest(record=record.record_id):
                with self.assertRaises(FeedbackError):
                    self.buffer.add(record)
        for ids in (["response_1", "response_0"], ["response_0", "response_0"]):
            with self.subTest(ids=ids):
                with self.assertRaises(FeedbackError):
                    self.buffer.records(ids)
        with self.assertRaises(KeyError):
            self.buffer.records(["missing"])
        self.assertEqual(len(self.buffer.pending()), 2)

    def test_identical_import_is_a_no_op_in_every_event_state_after_restart(self):
        for status in ("pending", "sealed", "failed", "committed"):
            with self.subTest(status=status):
                root = self.root / status
                buffer = FeedbackBuffer(root, capacity=3)
                for index in range(3):
                    buffer.add(response(index))
                if status != "pending":
                    buffer.seal_event("update_1", 2)
                if status == "failed":
                    buffer.fail_event("update_1", "publication failed")
                if status == "committed":
                    buffer.commit_event("update_1", "updated_draft", {})
                reopened = FeedbackBuffer(root, capacity=3)
                before = reopened.state_path.read_bytes()
                pending = reopened.pending()
                with patch("graduate.buffer.write_json", side_effect=AssertionError("Unexpected write")):
                    imported = reopened.add(response(0))
                imported.metadata["native"].clear()
                self.assertEqual(reopened.pending(), pending)
                self.assertEqual(reopened.records(["response_0"]), [response(0)])
                self.assertEqual(reopened.state_path.read_bytes(), before)

    def test_same_id_with_changed_metadata_is_rejected(self):
        self.buffer.add(response(0))
        values = response(0).model_dump(mode="json")
        values["metadata"]["native"]["token"] = 30.0
        with self.assertRaisesRegex(FeedbackError, "Changed content"):
            self.buffer.add(ResponseRecord.model_validate(values))
        self.assertEqual(self.buffer.pending(), [response(0)])

    def test_evaluation_is_rejected_at_the_record_boundary(self):
        values = response(0).model_dump(mode="json")
        values["purpose"] = "evaluation"
        with self.assertRaises(ValueError):
            ResponseRecord.model_validate(values)
        self.assertEqual(self.buffer.pending(), [])

    def test_capacity_counts_pending_and_sealed_records(self):
        self.add_responses(3)
        with self.assertRaises(BufferFullError):
            self.buffer.add(response(3))
        self.buffer.seal_event("update_1", 2)
        with self.assertRaises(BufferFullError):
            self.buffer.add(response(3))
        reopened = FeedbackBuffer(self.root, capacity=3)
        self.assertEqual(reopened.event("update_1")["status"], "sealed")
        with self.assertRaises(FeedbackError):
            reopened.seal_event("update_2", 1)

    def test_failed_event_remains_reserved_after_restart(self):
        self.add_responses(3)
        self.buffer.seal_event("update_1", 2)
        failure = self.buffer.fail_event("update_1", "publication failed")
        reopened = FeedbackBuffer(self.root, capacity=3)
        self.assertEqual(reopened.event("update_1"), failure)
        self.assertEqual(reopened.pending(), [response(2)])
        self.assertEqual(reopened.fail_event("update_1", "publication failed"), failure)
        with self.assertRaises(BufferFullError):
            reopened.add(response(3))
        with self.assertRaises(FeedbackError):
            reopened.seal_event("update_2", 1)
        with self.assertRaises(FeedbackError):
            reopened.commit_event("update_1", "updated_draft", {})
        with self.assertRaises(FeedbackError):
            reopened.fail_event("update_1", "different failure")

    def test_commit_releases_capacity_preserves_history_and_is_exactly_idempotent(self):
        self.add_responses(3)
        self.buffer.seal_event("update_1", 2)
        receipt = {"publication": {"checkpoint_id": "updated_draft"}}
        committed = self.buffer.commit_event("update_1", "updated_draft", receipt)
        self.assertEqual(self.buffer.commit_event("update_1", "updated_draft", receipt), committed)
        for checkpoint, changed in (("other_draft", receipt), ("updated_draft", {})):
            with self.subTest(checkpoint=checkpoint, receipt=changed):
                with self.assertRaises(FeedbackError):
                    self.buffer.commit_event("update_1", checkpoint, changed)
        with self.assertRaises(FeedbackError):
            self.buffer.fail_event("update_1", "too late")
        self.buffer.add(response(3))
        self.buffer.add(response(4))
        self.assertEqual(self.buffer.records(["response_0", "response_1"]), [response(0), response(1)])
        self.assertEqual(self.buffer.seal_event("update_2", 3)["record_ids"],
                         ["response_2", "response_3", "response_4"])

    def test_public_results_and_receipts_cannot_mutate_saved_state(self):
        original = response(0)
        added = self.buffer.add(original)
        original.metadata["native"]["token"] = 999
        added.metadata["native"]["token"] = 888
        self.buffer.pending()[0].metadata["native"]["token"] = 777
        self.buffer.records(["response_0"])[0].metadata["native"]["token"] = 666
        self.assertEqual(self.buffer.pending()[0], response(0))
        self.buffer.seal_event("update_1", 1)["record_ids"].clear()
        receipt = {"publication": {"checkpoint_id": "updated_draft"}}
        expected_receipt = deepcopy(receipt)
        returned = self.buffer.commit_event("update_1", "updated_draft", receipt)
        receipt["publication"].clear()
        returned["receipt"].clear()
        self.buffer.event("update_1")["record_ids"].clear()
        saved = self.buffer.event("update_1")
        self.assertEqual(saved["record_ids"], ["response_0"])
        self.assertEqual(saved["receipt"], expected_receipt)

    def test_invalid_counts_duplicate_events_and_changed_capacity_are_rejected(self):
        self.buffer.add(response(0))
        for count in (0, True, 2):
            with self.subTest(count=count):
                with self.assertRaises(FeedbackError):
                    self.buffer.seal_event("update_1", count)
        self.buffer.seal_event("update_1", 1)
        self.buffer.commit_event("update_1", "updated_draft", {})
        with self.assertRaises(FeedbackError):
            self.buffer.seal_event("update_1", 1)
        with self.assertRaises(FeedbackError):
            FeedbackBuffer(self.root, capacity=4)
        with self.assertRaises(FeedbackError):
            FeedbackBuffer(self.root, capacity=True)

    def test_failed_atomic_write_preserves_disk_and_memory(self):
        self.buffer.add(response(0))
        previous = self.buffer.state_path.read_bytes()
        with patch("graduate.storage.os.fsync", side_effect=OSError("flush failed")):
            with self.assertRaisesRegex(OSError, "flush failed"):
                self.buffer.add(response(1))
            with self.assertRaisesRegex(OSError, "flush failed"):
                self.buffer.seal_event("update_1", 1)
        self.assertEqual(self.buffer.state_path.read_bytes(), previous)
        self.assertEqual(self.buffer.pending(), [response(0)])
        with self.assertRaises(KeyError):
            self.buffer.event("update_1")
        self.buffer.seal_event("update_1", 1)
        previous = self.buffer.state_path.read_bytes()
        with patch("graduate.storage.os.fsync", side_effect=OSError("flush failed")):
            with self.assertRaisesRegex(OSError, "flush failed"):
                self.buffer.commit_event("update_1", "updated_draft", {})
        self.assertEqual(self.buffer.state_path.read_bytes(), previous)
        self.assertFalse((self.root / "state.json.tmp").exists())
        self.assertEqual(self.buffer.event("update_1")["status"], "sealed")
        reopened = FeedbackBuffer(self.root, capacity=3)
        self.assertEqual(reopened.event("update_1")["status"], "sealed")
        with self.assertRaises(FeedbackError):
            reopened.seal_event("update_2", 1)


if __name__ == "__main__":
    unittest.main()
