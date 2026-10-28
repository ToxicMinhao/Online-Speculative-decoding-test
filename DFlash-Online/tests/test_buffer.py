"""CPU checks for chronological, persistent feedback ownership."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.buffer import BufferFullError, FeedbackBuffer, FeedbackError


def record(index=0, *, session="s", record_id=None, purpose="serving"):
    return {
        "record_id": record_id or f"{session}-{index}",
        "session_id": session,
        "request_index": index,
        "checkpoint_id": "released",
        "purpose": purpose,
        "prompt_ids": [101, 102, 7],
        "response_ids": [31, 2],
        "finish_reason": "eos",
        "stop_token_ids": [2, 3],
        "max_new_tokens": 4,
        "metadata": {"source": "native", "nested": {"calls": 1}},
    }


class FeedbackBufferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="feedback_buffer_test_")
        self.root = Path(self.temp.name) / "feedback"

    def tearDown(self):
        self.temp.cleanup()

    def buffer(self, capacity=8):
        return FeedbackBuffer(self.root, capacity)

    def test_exact_tokens_and_metadata_survive_restart_without_aliasing(self):
        buffer = self.buffer()
        original = record()
        expected = copy.deepcopy(original)
        returned = buffer.add(original)
        original["prompt_ids"][0] = 999
        returned["metadata"]["nested"]["calls"] = 999
        reopened = self.buffer()
        self.assertEqual(reopened.pending(), [expected])
        reopened.pending()[0]["response_ids"][0] = 999
        reopened.records(["s-0"])[0]["metadata"].clear()
        self.assertEqual(reopened.records(["s-0"]), [expected])
        state = json.loads((self.root / "state.json").read_text())
        self.assertEqual(state["records"], [expected])

    def test_capacity_includes_reserved_records_and_commit_releases_space(self):
        buffer = self.buffer(1)
        buffer.add(record(0))
        with self.assertRaises(BufferFullError):
            buffer.add(record(1))
        buffer.seal_event("e0", 1)
        self.assertEqual(buffer.pending(), [])
        with self.assertRaises(BufferFullError):
            buffer.add(record(1))
        buffer.commit_event("e0", "checkpoint-1", {"published": True})
        buffer.add(record(1))
        self.assertEqual(buffer.pending(), [record(1)])
        self.assertEqual(buffer.records(["s-0", "s-1"]), [record(0), record(1)])

    def test_events_take_earliest_records_in_arrival_order(self):
        buffer = self.buffer()
        for index in range(3):
            buffer.add(record(index))
        first = buffer.seal_event("first", 2)
        self.assertEqual(first["record_ids"], ["s-0", "s-1"])
        self.assertEqual(buffer.pending(), [record(2)])
        first["record_ids"].clear()
        self.assertEqual(self.buffer().event("first")["record_ids"], ["s-0", "s-1"])
        with self.assertRaises(FeedbackError):
            self.buffer().seal_event("second", 1)
        buffer.commit_event("first", "c1", {})
        self.assertEqual(buffer.seal_event("second", 1)["record_ids"], ["s-2"])

    def test_failed_event_never_requeues_or_retrains_after_restart(self):
        buffer = self.buffer(2)
        buffer.add(record(0))
        buffer.add(record(1))
        buffer.seal_event("partial", 1)
        failed = buffer.fail_event("partial", "optimizer outcome unknown")
        reopened = self.buffer(2)
        self.assertEqual(reopened.pending(), [record(1)])
        self.assertEqual(reopened.event("partial"), failed)
        with self.assertRaises(FeedbackError):
            reopened.seal_event("retry", 1)
        with self.assertRaises(FeedbackError):
            reopened.commit_event("partial", "guessed", {})
        with self.assertRaises(BufferFullError):
            reopened.add(record(2))
        self.assertEqual(reopened.records(["s-0"]), [record(0)])

    def test_repeated_commit_cannot_change_the_publication_receipt(self):
        buffer = self.buffer()
        buffer.add(record())
        buffer.seal_event("event", 1)
        receipt = {"optimizer_step": 2, "checkpoint_path": "events/event/export"}
        committed = buffer.commit_event("event", "updated", receipt)
        receipt["optimizer_step"] = 99
        reopened = self.buffer()
        self.assertEqual(reopened.commit_event("event", "updated", committed["receipt"]), committed)
        with self.assertRaises(FeedbackError):
            reopened.commit_event("event", "different", {})
        with self.assertRaises(FeedbackError):
            reopened.fail_event("event", "late error")
        with self.assertRaises(FeedbackError):
            reopened.seal_event("event", 1)

    def test_evaluation_cannot_enter_feedback(self):
        buffer = self.buffer()
        before = (self.root / "state.json").read_bytes()
        for purpose in ("evaluation", "diagnostic", "source_replay"):
            with self.subTest(purpose=purpose), self.assertRaises(FeedbackError):
                buffer.add(record(purpose=purpose))
        self.assertEqual((self.root / "state.json").read_bytes(), before)

    def test_duplicate_record_rejected_even_after_commit(self):
        buffer = self.buffer()
        buffer.add(record())
        with self.assertRaises(FeedbackError):
            buffer.add(record())
        buffer.seal_event("event", 1)
        buffer.commit_event("event", "updated", {})
        with self.assertRaises(FeedbackError):
            buffer.add(record())
        self.assertEqual(buffer.records(["s-0"]), [record()])

    def test_request_indices_increase_within_each_session_and_may_skip(self):
        buffer = self.buffer()
        buffer.add(record(2))
        buffer.add(record(4))
        for index in (1, 2, 3, 4):
            with self.subTest(index=index), self.assertRaises(FeedbackError):
                buffer.add(record(index, record_id=f"different-{index}"))
        buffer.add(record(0, session="other"))
        self.assertEqual([row["request_index"] for row in buffer.pending()], [2, 4, 0])

    def test_retrieval_preserves_order_and_rejects_unknown_ids(self):
        buffer = self.buffer()
        buffer.add(record(0))
        buffer.add(record(1))
        for ids in (["s-1", "s-0"], ["s-0", "s-0"]):
            with self.subTest(ids=ids), self.assertRaises(FeedbackError):
                buffer.records(ids)
        with self.assertRaises(KeyError):
            buffer.records(["absent"])
        with self.assertRaises(KeyError):
            buffer.event("absent")

    def test_exact_token_lists_are_required(self):
        buffer = self.buffer()
        for tokens in ([], [True], [-1], [1.5], (1, 2)):
            for name in ("prompt_ids", "response_ids"):
                invalid = record()
                invalid[name] = tokens
                with self.subTest(name=name, tokens=tokens), self.assertRaises(FeedbackError):
                    buffer.add(invalid)
        self.assertEqual(buffer.pending(), [])
        # Training eligibility belongs to feature preparation, so a one-token
        # response still belongs in the record of what was actually served.
        short = record()
        short["response_ids"] = [2]
        buffer.add(short)
        self.assertEqual(buffer.pending(), [short])

    def test_configuration_and_event_count_checks(self):
        buffer = self.buffer(1)
        with self.assertRaises(FeedbackError):
            self.buffer(8)
        for capacity in (0, -1, True, 1.0):
            with self.subTest(capacity=capacity), self.assertRaises(FeedbackError):
                self.buffer(capacity)
        buffer.add(record())
        for count in (0, -1, True, 2):
            with self.subTest(count=count), self.assertRaises(FeedbackError):
                buffer.seal_event("event", count)

    def test_old_schema_requires_fresh_directory_and_is_not_modified(self):
        self.root.mkdir()
        state = self.root / "state.json"
        state.write_text('{"state":{"schema_version":1},"sha256":"old-evidence"}')
        before = state.read_bytes()
        with self.assertRaisesRegex(FeedbackError, "fresh output directory"):
            self.buffer()
        self.assertEqual(state.read_bytes(), before)

    def test_interrupted_add_keeps_previous_state(self):
        buffer = self.buffer()
        buffer.add(record(0))
        with patch("online_feedback.storage.os.replace", side_effect=OSError("replace interrupted")):
            with self.assertRaises(OSError):
                buffer.add(record(1))
        self.assertEqual(buffer.pending(), [record(0)])
        self.assertEqual(self.buffer().pending(), [record(0)])

    def test_interrupted_commit_keeps_event_reserved(self):
        buffer = self.buffer()
        buffer.add(record())
        sealed = buffer.seal_event("event", 1)
        with patch("online_feedback.storage.os.replace", side_effect=OSError("replace interrupted")):
            with self.assertRaises(OSError):
                buffer.commit_event("event", "updated", {})
        reopened = self.buffer()
        self.assertEqual(reopened.event("event"), sealed)
        self.assertEqual(reopened.pending(), [])
        with self.assertRaises(FeedbackError):
            reopened.seal_event("retry", 1)


if __name__ == "__main__":
    unittest.main()
