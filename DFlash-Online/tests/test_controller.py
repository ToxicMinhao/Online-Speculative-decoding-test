"""CPU lifecycle checks using the persistent buffer and simple model doubles."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_feedback.buffer import FeedbackError
from online_feedback.controller import OnlineController
from online_feedback.storage import write_json


class FakeTrainer:
    def __init__(self):
        self.calls = []
        self.steps = 0
        self.failure = None
        self.on_train = None

    def status(self):
        return {"global_step": self.steps, "completed_events": [call[0] for call in self.calls]}

    def train_event(self, event_id, paths, passes=2):
        self.calls.append((event_id, list(paths), passes))
        if self.on_train:
            self.on_train()
        if self.failure:
            raise self.failure
        self.steps += 2
        return {
            "checkpoint_path": "event-export",
            "full_state_path": "committed-training-state",
            "optimizer_steps": 2,
        }


class FakeBackend:
    def __init__(self):
        self.checkpoint_id = "released"
        self.generations = []
        self.prepared = []
        self.published = []
        self.response = [31, 2]
        self.wrong_prompt = False
        self.fail_publication = False
        self.on_generate = None
        self.on_publish = None

    def generate(self, prompt_ids, **kwargs):
        self.generations.append((list(prompt_ids), copy.deepcopy(kwargs)))
        record = {
            "record_id": kwargs["request_id"],
            "session_id": kwargs["session_id"],
            "request_index": kwargs["request_index"],
            "checkpoint_id": self.checkpoint_id,
            "purpose": kwargs["purpose"],
            "prompt_ids": [999] if self.wrong_prompt else list(prompt_ids),
            "response_ids": list(self.response),
            "finish_reason": "eos",
            "stop_token_ids": [2],
            "max_new_tokens": kwargs["max_new_tokens"],
            "metadata": {"verification_calls": 1, "request_latency_s": 0.5},
        }
        if self.on_generate:
            self.on_generate()
        write_json(kwargs["save_to"], record)
        return record

    def prepare_feedback(self, record, path):
        self.prepared.append(record["record_id"])
        if len(record["response_ids"]) < 2:
            return {"status": "ineligible_short_response", "record_id": record["record_id"]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"CPU feature fixture")
        return {"status": "prepared", "record_id": record["record_id"], "feature_path": str(path)}

    def publish(self, path):
        self.published.append(path)
        if self.on_publish:
            self.on_publish()
        if self.fail_publication:
            raise OSError("controlled publication failure")
        self.checkpoint_id = "updated"
        return {"checkpoint_id": "updated"}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="controller_contract_")
        self.root = Path(self.temp.name) / "loop"
        self.backend = FakeBackend()
        self.trainer = FakeTrainer()
        self.controller = self.reopen()

    def tearDown(self):
        self.temp.cleanup()

    def reopen(self):
        return OnlineController(self.backend, self.trainer, self.root, buffer_capacity=2)

    def serve(self, index=1, collect=True):
        return self.controller.serve(
            [101, 7], request_id=f"r{index}", session_id="s", request_index=index,
            max_new_tokens=4, collect_feedback=collect,
        )

    def evaluate(self, label="eval"):
        return self.controller.evaluate(
            [101, 7], canonical_response_ids=[31, 2], label=label, max_new_tokens=4,
        )

    def comparison(self, label):
        return json.loads((self.root / "evaluation" / f"{label}.comparison.json").read_text())

    def test_evaluation_never_enters_feedback_or_training(self):
        self.serve()
        before = (self.root / "feedback_buffer" / "state.json").read_bytes()
        with (
            patch.object(self.controller.buffer, "add", side_effect=AssertionError("evaluation added feedback")),
            patch.object(self.controller.buffer, "seal_event", side_effect=AssertionError("evaluation sealed event")),
            patch.object(self.backend, "prepare_feedback", side_effect=AssertionError("evaluation prepared training")),
            patch.object(self.trainer, "train_event", side_effect=AssertionError("evaluation trained")),
        ):
            result = self.evaluate()
        self.assertTrue(result["output_equal"])
        self.assertTrue(result["same_input"])
        self.assertTrue(result["learning_unchanged"])
        self.assertEqual(result["canonical_response_ids"], [31, 2])
        self.assertEqual(self.backend.generations[-1][1]["purpose"], "evaluation")
        self.assertEqual(before, (self.root / "feedback_buffer" / "state.json").read_bytes())

    def test_output_mismatch_preserves_raw_record_and_comparison(self):
        self.backend.response = [32, 2]
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint replay"):
            self.evaluate("bad_output")
        comparison = self.comparison("bad_output")
        self.assertFalse(comparison["output_equal"])
        self.assertTrue(comparison["same_input"])
        record = json.loads((self.root / "evaluation" / "bad_output.json").read_text())
        self.assertEqual(record["response_ids"], [32, 2])
        self.assertEqual(self.controller.buffer.pending(), [])
        self.assertEqual(self.trainer.calls, [])

    def test_input_mismatch_preserves_comparison(self):
        self.backend.wrong_prompt = True
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint replay"):
            self.evaluate("bad_input")
        self.assertFalse(self.comparison("bad_input")["same_input"])
        self.assertEqual(self.trainer.calls, [])

    def test_evaluation_detects_trainer_counter_mutation(self):
        self.backend.on_generate = lambda: setattr(self.trainer, "steps", 1)
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint replay"):
            self.evaluate("bad_learning")
        self.assertFalse(self.comparison("bad_learning")["learning_unchanged"])

    def test_serving_collection_can_be_disabled(self):
        record = self.serve(collect=False)
        self.assertEqual(record["purpose"], "serving")
        self.assertEqual(self.controller.buffer.pending(), [])
        self.assertTrue((self.root / "serving" / "r1.json").is_file())

    def test_success_trains_ordered_feedback_and_commits_after_publication(self):
        self.serve(1)
        self.serve(2)
        phases = []

        def during_training():
            phases.append("training")
            self.assertEqual(self.controller.buffer.event("e1")["status"], "sealed")
            self.assertEqual(self.backend.published, [])

        def during_publication():
            phases.append("publication")
            self.assertEqual(self.trainer.steps, 2)
            self.assertEqual(self.controller.buffer.event("e1")["status"], "sealed")

        self.trainer.on_train = during_training
        self.backend.on_publish = during_publication
        receipt = self.controller.update("e1", 2, passes=2)
        phases.append("committed")
        self.assertEqual(phases, ["training", "publication", "committed"])
        self.assertEqual(self.backend.prepared, ["r1", "r2"])
        self.assertEqual(self.backend.published, ["event-export"])
        self.assertEqual([path.name for path in self.trainer.calls[0][1]], ["r1.pt", "r2.pt"])
        self.assertEqual(self.trainer.calls[0][2], 2)
        event = self.reopen().buffer.event("e1")
        self.assertEqual(event["status"], "committed")
        self.assertEqual(event["checkpoint_id"], "updated")
        self.assertEqual(event["receipt"], receipt)
        self.assertEqual(self.controller.buffer.pending(), [])
        self.serve(3)
        self.assertEqual(self.controller.buffer.pending()[0]["checkpoint_id"], "updated")

    def test_failed_publication_blocks_retraining_under_same_or_new_event(self):
        self.serve(1)
        self.serve(2)
        self.backend.fail_publication = True
        with self.assertRaisesRegex(OSError, "publication failure"):
            self.controller.update("e1", 2)
        self.assertEqual(self.trainer.steps, 2)
        self.assertEqual(len(self.trainer.calls), 1)
        failed = self.controller.buffer.event("e1")
        self.assertEqual(failed["status"], "failed")
        self.assertIn("publication failure", failed["error"])
        self.assertEqual(self.backend.checkpoint_id, "released")
        reopened = self.reopen()
        for event_id in ("e1", "e2"):
            with self.subTest(event_id=event_id), self.assertRaises(FeedbackError):
                reopened.update(event_id, 2)
        self.assertEqual(len(self.trainer.calls), 1)
        self.assertTrue((self.root / "features" / "r1.pt").is_file())

    def test_short_response_stops_before_training_without_resampling(self):
        self.backend.response = [2]
        self.serve(1)
        self.serve(2)
        with self.assertRaisesRegex(ValueError, "No eligible training anchors"):
            self.controller.update("short", 2)
        self.assertEqual(len(self.backend.generations), 2)
        self.assertEqual(self.trainer.calls, [])
        self.assertEqual(self.controller.buffer.event("short")["status"], "failed")
        self.assertEqual(self.controller.buffer.records(["r1"])[0]["response_ids"], [2])

    def test_training_failure_never_publishes_or_retries(self):
        self.serve(1)
        self.serve(2)
        self.trainer.failure = RuntimeError("controlled training failure")
        with self.assertRaisesRegex(RuntimeError, "training failure"):
            self.controller.update("e1", 2)
        self.assertEqual(self.backend.published, [])
        self.assertEqual(self.controller.buffer.event("e1")["status"], "failed")
        with self.assertRaises(FeedbackError):
            self.controller.update("e2", 2)
        self.assertEqual(len(self.trainer.calls), 1)

    def test_keyboard_interrupt_also_leaves_event_blocked(self):
        self.serve()
        self.trainer.failure = KeyboardInterrupt("requested stop")
        with self.assertRaises(KeyboardInterrupt):
            self.controller.update("stopped", 1)
        self.assertIn("KeyboardInterrupt", self.reopen().buffer.event("stopped")["error"])
        with self.assertRaises(FeedbackError):
            self.reopen().update("retry", 1)

    def test_failure_to_save_error_does_not_hide_original_error(self):
        self.serve()
        self.trainer.failure = RuntimeError("original training failure")
        with patch.object(self.controller.buffer, "fail_event", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(RuntimeError, "original training failure") as caught:
                self.controller.update("e1", 1)
        self.assertIn("disk unavailable", caught.exception.__notes__[0])
        self.assertEqual(self.reopen().buffer.event("e1")["status"], "sealed")
        with self.assertRaises(FeedbackError):
            self.reopen().update("retry", 1)

    def test_costs_are_separate_from_feedback_and_training_receipts(self):
        record = self.serve()
        self.evaluate()
        receipt = self.controller.update("e1", 1)
        serving_costs = json.loads((self.root / "serving" / "r1.costs.json").read_text())
        evaluation_costs = json.loads((self.root / "evaluation" / "eval.costs.json").read_text())
        update_costs = json.loads((self.root / "events" / "e1.costs.json").read_text())
        self.assertGreaterEqual(serving_costs["buffer_persistence_s"], 0)
        self.assertEqual(evaluation_costs["native_request_latency_s"], 0.5)
        self.assertGreaterEqual(update_costs["update_call_s"], 0)
        self.assertNotIn("serve_call_s", record)
        self.assertNotIn("update_call_s", receipt)

    def test_ids_must_be_filename_components(self):
        with self.assertRaisesRegex(ValueError, "filename"):
            self.controller.serve([1], request_id="../outside", session_id="s", request_index=1, max_new_tokens=4)
        with self.assertRaisesRegex(ValueError, "filename"):
            self.controller.update("../outside", 1)
        with self.assertRaisesRegex(ValueError, "filename"):
            self.evaluate("../outside")
        self.assertEqual(self.backend.generations, [])


if __name__ == "__main__":
    unittest.main()
