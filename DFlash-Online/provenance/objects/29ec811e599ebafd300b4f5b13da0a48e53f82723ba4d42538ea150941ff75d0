"""Check the bounded qualification's causal ordering and failure evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from online_feedback.controller import OnlineController
from online_feedback.qualify_sglang import (
    admit_publication_resume, admit_storage, check_checkpoint_replays, check_recipe,
    exercise_loop, publication_resume_source,
)
from online_feedback.storage import write_json


class QualificationBackend:
    def __init__(self):
        self.checkpoint_id = "released"
        self.generated = []
        self.prepared = []
        self.published = []
        self.weights_match = True
        self.reload_calls = 1

    def generate(self, prompt_ids, **kwargs):
        self.generated.append(kwargs)
        cap = kwargs["max_new_tokens"]
        calls = 0 if cap == 1 else 1
        if kwargs["request_id"] == "updated_reloaded":
            calls = self.reload_calls
        record = {
            "record_id": kwargs["request_id"], "session_id": kwargs["session_id"],
            "request_index": kwargs["request_index"], "purpose": kwargs["purpose"],
            "checkpoint_id": self.checkpoint_id, "prompt_ids": list(prompt_ids),
            "response_ids": [31] if cap == 1 else [31, 2],
            "max_new_tokens": cap, "stop_token_ids": [2],
            "finish_reason": "length" if cap == 1 else "eos",
            "metadata": {"verification_calls": calls, "request_latency_s": 0.01},
        }
        write_json(kwargs["save_to"], record)
        return record

    def prepare_feedback(self, record, path):
        self.prepared.append(record["record_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        return {"status": "prepared", "feature_path": str(path)}

    def publish(self, checkpoint):
        self.published.append(checkpoint)
        self.checkpoint_id = checkpoint
        return {"checkpoint_id": checkpoint}

    def check_weights(self, checkpoint):
        return {"checkpoint": checkpoint, "all_equal": self.weights_match, "tensors": 58}


class QualificationTrainer:
    def __init__(self):
        self.events = []

    def status(self):
        return {
            "global_step": 2 * len(self.events), "presentations": 4 * len(self.events),
            "completed_events": list(self.events),
        }

    def train_event(self, event_id, feature_paths, passes):
        if len(feature_paths) != 2 or passes != 2:
            raise AssertionError("qualification changed the event size")
        self.events.append(event_id)
        return {"checkpoint_path": "updated", "optimizer_steps": 2, "response_presentations": 4}


class SGLangQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def exercise(self, backend):
        trainer = QualificationTrainer()
        controller = OnlineController(backend, trainer, self.root / "loop", buffer_capacity=2)
        summary = {"status": "running", "serving": [], "replays": []}
        self.summary = summary
        exercise_loop(controller, [[1, 10], [1, 11], [1, 12]], self.root, summary,
                      released_checkpoint="released")
        return trainer, summary

    def test_exactly_one_update_and_eight_generations(self):
        backend = QualificationBackend()
        trainer, summary = self.exercise(backend)
        self.assertEqual(trainer.events, ["event_1"])
        self.assertEqual(backend.prepared, ["qualification_1", "qualification_2"])
        self.assertEqual(len(backend.generated), 8)
        self.assertEqual([item["purpose"] for item in backend.generated].count("serving"), 2)
        self.assertEqual(backend.published, ["updated", "released", "updated"])
        self.assertTrue(all(item["learning_unchanged"] for item in summary["replays"]))

    def test_failed_native_publication_is_not_reported_as_qualified(self):
        backend = QualificationBackend()
        backend.weights_match = False
        with self.assertRaisesRegex(AssertionError, "native_publication"):
            self.exercise(backend)
        saved = json.loads((self.root / "PROGRESS.json").read_text())
        self.assertFalse(saved["publication"]["all_equal"])
        self.assertEqual(len(backend.generated), 5)
        self.assertEqual(len(backend.published), 1)

    def test_reload_call_change_fails_even_when_output_tokens_match(self):
        backend = QualificationBackend()
        backend.reload_calls = 2
        with self.assertRaisesRegex(AssertionError, "reloaded_verification_calls"):
            self.exercise(backend)
        self.assertTrue(self.summary["replays"][-1]["output_equal"])
        record = json.loads((self.root / "loop/evaluation/updated_reloaded.json").read_text())
        self.assertEqual(record["metadata"]["verification_calls"], 2)

    def test_storage_admission_records_failure_and_includes_reserve(self):
        draft = self.root / "draft"
        draft.mkdir()
        (draft / "model.safetensors").write_bytes(b"a" * 100)
        with patch("online_feedback.qualify_sglang.shutil.disk_usage",
                   return_value=SimpleNamespace(free=10 * 2**30)):
            with self.assertRaisesRegex(ValueError, "Insufficient storage"):
                admit_storage(self.root, draft)
        saved = json.loads((self.root / "STORAGE_ADMISSION.json").read_text())
        self.assertEqual(saved["required_bytes"], 420 + 10 * 2**30)
        self.assertEqual(saved["reserve_bytes"], 8 * 2**30)
        self.assertFalse(saved["admitted"])

    def test_recipe_drift_is_rejected_before_native_execution(self):
        config = Path(__file__).resolve().parents[1] / "configs/osd.yaml"
        self.assertEqual(check_recipe(config)["max_steps"], 7424)
        changed = self.root / "changed.yaml"
        changed.write_text(config.read_text().replace("learning_rate: 0.0003", "learning_rate: 0.0001"))
        with self.assertRaisesRegex(AssertionError, "learning_rate"):
            check_recipe(changed)

    def test_publication_recovery_does_not_serve_collect_or_train(self):
        backend = QualificationBackend()
        backend.publish("updated")
        trainer = QualificationTrainer()
        trainer.events = ["event_1"]
        controller = OnlineController(backend, trainer, self.root / "loop", buffer_capacity=2)
        summary = {"replays": []}
        with patch.object(controller, "serve", side_effect=AssertionError("unexpected serving")), \
                patch.object(controller, "update", side_effect=AssertionError("unexpected update")), \
                patch.object(trainer, "train_event", side_effect=AssertionError("unexpected training")), \
                patch.object(backend, "prepare_feedback", side_effect=AssertionError("unexpected feedback")):
            check_checkpoint_replays(controller, [[1], [2], [3]], self.root, summary,
                                     released_checkpoint="released", checkpoint="updated")
        self.assertEqual(len(backend.generated), 3)
        self.assertTrue(all(row["purpose"] == "evaluation" for row in backend.generated))
        self.assertEqual(trainer.status()["global_step"], 2)
        self.assertEqual(controller.buffer.pending(), [])

    def test_recovery_budget_does_not_duplicate_existing_checkpoint(self):
        with patch("online_feedback.qualify_sglang.shutil.disk_usage",
                   return_value=SimpleNamespace(free=9 * 2**30)):
            budget = admit_publication_resume(self.root)
        self.assertTrue(budget["admitted"])
        self.assertEqual(budget["required_bytes"], 9 * 2**30)

    def test_recovery_requires_committed_two_step_event_and_preserves_source(self):
        source = self.root / "old"
        event_root = source / "trainer/events/event_1"
        export = event_root / "export"
        export.mkdir(parents=True)
        state = source / "trainer/training_state.pt"
        state.write_bytes(b"retained state")
        write_json(source / "PLAN.json", {
            "target": "target", "draft": "draft", "max_new_tokens": 128,
            "block_size": 16, "context_limit": 2048, "seed": 17,
            "training_objective": "osd", "training_reduction": "response_mean",
        })
        write_json(source / "INPUTS.json", {"input_ids": [[1], [2], [3]]})
        write_json(source / "RESULT.json", {
            "status": "failed", "serving": [{}, {}],
            "replays": [{"output_equal": True, "learning_unchanged": True}] * 2,
            "error": {"message": "publication failed"},
        })
        training_path = event_root / "RESULT.json"
        training = {
            "status": "completed", "event_id": "event_1", "objective": "osd",
            "passes": 2, "optimizer_steps": 2, "response_presentations": 4,
            "checkpoint_path": str(export), "full_state_path": str(state),
        }
        write_json(training_path, training)
        source_files = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
        retained = publication_resume_source(source)
        self.assertEqual(retained["state"], state)
        self.assertEqual(retained["checkpoint"], export)
        self.assertEqual({path: path.read_bytes() for path in source_files}, source_files)
        training["optimizer_steps"] = 1
        write_json(training_path, training)
        with self.assertRaisesRegex(AssertionError, "optimizer_steps"):
            publication_resume_source(source)


if __name__ == "__main__":
    unittest.main()
