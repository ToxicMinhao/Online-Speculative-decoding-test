"""Target-objective event boundaries on the real upstream core and optimizer.

The scalar model keeps these lifecycle tests inexpensive. Objective mathematics
and actual DFlash forwards are covered in test_objective_integration.py.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch

from online_feedback import training as T
from test_training_contract import TinyTrainer, feature


def verified_feature(record_id="r1", response=(4, 5)):
    raw = feature(record_id, response)
    raw["label_binding"] = {
        "capture_backend": "same_frozen_HF_target_canonical_replay",
        "record_id": record_id,
        "prompt_length": raw["prompt_length"],
        "target_argmax_ids": list(response),
        "label_mismatches": [],
    }
    return raw


class EventBoundaryTrainer(TinyTrainer):
    """Instrument reference lifetime around real backwards and optimizer steps."""

    def _build_runtime(self):
        super()._build_runtime()
        self.contract["online_target_objective"] = self.objective
        self.phase_log = []
        self.reference_snapshots = []
        model = self.bundle.model
        model.reference = None

        def begin_event():
            if model.reference is not None:
                raise RuntimeError("Reference already active")
            weights = model.draft_model.weight.detach().clone()
            model.reference = SimpleNamespace(weights=weights)
            self.reference_snapshots.append(weights)
            self.phase_log.append("begin")

        def end_event():
            self.phase_log.append("end")
            model.reference = None
            return {
                "reference_setup_s": 0.125,
                "reference_forward_s": 0.25,
                "projection_s": 0.0625,
            }

        def observe_backward(gradient):
            if model.reference is None:
                raise RuntimeError("Backward lost its event reference")
            self.phase_log.append("backward")
            return gradient

        model.begin_event = begin_event
        model.end_event = end_event
        model.draft_model.weight.register_hook(observe_backward)

    def _checkpoint_state(self, event):
        if self.bundle.model.reference is not None:
            raise RuntimeError("Checkpoint must follow reference cleanup")
        self.phase_log.append("checkpoint")
        return super()._checkpoint_state(event)

    def _export(self, export_path):
        export_path.mkdir()
        (export_path / "model.safetensors").write_bytes(b"tiny fixture export")


class TargetBindingTests(unittest.TestCase):
    def test_verified_canonical_labels_are_accepted(self):
        raw = verified_feature()
        original_ids = raw["input_ids"].clone()
        T.validate_target_binding(raw)
        self.assertTrue(torch.equal(raw["input_ids"], original_ids))

    def test_target_mismatch_and_wrong_source_are_rejected(self):
        changes = {
            "target_argmax_ids": [4, 7],
            "capture_backend": "dataset_assistant_labels",
            "record_id": "another_response",
            "prompt_length": 1,
            "label_mismatches": [{"position": 3}],
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                raw = verified_feature()
                raw["label_binding"][field] = value
                with self.assertRaisesRegex(ValueError, "canonical target-greedy"):
                    T.validate_target_binding(raw)

    def test_missing_provenance_is_rejected(self):
        raw = verified_feature()
        del raw["label_binding"]
        with self.assertRaisesRegex(ValueError, "canonical target-greedy"):
            T.validate_target_binding(raw)


class TargetEventTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text("{}")
        self.features = []
        for index in range(2):
            path = self.root / f"response{index}.pt"
            torch.save(verified_feature(f"r{index}", (4, 5 + index)), path)
            self.features.append(path)
        logger = mock.patch("specforge.optimizer.print_on_rank0")
        logger.start()
        self.addCleanup(logger.stop)

    def trainer(self, name="run", objective="projected"):
        trainer = EventBoundaryTrainer(self.config, self.root / name, objective=objective)
        self.addCleanup(trainer.close)
        return trainer

    def test_one_reference_spans_all_passes_and_ends_after_backwards(self):
        trainer = self.trainer()
        initial_weights = trainer.bundle.draft_model.weight.detach().clone()
        receipt = trainer.train_event("event1", self.features, passes=2)
        self.assertEqual(
            trainer.phase_log,
            ["begin", "backward", "backward", "backward", "backward", "end", "checkpoint"],
        )
        self.assertIsNone(trainer.bundle.model.reference)
        self.assertEqual(receipt["response_presentations"], 4)
        self.assertEqual(receipt["optimizer_steps"], 2)
        self.assertEqual([row["pass"] for row in receipt["records"]], [0, 0, 1, 1])
        self.assertTrue(torch.equal(trainer.reference_snapshots[0], initial_weights))
        self.assertFalse(torch.equal(trainer.bundle.draft_model.weight, initial_weights))
        self.assertEqual(receipt["objective"], "projected")
        self.assertEqual(receipt["objective_work"], {
            "reference_setup_s": 0.125,
            "reference_forward_s": 0.25,
            "projection_s": 0.0625,
        })
        saved_receipt = json.loads((trainer.events_root / "event1" / "RESULT.json").read_text())
        saved_state = torch.load(trainer.latest, weights_only=False)
        self.assertEqual(saved_receipt["objective_work"], receipt["objective_work"])
        self.assertEqual(saved_state["events"][0]["objective_work"], receipt["objective_work"])

    def test_qualification_observer_leaves_training_and_rng_unchanged(self):
        from online_feedback.qualify import assert_state_equal
        from online_feedback.rng import capture_rng

        ordinary = self.trainer("ordinary")
        ordinary.train_event("event1", self.features)
        expected_rng = capture_rng()
        expected_optimizer = copy.deepcopy(ordinary.backend.optimizer.state_dict())
        expected_weights = copy.deepcopy(ordinary.bundle.draft_model.state_dict())

        checked = self.trainer("checked")
        phases = []

        def observe(model, phase):
            self.assertIsNotNone(model.reference)
            phases.append((phase, checked.global_step, len(checked.phase_log)))

        checked.train_event("event1", self.features, reference_check=observe)
        self.assertEqual(phases, [("before", 0, 1), ("after", 2, 5)])
        assert_state_equal(checked.bundle.draft_model.state_dict(), expected_weights)
        assert_state_equal(checked.backend.optimizer.state_dict(), expected_optimizer)
        assert_state_equal(capture_rng(), expected_rng)

    def test_qualification_failure_still_releases_reference_and_never_exports(self):
        for failed_phase in ("before", "after"):
            with self.subTest(phase=failed_phase):
                trainer = self.trainer("failure_" + failed_phase)

                def observe(model, phase):
                    if phase == failed_phase:
                        raise ValueError("reference qualification failed")

                with self.assertRaisesRegex(ValueError, "reference qualification failed"):
                    trainer.train_event("event1", self.features, reference_check=observe)
                self.assertIsNone(trainer.bundle.model.reference)
                self.assertTrue(trainer.failed)
                self.assertFalse(trainer.latest.exists())
                self.assertEqual(trainer.global_step, 0 if failed_phase == "before" else 2)
                self.assertEqual(trainer.phase_log[-1], "end")

    def test_logged_learning_rate_is_the_rate_actually_used_before_scheduler_step(self):
        trainer = self.trainer()
        optimizer = trainer.backend.optimizer
        actual_rates = []
        ordinary_step = optimizer.step

        def observe_step(*args, **kwargs):
            actual_rates.append(optimizer.get_learning_rate())
            return ordinary_step(*args, **kwargs)

        with mock.patch.object(optimizer, "step", side_effect=observe_step):
            receipt = trainer.train_event("event1", self.features)
        rates = [row["learning_rate"] for row in receipt["records"]]
        self.assertEqual(rates, [None, actual_rates[0], None, actual_rates[1]])
        self.assertNotEqual(actual_rates[-1], optimizer.get_learning_rate())

    def test_next_event_captures_updated_weights_without_restarting_optimizer(self):
        trainer = self.trainer()
        optimizer = trainer.backend.optimizer
        trainer.train_event("event1", self.features)
        before_second = trainer.bundle.draft_model.weight.detach().clone()
        trainer.train_event("event2", self.features)
        self.assertIs(trainer.backend.optimizer, optimizer)
        self.assertTrue(torch.equal(trainer.reference_snapshots[1], before_second))
        self.assertEqual(trainer.phase_log.count("begin"), 2)
        self.assertEqual(trainer.phase_log.count("end"), 2)
        self.assertEqual((trainer.global_step, trainer.presentations), (4, 8))

    def test_training_failure_releases_reference_before_recording_failure(self):
        trainer = self.trainer()
        original_step = trainer.core.train_step

        def fail_second(*args, **kwargs):
            if trainer.presentations == 1:
                raise RuntimeError("deliberate second-presentation failure")
            return original_step(*args, **kwargs)

        with mock.patch.object(trainer.core, "train_step", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "second-presentation"):
                trainer.train_event("failed", self.features)
        self.assertEqual(trainer.phase_log, ["begin", "backward", "end"])
        self.assertIsNone(trainer.bundle.model.reference)
        self.assertTrue(trainer.failed)
        self.assertFalse(trainer.latest.exists())
        failure = json.loads((trainer.events_root / "failed" / "FAILED.json").read_text())
        self.assertEqual(failure["presentations"], 1)
        self.assertFalse(failure["automatic_retry"])

    def test_bad_target_binding_stops_before_reference_creation_or_training(self):
        trainer = self.trainer()
        raw = verified_feature("r1", (4, 6))
        raw["label_binding"]["target_argmax_ids"] = [4, 5]
        torch.save(raw, self.features[1])
        with self.assertRaisesRegex(ValueError, "canonical target-greedy"):
            trainer.train_event("bad_binding", self.features)
        self.assertEqual(trainer.phase_log, [])
        self.assertEqual(trainer.author_strategy.calls, 0)
        self.assertIsNone(trainer.bundle.model.reference)
        self.assertFalse(trainer.failed)

    def test_resume_with_changed_objective_is_rejected_before_state_loading(self):
        projected = self.trainer("projected")
        projected.train_event("event1", self.features)
        control = self.trainer("control", objective="hard_ce")
        original_weights = copy.deepcopy(control.bundle.draft_model.state_dict())
        with self.assertRaisesRegex(ValueError, "algorithm or schedule"):
            control._restore(projected.latest)
        self.assertEqual((control.global_step, control.presentations), (0, 0))
        for name, value in control.bundle.draft_model.state_dict().items():
            self.assertTrue(torch.equal(value, original_weights[name]))


if __name__ == "__main__":
    unittest.main()
