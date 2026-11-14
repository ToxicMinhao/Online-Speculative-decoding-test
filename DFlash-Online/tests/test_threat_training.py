"""Replay integration through actual TrainerCore and persistent BF16 AdamW."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.ok_continuation.resume import differences
from experiments.threatened_repair_replay.qualification import qualify
from experiments.threatened_repair_replay.state import capture_state, copy_tree, restore_state, verify_state
from online_feedback.optimizer import DirectBF16AdamW
from test_threat_state import first_pair, tiny_trainer


def add_fixed_gradient(trainer):
    for parameter in trainer.bundle.draft_model.parameters():
        parameter.grad.add_(torch.full_like(parameter, .125))
    return dict(kind="synthetic CPU fixture", coefficient=.1)


def no_replay(_trainer):
    return dict(coefficient=0., skipped=True)


def padded_zero(trainer):
    parameters = tuple(trainer.bundle.draft_model.parameters())
    loss = trainer.bundle.draft_model(torch.ones((1, 3), dtype=torch.bfloat16)).float().sum() * 0.
    gradients = torch.autograd.grad(loss, parameters)
    if any(bool(gradient.any()) for gradient in gradients):
        raise AssertionError("Padded zero fixture unexpectedly has gradient")
    return dict(coefficient=0., skipped=False, padded_backward=True)


class TrainingTests(unittest.TestCase):
    def test_native_response_average_then_replay_then_one_clip_and_adam(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, features = tiny_trainer(directory)
            pair = first_pair(trainer, features)
            model = deepcopy(trainer.bundle.draft_model)
            optimizer = DirectBF16AdamW(model)
            optimizer.load_state_dict(copy_tree(trainer.backend.optimizer.state_dict()))
            for feature in pair:
                raw = torch.load(feature["path"], weights_only=True)
                loss = (model(raw["x"]).float() - raw["y"]).square().sum() / raw["denominator"]
                (loss / 2).backward()
            expected_native = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
            for parameter in model.parameters():
                parameter.grad.add_(torch.full_like(parameter, .125))
            expected_combined = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
            optimizer.step()
            clip = torch.nn.utils.clip_grad_norm_
            with patch("torch.nn.utils.clip_grad_norm_", wraps=clip) as clip_spy:
                result = trainer.run_pair(pair, 0, 0, replay=add_fixed_gradient, capture_gradients=True)
            self.assertEqual(clip_spy.call_count, 1)
            self.assertEqual(result["update"]["optimizer_applications"], 1)
            self.assertEqual(differences(result["preclip_gradients"]["native"], expected_native), [])
            self.assertEqual(differences(result["preclip_gradients"]["combined"], expected_combined), [])
            self.assertEqual(differences(trainer.bundle.draft_model.state_dict(), model.state_dict()), [])
            self.assertEqual(differences(trainer.backend.optimizer.state_dict(), optimizer.state_dict()), [])
            self.assertEqual(trainer.global_step, 769)
            self.assertEqual(trainer.presentations, 1538)
            self.assertEqual(trainer.core.accumulation_remainder, 0)
            self.assertEqual(len(trainer.response_records), 2)
            self.assertEqual(len(trainer.window_records), 1)

    def test_disabled_replay_and_disposable_restore_preserve_native_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, features = tiny_trainer(directory)
            pair = first_pair(trainer, features)
            before = capture_state(trainer)
            native = trainer.run_pair(pair, 0, 0, capture_gradients=True)
            after = capture_state(trainer)
            restore_state(trainer, before)
            disabled = trainer.run_pair(pair, 0, 0, label="disabled", category="disposable",
                replay=no_replay, reference_gradients=native["preclip_gradients"]["native"])
            self.assertEqual(disabled["update"]["gradient_change"]["difference_norm"], 0)
            self.assertEqual(differences(trainer.bundle.draft_model.state_dict(), after["scientific"]["draft_state_dict"]), [])
            restore_state(trainer, after)
            verify_state(trainer, after)
            self.assertEqual(trainer.response_records[0]["event_id"], "event_007")
            self.assertEqual(len(trainer.window_records), 1)

    def test_failure_receipt_written_before_exception_and_step_is_not_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, features = tiny_trainer(directory)
            pair = first_pair(trainer, features)
            def fail(_trainer):
                raise ValueError("controlled replay failure")
            with self.assertRaisesRegex(ValueError, "controlled replay failure"):
                trainer.run_pair(pair, 0, 0, replay=fail)
            record = json.loads((trainer.events_root / "event_007" / "PAIR_FAILED.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["optimizer_applications"], 0)
            self.assertEqual(record["completed_presentations"], 1)
            self.assertEqual(trainer.global_step, 768)
            self.assertEqual(trainer._active_event, "event_007")

    def test_failing_native_gradient_check_is_saved_without_running_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, features = tiny_trainer(directory)
            pair = first_pair(trainer, features)
            wrong = {name: torch.zeros_like(parameter) for name, parameter in trainer.bundle.draft_model.named_parameters()}
            called = []
            with self.assertRaisesRegex(ValueError, "different native gradients"):
                trainer.run_pair(pair, 0, 0, replay=lambda _: called.append(True), reference_gradients=wrong)
            self.assertFalse(called)
            record = json.loads((trainer.events_root / "event_007" / "PAIR_UPDATE.json").read_text())
            self.assertEqual(record["native_gradient_parity"]["status"], "failed")
            self.assertEqual(record["optimizer_applications"], 0)

    def test_consecutive_pairs_reuse_optimizer_and_refresh_native_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, features = tiny_trainer(directory)
            order = torch.randperm(len(features)).tolist()
            first = [features[index] for index in order[:2]]
            second = [features[index] for index in order[2:]]
            trainer.run_pair(first, 0, 0)
            trainer.run_pair(second, 0, 2, replay=add_fixed_gradient)
            self.assertEqual(trainer.global_step, 770)
            self.assertEqual(trainer.presentations, 1540)
            self.assertEqual(trainer._presentation_schedule.position, 4)
            self.assertEqual([row["presentation_ordinal"] for row in trainer.response_records], [0, 1, 2, 3])
            self.assertEqual([row["window"] for row in trainer.window_records], [1, 2])
            self.assertEqual([row["record_ids"] for row in trainer.window_records],
                             [[item["record_id"] for item in first], [item["record_id"] for item in second]])
            for state in trainer.backend.optimizer.optimizer.state.values():
                self.assertEqual(int(state["step"]), 770)

    def test_six_case_qualification_restores_origin_without_retaining_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, features = tiny_trainer(Path(directory) / "trainer", count=128)
            origin = capture_state(trainer)
            result = qualify(trainer, [Path(row["path"]) for row in features], Path(directory) / "qualification",
                dict(disabled=None, zero=no_replay, padded_zero=padded_zero,
                     U=add_fixed_gradient, T=add_fixed_gradient))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["optimizer_steps"], 6)
            self.assertEqual(result["native_response_presentations"], 12)
            self.assertEqual(len(result["cases"]), 6)
            verify_state(trainer, origin)
            self.assertFalse(list((Path(directory) / "qualification").rglob("*.pt")))

    def test_failed_qualification_does_not_continue_or_mutate_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, features = tiny_trainer(Path(directory) / "trainer", count=128)
            origin = capture_state(trainer)
            path = Path(directory) / "qualification"
            with self.assertRaisesRegex(ValueError, "qualification zero differs"):
                qualify(trainer, [Path(row["path"]) for row in features], path,
                    dict(disabled=None, zero=add_fixed_gradient, padded_zero=padded_zero,
                         U=add_fixed_gradient, T=add_fixed_gradient))
            result = json.loads((path / "RESULT.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(result["cases"]), 3)
            self.assertEqual(result["optimizer_steps"], 3)
            self.assertTrue((path / "FAILED.json").exists())
            verify_state(trainer, origin)


if __name__ == "__main__":
    unittest.main()
