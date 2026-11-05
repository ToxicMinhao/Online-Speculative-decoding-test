"""CPU integration of the bounded initialization qualification, not GPU proof.

The tiny fixture replaces loading/distributed wrapping/export only. Sampling,
losses, accumulation, BF16 AdamW, reference creation and the trial driver are real.
"""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest import mock

import numpy as np
import torch

import test_osd_recipe as legacy
from online_feedback import training as T
from online_feedback.qualify import assert_state_equal
from online_feedback.rng import capture_rng

import runner


LABELS = ("warmup", "Oa", "Ob", "U0", "P0", "U1", "P1")
STAGES = ("initial", "after_backward_0", "after_backward_1", "pre_optimizer", "final")


class ControlledQualificationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        # The upstream checkpoint backend consults accelerator availability even
        # for a CPU fixture. Prevent incidental CUDA RNG initialization on a
        # GPU-equipped test host; the real qualification does not use these mocks.
        for patch in (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU fixture initialized CUDA")),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        legacy.OSDRecipeTests.setUp(self)
        eos = mock.patch.object(T, "ranking_eos_token_ids", return_value=(28,))
        eos.start()
        self.addCleanup(eos.stop)
        for index in range(len(self.features), 128):
            raw = torch.load(self.features[index % 2], weights_only=True)
            raw["record_id"] = f"response_{index}"
            raw["label_binding"]["record_id"] = raw["record_id"]
            path = self.root / f"feedback_{index}.pt"
            torch.save(raw, path)
            self.features.append(path)
        ordinary = self.trainer("retained_ordinary", objective="osd")
        self.pin = self.root / "retained_initial.pt"
        torch.save(runner.scientific_state(ordinary), self.pin)
        ordinary.train_event("e1", self.features)
        self.receipt = ordinary.events_root / "e1/RESULT.json"
        ordinary.close()

    def run_controlled(self, destination, *, factory=None):
        with ExitStack() as stack:
            stack.enter_context(mock.patch("torch.cuda.reset_peak_memory_stats"))
            stack.enter_context(mock.patch("torch.cuda.max_memory_allocated", return_value=0))
            stack.enter_context(mock.patch("torch.cuda.max_memory_reserved", return_value=0))
            stack.enter_context(mock.patch.object(
                T.SpecForgeEventTrainer, "_export", side_effect=AssertionError("Qualification must not export")))
            if factory is not None:
                stack.enter_context(mock.patch.object(runner, "SpecForgeEventTrainer", side_effect=factory))
            return runner.run_qualification(self.config, self.receipt, self.pin, destination)

    @staticmethod
    def trial_label(output_root):
        return next(part for part in Path(output_root).parts if part in LABELS)

    def test_all_seven_windows_restore_pinned_rng_and_discard_warmup_state(self):
        factory = runner.SpecForgeEventTrainer
        constructed = []
        reference_presence = []
        disturbed_rng = []
        pin_hash = hashlib.sha256(self.pin.read_bytes()).hexdigest()
        receipt_hash = hashlib.sha256(self.receipt.read_bytes()).hexdigest()

        def construct(config, output_root, **kwargs):
            trainer = factory(config, output_root, **kwargs)
            label = self.trial_label(output_root)
            constructed.append(label)
            # Simulate variable post-construction library initialization. This
            # changes all available CPU RNGs, not only Python's state.
            for _ in range(len(constructed)):
                random.random()
                np.random.random()
                torch.rand(3)
            disturbed_rng.append(capture_rng())
            original_begin = trainer.bundle.model.begin_event

            def begin_event():
                result = original_begin()
                reference_presence.append((label, trainer.bundle.model.reference is not None))
                return result

            trainer.bundle.model.begin_event = begin_event
            return trainer

        destination = self.root / "qualification"
        result = self.run_controlled(destination, factory=construct)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(constructed, list(LABELS))
        self.assertEqual([row["label"] for row in result["trials"]], list(LABELS))
        self.assertEqual(reference_presence, [(label, label in ("U1", "P1")) for label in LABELS])
        self.assertEqual(hashlib.sha256(self.pin.read_bytes()).hexdigest(), pin_hash)
        self.assertEqual(hashlib.sha256(self.receipt.read_bytes()).hexdigest(), receipt_hash)
        expected_counts = {
            "backward_completed": 14,
            "temporary_optimizer_step_completed": 7,
            "current_training_forward_completed": 14,
            "reference_training_forward_completed": 4,
            "detached_comparison_forward_completed": 4,
            "response_presentations": 14,
        }
        for key, expected in expected_counts.items():
            self.assertEqual(result["work_counts"][key], expected, key)
        pin = torch.load(self.pin, weights_only=False)
        ordinary = json.loads(self.receipt.read_text())
        for label, trial in zip(LABELS, result["trials"]):
            for stage in STAGES:
                self.assertTrue((destination / label / f"{stage}.pt").is_file(), (label, stage))
            initial = torch.load(destination / label / "initial.pt", weights_only=False)
            for key in ("draft", "optimizer", "rng", "global_step", "presentations",
                        "accumulation_remainder", "training_modes"):
                assert_state_equal(initial[key], pin[key], f"{label}.pinned.{key}")
            self.assertEqual([row["record_id"] for row in trial["records"]],
                             [row["record_id"] for row in ordinary["records"][:2]])
            self.assertEqual([row["sampled_anchors"] for row in trial["records"]],
                             [row["sampled_anchors"] for row in ordinary["records"][:2]])
            if label in ("U1", "P1"):
                self.assertEqual(len(trial["prefix_checks"]), 2)
                self.assertEqual(len(trial["ranking_gradient_checks"]), 2)
            else:
                self.assertEqual(trial["prefix_checks"], [])
                self.assertEqual(trial["ranking_gradient_checks"], [])
        # Advancing the constructor RNG really created the condition the reset
        # must repair; merely restoring a fixed seed after building would fail.
        self.assertTrue(all(not torch.equal(state["torch"], pin["rng"]["torch"])
                            for state in disturbed_rng))
        warmup_final = torch.load(destination / "warmup/final.pt", weights_only=False)
        self.assertFalse(runner.compare_states_compact(warmup_final["draft"], pin["draft"])["exact_equal"])
        self.assertEqual(list(destination.rglob("export")), [])
        self.assertEqual(list(destination.rglob("training_state.pt")), [])
        self.assertFalse(torch.cuda.is_initialized())

    def test_post_step_mismatch_is_retained_before_exact_gate_stops_later_trials(self):
        factory = runner.SpecForgeEventTrainer
        constructed = []

        def construct(config, output_root, **kwargs):
            trainer = factory(config, output_root, **kwargs)
            label = self.trial_label(output_root)
            constructed.append(label)
            if label == "U0":
                original_step = trainer.backend.optimizer.step

                def corrupt_after_step(*args, **step_kwargs):
                    result = original_step(*args, **step_kwargs)
                    with torch.no_grad():
                        next(trainer.bundle.draft_model.parameters()).add_(0.125)
                    return result

                trainer.backend.optimizer.step = corrupt_after_step
            return trainer

        destination = self.root / "post_step_failure"
        with self.assertRaises(AssertionError):
            self.run_controlled(destination, factory=construct)
        result = json.loads((destination / "RESULT.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(constructed, ["warmup", "Oa", "Ob", "U0"])
        self.assertEqual(result["trials"][-1]["status"], "failed")
        for stage in STAGES:
            self.assertTrue((destination / "U0" / f"{stage}.pt").is_file(), stage)
        self.assertEqual(result["work_counts"]["temporary_optimizer_step_completed"], 4)
        self.assertEqual(result["work_counts"]["backward_completed"], 8)
        self.assertEqual(result["work_counts"].get("reference_training_forward_completed", 0), 0)
        self.assertFalse((destination / "P0").exists())
        expected = torch.load(destination / "Oa/final.pt", weights_only=False)
        actual = torch.load(destination / "U0/final.pt", weights_only=False)
        self.assertFalse(runner.compare_states_compact(expected["draft"], actual["draft"])["exact_equal"])

    def test_fresh_weight_mismatch_is_not_hidden_by_loading_the_pin(self):
        factory = runner.SpecForgeEventTrainer
        constructed = []

        def construct(config, output_root, **kwargs):
            trainer = factory(config, output_root, **kwargs)
            constructed.append(self.trial_label(output_root))
            with torch.no_grad():
                next(trainer.bundle.draft_model.parameters()).add_(0.125)
            return trainer

        destination = self.root / "initial_failure"
        with self.assertRaises(AssertionError):
            self.run_controlled(destination, factory=construct)
        result = json.loads((destination / "RESULT.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(constructed, ["warmup"])
        self.assertTrue((destination / "warmup/initial.pt").is_file())
        self.assertFalse((destination / "warmup/after_backward_0.pt").exists())
        self.assertEqual(result["work_counts"].get("backward_completed", 0), 0)
        self.assertEqual(result["work_counts"].get("temporary_optimizer_step_completed", 0), 0)


class AccountingTests(unittest.TestCase):
    def test_peak_budget_includes_all_retained_stages_and_reserve(self):
        weight_bytes = 1234567
        result = runner.storage_budget(weight_bytes)
        self.assertEqual(result["required_bytes"], 56 * weight_bytes + 10 * 1024 ** 3)

    def test_compact_difference_keeps_mismatches_and_finite_gate_rejects_nonfinite(self):
        left = {"same": torch.ones(3), "changed": torch.tensor([1.0, 2.0]), "step": 1}
        right = {"same": torch.ones(3), "changed": torch.tensor([1.0, 3.0]), "step": 2}
        result = runner.compare_states_compact(left, right)
        self.assertFalse(result["exact_equal"])
        self.assertNotIn("same", result["tensor_differences"])
        self.assertIn("changed", result["tensor_differences"])
        self.assertTrue(result["metadata_differences"])
        self.assertFalse(runner.compare_states_compact({}, {"empty": {}})["exact_equal"])
        self.assertFalse(runner.compare_states_compact({"x": [1]}, {"x": (1,)})["exact_equal"])
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                state = {"weights": torch.tensor([value])}
                with self.assertRaisesRegex(AssertionError, "Nonfinite state"):
                    runner.assert_finite(state)


if __name__ == "__main__":
    unittest.main()
