"""Tiny eager CPU qualification fixtures; no native GPU or scientific evidence."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import qualify_r as Q
import test_osd_recipe as legacy
from online_feedback import training as T
from online_feedback.qualify import assert_state_equal
from online_feedback.rng import capture_rng


class QualificationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        for patch in (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU fixture initialized CUDA")),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, "ranking_eos_token_ids", return_value=(28,))
        patch.start()
        self.addCleanup(patch.stop)
        for index in range(len(self.features), 128):
            raw = torch.load(self.features[index % 2], weights_only=True)
            raw["record_id"] = f"response_{index}"
            raw["label_binding"]["record_id"] = raw["record_id"]
            path = self.root / f"feedback_{index}.pt"
            torch.save(raw, path)
            self.features.append(path)
        ordinary = self.trainer("retained_ordinary", objective="osd")
        self.pin = self.root / "retained_initial.pt"
        torch.save(Q.controlled().scientific_state(ordinary), self.pin)
        ordinary.train_event("e1", self.features)
        self.receipt = ordinary.events_root / "e1/RESULT.json"
        ordinary.close()

    def run_driver(self, destination, factory=None):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                T.SpecForgeEventTrainer, "_export", side_effect=AssertionError("Qualification must not export")))
            if factory is not None:
                stack.enter_context(mock.patch.object(Q, "SpecForgeEventTrainer", side_effect=factory))
            return Q.run_qualification(self.config, self.receipt, self.pin, destination)

    @staticmethod
    def trial_label(path):
        return next(part for part in Path(path).parts if part in {row[0] for row in Q.TRIALS})

    def test_five_windows_exact_zero_lambda_active_R_and_fresh_main(self):
        factory = Q.SpecForgeEventTrainer
        constructed = []
        pin_hash = hashlib.sha256(self.pin.read_bytes()).hexdigest()

        def construct(config, root, **kwargs):
            trainer = factory(config, root, **kwargs)
            constructed.append(self.trial_label(root))
            for _ in constructed:
                random.random()
                np.random.random()
                torch.rand(3)
            return trainer

        destination = self.root / "qualification"
        result = self.run_driver(destination, construct)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(constructed, [row[0] for row in Q.TRIALS])
        self.assertEqual(result["work_counts"]["temporary_optimizer_step_completed"], 5)
        self.assertEqual(result["work_counts"]["backward_completed"], 10)
        self.assertEqual(result["work_counts"]["response_presentations"], 10)
        self.assertEqual(result["work_counts"]["current_training_forward_completed"], 10)
        self.assertEqual(result["work_counts"]["reference_training_forward_completed"], 4)
        self.assertEqual(result["work_counts"]["detached_comparison_forward_completed"], 4)
        self.assertEqual(result["work_counts"]["detached_logit_gradient_completed"], 20)
        ordinary = json.loads(self.receipt.read_text())
        for trial in result["trials"]:
            count = 4 if trial["label"] == "R1" else 2
            self.assertEqual([row["record_id"] for row in trial["records"]],
                             [row["record_id"] for row in ordinary["records"][:count]])
            self.assertEqual([row["sampled_anchors"] for row in trial["records"]],
                             [row["sampled_anchors"] for row in ordinary["records"][:count]])
        r = result["trials"][-1]
        self.assertEqual(len(r["prefix_checks"]), 4)
        self.assertEqual(len(r["ranking_gradient_checks"]), 4)
        self.assertEqual(r["post_update_presentations"], 2)
        self.assertTrue(all(row["equality_value_and_gradient_zero"] for row in r["ranking_gradient_checks"]))
        for stage in ("pre_optimizer_step_1", "pre_optimizer_step_2", "after_backward_2", "after_backward_3"):
            self.assertTrue((destination / "R1" / (stage + ".pt")).is_file())
        for filename in ("initial_vs_Oa.json", "after_backward_0_vs_Oa.json", "after_backward_1_vs_Oa.json",
                         "pre_optimizer_vs_Oa.json", "final_vs_Oa.json", "records_vs_Oa.json"):
            self.assertTrue(json.loads((destination / "R0" / filename).read_text())["exact_equal"])
        self.assertEqual(list(destination.rglob("export")), [])
        self.assertEqual(list(destination.rglob("training_state.pt")), [])
        self.assertEqual(hashlib.sha256(self.pin.read_bytes()).hexdigest(), pin_hash)

        # Main preparation must restore all RNGs after constructor disturbance,
        # without creating a reference or inheriting a qualification update.
        main = self.trainer("main", objective="osd_reference_margin")
        random.random()
        np.random.random()
        torch.rand(4)
        initialization = Q.prepare_main(main, self.pin, self.root / "main_initialization")
        json.dumps(initialization)  # The campaign stores this compact receipt.
        self.assertGreaterEqual(initialization["wall_s"], 0)
        self.assertTrue(initialization["exact_pinned_initial_state"])
        self.assertFalse(initialization["reference_created"])
        self.assertFalse(initialization["inherited_warmup_state"])
        pin = torch.load(self.pin, weights_only=False)
        assert_state_equal(capture_rng(), pin["rng"], "main_complete_rng")
        self.assertIsNone(main.bundle.model.reference)
        self.assertEqual((main.global_step, main.presentations), (0, 0))
        self.assertEqual(main.backend.optimizer.state_dict()["optimizer_state_dict"]["state"], {})
        main.bundle.model.begin_event()
        Q.controlled().validate_reference(main, pin)
        main.bundle.model.end_event()
        main.close()
        self.assertFalse(torch.cuda.is_initialized())

    def test_post_step_R0_mismatch_is_saved_and_stops_R1(self):
        factory = Q.SpecForgeEventTrainer
        constructed = []

        def construct(config, root, **kwargs):
            trainer = factory(config, root, **kwargs)
            label = self.trial_label(root)
            constructed.append(label)
            if label == "R0":
                original = trainer.backend.optimizer.step

                def corrupt(*args, **kwargs):
                    result = original(*args, **kwargs)
                    with torch.no_grad():
                        next(trainer.bundle.draft_model.parameters()).add_(0.125)
                    return result

                trainer.backend.optimizer.step = corrupt
            return trainer

        root = self.root / "failure"
        with self.assertRaisesRegex(AssertionError, "Exact state gate failed"):
            self.run_driver(root, construct)
        self.assertEqual(constructed, ["warmup", "Oa", "R0"])
        result = json.loads((root / "RESULT.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["work_counts"]["temporary_optimizer_step_completed"], 3)
        self.assertTrue((root / "R0/final.pt").is_file())
        self.assertTrue((root / "R0/failure_state.pt").is_file())
        self.assertFalse((root / "R1").exists())

    def test_wrong_reference_values_stop_before_R_current_forward(self):
        factory = Q.SpecForgeEventTrainer

        def construct(config, root, **kwargs):
            trainer = factory(config, root, **kwargs)
            if self.trial_label(root) == "R1":
                original = trainer.bundle.model._reference_feedback

                def corrupt(*args, **kwargs):
                    prefix, values, correct = original(*args, **kwargs)
                    return prefix, values + 0.125, correct

                trainer.bundle.model._reference_feedback = corrupt
            return trainer

        root = self.root / "reference_failure"
        with self.assertRaisesRegex(AssertionError, "reference_own_rival_values"):
            self.run_driver(root, construct)
        result = json.loads((root / "RESULT.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["work_counts"]["temporary_optimizer_step_completed"], 3)
        self.assertEqual(result["work_counts"]["current_training_forward_completed"], 6)
        self.assertTrue((root / "R1/failure_state.pt").is_file())


class PrimitiveTests(unittest.TestCase):
    def test_budget_and_detached_gradient_checks(self):
        weight_bytes = 1234567
        self.assertEqual(Q.storage_budget(weight_bytes)["required_bytes"], 38 * weight_bytes + 10 * 2**30)
        with tempfile.TemporaryDirectory() as directory:
            log = Q.WorkLog(directory)
            logits = torch.tensor([[3., 2., 1.], [1., 2., 3.]], dtype=torch.bfloat16)
            labels = torch.tensor([0, 1])
            references = Q.softplus_oracle(logits, labels)
            result = Q.check_reference_gradients(logits, labels, references, log)
            self.assertTrue(result["equality_value_and_gradient_zero"])
            self.assertTrue(result["synthetic_changed_rivals_checked"])
            self.assertEqual(log.counts["detached_logit_gradient_completed"], 5)
        self.assertFalse(torch.cuda.is_initialized())

    def test_prepare_main_rejects_wrong_arm_before_state_loading(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            for objective, ranking_lambda in (("osd", 1.), ("osd_reference_margin", 0.)):
                with self.subTest(objective=objective, ranking_lambda=ranking_lambda):
                    with self.assertRaisesRegex(AssertionError, "fresh R with lambda one"):
                        Q.prepare_main(SimpleNamespace(objective=objective, ranking_lambda=ranking_lambda),
                                       "missing", Path(directory) / "not_created")
                    self.assertFalse((Path(directory) / "not_created").exists())


if __name__ == "__main__":
    unittest.main()
