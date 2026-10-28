"""Reference-margin CPU event integration using the existing tiny author fixture."""
from copy import deepcopy
import unittest
from unittest import mock

from online_feedback import training as T
from online_feedback.objective import OSDTrainStrategy
from online_feedback.optimizer import DirectBF16AdamW
import test_osd_recipe as legacy
from test_training_contract import trees_equal


ARM = "osd_reference_margin"


class ReferenceMarginTrainerTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, "ranking_eos_token_ids", return_value=(28,))
        self.eos = patch.start()
        self.addCleanup(patch.stop)

    def test_default_arm_preserves_recipe_schedule_accumulation_and_logs_additive_diagnostics(self):
        ordinary_trainer = self.trainer("ordinary", objective="osd")
        ordinary = ordinary_trainer.train_event("e1", self.features)
        trainer = self.trainer("reference_margin", objective=ARM)
        self.assertIsInstance(trainer.author_strategy, OSDTrainStrategy)
        self.assertIsInstance(trainer.backend.optimizer, DirectBF16AdamW)
        self.assertEqual(trainer.bundle.model.ranking_lambda, 1.0)
        result = trainer.train_event("e1", self.features, presentation_schedule=ordinary)
        for key in ("record_id", "pass", "sampled_anchors", "optimizer_stepped", "learning_rate"):
            self.assertEqual([row[key] for row in result["records"]],
                             [row[key] for row in ordinary["records"]], key)
        self.assertEqual((result["response_presentations"], result["optimizer_steps"]), (16, 8))
        self.assertEqual(trainer.core.accumulation_remainder, 0)
        self.assertIsNone(trainer.bundle.model.reference)
        self.assertIsNone(trainer._presentation_schedule)
        contract = self.saved(trainer)["algorithm_contract"]["online_target_objective"]
        self.assertEqual(contract["name"], ARM)
        self.assertEqual(contract["ranking_lambda"], 1.0)
        self.assertEqual(contract["eos_token_ids"], [28])
        self.assertEqual(contract["loss_precision"], "bfloat16")
        self.assertEqual(contract["teacher"], "frozen_target_predecessor_hidden")
        for row in result["records"]:
            metrics = row["metrics"]
            for name in ("kl_loss_numerator", "auxiliary_loss_numerator", "loss_denominator",
                         "protected_active_positions", "protected_positions",
                         "reference_correct_positions", "reference_correct_current_incorrect_positions",
                         "protected_activation_rate", "reference_correct_to_incorrect_rate"):
                self.assertIn(name, metrics)
            self.assertGreater(metrics["loss_denominator"], 0)
            self.assertLessEqual(metrics["protected_active_positions"], metrics["protected_positions"])
            self.assertLessEqual(metrics["reference_correct_current_incorrect_positions"],
                                 metrics["reference_correct_positions"])

    def test_zero_lambda_matches_ordinary_complete_trainer_state_and_metrics_exactly(self):
        ordinary_trainer = self.trainer("ordinary", objective="osd")
        ordinary = ordinary_trainer.train_event("e1", self.features)
        expected = self.saved(ordinary_trainer)
        trainer = self.trainer("reference_margin_zero", objective=ARM, ranking_lambda=0)
        with mock.patch.object(trainer.bundle.model, "_reference_feedback", side_effect=AssertionError("unexpected reference work")):
            result = trainer.train_event("e1", self.features, presentation_schedule=ordinary)
        actual = self.saved(trainer)
        for key in ("draft_state_dict", "backend", "rng", "global_step", "presentations"):
            self.assertTrue(trees_equal(expected[key], actual[key]), key)
        for key in ("record_id", "pass", "sampled_anchors", "optimizer_stepped", "learning_rate", "metrics"):
            self.assertTrue(trees_equal([row[key] for row in ordinary["records"]],
                                       [row[key] for row in result["records"]]), key)
        self.assertEqual(result["objective_work"]["reference_setup_s"], 0)
        self.assertEqual(result["objective_work"]["reference_forward_s"], 0)
        self.assertEqual(result["objective_work"]["reference_head_s"], 0)

    def test_resume_keeps_reference_margin_optimizer_rng_schedule_and_next_event_exact(self):
        uninterrupted = self.trainer("uninterrupted", objective=ARM)
        uninterrupted.train_event("e1", self.features)
        checkpoint = self.root / "reference_margin_checkpoint.pt"
        checkpoint.write_bytes(uninterrupted.latest.read_bytes())
        expected_event = uninterrupted.train_event("e2", self.features)
        expected = self.saved(uninterrupted)
        resumed = self.trainer("resumed", objective=ARM, resume_state=checkpoint)
        actual_event = resumed.train_event("e2", self.features)
        actual = self.saved(resumed)
        for key in ("draft_state_dict", "backend", "rng", "global_step", "presentations"):
            self.assertTrue(trees_equal(expected[key], actual[key]), key)
        for key in ("record_id", "pass", "sampled_anchors", "optimizer_stepped", "learning_rate", "metrics"):
            self.assertTrue(trees_equal([row[key] for row in expected_event["records"]],
                                       [row[key] for row in actual_event["records"]]), key)

    def test_reference_margin_resume_identity_rejects_other_arms_lambda_and_eos(self):
        trained = self.trainer("trained", objective=ARM)
        trained.train_event("e1", self.features)
        alternatives = [
            ("ordinary", {"objective": "osd"}, (28,)),
            ("uniform", {"objective": "osd_uniform"}, (28,)),
            ("prefix", {"objective": "osd_prefix"}, (28,)),
            ("lambda", {"ranking_lambda": 0.0}, (28,)),
            ("eos", {}, (27, 28)),
        ]
        for label, options, eos in alternatives:
            with self.subTest(change=label):
                self.eos.return_value = eos
                trainer = self.trainer(label, **{"objective": ARM, **options})
                before = deepcopy(trainer.bundle.draft_model.state_dict())
                with self.assertRaisesRegex(ValueError, "Resume algorithm or schedule differs"):
                    trainer._restore(trained.latest)
                self.assertTrue(trees_equal(before, trainer.bundle.draft_model.state_dict()))
                self.assertEqual(trainer.global_step, 0)

    def test_changed_schedule_is_rejected_before_reference_or_current_forward(self):
        ordinary = self.trainer("ordinary", objective="osd").train_event("e1", self.features)
        wrong = deepcopy(ordinary)
        wrong["records"][0]["sampled_anchors"][0]["positions"][0][0] += 1
        trainer = self.trainer("reference_margin", objective=ARM)
        with mock.patch.object(trainer.bundle.model, "_reference_feedback") as reference, \
                mock.patch.object(trainer.bundle.draft_model, "forward") as current:
            with self.assertRaisesRegex(ValueError, "Sampled anchors differ"):
                trainer.train_event("e1", self.features, presentation_schedule=wrong)
        reference.assert_not_called()
        current.assert_not_called()
        self.assertEqual((trainer.global_step, trainer.presentations), (0, 0))
        self.assertFalse(trainer.latest.exists())
        self.assertIsNone(trainer.bundle.model.reference)


if __name__ == "__main__":
    unittest.main()
