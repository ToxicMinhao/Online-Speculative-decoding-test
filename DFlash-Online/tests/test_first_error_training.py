"""CPU end-to-end wiring for fresh C/K/A, response means and counted work."""

from copy import deepcopy
import unittest
from unittest import mock

import torch

from online_feedback import training as T
from online_feedback.objective import OSDTrainStrategy
from online_feedback.optimizer import DirectBF16AdamW
import test_osd_recipe as legacy
from test_first_error_objective import NEW_ARMS, CE_ARMS
from test_training_contract import trees_equal


class FirstErrorTrainerTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, "ranking_eos_token_ids", return_value=(28,))
        self.eos = patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), "CPU test initialized CUDA")

    def test_new_arms_start_fresh_and_reuse_the_ordinary_presentation_anchor_schedule(self):
        baseline = self.trainer("ordinary", objective="osd")
        initial = deepcopy(baseline.bundle.draft_model.state_dict())
        ordinary = baseline.train_event("e1", self.features[:2])
        for arm in NEW_ARMS:
            with self.subTest(arm=arm):
                trainer = self.trainer(arm, objective=arm)
                self.assertTrue(trees_equal(initial, trainer.bundle.draft_model.state_dict()))
                self.assertEqual((trainer.global_step, trainer.presentations), (0, 0))
                self.assertEqual(len(trainer.backend.optimizer.optimizer.state), 0)
                self.assertIsInstance(trainer.backend.optimizer, DirectBF16AdamW)
                self.assertEqual(trainer.bundle.model.loss_dtype, torch.bfloat16)
                if arm == "osd_first_error":
                    self.assertIsInstance(trainer.author_strategy, OSDTrainStrategy)
                with mock.patch.object(trainer.bundle.model, "_reference_feedback",
                                       side_effect=AssertionError("Unexpected reference forward")):
                    result = trainer.train_event("e1", self.features[:2], presentation_schedule=ordinary)
                self.assertEqual((result["response_presentations"], result["optimizer_steps"]), (4, 2))
                for key in ("record_id", "pass", "sampled_anchors", "optimizer_stepped", "learning_rate"):
                    self.assertEqual([row[key] for row in ordinary["records"]],
                                     [row[key] for row in result["records"]], key)
                self.assertIsNone(trainer.bundle.model.reference)
                self.assertIsNone(trainer._presentation_schedule)
                contract = self.saved(trainer)["algorithm_contract"]["online_target_objective"]
                self.assertEqual(contract["name"], arm)
                self.assertEqual(contract["loss_precision"], "bfloat16")
                self.assertEqual(contract["denominator_epsilon"], 1e-6)
                self.assertIsNone(contract["reference"])
                self.assertNotIn("ranking_lambda", contract)
                self.assertEqual(contract["main_loss"], "canonical_ce" if arm in CE_ARMS else "forward_kl")
                self.assertEqual(contract["eos_token_ids"], [28])
                for name in ("reference_setup_s", "reference_forward_s", "reference_head_s", "projection_s"):
                    self.assertEqual(result["objective_work"][name], 0)
                self.assertGreaterEqual(result["objective_work"]["selection_s"], 0)
                if arm in CE_ARMS:
                    self.assertEqual(result["objective_work"]["teacher_head_s"], 0)
                self.assertGreater(result["costs"]["training_s"], 0)

    def test_event_diagnostics_count_each_response_once_despite_accumulation_and_recomputation(self):
        for arm in NEW_ARMS:
            with self.subTest(arm=arm):
                trainer = self.trainer(arm, objective=arm)
                observed = []
                model = trainer.bundle.model
                original_forward = model.forward

                def record_original_forward(*args, **kwargs):
                    output = original_forward(*args, **kwargs)
                    observed.append({key: float(value) for key, value in output[2]["sum_metrics"].items()})
                    return output

                with mock.patch.object(model, "forward", side_effect=record_original_forward), \
                        mock.patch.object(model, "_chunk_loss", wraps=model._chunk_loss) as chunks:
                    result = trainer.train_event("e1", self.features[:2])
                self.assertEqual(len(observed), 4)
                # The 5/7 block fixtures each fit in one configured eight-block
                # vocabulary chunk. Backward must call it again, without new rows.
                self.assertGreater(chunks.call_count, len(observed))
                self.assertEqual(result["selection_totals_source"],
                                 "optimizer_boundary_accumulation_windows_once")
                totals = result["selection_totals"]
                self.assertEqual(totals["response_presentations"], 4)
                for name in totals:
                    expected = 4 if name == "response_presentations" else sum(row[name] for row in observed)
                    self.assertAlmostEqual(totals[name], expected, delta=max(1e-6, abs(expected) * 1e-7), msg=name)
                self.assertEqual([row["metrics_scope"] for row in result["records"]],
                                 ["presentation", "accumulation_window", "presentation", "accumulation_window"])
                naive_count = sum(row["metrics"]["available_position_occurrences"] for row in result["records"])
                self.assertGreater(naive_count, totals["available_position_occurrences"])
                if arm == "osd_ce":
                    self.assertEqual(totals["selected_position_occurrences"], totals["available_position_occurrences"])
                else:
                    self.assertLess(totals["selected_position_occurrences"], totals["available_position_occurrences"])

    def test_response_mean_adapter_keeps_each_response_equal_despite_different_selected_weights(self):
        for arm in NEW_ARMS:
            with self.subTest(arm=arm):
                trainer = self.trainer(arm, objective=arm)
                outputs = []
                trainer.bundle.model.begin_event()
                for path in self.features[:2]:
                    raw = torch.load(path, weights_only=True)
                    output = trainer.strategy.forward_loss(trainer._batch(raw))
                    outputs.append(output)
                    self.assertEqual(float(output.loss_terms[1]), 1)
                    torch.testing.assert_close(output.loss_terms[0], output.loss, rtol=0, atol=0)
                    self.assertEqual(float(output.sum_metrics["response_presentations"]), 1)
                    author_numerator, author_denominator = output.ratio_metrics["author_weighted_objective"]
                    torch.testing.assert_close(output.loss, author_numerator / author_denominator, rtol=0, atol=0)
                self.assertNotEqual(float(outputs[0].ratio_metrics["author_weighted_objective"][1]),
                                    float(outputs[1].ratio_metrics["author_weighted_objective"][1]))
                self.assertEqual(trainer.accumulation, 2)
                self.assertEqual(trainer.reduction, "response_mean")
                trainer.bundle.model.end_event()

    def test_all_new_arm_resume_contracts_reject_other_arms_before_mutating_weights(self):
        trained = self.trainer("trained", objective="osd_ce_first_error")
        trained.train_event("e1", self.features[:2])
        for arm in ("osd", "osd_ce", "osd_first_error", "osd_uniform", "osd_prefix", "osd_reference_margin"):
            trainer = self.trainer(arm, objective=arm)
            before = deepcopy(trainer.bundle.draft_model.state_dict())
            with self.assertRaisesRegex(ValueError, "Resume algorithm or schedule differs"):
                trainer._restore(trained.latest)
            self.assertTrue(trees_equal(before, trainer.bundle.draft_model.state_dict()))
            self.assertEqual(trainer.global_step, 0)

    def test_changed_retained_anchor_is_rejected_before_current_forward(self):
        ordinary = self.trainer("ordinary", objective="osd").train_event("e1", self.features[:2])
        wrong = deepcopy(ordinary)
        wrong["records"][0]["sampled_anchors"][0]["positions"][0][0] += 1
        for arm in NEW_ARMS:
            trainer = self.trainer(arm, objective=arm)
            with mock.patch.object(trainer.bundle.draft_model, "forward") as forward:
                with self.assertRaisesRegex(ValueError, "Sampled anchors differ"):
                    trainer.train_event("e1", self.features[:2], presentation_schedule=wrong)
            forward.assert_not_called()
            self.assertEqual((trainer.global_step, trainer.presentations), (0, 0))
            self.assertFalse(trainer.latest.exists())


if __name__ == "__main__":
    unittest.main()
