"""CPU event wiring: ordinary recipe, matched draws and isolated arm contracts."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from online_feedback import training as T
from online_feedback.objective import OSDTrainStrategy
from online_feedback.optimizer import DirectBF16AdamW
import test_osd_recipe as legacy
from test_training_contract import trees_equal


class RankingTrainerTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, "ranking_eos_token_ids", return_value=(28,))
        self.eos = patch.start()
        self.addCleanup(patch.stop)

    def test_both_ranking_arms_use_osd_recipe_and_retained_ordinary_schedule(self):
        baseline = self.trainer("ordinary", objective="osd")
        ordinary = baseline.train_event("e1", self.features)
        for arm in ("osd_uniform", "osd_prefix"):
            with self.subTest(arm=arm):
                trainer = self.trainer(arm, objective=arm)
                self.assertIsInstance(trainer.author_strategy, OSDTrainStrategy)
                self.assertIsInstance(trainer.backend.optimizer, DirectBF16AdamW)
                result = trainer.train_event("e1", self.features, presentation_schedule=ordinary)
                for key in ("record_id", "pass", "sampled_anchors", "optimizer_stepped", "learning_rate"):
                    self.assertEqual([r[key] for r in result["records"]],
                                     [r[key] for r in ordinary["records"]], key)
                self.assertEqual((result["response_presentations"], result["optimizer_steps"]), (16, 8))
                self.assertIsNone(trainer.bundle.model.reference)
                self.assertIsNone(trainer._presentation_schedule)
                contract = self.saved(trainer)["algorithm_contract"]["online_target_objective"]
                self.assertEqual(contract["name"], arm)
                self.assertEqual(contract["ranking_lambda"], 1.0)
                self.assertEqual(contract["eos_token_ids"], [28])
                self.assertEqual(contract["loss_precision"], "bfloat16")
                self.assertEqual(contract["teacher"], "frozen_target_predecessor_hidden")

    def test_zero_lambda_matches_ordinary_weights_optimizer_rng_and_exposure(self):
        baseline = self.trainer("ordinary", objective="osd")
        event = baseline.train_event("e1", self.features)
        expected = self.saved(baseline)
        for arm in ("osd_uniform", "osd_prefix"):
            trainer = self.trainer(arm, objective=arm, ranking_lambda=0)
            trainer.train_event("e1", self.features, presentation_schedule=event)
            actual = self.saved(trainer)
            for key in ("draft_state_dict", "backend", "rng", "global_step", "presentations"):
                self.assertTrue(trees_equal(expected[key], actual[key]), (arm, key))

    def test_changed_order_is_rejected_before_current_forward(self):
        ordinary = self.trainer("ordinary", objective="osd").train_event("e1", self.features)
        wrong = deepcopy(ordinary)
        wrong["records"][0], wrong["records"][1] = wrong["records"][1], wrong["records"][0]
        trainer = self.trainer("ranked", objective="osd_prefix")
        with mock.patch.object(trainer.core, "train_step", wraps=trainer.core.train_step) as forward:
            with self.assertRaisesRegex(ValueError, "Response ordering differs"):
                trainer.train_event("e1", self.features, presentation_schedule=wrong)
            forward.assert_not_called()
        self.assertEqual((trainer.global_step, trainer.presentations), (0, 0))
        self.assertTrue(trainer.failed)
        self.assertIsNone(trainer._presentation_schedule)

    def test_changed_anchor_is_rejected_before_draft_model_forward(self):
        ordinary = self.trainer("ordinary", objective="osd").train_event("e1", self.features)
        wrong = deepcopy(ordinary)
        wrong["records"][0]["sampled_anchors"][0]["positions"][0][0] += 1
        trainer = self.trainer("ranked", objective="osd_prefix")
        with mock.patch.object(trainer.bundle.draft_model, "forward", wraps=trainer.bundle.draft_model.forward) as forward:
            with self.assertRaisesRegex(ValueError, "Sampled anchors differ"):
                trainer.train_event("e1", self.features, presentation_schedule=wrong)
            forward.assert_not_called()
        self.assertEqual((trainer.global_step, trainer.presentations), (0, 0))
        self.assertFalse(trainer.latest.exists())

    def test_ranking_resume_contract_rejects_lambda_eos_and_other_arm(self):
        trained = self.trainer("trained", objective="osd_prefix")
        trained.train_event("e1", self.features)
        for label, options, eos in (("lambda", {"ranking_lambda": 0}, (28,)),
                                    ("eos", {}, (27, 28)),
                                    ("arm", {"objective": "osd_uniform"}, (28,))):
            self.eos.return_value = eos
            trainer = self.trainer(label, **{"objective": "osd_prefix", **options})
            before = deepcopy(trainer.bundle.draft_model.state_dict())
            with self.assertRaisesRegex(ValueError, "Resume algorithm or schedule differs"):
                trainer._restore(trained.latest)
            self.assertTrue(trees_equal(before, trainer.bundle.draft_model.state_dict()))
            self.assertEqual(trainer.global_step, 0)

    def test_relocated_feedback_is_rejected_before_reference_or_training(self):
        ordinary = self.trainer("ordinary", objective="osd").train_event("e1", self.features)
        copied = self.root / "renamed_feedback.pt"
        copied.write_bytes(self.features[0].read_bytes())
        altered_paths = [copied, *self.features[1:]]
        trainer = self.trainer("ranked", objective="osd_prefix")
        with mock.patch.object(trainer.bundle.model, "begin_event") as reference:
            with self.assertRaisesRegex(ValueError, "feature paths or metadata differ"):
                trainer.train_event("e1", altered_paths, presentation_schedule=ordinary)
            reference.assert_not_called()
        self.assertEqual((trainer.global_step, trainer.presentations), (0, 0))
        self.assertFalse((trainer.events_root / "e1").exists())


class RankingEOSMetadataTests(unittest.TestCase):
    def test_uses_serving_precedence_and_tokenizer_without_target_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"eos_token_id": 1}))
            (root / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 3]}))
            with mock.patch("transformers.AutoTokenizer.from_pretrained",
                            return_value=SimpleNamespace(eos_token_id=4)) as tokenizer:
                self.assertEqual(T.ranking_eos_token_ids(root), (2, 3, 4))
                tokenizer.assert_called_once_with(root, local_files_only=True)


if __name__ == "__main__":
    unittest.main()
