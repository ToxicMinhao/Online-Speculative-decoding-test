"""Read-only continuation contracts and one tiny CPU event-five integration."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from experiments.survival_head import training as T
from experiments.survival_head.contract import expected_work
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch
from online_feedback.survival_head import ScalarLinearHead, survival_head_contract
from test_role_loss_campaign_training import pilot_state
import test_full_k_training as fixtures


BINDING = "b" * 64


def head_state(arm, steps):
    head = ScalarLinearHead(4, device="cpu")
    optimizer = DirectBF16AdamW(head)
    if steps:
        head(torch.ones(2, 4, dtype=torch.bfloat16)).float().sum().backward()
        optimizer.step()
        for state in optimizer.optimizer.state.values():
            state["step"].fill_(steps)
    return dict(contract=survival_head_contract(arm), head=head.state_dict(),
                optimizer=optimizer.state_dict(), steps=steps)


def continuation(root, arm="SURVIVAL", event=5):
    origin, original_args = pilot_state(4, "KPLUS", "gsm8k")
    state, _ = pilot_state(event, "KPLUS", "gsm8k")
    state["events"][:4] = deepcopy(origin["events"])
    state["algorithm_contract"] = T.continuation_contract(origin, binding=BINDING,
                                                           arm=arm, dataset="gsm8k")
    if event == 5:
        receipt = state["events"][4]
        head_receipt = None
        if arm != "KPLUS":
            path = root / (arm + "_head.pt")
            save_torch(path, head_state(arm, 128))
            head_receipt = dict(contract=survival_head_contract(arm), steps=128,
                                state_path=str(path), state_identity=T.identity(path))
        receipt.update(method=arm, dataset="gsm8k", kplus_loss=T.kplus_contract(),
                       survival_head=head_receipt, survival_work=expected_work(arm))
        for row in receipt["records"]:
            observation = row["response_observation"]
            observation["loss"] = 1.25 if arm != "KPLUS" else 1.0
            if arm != "KPLUS":
                observation["sum_metrics"].update(head_response_count=1, head_valid_proposals=2,
                    head_bce_numerator=0.5, head_response_bce_sum=0.25, head_native_kplus_loss_sum=1.0)
                if row["optimizer_stepped"]:
                    row["metrics"].update(head_optimizer_steps=1, head_grad_norm=0.1)
    args = dict(origin=origin, binding=BINDING, arm=arm, dataset="gsm8k",
        config_text=state["config_text"], original_config_text=original_args["original_config_text"],
        completed_event=event, expected_recipe=origin["backend"]["optimizer"]["recipe"],
        expected_cuda_rng_states=0)
    return origin, state, args


class SurvivalTrainingStateTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_all_three_arms_and_both_boundaries_preserve_origin_rng_and_history(self):
        for arm in T.ARMS:
            for event in (4, 5):
                origin, state, args = continuation(self.root, arm, event)
                original, current, rng = deepcopy(origin), deepcopy(state), capture_rng()
                receipt = T.validate_snapshot(state, **args)
                self.assertEqual(receipt["global_step"], event * 128)
                self.assertEqual(receipt["maximum_adam_step"], 640)
                self.assertFalse(T.differences(origin, original))
                self.assertFalse(T.differences(state, current))
                self.assertFalse(T.differences(capture_rng(), rng))

    def test_origin_migration_reuses_scientific_payload_without_changes(self):
        origin, _, args = continuation(self.root, event=4)
        state = T.migrate_origin(origin, binding=BINDING, arm="SURVIVAL", dataset="gsm8k",
            config_text=origin["config_text"], run_id="new-survival",
            original_config_text=args["original_config_text"], expected_recipe=args["expected_recipe"],
            expected_cuda_rng_states=0)
        for name in ("draft_state_dict", "backend", "rng", "events"):
            self.assertIs(state[name], origin[name])
        self.assertEqual(state["global_step"], 512)
        self.assertEqual(state["presentations"], 1024)
        self.assertNotIn("survival_head", state)

    def test_foreign_origin_changed_history_extra_event_and_lost_moments_fail(self):
        origin, state, args = continuation(self.root)
        with self.assertRaises(ValueError):
            T.validate_snapshot(state, **dict(args, completed_event=6))
        with self.assertRaises(ValueError):
            T.validate_origin(origin, dataset="mathqa", original_config_text=args["original_config_text"])
        with self.assertRaises(ValueError):
            T.validate_origin(state, dataset="gsm8k", original_config_text=args["original_config_text"])
        for mutate in (
            lambda item: item["events"][0].update(method="LOCAL"),
            lambda item: item["algorithm_contract"][T.CONTRACT_KEY].update(final_adam_step=768),
            lambda item: item["backend"]["optimizer"]["optimizer_state_dict"]["state"].clear(),
        ):
            broken = deepcopy(state)
            mutate(broken)
            with self.assertRaises(ValueError):
                T.validate_snapshot(broken, **args)

    def test_head_payload_identity_steps_precision_and_recipe_fail_closed(self):
        _, state, args = continuation(self.root)
        head = state["events"][4]["survival_head"]
        Path(head["state_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            T.validate_snapshot(state, **args)
        good = head_state("SURVIVAL", 128)
        for mutate in (
            lambda item: item.update(steps=127),
            lambda item: item["head"].update(weight=item["head"]["weight"].float()),
            lambda item: item["optimizer"]["optimizer_state_dict"]["state"].clear(),
            lambda item: item["optimizer"]["recipe"]["optimizer_options"].update(weight_decay=0),
            lambda item: item["optimizer"]["optimizer_state_dict"]["state"][0]["exp_avg_sq"].fill_(float("nan")),
        ):
            broken = deepcopy(good)
            mutate(broken)
            with self.assertRaises(ValueError):
                T.validate_head_state(broken, arm="SURVIVAL", expected_steps=128,
                                      native_recipe=args["expected_recipe"])

    def test_auxiliary_coefficient_and_step_observations_are_validated(self):
        _, state, args = continuation(self.root)
        for mutate in (
            lambda item: item["events"][4]["records"][0]["response_observation"].update(loss=1.5),
            lambda item: item["events"][4]["records"][1]["metrics"].update(head_optimizer_steps=2),
            lambda item: item["events"][4]["survival_work"].update(response_backwards=512),
        ):
            broken = deepcopy(state)
            mutate(broken)
            with self.assertRaises(ValueError):
                T.validate_snapshot(broken, **args)

    def test_active_training_requires_install_and_sixth_event_is_forbidden(self):
        trainer = object.__new__(T.SurvivalHeadTrainer)
        trainer.events = [{}] * 4
        trainer.arm, trainer.head_installed = "LOCAL", False
        with self.assertRaisesRegex(ValueError, "Install the head"):
            trainer._prepare_event("event_005", [], 2)
        trainer.events.append({})
        with self.assertRaisesRegex(ValueError, "stops after event five"):
            trainer._prepare_event("event_006", [], 2)


class SurvivalTrainingTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("SURVIVAL_TINY_CHILD") == "1":
            fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("SURVIVAL_TINY_CHILD") == "1":
            fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get("SURVIVAL_TINY_CHILD") == "1":
            fixtures.FullKTinyIntegrationTests.setUp(self)

    def test_tiny_event_five_saves_separate_head_and_restores_both_states(self):
        if os.environ.get("SURVIVAL_TINY_CHILD") != "1":
            root = Path(__file__).resolve().parents[1]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", SURVIVAL_TINY_CHILD="1")
            env["PYTHONPATH"] = os.pathsep.join(str(root / name) for name in (".", "tests", "training", "inference"))
            result = subprocess.run([sys.executable, "-m", "unittest", "-v",
                __name__ + ".SurvivalTrainingTinyTests." + self._testMethodName],
                cwd=root, env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return

        # This is a synthetic fixture with real five-layer tiny-model shapes;
        # it does not represent a replay of any historical experiment.
        base = fixtures.FullKTinyIntegrationTests.build(self, "fixture_source")
        origin = deepcopy(T.native.live_state(base))
        fake, _ = pilot_state(4, "KPLUS", "gsm8k")
        origin.update(events=fake["events"], global_step=512, presentations=1024)
        origin["algorithm_contract"].pop(T.native.CONTRACT_KEY)
        origin["algorithm_contract"][T.historical.CONTRACT_KEY] = T.historical.experiment_contract(
            fixtures.BINDING, "KPLUS", "gsm8k")
        optimizer = origin["backend"]["optimizer"]["optimizer_state_dict"]
        for index, parameter in zip(optimizer["param_groups"][0]["params"], base.backend.optimizer.model_params):
            optimizer["state"][index] = dict(step=torch.tensor(512.0),
                exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
        origin_file = self.root / "synthetic_origin.pt"
        save_torch(origin_file, origin)
        export = self.root / "origin_export"
        export.mkdir()
        save_file(origin["draft_state_dict"], export / "model.safetensors")
        base.close()
        options = dict(arm="SURVIVAL", binding=BINDING, original_config_text=self.original_config_text,
            origin_state=origin_file, origin_export=export, expected_cuda_rng_states=0)
        trainer = T.SurvivalHeadTrainer(self.config, self.root / "survival", **options)
        self.addCleanup(trainer.close)
        self.assertFalse(hasattr(trainer.bundle.model, "survival_head_extension"))
        before = capture_rng()
        self.assertTrue(trainer.install_head())
        self.assertFalse(T.differences(before, capture_rng()))
        result = trainer.train_event("event_005", self.features)
        checked = trainer.verify_boundary(result["export_path"])
        self.assertTrue(checked["head_live_exact"])
        self.assertEqual(result["survival_work"], expected_work("SURVIVAL"))
        self.assertEqual(result["survival_head"]["steps"], 128)
        self.assertEqual(trainer.global_step, 640)
        self.assertEqual(trainer.presentations, 1280)
        self.assertEqual(len(trainer.window_records), 128)
        saved = trainer.latest
        trainer.close()
        restored = T.SurvivalHeadTrainer(self.config, self.root / "restored", **options,
                                        resume_state=saved, completed_event=5)
        self.addCleanup(restored.close)
        restored.install_head()
        self.assertEqual(restored.core.extension.steps, 128)
        with self.assertRaisesRegex(ValueError, "stops after event five"):
            restored.train_event("event_006", self.features)
        self.assertFalse(T.differences(torch.load(origin_file, weights_only=False), origin))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
