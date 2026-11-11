"""CPU checks for inherited state/configuration and no-grad exposure arithmetic."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from experiments.second_follower import training as T
from experiments.second_follower.contract import expected_work
from online_feedback.rng import capture_rng, restore_rng
from online_feedback.second_follower import reduce_second_follower
from online_feedback.storage import save_torch, write_json
from safetensors.torch import save_file
from test_role_loss_campaign_training import pilot_state
from test_second_follower import fixture, objective_terms
import test_full_k_training as fixtures

BINDING = "b" * 64


def continuation(arm, event):
    origin, original_arguments = pilot_state(4, "KPLUS", "gsm8k")
    state, _ = pilot_state(event, "KPLUS", "gsm8k")
    state["events"][:4] = deepcopy(origin["events"])
    state["algorithm_contract"] = T.continuation_contract(
        origin, binding=BINDING, arm=arm
    )
    if event == 5:
        state["events"][4].update(
            method=arm,
            dataset="gsm8k",
            second_follower_work=expected_work(arm),
            kplus_loss=T.kplus_contract(),
        )
        if arm == "SECOND":
            for row in state["events"][4]["records"]:
                row["response_observation"]["sum_metrics"]["second_follower_disabled"] = 0.
    arguments = dict(
        origin=origin,
        binding=BINDING,
        arm=arm,
        dataset="gsm8k",
        config_text=state["config_text"],
        original_config_text=original_arguments["original_config_text"],
        completed_event=event,
        expected_recipe=origin["backend"]["optimizer"]["recipe"],
        expected_cuda_rng_states=0,
    )
    return origin, state, arguments


class CampaignContractTests(unittest.TestCase):
    def setUp(self):
        rng = capture_rng()
        self.addCleanup(restore_rng, rng)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_parent_and_endpoint_preserve_history_rng_and_optimizer_contract(self):
        for arm in ("NATIVE", "SECOND"):
            for event in (4, 5):
                with self.subTest(arm=arm, event=event):
                    _, state, arguments = continuation(arm, event)
                    before, rng = deepcopy(state), capture_rng()
                    result = T.validate_snapshot(state, **arguments)
                    self.assertEqual(result["global_step"], 128 * event)
                    self.assertEqual(state["horizon"], 7424)
                    self.assertFalse(T.differences(before, state))
                    self.assertFalse(T.differences(rng, capture_rng()))

    def test_migration_reuses_complete_parent_and_does_not_reset_scientific_state(self):
        origin, _, arguments = continuation("SECOND", 4)
        migrated = T.migrate_origin(
            origin, binding=BINDING, arm="SECOND", dataset="gsm8k",
            config_text=origin["config_text"], run_id="second_fixture",
            original_config_text=arguments["original_config_text"],
            expected_recipe=arguments["expected_recipe"], expected_cuda_rng_states=0,
        )
        for key in ("draft_state_dict", "backend", "rng", "events"):
            self.assertIs(migrated[key], origin[key])
        self.assertEqual(migrated["global_step"], 512)
        self.assertEqual(migrated["presentations"], 1024)
        self.assertNotIn(T.CONTRACT_KEY, origin["algorithm_contract"])
        self.assertEqual(migrated["algorithm_contract"][T.CONTRACT_KEY]["relative_transfer_cap"], .125)

    def test_changed_history_empty_moments_new_horizon_or_extra_event_rejected(self):
        _, state, arguments = continuation("SECOND", 5)
        for mutate in (
            lambda value: value["events"][0].update(method="foreign"),
            lambda value: value["backend"]["optimizer"]["optimizer_state_dict"]["state"].clear(),
            lambda value: value["algorithm_contract"][T.CONTRACT_KEY].update(final_adam_step=768),
            lambda value: value["events"][4]["second_follower_work"].update(optimizer_steps=129),
            lambda value: value["events"][4]["records"][0]["response_observation"]["sum_metrics"].update(second_follower_disabled=1.),
            lambda value: value.update(horizon=640),
        ):
            broken = deepcopy(state)
            mutate(broken)
            with self.assertRaises(ValueError):
                T.validate_snapshot(broken, **arguments)
        with self.assertRaises(ValueError):
            T.validate_snapshot(state, **dict(arguments, completed_event=6))

    def test_inherited_config_allows_identity_changes_only(self):
        _, state, arguments = continuation("SECOND", 4)
        original = arguments["original_config_text"]
        config = yaml.safe_load(state["config_text"])
        config.update(run_id="second", output_dir="/new/output")
        T.native.validate_config_text(yaml.safe_dump(config), "K", original)
        for section, key, changed in (
            ("training", "num_anchors", 256),
            ("training", "learning_rate", 1e-4),
            ("training", "seed", 18),
            ("online_feedback", "max_events", 5),
            ("online_feedback", "weight_decay", 0.),
        ):
            bad = deepcopy(config)
            bad[section][key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                T.native.validate_config_text(yaml.safe_dump(bad), "K", original)

    def test_no_grad_exposure_keeps_same_classes_transfer_and_loss_without_rng_draw(self):
        proposals, labels, weights = fixture()
        rows = torch.linspace(.01, 3., labels.numel(), dtype=torch.bfloat16)
        rows = rows.reshape_as(weights).requires_grad_()
        rng = capture_rng()
        native_loss, native_metrics, _ = reduce_second_follower(
            objective_terms(rows, proposals, labels, weights)
        )
        with torch.no_grad():
            probe_loss, probe_metrics, _ = reduce_second_follower(
                objective_terms(rows, proposals, labels, weights)
            )
        self.assertTrue(torch.equal(native_loss, probe_loss))
        self.assertTrue(native_loss.requires_grad)
        self.assertFalse(probe_loss.requires_grad)
        self.assertGreater(float(probe_metrics["second_follower_transferred_weight"]), 0)
        self.assertEqual(native_metrics.keys(), probe_metrics.keys())
        for field in native_metrics:
            self.assertTrue(torch.equal(native_metrics[field], probe_metrics[field]), field)
        self.assertIsNone(rows.grad)
        self.assertFalse(T.differences(rng, capture_rng()))


class TinyCampaignTests(unittest.TestCase):
    """Exercise the real trainer adapter on the inherited small CPU model fixture."""

    @classmethod
    def setUpClass(cls):
        if os.environ.get("SECOND_FOLLOWER_TINY_CHILD") == "1":
            fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("SECOND_FOLLOWER_TINY_CHILD") == "1":
            fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get("SECOND_FOLLOWER_TINY_CHILD") == "1":
            fixtures.FullKTinyIntegrationTests.setUp(self)

    def test_one_complete_candidate_event_preserves_parent_and_stops_at640(self):
        if os.environ.get("SECOND_FOLLOWER_TINY_CHILD") != "1":
            root = Path(__file__).resolve().parents[1]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", SECOND_FOLLOWER_TINY_CHILD="1")
            env["PYTHONPATH"] = os.pathsep.join(str(root / p) for p in (".", "tests", "training", "inference"))
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "-v", __name__ + ".TinyCampaignTests." + self._testMethodName],
                cwd=root, env=env, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        base = fixtures.FullKTinyIntegrationTests.build(self, "parent_fixture")
        origin = deepcopy(T.native.live_state(base))
        old, _ = pilot_state(4, "KPLUS", "gsm8k")
        origin.update(events=old["events"], global_step=512, presentations=1024)
        origin["algorithm_contract"].pop(T.native.CONTRACT_KEY)
        origin["algorithm_contract"][T.historical.CONTRACT_KEY] = T.historical.experiment_contract(
            fixtures.BINDING, "KPLUS", "gsm8k"
        )
        optimizer = origin["backend"]["optimizer"]["optimizer_state_dict"]
        for index, parameter in zip(optimizer["param_groups"][0]["params"], base.backend.optimizer.model_params):
            optimizer["state"][index] = dict(
                step=torch.tensor(512.), exp_avg=torch.zeros_like(parameter),
                exp_avg_sq=torch.zeros_like(parameter),
            )
        source = self.root / "origin.pt"
        save_torch(source, origin)
        export = self.root / "origin_export"
        export.mkdir()
        save_file(origin["draft_state_dict"], export / "model.safetensors")
        base.close()
        trainer = T.SecondFollowerTrainer(
            self.config, self.root / "second", arm="SECOND", binding=BINDING,
            original_config_text=self.original_config_text,
            origin_state=source, origin_export=export, expected_cuda_rng_states=0,
        )
        self.addCleanup(trainer.close)
        rng = capture_rng()
        trainer.install_second()
        self.assertFalse(T.differences(rng, capture_rng()))
        result = trainer.train_event("event_005", self.features)
        self.assertEqual((trainer.global_step, trainer.presentations), (640, 1280))
        self.assertEqual(result["second_follower_work"], expected_work("SECOND"))
        self.assertEqual(len(result["records"]), 256)
        for row in result["records"]:
            sums = row["response_observation"]["sum_metrics"]
            self.assertEqual(sums["second_follower_disabled"], 0.)
            self.assertIn("second_follower_transferred_weight", sums)
        self.assertTrue(trainer.verify_boundary(result["checkpoint_path"])["export_exact"])
        self.assertFalse(T.differences(torch.load(source, weights_only=False), origin))
        with self.assertRaises(ValueError):
            trainer.train_event("event_006", self.features)
        self.assertFalse(torch.cuda.is_initialized())


class RuntimeCloseoutTests(unittest.TestCase):
    def test_inactive_event_skips_serving_but_requires_successful_postflight(self):
        from experiments.second_follower import runtime as R

        for fail_postflight in (False, True):
            with self.subTest(fail_postflight=fail_postflight), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                export = root / "export"
                export.mkdir()
                (export / "model.safetensors").write_bytes(b"fixture weights")
                plan = dict(
                    root=str(root), execution_source=dict(commit="fixture"),
                    checkpoints=dict(NATIVE=dict(
                        checkpoint_path="/native", weights_identity=dict(sha256="native", bytes=1),
                    )),
                )
                write_json(root / "sealed/PLAN.json", plan)
                write_json(root / "qualification/exposure/RESULT.json", dict(first_active_window=0))
                for case in ("NATIVE", "ZERO", "ACTIVE"):
                    write_json(root / f"qualification/{case}/RESULT.json", dict(
                        status="completed", finite=dict(status="passed"),
                        exact=dict(exact_equal=True), gradient_changes=dict(changed=1),
                        weight_changes=dict(changed=1), transferred_weight=1.,
                    ))
                write_json(root / "training/SECOND/RESULT.json", dict(
                    checkpoint_path=str(export), weights_identity=dict(sha256="second", bytes=2),
                    records=[dict(response_observation=dict(sum_metrics=dict(second_follower_transferred_weight=0.)))],
                ))
                postflights = []

                def checked(_plan, *, postflight=False):
                    postflights.append(postflight)
                    if postflight and fail_postflight:
                        raise ValueError("fixture postflight failure")
                    return dict(status="passed")

                with patch.object(R, "preflight", side_effect=checked), patch.object(
                    R, "launch", return_value=dict(status="completed")
                ) as launched:
                    if fail_postflight:
                        with self.assertRaisesRegex(ValueError, "fixture postflight"):
                            R.run(root)
                    else:
                        R.run(root)
                self.assertEqual(postflights, [False, True])
                self.assertEqual(launched.call_count, 5)
                modules = [call.args[1] for call in launched.call_args_list]
                self.assertNotIn("experiments.second_follower.runtime", modules)
                self.assertNotIn("experiments.second_follower.matched_start", modules)
                status = R.read(root / "STATUS.json")
                self.assertEqual(status["status"], "failed" if fail_postflight else "completed_inactive")


if __name__ == "__main__":
    unittest.main()
