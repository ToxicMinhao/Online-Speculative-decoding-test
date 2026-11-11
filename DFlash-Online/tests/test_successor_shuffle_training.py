"""Original-state integrity and a real tiny CPU event-five continuation."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from experiments.successor_shuffle import training as T
from experiments.successor_shuffle.contract import expected_work
from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch
from test_role_loss_campaign_training import pilot_state
import test_full_k_training as fixtures

BINDING = "b" * 64


def continuation(arm, event):
    origin, old = pilot_state(4, "KPLUS", "gsm8k")
    state, _ = pilot_state(event, "KPLUS", "gsm8k")
    state["events"][:4] = deepcopy(origin["events"])
    state["algorithm_contract"] = T.continuation_contract(
        origin, binding=BINDING, arm=arm
    )
    if event == 5:
        receipt = state["events"][4]
        receipt.update(
            method=arm,
            dataset="gsm8k",
            successor_work=expected_work(arm),
            kplus_loss=None if arm == "K" else T.kplus_contract(),
        )
        if arm == "K":
            for row in receipt["records"]:
                sums = row["response_observation"]["sum_metrics"]
                for k in list(sums):
                    if k.startswith("kplus_"):
                        del sums[k]
    args = dict(
        origin=origin,
        binding=BINDING,
        arm=arm,
        dataset="gsm8k",
        config_text=state["config_text"],
        original_config_text=old["original_config_text"],
        completed_event=event,
        expected_recipe=origin["backend"]["optimizer"]["recipe"],
        expected_cuda_rng_states=0,
    )
    return origin, state, args


class SuccessorStateTests(unittest.TestCase):
    def test_original_boundaries_and_native_endpoints_preserve_rng_and_history(self):
        for arm, event in [
            ("K", 4),
            ("TRUE", 4),
            ("SHUFFLED", 4),
            ("K", 5),
            ("TRUE", 5),
        ]:
            origin, state, args = continuation(arm, event)
            before = deepcopy(state)
            rng = capture_rng()
            checked = T.validate_snapshot(state, **args)
            self.assertEqual(checked["global_step"], event * 128)
            self.assertFalse(T.differences(before, state))
            self.assertFalse(T.differences(rng, capture_rng()))

    def test_foreign_history_missing_adam_and_sixth_event_fail(self):
        _, state, args = continuation("TRUE", 5)
        for mutate in (
            lambda x: x["events"][0].update(method="K"),
            lambda x: x["backend"]["optimizer"]["optimizer_state_dict"][
                "state"
            ].clear(),
            lambda x: x["algorithm_contract"][T.CONTRACT_KEY].update(
                final_adam_step=768
            ),
        ):
            broken = deepcopy(state)
            mutate(broken)
            with self.assertRaises(ValueError):
                T.validate_snapshot(broken, **args)
        with self.assertRaises(ValueError):
            T.validate_snapshot(state, **dict(args, completed_event=6))

    def test_migration_reuses_complete_payload_without_reinitialization(self):
        origin, _, args = continuation("SHUFFLED", 4)
        state = T.migrate_origin(
            origin,
            binding=BINDING,
            arm="SHUFFLED",
            dataset="gsm8k",
            config_text=origin["config_text"],
            run_id="new",
            original_config_text=args["original_config_text"],
            expected_recipe=args["expected_recipe"],
            expected_cuda_rng_states=0,
        )
        for key in ("draft_state_dict", "backend", "rng", "events"):
            self.assertIs(state[key], origin[key])


class SuccessorTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("SUCCESSOR_TINY_CHILD") == "1":
            fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("SUCCESSOR_TINY_CHILD") == "1":
            fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get("SUCCESSOR_TINY_CHILD") == "1":
            fixtures.FullKTinyIntegrationTests.setUp(self)

    def test_tiny_complete_shuffled_event_records_all_assignments_and_restores_original(
        self,
    ):
        if os.environ.get("SUCCESSOR_TINY_CHILD") != "1":
            root = Path(__file__).resolve().parents[1]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", SUCCESSOR_TINY_CHILD="1")
            env["PYTHONPATH"] = os.pathsep.join(
                str(root / name) for name in (".", "tests", "training", "inference")
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "-v",
                    __name__ + ".SuccessorTinyTests." + self._testMethodName,
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        base = fixtures.FullKTinyIntegrationTests.build(self, "fixture_source")
        origin = deepcopy(T.native.live_state(base))
        fake, _ = pilot_state(4, "KPLUS", "gsm8k")
        origin.update(events=fake["events"], global_step=512, presentations=1024)
        origin["algorithm_contract"].pop(T.native.CONTRACT_KEY)
        origin["algorithm_contract"][T.historical.CONTRACT_KEY] = (
            T.historical.experiment_contract(fixtures.BINDING, "KPLUS", "gsm8k")
        )
        optimizer = origin["backend"]["optimizer"]["optimizer_state_dict"]
        for index, p in zip(
            optimizer["param_groups"][0]["params"], base.backend.optimizer.model_params
        ):
            optimizer["state"][index] = dict(
                step=torch.tensor(512.0),
                exp_avg=torch.zeros_like(p),
                exp_avg_sq=torch.zeros_like(p),
            )
        source = self.root / "origin.pt"
        save_torch(source, origin)
        export = self.root / "origin_export"
        export.mkdir()
        save_file(origin["draft_state_dict"], export / "model.safetensors")
        base.close()
        trainer = T.SuccessorShuffleTrainer(
            self.config,
            self.root / "shuffled",
            arm="SHUFFLED",
            binding=BINDING,
            original_config_text=self.original_config_text,
            origin_state=source,
            origin_export=export,
            expected_cuda_rng_states=0,
        )
        self.addCleanup(trainer.close)
        rng = capture_rng()
        trainer.install_shuffle()
        self.assertFalse(T.differences(rng, capture_rng()))
        result = trainer.train_event("event_005", self.features)
        self.assertEqual(trainer.global_step, 640)
        self.assertEqual(trainer.presentations, 1280)
        self.assertEqual(result["successor_work"], expected_work("SHUFFLED"))
        self.assertEqual(len(result["records"]), 256)
        self.assertTrue(
            all(
                r["response_observation"]["successor_shuffle"]["status"] == "passed"
                for r in result["records"]
            )
        )
        self.assertTrue(
            trainer.verify_boundary(result["checkpoint_path"])["export_exact"]
        )
        self.assertFalse(T.differences(torch.load(source, weights_only=False), origin))
        with self.assertRaises(ValueError):
            trainer.train_event("event_006", self.features)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
