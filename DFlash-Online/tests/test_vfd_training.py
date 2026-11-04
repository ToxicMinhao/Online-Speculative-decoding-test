"""Native K+ equivalence, complete branch restoration and subset continuity."""

from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import torch

from experiments.vfd_second_pass import contract as C
from experiments.vfd_second_pass import training as T
from online_feedback.kplus import install_kplus
from online_feedback.rng import capture_rng, preserve_rng
from online_feedback.storage import save_torch
from experiments.ok_events_5_8.campaign import identity
import test_full_k_training as native_fixtures


class VFDContractTests(unittest.TestCase):
    def test_declared_physical_budget_and_per_arm_steps(self):
        self.assertEqual(
            C.physical_training(4),
            dict(optimizer_steps=1664, training_presentations=3328),
        )
        self.assertEqual(
            [C.logical_steps(arm, 1) for arm in C.ARMS], [128, 64, 96, 96, 96]
        )
        order = list(range(128))
        selected = list(reversed(order[::2]))
        self.assertEqual(C.validate_subset(order, selected, "PROGRESS64"), order[::2])
        for bad in (selected[:-1], [0] * 64, list(range(100, 164))):
            with self.assertRaises(ValueError):
                C.validate_subset(order, bad, "PROGRESS64")

    def test_native_boundary_rejects_changed_counters_moments_precision_and_rng(self):
        state, _ = native_fixtures.own_state("K", 1)
        recipe = deepcopy(state["backend"]["optimizer"]["recipe"])
        T.check_native_boundary(
            state, expected_steps=128, recipe=recipe, expected_cuda_rng_states=0
        )
        first_id = state["backend"]["optimizer"]["optimizer_state_dict"][
            "param_groups"
        ][0]["params"][0]
        mutations = [
            lambda value: value.update(presentations=255),
            lambda value: value["backend"]["optimizer"]["optimizer_state_dict"][
                "state"
            ][first_id].update(step=torch.tensor(0.0)),
            lambda value: value["backend"]["optimizer"]["optimizer_state_dict"][
                "state"
            ][first_id].pop("exp_avg"),
            lambda value: value["rng"].update(cuda=[torch.zeros(2, dtype=torch.uint8)]),
        ]
        for mutate in mutations:
            bad = deepcopy(state)
            mutate(bad)
            with self.assertRaises((ValueError, KeyError)):
                T.check_native_boundary(
                    bad, expected_steps=128, recipe=recipe, expected_cuda_rng_states=0
                )
        with self.assertRaises(ValueError):
            T.check_native_boundary(state, expected_steps=128, recipe=recipe)


class VFDTinyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("VFD_TINY_CPU_CHILD") == "1":
            native_fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("VFD_TINY_CPU_CHILD") == "1":
            native_fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get("VFD_TINY_CPU_CHILD") == "1":
            native_fixtures.FullKTinyIntegrationTests.setUp(self)

    def build(self, name):
        path = self.root / (name + ".yaml")
        path.write_text(self.original_config_text)
        trainer = T.VFDTrainer(
            path,
            self.root / name,
            original_config_text=self.original_config_text,
            expected_cuda_rng_states=0,
        )
        self.addCleanup(trainer.close)
        return trainer

    def assert_science_equal(self, before, after):
        for key in (
            "draft_state_dict",
            "backend",
            "rng",
            "global_step",
            "presentations",
        ):
            self.assertFalse(T.differences(before[key], after[key]), key)

    def test_native_equivalence_forks_subsets_and_continued_moments(self):
        if os.environ.get("VFD_TINY_CPU_CHILD") != "1":
            repository = Path(__file__).resolve().parents[1]
            environment = dict(
                os.environ, CUDA_VISIBLE_DEVICES="", VFD_TINY_CPU_CHILD="1"
            )
            environment["PYTHONPATH"] = os.pathsep.join(
                (
                    str(repository),
                    str(repository / "training"),
                    str(repository / "tests"),
                )
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "-v",
                    "test_vfd_training.VFDTinyIntegrationTests.test_native_equivalence_forks_subsets_and_continued_moments",
                ],
                cwd=repository,
                env=environment,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        trainer = self.build("full")
        initial = deepcopy(T.live_state(trainer))
        features = trainer.features(1, self.features)
        before = capture_rng()
        for bad in (
            [],
            [features[0]["record_id"]] * 2,
            ["foreign", features[0]["record_id"]],
            [features[0]["record_id"]],
        ):
            with self.assertRaises(ValueError):
                trainer.run_pass(features, event=1, pass_index=2, selected=bad)
            self.assertFalse(T.differences(before, capture_rng()))
            self.assertEqual((trainer.global_step, trainer.presentations), (0, 0))
        first = trainer.run_pass(features, event=1, pass_index=1)
        shared = deepcopy(T.live_state(trainer))
        shared_path = self.root / "shared.pt"
        committed = trainer.save_boundary(
            shared_path, arm="FULL", event=1, shared_fork=True
        )
        self.assertEqual(committed["identity"], identity(shared_path))
        self.assertFalse(
            shared_path.with_name(shared_path.name + ".candidate").exists()
        )

        # A decoded invalid successor must not replace the current recovery file.
        def corrupt_staging(path, payload):
            damaged = deepcopy(payload)
            damaged["continuation"]["arm"] = "INVALID"
            save_torch(path, damaged)

        with mock.patch.object(T, "save_torch", side_effect=corrupt_staging):
            with self.assertRaisesRegex(ValueError, "Persisted complete boundary"):
                trainer.save_boundary(
                    shared_path, arm="FULL", event=1, shared_fork=True
                )
        self.assertEqual(committed["identity"], identity(shared_path))
        candidate_path = shared_path.with_name(shared_path.name + ".candidate")
        self.assertTrue(candidate_path.exists())
        candidate_path.unlink()  # Disposable CPU test evidence, inside TemporaryDirectory.
        self.assertFalse(T.differences(shared["rng"], capture_rng()))
        second = trainer.run_pass(features, event=1, pass_index=2)
        full = deepcopy(T.live_state(trainer))
        self.assertEqual((full["global_step"], full["presentations"]), (128, 256))
        trainer.close()

        # Original generic trainer, native loss, native two-pass loop, same start.
        plain = T.SpecForgeEventTrainer(
            self.config,
            self.root / "plain",
            objective="osd_first_error",
            reduction="response_mean",
            ranking_lambda=0.0,
        )
        self.addCleanup(plain.close)
        install_kplus(plain.bundle.model)
        T.VFDTrainer._load_native(plain, initial)
        native_features = plain._prepare_event("event_001", self.features, 2)
        plain.bundle.model.begin_event()
        try:
            native_rows = plain._train_passes(native_features, 2)
        finally:
            plain.bundle.model.end_event()
        self.assert_science_equal(full, T.live_state(plain))
        for key in (
            "record_id",
            "metrics",
            "sampled_anchors",
            "optimizer_stepped",
            "optimizer_step_after",
        ):
            self.assertEqual(
                [row[key] for row in first["records"] + second["records"]],
                [row[key] for row in native_rows],
                key,
            )
        plain.close()

        candidate = self.build("candidate")
        for arm in C.ARMS:
            candidate.restore_boundary(shared_path, arm=arm, event=1, shared_fork=True)
            self.assertFalse(
                T.differences(
                    T.scientific_state(shared),
                    T.scientific_state(T.live_state(candidate)),
                )
            )
        # ONCE performs no second-pass call or permutation; persistence is RNG neutral.
        once_path = self.root / "once.pt"
        candidate.save_boundary(
            once_path, arm="ONCE", event=1, event_record=dict(event_id="event_001")
        )
        once = torch.load(once_path, map_location="cpu", weights_only=False)["native"]
        self.assert_science_equal(shared, once)
        candidate.restore_boundary(
            shared_path, arm="PROGRESS64", event=1, shared_fork=True
        )
        full_order = second["executed_order"]
        chosen = [feature["record_id"] for feature in features][::2]
        with preserve_rng():
            own_permutation = [
                features[index]["record_id"] for index in torch.randperm(128).tolist()
            ]
        subset = candidate.run_pass(
            features,
            event=1,
            pass_index=2,
            full_order=full_order,
            selected=list(reversed(chosen)),
        )
        self.assertEqual(subset["own_permutation"], own_permutation)
        self.assertEqual(
            subset["executed_order"], [item for item in full_order if item in chosen]
        )
        self.assertEqual(
            (subset["optimizer_steps"], subset["response_presentations"]), (32, 64)
        )
        self.assertEqual((candidate.global_step, candidate.presentations), (96, 192))
        self.assertEqual(len(subset["records"]), 64)
        self.assertTrue(
            all(len(row["sampled_anchors"]) == 1 for row in subset["records"])
        )
        self.assertTrue(
            all(
                value["step"].item() == 96
                for value in candidate.backend.optimizer.optimizer.state.values()
            )
        )
        boundary = self.root / "progress_event1.pt"
        candidate.save_boundary(
            boundary, arm="PROGRESS64", event=1, event_record=dict(event_id="event_001")
        )
        saved = torch.load(boundary, map_location="cpu", weights_only=False)["native"]
        with self.assertRaisesRegex(ValueError, "another policy"):
            candidate.restore_boundary(boundary, arm="RANDOM64", event=1)
        self.assert_science_equal(saved, T.live_state(candidate))

        next_paths = []
        for index, path in enumerate(self.features):
            raw = torch.load(path, weights_only=True)
            raw["record_id"] = f"event2_response_{index}"
            raw["label_binding"]["record_id"] = raw["record_id"]
            destination = self.root / f"event2_{index}.pt"
            torch.save(raw, destination)
            next_paths.append(destination)
        next_features = candidate.features(2, next_paths)
        next_order = list(reversed([feature["record_id"] for feature in next_features]))
        continuous = candidate.run_pass(
            next_features, event=2, pass_index=1, full_order=next_order
        )
        continued = deepcopy(T.live_state(candidate))
        self.assertEqual((candidate.global_step, candidate.presentations), (160, 320))
        candidate.close()
        restored = self.build("restored")
        restored.restore_boundary(boundary, arm="PROGRESS64", event=1)
        self.assert_science_equal(saved, T.live_state(restored))
        self.assertTrue(
            all(
                value["step"].item() == 96
                for value in restored.backend.optimizer.optimizer.state.values()
            )
        )
        replay = restored.run_pass(
            next_features, event=2, pass_index=1, full_order=next_order
        )
        self.assert_science_equal(continued, T.live_state(restored))
        for key in ("record_id", "metrics", "sampled_anchors"):
            self.assertEqual(
                [row[key] for row in continuous["records"]],
                [row[key] for row in replay["records"]],
                key,
            )
        self.assertTrue(
            all(
                value["step"].item() == 160
                for value in restored.backend.optimizer.optimizer.state.values()
            )
        )
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
