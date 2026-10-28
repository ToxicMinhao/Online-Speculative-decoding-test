"""Receipt-bound main startup with tiny CPU models and no native serving."""
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.prefix_ranking import controlled_startup as startup_module
from experiments.prefix_ranking import run as campaign
from online_feedback import training as T
from online_feedback.qualify import assert_state_equal
from online_feedback.rng import capture_rng
from test_training_contract import trees_equal
import test_osd_recipe as legacy


TRIALS = (("warmup", "osd", 0.0), ("Oa", "osd", 0.0), ("Ob", "osd", 0.0),
          ("U0", "osd_uniform", 0.0), ("P0", "osd_prefix", 0.0),
          ("U1", "osd_uniform", 1.0), ("P1", "osd_prefix", 1.0))
COUNTS = dict(current_training_forward_completed=14, reference_training_forward_completed=4,
              detached_comparison_forward_completed=4, temporary_optimizer_step_completed=7,
              response_presentations=14, backward_completed=14)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


def qualification_paths():
    """Find retained run-local helpers in the workspace or deployed run tree."""
    worktree = Path(__file__).resolve().parents[1]
    roots = (worktree.parent.parent, worktree.parent / "runs")
    result = []
    for variable, directory in (
        ("DFLASH_QUALIFICATION_RUNNER", "dflash_prefix_ranking_controlled_qualification_20261026"),
        ("DFLASH_DIAGNOSTIC_RUNNER", "dflash_prefix_ranking_gpu_diagnostic_20261026"),
    ):
        candidates = ([Path(os.environ[variable])] if variable in os.environ else
                      [root / directory / "runner.py" for root in roots])
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise RuntimeError(f"Set {variable} to the retained run-local runner for this integration test")
        result.append(path)
    return tuple(result)


def synthetic_qualification(root, config, ordinary, pin, qualification_runner, diagnostic_runner):
    """Build small receipt fixtures; these are never claimed as GPU evidence."""
    receipt_path = root / "qualification/RESULT.json"
    seal_path = root / "SEAL.json"
    counts = dict(COUNTS)
    for key, value in COUNTS.items():
        if key.endswith("_completed"):
            counts[key.removesuffix("_completed") + "_attempted"] = value
    trials = []
    for label, objective, ranking_lambda in TRIALS:
        trials.append(dict(label=label, objective=objective, ranking_lambda=ranking_lambda,
                           status="discarded_warmup" if label == "warmup" else "passed",
                           records=[{"record_id": "fixture_0"}, {"record_id": "fixture_1"}],
                           frozen_target_version_layout_unchanged=True,
                           exact_baseline_compatibility=label in ("Ob", "U0", "P0"),
                           reference_rng_isolation=bool(ranking_lambda),
                           prefix_checks=[{"prefix_equal": True, "reference_rng_equal": True}] * 2
                           if ranking_lambda else [],
                           ranking_gradient_checks=[{"max_gradient_error": 0.0}] * 2
                           if ranking_lambda else []))
        names = ["initial_vs_pinned"]
        if label in ("Ob", "U0", "P0"):
            names += ["initial_vs_Oa", "after_backward_0_vs_Oa", "after_backward_1_vs_Oa",
                      "pre_optimizer_vs_Oa", "final_vs_Oa", "records_vs_Oa"]
        if ranking_lambda:
            names.append("rng_vs_Oa")
        for name in names:
            write_json(receipt_path.parent / label / f"{name}.json", {"exact_equal": True})
    result = dict(status="passed", trials=trials, work_counts=counts, wall_s=1.0,
                  generations=0, main_optimizer_steps=0, main_campaign_started=False,
                  baseline_tolerance="exact equality", config=str(config),
                  ordinary_receipt=str(ordinary), pinned_initial=str(pin),
                  diagnostic_runner=str(diagnostic_runner),
                  config_sha256=digest(config), ordinary_receipt_sha256=digest(ordinary),
                  pinned_initial_sha256=digest(pin), diagnostic_runner_sha256=digest(diagnostic_runner),
                  plan=dict(temporary_optimizer_steps=7, response_presentations=14,
                            current_training_forwards=14, reference_training_forwards=4,
                            detached_comparison_forwards=4, backwards=14, generations=0,
                            main_optimizer_steps=0, teacher_model_forwards=0))
    write_json(receipt_path, result)
    write_json(seal_path, dict(diagnostic_sha256={"runner.py": digest(qualification_runner)},
                               dependency_sha256={str(diagnostic_runner): digest(diagnostic_runner),
                                                  str(pin): digest(pin)}))
    return (config, ordinary, receipt_path, seal_path, qualification_runner, diagnostic_runner, pin)


class QualificationReceiptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = []
        for name in ("config.yaml", "ordinary.json", "pin.pt", "qualified_runner.py", "diagnostic_runner.py"):
            path = self.root / name
            path.write_bytes(name.encode())
            self.paths.append(path)
        self.args = synthetic_qualification(self.root, *self.paths)

    def test_complete_bound_receipt_is_accepted_without_loading_a_trainer(self):
        with mock.patch.object(T, "SpecForgeEventTrainer", side_effect=AssertionError("Receipt verification loaded a model")):
            result = startup_module.verify_qualification(*self.args)
        self.assertIsInstance(result, dict)
        self.assertFalse(torch.cuda.is_initialized())

    def test_failed_incomplete_or_unbound_receipt_is_rejected(self):
        path = self.args[2]
        original = json.loads(path.read_text())
        invalid = []
        for key, value in (("status", "failed"), ("main_campaign_started", True),
                           ("config_sha256", "0" * 64), ("ordinary_receipt_sha256", "0" * 64),
                           ("pinned_initial_sha256", "0" * 64), ("diagnostic_runner_sha256", "0" * 64)):
            row = copy.deepcopy(original)
            row[key] = value
            invalid.append((key, row))
        row = copy.deepcopy(original)
        row["work_counts"]["backward_completed"] = 13
        invalid.append(("short_work", row))
        row = copy.deepcopy(original)
        row["trials"][-1]["status"] = "failed"
        invalid.append(("failed_active_trial", row))
        for name, row in invalid:
            with self.subTest(name=name):
                write_json(path, row)
                with self.assertRaises((AssertionError, RuntimeError, ValueError)):
                    startup_module.verify_qualification(*self.args)
        write_json(path, original)

    def test_each_exact_gate_and_runner_source_remains_bound(self):
        gate = self.args[2].parent / "P0/final_vs_Oa.json"
        write_json(gate, {"exact_equal": False})
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            startup_module.verify_qualification(*self.args)
        write_json(gate, {"exact_equal": True})
        self.args[4].write_bytes(b"changed qualification runner")
        with self.assertRaises((AssertionError, RuntimeError, ValueError)):
            startup_module.verify_qualification(*self.args)

    def test_campaign_reuses_receipt_then_warms_once_before_u_p_and_serving(self):
        config, ordinary, qualification, seal, qualified_runner, diagnostic, pin = self.args
        write_json(ordinary, {"checkpoint_path": "retained_O"})
        manifest = self.root / "manifest.json"
        write_json(manifest, {"eval_requests": []})
        destination = self.root / "campaign"
        args = SimpleNamespace(config=config, ordinary_receipt=ordinary, qualification_receipt=qualification,
                               qualification_seal=seal, qualification_runner=qualified_runner,
                               diagnostic_runner=diagnostic, pinned_initial=pin, manifest=manifest,
                               ordinary_measurements=self.root / "unused_measurements", output=destination,
                               worker_python="unused_python", serving_gpu="0")
        order = []

        def warmup(root):
            order.append("warmup")
            return {"status": "passed", "work_counts": {"temporary_optimizer_step_completed": 1}}

        startup = SimpleNamespace(receipt={"status": "passed", "reused": True}, warmup=mock.Mock(side_effect=warmup))

        def train(config, ordinary, root, arm, *, startup):
            order.append(arm)
            return str(root / arm / "export"), {"event_wall_s": 0.0}

        backend = SimpleNamespace(loading_s=0.0, close=mock.Mock())

        def load_backend(*args, **kwargs):
            self.assertEqual(order, ["warmup", "U", "P"])
            order.append("load_native")
            return backend

        def evaluate(*args, **kwargs):
            order.append("evaluate")
            return {"status": "completed"}

        model = SimpleNamespace(draft_checkpoint_path="released_F", target_model_path="frozen_target")
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(startup_module, "ControlledStartup", return_value=startup))
            stack.enter_context(mock.patch.object(campaign, "admit_storage"))
            stack.enter_context(mock.patch.object(campaign, "train_arm", side_effect=train))
            stack.enter_context(mock.patch("online_feedback.recipe.load_event_config", return_value=(SimpleNamespace(model=model), None)))
            stack.enter_context(mock.patch("online_feedback.qualify_sglang.check_recipe"))
            stack.enter_context(mock.patch("experiments.osd_one_event.protocol.validate_manifest"))
            stack.enter_context(mock.patch("experiments.prefix_ranking.protocol.load_ordinary_counts", return_value={}))
            stack.enter_context(mock.patch("online_feedback.sglang_backend.SGLangDFlashBackend", side_effect=load_backend))
            stack.enter_context(mock.patch("experiments.prefix_ranking.evaluate.evaluate", side_effect=evaluate))
            qualifier = stack.enter_context(mock.patch("experiments.prefix_ranking.qualify.run_qualification",
                                                      side_effect=AssertionError("Main reran qualification")))
            campaign.run(args)
        qualifier.assert_not_called()
        startup.warmup.assert_called_once_with(destination / "process_warmup")
        backend.close.assert_called_once_with()
        self.assertEqual(order, ["warmup", "U", "P", "load_native", "evaluate"])
        status = json.loads((destination / "STATUS.json").read_text())
        self.assertEqual(status["status"], "completed")
        self.assertEqual((status["main_optimizer_steps"], status["main_response_presentations"]), (256, 512))
        self.assertTrue(status["qualification"]["reused"])


class ControlledStartupIntegrationTests(unittest.TestCase):
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
        self.pinned = dict(draft=copy.deepcopy(ordinary.bundle.draft_model.state_dict()),
                           optimizer=copy.deepcopy(ordinary.backend.optimizer.state_dict()), rng=capture_rng(),
                           global_step=ordinary.global_step, presentations=ordinary.presentations,
                           accumulation_remainder=ordinary.core.accumulation_remainder,
                           training_modes={name: module.training for name, module in ordinary.bundle.model.named_modules()})
        self.pin_path = self.root / "initial.pt"
        torch.save(self.pinned, self.pin_path)
        self.ordinary = ordinary.train_event("event_001", self.features)
        self.ordinary_path = ordinary.events_root / "event_001/RESULT.json"
        ordinary.close()
        qualification_runner, diagnostic_runner = qualification_paths()
        self.args = synthetic_qualification(self.root / "qualified_fixture", self.config,
                                            self.ordinary_path, self.pin_path,
                                            qualification_runner, diagnostic_runner)

    def test_warmup_then_full_u_and_p_restore_independent_initial_state_and_original_schedule(self):
        startup = startup_module.ControlledStartup(*self.args)
        destination = self.root / "main"
        destination.mkdir()
        initial = {}
        original_prepare = startup.prepare
        original_constructor = T.SpecForgeEventTrainer

        def disturbed_constructor(*args, **kwargs):
            trainer = original_constructor(*args, **kwargs)
            random.random()
            np.random.random()
            torch.rand(9)
            return trainer

        def prepare(trainer, path):
            result = original_prepare(trainer, path)
            initial[Path(path).parent.name] = startup.driver.scientific_state(trainer)
            return result

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(T, "SpecForgeEventTrainer", side_effect=disturbed_constructor))
            stack.enter_context(mock.patch.object(startup_module, "SpecForgeEventTrainer", side_effect=disturbed_constructor))
            stack.enter_context(mock.patch.object(startup, "prepare", side_effect=prepare))
            stack.enter_context(mock.patch("experiments.prefix_ranking.qualify.run_qualification",
                                           side_effect=AssertionError("Main startup reran qualification")))
            stack.enter_context(mock.patch("torch.cuda.init"))
            stack.enter_context(mock.patch("torch.cuda.reset_peak_memory_stats"))
            stack.enter_context(mock.patch("torch.cuda.max_memory_allocated", return_value=0))
            stack.enter_context(mock.patch("torch.cuda.max_memory_reserved", return_value=0))
            warmup = startup.warmup(destination / "process_warmup")
            checkpoints = {}
            for arm in ("U", "P"):
                checkpoints[arm], _ = campaign.train_arm(self.config, self.ordinary, destination, arm,
                                                        startup=startup)
        self.assertEqual(warmup["status"], "passed")
        self.assertEqual(warmup["work_counts"]["backward_completed"], 2)
        self.assertEqual(warmup["work_counts"]["temporary_optimizer_step_completed"], 1)
        self.assertEqual(warmup["work_counts"]["response_presentations"], 2)
        for arm in ("U", "P"):
            assert_state_equal(initial[arm], self.pinned, arm + ".fresh_initial")
            receipt = json.loads((destination / arm / "events/event_001/RESULT.json").read_text())
            self.assertEqual((receipt["optimizer_steps"], receipt["response_presentations"]), (128, 256))
            for field in ("record_id", "pass", "sampled_anchors"):
                self.assertEqual([row[field] for row in receipt["records"]],
                                 [row[field] for row in self.ordinary["records"]], (arm, field))
            self.assertTrue(Path(checkpoints[arm]).exists())
        self.assertNotEqual(checkpoints["U"], checkpoints["P"])
        self.assertEqual(list((destination / "process_warmup").rglob("export")), [])
        self.assertEqual(list((destination / "process_warmup").rglob("training_state.pt")), [])
        self.assertFalse(torch.cuda.is_initialized())

    def test_prepare_rejects_changed_fresh_weights_without_overwriting_them(self):
        startup = startup_module.ControlledStartup(*self.args)
        trainer = self.trainer("invalid_fresh", objective="osd_uniform", ranking_lambda=1.0)
        with self.assertRaisesRegex(RuntimeError, "single discarded process warmup"):
            startup.prepare(trainer, self.root / "unwarmed_initialization")
        startup.warmup(self.root / "process_warmup")
        with self.assertRaisesRegex(RuntimeError, "only once"):
            startup.warmup(self.root / "second_warmup")
        with torch.no_grad():
            next(trainer.bundle.draft_model.parameters()).add_(0.125)
        changed = copy.deepcopy(trainer.bundle.draft_model.state_dict())
        with self.assertRaises(AssertionError):
            startup.prepare(trainer, self.root / "invalid_fresh/initialization")
        comparison = json.loads((self.root / "invalid_fresh/initialization/initial_vs_pinned.json").read_text())
        self.assertFalse(comparison["exact_equal"])
        self.assertTrue(trees_equal(changed, trainer.bundle.draft_model.state_dict()))
        self.assertEqual((trainer.global_step, trainer.presentations), (0, 0))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
