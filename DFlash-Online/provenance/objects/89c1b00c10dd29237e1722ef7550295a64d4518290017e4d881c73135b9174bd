"""CPU execution integration tests; fixtures never qualify native GPU behavior."""

from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch

from online_feedback.storage import write_json
from experiments.first_error.campaign import QUALIFICATION_GATES
from test_osd_one_event import request


def qualification_receipt():
    return {"scope": "first_error_OCKA", "status": "passed", "wall_s": 1.,
            "checks": {name: True for name in QUALIFICATION_GATES}}


def load_campaign():
    """Load orchestration with the separate native qualifier safely stubbed."""
    qualifier = ModuleType("qualify_first_error")
    qualifier.__file__ = __file__
    qualifier.run_qualification = mock.Mock(return_value=qualification_receipt())
    qualifier.prepare_main = mock.Mock(return_value={"wall_s": 0., "exact_pinned_initial_state": True})
    qualifier.save_failure = mock.Mock()
    spec = importlib.util.spec_from_file_location(
        "first_error_execution_campaign_for_tests", Path(__file__).with_name("campaign.py"))
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"qualify_first_error": qualifier}):
        spec.loader.exec_module(module)
    return module


def fixture_inputs(root):
    training = []
    for index in range(128):
        row = request(index)
        row.update(request_id=f"train_{index}", prompt_ids=[1, 10000 + index])
        training.append(row)
    evaluation = []
    for index in range(64):
        row = request(index)
        row["prompt_ids"] = [1, 20000 + index]
        evaluation.append(row)
    manifest = {"train_requests": training, "eval_requests": evaluation,
                "policy": {"train_context_limit": 2272, "serving_context_limit": 4096}}
    features = [{"path": str(root / f"feedback_{index}.pt"), "record_id": row["request_id"]}
                for index, row in enumerate(training)]
    records = []
    for pass_index in range(2):
        for index, row in enumerate(training):
            records.append({"record_id": row["request_id"], "pass": pass_index,
                            "optimizer_stepped": bool(index % 2),
                            "optimizer_step_after": pass_index * 64 + (index + 1) // 2,
                            "sampled_anchors": [{"positions": [[2, 3, 4, 5]],
                                                 "keep_mask": [[True, True, True, True]]}]})
    ordinary = {"status": "completed", "objective": "osd", "checkpoint_path": "/retained/O",
                "features": features, "records": records, "passes": 2, "start_step": 0,
                "end_step": 128, "optimizer_steps": 128, "response_presentations": 256}
    config = root / "config.yaml"
    config.write_text("CPU fixture: model loading is replaced\n")
    manifest_path, ordinary_path = root / "manifest.json", root / "ordinary.json"
    write_json(manifest_path, manifest)
    write_json(ordinary_path, ordinary)
    args = SimpleNamespace(config=config, manifest=manifest_path, ordinary_receipt=ordinary_path,
                           ordinary_measurements=root / "measurements.jsonl", output=root / "campaign",
                           worker_python="/unused/python", serving_gpu="0",
                           pinned_initial=root / "initial.pt", controlled_runner=root / "controlled.py",
                           diagnostic_runner=root / "diagnostic.py")
    for path in (args.pinned_initial, args.controlled_runner, args.diagnostic_runner):
        path.write_bytes(path.name.encode())
    return args, manifest, ordinary


class ExecutionCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.campaign = load_campaign()
        self.args, self.manifest, self.ordinary = fixture_inputs(self.root)
        self.addCleanup(self.assert_cuda_unused)

    def assert_cuda_unused(self):
        self.assertFalse(torch.cuda.is_initialized(), "CPU fixture initialized CUDA")

    def bound_qualification(self):
        result = qualification_receipt()
        result.update(generations=0, main_optimizer_steps=0, work_counts={
            "response_presentations": 14, "temporary_optimizer_step_completed": 7,
            "temporary_optimizer_step_attempted": 7, "backward_completed": 14,
            "backward_attempted": 14, "current_training_forward_completed": 14,
            "current_training_forward_attempted": 14,
            "reference_training_forward_completed": 0, "reference_training_forward_attempted": 0,
        })
        result["input_sha256"] = {str(path): self.campaign.digest(path) for path in (
            self.args.config, self.args.ordinary_receipt, self.args.pinned_initial,
            self.args.controlled_runner, self.args.diagnostic_runner,
            self.campaign.qualify_first_error.__file__)}
        return result

    def status(self):
        return json.loads((self.args.output / "STATUS.json").read_text())

    def dependencies(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("No CUDA in CPU test")))
        load = stack.enter_context(mock.patch("online_feedback.recipe.load_event_config", return_value=(
            SimpleNamespace(model=SimpleNamespace(draft_checkpoint_path="released_DFlash", target_model_path="target")), None)))
        stack.enter_context(mock.patch("online_feedback.qualify_sglang.check_recipe", return_value={"passed": True}))
        stack.enter_context(mock.patch.object(self.campaign, "load_ordinary_counts", return_value={
            row["request_id"]: {"F": 5, "O": 2} for row in self.manifest["eval_requests"]}))
        order, backend = [], mock.Mock(loading_s=.5)

        def qualify(*args, **kwargs):
            order.append("qualify")
            return self.bound_qualification()

        def train(config, ordinary, root, arm, *, startup):
            self.assertEqual(ordinary, self.ordinary)
            self.assertEqual(config, self.args.config)
            self.assertIn(arm, ("C", "K", "A"))
            startup.prepare("fresh_" + arm, Path(root) / arm / "initialization")
            write_json(Path(root) / f"{arm}_CLOSED.json", {
                "global_step": 128, "presentations": 256, "completed_events": ["event_001"],
                "accumulation_remainder": 0, "closed": True})
            order.append(arm + "_closed")
            return "/fresh/" + arm, {"event_wall_s": 1., "new_feedback_generations": 0}

        def native(*args, **kwargs):
            self.assertEqual(order[-3:], ["C_closed", "K_closed", "A_closed"])
            for arm in ("C", "K", "A"):
                self.assertTrue(json.loads((self.args.output / f"{arm}_CLOSED.json").read_text())["closed"])
            order.append("native")
            return backend

        def evaluate(actual_backend, requests, checkpoints, root, **kwargs):
            self.assertIs(actual_backend, backend)
            self.assertEqual(requests, self.manifest["eval_requests"])
            self.assertEqual(checkpoints, {"O": "/retained/O", "C": "/fresh/C", "K": "/fresh/K", "A": "/fresh/A"})
            self.assertNotIn("F", checkpoints)
            order.append("evaluate")
            return {"status": "completed", "planned_generations": 768, "returned_generations": 768,
                    "attempted_generations": 768, "unknown_generation_attempts": 0,
                    "unattempted_generations": 0}

        self.campaign.qualify_first_error.run_qualification.side_effect = qualify
        train_patch = stack.enter_context(mock.patch.object(self.campaign, "train_arm", side_effect=train))
        native_patch = stack.enter_context(mock.patch("online_feedback.sglang_backend.SGLangDFlashBackend", side_effect=native))
        evaluate_patch = stack.enter_context(mock.patch.object(self.campaign, "evaluate", side_effect=evaluate))
        return SimpleNamespace(order=order, backend=backend, train=train_patch, native=native_patch,
                               evaluate=evaluate_patch, load=load)

    def test_scope_binds_exact_features_presentations_accumulation_and_panel(self):
        self.campaign.validate_scope(self.manifest, self.ordinary)
        bad_cases = []
        for key, value in (("optimizer_steps", 127), ("response_presentations", 255),
                           ("passes", 1), ("objective", "osd_ce"), ("start_step", 128)):
            ordinary = deepcopy(self.ordinary)
            ordinary[key] = value
            bad_cases.append((key, self.manifest, ordinary))
        for label, change in (
                ("features127", lambda value: value["features"].pop()),
                ("duplicate_identity", lambda value: value["features"][1].update(record_id="train_0")),
                ("duplicate_path", lambda value: value["features"][1].update(path=value["features"][0]["path"])),
                ("records255", lambda value: value["records"].pop()),
                ("missing_anchor", lambda value: value["records"][0].update(sampled_anchors=[])),
                ("multiple_anchor_draws", lambda value: value["records"][0]["sampled_anchors"].append({})),
                ("wrong_pass", lambda value: value["records"][0].update({"pass": 1})),
                ("duplicate_presentation", lambda value: value["records"][1].update(record_id="train_0")),
                ("wrong_step", lambda value: value["records"][0].update(optimizer_stepped=True)),
                ("wrong_counter", lambda value: value["records"][1].update(optimizer_step_after=0))):
            ordinary = deepcopy(self.ordinary)
            change(ordinary)
            bad_cases.append((label, self.manifest, ordinary))
        for label, change in (
                ("requests63", lambda value: value["eval_requests"].pop()),
                ("wrong_ordinal", lambda value: value["eval_requests"][0].update(panel_ordinal=2)),
                ("wrong_cap", lambda value: value["eval_requests"][0].update(max_new_tokens=1024)),
                ("prompt_overlap", lambda value: value["eval_requests"][0].update(prompt_ids=value["train_requests"][0]["prompt_ids"]))):
            manifest = deepcopy(self.manifest)
            change(manifest)
            bad_cases.append((label, manifest, self.ordinary))
        for label, manifest, ordinary in bad_cases:
            with self.subTest(case=label), self.assertRaises(ValueError):
                self.campaign.validate_scope(manifest, ordinary)

    def test_invalid_scope_fails_before_qualification_or_runtime_loading(self):
        patched = self.dependencies()
        self.ordinary["records"].pop()
        write_json(self.args.ordinary_receipt, self.ordinary)
        with self.assertRaises(ValueError):
            self.campaign.run(self.args)
        self.campaign.qualify_first_error.run_qualification.assert_not_called()
        patched.load.assert_not_called()
        patched.train.assert_not_called()
        patched.native.assert_not_called()

    def test_qualification_requires_every_new_gate_exact_pins_and_bounded_completed_work(self):
        valid = self.bound_qualification()
        self.campaign.validate_qualification(valid, self.args)
        invalid = []
        for name in QUALIFICATION_GATES:
            row = deepcopy(valid)
            row["checks"][name] = False
            invalid.append((name, row))
        for name in valid["input_sha256"]:
            row = deepcopy(valid)
            row["input_sha256"][name] = "0" * 64
            invalid.append(("pin " + name, row))
        for name in ("response_presentations", "temporary_optimizer_step_completed", "backward_completed",
                     "current_training_forward_completed", "temporary_optimizer_step_attempted",
                     "backward_attempted", "current_training_forward_attempted",
                     "reference_training_forward_completed", "reference_training_forward_attempted"):
            row = deepcopy(valid)
            row["work_counts"][name] += 1
            invalid.append((name, row))
        for name in ("generations", "main_optimizer_steps"):
            row = deepcopy(valid)
            row[name] = 1
            invalid.append((name, row))
        row = deepcopy(valid)
        row["scope"] = "prefix_ranking_UP"
        invalid.append(("old qualification", row))
        row = deepcopy(valid)
        row["work_counts"]["reference_draft_forwards_observed"] = 1
        invalid.append(("reference actually observed", row))
        for name, row in invalid:
            with self.subTest(case=name), self.assertRaises(ValueError):
                self.campaign.validate_qualification(row, self.args)
        self.args.pinned_initial.write_bytes(b"changed pinned state")
        with self.assertRaisesRegex(ValueError, "binding differs"):
            self.campaign.validate_qualification(valid, self.args)

    def test_success_qualifies_once_trains_only_c_k_a_and_closes_before_768_replays(self):
        patched = self.dependencies()
        self.campaign.run(self.args)
        self.assertEqual(patched.order, ["qualify", "C_closed", "K_closed", "A_closed", "native", "evaluate"])
        self.assertEqual(self.campaign.qualify_first_error.run_qualification.call_count, 1)
        self.assertEqual([call.args[3] for call in patched.train.call_args_list], ["C", "K", "A"])
        self.assertEqual(self.campaign.qualify_first_error.prepare_main.call_count, 3)
        self.assertEqual([call.args[0] for call in self.campaign.qualify_first_error.prepare_main.call_args_list],
                         ["fresh_C", "fresh_K", "fresh_A"])
        patched.backend.close.assert_called_once_with()
        status = self.status()
        self.assertEqual(status["status"], "completed")
        self.assertEqual((status["main_optimizer_steps"], status["main_response_presentations"]), (384, 768))
        self.assertEqual(status["main_steps_remaining"], 0)
        self.assertEqual(status["planned_evaluation_generations"], 768)
        self.assertEqual(status["new_feedback_generations"], 0)
        self.assertEqual(status["new_teacher_preparation_forwards"], 0)
        self.assertFalse(status["automatic_retry"])
        costs = json.loads((self.args.output / "COSTS.json").read_text())
        self.assertGreaterEqual(costs["qualification_wall_s"], 0)
        self.assertEqual(costs["qualification_driver_wall_s"], 1.)

    def test_failed_qualification_blocks_all_main_and_native_work_without_retry(self):
        patched = self.dependencies()
        failed = self.bound_qualification()
        failed["checks"]["first_error_alignment"] = False
        self.campaign.qualify_first_error.run_qualification.side_effect = None
        self.campaign.qualify_first_error.run_qualification.return_value = failed
        with self.assertRaises(ValueError):
            self.campaign.run(self.args)
        patched.train.assert_not_called()
        patched.native.assert_not_called()
        patched.evaluate.assert_not_called()
        self.assertEqual(self.campaign.qualify_first_error.run_qualification.call_count, 1)
        self.assertEqual(self.status()["status"], "failed")
        self.assertEqual(self.status()["main_optimizer_steps"], 0)
        costs = json.loads((self.args.output / "COSTS.json").read_text())
        self.assertIn("qualification_wall_s", costs)

    def test_partial_k_failure_keeps_c_and_actual_k_counts_and_never_starts_a_or_native(self):
        patched = self.dependencies()
        original_train = patched.train.side_effect

        def fail_k(config, ordinary, root, arm, **kwargs):
            if arm == "K":
                write_json(Path(root) / "K_CLOSED.json", {"global_step": 3, "presentations": 7,
                           "accumulation_remainder": 1, "closed": True})
                raise RuntimeError("K failed after partial work")
            return original_train(config, ordinary, root, arm, **kwargs)

        patched.train.side_effect = fail_k
        with self.assertRaisesRegex(RuntimeError, "partial work"):
            self.campaign.run(self.args)
        self.assertEqual([call.args[3] for call in patched.train.call_args_list], ["C", "K"])
        patched.native.assert_not_called()
        patched.evaluate.assert_not_called()
        status = self.status()
        self.assertEqual((status["main_optimizer_steps"], status["main_response_presentations"]), (131, 263))
        self.assertEqual(status["main_steps_remaining"], 253)
        self.assertNotIn("A", status["arm_counters"])
        self.assertEqual(status["status"], "failed")

    def test_unclosed_trainer_stops_before_next_arm_and_native_load(self):
        patched = self.dependencies()
        original_train = patched.train.side_effect

        def unclosed(config, ordinary, root, arm, **kwargs):
            result = original_train(config, ordinary, root, arm, **kwargs)
            path = Path(root) / f"{arm}_CLOSED.json"
            state = json.loads(path.read_text())
            state["closed"] = False
            write_json(path, state)
            return result

        patched.train.side_effect = unclosed
        with self.assertRaisesRegex(RuntimeError, "did not close"):
            self.campaign.run(self.args)
        self.assertEqual(patched.train.call_count, 1)
        patched.native.assert_not_called()

    def test_missing_partial_arm_receipt_marks_counts_unknown_and_remaining_unavailable(self):
        patched = self.dependencies()
        original_train = patched.train.side_effect

        def interrupted(config, ordinary, root, arm, **kwargs):
            if arm == "K":
                raise RuntimeError("interrupted before closure receipt")
            return original_train(config, ordinary, root, arm, **kwargs)

        patched.train.side_effect = interrupted
        with self.assertRaisesRegex(RuntimeError, "closure receipt"):
            self.campaign.run(self.args)
        status = self.status()
        self.assertEqual(status["unknown_arm_counters"], "K")
        self.assertIsNone(status["main_steps_remaining"])
        self.assertEqual((status["main_optimizer_steps"], status["main_response_presentations"]), (128, 256))
        self.assertEqual(list(status["arm_counters"]), ["C"])
        patched.native.assert_not_called()

    def test_evaluation_failure_closes_backend_and_retains_completed_training_counts(self):
        patched = self.dependencies()
        patched.evaluate.side_effect = RuntimeError("native token mismatch")
        with self.assertRaisesRegex(RuntimeError, "token mismatch"):
            self.campaign.run(self.args)
        patched.backend.close.assert_called_once_with()
        self.assertEqual(patched.train.call_count, 3)
        self.assertEqual(patched.evaluate.call_count, 1)
        status = self.status()
        self.assertEqual(status["status"], "failed")
        self.assertEqual((status["main_optimizer_steps"], status["main_response_presentations"]), (384, 768))
        self.assertEqual(status["main_steps_remaining"], 0)

    def test_main_failure_snapshot_precedes_close_and_snapshot_errors_preserve_original_failure(self):
        for capture_fails in (False, True):
            with self.subTest(capture_fails=capture_fails):
                order = []
                trainer = SimpleNamespace(failed=False, closed=False)
                trainer.status = lambda: {"global_step": 0, "presentations": 0,
                                          "completed_events": [], "accumulation_remainder": 0,
                                          "closed": trainer.closed}

                def train(*args, **kwargs):
                    order.append("training_failure")
                    raise RuntimeError("original training error")

                def close():
                    trainer.closed = True
                    order.append("close")

                def save_failure(actual_trainer, path):
                    self.assertIs(actual_trainer, trainer)
                    self.assertFalse(trainer.closed)
                    order.append("capture")
                    if capture_fails:
                        raise OSError("snapshot storage failed")
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_bytes(b"synthetic partial state")

                trainer.train_event, trainer.close = train, close
                startup = SimpleNamespace(prepare=lambda *args: {"wall_s": 0.}, save_failure=save_failure)
                destination = self.root / ("failed_capture" if capture_fails else "saved_capture")
                with mock.patch("online_feedback.training.SpecForgeEventTrainer", return_value=trainer):
                    with self.assertRaisesRegex(RuntimeError, "original training error"):
                        self.campaign.train_arm(self.args.config, self.ordinary, destination, "C", startup=startup)
                self.assertEqual(order, ["training_failure", "capture", "close"])
                self.assertTrue(trainer.failed)
                self.assertTrue(trainer.closed)
                failure = json.loads((destination / "C_FAILURE.json").read_text())
                self.assertIn("original training error", failure["error"])
                if capture_fails:
                    self.assertIn("snapshot storage failed", failure["capture_error"])
                    self.assertNotIn("state_path", failure)
                else:
                    self.assertEqual(Path(failure["state_path"]).read_bytes(), b"synthetic partial state")


if __name__ == "__main__":
    unittest.main()
