"""CPU-only campaign wiring and native-record fixtures for the bounded R event.

These tests never load a released model or start remote/GPU work. The native
backend below is the retained small deterministic unit-test fixture.
"""
import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from test_osd_one_event import Backend, request
from experiments.prefix_ranking.evaluate import evaluate
from online_feedback.storage import write_json


ARMS = ("F", "O", "P", "R")
COMPARISONS = (("R", "O"), ("R", "P"), ("R", "F"), ("O", "F"), ("P", "O"))


def load_campaign_with_qualification_stub():
    """Test campaign boundaries without executing the separate GPU qualifier."""
    qualification = ModuleType("qualify_r")
    qualification.run_qualification = mock.Mock()
    qualification.prepare_main = mock.Mock(return_value={"wall_s": 0.0})
    qualification.save_failure = mock.Mock()
    spec = importlib.util.spec_from_file_location(
        "reference_margin_campaign_for_tests", Path(__file__).with_name("campaign.py"))
    campaign = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"qualify_r": qualification}):
        spec.loader.exec_module(campaign)
    return campaign


class ReferenceMarginCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.campaign = load_campaign_with_qualification_stub()

    def metrics(self, scale):
        return {
            "kl_loss_numerator": 4.0 * scale,
            "auxiliary_loss_numerator": 2.0 * scale,
            "loss_denominator": 8.0 * scale,
            "protected_active_positions": 2 * scale,
            "protected_positions": 5 * scale,
            "reference_correct_positions": 7 * scale,
            "reference_correct_current_incorrect_positions": scale,
            "supervised_position_exposure": 9 * scale,
        }

    def receipt(self):
        records = [dict(optimizer_stepped=bool(index % 2),
                        metrics=self.metrics(2 if index % 2 else 1),
                        sampled_anchors=[{"keep_mask": [[True, False]]}])
                   for index in range(256)]
        return dict(optimizer_steps=128, response_presentations=256, start_step=0,
                    end_step=128, passes=2, objective="osd_reference_margin", status="completed",
                    records=records, costs={"export_s": 0.0}, objective_work={"reference_forward_s": 0.0},
                    checkpoint_path=str(self.root / "R/export"))

    def trainer_fixture(self):
        order = []
        trainer = SimpleNamespace(failed=False, closed=False, steps=0, presentations=0)

        def status():
            return dict(global_step=trainer.steps, presentations=trainer.presentations,
                        completed_events=["event_001"] if trainer.steps == 128 else [],
                        accumulation_remainder=0, closed=trainer.closed)

        def train(*args, **kwargs):
            order.append("train")
            trainer.steps, trainer.presentations = 128, 256
            return self.receipt()

        def close():
            trainer.closed = True
            order.append("close")

        trainer.status = mock.Mock(side_effect=status)
        trainer.train_event = mock.Mock(side_effect=train)
        trainer.close = mock.Mock(side_effect=close)
        return trainer, order

    def patch_cpu_trainer(self, trainer):
        stack = ExitStack()
        self.addCleanup(stack.close)
        constructor = stack.enter_context(mock.patch("online_feedback.training.SpecForgeEventTrainer",
                                                      return_value=trainer))
        stack.enter_context(mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU fixture initialized CUDA")))
        stack.enter_context(mock.patch("torch.cuda.reset_peak_memory_stats"))
        stack.enter_context(mock.patch("torch.cuda.max_memory_allocated", return_value=0))
        stack.enter_context(mock.patch("torch.cuda.max_memory_reserved", return_value=0))
        return constructor

    def test_diagnostics_sum_only_boundaries_with_explicit_pooled_denominators(self):
        rows = [dict(optimizer_stepped=stepped, metrics=self.metrics(scale))
                for stepped, scale in ((False, 100), (True, 2), (False, 200), (True, 3))]
        result = self.campaign.diagnostic_totals({"records": rows})
        self.assertEqual(result["boundary_rows"], 2)
        self.assertEqual(result["sums"], {key: self.metrics(5)[key] for key in self.campaign.DIAGNOSTIC_SUMS})
        self.assertEqual(result["ratios"]["protected_activation_rate"], 2 / 5)
        self.assertEqual(result["ratios"]["reference_correct_to_incorrect_rate"], 1 / 7)
        self.assertEqual(result["ratios"]["pooled_weighted_forward_kl"], 0.5)
        self.assertEqual(result["ratios"]["pooled_weighted_auxiliary_loss"], 0.25)
        empty = self.campaign.diagnostic_totals(
            {"records": [dict(optimizer_stepped=True, metrics=self.metrics(0))]})
        self.assertTrue(all(value is None for value in empty["ratios"].values()))
        self.assertTrue(all(value == 0 for value in empty["sums"].values()))

    def test_train_constructs_only_fresh_r_then_prepares_and_closes_exact_scheduled_event(self):
        trainer, order = self.trainer_fixture()
        constructor = self.patch_cpu_trainer(trainer)
        ordinary = {"features": [{"path": str(self.root / f"feature_{index}.pt")} for index in range(128)]}

        def prepare(actual_trainer, pin, path):
            self.assertIs(actual_trainer, trainer)
            self.assertEqual(pin, self.root / "initial.pt")
            self.assertEqual(path, self.root / "R/initialization")
            self.assertFalse(trainer.closed)
            order.append("prepare")
            return {"wall_s": 0.0, "full_rng_restored_after_construction": True}

        self.campaign.qualify_r.prepare_main.side_effect = prepare
        checkpoint, costs = self.campaign.train_r("config.yaml", ordinary, self.root,
                                                pinned_initial=self.root / "initial.pt")
        constructor.assert_called_once_with(
            "config.yaml", self.root / "R", objective="osd_reference_margin",
            ranking_lambda=1.0, reduction="response_mean")
        trainer.train_event.assert_called_once_with(
            "event_001", [Path(row["path"]) for row in ordinary["features"]],
            passes=2, presentation_schedule=ordinary)
        self.assertIs(trainer.train_event.call_args.kwargs["presentation_schedule"], ordinary)
        self.assertEqual(order, ["prepare", "train", "close"])
        self.assertEqual(checkpoint, self.receipt()["checkpoint_path"])
        self.assertEqual(costs["diagnostics"]["boundary_rows"], 128)
        self.assertEqual(costs["new_feedback_generations"], 0)
        self.assertEqual(costs["teacher_preparation_s"], 0.0)
        closed = json.loads((self.root / "R_CLOSED.json").read_text())
        self.assertTrue(closed["closed"])
        self.assertEqual((closed["global_step"], closed["presentations"]), (128, 256))
        self.campaign.qualify_r.run_qualification.assert_not_called()

    def test_startup_failure_prevents_event_and_closes_without_retry(self):
        trainer, order = self.trainer_fixture()
        self.patch_cpu_trainer(trainer)
        self.campaign.qualify_r.prepare_main.side_effect = AssertionError("fresh state differs")
        with self.assertRaisesRegex(AssertionError, "fresh state differs"):
            self.campaign.train_r("config.yaml", {"features": []}, self.root,
                                  pinned_initial="initial.pt")
        trainer.train_event.assert_not_called()
        self.assertEqual(order, ["close"])
        self.assertTrue(trainer.failed)
        self.campaign.qualify_r.save_failure.assert_called_once_with(trainer, self.root / "R/failure_state.pt")
        self.assertTrue((self.root / "R_FAILURE.json").is_file())

    def test_wrong_training_budget_is_rejected_and_failure_state_preserved(self):
        trainer, order = self.trainer_fixture()
        self.patch_cpu_trainer(trainer)
        receipt = self.receipt()
        receipt["optimizer_steps"] = 127
        trainer.train_event.side_effect = None
        trainer.train_event.return_value = receipt
        with self.assertRaises(RuntimeError):
            self.campaign.train_r("config.yaml", {"features": []}, self.root,
                                  pinned_initial="initial.pt")
        self.assertTrue(trainer.failed)
        self.assertTrue(trainer.closed)
        self.assertEqual(trainer.train_event.call_count, 1)
        self.campaign.qualify_r.save_failure.assert_called_once()

    def run_fixture(self):
        config = self.root / "config.yaml"
        config.write_text("synthetic config")
        manifest = self.root / "manifest.json"
        write_json(manifest, {"eval_requests": [request(index) for index in range(64)]})
        ordinary = self.root / "ordinary.json"
        write_json(ordinary, {"checkpoint_path": "retained_O", "features": [{"path": f"retained_{i}.pt"} for i in range(128)]})
        prefix = self.root / "prefix.json"
        write_json(prefix, {"checkpoint_path": "retained_P"})
        return SimpleNamespace(config=config, manifest=manifest, ordinary_receipt=ordinary,
                               prefix_receipt=prefix, ordinary_measurements=self.root / "ordinary_measurements.jsonl",
                               output=self.root / "campaign", pinned_initial=self.root / "pin.pt",
                               controlled_runner=self.root / "controlled.py", diagnostic_runner=self.root / "diagnostic.py",
                               worker_python="unused_python", serving_gpu="0")

    def patch_run_dependencies(self, args, order):
        stack = ExitStack()
        self.addCleanup(stack.close)
        model = SimpleNamespace(draft_checkpoint_path="released_F", target_model_path="frozen_target")
        stack.enter_context(mock.patch("online_feedback.recipe.load_event_config",
                                       return_value=(SimpleNamespace(model=model), None)))
        stack.enter_context(mock.patch("online_feedback.qualify_sglang.check_recipe"))
        stack.enter_context(mock.patch("experiments.osd_one_event.protocol.validate_manifest"))
        stack.enter_context(mock.patch.object(self.campaign, "load_ordinary_counts", return_value={}))

        def qualify(*args, **kwargs):
            order.append("qualify")
            return {"status": "passed", "wall_s": 1.0}

        def train(*args, **kwargs):
            self.assertEqual(order, ["qualify"])
            order.append("train_closed")
            return "fresh_R", {"event_wall_s": 1.0}

        backend = SimpleNamespace(loading_s=0.5, close=mock.Mock())

        def load_backend(*args, **kwargs):
            self.assertEqual(order, ["qualify", "train_closed"])
            order.append("native")
            return backend

        def replay(*args, **kwargs):
            self.assertEqual(order, ["qualify", "train_closed", "native"])
            order.append("evaluate")
            self.assertEqual(args[2], {"F": "released_F", "O": "retained_O", "P": "retained_P", "R": "fresh_R"})
            self.assertEqual(len(args[1]), 64)
            self.assertEqual(kwargs["arms"], ARMS)
            self.assertEqual(kwargs["comparisons"], COMPARISONS)
            return {"status": "completed", "returned_generations": 768}

        self.campaign.qualify_r.run_qualification.side_effect = qualify
        train_mock = stack.enter_context(mock.patch.object(self.campaign, "train_r", side_effect=train))
        backend_mock = stack.enter_context(mock.patch("online_feedback.sglang_backend.SGLangDFlashBackend",
                                                      side_effect=load_backend))
        evaluate_mock = stack.enter_context(mock.patch.object(self.campaign, "evaluate", side_effect=replay))
        return train_mock, backend_mock, evaluate_mock, backend

    def test_run_qualifies_once_then_only_r_then_loads_capture_disabled_evaluation(self):
        args, order = self.run_fixture(), []
        train, native, replay, backend = self.patch_run_dependencies(args, order)
        self.campaign.run(args)
        self.assertEqual(order, ["qualify", "train_closed", "native", "evaluate"])
        self.assertEqual(self.campaign.qualify_r.run_qualification.call_count, 1)
        self.assertEqual(train.call_count, 1)
        backend.close.assert_called_once_with()
        status = json.loads((args.output / "STATUS.json").read_text())
        self.assertEqual(status["status"], "completed")
        self.assertEqual((status["main_optimizer_steps"], status["main_response_presentations"]), (128, 256))
        self.assertEqual(status["new_feedback_generations"], 0)
        self.assertEqual(status["new_teacher_preparation_forwards"], 0)
        self.assertFalse(status["automatic_retry"])

    def test_failed_qualification_blocks_main_training_and_native_loading(self):
        args, order = self.run_fixture(), []
        train, native, replay, backend = self.patch_run_dependencies(args, order)
        self.campaign.qualify_r.run_qualification.side_effect = None
        self.campaign.qualify_r.run_qualification.return_value = {"status": "failed", "wall_s": 1.0}
        with self.assertRaisesRegex(RuntimeError, "qualification did not pass"):
            self.campaign.run(args)
        train.assert_not_called()
        native.assert_not_called()
        replay.assert_not_called()
        backend.close.assert_not_called()
        self.assertEqual(self.campaign.qualify_r.run_qualification.call_count, 1)
        status = json.loads((args.output / "STATUS.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual((status["main_optimizer_steps"], status["main_response_presentations"]), (0, 0))

    def test_failed_training_blocks_evaluation_and_reports_partial_closed_counters(self):
        args, order = self.run_fixture(), []
        train, native, replay, backend = self.patch_run_dependencies(args, order)

        def failed_train(*args, **kwargs):
            write_json(Path(args[2]) / "R_CLOSED.json", {"global_step": 1, "presentations": 2})
            raise RuntimeError("training failed")

        train.side_effect = failed_train
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            self.campaign.run(args)
        native.assert_not_called()
        replay.assert_not_called()
        self.assertEqual(train.call_count, 1)
        status = json.loads((args.output / "STATUS.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual((status["main_optimizer_steps"], status["main_response_presentations"]), (1, 2))

    def test_evaluation_failure_closes_backend_and_keeps_completed_training_counts(self):
        args, order = self.run_fixture(), []
        train, native, replay, backend = self.patch_run_dependencies(args, order)
        replay.side_effect = RuntimeError("invalid native output")
        with self.assertRaisesRegex(RuntimeError, "invalid native output"):
            self.campaign.run(args)
        backend.close.assert_called_once_with()
        self.assertEqual(train.call_count, 1)
        self.assertEqual(replay.call_count, 1)
        status = json.loads((args.output / "STATUS.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual((status["main_optimizer_steps"], status["main_response_presentations"]), (128, 256))


class ReferenceMarginEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = [request(index) for index in range(64)]
        self.checkpoints = {arm: "/" + arm for arm in ("F", "O", "P", "R")}
        self.counts = {row["request_id"]: {"F": 2, "O": 2} for row in self.requests}

    def evaluate(self, backend, name="evaluation"):
        return evaluate(
            backend, self.requests, self.checkpoints, self.root / name,
            ordinary_counts=self.counts, arms=ARMS, comparisons=COMPARISONS)

    def summary(self, name="evaluation"):
        return json.loads((self.root / name / "report/SUMMARY.json").read_text())

    def test_exact_768_budget_contains_96_generation_gate_and_only_f_o_p_r(self):
        backend = Backend(self.requests)
        status = self.evaluate(backend)
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["planned_generations"], 768)
        self.assertEqual(status["attempted_generations"], 768)
        self.assertEqual(status["returned_generations"], 768)
        self.assertEqual(status["unknown_generation_attempts"], 0)
        self.assertEqual(status["unattempted_generations"], 0)
        self.assertEqual(status["optimizer_steps"], 0)
        self.assertEqual(status["feedback_collections"], 0)
        gate = json.loads((self.root / "evaluation/QUALIFICATION.json").read_text())
        self.assertEqual((gate["requests"], gate["generations"]), (8, 96))
        self.assertEqual(backend.weight_checks, ["/F", "/O", "/P", "/R"])
        position_counts = {arm: [0] * 4 for arm in self.checkpoints}
        for ordinal in range(64):
            for repeat in range(3):
                offset = (ordinal * 3 + repeat) * 4
                actual = [entry[0][1:] for entry in backend.executions[offset:offset + 4]]
                shift = (ordinal + repeat) % 4
                arms = ["F", "O", "P", "R"]
                self.assertEqual(actual, arms[shift:] + arms[:shift])
                for position, arm in enumerate(actual):
                    position_counts[arm][position] += 1
        self.assertEqual(position_counts, {arm: [48] * 4 for arm in self.checkpoints})
        for checkpoint, prompt, options in backend.executions:
            self.assertIn(checkpoint, self.checkpoints.values())
            self.assertEqual(options["purpose"], "evaluation")
            self.assertEqual(options["max_new_tokens"], 2048)
            self.assertNotIn("response_ids", options)
            self.assertNotIn("collect_features", options)
            self.assertEqual(prompt, self.requests[options["request_index"]]["prompt_ids"])
        summary = self.summary()
        self.assertEqual(summary["valid_requests"], 64)
        self.assertEqual(summary["primary_comparison"], "R_vs_O")
        for name in ("R_vs_O", "R_vs_P", "R_vs_F"):
            self.assertIn(name, summary["comparisons"])
        self.assertNotIn("U", summary)
        self.assertIn("F/O/P/R", summary["cohort"])
        text = (self.root / "evaluation/report/REPORT.md").read_text()
        self.assertIn("**R versus O is primary.**", text)
        self.assertNotIn("**P versus O is primary.**", text)
        self.assertNotIn("Beating U alone", text)
        protocol = json.loads((self.root / "evaluation/PROTOCOL.json").read_text())
        self.assertEqual(protocol["arms"], ["F", "O", "P", "R"])
        self.assertFalse(protocol["capture"])

    def test_correct_but_harmful_r_continues_and_remains_in_primary_comparison(self):
        def mutation(record, options):
            if record["checkpoint_id"] != "/R":
                return
            record["metadata"].update(
                verification_calls=3, accepted_draft_tokens=1,
                trace=[
                    dict(output_start=0, anchor_token=10, committed_progress=1,
                         committed_tokens=[11], accepted_draft_tokens=0),
                    dict(output_start=1, anchor_token=11, committed_progress=1,
                         committed_tokens=[12], accepted_draft_tokens=0),
                    dict(output_start=2, anchor_token=12, committed_progress=2,
                         committed_tokens=[13, 99], accepted_draft_tokens=1)])

        status = self.evaluate(Backend(self.requests, mutation))
        self.assertEqual(status["returned_generations"], 768)
        comparison = self.summary()["comparisons"]["R_vs_O"]
        self.assertEqual(comparison["call_reduction_pct"], -50.0)
        self.assertEqual(comparison["call_request_directions"],
                         {"improved": 0, "regressed": 64, "tied": 0})

    def test_output_mismatch_stops_and_preserves_invalid_record_without_retry(self):
        def mutation(record, options):
            if record["checkpoint_id"] == "/R":
                record["response_ids"][1] = 777
                record["metadata"]["trace"][0]["committed_tokens"][0] = 777

        backend = Backend(self.requests, mutation)
        with self.assertRaisesRegex(RuntimeError, "reference_response_ids"):
            self.evaluate(backend)
        status = json.loads((self.root / "evaluation/STATUS.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["returned_generations"], 4)
        self.assertEqual(status["unattempted_generations"], 764)
        self.assertEqual(len(backend.executions), 4)
        rows = [json.loads(line) for line in
                (self.root / "evaluation/MEASUREMENTS.jsonl").read_text().splitlines()]
        self.assertEqual(rows[-1]["method"], "R")
        self.assertFalse(rows[-1]["valid_measurement"])
        self.assertTrue(Path(rows[-1]["record_path"]).is_file())
        self.assertEqual(self.summary()["valid_requests"], 0)

    def test_capture_and_retained_ordinary_count_mismatches_are_failures(self):
        def capture(record, options):
            record["metadata"]["feedback_capture_in_timed_request"] = True
            record["metadata"]["capture"]["collect_features"] = True

        with self.assertRaisesRegex(RuntimeError, "capture"):
            self.evaluate(Backend(self.requests, capture), "capture")
        self.counts["q0"]["O"] = 999
        with self.assertRaisesRegex(RuntimeError, "retained_one_event_count_mismatch"):
            self.evaluate(Backend(self.requests), "ordinary_mismatch")

    def test_existing_prefix_evaluation_globals_are_preserved_after_success_and_failure(self):
        from experiments.prefix_ranking import evaluate, protocol, report

        def configuration():
            return (protocol.ARMS, protocol.COMPARISONS, evaluate.ARMS,
                    report.ARMS, report.COMPARISONS)

        expected = copy.deepcopy(configuration())
        self.evaluate(Backend(self.requests), "success")
        self.assertEqual(configuration(), expected)
        self.counts["q0"]["F"] = 999
        with self.assertRaises(RuntimeError):
            self.evaluate(Backend(self.requests), "failure")
        self.assertEqual(configuration(), expected)

    def test_omitted_parameters_keep_existing_f_o_u_p_defaults(self):
        requests = self.requests[:2]
        checkpoints = {arm: "/" + arm for arm in ("F", "O", "U", "P")}
        counts = {row["request_id"]: self.counts[row["request_id"]] for row in requests}
        backend = Backend(requests)
        result = evaluate(backend, requests, checkpoints, self.root / "legacy",
                          ordinary_counts=counts, qualification_requests=1)
        self.assertEqual(result["returned_generations"], 24)
        self.assertEqual(backend.weight_checks, ["/F", "/O", "/U", "/P"])
        summary = self.summary("legacy")
        self.assertEqual(summary["primary_comparison"], "P_vs_O")
        self.assertEqual(set(summary["comparisons"]), {"P_vs_O", "P_vs_U", "P_vs_F", "O_vs_F"})
        self.assertIn("F/O/U/P", summary["cohort"])
        text = (self.root / "legacy/report/REPORT.md").read_text()
        self.assertIn("**P versus O is primary.**", text)
        self.assertIn("Beating U alone does not establish improvement over ordinary OSD.", text)

    def test_unknown_arm_or_comparison_rejected_before_backend_work(self):
        for name, options in (
                ("arms", {"arms": ("F", "O", "P", "P"), "comparisons": COMPARISONS}),
                ("comparison", {"arms": ARMS, "comparisons": (("R", "U"),)})):
            with self.subTest(case=name):
                backend = Backend(self.requests)
                with self.assertRaises(ValueError):
                    evaluate(backend, self.requests, self.checkpoints, self.root / name,
                             ordinary_counts=self.counts, **options)
                self.assertEqual(backend.executions, [])
                self.assertEqual(backend.switches, [])
                self.assertFalse((self.root / name).exists())


if __name__ == "__main__":
    unittest.main()
