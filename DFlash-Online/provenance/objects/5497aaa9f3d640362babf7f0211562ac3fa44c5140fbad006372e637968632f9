"""CPU-only campaign contracts; fake generation is not GPU qualification."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.first_error import campaign
from experiments.prefix_ranking.evaluate import evaluate as shared_evaluate
from experiments.prefix_ranking.protocol import method_order, validate_inputs
from test_osd_one_event import Backend, request
from test_prefix_ranking_evaluation import harmful


def qualified_fixture():
    return {"scope": "first_error_OCKA", "status": "passed",
            "checks": {name: True for name in campaign.QUALIFICATION_GATES}}


def fake_trainer(objective="osd_ce"):
    trainer = mock.Mock()
    trainer.objective = objective
    trainer.failed = False
    trainer.bundle.model.ranking_enabled = False
    trainer.bundle.model.reference = None
    fresh = {"global_step": 0, "presentations": 0, "completed_events": [], "accumulation_remainder": 0}
    complete = {"global_step": 128, "presentations": 256, "completed_events": ["event_001"],
                "accumulation_remainder": 0}
    trainer.status.return_value = fresh
    receipt = {"optimizer_steps": 128, "response_presentations": 256, "start_step": 0,
               "end_step": 128, "passes": 2, "objective": objective, "status": "completed",
               "checkpoint_path": "/new/" + objective, "costs": {"training_s": 5.0},
               "objective_work": {"selection_s": .1}, "selection_totals": {"response_presentations": 256}}

    def event(*args, **kwargs):
        trainer.status.return_value = complete
        return receipt

    trainer.train_event.side_effect = event
    return trainer, receipt


class FirstErrorCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests = [request(index) for index in range(3)]
        self.checkpoints = {arm: "/" + arm for arm in campaign.ARMS}
        self.counts = {row["request_id"]: {"F": 7 + index, "O": 2}
                       for index, row in enumerate(self.requests)}
        self.ordinary = {"checkpoint_path": "/retained/O", "features": [
            {"path": f"/retained/feedback_{index}.pt"} for index in range(128)],
            "records": [{"retained_schedule": True}]}

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized(), "CPU test initialized CUDA")

    def evaluate(self, backend, name="evaluation"):
        return campaign.evaluate(backend, self.requests, self.checkpoints, self.root / name,
                                 ordinary_counts=self.counts, qualification_requests=1)

    def read(self, relative):
        return json.loads((self.root / relative).read_text())

    def measurements(self, name="evaluation"):
        return [json.loads(line) for line in (self.root / name / "MEASUREMENTS.jsonl").read_text().splitlines()]

    def test_exact_768_budget_rotating_fresh_ocka_without_f_generations_or_latency(self):
        requests = [request(index) for index in range(64)]
        counts = {row["request_id"]: {"F": 7, "O": 2} for row in requests}
        backend = Backend(requests)
        result = campaign.evaluate(backend, requests, self.checkpoints, self.root / "full",
                                   ordinary_counts=counts)
        self.assertEqual((result["planned_generations"], result["returned_generations"]), (768, 768))
        self.assertEqual((result["optimizer_steps"], result["feedback_collections"]), (0, 0))
        self.assertEqual(result["unattempted_generations"], 0)
        self.assertEqual(backend.weight_checks, ["/O", "/C", "/K", "/A"])
        positions = {arm: [0] * 4 for arm in campaign.ARMS}
        for ordinal in range(64):
            expected = [arm for repeat in range(3)
                        for arm in method_order(ordinal, repeat, arms=campaign.ARMS)]
            actual = [row[0][1:] for row in backend.executions[ordinal * 12:(ordinal + 1) * 12]]
            self.assertEqual(actual, expected)
            for repeat in range(3):
                for position, arm in enumerate(actual[repeat * 4:(repeat + 1) * 4]):
                    positions[arm][position] += 1
        self.assertEqual(positions, {arm: [48] * 4 for arm in campaign.ARMS})
        for checkpoint, prompt, options in backend.executions:
            self.assertNotEqual(checkpoint, "/F")
            self.assertEqual(options["purpose"], "evaluation")
            self.assertNotIn("response_ids", options)
            self.assertEqual(options["max_new_tokens"], 2048)
            self.assertEqual(prompt, requests[options["request_index"]]["prompt_ids"])
        summary = self.read("full/report/SUMMARY.json")
        self.assertNotIn("F", summary)
        self.assertNotIn("primary_comparison", summary)
        self.assertEqual(set(summary["comparisons"]), {"K_vs_O", "A_vs_C", "C_vs_O", "A_vs_K"})
        self.assertEqual(summary["comparison_axes"], {"position_selection": ["K_vs_O", "A_vs_C"],
                                                      "loss_choice": ["C_vs_O", "A_vs_K"]})
        historical = self.read("full/report/HISTORICAL_F_COUNTS.json")
        self.assertEqual((historical["valid_requests"], historical["verification_calls"]), (64, 448))
        self.assertFalse(any("latency" in key for key in historical))
        protocol = self.read("full/PROTOCOL.json")
        self.assertEqual(protocol["required_arms"], ["O"])
        self.assertEqual(self.read("full/QUALIFICATION.json")["generations"], 96)

    def test_report_four_comparisons_use_the_requested_axes_and_direction(self):
        def mutate(record, options):
            arm = record["checkpoint_id"][1:]
            if arm in ("C", "A"):
                harmful(record)
            record["metadata"]["request_latency_s"] = {"O": 4., "C": 2., "K": 3., "A": 1.}[arm]
            record["metadata"]["native_generate_s"] = record["metadata"]["request_latency_s"] * .9

        result = self.evaluate(Backend(self.requests, mutate))
        self.assertEqual(result["returned_generations"], 36)
        comparisons = self.read("evaluation/report/SUMMARY.json")["comparisons"]
        expected = {"K_vs_O": (0., 25.), "A_vs_C": (0., 50.),
                    "C_vs_O": (-50., 50.), "A_vs_K": (-50., 100. * (1 - 1 / 3))}
        for label, (calls, latency) in expected.items():
            self.assertAlmostEqual(comparisons[label]["call_reduction_pct"], calls)
            self.assertAlmostEqual(comparisons[label]["request_time_reduction_pct"], latency)

    def test_late_mismatch_stops_retains_raw_and_historical_f_uses_common_valid_cohort(self):
        def mutate(record, options):
            if options["request_index"] == 1 and record["checkpoint_id"] == "/C":
                record["response_ids"][1] = 777
                record["metadata"]["trace"][0]["committed_tokens"][0] = 777

        with self.assertRaisesRegex(RuntimeError, "reference_response_ids"):
            self.evaluate(Backend(self.requests, mutate))
        status = self.read("evaluation/STATUS.json")
        self.assertEqual(status["qualification"], "passed")
        self.assertEqual((status["returned_generations"], status["unattempted_generations"]), (13, 23))
        row = self.measurements()[-1]
        self.assertFalse(row["valid_measurement"])
        self.assertEqual(json.loads(Path(row["record_path"]).read_text())["response_ids"][1], 777)
        summary = self.read("evaluation/report/SUMMARY.json")
        self.assertEqual(summary["valid_requests"], 1)
        self.assertEqual(summary["invalid_or_incomplete_requests"], 2)
        historical = self.read("evaluation/report/HISTORICAL_F_COUNTS.json")
        self.assertEqual(historical["per_request"], {"q0": 7})
        self.assertEqual((historical["valid_requests"], historical["verification_calls"]), (1, 7))

    def test_termination_mismatch_is_a_correctness_failure(self):
        def mutate(record, options):
            record["finish_reason"] = "length"

        with self.assertRaisesRegex(RuntimeError, "finish_reason"):
            self.evaluate(Backend(self.requests, mutate))
        self.assertEqual(self.read("evaluation/STATUS.json")["returned_generations"], 1)
        self.assertEqual(self.read("evaluation/report/HISTORICAL_F_COUNTS.json")["valid_requests"], 0)

    def test_retained_o_counts_required_but_historical_f_counts_are_not_a_fresh_gate(self):
        for row in self.requests:
            row["reference_calls"] = {"F": 999, "O": 999}
        self.assertEqual(self.evaluate(Backend(self.requests))["status"], "completed")
        self.counts["q0"]["O"] = 999
        with self.assertRaisesRegex(RuntimeError, "retained_one_event_count_mismatch"):
            self.evaluate(Backend(self.requests), "bad_o")
        self.assertEqual(len(self.measurements("bad_o")), 1)
        self.assertEqual(self.read("bad_o/report/HISTORICAL_F_COUNTS.json")["verification_calls"], 0)
        del self.counts["q0"]["O"]
        with self.assertRaisesRegex(ValueError, "Both retained"):
            self.evaluate(Backend(self.requests), "missing_o")

    def test_shared_evaluator_keeps_legacy_required_f_o_default(self):
        with self.assertRaisesRegex(ValueError, "Four distinct"):
            shared_evaluate(Backend(self.requests), self.requests, self.checkpoints, self.root / "default",
                            ordinary_counts=self.counts, arms=campaign.ARMS, comparisons=campaign.COMPARISONS)
        validate_inputs(self.requests, self.checkpoints, self.counts,
                        arms=campaign.ARMS, required_arms=("O",))
        with self.assertRaisesRegex(ValueError, "Four distinct"):
            validate_inputs(self.requests, {**self.checkpoints, "O": "/C"}, self.counts,
                            arms=campaign.ARMS, required_arms=("O",))

    def test_rejected_existing_evaluation_root_preserves_all_retained_report_bytes(self):
        self.evaluate(Backend(self.requests))
        root = self.root / "evaluation"
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        self.counts["q0"]["F"] = 999
        backend = Backend(self.requests)
        with self.assertRaises(FileExistsError):
            self.evaluate(backend)
        self.assertEqual(backend.executions, [])
        after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_only_c_k_a_train_after_new_qualification_and_retained_o_is_unchanged(self):
        ordinary = deepcopy(self.ordinary)
        startup = object()

        def train(config, retained, root, arm, *, startup):
            self.assertIs(retained, self.ordinary)
            return "/new/" + arm, {"arm": arm}

        with mock.patch.object(campaign, "train_arm", side_effect=train) as train_arm:
            checkpoints, costs = campaign.train_new_arms("config", self.ordinary, self.root,
                                                         startup=startup, qualification=qualified_fixture())
        self.assertEqual([call.args[3] for call in train_arm.call_args_list], ["C", "K", "A"])
        self.assertTrue(all(call.kwargs["startup"] is startup for call in train_arm.call_args_list))
        self.assertEqual(checkpoints, {"O": "/retained/O", "C": "/new/C", "K": "/new/K", "A": "/new/A"})
        self.assertEqual(set(costs), {"C", "K", "A"})
        self.assertEqual(self.ordinary, ordinary)
        invalid = [{"status": "passed", "scope": "prefix_ranking_UP"},
                   {**qualified_fixture(), "status": "failed"}]
        for gate in campaign.QUALIFICATION_GATES:
            receipt = qualified_fixture()
            receipt["checks"][gate] = False
            invalid.append(receipt)
        for receipt in invalid:
            with mock.patch.object(campaign, "train_arm") as train_arm:
                with self.assertRaisesRegex(ValueError, "separate passed"):
                    campaign.train_new_arms("config", self.ordinary, self.root,
                                            startup=startup, qualification=receipt)
                train_arm.assert_not_called()

    def test_train_arm_binds_fresh_state_exact_schedule_and_fixed_128_steps(self):
        for arm, objective in campaign.OBJECTIVES.items():
            with self.subTest(arm=arm):
                trainer, receipt = fake_trainer(objective)
                startup = mock.Mock()
                startup.prepare.return_value = {"wall_s": .2, "exact_pinned_initial_state": True}
                with mock.patch("online_feedback.training.SpecForgeEventTrainer", return_value=trainer) as construct:
                    checkpoint, costs = campaign.train_arm("config", self.ordinary, self.root, arm, startup=startup)
                construct.assert_called_once_with("config", self.root / arm, objective=objective,
                                                  ranking_lambda=0., reduction="response_mean")
                startup.prepare.assert_called_once_with(trainer, self.root / arm / "initialization")
                trainer.train_event.assert_called_once_with("event_001", [Path(row["path"]) for row in self.ordinary["features"]],
                                                            passes=2, presentation_schedule=self.ordinary)
                self.assertEqual(checkpoint, receipt["checkpoint_path"])
                self.assertEqual(costs["selection_totals"]["response_presentations"], 256)
                self.assertEqual(costs["new_feedback_generations"], 0)
                self.assertEqual(costs["new_teacher_preparation_s"], 0)
                self.assertEqual((costs["peak_allocated_bytes"], costs["peak_reserved_bytes"]), (0, 0))
                trainer.close.assert_called_once()
                self.assertFalse(trainer.failed)
                self.assertTrue((self.root / f"{arm}_CLOSED.json").is_file())

    def test_train_arm_closes_and_records_initialization_training_and_accounting_failures(self):
        for failure in ("initialization", "dirty_state", "training", "127_steps"):
            with self.subTest(failure=failure):
                trainer, receipt = fake_trainer()
                startup = mock.Mock()
                startup.prepare.return_value = {"wall_s": .1}
                if failure == "initialization":
                    startup.prepare.side_effect = RuntimeError("bad initialization")
                elif failure == "dirty_state":
                    trainer.status.return_value["accumulation_remainder"] = 1
                elif failure == "training":
                    trainer.train_event.side_effect = RuntimeError("training failed")
                else:
                    receipt["optimizer_steps"] = 127
                destination = self.root / failure
                with mock.patch("online_feedback.training.SpecForgeEventTrainer", return_value=trainer):
                    with self.assertRaises(RuntimeError):
                        campaign.train_arm("config", self.ordinary, destination, "C", startup=startup)
                self.assertTrue(trainer.failed)
                trainer.close.assert_called_once()
                self.assertTrue((destination / "C_FAILURE.json").is_file())
                self.assertTrue((destination / "C_CLOSED.json").is_file())
                if failure in ("initialization", "dirty_state"):
                    trainer.train_event.assert_not_called()

    def test_first_error_startup_validates_fresh_states_once_in_c_k_a_order_without_reference(self):
        startup = object.__new__(campaign.FirstErrorStartup)
        startup._warmup_done = True
        startup._prepared = []
        startup.pinned = object()
        startup.driver = mock.Mock()
        trainers = [fake_trainer(objective)[0] for objective in campaign.OBJECTIVES.values()]
        with self.assertRaisesRegex(RuntimeError, "in order"):
            startup.prepare(trainers[1], self.root / "early_k")
        startup.driver.validate_fresh.assert_not_called()
        for arm, trainer in zip(campaign.OBJECTIVES, trainers):
            receipt = startup.prepare(trainer, self.root / arm)
            startup.driver.validate_fresh.assert_called_with(trainer, startup.pinned, self.root / arm)
            self.assertTrue(receipt["exact_pinned_initial_state"])
            self.assertTrue(receipt["full_rng_restored_after_construction"])
            self.assertFalse(receipt["inherited_warmup_state"])
            self.assertFalse(receipt["reference_created"])
        self.assertEqual(startup._prepared, list(campaign.OBJECTIVES.values()))
        with self.assertRaisesRegex(RuntimeError, "in order"):
            startup.prepare(trainers[0], self.root / "repeat")
        self.assertEqual(startup.driver.validate_fresh.call_count, 3)

    def test_first_error_startup_rejects_missing_warmup_reference_and_auxiliary_before_validation(self):
        for failure in ("warmup", "reference", "auxiliary"):
            with self.subTest(failure=failure):
                startup = object.__new__(campaign.FirstErrorStartup)
                startup._warmup_done = failure != "warmup"
                startup._prepared = []
                startup.pinned = object()
                startup.driver = mock.Mock()
                trainer, _ = fake_trainer()
                if failure == "reference":
                    trainer.bundle.model.reference = object()
                if failure == "auxiliary":
                    trainer.bundle.model.ranking_enabled = True
                with self.assertRaises(RuntimeError):
                    startup.prepare(trainer, self.root / failure)
                startup.driver.validate_fresh.assert_not_called()
                self.assertEqual(startup._prepared, [])

    def test_peak_storage_counts_three_states_exports_initial_failure_staging_and_reserve(self):
        (self.root / "model-00001.safetensors").write_bytes(b"0" * 100)
        (self.root / "model-00002.safetensors").write_bytes(b"0" * 200)
        with mock.patch.object(campaign.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 2**30)) as usage:
            budget = campaign.storage_budget(self.root, self.root)
            self.assertEqual(budget["model_bytes"], 300)
            self.assertEqual(budget["state_export_initial_bytes"], int(16 * 300 * 1.10))
            self.assertEqual(budget["failure_overlap_bytes"], int(4 * 300 * 1.10))
            self.assertEqual(budget["required_bytes"], int(16 * 300 * 1.10) + int(4 * 300 * 1.10) + 16 * 2**30)
            usage.return_value.free = budget["required_bytes"]
            self.assertTrue(campaign.storage_budget(self.root, self.root)["admitted"])
            usage.return_value.free -= 1
            with self.assertRaisesRegex(RuntimeError, "peak-storage"):
                campaign.storage_budget(self.root, self.root)
            self.assertFalse(self.read("STORAGE_ADMISSION.json")["admitted"])
            usage.return_value.free = 100 * 2**30
            empty = self.root / "empty_model"
            empty.mkdir()
            with self.assertRaisesRegex(RuntimeError, "peak-storage"):
                campaign.storage_budget(self.root, empty)


if __name__ == "__main__":
    unittest.main()
