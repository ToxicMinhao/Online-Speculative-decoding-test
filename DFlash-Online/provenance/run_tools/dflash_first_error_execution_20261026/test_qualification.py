"""CPU-only tests for the disposable native qualification driver."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import qualify_first_error as Q
from online_feedback.first_error import first_error_mask, selection_statistics
from online_feedback.objective import TargetObjectiveModel
from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch, write_json
from online_feedback import training as T
import test_osd_recipe as legacy
from test_objective_integration import tiny_author
from test_osd_objective import osd_batch
from test_training_contract import trees_equal


class QualificationUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()

    @classmethod
    def tearDownClass(cls):
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(cls.threads)

    def test_plan_exactly_seven_windows_no_reference_or_generation(self):
        self.assertEqual(len(Q.TRIALS), 7)
        self.assertEqual([row[0] for row in Q.TRIALS],
                         ["warmup", "O", "C", "K_all", "A_all", "K_natural", "A_natural"])
        self.assertEqual((Q.PLAN["backwards"], Q.PLAN["temporary_optimizer_steps"]), (14, 7))
        self.assertEqual(Q.PLAN["detached_logit_gradient_calls"], 30)
        for key in ("reference_training_forwards", "detached_comparison_forwards", "generations",
                    "teacher_model_forwards", "main_optimizer_steps"):
            self.assertEqual(Q.PLAN[key], 0)
        budget = Q.storage_budget(100)
        self.assertEqual(budget["required_bytes"], 2800 + 10 * 2**30)
        self.assertEqual(budget["retained_qualification_tensors_after_success"], 0)
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                Q.storage_budget(invalid)

    def test_cpu_loop_oracle_handles_first_error_gap_eos_and_empty(self):
        labels = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 9, 4],
                                [9, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]])
        proposals = labels.clone()
        proposals[0, 0, 1] = 8
        proposals[0, 1, 3] = 8
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights[..., 0] = 0
        weights[0, 4, 2] = 0
        weights[0, 5, 1] = 0
        expected, counts = Q.first_error_oracle(proposals, labels, weights, (9,))
        self.assertEqual(expected.tolist(), [[[False, True, False, False], [False, True, True, True],
                                              [False, True, True, False], [False, False, False, False],
                                              [False, True, False, False], [False, False, False, False]]])
        self.assertEqual(counts, (2, 2, [1, 0, 1]))
        self.assertTrue(torch.equal(expected, first_error_mask(proposals, labels, weights, (9,))))
        self.assertTrue(torch.equal(Q.expected_statistics(proposals, labels, weights, expected, (9,)),
                                    selection_statistics(proposals, labels, weights, expected, (9,))))

    def test_current_forward_observer_checks_real_gradients_and_checkpoint_counts(self):
        for arm, forced in (("osd_ce", False), ("osd_first_error", False),
                            ("osd_ce_first_error", False), ("osd_first_error", True),
                            ("osd_ce_first_error", True)):
            with self.subTest(arm=arm, forced=forced), tempfile.TemporaryDirectory() as folder:
                model = TargetObjectiveModel(tiny_author(), arm, loss_precision="bfloat16", eos_token_ids=(28,))
                model.begin_event()
                log = Q.WorkLog(folder)
                observer = Q.FirstErrorObserver(model, log, force_all=forced)
                with observer.installed():
                    loss, _, _ = model(**osd_batch(model))
                    chunks_before = log.counts["objective_chunk_calls_including_recompute"]
                    (loss / 2).backward()
                    self.assertGreater(log.counts["objective_chunk_calls_including_recompute"], chunks_before)
                    observer.check_saved_gradients()
                self.assertEqual((len(observer.loss_checks), len(observer.gradient_checks)), (1, 1))
                self.assertEqual(log.counts["detached_logit_gradient_completed"], 3)
                self.assertEqual(log.counts["current_training_forward_completed"], 1)
                self.assertEqual(log.counts["current_draft_forwards_observed"], 1)
                self.assertLessEqual(observer.gradient_checks[0]["rows"], 8)
                self.assertTrue(observer.gradient_checks[0]["actual_excluded_gradient_zero"])
                self.assertIsNone(model.reference)
                if arm in Q.CE_OBJECTIVES:
                    self.assertEqual(model.end_event()["teacher_head_s"], 0)

    def test_observer_does_not_change_model_gradients_or_rng(self):
        for arm in Q.NEW_OBJECTIVES:
            author = tiny_author()
            results = []
            for observed in (False, True):
                with tempfile.TemporaryDirectory() as folder:
                    model = TargetObjectiveModel(copy.deepcopy(author), arm, loss_precision="bfloat16")
                    torch.manual_seed(17)
                    model.begin_event()
                    if observed:
                        observer = Q.FirstErrorObserver(model, Q.WorkLog(folder))
                        with observer.installed():
                            loss, _, metrics = model(**osd_batch(model))
                            loss.backward()
                            observer.check_saved_gradients()
                    else:
                        loss, _, metrics = model(**osd_batch(model))
                        loss.backward()
                    results.append((loss.detach(), metrics,
                                    {name: parameter.grad for name, parameter in model.draft_model.named_parameters()},
                                    capture_rng()))
            self.assertTrue(trees_equal(results[0], results[1]), arm)

    def test_common_core_omits_only_extra_diagnostics_and_keeps_scientific_values(self):
        core = dict(micro=1, ratio_totals={"loss": (torch.tensor(1.), torch.tensor(1.)),
                                        "objective_loss": (torch.tensor(2.), torch.tensor(3.)),
                                        "selection_rate": (torch.tensor(1.), torch.tensor(2.))},
                    sum_totals={"response_presentations": torch.tensor(1.),
                                "supervised_position_exposure": torch.tensor(5.),
                                "first_error_position_1": torch.tensor(2.)})
        common = Q.common_core(core)
        self.assertEqual(common["micro"], 1)
        self.assertEqual(set(common["ratio_totals"]), {"loss", "objective_loss"})
        self.assertEqual(set(common["sum_totals"]), {"response_presentations", "supervised_position_exposure"})
        changed = copy.deepcopy(common)
        changed["ratio_totals"]["loss"][0].add_(1)
        self.assertFalse(trees_equal(common, changed))

    def test_retirement_requires_evidence_and_refuses_foreign_or_failure_tensors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "qualification"
            root.mkdir()
            state = root / "trial.pt"
            state.write_bytes(b"temporary state")
            evidence = root / "comparison.json"
            evidence.write_text('{"exact_equal": true}')
            driver = SimpleNamespace(sha256=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
            with mock.patch.object(Q, "controlled", return_value=driver):
                with self.assertRaises(AssertionError):
                    Q.retire_tensors([state], root, reason="test", evidence=[])
                self.assertTrue(state.exists())
                for forbidden in (Path(folder) / "old.pt", root / "failure_state.pt", root / "REPORT.md"):
                    forbidden.write_bytes(b"keep")
                    with self.assertRaises(AssertionError):
                        Q.retire_tensors([forbidden], root, reason="test", evidence=[evidence])
                    self.assertTrue(forbidden.exists())
                rows = Q.retire_tensors([state], root, reason="exact comparison passed", evidence=[evidence])
                self.assertFalse(state.exists())
                self.assertTrue(rows[0]["removed"])
                self.assertEqual(rows[0]["sha256"], hashlib.sha256(b"temporary state").hexdigest())
                self.assertIn("lost_capability", rows[0])
                receipt = json.loads((root / "RECLAMATION.json").read_text())
                self.assertEqual(len(receipt["entries"]), 1)
                self.assertEqual(receipt["batches"][0]["status"], "completed")
                self.assertIn("free_bytes_before", receipt["batches"][0])
                self.assertIn("free_bytes_after", receipt["batches"][0])

    def test_prepare_main_is_json_safe_and_rejects_reference_or_old_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            driver = SimpleNamespace(load_state=mock.Mock(return_value={}),
                                     validate_fresh=mock.Mock(return_value={"tensor": torch.ones(1)}))
            for index, arm in enumerate(sorted(Q.NEW_OBJECTIVES)):
                model = SimpleNamespace(ranking_enabled=False, reference=None)
                trainer = SimpleNamespace(objective=arm, bundle=SimpleNamespace(model=model))
                with mock.patch.object(Q, "controlled", return_value=driver):
                    receipt = Q.prepare_main(trainer, "pin.pt", Path(folder) / str(index))
                json.dumps(receipt)
                self.assertTrue(receipt["exact_pinned_initial_state"])
                self.assertFalse(receipt["inherited_warmup_state"])
            for objective, ranking, reference in (("osd", False, None), ("osd_ce", True, None),
                                                  ("osd_ce", False, object())):
                trainer = SimpleNamespace(objective=objective, bundle=SimpleNamespace(model=SimpleNamespace(
                    ranking_enabled=ranking, reference=reference)))
                with self.assertRaises(AssertionError):
                    Q.prepare_main(trainer, "pin.pt", Path(folder) / "rejected")


class FullQualifierCPUIntegrationTests(unittest.TestCase):
    """Real tiny trainer/Adam/schedules and exact gates across all seven windows."""
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, "ranking_eos_token_ids", return_value=(28,))
        patch.start()
        self.addCleanup(patch.stop)
        assert not torch.cuda.is_initialized()

    def retained_fixture(self):
        for index in range(len(self.features), 128):
            raw = torch.load(self.features[index % 2], weights_only=True)
            raw["record_id"] = f"response_{index}"
            raw["label_binding"]["record_id"] = raw["record_id"]
            path = self.root / f"feedback_{index}.pt"
            torch.save(raw, path)
            self.features.append(path)
        controlled_runner = Q.DEFAULT_CONTROLLED
        diagnostic_runner = controlled_runner.parent.parent / "dflash_prefix_ranking_gpu_diagnostic_20261026/runner.py"
        driver = Q.controlled(controlled_runner, diagnostic_runner)
        ordinary_trainer = self.trainer("retained_ordinary", objective="osd")
        pin_path = self.root / "retained_initial.pt"
        save_torch(pin_path, driver.scientific_state(ordinary_trainer))
        ordinary = ordinary_trainer.train_event("event_001", self.features, passes=2)
        ordinary_trainer.close()
        ordinary_path = self.root / "retained_ordinary.json"
        write_json(ordinary_path, ordinary)
        return driver, controlled_runner, diagnostic_runner, pin_path, ordinary_path

    def test_all_seven_windows_exact_gates_original_schedule_and_rolling_retirement(self):
        driver, controlled_runner, diagnostic_runner, pin_path, ordinary_path = self.retained_fixture()
        root = self.root / "qualification"
        result = Q.run_qualification(self.config, ordinary_path, pin_path, root,
                                     controlled_runner=controlled_runner, diagnostic_runner=diagnostic_runner)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["scope"], "first_error_OCKA")
        self.assertEqual(result["work_counts"]["response_presentations"], 14)
        self.assertEqual(result["work_counts"]["temporary_optimizer_step_completed"], 7)
        self.assertEqual(result["work_counts"]["backward_completed"], 14)
        self.assertEqual(result["work_counts"]["current_draft_forwards_observed"], 14)
        self.assertEqual(result["work_counts"]["detached_logit_gradient_completed"], 30)
        self.assertEqual(result["generations"], 0)
        self.assertTrue(result["all_temporary_tensors_retired"])
        self.assertFalse(list(root.rglob("*.pt")))
        self.assertTrue(pin_path.exists())
        for label in ("K_all", "A_all"):
            for stage in ("after_backward_0", "after_backward_1", "pre_optimizer", "final", "records"):
                self.assertTrue(json.loads((root / label / f"{stage}_vs_baseline.json").read_text())["exact_equal"])
        for trial in result["trials"]:
            self.assertEqual(trial["work_counts"]["current_training_forward_completed"], 2)
            self.assertEqual(trial["work_counts"]["temporary_optimizer_step_completed"], 1)
            self.assertEqual(trial["objective_work"]["reference_setup_s"], 0)
            if trial["objective"] in Q.NEW_OBJECTIVES:
                self.assertEqual(len(trial["loss_checks"]), 2)
                self.assertEqual(len(trial["gradient_checks"]), 2)
                self.assertEqual(trial["work_counts"]["detached_logit_gradient_completed"], 6)
        reclamation = json.loads((root / "RECLAMATION.json").read_text())
        self.assertEqual(len(reclamation["entries"]), 35)
        for record in reclamation["entries"]:
            self.assertTrue(record["removed"])
            self.assertEqual(len(record["sha256"]), 64)
            for path, digest in record["retained_evidence_sha256"].items():
                self.assertEqual(driver.sha256(path), digest)
        fresh = self.trainer("main_C", objective="osd_ce")
        receipt = Q.prepare_main(fresh, pin_path, self.root / "main_C_initial")
        json.dumps(receipt)
        self.assertEqual((fresh.global_step, fresh.presentations, fresh.events), (0, 0, []))
        self.assertFalse(fresh.backend.optimizer.state_dict()["optimizer_state_dict"]["state"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_failed_native_gate_stops_and_keeps_failure_and_needed_comparator_tensors(self):
        _, controlled_runner, diagnostic_runner, pin_path, ordinary_path = self.retained_fixture()
        root = self.root / "failed_qualification"
        with mock.patch.object(Q, "check_prediction_gradients", side_effect=AssertionError("injected gradient gate")):
            with self.assertRaisesRegex(AssertionError, "injected gradient gate"):
                Q.run_qualification(self.config, ordinary_path, pin_path, root,
                                    controlled_runner=controlled_runner, diagnostic_runner=diagnostic_runner)
        result = json.loads((root / "RESULT.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertFalse(any(result["checks"].values()))
        self.assertEqual([trial["label"] for trial in result["trials"]], ["warmup", "O", "C"])
        self.assertEqual(result["work_counts"]["temporary_optimizer_step_completed"], 2)
        self.assertEqual(result["work_counts"]["backward_completed"], 5)
        self.assertTrue((root / "C/failure_state.pt").is_file())
        self.assertTrue((root / "O/final.pt").is_file())
        self.assertTrue((root / "O/pre_optimizer.pt").is_file())
        self.assertFalse((root / "K_all").exists())
        self.assertFalse(result["main_campaign_started"])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
