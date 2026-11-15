"""CPU orchestration checks: these tests never initialize a model or GPU."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.native_update_size import campaign
from online_feedback.storage import write_json
from test_native_update_size_analysis import make_grid


def plan_fixture(root):
    gate = dict(status="passed", cuda_initialized=False, optimizer_steps=0, cpu_wall_s=.25)
    write_json(root / "gate.json", gate)
    write_json(root / "reuse.json", dict(status="passed", matched_rows=12288))
    write_json(root / "preparation/RESOURCE_PREFLIGHT.json", dict(status="passed"))
    keys, _ = make_grid(64, windows=tuple(range(8, 129, 8)))
    write_json(root / "keys.json", keys)
    return dict(root=str(root), cpu_gate0=str(root / "gate.json"),
        reuse_identity_receipt=str(root / "reuse.json"), source_manifest="/source.json",
        source_identity=dict(bytes=1, sha256="source"), execution_commit="test",
        code_path=str(Path(campaign.__file__).resolve().parents[2]), live_input_identities={},
        training_host_required_bytes=80 * 1024**3, host_reserve_bytes=32 * 1024**3,
        peak_new_bytes=100_000_000_000, probe_keys=str(root / "keys.json"), parent_records={},
        serving_gpu=0, serving_gpu_identity="GPU-qualified")


def resources_fixture(jobs=""):
    return dict(host=dict(MemAvailable=500 * 1024**3),
        gpu=dict(devices=dict(status="passed", rows=[dict(index="0", uuid="GPU-qualified")]),
            compute_jobs=dict(status="passed", stdout=jobs,
            rows=[] if not jobs else [dict(pid="123", gpu_uuid="GPU-0")])) )


class AdmissionTests(unittest.TestCase):
    def preflight(self, plan, resources, source_path=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(campaign, "authenticate_source", return_value=dict(
                source_path=source_path or plan["code_path"])))
            stack.enter_context(patch.object(campaign, "check_identity"))
            stack.enter_context(patch.object(campaign, "lightweight_snapshot", return_value=resources))
            stack.enter_context(patch.object(campaign, "check_write", return_value=dict(status="passed")))
            stack.enter_context(patch("experiments.native_update_size.probe_client.resource_estimate",
                return_value=dict(conservative_precompaction_peak_bytes=1)))
            return campaign.preflight(plan)

    def test_failed_cpu_gate_stops_before_source_or_model_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = plan_fixture(root)
            for changed in (dict(status="failed", cuda_initialized=False, optimizer_steps=0),
                            dict(status="passed", cuda_initialized=True, optimizer_steps=0),
                            dict(status="passed", cuda_initialized=False, optimizer_steps=1)):
                write_json(plan["cpu_gate0"], changed)
                with patch.object(campaign, "authenticate_source") as source:
                    with self.assertRaises(ValueError):
                        campaign.preflight(plan)
                    source.assert_not_called()

    def test_one_existing_compute_process_is_not_mistaken_for_header(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_fixture(Path(folder))
            with self.assertRaises(RuntimeError):
                self.preflight(plan, resources_fixture("GPU-0, 123, python, 1024\n"))

    def test_selected_gpu_index_must_match_sealed_physical_uuid(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_fixture(Path(folder))
            resources = resources_fixture()
            resources["gpu"]["devices"]["rows"][0]["uuid"] = "GPU-other"
            with self.assertRaisesRegex(ValueError, "physical GPU"):
                self.preflight(plan, resources)
            result = self.preflight(plan, resources_fixture())
            self.assertEqual(result["status"], "passed")

    def test_manifest_source_must_be_current_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_fixture(Path(folder))
            with self.assertRaises(ValueError):
                self.preflight(plan, resources_fixture(), source_path="/different/source")

    def test_exact_required_host_ram_gate_is_applied(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = plan_fixture(Path(folder))
            resources = resources_fixture()
            resources["host"]["MemAvailable"] = 79 * 1024**3
            with self.assertRaises(MemoryError):
                self.preflight(plan, resources)

    def test_runtime_import_mismatch_blocks_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = plan_fixture(root)
            write_json(root / "preparation/RESOURCE_PREFLIGHT.json", dict(status="incomplete_or_mismatched"))
            with self.assertRaisesRegex(ValueError, "runtime/import"):
                self.preflight(plan, resources_fixture())

    def test_gate_failure_never_starts_gpu_or_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = plan_fixture(root)
            write_json(root / "plan.json", plan)
            budget = Mock()
            budget.gpu_seconds = 0
            budget.summary.return_value = {}
            with ExitStack() as stack:
                stack.enter_context(patch.object(campaign, "CampaignBudget", return_value=budget))
                service = stack.enter_context(patch.object(campaign, "BudgetService"))
                stack.enter_context(patch.object(campaign, "preflight", side_effect=ValueError("CPU Gate0 failed")))
                launch = stack.enter_context(patch.object(campaign, "launch_training"))
                scoring = stack.enter_context(patch("experiments.native_update_size.probe_client.run"))
                stack.enter_context(patch.object(campaign, "write_report", return_value=dict(status="incomplete")))
                stack.enter_context(patch.object(campaign.signal, "signal"))
                stack.enter_context(patch.object(campaign.signal, "setitimer"))
                with self.assertRaisesRegex(ValueError, "Gate0"):
                    campaign.run(root / "plan.json")
            budget.start_gpu.assert_not_called()
            launch.assert_not_called()
            scoring.assert_not_called()
            service.return_value.close.assert_called_once()
            self.assertEqual(json.loads((root / "STATUS.json").read_text())["status"], "failed")

    def test_native_budget_includes_one_reconstruction_and_one_engine_startup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {}
            for i in range(64):
                path = root / f"r{i}.json"
                write_json(path, dict(metadata=dict(trace=[{}] * 4)))
                paths[str(i)] = str(path)
            limits = campaign.limits(dict(parent_records=paths))
            self.assertEqual(limits["draft_block"], 8192 + 32 + 256 + 128)
            self.assertEqual(limits["model_load"], 40)

    def test_report_validation_failure_clears_completed_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = plan_fixture(root)
            plan["serving_gpu"] = 0
            write_json(root / "plan.json", plan)
            budget = Mock()
            budget.gpu_seconds = 0
            budget.summary.return_value = {}
            with ExitStack() as stack:
                stack.enter_context(patch.object(campaign, "CampaignBudget", return_value=budget))
                stack.enter_context(patch.object(campaign, "BudgetService"))
                stack.enter_context(patch.object(campaign, "preflight", return_value=dict(status="passed")))
                stack.enter_context(patch.object(campaign, "launch_training", return_value=dict(probe_exports=[])))
                stack.enter_context(patch("experiments.native_update_size.probe_client.run"))
                stack.enter_context(patch.object(campaign, "write_report", side_effect=ValueError("Invalid outcome grid")))
                stack.enter_context(patch.object(campaign.signal, "signal"))
                stack.enter_context(patch.object(campaign.signal, "setitimer"))
                with self.assertRaisesRegex(ValueError, "Invalid outcome grid"):
                    campaign.run(root / "plan.json")
            status = json.loads((root / "STATUS.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertTrue((root / "REPORT_FAILURE.json").is_file())


class ReportingTests(unittest.TestCase):
    def completed_receipts(self, root):
        training = dict(qualification=dict(status="passed"),
            windows=[dict(window=w, status="completed") for w in range(8, 129, 8)],
            final_native_parity=dict(status="passed"), calibration_trials=[], geometry=[])
        scoring = dict(reuse_sentinel_status="passed", new_logical_control_rows=8192,
            invalid_control_rows=0, unknown_control_rows=0)
        write_json(root / "training/RESULT.json", training)
        write_json(root / "probes/STATUS.json", scoring)
        return training, scoring

    def report(self, plan, receipt):
        budget = Mock()
        budget.summary.return_value = dict(completed={}, failed={}, unknown={}, gpu_seconds=1)
        return campaign.write_report(plan, receipt, budget)

    def test_completed_grid_uses_actual_qualification_status_and_all_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = plan_fixture(root)
            self.completed_receipts(root)
            _, rows = make_grid(64, windows=tuple(range(8, 129, 8)))
            write_json(root / "probes/PROBE_OUTCOMES.json", rows)
            result = self.report(plan, dict(status="completed", phases={}, wall_s=1))
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["validity"]["gates"]["native_six_case_qualification"])
            self.assertEqual(result["comparison"]["paired_block_windows"], 4096)

    def test_incomplete_append_only_rows_are_preserved_without_final_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = plan_fixture(root)
            _, rows = make_grid(64, windows=tuple(range(8, 129, 8)))
            partial = rows[:2] + [dict(rows[2], valid=False, status="unknown")]
            (root / "probes").mkdir()
            (root / "probes/PROBE_OUTCOMES.jsonl").write_text("".join(json.dumps(r) + "\n" for r in partial))
            result = self.report(plan, dict(status="failed", phases={}, wall_s=1, error="interrupted"))
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["grid"]["recorded"], 3)
            self.assertEqual(result["grid"]["unknown"], 1)
            self.assertNotIn("comparison", result)

    def test_bad_grid_cannot_leave_completed_campaign_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = plan_fixture(root)
            self.completed_receipts(root)
            _, rows = make_grid(64, windows=tuple(range(8, 129, 8)))
            write_json(root / "probes/PROBE_OUTCOMES.json", rows[:-1])
            with self.assertRaises(ValueError):
                self.report(plan, dict(status="completed", phases={}, wall_s=1))
            result = json.loads((root / "final_report/SUMMARY.json").read_text())
            self.assertEqual(result["status"], "invalid")
            self.assertNotIn("comparison", result)

    def test_failed_qualification_blocks_efficacy_even_if_rows_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = plan_fixture(root)
            training, _ = self.completed_receipts(root)
            training["qualification"]["status"] = "failed"
            write_json(root / "training/RESULT.json", training)
            _, rows = make_grid(64, windows=tuple(range(8, 129, 8)))
            write_json(root / "probes/PROBE_OUTCOMES.json", rows)
            result = self.report(plan, dict(status="failed", phases={}, wall_s=1))
            self.assertEqual(result["status"], "incomplete")
            self.assertNotIn("comparison", result)


if __name__ == "__main__":
    unittest.main()
