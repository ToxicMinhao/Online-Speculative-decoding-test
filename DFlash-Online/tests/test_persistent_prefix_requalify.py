"""Qualification-only retry tests; all native child work is mocked."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from experiments.persistent_prefix_withholding import requalify as R
from experiments.persistent_prefix_withholding import qualification as Q
from experiments.persistent_prefix_withholding import serving_qualification as S
from experiments.persistent_prefix_withholding import admission as A
from experiments.persistent_prefix_withholding import campaign as C
from experiments.persistent_prefix_withholding.resources import LIMITS, ELAPSED_SECONDS
from online_feedback.storage import write_json
from test_persistent_prefix_launch import training_receipts, serving_receipts


def write_ledger(path, counts, unknown=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    serial = 0
    for kind, count in counts.items():
        for _ in range(count):
            serial += 1
            rows.extend([dict(serial=serial, kind=kind, phase="started"),
                         dict(serial=serial, kind=kind, phase="completed")])
    if unknown:
        rows.append(dict(serial=serial+1, kind=unknown, phase="started"))
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


class RequalificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.old, self.root = base/"failed", base/"retry"
        self.root.mkdir()
        self.plan = dict(root=str(self.root), source_root=str(base), execution_commit="b"*40,
            training_python="/qualified/python", devices={"A":"GPU-A", "B":"GPU-B"},
            compiler_root="/same/cache", compiler_baseline_allocated_bytes=5022478336,
            compiler_growth_reserved_bytes=4*2**30)
        prior_plan = dict(self.plan, root=str(self.old), execution_commit="a"*40)
        write_json(self.old/"sealed/PLAN.json", prior_plan)
        self.old_clock = dict(first_model_load_epoch=time.time()-1000,
            completed_card_s=89.01599144935608, active={},
            finished=[dict(occupied_card_s=89.01599144935608)])
        write_json(self.old/"CLOCK.json", self.old_clock)
        write_json(self.old/"LAUNCH.json", dict(status="stopped", execution_commit="a"*40,
                   completed_phases=["cpu_preflight"]))
        counts = dict(model_load=1, response_forward=6, response_backward=6, optimizer_step=2)
        write_ledger(self.old/"qualification/A/PHYSICAL_WORK.jsonl", counts)
        write_json(self.old/"qualification/A/STATUS.json", dict(status="failed",
            execution_commit="a"*40, no_production_training=True, no_export=True,
            complete_generations=0, physical_work=dict(attempted=counts, completed=counts, failed={})))
        authorization = base/"AUTHORIZATION.json"
        write_json(authorization, dict(scope="qualification_only", prior_root=str(self.old),
            new_root=str(self.root), new_adam_steps=8, new_response_forwards=20,
            new_response_backwards=20, new_initializations=2, new_generations=8,
            preserve_original_clock=True, main_campaign_authorized=False))
        self.plan["prior_qualification_attempt"] = dict(root=str(self.old), execution_commit="a"*40)
        self.plan["qualification_retry_authorization"] = dict(path=str(authorization),
                                                              identity=R.file_identity(authorization))
        self.plan_path = self.root/"sealed/PLAN.json"
        self.reseal()
        storage = patch.object(R, "storage_boundary", return_value=dict(status="passed"))
        self.storage = storage.start()
        self.addCleanup(storage.stop)

    def reseal(self):
        self.plan["prior_qualification_attempt"]["files"] = {
            name:R.file_identity(self.old/name) for name in R.PRIOR_FILES}
        write_json(self.plan_path, self.plan)

    def cpu_pass(self, *args, **kwargs):
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        write_json(self.root/"CPU_PREFLIGHT.json", dict(status="passed",
                                                     execution_commit=self.plan["execution_commit"]))

    def child_pass(self, path, plan, kind, slot):
        if kind == "training":
            report = training_receipts()[slot]
            write_ledger(self.root/"qualification"/slot/"PHYSICAL_WORK.jsonl",
                         report["physical_work"]["completed"])
        else:
            report = serving_receipts()[slot]
            write_json(self.root/"serving_qualification"/slot/"calls/RESULT.json",
                       report["qualification"])
        return report

    def patches(self):
        # Every test retains the production helper, but never calls a real child.
        return patch.object(R.subprocess, "run", side_effect=self.cpu_pass)

    def test_success_carries_old_work_and_clock_but_never_starts_main(self):
        originals = {name:(self.old/name).read_bytes() for name in R.PRIOR_FILES}
        limits = dict(LIMITS)
        with self.patches(), patch.object(R, "run_qualification_child", side_effect=self.child_pass) as child, \
                patch.object(Q, "compare_cards", return_value=dict(status="passed")), \
                patch.object(S, "compare_serving_cards", return_value=dict(status="passed")), \
                patch.object(A, "qualify_resources", return_value=dict(status="passed")), \
                patch.object(C, "run_campaign") as production:
            result = R.run(self.plan_path)
        self.assertEqual(result["status"], "qualified")
        self.assertEqual([call.args[-2:] for call in child.call_args_list],
                         [("training","A"),("training","B"),("serving","A"),("serving","B")])
        production.assert_not_called()
        self.assertFalse((self.root/"WORK.json").exists())
        self.assertTrue((self.root/"REQUALIFICATION_WORK.json").exists())
        self.assertEqual(R.read(self.root/"CLOCK.json"), self.old_clock)
        work = R.read(self.root/"CUMULATIVE_WORK.json")
        expected = dict(adam=10, response_forward=26, response_backward=26,
                        training_initialization=3, qualification_generation=8)
        for key, value in expected.items():
            self.assertEqual(work["cumulative"]["completed"][key], value)
        self.assertEqual(work["unresolved_sources"], [])
        self.assertTrue(all(value == 0 for value in work["unattempted"].values()))
        self.assertEqual({name:(self.old/name).read_bytes() for name in R.PRIOR_FILES}, originals)
        self.assertEqual(LIMITS, limits)
        self.assertEqual(R.read(self.plan_path)["compiler_baseline_allocated_bytes"], 5022478336)
        self.assertEqual(self.storage.call_count, 5)

    def test_tampered_evidence_or_wrong_authorization_stops_before_cpu(self):
        (self.old/"CLOCK.json").write_text("{}")
        with patch.object(R.subprocess, "run") as cpu, patch.object(R, "run_qualification_child") as child:
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                R.run(self.plan_path)
        cpu.assert_not_called()
        child.assert_not_called()
        self.assertFalse((self.root/"CLOCK.json").exists())

    def test_unknown_prior_work_and_active_leases_are_rejected(self):
        write_ledger(self.old/"qualification/A/PHYSICAL_WORK.jsonl",
            dict(model_load=1, response_forward=6, response_backward=6, optimizer_step=2),
            unknown="response_forward")
        self.reseal()
        with self.assertRaisesRegex(ValueError, "unknown work"):
            R.authenticate_prior(self.plan)
        write_json(self.old/"CLOCK.json", dict(self.old_clock, active={"worker":{}}))
        self.reseal()
        with self.assertRaisesRegex(ValueError, "active leases"):
            R.authenticate_prior(self.plan)

    def test_compiler_baseline_cannot_be_reset(self):
        self.plan["compiler_baseline_allocated_bytes"] += 16523264
        with self.assertRaisesRegex(ValueError, "compiler storage baseline"):
            R.authenticate_prior(self.plan)

    def test_failed_cpu_preflight_launches_no_gpu(self):
        with patch.object(R.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "cpu")), \
                patch.object(R, "run_qualification_child") as child:
            with self.assertRaises(subprocess.CalledProcessError):
                R.run(self.plan_path)
        child.assert_not_called()
        work = R.read(self.root/"CUMULATIVE_WORK.json")
        self.assertEqual(work["cumulative"]["completed"]["adam"], 2)
        self.assertEqual(work["attempt"]["attempted"]["adam"], 0)
        self.assertEqual(R.read(self.root/"REQUALIFICATION.json")["status"], "stopped")

    def test_wrong_source_cpu_receipt_is_not_a_pass(self):
        def wrong(*args, **kwargs):
            write_json(self.root/"CPU_PREFLIGHT.json", dict(status="passed", execution_commit="other"))
        with patch.object(R.subprocess, "run", side_effect=wrong), \
                patch.object(R, "run_qualification_child") as child:
            with self.assertRaisesRegex(ValueError, "execution source"):
                R.run(self.plan_path)
        child.assert_not_called()

    def test_expired_original_clock_is_not_reset(self):
        self.old_clock["first_model_load_epoch"] = time.time()-ELAPSED_SECONDS-1
        write_json(self.old/"CLOCK.json", self.old_clock)
        self.reseal()
        with self.patches(), patch.object(R, "run_qualification_child") as child:
            with self.assertRaisesRegex(TimeoutError, "Original qualification clock"):
                R.run(self.plan_path)
        child.assert_not_called()
        self.assertEqual(R.read(self.root/"CLOCK.json"), self.old_clock)

    def test_native_failure_retains_partial_work_and_does_not_retry(self):
        def fail(path, plan, kind, slot):
            write_ledger(self.root/"qualification/A/PHYSICAL_WORK.jsonl",
                         dict(model_load=1, response_forward=4, response_backward=4, optimizer_step=1),
                         unknown="response_forward")
            raise RuntimeError("native failure")
        with self.patches(), patch.object(R, "run_qualification_child", side_effect=fail) as child:
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                R.run(self.plan_path)
        child.assert_called_once()
        work = R.read(self.root/"CUMULATIVE_WORK.json")
        self.assertEqual(work["cumulative"]["completed"]["adam"], 3)
        self.assertEqual(work["cumulative"]["completed"]["response_forward"], 10)
        self.assertEqual(work["cumulative"]["unknown"]["response_forward"], 1)
        with patch.object(R, "run_qualification_child") as repeated:
            with self.assertRaises(FileExistsError):
                R.run(self.plan_path)
        repeated.assert_not_called()

    def test_cross_card_failure_stops_before_serving(self):
        with self.patches(), patch.object(R, "run_qualification_child", side_effect=self.child_pass) as child, \
                patch.object(Q, "compare_cards", side_effect=ValueError("cross-card")):
            with self.assertRaisesRegex(ValueError, "cross-card"):
                R.run(self.plan_path)
        self.assertEqual(child.call_count, 2)
        work = R.read(self.root/"CUMULATIVE_WORK.json")
        self.assertEqual(work["cumulative"]["completed"]["adam"], 10)
        self.assertEqual(work["cumulative"]["completed"]["qualification_generation"], 0)

    def test_missing_started_child_ledger_is_unknown_not_unattempted(self):
        work = R.accounting(self.root, R.authenticate_prior(self.plan), ["training_A"])
        self.assertTrue(work["counts_are_lower_bounds"])
        self.assertIsNone(work["unattempted"]["adam"])
        self.assertEqual(work["cumulative"]["completed"]["adam"], 2)

    def test_storage_gate_stops_before_native_work(self):
        self.storage.side_effect = RuntimeError("compiler growth")
        with self.patches(), patch.object(R, "run_qualification_child") as child:
            with self.assertRaisesRegex(RuntimeError, "compiler growth"):
                R.run(self.plan_path)
        child.assert_not_called()
        self.assertIn("compiler growth", R.read(self.root/"REQUALIFICATION.json")["final_storage_error"])

    def test_final_storage_failure_preserves_receipts_and_returns_failure(self):
        self.storage.side_effect = [dict(status="passed")]*4+[RuntimeError("final storage")]
        with self.patches(), patch.object(R, "run_qualification_child", side_effect=self.child_pass), \
                patch.object(Q, "compare_cards", return_value=dict(status="passed")), \
                patch.object(S, "compare_serving_cards", return_value=dict(status="passed")), \
                patch.object(A, "qualify_resources", return_value=dict(status="passed")):
            with self.assertRaisesRegex(RuntimeError, "final storage or accounting"):
                R.run(self.plan_path)
        result = R.read(self.root/"REQUALIFICATION.json")
        self.assertEqual(result["status"], "stopped")
        self.assertIn("final storage", result["final_storage_error"])
        self.assertEqual(R.read(self.root/"CUMULATIVE_WORK.json")["cumulative"]["completed"]["adam"], 10)


if __name__ == "__main__":
    unittest.main()
