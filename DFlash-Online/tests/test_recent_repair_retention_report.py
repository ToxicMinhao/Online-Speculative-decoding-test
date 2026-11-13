"""Reporting must not turn sparse/invalid evidence into a performance claim."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.recent_repair_retention import report as R


def observation(pid, value):
    return dict(record_id=f"train_{pid:05d}", **{"pass": 0}, optimizer_step_after=768+(pid+1)//2,
        optimizer_stepped=bool(pid % 2), training_s=1., metrics={"retention_firing_count": 10000},
        response_observation=dict(record_id=f"train_{pid:05d}", presentation_ordinal=pid,
            sum_metrics=dict(kplus_response_count=1, kplus_N_K=2., kplus_S=4.,
                retention_coefficient=.1, retention_firing_count=value,
                retention_auxiliary_loss=.25)))


def setup(root, status="failed"):
    plan = dict(root=str(root), execution_commit="a"*40, panel_manifest=str(root/"PANEL.json"),
        timing_schedule=str(root/"SCHEDULE.json"), path_request_ids=["eval1"],
        qualification_root=str(root/"qualification"))
    R.write(root/"sealed/PLAN.json", plan)
    R.write(root/"STATUS.json", dict(status=status, phase="training", execution_commit="a"*40, wall_s=10.))
    R.write(root/"PANEL.json", dict(requests=[dict(request_id="eval1", source_id="gsm8k:main:test:1",
        prompt_ids=[999], question="DO NOT PUBLISH PRIVATE TEXT")], selected_tier_counts={"1":1}))
    R.write(root/"SCHEDULE.json", [])
    return plan


class RetentionReportTests(unittest.TestCase):
    def test_response_not_accumulation_rows_and_duplicate_rejected(self):
        records = [observation(0, 2), observation(1, 3)]
        rows, totals = R.response_exposure(records, "RECENT")
        self.assertEqual(totals["retention_firing_count"], 5)
        self.assertEqual(totals["retention_auxiliary_loss"], .5)
        self.assertEqual(totals["retention_coefficient"], .1)
        self.assertEqual(totals["adam_steps"], 1)
        self.assertAlmostEqual(rows[0]["ordinary_selected_kl"], 2/(4+1e-6))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            R.response_exposure(records + [records[0]], "RECENT")

    def test_incomplete_report_preserves_counts_without_gain_or_private_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setup(root)
            R.write(root/"evaluation/STATUS.json", dict(status="failed", planned_generations=6,
                attempted_generations=1, returned_generations=1, invalid_generations=1,
                unattempted_generations=5))
            row = dict(request_id="eval1", method="NATIVE", repeat=0, output_tokens=1,
                verification_calls=1, valid_measurement=False, errors=["private prompt"],
                canonical_token=999, rival_token=333, prompt_ids=[999])
            (root/"evaluation/MEASUREMENTS.jsonl").write_text(json.dumps(row)+"\n")
            R.write(root/"ASSIGNMENTS.json", dict(eligibility=[dict(request_id="train_1",
                status="eligible", canonical_token=999, rival_token=333)], arms={}, summary={}))
            summary = R.generate_report(root)
            self.assertFalse(summary["complete_valid_comparison"])
            self.assertEqual(summary["comparisons"], {})
            self.assertEqual(summary["serving_audit"]["progress"]["unattempted_generations"], 5)
            text = "\n".join(path.read_text() for path in (root/"final_report").iterdir())
            for private in ("DO NOT PUBLISH PRIVATE TEXT", '"prompt_ids"', '"canonical_token"',
                            '"rival_token"', "private prompt"):
                self.assertNotIn(private, text)

    def test_feasibility_stop_never_requires_missing_serving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setup(root, "completed_no_treatment")
            result = R.generate_report(root)
            self.assertEqual(result["decision"]["decision"], "no_training_eligible_slots")
            self.assertFalse(result["complete_valid_comparison"])

    def test_complete_endpoint_uses_clipped_acceptance_and_two_sweep_means(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            setup(root, "completed")
            R.write(root/"EXPORTS.json", {arm:"/"+arm for arm in R.ARMS})
            R.write(root/"canonicals/CANONICAL.json", {})
            R.write(root/"evaluation/STATUS.json", dict(status="completed"))
            R.write(root/"fixed_starts/STATUS.json", dict(status="completed"))
            R.write(root/"fixed_starts/STARTS.json", [])
            R.write(root/"fixed_starts/OUTCOMES.json", [])
            for case in ("NATIVE","DISABLED","ZERO","ACTIVE"):
                R.write(root/"qualification"/case/"RESULT.json",dict(status="passed"))
            rows=[]
            for arm in R.ARMS:
                for repeat in (0,1):
                    rows.append(dict(request_id="eval1",method=arm,repeat=repeat,valid_measurement=True,
                        errors=[],verification_calls=2,accepted_draft_tokens=3,native_accepted_draft_tokens=9,
                        output_tokens=5,committed_progress=5,finish_reason="eos",request_latency_s=2.+repeat,
                        native_generate_s=1.+repeat,cache_maintenance_s=1.))
            trained={arm:dict(status="completed",response_presentations=256,adam_steps=128,
                final_vs_native=dict(status="passed")) for arm in R.ARMS}
            with patch.object(R.evaluation,"audit_stage",return_value=dict(status="passed",errors=[],
                    rows=rows,progress=dict(status="completed"))), patch.object(R,"training_evidence",
                        return_value=([],[],trained,[],[])):
                result=R.generate_report(root)
            self.assertTrue(result["complete_valid_comparison"])
            self.assertEqual(result["comparisons"]["NATIVE"]["recent_accepted"],3)
            self.assertEqual(result["comparisons"]["NATIVE"]["recent_accepted_per_call"],1.5)
            self.assertEqual(result["comparisons"]["NATIVE"]["recent_request_s"],2.5)
            self.assertEqual(result["sweep_totals"][0]["request_latency_s"],2.)
            self.assertEqual(result["fixed_starts"]["contrasts"]["NATIVE"]["exact_slot_relapse"]
                             ["interval"]["undefined_draws"],10000)
            self.assertEqual(R.read(root/"final_report/REPORT_SELF_CHECKS.json")["status"],"passed")

    def test_bad_raw_audit_blocks_otherwise_completed_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            setup(root,"completed")
            for name in ("EXPORTS.json","canonicals/CANONICAL.json","evaluation/STATUS.json"):
                R.write(root/name,{})
            with patch.object(R.evaluation,"audit_stage",return_value=dict(status="invalid_or_incomplete",
                    errors=["raw output mismatch"],rows=[],progress={})):
                result=R.generate_report(root)
            self.assertFalse(result["complete_valid_comparison"])
            self.assertEqual(result["comparisons"],{})

    def test_partial_response_observations_do_not_infer_adam_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            setup(root)
            observed=observation(0,1)["response_observation"]
            observed["pass_index"]=0
            path=root/"training/RECENT/trainer/events/event_007/RESPONSE_OBSERVATIONS.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(observed)+"\n")
            values=R.training_evidence(root)
            self.assertEqual(len(values[0]),1)
            self.assertIsNone(values[2]["RECENT"]["adam_steps"])

    def test_evaluation_recovery_preserves_commits_and_counts_prior_cost_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"new"
            root.mkdir()
            setup(root)
            prior=Path(directory)/"old"
            R.write(prior/"final_report/SUMMARY.json",dict(resources=dict(measured_allocated_phase_window_s=100.)))
            R.write(root/"STATUS.json",dict(status="failed",wall_s=40.,combined_campaign_wall_s=240.))
            R.write(root/"CONTINUATION.json",dict(previous_execution_root=str(prior),
                training_execution_commit="b"*40,evaluation_execution_commit="a"*40,
                prior_campaign_wall_s=200.,prior_fixed_start_gpu_s=24.23562198318541,
                failed_startup_work=dict(completed=dict(draft_block=51,target_call=48)),
                actual_parent_generations_before=0,pending_parent_generations=64,
                pending_timed_completions=1536,new_optimizer_steps=0))
            R.write(root/"processes/fixed_starts.json",dict(wall_s=10.))
            R.write(root/"SERVING_RESULT.json",dict(wall_s=20.))
            R.write(root/"canonicals/STATUS.json",dict(wall_s=5.))
            result=R.generate_report(root)
            self.assertEqual(result["training_execution_commit"],"b"*40)
            self.assertEqual(result["evaluation_execution_commit"],"a"*40)
            self.assertEqual(result["combined_campaign_wall_s"],240.)
            self.assertEqual(result["resources"]["measured_allocated_phase_window_s"],130.)
            with (root/"final_report/costs.csv").open() as stream:
                costs=list(csv.DictReader(stream))
            failed=[r for r in costs if r["phase"]=="previous_failed_fixed_start" and r["metric"]=="gpu_seconds"]
            self.assertEqual(len(failed),1)
            self.assertFalse(result["complete_valid_comparison"])


if __name__ == "__main__":
    unittest.main()
