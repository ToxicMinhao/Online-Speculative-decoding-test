"""Private-data boundaries and complete/partial report contracts."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.lost_repair.report import report


def small_fixture():
    census = [dict(request_id="r0", canonical_start=start, a4=8, a5=a5, a6=8,
                   valid_proposals=15, genuine_k6_rejection=True, parity_valid=True)
              for start, a5 in ((0, 15), (16, 8))]
    rows = [dict(request_id="r0", canonical_start=start, presentation_id=pid, j=9,
                 priority="high", first_error_weight=2.0, wrong_margin=1.0)
            for pid in (0, 1) for start in (0, 16)]
    presentations = [dict(request_id="r0", presentation_id=pid, selected_weight=20.0,
                          expected_anchor_count=2, first_error_weight=4.0) for pid in (0, 1)]
    return census, rows, presentations


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_run(root, *, complete=True):
    census, rows, presentations = small_fixture()
    plan = dict(requests=["r0"], execution_commit="a" * 40,
                checkpoint_identities={"KPLUS6": "b" * 64},
                manifest_identity={"sha256": "c" * 64},
                source_manifest_path="/private/source/MANIFEST.json", gpu_seconds_limit=14400)
    save(root / "PLAN.json", plan)
    save(root / "census/CENSUS.json", census)
    save(root / "training/RESULT.json", dict(status="completed", training_rows=rows,
                                             presentation_rows=presentations))
    save(root / "STATUS.json", dict(status="completed" if complete else "failed",
                                    census_complete=complete, training_complete=complete))
    save(root / "BUDGET.json", dict(occupied_gpu_seconds=123, limit_seconds=14400))
    save(root / "RESULT.json", dict(phase_costs={"native_replay_s": 100},
                                   work={"optimizer_steps": 0, "training_forwards": 4}))
    return plan


class ReportTests(unittest.TestCase):
    def test_complete_receipts_emit_every_required_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_run(root)
            outcome = report(root / "PLAN.json", root)
            self.assertEqual(outcome["status"], "COMPLETE")
            self.assertEqual(outcome["gate_decision"], "FAIL")
            destination = root / "final_report"
            self.assertEqual({p.name for p in destination.iterdir()}, {
                "REPORT.md", "PER_REQUEST.csv", "ANCHOR_COVERAGE.csv", "NATIVE_WEIGHTS.csv",
                "PRESENTATION_WEIGHTS.csv", "MATCHED_PAIRS.csv", "GATE.json", "SUMMARY.json",
            })
            summary = json.loads((destination / "SUMMARY.json").read_text())
            self.assertEqual(summary["provenance"]["execution_commit"], "a" * 40)
            self.assertEqual(summary["costs"]["phase_costs"]["native_replay_s"], 100)
            self.assertEqual(summary["costs"]["work"]["optimizer_steps"], 0)
            self.assertEqual(summary["evidence_missing"], [])
            self.assertEqual(summary["native_H_weight_before_matching"], 4)
            self.assertIn("no independent native replication", (destination / "REPORT.md").read_text())

    def test_missing_training_preserves_census_and_does_not_pass_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = write_run(root)
            (root / "training/RESULT.json").unlink()
            outcome = report(plan, root)
            self.assertEqual(outcome["gate_decision"], "INCOMPLETE")
            summary = json.loads((root / "final_report/SUMMARY.json").read_text())
            self.assertEqual(summary["H_observed_count"], 1)
            self.assertEqual(summary["full_natural_denominator"], 2)
            self.assertEqual(summary["observed_presentations"], 0)
            self.assertTrue(all(v["passed"] is None for v in summary["gate"].values()))
            with (root / "final_report/NATIVE_WEIGHTS.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_partial_census_keeps_observed_numerator_and_missing_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = write_run(root, complete=False)
            census = json.loads((root / "census/CENSUS.json").read_text())
            save(root / "census/CENSUS.json", census[:1])
            report(plan, root)
            summary = json.loads((root / "final_report/SUMMARY.json").read_text())
            self.assertEqual(summary["H_observed_count"], 1)
            self.assertEqual(summary["observed_natural_starts"], 1)
            self.assertIsNone(summary["full_natural_denominator"])
            self.assertIsNone(summary["H_share_all_natural_starts"])
            self.assertEqual(summary["gate_decision"], "INCOMPLETE")

    def test_missing_final_training_recovers_durable_presentations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = write_run(root)
            training = json.loads((root / "training/RESULT.json").read_text())
            (root / "training/RESULT.json").unlink()
            save(root / "training/presentations/000.json", dict(
                training_rows=training["training_rows"][:2],
                presentation=training["presentation_rows"][0]))
            report(plan, root)
            summary = json.loads((root / "final_report/SUMMARY.json").read_text())
            self.assertEqual(summary["observed_presentations"], 1)
            self.assertEqual(summary["native_H_weight_before_matching"], 2)
            self.assertEqual(summary["native_first_error_weight"], 4)
            self.assertEqual(summary["selected_weight"], 20)
            self.assertEqual(summary["projected_matched_weight"], 0.5)
            self.assertEqual(summary["full_natural_denominator"], 2)
            self.assertIsNone(summary["full_native_first_error_weight"])
            self.assertIsNone(summary["projected_weight_fraction_all_first_error"])
            self.assertEqual(summary["gate_decision"], "INCOMPLETE")
            self.assertIn("training/presentations/000.json", summary["provenance"]["durable_presentation_file_identities"])
            with (root / "final_report/NATIVE_WEIGHTS.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_all_durable_presentations_without_final_receipt_remain_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = write_run(root)
            training = json.loads((root / "training/RESULT.json").read_text())
            (root / "training/RESULT.json").unlink()
            for pid in (0, 1):
                save(root / f"training/presentations/{pid:03}.json", dict(
                    training_rows=training["training_rows"][pid * 2:pid * 2 + 2],
                    presentation=training["presentation_rows"][pid]))
            report(plan, root)
            summary = json.loads((root / "final_report/SUMMARY.json").read_text())
            self.assertEqual(summary["observed_presentations"], 2)
            self.assertEqual(summary["native_first_error_weight"], 8)
            self.assertEqual(summary["gate_decision"], "INCOMPLETE")
            self.assertIsNone(summary["full_native_first_error_weight"])

    def test_final_training_receipt_not_double_counted_with_durable_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = write_run(root)
            training = json.loads((root / "training/RESULT.json").read_text())
            save(root / "training/presentations/000.json", dict(
                training_rows=training["training_rows"][:2],
                presentation=training["presentation_rows"][0]))
            report(plan, root)
            summary = json.loads((root / "final_report/SUMMARY.json").read_text())
            self.assertEqual(summary["observed_presentations"], 2)
            self.assertEqual(summary["native_first_error_weight"], 8)
            self.assertEqual(summary["status"], "COMPLETE")

    def test_actual_plan_checkpoint_mapping_is_published(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = write_run(root)
            plan["checkpoints"] = {"K6": dict(export="/private/checkpoint", weights_identity={"sha256": "d" * 64})}
            report(plan, root)
            summary = json.loads((root / "final_report/SUMMARY.json").read_text())
            self.assertEqual(summary["provenance"]["checkpoint_identities"], plan["checkpoints"])

    def test_missing_cost_receipts_remain_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = write_run(root)
            (root / "BUDGET.json").unlink()
            report(plan, root)
            summary = json.loads((root / "final_report/SUMMARY.json").read_text())
            self.assertTrue(summary["costs"]["unavailable"])
            self.assertEqual(summary["costs"]["occupied_gpu_budget"], {})
            self.assertEqual(summary["gate_decision"], "INCOMPLETE")

    def test_private_tokens_and_payloads_are_not_exported(self):
        secret = "PRIVATE_PROMPT_CANARY_7395"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = write_run(root)
            training = json.loads((root / "training/RESULT.json").read_text())
            for row in training["training_rows"] + training["presentation_rows"]:
                row.update(prompt=secret, canonical_token=938432, wrong_token=812373,
                           proposals=[84373, 789732], sampled_anchors={"private": secret})
            save(root / "training/RESULT.json", training)
            save(root / "RESULT.json", dict(phase_costs={"native_replay_s": 100},
                                             work={"optimizer_steps": 0}, private_prompt=secret))
            report(plan, root)
            for path in (root / "final_report").iterdir():
                text = path.read_text()
                self.assertNotIn(secret, text)
                self.assertNotIn("938432", text)
                self.assertNotIn("812373", text)
                self.assertNotIn("789732", text)
            with (root / "final_report/NATIVE_WEIGHTS.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)

    def test_malformed_or_identity_invalid_evidence_still_reports_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = write_run(root)
            training = json.loads((root / "training/RESULT.json").read_text())
            training["training_rows"].append(dict(training["training_rows"][0]))
            save(root / "training/RESULT.json", training)
            outcome = report(plan, root)
            self.assertEqual(outcome["gate_decision"], "INCOMPLETE")
            summary = json.loads((root / "final_report/SUMMARY.json").read_text())
            self.assertTrue(summary["evidence_errors"])
            self.assertIsNone(summary["H_observed_count"])
            self.assertIn("Duplicate sampled anchor", (root / "final_report/REPORT.md").read_text())

    def test_invalid_json_is_named_not_imputed_zero_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = write_run(root)
            (root / "training/RESULT.json").write_text("{\"unfinished\":")
            report(plan, root)
            summary = json.loads((root / "final_report/SUMMARY.json").read_text())
            self.assertTrue(summary["evidence_errors"])
            self.assertEqual(summary["gate_decision"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
