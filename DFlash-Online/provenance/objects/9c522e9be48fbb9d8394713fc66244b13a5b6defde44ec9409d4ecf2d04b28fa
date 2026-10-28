"""Report contracts: matched denominators, failure visibility and nonoverlapping costs."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.projected_target import osd_report as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    source = root / "original"
    requests = [{"request_id": name, "role": role, "prompt_ids": [10, index]}
                for index, (name, role) in enumerate((("a1", "A"), ("a2", "A"), ("b1", "B")))]
    write(root / "manifest.json", {"groups": [{"name": "set", "requests": requests}]})
    write(root / "PLAN.json", {"source_campaign": str(source), "output_cap": 101})
    write(root / "STATUS.json", {"status": "completed", "wall_s": 80})
    write(root / "set/QUALIFICATION.json", {
        "status": "passed", **dict.fromkeys(report.QUALIFICATION_FIELDS, True),
        "arms": {"O": dict.fromkeys(report.ARM_CHECKS, True)},
    })
    for arm, calls, times in (("F", [10, 100, 10], [1, 9, 2]), ("O", [5, 90, 20], [0.8, 8.2, 3])):
        for request, count, latency in zip(requests, calls, times):
            record = {
                "record_id": request["request_id"], "prompt_ids": request["prompt_ids"],
                "response_ids": [1] * 101, "finish_reason": "length", "stop_token_ids": [99],
                "max_new_tokens": 101, "metadata": {"verification_calls": count,
                    "accepted_draft_tokens": 100 - count, "committed_progress": 100,
                    "request_latency_s": latency},
            }
            write(root / "set" / arm / "records" / (request["request_id"] + ".json"), record)
            if arm == "F":
                record["metadata"]["request_latency_s"] = 999  # Historical timing must never enter fresh ratios.
                write(source / "set/F/records" / (request["request_id"] + ".json"), record)
    return root


def comparison(summary, role="A", group="ALL"):
    return next(r for r in summary["comparisons"] if r["role"] == role and r["group"] == group)


class OSDFollowupReportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = fixture(Path(directory.name))

    def mutate(self, relative, edit):
        path = self.root / relative
        value = json.loads(path.read_text())
        edit(value)
        write(path, value)

    def rows(self):
        with (self.root / "PER_REQUEST.csv").open() as handle:
            return list(csv.DictReader(handle))

    def test_summed_a_b_ratios_and_fresh_timing(self):
        summary = report.build_report(self.root)
        self.assertTrue(summary["all_planned_valid"])
        a, b = comparison(summary), comparison(summary, "B")
        self.assertAlmostEqual(a["call_change_pct"], 100 * (95 / 110 - 1))
        self.assertAlmostEqual(a["request_time_change_pct"], -10)
        self.assertEqual((a["gains"], a["regressions"], a["ties"]), (2, 0, 0))
        self.assertEqual(b["call_change_pct"], 100)
        self.assertEqual(b["request_time_change_pct"], 50)
        self.assertEqual(a["F_request_latency_s_sum"], 10)
        self.assertAlmostEqual(a["O_calls_per_1000_output_tokens"], 95000 / 202)
        self.assertAlmostEqual(a["O_committed_progress_per_call"], 200 / 95)
        self.assertAlmostEqual(a["O_accepted_draft_tokens_per_call"], 105 / 95)
        self.assertEqual(a["O_cap_hits"], 2)
        for name in ("PER_REQUEST.csv", "POOLED.csv", "SUMMARY.json", "COSTS.json", "REPORT.md"):
            self.assertTrue((self.root / name).is_file())

    def test_missing_measurement_has_no_full_planned_gain(self):
        (self.root / "set/O/records/a2.json").unlink()
        summary = report.build_report(self.root)
        a = comparison(summary)
        self.assertEqual((a["planned_pairs"], a["valid_pairs"], a["invalid_or_missing_pairs"]), (2, 1, 1))
        self.assertIsNone(a["call_change_pct"])
        self.assertIsNone(a["request_time_change_pct"])
        self.assertEqual(a["available_valid_pairs_call_change_pct"], -50)
        self.assertEqual(summary["present_generations"], 5)
        self.assertEqual(len(self.rows()), 6)
        missing = next(r for r in self.rows() if r["arm"] == "O" and r["request_id"] == "a2")
        self.assertIn("missing_file", missing["validity_reason"])

    def test_measurement_validator_and_output_equality_are_reapplied(self):
        edits = [
            (lambda x: x["response_ids"].__setitem__(5, 2), "reference_response_ids_mismatch"),
            (lambda x: x["metadata"].update(accepted_draft_tokens=101), "accepted_exceeds_progress"),
            (lambda x: x["metadata"].update(verification_calls=-1), "invalid_verification_calls"),
            (lambda x: x.update(finish_reason="eos"), "eos_token_finish_mismatch"),
        ]
        for edit, reason in edits:
            with self.subTest(reason=reason):
                fixture(self.root)
                self.mutate("set/O/records/a1.json", edit)
                summary = report.build_report(self.root)
                self.assertEqual(comparison(summary)["valid_pairs"], 1)
                self.assertIsNone(comparison(summary)["call_change_pct"])
                bad = next(r for r in self.rows() if r["arm"] == "O" and r["request_id"] == "a1")
                self.assertIn(reason, bad["validity_reason"])

    def test_frozen_reproduction_is_independent_of_stale_qualification(self):
        self.mutate("set/F/records/a1.json", lambda x: x["metadata"].update(verification_calls=9))
        summary = report.build_report(self.root)
        self.assertEqual(comparison(summary)["valid_pairs"], 1)
        row = next(r for r in self.rows() if r["arm"] == "F" and r["request_id"] == "a1")
        self.assertIn("F_retained_verification_calls_mismatch", row["validity_reason"])

    def test_each_qualification_failure_blocks_gain_claims(self):
        for field in report.QUALIFICATION_FIELDS:
            with self.subTest(field=field):
                fixture(self.root)
                self.mutate("set/QUALIFICATION.json", lambda x: x.update({field: False}))
                summary = report.build_report(self.root)
                self.assertEqual(summary["valid_generations"], 0)
                self.assertEqual(comparison(summary)["gains"], 0)
        fixture(self.root)
        self.mutate("set/QUALIFICATION.json", lambda x: x["arms"]["O"].update(publication_equal=False))
        self.assertEqual(report.build_report(self.root)["valid_generations"], 0)
        (self.root / "set/QUALIFICATION.json").unlink()
        self.assertEqual(report.build_report(self.root)["valid_generations"], 0)

    def test_bad_time_does_not_enter_latency_ratio_or_erase_valid_counts(self):
        self.mutate("set/O/records/a1.json", lambda x: x["metadata"].update(request_latency_s=-1))
        a = comparison(report.build_report(self.root))
        self.assertTrue(a["complete"])
        self.assertFalse(a["timing_complete"])
        self.assertEqual(a["timed_pairs"], 1)
        self.assertIsNone(a["request_time_change_pct"])
        self.assertAlmostEqual(a["available_valid_timed_pairs_time_change_pct"], 100 * (8.2 / 9 - 1))

    def test_zero_call_eos_does_not_invent_progress_ratios(self):
        def eos(record):
            record.update(response_ids=[99], finish_reason="eos", metadata={
                "verification_calls": 0, "accepted_draft_tokens": 0, "committed_progress": 0,
                "request_latency_s": 0.3})
        for path in ("set/F/records/b1.json", "set/O/records/b1.json", "original/set/F/records/b1.json"):
            self.mutate(path, eos)
        b = comparison(report.build_report(self.root), "B")
        self.assertTrue(b["complete"])
        self.assertIsNone(b["call_change_pct"])
        self.assertIsNone(b["O_committed_progress_per_call"])
        self.assertEqual(b["O_eos_count"], 1)

    def test_costs_preserve_nested_and_historical_boundaries(self):
        write(self.root / "original/set/F/PREPARATION.json", {"records": [
            {"teacher_preparation_s": 4, "feature_persistence_s": 0.5},
            {"teacher_preparation_s": 6, "feature_persistence_s": 0.5}]})
        write(self.root / "set/O/TRAINING.json", {
            "costs": {"training_s": 20, "checkpoint_save_s": 7, "export_s": 4, "event_wall_s_before_commit": 32},
            "objective_work": {"teacher_head_s": 3}, "optimizer_steps": 8,
            "response_presentations": 16, "actual_learning_rates": [0.1] * 8,
            "exposure": {"total_supervised_positions": 100}, "peak_allocated_bytes": 1234})
        write(self.root / "set/O/ACTIVATION.json", {"activation_s": 2, "native_activation_s": 2})
        report.build_report(self.root)
        all_costs = json.loads((self.root / "COSTS.json").read_text())
        costs = all_costs["groups"]["set"]
        self.assertEqual(all_costs["actual_campaign_wall_s"], 80)
        self.assertEqual(costs["teacher_preparation"]["fresh_teacher_preparation_s"], 0)
        self.assertEqual(costs["teacher_preparation"]["historical_cost_for_reconstructed_workflow"]["teacher_preparation_s"], 10)
        self.assertEqual(costs["training"]["training_s"], 20)
        self.assertEqual(costs["nested_objective_work"]["teacher_head_s"], 3)
        self.assertEqual(costs["activation"]["activation_s"], 2)
        self.assertEqual(costs["exposure"]["response_presentations"], 16)
        self.assertEqual(costs["serving"]["F"]["observed_request_latency_s"], 12)


if __name__ == "__main__":
    unittest.main()
