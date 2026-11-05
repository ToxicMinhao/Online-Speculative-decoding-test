"""CPU-only regression checks for planned denominators and qualified comparisons."""
import csv
import importlib.util
import json
from pathlib import Path

import tempfile
import unittest


spec = importlib.util.spec_from_file_location("feasibility_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def fixture(root):
    requests = [{"request_id": name, "role": role, "prompt_ids": [10, index]}
                for index, (name, role) in enumerate((("a1", "A"), ("a2", "A"), ("b1", "B")))]
    write(root / "manifest.json", {"max_new_tokens": 101, "groups": [{"name": "set", "requests": requests}]})
    write(root / "set" / "QUALIFICATION.json", {
        "status": "passed", "matched_anchors": True, "matched_exposure": True, "matched_final_rng": True})
    counts = {"F": [10, 100, 10], "C": [5, 90, 20], "P": [4, 80, 15]}
    for arm, values in counts.items():
        for request, calls in zip(requests, values):
            write(root / "set" / arm / "records" / (request["request_id"] + ".json"), {
                "record_id": request["request_id"], "prompt_ids": request["prompt_ids"],
                "response_ids": [1] * 101, "finish_reason": "length", "stop_token_ids": [99],
                "max_new_tokens": 101, "metadata": {"verification_calls": calls,
                    "accepted_draft_tokens": 100 - calls, "committed_progress": 100, "request_latency_s": 0.5}})
    return root


def comparison(summary, role="A", arm="C", baseline="F", group="ALL"):
    return next(row for row in summary["comparisons"]
                if (row["group"], row["role"], row["arm"], row["baseline"]) == (group, role, arm, baseline))


def mutate(root, arm, request, edit):
    path = root / "set" / arm / "records" / (request + ".json")
    value = json.loads(path.read_text())
    edit(value)
    write(path, value)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = fixture(Path(self.temporary.name))

    def test_sum_ratios_and_a_b_remain_separate(self):
        summary = report.build_report(self.root)
        a, b = comparison(summary), comparison(summary, role="B")
        self.assertTrue(summary["all_planned_valid"])
        self.assertAlmostEqual(a["change_pct"], 100 * (95 / 110 - 1))
        self.assertNotAlmostEqual(a["change_pct"], (-50 - 10) / 2)
        self.assertEqual((a["gains"], a["regressions"], a["ties"]), (2, 0, 0))
        self.assertEqual(b["change_pct"], 100)
        self.assertEqual((b["gains"], b["regressions"], b["ties"]), (0, 1, 0))
        self.assertAlmostEqual(comparison(summary, arm="P", baseline="C")["change_pct"], 100 * (84 / 95 - 1))
        self.assertEqual(len(self.read_rows()), 9)
        for name in ("REPORT.md", "SUMMARY.json", "POOLED.csv", "COSTS.json"):
            self.assertTrue((self.root / name).exists())

    def read_rows(self):
        with (self.root / "PER_REQUEST.csv").open() as handle:
            return list(csv.DictReader(handle))

    def test_missing_request_keeps_planned_denominator(self):
        (self.root / "set" / "C" / "records" / "a2.json").unlink()
        summary = report.build_report(self.root)
        result = comparison(summary)
        self.assertEqual((result["planned_pairs"], result["valid_pairs"], result["invalid_or_missing_pairs"]), (2, 1, 1))
        self.assertIsNone(result["change_pct"])
        self.assertEqual(result["available_valid_pairs_change_pct"], -50)
        self.assertEqual(summary["present_generations"], 8)
        rows = self.read_rows()
        missing = next(r for r in rows if r["arm"] == "C" and r["request_id"] == "a2")
        self.assertEqual(len(rows), 9)
        self.assertEqual(missing["validity_reason"], "missing_file")

    def test_invalid_outputs_are_visible_and_excluded(self):
        edits = [
            (lambda x: x["response_ids"].__setitem__(5, 2), "canonical_F_response_ids_mismatch"),
            (lambda x: x["prompt_ids"].append(8), "prompt_ids_mismatch"),
            (lambda x: x.update(finish_reason="eos"), "eos_token_finish_mismatch"),
            (lambda x: x.update(max_new_tokens=102), "max_new_tokens_mismatch"),
            (lambda x: x["metadata"].update(committed_progress=99), "progress_length_mismatch"),
            (lambda x: x["metadata"].update(verification_calls=-1), "invalid_verification_calls"),
            (lambda x: x["metadata"].update(accepted_draft_tokens=101), "accepted_exceeds_progress"),
        ]
        for edit, reason in edits:
            with self.subTest(reason=reason):
                fixture(self.root)
                mutate(self.root, "P", "a1", edit)
                summary = report.build_report(self.root)
                self.assertEqual(comparison(summary, arm="P")["valid_pairs"], 1)
                self.assertIsNone(comparison(summary, arm="P", baseline="C")["change_pct"])
                row = next(r for r in self.read_rows() if r["arm"] == "P" and r["request_id"] == "a1")
                self.assertIn(reason, row["validity_reason"])

    def test_pending_qualification_never_counts_as_gain(self):
        write(self.root / "set" / "QUALIFICATION.json", {"status": "pending"})
        summary = report.build_report(self.root)
        self.assertEqual(summary["valid_generations"], 0)
        self.assertEqual(comparison(summary)["gains"], 0)
        self.assertIsNone(comparison(summary)["change_pct"])
        self.assertFalse(summary["qualification"]["set"]["passed"])

    def test_cost_boundaries_and_exact_exposure_are_preserved(self):
        write(self.root / "set" / "P" / "TRAINING.json", {
            "costs": {"training_s": 20, "export_s": 4, "event_wall_s_before_commit": 30},
            "objective_work": {"reference_setup_s": 2, "reference_forward_s": 3, "projection_s": 1},
            "response_presentations": 16, "optimizer_steps": 8, "actual_learning_rates": [0.1] * 8,
            "exposure": {"total_supervised_positions": 50, "total_weight": 45.5},
            "records": [{"record_id": "a1", "pass": 0, "optimizer_stepped": False, "metrics": {"loss": 9}}]})
        report.build_report(self.root)
        costs = json.loads((self.root / "COSTS.json").read_text())["groups"]["set"]["P"]
        self.assertEqual(costs["training"]["training_s"], 20)
        self.assertNotIn("export_s", costs["training"])
        self.assertEqual(costs["export"], {"export_s": 4})
        self.assertEqual(costs["reference_and_projection"]["reference_forward_s"], 3)
        self.assertEqual(costs["exposure"]["supervision"]["total_weight"], 45.5)
        self.assertNotIn("loss", json.dumps(costs))

    def test_zero_calls_single_eos_has_no_invented_ratio(self):
        for arm in report.ARMS:
            mutate(self.root, arm, "b1", lambda x: x.update(
                response_ids=[99], finish_reason="eos", metadata={
                    "verification_calls": 0, "accepted_draft_tokens": 0, "committed_progress": 0}))
        summary = report.build_report(self.root)
        result = comparison(summary, role="B")
        self.assertTrue(result["complete"])
        self.assertEqual(result["valid_pairs"], 1)
        self.assertIsNone(result["change_pct"])
        self.assertEqual(result["ratio_note"], "zero_baseline_calls")


if __name__ == "__main__":
    unittest.main()
