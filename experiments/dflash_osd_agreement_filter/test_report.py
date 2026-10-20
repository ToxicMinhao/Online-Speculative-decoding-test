import copy
import json
from pathlib import Path
import tempfile
import unittest

from .experiment import run_stream
from .report import comparison, summarize, write_reports
from .test_experiment import Backend, references, stream


def result():
    source = stream()
    return run_stream(source, references(source), lambda arm, name: Backend(arm))


class ReportTests(unittest.TestCase):
    def test_primary_baseline_tradeoff_not_success(self):
        data = result()
        for row in data["scores"]:
            row["verification_calls"] = {"F": 7, "H": 2, "L": 5, "G": 3, "N": 4}[row["arm"]]
        report = summarize([data])
        self.assertEqual({r["interpretation"] for r in report["interpretations"]},
                         {"tradeoff_not_improvement_over_ordinary_OSD"})

    def test_pool_sums_counts_not_mean_percentages(self):
        data = result()
        for row in data["scores"]:
            if row["arm"] == "F":
                row["verification_calls"] = 10 if row["turn"] == 2 else 90
            elif row["arm"] == "G":
                row["verification_calls"] = 5 if row["turn"] == 2 else 90
        report = summarize([data])
        pooled = next(r for r in report["pooled"] if r["source"] == "all" and r["turn"] == "all" and
                      r["candidate"] == "G" and r["control"] == "F")
        self.assertAlmostEqual(pooled["verification_call_reduction_percent"], 5)
        self.assertEqual(pooled["candidate_metrics"]["verification_calls"], 95)
        self.assertEqual(pooled["valid_requests"], 2)

    def test_invalid_output_stays_visible_but_cannot_claim_gain(self):
        data = result()
        row = next(r for r in data["scores"] if r["arm"] == "G" and r["turn"] == 2)
        row["output_ids"] = [99]
        row["verification_calls"] = 0
        value, _ = comparison(data, 2, "G", "F")
        self.assertFalse(value["valid"])
        self.assertIsNone(value["verification_call_reduction_percent"])
        self.assertIn("G:exact_output_ids_mismatch", value["invalid_reasons"])
        report = summarize([data])
        self.assertEqual(len(report["scores"]), 10)

    def test_turns_and_sources_remain_separate_and_costs_not_added(self):
        data = result()
        for arm in data["arms"].values():
            for update in arm["updates"]:
                update.update(training_s=1., export_s=2., observed_update_call_s=4.)
        report = summarize([data])
        self.assertEqual({r["turn"] for r in report["pooled"]}, {2, 3, "all"})
        self.assertEqual({r["source"] for r in report["pooled"]}, {"all", "wildchat"})
        self.assertTrue(all(r["training_s"] == 1 and r["export_s"] == 2 and r["observed_update_call_s"] == 4
                            for r in report["updates"]))
        self.assertIn("not deployment", report["scope"])

    def test_disk_round_trip_and_complete_outputs(self):
        data = json.loads(json.dumps(result()))
        with tempfile.TemporaryDirectory() as directory:
            report = write_reports([data], directory)
            self.assertTrue(all(r["valid"] for r in report["comparisons"]))
            self.assertEqual(set(p.name for p in Path(directory).iterdir()),
                {"SUMMARY.json", "REPORT.md", "COMPARISONS.csv", "POOLED.csv", "SCORES.csv", "PER_REQUEST.csv", "UPDATES.csv", "CANONICAL_COSTS.csv", "INTERPRETATIONS.csv"})
            self.assertEqual(len(report["per_request"]), 10)
            self.assertTrue(all(r["committed_progress_per_call"] == 1.5 for r in report["per_request"]))


if __name__ == "__main__":
    unittest.main()
