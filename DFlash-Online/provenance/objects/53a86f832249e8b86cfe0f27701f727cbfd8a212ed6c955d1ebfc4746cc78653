"""Scientific accounting and cohort failures for the bounded component diagnostic."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.osd_components.analysis import ARMS, summarize, write_report


def fixture(counts, times=None):
    requests = [{"request_id": f"request-{i}", "panel_ordinal": i, "original_index": 11 + i * 7}
                for i in range(len(counts["F"]))]
    rows = []
    for i, request in enumerate(requests):
        for arm in ARMS:
            for repeat in range(3):
                calls = counts[arm][i]
                rows.append({**request, "method": arm, "repeat": repeat,
                             "valid_measurement": True, "errors": [],
                             "verification_calls": calls, "output_tokens": 101,
                             "accepted_draft_tokens": 100 - calls, "committed_progress": 100,
                             "request_latency_s": (times[arm][i] if times else 1 + i) * (repeat + 1),
                             "cap_hit": i % 2 == 0, "finish_reason": "length" if i % 2 == 0 else "eos"})
    return requests, rows


def pooled(summary, arm):
    return next(row for row in summary["pooled"] if row["method"] == arm)


class CheckpointComponentAnalysisTests(unittest.TestCase):
    def test_one_common_cohort_retains_all_failures_and_missing_work(self):
        requests, rows = fixture({arm: [10, 20, 30, 40] for arm in ARMS})
        bad = next(row for row in rows if row["request_id"] == "request-1"
                   and row["method"] == "O" and row["repeat"] == 1)
        bad.update(valid_measurement=False, errors=["O_historical_reference_count_mismatch"])
        rows = [row for row in rows if not (row["request_id"] == "request-2"
                                            and row["method"] == "B-only" and row["repeat"] == 2)]
        next(row for row in rows if row["request_id"] == "request-3"
             and row["method"] == "C-only" and row["repeat"] == 0)["output_tokens"] = 100
        summary = summarize(rows, requests, bootstrap_samples=100)
        self.assertEqual(summary["cohort"]["valid_request_ids"], ["request-0"])
        self.assertEqual(summary["cohort"]["status"], "PARTIAL")
        self.assertEqual(summary["cohort"]["missing_slots"], 1)
        self.assertEqual(len(summary["measurements"]), 48)
        for arm in ARMS:
            self.assertEqual(pooled(summary, arm)["N_X"], 10)
            self.assertEqual(pooled(summary, arm)["output_tokens"], 101)
            self.assertEqual(sum(map(sum, summary["cross_tabs"].get(arm, {"counts": [[1]]})["counts"])), 1)
        invalid = summary["per_request"][1]
        self.assertIn("O_historical_reference_count_mismatch", invalid["failure_reasons"][0]["errors"])
        self.assertEqual(invalid["validity"]["F"][0]["status"], "valid")
        self.assertFalse(invalid["in_common_cohort"])
        self.assertIsNone(invalid["calls_saved_vs_F"]["F"])
        self.assertEqual(summary["bootstrap"]["cohort_size"], 1)

    def test_calls_are_not_tripled_and_timing_is_a_pooled_ratio(self):
        requests, rows = fixture(
            {"F": [10, 100], "O": [8, 95], "C-only": [9, 96], "B-only": [8, 99]},
            {"F": [1, 9], "O": [0.5, 8], "C-only": [1, 8.5], "B-only": [0.8, 9]},
        )
        summary = summarize(rows, requests, bootstrap_samples=100)
        f, o = pooled(summary, "F"), pooled(summary, "O")
        self.assertEqual(f["N_X"], 110)
        self.assertEqual(f["output_tokens"], 202)
        self.assertEqual(o["D_X"], 7)
        self.assertEqual(o["accepted_draft_tokens"], 97)
        self.assertEqual(o["committed_progress"], 200)
        self.assertEqual(f["request_latency_s_by_repeat"], [10, 20, 30])
        self.assertEqual(f["request_latency_s_mean_total"], 20)
        self.assertEqual(o["request_latency_s_all_repeats"], 51)
        self.assertAlmostEqual(o["time_reduction_pct"], 15)
        self.assertAlmostEqual(o["calls_per_1000_tokens"], 103000 / 202)
        self.assertEqual(o["cap_hits"], 1)
        self.assertEqual(o["improved_requests"], 2)

    def test_gross_savings_harm_cross_tabs_and_interaction(self):
        requests, rows = fixture({"F": [10, 20, 30], "O": [8, 25, 25],
                                  "C-only": [9, 23, 26], "B-only": [7, 26, 28]})
        summary = summarize(rows, requests, bootstrap_samples=1000)
        o, c, b = (pooled(summary, arm) for arm in ("O", "C-only", "B-only"))
        self.assertEqual((o["G_X"], o["H_X"], o["D_X"]), (7, 5, 2))
        self.assertEqual((c["G_X"], c["H_X"], c["D_X"]), (5, 3, 2))
        self.assertEqual((b["G_X"], b["H_X"], b["D_X"]), (5, 6, -1))
        self.assertEqual(b["R_X"], -0.5)
        self.assertEqual((o["improved_requests"], o["tied_requests"], o["regressed_requests"]), (2, 0, 1))
        effects = {row["effect"]: row["calls_saved"] for row in summary["interaction"]}
        self.assertEqual(effects, {"M_C_given_BF": 2, "M_C_given_BO": 3,
                                   "M_B_given_CF": -1, "M_B_given_CO": 0, "I": 1})
        self.assertTrue(all(summary["accounting_checks"].values()))
        self.assertEqual(summary["cross_tabs"]["C-only"]["counts"], [[2, 0, 0], [0, 0, 0], [0, 0, 1]])

    def test_missing_and_changed_repeat_never_substitute_repeat_zero(self):
        requests, rows = fixture({arm: [10, 20] for arm in ARMS})
        rows = [row for row in rows if not (row["request_id"] == "request-0"
                                            and row["method"] == "O" and row["repeat"] == 0)]
        next(row for row in rows if row["request_id"] == "request-1"
             and row["method"] == "C-only" and row["repeat"] == 2)["verification_calls"] = 19
        summary = summarize(rows, requests, bootstrap_samples=100)
        self.assertEqual(summary["cohort"]["status"], "UNAVAILABLE")
        self.assertEqual(summary["cohort"]["valid_requests"], 0)
        self.assertEqual(summary["bootstrap"]["samples"], 0)
        self.assertEqual(summary["bootstrap"]["absolute_call_effects"], {})
        self.assertIsNone(pooled(summary, "F")["N_X"])
        self.assertIsNone(pooled(summary, "C-only")["D_X"])
        self.assertEqual(summary["per_request"][0]["validity"]["O"][0]["status"], "missing")
        for repeat in summary["per_request"][1]["validity"]["C-only"]:
            self.assertIn("verification_calls_mismatch_across_repeats", repeat["errors"])

    def test_bootstrap_is_paired_at_request_level(self):
        # Constant within-request effects remain exact under paired resampling,
        # despite substantial differences between requests and timing repeats.
        requests, rows = fixture({"F": [10, 50, 90], "O": [8, 48, 88],
                                  "C-only": [9, 49, 89], "B-only": [10, 50, 90]})
        summary = summarize(rows, requests, bootstrap_samples=1000)
        intervals = summary["bootstrap"]["absolute_call_effects"]
        self.assertEqual(intervals["D_O"], [6, 6])
        self.assertEqual(intervals["E_C-only"], [3, 3])
        self.assertEqual(intervals["I"], [3, 3])
        self.assertEqual(summary, summarize(rows, requests, bootstrap_samples=1000))

    def test_zero_bootstrap_denominators_preclude_filtered_ratio_interval(self):
        requests, rows = fixture({"F": [10, 10], "O": [9, 11],
                                  "C-only": [9, 9], "B-only": [10, 10]})
        summary = summarize(rows, requests)
        self.assertEqual(pooled(summary, "C-only")["retained_gain_status"], "zero")
        self.assertIsNone(pooled(summary, "C-only")["R_X"])
        states = summary["bootstrap"]["denominator_states"]
        self.assertGreater(states["negative"]["count"], 0)
        self.assertGreater(states["zero"]["count"], 0)
        self.assertGreater(states["positive_at_least_2pct"]["count"], 0)
        self.assertEqual(sum(state["count"] for state in states.values()), 10000)
        self.assertAlmostEqual(sum(state["fraction"] for state in states.values()), 1)
        for ratio in summary["bootstrap"]["retained_gain_ratios"].values():
            self.assertIsNone(ratio["ci95"])
            self.assertEqual(ratio["undefined_samples"], states["zero"]["count"])
            self.assertEqual(ratio["filtered_resamples"], 0)

    def test_negative_and_small_positive_denominators_are_explicit(self):
        for o_calls, status, ratio in ((12, "negative", -0.5), (99, "small_positive", 1),
                                       (98, "positive_at_least_2pct", 0.5)):
            f_calls = 10 if o_calls == 12 else 100
            requests, rows = fixture({"F": [f_calls], "O": [o_calls],
                                      "C-only": [f_calls - 1], "B-only": [f_calls]})
            summary = summarize(rows, requests, bootstrap_samples=100)
            with self.subTest(status=status):
                self.assertEqual(pooled(summary, "C-only")["retained_gain_status"], status)
                self.assertEqual(pooled(summary, "C-only")["R_X"], ratio)
                self.assertEqual(summary["bootstrap"]["denominator_states"][status]["count"], 100)
                self.assertEqual(summary["bootstrap"]["retained_gain_ratios"]["C-only"]["ci95"], [ratio, ratio])
                if status in ("negative", "small_positive"):
                    self.assertIsNone(pooled(summary, "C-only")["descriptive_80pct_screen"])
                else:
                    self.assertFalse(pooled(summary, "C-only")["descriptive_80pct_screen"])

    def test_native_token_accounting_is_checked_even_when_runner_marks_valid(self):
        for field, value, reason in (("accepted_draft_tokens", 101, "accepted_exceeds_progress"),
                                     ("committed_progress", 99, "progress_length_mismatch")):
            requests, rows = fixture({arm: [10, 20] for arm in ARMS})
            bad = next(row for row in rows if row["request_id"] == "request-0"
                       and row["method"] == "C-only" and row["repeat"] == 1)
            bad[field] = value
            summary = summarize(rows, requests, bootstrap_samples=100)
            with self.subTest(field=field):
                self.assertEqual(summary["cohort"]["valid_request_ids"], ["request-1"])
                measurement = next(row for row in summary["measurements"]
                                   if row["request_id"] == "request-0"
                                   and row["method"] == "C-only" and row["repeat"] == 1)
                self.assertTrue(measurement["valid_measurement"])
                self.assertFalse(measurement["analysis_valid"])
                self.assertIn(reason, measurement["errors"])
                self.assertEqual(pooled(summary, "F")["N_X"], 20)

    def test_beating_osd_but_losing_to_frozen_is_reduced_harm(self):
        requests, rows = fixture({"F": [10], "O": [14], "C-only": [12], "B-only": [11]})
        summary = summarize(rows, requests, bootstrap_samples=100)
        c = pooled(summary, "C-only")
        self.assertEqual((c["D_X"], c["E_X"], c["G_X"], c["H_X"]), (-2, -2, 0, 2))
        self.assertEqual(c["regressed_requests"], 1)
        self.assertEqual(c["retained_gain_status"], "negative")
        self.assertIsNone(c["descriptive_80pct_screen"])

    def test_duplicate_and_unexpected_measurements_are_retained(self):
        requests, rows = fixture({arm: [10, 20] for arm in ARMS})
        rows.append(dict(rows[0], record_path="duplicate.json"))
        rows.append(dict(rows[-2], repeat=3, record_path="extra-repeat.json"))
        summary = summarize(rows, requests, bootstrap_samples=100)
        self.assertEqual(summary["cohort"]["duplicate_slots"], 1)
        self.assertEqual(summary["cohort"]["unexpected_measurements"], 1)
        self.assertEqual(summary["cohort"]["valid_requests"], 0)
        self.assertEqual(summary["measurements"][0]["source_rows"][1]["record_path"], "duplicate.json")
        self.assertEqual(summary["unexpected_measurements"][0]["record_path"], "extra-repeat.json")

    def test_full_panel_and_report_preserve_all_measurements(self):
        requests, rows = fixture({"F": [10] * 64, "O": [8] * 64,
                                  "C-only": [9] * 64, "B-only": [8] * 64})
        summary = summarize(rows, requests, bootstrap_samples=100)
        self.assertEqual(summary["cohort"]["status"], "FULL")
        self.assertEqual(summary["cohort"]["expected_measurements"], 768)
        self.assertEqual(pooled(summary, "F")["N_X"], 640)
        self.assertTrue(pooled(summary, "B-only")["descriptive_80pct_screen"])
        with tempfile.TemporaryDirectory() as directory:
            files = write_report(summary, directory)
            self.assertEqual(json.loads(Path(files["summary"]).read_text()), summary)
            with Path(files["measurements"]).open() as handle:
                written = list(csv.DictReader(handle))
            self.assertEqual(len(written), 768)
            self.assertEqual({row["repeat"] for row in written}, {"0", "1", "2"})
            self.assertIn("request-composition sensitivity", Path(files["report"]).read_text())


if __name__ == "__main__":
    unittest.main()
