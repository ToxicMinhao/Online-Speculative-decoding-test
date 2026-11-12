"""Fixed-fixture checks of outcome-blind matching and full-denominator risks."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.repair_history_margin.analysis import (
    analyze_k6, build_pre_k6, classify_k6, cluster_bootstrap, margin_balance,
    match_tie_key, payload_sha256, validate_seal,
)


def start(source="r0", offset=0, a4=1, a5=3, margin=1.0, limit=15, **extra):
    return dict(source_id=source, start_offset=offset, a4=a4, a5=a5,
                valid_proposal_count=limit, k4_first_error=a4 < limit,
                margins=[dict(slot=j, margin=margin) for j in range(1, a5 + 1)], **extra)


def outcome(source="r0", offset=0, a6=1, limit=15, **extra):
    return dict(source_id=source, start_offset=offset, a6=a6,
                valid_proposal_count=limit, k6_first_error=a6 < limit,
                direct_correct={j: j <= a6 for j in range(1, limit + 1)}, **extra)


def basic_seal():
    return build_pre_k6([start(), start(offset=16, a4=3)])


class MatchingTests(unittest.TestCase):
    def test_label_requires_genuine_error_and_repair_at_same_slot(self):
        rows = [start(), start(offset=16, a4=3), start(offset=32, a4=1, a5=1),
                start(offset=48, a4=3, a5=3, limit=3)]
        seal = build_pre_k6(rows)
        self.assertEqual(seal["coverage"]["all_repairs"], 1)
        self.assertEqual(seal["repairs"][0]["j"], 2)
        self.assertEqual(seal["coverage"]["matched_repairs"], 1)

    def test_control_requires_same_request_depth_and_a5(self):
        seal = build_pre_k6([start(), start("other", offset=16, a4=3),
                             start(offset=32, a4=3, a5=4), start(offset=48, a4=0),
                             start(offset=64, a4=2)])
        pool = seal["structural_pools"][0]
        self.assertEqual([c["start_offset"] for c in pool["comparators"]], [64])
        self.assertEqual(seal["matched_pairs"][0]["comparator_start"], 64)

    def test_nearest_control_and_with_replacement(self):
        seal = build_pre_k6([start(offset=0, margin=1), start(offset=8, margin=1.125),
                             start(offset=16, a4=3, margin=1.0625),
                             start(offset=32, a4=3, margin=1.25)])
        self.assertEqual([p["comparator_start"] for p in seal["matched_pairs"]], [16, 16])
        self.assertEqual(seal["coverage"]["maximum_comparator_reuse"], 2)
        reversed_seal = build_pre_k6(list(reversed([
            start(offset=0, margin=1), start(offset=8, margin=1.125),
            start(offset=16, a4=3, margin=1.0625), start(offset=32, a4=3, margin=1.25)])))
        self.assertEqual(sorted((p["repaired_start"], p["comparator_start"]) for p in reversed_seal["matched_pairs"]), [(0, 16), (8, 16)])

    def test_caliper_inclusive_and_zero_only_to_zero(self):
        self.assertEqual(build_pre_k6([start(), start(offset=16, a4=3, margin=1.25)])["coverage"]["matched_repairs"], 1)
        self.assertEqual(build_pre_k6([start(), start(offset=16, a4=3, margin=1.250001)])["coverage"]["matched_repairs"], 0)
        self.assertEqual(build_pre_k6([start(margin=0), start(offset=16, a4=3, margin=.01)])["coverage"]["matched_repairs"], 0)
        seal = build_pre_k6([start(margin=0), start(offset=16, a4=3, margin=0)])
        self.assertEqual(seal["coverage"]["matched_repairs"], 1)
        self.assertEqual(seal["balance"]["absolute_smd"], 0)

    def test_tie_break_is_exact_compact_utf8_sha(self):
        expected = hashlib.sha256(json.dumps([20261115, "rα", 0, 16, 2], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(match_tie_key("rα", 0, 16, 2), expected)
        seal = build_pre_k6([start(), start(offset=16, a4=3, margin=.875), start(offset=32, a4=3, margin=1.125)])
        expected_start = min((16, 32), key=lambda c: match_tie_key("r0", 0, c, 2))
        self.assertEqual(seal["matched_pairs"][0]["comparator_start"], expected_start)

    def test_all_structural_pools_sealed_but_work_uses_matched_population(self):
        seal = build_pre_k6([start(), start(offset=16, a4=3),
                             start("r1", margin=0), start("r1", offset=16, a4=3, margin=5)])
        self.assertEqual(len(seal["structural_pools"]), 2)
        self.assertEqual({r["source_id"] for r in seal["required_k6_starts"]}, {"r0"})
        validate_seal(json.loads(json.dumps(seal)))
        changed = copy.deepcopy(seal)
        changed["structural_pools"][1]["comparators"][0]["margin"] = 2
        with self.assertRaisesRegex(ValueError, "seal changed"):
            validate_seal(changed)

    def test_every_accepted_margin_required_and_nonnegative(self):
        missing = start(); missing["margins"].pop()
        negative = start(); negative["margins"][0]["margin"] = -.01
        nonfinite = start(); nonfinite["margins"][0]["margin"] = float("nan")
        duplicate = start(); duplicate["margins"].append(duplicate["margins"][0])
        for row in (missing, negative, nonfinite, duplicate):
            with self.assertRaises(ValueError):
                build_pre_k6([row])

    def test_invalid_and_empty_inputs_explicit(self):
        bad = dict(source_id="bad", start_offset=0, valid=False, reason="cache_failure")
        seal = build_pre_k6([bad])
        self.assertEqual(seal["coverage"]["unavailable_starts"], 1)
        self.assertEqual(seal["unavailable_starts"][0]["reason"], "cache_failure")
        self.assertTrue(seal["support_flags"])
        self.assertEqual(build_pre_k6([])["coverage"]["matched_repairs"], 0)

    def test_smd_uses_matched_reuse_weights_and_sample_variance(self):
        pairs = [dict(repaired_margin=1, comparator_margin=1), dict(repaired_margin=3, comparator_margin=2)]
        balance = margin_balance(pairs)
        self.assertAlmostEqual(balance["absolute_smd"], .5 / np.sqrt((2 + .5) / 2))
        self.assertEqual(balance["absolute_difference"]["maximum"], 1)
        constant = margin_balance([dict(repaired_margin=1, comparator_margin=1.1)] * 2)
        self.assertTrue(constant["undefined_nonidentical_constant_groups"])
        self.assertIsNone(constant["absolute_smd"])


class OutcomeTests(unittest.TestCase):
    def test_three_exhaustive_classes_keep_earlier_errors(self):
        for a6, expected in [(0, "earlier_rejection"), (1, "exact_slot_rejection"), (2, "accepts_through")]:
            self.assertEqual(classify_k6(outcome(a6=a6), 2)["outcome"], expected)
        row = outcome(a6=0); row["direct_correct"][2] = True
        checked = classify_k6(row, 2)
        self.assertEqual(checked["outcome"], "earlier_rejection")
        self.assertTrue(checked["direct_correct"])

    def test_eos_cap_bounds_and_inconsistent_tokens_invalid(self):
        self.assertFalse(classify_k6(outcome(a6=1, limit=1), 2)["valid"])
        row = outcome(a6=1); row["k6_first_error"] = False
        self.assertFalse(classify_k6(row, 2)["valid"])
        row = outcome(a6=1); row["direct_correct"][2] = True
        self.assertFalse(classify_k6(row, 2)["valid"])
        self.assertEqual(classify_k6(outcome(a6=2, limit=2), 2)["outcome"], "accepts_through")

    def test_paired_risk_and_structural_gap_use_same_repairs(self):
        seal = build_pre_k6([start(), start(offset=8, margin=1.125),
                             start(offset=16, a4=3, margin=1), start(offset=32, a4=3, margin=5)])
        outcomes = [outcome(a6=1), outcome(offset=8, a6=0), outcome(offset=16, a6=3), outcome(offset=32, a6=1)]
        result = analyze_k6(seal, outcomes)
        self.assertEqual(result["primary"]["denominator"], 2)
        self.assertEqual(result["primary"]["delta"], .5)
        self.assertEqual(result["primary"]["earlier_break_difference"], .5)
        self.assertEqual(result["primary"]["established"]["counts"]["accepts_through"], 2)
        self.assertEqual(result["structural"]["denominator"], 2)
        self.assertEqual(result["structural"]["delta"], 0)
        self.assertEqual(result["decision"], "unresolved")  # sparse fixture, not efficacy evidence

    def test_missing_never_becomes_non_event_or_complete_case_primary(self):
        result = analyze_k6(basic_seal(), [outcome()])
        self.assertEqual(result["primary"]["denominator"], 1)
        self.assertEqual(result["primary"]["valid_pairs"], 0)
        self.assertIsNone(result["primary"]["delta"])
        self.assertIsNone(result["primary"]["established"]["rates"]["exact_slot_rejection"])
        self.assertEqual(result["decision"], "unresolved")
        self.assertEqual(result["missing_k6_starts"], 1)

    def test_replayed_start_uniqueness_and_sealed_work_only(self):
        for rows in ([outcome(), outcome()], [outcome(offset=999)]):
            with self.assertRaises(ValueError):
                analyze_k6(basic_seal(), rows)

    def test_direct_proposal_comparison_supports_token_arrays(self):
        row = outcome(); del row["direct_correct"]
        row.update(proposals=[10, 99, 30], canonical=[10, 20, 30])
        self.assertEqual(classify_k6(row, 2)["outcome"], "exact_slot_rejection")
        self.assertFalse(classify_k6(row, 2)["direct_correct"])


class BootstrapTests(unittest.TestCase):
    def test_request_resampling_retains_pairs_and_reuse_weight(self):
        rows = [dict(source_id="a", value=1)] * 3 + [dict(source_id="b", value=-1)]
        result = cluster_bootstrap(rows, "value")
        rng = np.random.Generator(np.random.PCG64(20261115))
        indices = rng.integers(2, size=(10000, 2))
        sums = np.asarray([3, -1])[indices].sum(axis=1)
        denominators = np.asarray([3, 1])[indices].sum(axis=1)
        expected = np.quantile(sums / denominators, [.025, .975])
        self.assertEqual([result["lower"], result["upper"]], list(expected))
        self.assertEqual(result["undefined_draws"], 0)
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result, cluster_bootstrap(list(reversed(rows)), "value"))

    def test_empty_bootstrap_reports_undefined_draws(self):
        result = cluster_bootstrap([], "value")
        self.assertEqual(result["undefined_draws"], 10000)
        self.assertIsNone(result["lower"])

    def test_decision_threshold_and_sparse_population(self):
        starts, outcomes = [], []
        for i in range(40):
            source = f"r{i}"
            for offset in (0, 4, 8):
                starts.append(start(source, offset=offset))
                outcomes.append(outcome(source, offset=offset))
            starts.append(start(source, offset=16, a4=3))
            outcomes.append(outcome(source, offset=16, a6=3))
        seal = build_pre_k6(starts)
        self.assertEqual(seal["support_flags"], [])
        result = analyze_k6(seal, outcomes)
        self.assertEqual(result["decision"], "additional_risk_above_five_percentage_points_supported")
        for row in outcomes:
            row.update(a6=3, direct_correct={j: j <= 3 for j in range(1, 16)})
        result = analyze_k6(seal, outcomes)
        self.assertEqual(result["decision"], "prespecified_five_point_positive_association_ruled_out")
        self.assertEqual(analyze_k6(seal, outcomes, execution_complete=False)["decision"], "unresolved")


class ReportTests(unittest.TestCase):
    def test_complete_report_public_tables_do_not_copy_token_payload(self):
        from experiments.repair_history_margin.report import report
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            starts = [start(), start(offset=16, a4=3)]
            starts[0]["canonical_token_ids"] = [87654321]
            seal = build_pre_k6(starts)
            for name, payload in (("STARTS.json", starts), ("PRE_K6.json", seal),
                                  ("OUTCOMES.json", [outcome(), outcome(offset=16, a6=3)]),
                                  ("RESULT.json", dict(status="completed"))):
                (output / name).write_text(json.dumps(payload))
            summary = report(dict(execution_commit="fixture"), output)
            self.assertTrue(summary["durable_matching_reconstruction_passed"])
            text = "".join(p.read_text() for p in (output / "final_report").iterdir())
            self.assertNotIn("87654321", text)
            self.assertIn("percentage points", text)

    def test_partial_work_includes_failed_capture_and_unattempted_blocks(self):
        from experiments.repair_history_margin.report import work_accounting
        plan = dict(requests=[{}, {}], scope=dict(archived_k5_rounds=6))
        budget = dict(attempted=dict(parent_walk=1, neutral_block=1),
                      completed=dict(neutral_block=1), failed=dict(parent_walk=1), unknown={})
        result = work_accounting(plan, budget, None)
        rows = {r["operation"]: r for r in result["rows"]}
        self.assertEqual(rows["parent_walk"]["failed"], 1)
        self.assertEqual(rows["parent_walk"]["unattempted"], 1)
        self.assertEqual(rows["neutral_block"]["unattempted"], 5)
        self.assertEqual(rows["historical_block"]["unattempted"], 6)
        self.assertIsNone(rows["outcome_block"]["planned"])
        self.assertIsNone(rows["outcome_block"]["unattempted"])
        self.assertEqual(result["status"], "passed")
        budget["unknown"]["neutral_block"] = 1
        budget["completed"]["neutral_block"] = 0
        self.assertEqual(work_accounting(plan, budget, None)["rows"][1]["unknown"], 1)

    def test_incomplete_prephase_is_not_retrospectively_sealed(self):
        from experiments.repair_history_margin.report import report
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "STARTS.json").write_text(json.dumps([start()]))
            report(dict(execution_commit="fixture"), output)
            self.assertFalse((output / "PRE_K6.json").exists())
            self.assertIn("no pre-K+6 comparison seal", (output / "final_report/REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main()
