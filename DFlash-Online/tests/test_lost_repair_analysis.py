"""CPU contracts for the frozen-checkpoint lost-repair gate."""

import copy
import math
import unittest

from experiments.lost_repair.analysis import analyze, classify_start, match_occurrences, stable_key


def native(request="r0", start=0, a4=8, a5=15, a6=8, valid=15, **extra):
    return dict(request_id=request, canonical_start=start, a4=a4, a5=a5, a6=a6,
                valid_proposals=valid, genuine_k6_rejection=a6 < valid,
                parity_valid=True, **extra)


def occurrence(request="r0", start=0, presentation=0, j=9,
               priority="high", weight=2.0, margin=1.0):
    return dict(request_id=request, canonical_start=start, presentation_id=presentation,
                j=j, priority=priority, first_error_weight=weight, wrong_margin=margin)


def presentation(request="r0", pid=0, anchors=2, selected=20.0, error_weight=4.0):
    return dict(request_id=request, presentation_id=pid,
                selected_weight=selected, expected_anchor_count=anchors, first_error_weight=error_weight)


def small_fixture():
    census = [native(start=0), native(start=16, a5=8)]
    rows = [occurrence(start=start, presentation=p) for p in (0, 1) for start in (0, 16)]
    presentations = [presentation(pid=p) for p in (0, 1)]
    return census, rows, presentations


def evaluate(census, rows, presentations, ids=("r0",), **kwargs):
    return analyze(census, rows, presentations, ids, census_complete=True,
                   training_complete=True, **kwargs)


class ClassificationTests(unittest.TestCase):
    def test_exact_lost_repair_and_persistent_error(self):
        h = classify_start(native())
        self.assertEqual((h["history"], h["j"], h["previously_attained_then_lost"]), ("H", 9, 7))
        self.assertTrue(h["k6_earlier_break_vs_k5"])
        self.assertFalse(h["k6_earlier_break_vs_k4"])
        m = classify_start(native(a5=8))
        self.assertEqual(m["history"], "M")

    def test_older_prefix_break_does_not_become_lost_repair(self):
        row = classify_start(native(a4=7))
        self.assertEqual(row["history"], "other_rejection_history")
        self.assertTrue(row["k4_earlier_break_vs_k6"])
        self.assertFalse(row["H"])

    def test_converse_can_coexist_with_later_k6_rejection(self):
        row = classify_start(native(a4=3, a5=3, a6=8))
        self.assertTrue(row["converse"])
        self.assertEqual(row["converse_j"], 4)
        self.assertEqual(row["converse_prefix_gain"], 5)
        self.assertEqual(row["history"], "other_rejection_history")

    def test_eos_or_cap_boundary_is_not_a_rejection(self):
        for valid in (0, 3, 15):
            row = classify_start(native(a4=valid, a5=valid, a6=valid, valid=valid))
            self.assertEqual(row["history"], "all_correct_or_terminal")
            self.assertIsNone(row["j"])
        row = native(a4=2, a5=4, a6=2, valid=4)
        row["genuine_k6_rejection"] = False
        self.assertFalse(classify_start(row)["H"])

    def test_cap_validity_required_at_rejected_slot(self):
        row = native(a4=3, a5=3, a6=3, valid=3)
        row["genuine_k6_rejection"] = True
        with self.assertRaisesRegex(ValueError, "canonical slot"):
            classify_start(row)

    def test_missing_invalid_and_out_of_bound_replays(self):
        row = native()
        row["a5"] = None
        self.assertEqual(classify_start(row)["history"], "missing_replay")
        row = native()
        row["parity_valid"] = False
        self.assertEqual(classify_start(row)["history"], "invalid_replay")
        with self.assertRaisesRegex(ValueError, "EOS/cap"):
            classify_start(native(a5=16))


class MatchingTests(unittest.TestCase):
    @staticmethod
    def marked(start, label, margin, **kwargs):
        return dict(occurrence(start=start, margin=margin, **kwargs), matched_history=label)

    def test_nearest_margin_without_replacement(self):
        rows = [self.marked(0, "H", 1), self.marked(16, "H", 4),
                self.marked(32, "M", 1.1), self.marked(48, "M", 4.1)]
        pairs = match_occurrences(rows)
        self.assertEqual({(p["H_start"], p["M_start"]) for p in pairs}, {(0, 32), (16, 48)})
        self.assertEqual(len({p["M_start"] for p in pairs}), 2)

    def test_h_order_and_margin_tie_use_declared_hash(self):
        rows = [self.marked(0, "H", 1), self.marked(16, "H", 1), self.marked(32, "M", 1)]
        pair = match_occurrences(rows)[0]
        self.assertEqual(pair["H_start"], min((0, 16), key=lambda s: stable_key("r0", s)))
        rows = [self.marked(0, "H", 1), self.marked(16, "M", 0.5), self.marked(32, "M", 1.5)]
        pair = match_occurrences(rows)[0]
        self.assertEqual(pair["M_start"], min((16, 32), key=lambda s: stable_key("r0", s)))
        self.assertEqual(match_occurrences(rows), match_occurrences(list(reversed(rows))))

    def test_no_cross_response_depth_priority_or_presentation_match(self):
        h = self.marked(0, "H", 1)
        for change in ({"request_id": "r1"}, {"j": 8}, {"priority": "low"}, {"presentation_id": 1}):
            m = dict(self.marked(16, "M", 1), **change)
            self.assertEqual(match_occurrences([h, m]), [])

    def test_ineligible_retained_as_own_matching_class(self):
        rows = [self.marked(0, "H", 0, priority="ineligible", weight=0.75),
                self.marked(16, "M", 0, priority="ineligible", weight=0.5)]
        pairs = match_occurrences(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["projected_weight"], 0.125)
        self.assertEqual(pairs[0]["absolute_margin_difference"], 0)


class ExposureAndGateTests(unittest.TestCase):
    def test_unique_support_and_repeated_exposure_have_different_counts(self):
        result = evaluate(*small_fixture())
        s = result["summary"]
        self.assertEqual(s["exposure_funnel"]["matched"], dict(occurrences=2, distinct_anchors=1, distinct_requests=1))
        self.assertEqual(s["native_H_weight_before_matching"], 4)
        self.assertEqual(s["native_first_error_weight"], 8)
        self.assertEqual(s["selected_weight"], 40)
        self.assertEqual(s["projected_matched_weight"], 1)
        self.assertEqual(s["H_weight_fraction_all_first_error"], 0.5)
        self.assertEqual(s["H_weight_fraction_all_selected"], 0.1)
        self.assertEqual(s["gate_decision"], "FAIL")
        self.assertTrue(s["gate"]["projected_weight_fraction_all_native_first_error"]["passed"])

    def test_same_j_filter_precedes_matching_and_weight(self):
        census, rows, pp = small_fixture()
        rows[0]["j"] = 8
        result = evaluate(census, rows, pp)
        s = result["summary"]
        self.assertEqual(s["exposure_funnel"]["scheduled"]["occurrences"], 2)
        self.assertEqual(s["exposure_funnel"]["same_j"]["occurrences"], 1)
        self.assertEqual(s["native_H_weight_before_matching"], 2)
        self.assertEqual(s["native_first_error_weight"], 8)

    def test_unmatched_h_keeps_weight_and_non_natural_errors_in_denominator(self):
        census, rows, pp = small_fixture()
        for row in rows:
            if row["canonical_start"] == 16:
                row["priority"] = "low"
        rows.extend([occurrence(start=99, presentation=p, weight=10) for p in (0, 1)])
        for row in pp:
            row["expected_anchor_count"] = 3
            row["first_error_weight"] = 14
        s = evaluate(census, rows, pp)["summary"]
        self.assertEqual(s["H_unmatched_occurrences"], 2)
        self.assertEqual(s["native_H_weight_before_matching"], 4)
        self.assertEqual(s["native_first_error_weight"], 28)
        self.assertEqual(s["projected_matched_weight"], 0)

    def test_first_error_free_blocks_contribute_no_error_weight(self):
        census, rows, pp = small_fixture()
        rows[0].update(j=None, first_error_weight=0, wrong_margin=None, priority="none")
        pp[0]["first_error_weight"] = 2
        s = evaluate(census, rows, pp)["summary"]
        self.assertEqual(s["native_first_error_weight"], 6)
        self.assertEqual(s["exposure_funnel"]["same_j"]["occurrences"], 1)

    def test_gate_exact_thresholds_pass_without_double_counting_anchor_support(self):
        census, rows, pp, ids = [], [], [], []
        for n in range(16):
            rid = f"r{n}"
            ids.append(rid)
            census.extend([native(rid, start) for start in (0, 16)])
            census.extend([native(rid, start, a5=8) for start in (32, 48)])
            for p in range(2):
                pid = 2 * n + p
                pp.append(presentation(rid, pid, anchors=4, selected=100, error_weight=8))
                rows.extend([occurrence(rid, start, pid) for start in (0, 16, 32, 48)])
        result = evaluate(census, rows, pp, ids)
        s = result["summary"]
        self.assertEqual(s["gate_decision"], "PASS")
        self.assertEqual(s["gate"]["distinct_matched_H_anchors"]["value"], 32)
        self.assertEqual(s["gate"]["distinct_matched_H_requests"]["value"], 16)
        self.assertEqual(s["exposure_funnel"]["matched"]["occurrences"], 64)

    def test_missing_native_replay_cannot_decide_gate(self):
        census, rows, pp = small_fixture()
        census[1]["a4"] = None
        s = evaluate(census, rows, pp)["summary"]
        self.assertEqual(s["gate_decision"], "INCOMPLETE")
        self.assertIsNone(s["full_natural_denominator"])
        self.assertTrue(all(x["passed"] is None for x in s["gate"].values()))
        self.assertEqual(s["H_observed_count"], 1)

    def test_missing_training_does_not_erase_completed_census_denominator(self):
        census, rows, pp = small_fixture()
        s = evaluate(census, rows[:-1], pp)["summary"]
        self.assertEqual(s["gate_decision"], "INCOMPLETE")
        self.assertEqual(s["full_natural_denominator"], 2)
        self.assertEqual(s["H_share_all_natural_starts"], 0.5)
        self.assertFalse(s["training_anchor_coverage_complete"])

    def test_missing_presentation_and_full_census_count_detected(self):
        census, rows, pp = small_fixture()
        s = evaluate(census, rows[:2], pp[:1])["summary"]
        self.assertEqual(s["gate_decision"], "INCOMPLETE")
        s = evaluate(census, rows, pp, expected_natural_starts=3)["summary"]
        self.assertEqual(s["known_unobserved_starts"], 1)
        self.assertEqual(s["gate_decision"], "INCOMPLETE")

    def test_identity_failures_and_nonfinite_measurements_rejected(self):
        for mode in ("duplicate", "nonfinite", "wrong_response"):
            census, rows, pp = small_fixture()
            if mode == "duplicate":
                rows.append(dict(rows[0]))
            elif mode == "nonfinite":
                rows[0]["wrong_margin"] = math.nan
            else:
                rows[0]["request_id"] = "r1"
            with self.assertRaises(ValueError):
                evaluate(census, rows, pp)

    def test_analysis_does_not_mutate_input_records(self):
        fixture = small_fixture()
        original = copy.deepcopy(fixture)
        evaluate(*fixture)
        self.assertEqual(fixture, original)

    def test_native_response_mass_is_authoritative_with_bounded_fp32_rounding(self):
        census, rows, pp = small_fixture()
        pp[0]["first_error_weight"] = 4.000001
        result = evaluate(census, rows, pp)
        self.assertAlmostEqual(result["summary"]["native_first_error_weight"], 8.000001)
        self.assertEqual(result["summary"]["serialized_anchor_first_error_weight"], 8)
        self.assertEqual(result["summary"]["first_error_mass_consistency"]["checked_presentations"], 2)
        self.assertAlmostEqual(result["summary"]["projected_weight_fraction_all_first_error"], 1 / 8.000001)
        pp[0]["first_error_weight"] = 4.0001
        with self.assertRaisesRegex(ValueError, "native response first-error mass"):
            evaluate(census, rows, pp)

    def test_missing_native_mass_is_not_silently_replaced_by_anchor_sum(self):
        census, rows, pp = small_fixture()
        del pp[0]["first_error_weight"]
        result = evaluate(census, rows, pp)
        self.assertEqual(result["summary"]["gate_decision"], "INCOMPLETE")
        self.assertIsNone(result["summary"]["native_first_error_weight"])
        self.assertEqual(result["summary"]["serialized_anchor_first_error_weight"], 8)


if __name__ == "__main__":
    unittest.main()
