"""Pure CPU checks for first-transition accounting and paired request inference."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.verification_transition import analysis as A


def walk(progress):
    canonical = list(range(100, 101 + sum(progress)))
    offset = 0
    trace = []
    for index, count in enumerate(progress):
        target = (canonical[offset + 1:] + [-9] * 16)[:16]
        draft = target[:15]
        if count < 16:
            draft[count - 1] = -3
        trace.append(dict(output_start=offset, anchor_token=canonical[offset],
                          committed_progress=count, committed_tokens=canonical[offset + 1:offset + count + 1],
                          positions=list(range(2 + offset, 18 + offset)),
                          draft_tokens=draft, target_argmax=target,
                          finished=index == len(progress) - 1))
        offset += count
    return dict(prompt_ids=[9, 8], response_ids=canonical, finish_reason="eos", max_new_tokens=2048,
                metadata=dict(trace=trace, verification_calls=len(trace)))


def cell(calls):
    return dict(status="completed", verification_calls=calls,
                output_matches_canonical=True, termination_matches=True)


def advanced(rank=0):
    # Both have99 committed tokens. The first K6 advance is interior; total K6
    # calls are lower, but either fixed checkpoint can prefer the earlier start.
    return A.request_row("gsm8k", f"r{rank}", rank, walk([2] + [16] * 6 + [1]),
                         walk([3] + [16] * 6), crossed=dict(C46=cell(8), C64=cell(5)))


def panel(advance_count=8):
    rows = []
    for rank in range(64):
        if rank < advance_count:
            row = advanced(rank)
        else:
            record = walk([16, 16, 16])
            row = A.request_row("gsm8k", f"r{rank}", rank, record, deepcopy(record))
        rows.append(row)
    return rows


class TransitionAnalysisTests(unittest.TestCase):
    def test_four_cells_decompose_and_repair_first_error(self):
        row = advanced()
        self.assertEqual([row[k] for k in A.CELLS], [7, 8, 5, 6])
        self.assertEqual([row[k] for k in A.EFFECTS], [-1, 1, 1, -2, -2])
        self.assertTrue(row["both_restart_positive"])
        self.assertTrue(row["interior"])
        self.assertEqual(row["first_rejected_proposal_position"], 2)
        self.assertTrue(row["first_rejected_position_repaired"])

    def test_first_actual_progress_not_first_proposal_difference(self):
        left, right = walk([3, 3, 3]), walk([3, 4, 2])
        right["metadata"]["trace"][0]["draft_tokens"][8] = -99
        result = A.find_transition(left, right)
        self.assertEqual(result["divergence_round"], 2)
        self.assertEqual(result["prior_proposal_difference_rounds"], [1])
        self.assertEqual(result["endpoint_call_difference"], 0)
        self.assertEqual(result["transition_kind"], "advance")

    def test_no_divergence_is_na_but_effects_are_zero(self):
        left, right = walk([3, 3]), walk([3, 3])
        right["metadata"]["trace"][0]["draft_tokens"][8] = -99
        row = A.request_row("gsm8k", "r", 0, left, right)
        self.assertEqual(row["transition_kind"], "none")
        self.assertTrue(all(row[name] is None for name in A.CELLS))
        self.assertTrue(all(row[name] == 0 for name in A.EFFECTS))
        self.assertTrue(all(row[name + "_status"] == "not_applicable" for name in A.CELLS))
        self.assertEqual(row["prior_proposal_difference_count"], 1)

    def test_terminal_successor_zero_and_original_boundary(self):
        row = A.request_row("gsm8k", "r", 0, walk([3, 4]), walk([7]),
                            crossed={"C64": cell(1)})
        self.assertEqual(row["C46"], 0)
        self.assertEqual(row["C46_status"], "terminal_zero")
        self.assertEqual(row["remaining_tokens6"], 0)
        self.assertTrue(row["boundary"])
        self.assertEqual(row["D"], -1)

    def test_boundary_includes_exactly_sixteen_remaining(self):
        row = A.request_row("gsm8k", "r", 0, walk([2, 16, 1]), walk([3, 16]),
                            crossed={"C46": cell(1), "C64": cell(2)})
        self.assertEqual(row["remaining_tokens6"], 16)
        self.assertTrue(row["boundary"])
        self.assertFalse(row["interior"])

    def test_initial_regression_is_not_an_advance(self):
        row = A.request_row("gsm8k", "r", 0, walk([4, 4]), walk([3, 5]),
                            crossed={"C46": cell(1), "C64": cell(1)})
        self.assertEqual(row["transition_kind"], "regression")
        self.assertFalse(row["first_rejected_position_repaired"])

    def test_later_repair_without_prior_matches_is_not_a_prefix_repair(self):
        left, right = walk([3, 6]), walk([4, 5])
        right["metadata"]["trace"][0]["draft_tokens"][0] = -99
        result = A.find_transition(left, right)
        self.assertFalse(result["preceding_matches_preserved"])
        self.assertFalse(result["first_rejected_position_repaired"])

    def test_rejection_beyond_canonical_stop_is_not_an_available_repair(self):
        left, right = walk([3]), walk([3])
        for record in (left, right):
            row = record["metadata"]["trace"][0]
            row["draft_tokens"] = row["target_argmax"][:15]
            row["draft_tokens"][10] = -123
        result = A._repair_check(left["metadata"]["trace"][0], right["metadata"]["trace"][0], left["response_ids"])
        self.assertIsNone(result["first_rejected_proposal_position"])
        self.assertEqual(result["repair_check"], "no_valid_first_rejection")

    def test_incomplete_or_invalid_crossed_cells_never_become_zero(self):
        for candidate in (None, dict(status="incomplete", verification_calls=9),
                          dict(cell(2), output_matches_canonical=False)):
            row = A.request_row("gsm8k", "r", 0, walk([3, 3, 3]), walk([3, 4, 2]),
                                crossed={"C46": candidate, "C64": cell(1)})
            self.assertEqual(row["status"], "incomplete")
            self.assertIsNone(row["C46"])
            self.assertTrue(all(row[name] is None for name in A.EFFECTS))

    def test_corrupt_native_commit_or_termination_is_rejected(self):
        valid = walk([3, 4])
        for field, value in (("committed_progress", 4), ("committed_tokens", [2, 3, 4]),
                             ("anchor_token", 555), ("finished", True)):
            invalid = deepcopy(valid)
            invalid["metadata"]["trace"][0][field] = value
            with self.assertRaises(ValueError):
                A.find_transition(invalid, valid)

    def test_paired_bootstrap_recomputes_stratum_from_full_panel(self):
        rows = panel()
        with patch.object(A, "RESAMPLES", 200):
            summary = A.summarize(rows, dataset="gsm8k")
        indices = np.random.Generator(np.random.PCG64(20261110)).integers(0, 64, (200, 64))
        eligible = indices < 8
        counts = eligible.sum(1)
        expected = counts / 64
        effect = summary["strata"]["interior_advances"]["R4"]
        self.assertEqual(effect["unconditional"]["mean"], 8 / 64)
        self.assertEqual(effect["unconditional_ci95"], np.quantile(expected, [.025, .975], method="linear").tolist())
        self.assertEqual(effect["conditional_ci95"], [1., 1.])
        self.assertEqual(effect["empty_stratum_resamples"], int((counts == 0).sum()))
        self.assertEqual(effect["concentration"]["largest_absolute_share"], 1 / 8)
        self.assertTrue(effect["loo_all_conditional_means_positive"])

    def test_sparse_empty_and_singleton_strata_stay_descriptive(self):
        with patch.object(A, "RESAMPLES", 100):
            empty = A.summarize(panel(0), dataset="gsm8k")["strata"]["all_advances"]["R4"]
            one = A.summarize(panel(1), dataset="gsm8k")["strata"]["all_advances"]["R4"]
        self.assertEqual(empty["empty_stratum_resamples"], 100)
        self.assertIsNone(empty["conditional"]["mean"])
        self.assertIsNone(empty["conditional_ci95"])
        self.assertEqual(one["conditional"]["mean"], 1.)
        self.assertGreater(one["empty_stratum_resamples"], 0)
        self.assertIsNone(one["conditional_ci95"])
        self.assertEqual(one["loo_conditional_signs"]["empty"], 1)
        self.assertFalse(one["loo_all_conditional_means_positive"])

    def test_missing_selected_request_retained_without_inference(self):
        rows = panel()
        rows[3] = A.request_row("gsm8k", "r3", 3)
        summary = A.summarize(rows, dataset="gsm8k")
        self.assertEqual(summary["status"], "incomplete")
        self.assertIsNone(summary["strata"])
        self.assertEqual(summary["complete_requests"], 63)
        with self.assertRaises(ValueError):
            A.summarize(rows[:-1], dataset="gsm8k")

    def test_dataset_separation_and_unique_selected_ids(self):
        rows = panel()
        rows[0]["dataset"] = "mathqa"
        with self.assertRaises(ValueError):
            A.summarize(rows, dataset="gsm8k")
        rows = panel()
        rows[1]["request_id"] = rows[0]["request_id"]
        with self.assertRaises(ValueError):
            A.summarize(rows, dataset="gsm8k")

    def test_summary_rejects_zero_filled_missing_cells_and_bad_effects(self):
        rows = panel()
        rows[0]["C46"] = None
        with self.assertRaises(ValueError):
            A.summarize(rows, dataset="gsm8k")
        rows = panel()
        rows[0]["M4"] += 1
        with self.assertRaises(ValueError):
            A.summarize(rows, dataset="gsm8k")

    def test_negative_contributions_and_loo_sign_change_are_preserved(self):
        values = np.array([5., -2., -2., 0.])
        selected = np.array([True, True, True, False])
        indices = np.array([[0, 1, 2, 3], [1, 1, 2, 3]])
        result = A._effect_summary(values, selected, indices, ["a", "b", "c", "d"])
        self.assertEqual(result["conditional"]["negative"], 2)
        self.assertEqual(result["concentration"]["largest_absolute_share"], 5 / 9)
        self.assertEqual(result["leave_one_out"][0]["conditional_mean"], -2.)
        self.assertFalse(result["loo_all_conditional_means_positive"])

    def test_report_and_tables_preserve_missing_na_and_no_speed_claim(self):
        rows = panel(0)
        with patch.object(A, "RESAMPLES", 20):
            summary = A.summarize(rows, dataset="gsm8k")
        with tempfile.TemporaryDirectory() as folder:
            A.write_tables(folder, rows, {"gsm8k": summary})
            saved = json.loads((Path(folder) / "SUMMARY.json").read_text())
            self.assertEqual(saved["gsm8k"]["transitions"], {"none": 64})
            self.assertIn("not_applicable", (Path(folder) / "PER_REQUEST.csv").read_text())
        report = A.render_report({"gsm8k": summary}, execution_commit="test")
        self.assertIn("does not establish serving speedup", report)
        self.assertIn("No automatic support label", report)


if __name__ == "__main__":
    unittest.main()
