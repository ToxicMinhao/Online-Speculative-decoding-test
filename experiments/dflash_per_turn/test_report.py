"""CPU-only checks of paired evidence, case selection and cost boundaries."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("per_turn_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture(name="wildchat_00", source="allenai/WildChat-4.8M", n=4):
    result = dict(episode_id=name, source=source, synthetic="PersonaMem" in source,
                  status="complete", planned_requests=n, canonical=[], scores=[], updates=[], support=[],
                  diagnostics=[], cleanup={"status": "passed"}, actual_wall_s=1000.)
    for turn in range(1, n + 1):
        current, previous = report._methods(turn)
        item = dict(episode_id=name, turn=turn, input_ids=[20, 30, turn], output_ids=[40, 41, 42],
                    checkpoint_id=current, checkpoint_identity_sha256="hash-" + current,
                    request_s=2., verification_calls=10, accepted_draft_tokens=1,
                    verification_committed_tokens=2, capture=False, feedback=None, capture_costs=None,
                    activation_s=.2, native_activation_s=.05, cache_maintenance_s=.1, cap_hit=False)
        result["canonical"].append(item)
        for repeat in range(3):
            for checkpoint in dict.fromkeys(method for method in ("F", current, previous) if method is not None):
                row = {**copy.deepcopy(item), "checkpoint_id": checkpoint,
                       "checkpoint_identity_sha256": "hash-" + checkpoint, "repeat": repeat,
                       "input_match": True, "canonical_output_match": True, "checkpoint_identity_match": True,
                       "same_generation_valid": True, "activation_s": 10000., "cache_maintenance_s": 10000.}
                # D1 useful; D2 partially erodes that gain; D3 worse than frozen.
                row.update(request_s={"F": 2., "D1": 1., "D2": 1.5, "D3": 2.5}[checkpoint],
                           verification_calls={"F": 10, "D1": 5, "D2": 7, "D3": 12}[checkpoint])
                result["scores"].append(row)
        if turn < n:
            result["support"].append(dict(turn=turn, observed_support_prepare_s=.3, collection_s=.1))
            result["updates"].append(dict(event=turn, optimizer_steps=2 * turn, checkpoint="D" + str(turn),
                observed_update_call_s=10., training_s=7., export_s=1., optimizer_state_save_s=1.,
                trainer_startup_s=.5 if turn == 1 else 0.,
                accounting={"optimizer_steps_this_event": 2, "response_presentations_this_event": 2,
                    "costs": {"training_s": 7., "teacher_preparation_s": 3., "draft_training_and_optimizer_s": 4.},
                    "response_audits": [{"sampled_anchor_count": 4}],
                    "batches": [{"presentations": [{"sampled_blocks": 4, "eligible_positions": 8,
                        "unique_supervised_positions": 3, "weighted_position_sum": 5.}]}]}))
    return result


def change(result, turn, checkpoint, **values):
    for row in result["scores"]:
        if row["turn"] == turn and row["checkpoint_id"] == checkpoint:
            row.update(values)


class ReportTests(unittest.TestCase):
    def test_counts_dedup_and_two_distinct_comparisons(self):
        result = fixture()
        summary = report.summarize([result])
        conversation = summary["conversations"][0]
        self.assertEqual(conversation["expected_score_executions"], 27)
        self.assertEqual(conversation["planned_optimizer_steps"], 6)
        self.assertEqual(conversation["score_executions_observed"], 27)
        first = conversation["turns"][0]
        self.assertEqual({row["comparison"] for row in first["comparisons"]}, {"online_vs_frozen"})
        self.assertEqual(first["comparisons"][-1]["latency_reduction_percent"], 0)
        third = conversation["turns"][2]
        pooled = {row["comparison"]: row for row in third["comparisons"] if row["repeat"] == "pooled"}
        self.assertAlmostEqual(pooled["online_vs_frozen"]["latency_reduction_percent"], 25.)
        self.assertAlmostEqual(pooled["current_vs_previous"]["latency_reduction_percent"], -50.)
        self.assertEqual(pooled["current_vs_previous"]["candidate_metrics"]["accepted_draft_tokens"], 3)
        self.assertEqual(pooled["current_vs_previous"]["candidate_metrics"]["verification_committed_tokens"], 6)
        self.assertEqual(pooled["current_vs_previous"]["candidate_metrics"]["output_tokens"], 9)

    def test_ratio_of_sums_not_average_percentages(self):
        result = fixture()
        change(result, 2, "F", request_s=20.)
        change(result, 2, "D1", request_s=18.)
        pooled = report.summarize([result])["conversations"][0]["comparisons"][0]
        expected = 100 * (1 - (2 + 18 + 1.5 + 2.5) / (2 + 20 + 2 + 2))
        self.assertAlmostEqual(pooled["latency_reduction_percent"], expected)
        self.assertEqual(pooled["expected_contexts"], 4)
        self.assertEqual(pooled["valid_pairs"], 12)

    def test_invalid_repeat_kept_and_excludes_whole_context_from_pool(self):
        for mutation in (lambda row: row.update(input_ids=[20, 31, 2]),
                         lambda row: row.update(output_ids=[40, 999, 42]),
                         lambda row: row.update(checkpoint_identity_sha256="different"),
                         lambda row: row.update(capture=True),
                         lambda row: row.update(canonical_output_match=False),
                         lambda row: row.update(request_s=float("nan"))):
            result = fixture()
            row = next(row for row in result["scores"] if row["turn"] == 2 and row["checkpoint_id"] == "D1")
            mutation(row)
            summary = report.summarize([result])
            turn = summary["conversations"][0]["turns"][1]
            invalid = next(row for row in turn["comparisons"] if row["repeat"] == "pooled")
            self.assertFalse(invalid["paired_same_generation_valid"])
            self.assertIsNone(invalid["latency_reduction_percent"])
            self.assertEqual(summary["conversations"][0]["score_executions_observed"], 27)
            self.assertFalse(summary["sources"][0]["comparisons"][0]["coverage_complete"])
            self.assertIsNone(report.classify_update(result, 2)["kind"])

    def test_missing_duplicate_repeat_does_not_pass_count_only_check(self):
        result = fixture()
        rows = [row for row in result["scores"] if row["turn"] == 3 and row["checkpoint_id"] == "D2"]
        rows[-1]["repeat"] = 1
        self.assertIsNone(report.classify_update(result, 3)["kind"])
        output = report.summarize([result])["conversations"][0]["turns"][2]
        self.assertFalse(next(row for row in output["comparisons"] if row["repeat"] == "pooled")["paired_same_generation_valid"])

    def test_complete_cost_counts_components_once_and_keeps_exposure(self):
        result = report.summarize([fixture()])["conversations"][0]["cost_breakdown"]
        self.assertTrue(result["complete_nonoverlapping_accounting"])
        self.assertAlmostEqual(result["reconstructed_recurring_session_s"], 8 + .8 + .4 + .9 + 30 - .5)
        self.assertEqual(result["one_time_loading"]["trainer_startup_s"], .5)
        self.assertEqual(result["observed_update_calls_including_startup_s"], 30.)
        self.assertEqual(result["diagnostic_recovery_io"]["optimizer_state_save_s"], 3.)
        self.assertEqual(result["actual_experiment_wall_s"], 1000.)
        self.assertEqual(result["per_update"][0]["accounting_costs"]["teacher_preparation_s"], 3.)
        exposure = result["per_update"][0]["supervised_exposure"]
        self.assertEqual(exposure["sampled_presentation_totals"]["sampled_blocks"], 4)
        self.assertEqual(exposure["response_presentations_this_event"], 2)
        self.assertGreater(result["benchmark_timers"]["activation_s"]["sum_s"], 10000.)

    def test_missing_costs_not_assumed_free_and_partial_denominator_stays_planned(self):
        for mutation in (lambda x: x["canonical"][0].pop("activation_s"),
                         lambda x: x["support"][0].pop("observed_support_prepare_s"),
                         lambda x: x["updates"][0].pop("observed_update_call_s"),
                         lambda x: x["updates"][0].pop("trainer_startup_s"),
                         lambda x: x.update(status="failed")):
            result = fixture()
            mutation(result)
            self.assertIsNone(report.cost_breakdown(result)["reconstructed_recurring_session_s"])
        result = fixture()
        result["canonical"] = result["canonical"][:2]
        result["scores"] = [row for row in result["scores"] if row["turn"] <= 2]
        summary = report.summarize([result])["conversations"][0]
        self.assertEqual(summary["expected_score_executions"], 27)
        self.assertEqual(summary["comparisons"][0]["expected_contexts"], 4)
        self.assertEqual(summary["comparisons"][0]["valid_contexts"], 2)
        self.assertIsNone(summary["cost_breakdown"]["reconstructed_recurring_session_s"])

    def test_selection_is_chronological_and_bounded_not_largest_effect(self):
        result = fixture()
        self.assertEqual(report.classify_update(result, 2)["kind"], "beneficial")
        self.assertEqual(report.classify_update(result, 3)["kind"], "harmful")
        self.assertEqual(report.classify_update(result, 4)["kind"], "harmful")
        selected = report.select_diagnostic_events([result, fixture(name="later")])
        self.assertEqual([(row["event"], row["kind"]) for row in selected], [(1, "beneficial"), (2, "harmful"), (3, "harmful")])
        self.assertTrue(all(row["episode_id"] == result["episode_id"] for row in selected))

    def test_timing_only_harm_and_call_timing_tradeoff_are_explicit(self):
        result = fixture()
        change(result, 3, "D2", verification_calls=5)
        decision = report.classify_update(result, 3)
        self.assertEqual(decision["metric_basis"], "timing_only")
        self.assertEqual(decision["kind"], "harmful")
        self.assertIn("no demonstrated drafting harm", decision["reason"])
        change(result, 3, "D2", verification_calls=7, request_s=.9)
        decision = report.classify_update(result, 3)
        self.assertEqual(decision["metric_basis"], "verification_calls")
        self.assertTrue(decision["tradeoff"])
        change(result, 3, "D2", verification_calls=4, request_s=1.5)
        decision = report.classify_update(result, 3)
        self.assertIsNone(decision["kind"])
        self.assertTrue(decision["tradeoff"])

    def test_one_opposite_repeat_prevents_drafting_harm_selection(self):
        result = fixture()
        row = next(row for row in result["scores"] if row["turn"] == 3 and row["checkpoint_id"] == "D2" and row["repeat"] == 2)
        row["verification_calls"] = 4
        self.assertIsNone(report.classify_update(result, 3)["kind"])

    def test_zero_verification_calls_remain_valid_without_division_by_zero(self):
        result = fixture()
        for row in result["canonical"] + result["scores"]:
            row.update(output_ids=[128009], accepted_draft_tokens=0, verification_committed_tokens=0,
                       verification_calls=0)
        output = report.summarize([result])
        comparison = output["conversations"][0]["comparisons"][0]
        self.assertTrue(comparison["coverage_complete"])
        self.assertIsNotNone(comparison["latency_reduction_percent"])
        self.assertIsNone(comparison["verification_call_reduction_percent"])
        self.assertEqual(comparison["candidate_metrics"]["calls_per_1000_output_tokens"], 0.)
        self.assertIsNone(comparison["candidate_metrics"]["accepted_draft_tokens_per_verification"])
        self.assertEqual(report.classify_update(result, 2)["metric_basis"], "timing_only")
        change(result, 2, "D1", verification_calls=1)
        self.assertEqual(report.classify_update(result, 2)["kind"], "harmful")

    def test_sources_remain_separate_and_duplicate_conversation_rejected(self):
        real = fixture()
        synthetic = fixture("personamem_persona42", "bowen-upenn/PersonaMem-v2")
        output = report.summarize([real, synthetic])
        self.assertEqual(len(output["sources"]), 2)
        self.assertEqual([row["synthetic"] for row in output["sources"]], [False, True])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report.summarize([real, real])
        real.pop("planned_requests")
        with self.assertRaisesRegex(ValueError, "planned_requests"):
            report.summarize([real])

    def test_diagnostic_pre_post_and_smaller_rate_use_identical_inputs(self):
        result = fixture()
        case = dict(event=2, kind="harmful", next_turn=3, earlier_turn=1, status="complete", scores=[],
                    branch={"main_state_restored": True}, selection=report.classify_update(result, 3))
        for window, turn in (("next", 3), ("earlier", 1)):
            canonical = result["canonical"][turn - 1]
            for role, checkpoint, calls in (("pre", "D1", 5), ("post", "D2", 7), ("low_lr", "L2", 6)):
                for repeat in range(3):
                    row = {**copy.deepcopy(canonical), "checkpoint_id": checkpoint, "checkpoint_identity_sha256": "hash-" + checkpoint,
                           "window": window, "role": role, "repeat": repeat, "verification_calls": calls,
                           "request_s": calls / 5, "input_match": True, "canonical_output_match": True,
                           "checkpoint_identity_match": True, "same_generation_valid": True}
                    case["scores"].append(row)
        result["diagnostics"].append(case)
        output = report.summarize([result])["conversations"][0]["diagnostics"][0]
        self.assertEqual(len(output["comparisons"]), 6)
        self.assertTrue(all(row["paired_same_generation_valid"] for row in output["comparisons"]))
        low_post = next(row for row in output["comparisons"] if row["window"] == "next" and row["control_role"] == "post")
        self.assertAlmostEqual(low_post["verification_call_reduction_percent"], 100 * (1 - 6 / 7))
        case["scores"][-1]["input_ids"] = [1, 2]
        output = report.summarize([result])["conversations"][0]["diagnostics"][0]
        self.assertFalse(next(row for row in output["comparisons"] if row["window"] == "earlier" and row["control_role"] == "post")["paired_same_generation_valid"])

    def test_outputs_preserve_all_raw_repeats_and_invalid_records(self):
        result = fixture()
        result["scores"][0]["canonical_output_match"] = False
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "reports"
            summary = report.write_reports([result], out)
            self.assertEqual(json.loads((out / "SUMMARY.json").read_text()), summary)
            with (out / "raw_repeats.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 27)
            self.assertEqual(rows[0]["report_valid"], "False")
            self.assertIn("canonical_output_match", rows[0]["invalid_reasons"])
            with (out / "per_turn.csv").open(newline="") as handle:
                comparisons = list(csv.DictReader(handle))
            self.assertEqual(len(comparisons), 28)
            self.assertIn("available checkpoint", (out / "REPORT.md").read_text())
            self.assertIn("No diagnostic branch results", (out / "CASES.md").read_text())
            with self.assertRaises(FileExistsError):
                report.write_reports([result], out)


if __name__ == "__main__":
    unittest.main()
