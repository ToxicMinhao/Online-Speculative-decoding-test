"""CPU arithmetic checks: paired denominators, failures and nonoverlapping cost."""
import copy
import importlib.util
from pathlib import Path
import unittest


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


report = module("followup_report", "report.py")
fixtures = module("followup_report_fixture", "test_experiment.py")


def example(mode="author-path", replica=1, scenario="STABLE_CODE"):
    rows = fixtures.episode_rows(replica, scenario)
    backend = fixtures.FakeBackend(divergent=True)
    run = fixtures.experiment.run_author_episode if mode == "author-path" else fixtures.experiment.run_checkpoint_comparison
    result = run(rows, fixtures.original_reference(rows), backend)
    assert result["status"] == "complete", result
    for row in result["scores"]:
        method = row["checkpoint_id"]
        row["request_s"] = {"F": 2., "C1": 1.8, "C2": 1.6, "A1": 1.7, "A2": 1.5}[method]
        row["verification_calls"] = {"F": 10, "C1": 9, "C2": 8, "A1": 8, "A2": 7}[method]
        row["verification_committed_tokens"] = 1
        row["cache_maintenance_s"] = 10000.  # diagnostic switches must not enter session cost
        row["activation_s"] = 20000.
    for row in result["canonical"]:
        row.update(request_s=1., cache_maintenance_s=.1, activation_s=.2, native_activation_s=.1)
    for row in result["support"]:
        row.update(observed_support_prepare_s=.05, collection_s=.04)
    for event, update in enumerate(result["updates"], 1):
        update.update(observed_update_call_s=10. * event, training_s=5. * event,
                      export_s=2., optimizer_state_save_s=1.)
        update["accounting"] = {"costs": {"training_s": 5. * event, "teacher_preparation_s": 2. * event,
                                           "draft_training_and_optimizer_s": 3. * event},
                                "response_presentations_this_event": 16,
                                "eligible_position_presentations_this_event": 128,
                                "response_audits": [{"sampled_anchor_count": 8, "eligible_anchor_count": 10}],
                                "batches": [{"presentations": [{"sampled_blocks": 8, "eligible_blocks": 8,
                                                               "eligible_positions": 64, "unique_supervised_positions": 20,
                                                               "weighted_position_sum": 40., "loss": 100000.}]}]}
    return result


def cell(result, *, window="25-32", repeat="pooled"):
    return next(row for row in result["episodes"][0]["windows"] if row["window"] == window and row["repeat"] == repeat)


def comparison(row, candidate, control):
    return next(value for value in row["comparisons"] if value["candidate"] == candidate and value["control"] == control)


class FollowupReportTests(unittest.TestCase):
    def test_requested_same_input_comparisons_and_ratio_denominators(self):
        output = report.summarize([example()])
        final = cell(output)
        self.assertAlmostEqual(comparison(final, "C2", "C1")["latency_reduction_percent"], 100 * (1 - 1.6 / 1.8))
        self.assertAlmostEqual(comparison(final, "A2", "C2")["latency_reduction_percent"], 6.25)
        early = cell(output, window="17-24")
        self.assertAlmostEqual(comparison(early, "A1", "C1")["latency_reduction_percent"], 100 * (1 - 1.7 / 1.8))
        frozen = final["metrics"]["F"]
        self.assertEqual(frozen["executions"], 24)
        self.assertEqual(frozen["distinct_request_contexts"], 8)
        self.assertEqual(frozen["output_tokens"], 48)
        self.assertEqual(frozen["verification_calls"], 240)
        self.assertAlmostEqual(frozen["calls_per_1000_output_tokens"], 5000.)
        self.assertAlmostEqual(frozen["committed_output_tokens_per_verification"], .2)
        self.assertAlmostEqual(frozen["verification_committed_tokens_per_call"], .1)

    def test_nonoverlapping_cost_excludes_scores_and_nested_timers(self):
        result = report.summarize([example()])["episodes"][0]["cost_breakdown"]
        self.assertTrue(result["complete_nonoverlapping_accounting"])
        self.assertAlmostEqual(result["accounted_author_episode_s"], 32 + 3.2 + 6.4 + .8 + 30)
        self.assertAlmostEqual(result["support_preparation_timers"]["collection_s"]["sum_s"], .64)
        self.assertEqual(result["per_update"][0]["accounting_costs"]["teacher_preparation_s"], 2.)
        self.assertIn("unavailable", result["frozen_full_session_comparison"])

    def test_missing_required_timers_and_partial_trajectory_never_become_free(self):
        for mutation in (lambda x: x["canonical"][4].pop("cache_maintenance_s"),
                         lambda x: x["canonical"][4].pop("activation_s"),
                         lambda x: x["support"][0].pop("observed_support_prepare_s"),
                         lambda x: x["updates"][0].pop("observed_update_call_s"),
                         lambda x: x["canonical"].pop()):
            episode = example()
            mutation(episode)
            result = report.summarize([episode])["episodes"][0]["cost_breakdown"]
            self.assertFalse(result["complete_nonoverlapping_accounting"])
            self.assertIsNone(result["accounted_author_episode_s"])

    def test_mismatched_or_missing_pair_suppresses_gain_but_keeps_metrics(self):
        for mutation in (lambda x: x["scores"].pop(next(i for i, r in enumerate(x["scores"])
                                                       if r["checkpoint_id"] == "C2")),
                         lambda x: next(r for r in x["scores"] if r["checkpoint_id"] == "C2").update(output_ids=[77]),
                         lambda x: next(r for r in x["scores"] if r["checkpoint_id"] == "C2").update(input_ids_sha256="bad"),
                         lambda x: next(r for r in x["scores"] if r["checkpoint_id"] == "C2").update(checkpoint_identity_sha256="changed")):
            episode = example()
            mutation(episode)
            output = report.summarize([episode])
            paired = comparison(cell(output), "C2", "C1")
            self.assertFalse(paired["paired_same_generation_valid"])
            self.assertIsNone(paired["latency_reduction_percent"])
            self.assertGreater(cell(output)["metrics"]["C2"]["executions"], 0)

    def test_scenario_and_pooled_repeat_summaries_do_not_inflate_contexts(self):
        episodes = [example(replica=replica, scenario=scenario)
                    for replica in (1, 2) for scenario in ("STABLE_CODE", "STABLE_MATH")]
        for row in episodes[1]["scores"]:
            row["request_s"] *= 2
        output = report.summarize(episodes)
        pooled = next(row for row in output["pooled"] if row["window"] == "25-32" and row["repeat"] == "pooled")
        self.assertEqual(pooled["metrics"]["F"]["executions"], 96)
        self.assertEqual(pooled["metrics"]["F"]["distinct_request_contexts"], 32)
        self.assertEqual(pooled["expected_distinct_contexts"], 32)
        self.assertEqual(len(output["scenarios"]), 16)
        expected_new = sum(r["request_s"] for e in episodes for r in e["scores"] if r["turn"] >= 25 and r["checkpoint_id"] == "C2")
        expected_old = sum(r["request_s"] for e in episodes for r in e["scores"] if r["turn"] >= 25 and r["checkpoint_id"] == "C1")
        self.assertAlmostEqual(comparison(pooled, "C2", "C1")["latency_reduction_percent"], 100 * (1 - expected_new / expected_old))

    def test_checkpoint_mode_has_no_author_or_full_session_claim(self):
        output = report.summarize([example("checkpoints-only")])
        self.assertEqual({row["window"] for row in output["episodes"][0]["windows"]}, {"25-32"})
        self.assertEqual(set(cell(output)["metrics"]), {"F", "C1", "C2"})
        self.assertIsNone(output["episodes"][0]["cost_breakdown"]["accounted_author_episode_s"])
        self.assertTrue(comparison(cell(output), "C2", "C1")["paired_same_generation_valid"])

    def test_own_history_mismatches_not_confused_with_common_input_failure(self):
        output = report.summarize([example()])
        trajectory = output["episodes"][0]["trajectory"]
        self.assertEqual(trajectory["input_mismatch_turns"], list(range(18, 33)))
        self.assertEqual(trajectory["output_mismatch_turns"], list(range(17, 33)))
        self.assertTrue(all(r["paired_same_generation_valid"] for r in cell(output)["comparisons"]))

    def test_exposure_is_measured_and_loss_values_not_compared(self):
        output = report.summarize([example()])
        exposure = output["episodes"][0]["cost_breakdown"]["per_update"][0]["supervised_exposure"]
        self.assertEqual(exposure["response_presentations_this_event"], 16)
        self.assertEqual(exposure["sampled_presentation_totals"]["unique_supervised_positions"], 20)
        self.assertEqual(exposure["response_audits"][0]["sampled_anchor_count"], 8)
        self.assertNotIn("loss", str(exposure))

    def test_missing_optional_progress_is_unmeasured(self):
        episode = example()
        for row in episode["scores"]:
            row.pop("verification_committed_tokens")
        output = report.summarize([episode])
        self.assertIsNone(cell(output)["metrics"]["F"]["verification_committed_tokens_per_call"])
        self.assertIsNotNone(cell(output)["metrics"]["F"]["committed_output_tokens_per_verification"])

    def test_duplicate_episode_mode_rejected_and_modes_remain_separate(self):
        episode = example()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report.summarize([episode, copy.deepcopy(episode)])
        output = report.summarize([episode, example("checkpoints-only")])
        self.assertEqual({row["mode"] for row in output["pooled"]}, {"author-path", "checkpoints-only"})


if __name__ == "__main__":
    unittest.main()
