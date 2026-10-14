"""Synthetic reporting regressions. No models, data files or torch are needed."""
import copy
import unittest

try:
    from . import report
except ImportError:
    import report


def score(turn, checkpoint, repeat=0, *, seconds=None):
    # Alternating short/long responses expose accidental means of per-request
    # rates: sums use80tokens and16calls rather than equal request weighting.
    short = turn % 2 == 1
    tokens = 2 if short else 18
    calls = 1 if short else 3
    if seconds is None:
        seconds = (1.0 if short else 3.0) * (1.25 if checkpoint in ("U1", "U2") else 1.0)
    return dict(turn=turn, question_id=f"q{turn}", input_ids_sha256=f"input{turn}",
                repeat=repeat, checkpoint_id=checkpoint, output_ids=list(range(tokens)),
                verification_calls=0 if checkpoint == "AR" else calls,
                accepted_draft_tokens=0 if checkpoint == "AR" else (0 if short else 8),
                request_s=seconds, same_generation_valid=True, canonical_output_match=True,
                cache_maintenance_s=0.2)


def episode(*, name="episode1", scenario="STABLE_CODE", replica=1):
    canonical = []
    for turn in range(1, 33):
        checkpoint = "F" if turn <= 16 else "U1" if turn <= 24 else "U2"
        canonical.append(dict(score(turn, checkpoint, seconds=2.0), activation_s=0.1,
            capture_s=800.0, feedback={"costs": {"capture_s": 900.0}},
            nested_phase_seconds={"capture_s": 1000.0}))
    scores = [score(turn, arm, repeat) for repeat in range(3)
              for low, adapted in ((17, "U1"), (25, "U2"))
              for arm in ("F", adapted, "AR") for turn in range(low, low + 8)]
    return dict(episode_id=name, scenario=scenario, replica=replica, status="complete",
                canonical=canonical, scores=scores,
                updates=[dict(event=1, checkpoint_id="U1", optimizer_steps=8,
                              observed_update_call_s=10.0,
                              costs={"training_s": 8.0, "replay_s": 5.0, "activation_s": 2.0}),
                         dict(event=2, checkpoint_id="U2", optimizer_steps=16,
                              observed_update_call_s=20.0,
                              costs={"training_s": 16.0, "replay_s": 10.0, "activation_s": 4.0})],
                baseline_early=[score(t, "F", seconds=1.0) for t in range(1, 17)],
                experiment_wall_s=9999.0, cleanup={"initial_loading_s": 123.0})


def summarized(value):
    return report.summarize([value])["episodes"][0]


def window(value, turns="17–24", repeat=0):
    return next(row for row in value["windows"] if row["turns"] == turns and row["repeat"] == repeat)


class ReportTests(unittest.TestCase):
    def test_weighted_call_acceptance_and_throughput_metrics(self):
        result = summarized(episode())
        metrics = window(result)["metrics"]["F"]
        self.assertEqual(metrics["requests"], 8)
        self.assertEqual(metrics["output_tokens"], 80)
        self.assertEqual(metrics["verification_calls"], 16)
        self.assertEqual(metrics["accepted_draft_tokens"], 32)
        self.assertEqual(metrics["complete_request_s"], 16.0)
        self.assertEqual(metrics["calls_per_1000_output_tokens"], 200.0)
        self.assertEqual(metrics["accepted_draft_tokens_per_verification"], 2.0)
        self.assertEqual(metrics["committed_output_tokens_per_verification"], 5.0)
        self.assertEqual(metrics["output_tokens_per_second"], 5.0)
        # Averaging the short/long call rates gives333.33, not200.
        self.assertNotAlmostEqual(metrics["calls_per_1000_output_tokens"], (1000 / 2 + 3000 / 18) / 2)
        self.assertNotAlmostEqual(metrics["output_tokens_per_second"], (2 / 1 + 18 / 3) / 2)

    def test_all_repeats_windows_episodes_and_negative_effects_are_retained(self):
        source = [episode(), episode(name="episode2", scenario="RANDOM_MIX", replica=2)]
        result = report.summarize(source)
        self.assertEqual(len(result["episodes"]), 2)
        for item in result["episodes"]:
            self.assertEqual(len(item["windows"]), 6)
            self.assertEqual({(w["turns"], w["repeat"]) for w in item["windows"]},
                             {(block, repeat) for block in ("17–24", "25–32") for repeat in range(3)})
            for w in item["windows"]:
                self.assertTrue(w["paired_same_generation_valid"])
                self.assertAlmostEqual(w["latency_reduction_percent"], -25.0)

    def test_positive_effect_is_sum_of_paired_times_not_mean_of_percentages(self):
        source = episode()
        for row in source["scores"]:
            if row["checkpoint_id"] == "U1":
                row["request_s"] = 0.5 if row["turn"] % 2 else 2.7
        result = window(summarized(source))
        self.assertTrue(result["paired_same_generation_valid"])
        self.assertAlmostEqual(result["latency_reduction_percent"], 20.0)
        self.assertNotAlmostEqual(result["latency_reduction_percent"], (50.0 + 10.0) / 2)

    def test_mismatch_flag_retains_metrics_but_never_claims_gain(self):
        source = episode()
        source["scores"][0]["same_generation_valid"] = False
        source["scores"][0]["output_ids"] = [777]
        result = window(summarized(source))
        self.assertFalse(result["paired_same_generation_valid"])
        self.assertIsNone(result["latency_reduction_percent"])
        self.assertEqual(result["metrics"]["F"]["requests"], 8)
        self.assertEqual(result["metrics"]["F"]["output_tokens"], 79)

    def test_partial_window_input_mismatch_or_question_mismatch_never_claims_gain(self):
        for case in ("missing", "input", "question"):
            source = episode()
            index = next(i for i, row in enumerate(source["scores"]) if row["checkpoint_id"] == "U1")
            if case == "missing":
                source["scores"].pop(index)
            elif case == "input":
                source["scores"][index]["input_ids_sha256"] = "different-input"
            else:
                source["scores"][index]["question_id"] = "different-question"
            with self.subTest(case=case):
                result = window(summarized(source))
                self.assertFalse(result["paired_same_generation_valid"])
                self.assertIsNone(result["latency_reduction_percent"])

    def test_duplicate_rows_do_not_count_as_a_complete_eight_turn_window(self):
        source = episode()
        source["scores"] = [copy.deepcopy(score(17, arm)) for arm in ("F", "U1", "AR") for _ in range(8)]
        result = window(summarized(source))
        self.assertFalse(result["paired_same_generation_valid"])
        self.assertIsNone(result["latency_reduction_percent"])
        self.assertFalse(result["AR_same_generation_valid"])

    def test_wrong_turn_range_does_not_count_as_expected_window(self):
        source = episode()
        source["scores"] = [score(turn, arm) for arm in ("F", "U1", "AR") for turn in range(9, 17)]
        result = window(summarized(source))
        self.assertFalse(result["paired_same_generation_valid"])
        self.assertIsNone(result["latency_reduction_percent"])

    def test_raw_output_mismatch_overrides_contradictory_validity_flags(self):
        source = episode()
        item = next(row for row in source["scores"] if row["checkpoint_id"] == "U1")
        item["output_ids"] = [777] * len(item["output_ids"])
        result = window(summarized(source))
        self.assertFalse(result["paired_same_generation_valid"])
        self.assertIsNone(result["latency_reduction_percent"])

    def test_ar_zero_verifications_and_mismatch_are_separate_reference_limits(self):
        source = episode()
        for row in source["scores"]:
            if row["checkpoint_id"] == "AR":
                row["same_generation_valid"] = False
        result = window(summarized(source))
        self.assertTrue(result["paired_same_generation_valid"])
        self.assertFalse(result["AR_same_generation_valid"])
        self.assertEqual(result["metrics"]["AR"]["verification_calls"], 0)
        self.assertEqual(result["metrics"]["AR"]["calls_per_1000_output_tokens"], 0.0)
        self.assertIsNone(result["metrics"]["AR"]["accepted_draft_tokens_per_verification"])
        self.assertIsNone(result["metrics"]["AR"]["committed_output_tokens_per_verification"])
        self.assertGreater(result["metrics"]["AR"]["output_tokens_per_second"], 0)

    def test_ar_raw_output_or_input_mismatch_overrides_validity_flags(self):
        for case in ("output", "input"):
            source = episode()
            row = next(r for r in source["scores"] if r["checkpoint_id"] == "AR")
            if case == "output":
                row["output_ids"] = [777]
            else:
                row["input_ids_sha256"] = "another-input"
            with self.subTest(case=case):
                result = window(summarized(source))
                self.assertTrue(result["paired_same_generation_valid"])
                self.assertFalse(result["AR_same_generation_valid"])

    def test_nested_capture_replay_training_and_wall_time_are_not_added_twice(self):
        source = episode()
        result = summarized(source)
        self.assertEqual(result["canonical_serving_s"], 64.0)
        self.assertAlmostEqual(result["canonical_cache_maintenance_s"], 6.4)
        self.assertAlmostEqual(result["canonical_publication_s"], 3.2)
        self.assertEqual(result["observed_synchronous_update_wait_s"], 30.0)
        self.assertAlmostEqual(result["accounted_adaptive_episode_s"], 103.6)
        self.assertEqual(result["experiment_wall_s"], 9999.0)
        self.assertEqual(result["updates"], source["updates"])
        # Initial16 frozen requests are reused, then each repeat contributes
        # exactly16frozen scored requests; all include cache cost once.
        self.assertEqual(len(result["frozen_episode_reference_s_by_repeat"]), 3)
        for value in result["frozen_episode_reference_s_by_repeat"]:
            self.assertAlmostEqual(value, 19.2 + 35.2)

    def test_incomplete_or_failed_episode_has_no_whole_episode_cost_claim(self):
        for case in ("failed", "missing_turn", "missing_update"):
            source = episode()
            if case == "failed":
                source.update(status="failed", error="fixture failure")
            elif case == "missing_turn":
                source["canonical"].pop()
            else:
                source["updates"].pop()
            with self.subTest(case=case):
                result = summarized(source)
                self.assertIsNone(result["accounted_adaptive_episode_s"])
                self.assertEqual(result["frozen_episode_reference_s_by_repeat"], [None] * 3)
                self.assertEqual(result["status"], source["status"])
                self.assertGreater(result["canonical_serving_s"], 0)

    def test_duplicate_canonical_turns_or_updates_do_not_complete_episode(self):
        for case in ("turn", "update"):
            source = episode()
            if case == "turn":
                source["canonical"][-1] = copy.deepcopy(source["canonical"][-2])
            else:
                source["updates"][-1] = copy.deepcopy(source["updates"][0])
            with self.subTest(case=case):
                result = summarized(source)
                self.assertIsNone(result["accounted_adaptive_episode_s"])
                self.assertEqual(result["frozen_episode_reference_s_by_repeat"], [None] * 3)

    def test_duplicate_early_baseline_turns_do_not_create_full_session_reference(self):
        source = episode()
        source["baseline_early"][-1] = copy.deepcopy(source["baseline_early"][-2])
        result = summarized(source)
        self.assertIsNotNone(result["accounted_adaptive_episode_s"])
        self.assertEqual(result["frozen_episode_reference_s_by_repeat"], [None] * 3)

    def test_empty_failed_episode_is_retained_without_nan_or_rates(self):
        source = episode()
        source.update(status="failed", error="failed before serving", canonical=[], scores=[], updates=[], baseline_early=[])
        result = summarized(source)
        self.assertEqual(result["error"], "failed before serving")
        self.assertIsNone(result["accounted_adaptive_episode_s"])
        for item in result["windows"]:
            self.assertFalse(item["paired_same_generation_valid"])
            self.assertIsNone(item["latency_reduction_percent"])
            for metric in item["metrics"].values():
                self.assertEqual(metric["requests"], 0)
                self.assertIsNone(metric["calls_per_1000_output_tokens"])
                self.assertIsNone(metric["output_tokens_per_second"])

    def test_zero_base_timer_never_divides_by_zero_or_claims_a_gain(self):
        source = episode()
        for row in source["scores"]:
            if row["checkpoint_id"] == "F":
                row["request_s"] = 0.0
        result = window(summarized(source))
        self.assertTrue(result["paired_same_generation_valid"])
        self.assertIsNone(result["latency_reduction_percent"])
        self.assertIsNone(result["metrics"]["F"]["output_tokens_per_second"])

    def test_summarization_does_not_mutate_source_measurements(self):
        source = episode()
        before = copy.deepcopy(source)
        report.summarize([source])
        self.assertEqual(source, before)


if __name__ == "__main__":
    unittest.main()
