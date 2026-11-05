import copy
import unittest

import numpy as np

from experiments.block_depth.gate import (select_gate_starts, progress_for_proposals,
    pooled_features, weighted_ridge, fit_gate)
from experiments.block_depth.endpoint import endpoint_schedule, validate_execution, analyze_endpoint, paired_bootstrap


def panel():
    return [dict(request_id=f"final-{i}", prompt_ids=[i, 98], stop_token_ids=[99], max_new_tokens=2048) for i in range(384)]


def canonical(request):
    return dict(prompt_ids=request["prompt_ids"], response_ids=[10, 11, 12, 99], finish_reason="eos")


def execution(request, arm="FIVE", repeat=0, latency=1.0):
    return dict(request, arm=arm, repeat=repeat, response_ids=[10, 11, 12, 99], finish_reason="eos", metadata=dict(
        verification_calls=1, accepted_draft_tokens=2, committed_progress=3,
        shallow_rounds=int(arm != "FIVE"), deep_rounds=int(arm == "FIVE"),
        request_latency_s=latency, native_generate_s=latency * .9,
        decode_round_wall_s=latency * .8, cache_maintenance_s=latency * .1,
        feedback_capture_in_timed_request=False, trace=[]))


class GateTests(unittest.TestCase):
    def test_prefix_first_error_bonus_and_seed_exclusion(self):
        result = progress_for_proposals([1, 2, 8] + [0] * 12, [1, 2, 3, 4, 99], [99])
        self.assertEqual(result["accepted_draft_tokens"], 2)
        self.assertEqual(result["committed_progress"], 3)
        result = progress_for_proposals(list(range(15)), list(range(16)), [99])
        self.assertEqual(result["committed_progress"], 16)
        self.assertEqual(result["accepted_draft_tokens"], 15)

    def test_eos_match_repair_and_cap_boundaries(self):
        for proposals, suffix, accepted, progress, matched, cap in (
            ([1, 99] + [0] * 13, [1, 99], 2, 2, True, False),
            ([1, 8] + [0] * 13, [1, 99], 1, 2, False, False),
            ([1, 2] + [0] * 13, [1, 2], 2, 2, False, True),
            ([1] * 15, [], 0, 0, False, False),
        ):
            with self.subTest(suffix=suffix, proposals=proposals[:2]):
                result = progress_for_proposals(proposals, suffix, [99])
                self.assertEqual(result, dict(accepted_draft_tokens=accepted, committed_progress=progress,
                                              matched_eos=matched, censored_at_cap=cap))

    def test_progress_signed_difference_can_be_negative(self):
        shallow = progress_for_proposals([1, 2, 3] + [0] * 12, [1, 2, 3, 99], [99])
        deep = progress_for_proposals([0] * 15, [1, 2, 3, 99], [99])
        self.assertEqual(deep["committed_progress"] - shallow["committed_progress"], -3)

    def test_hash_anchor_sampling_deterministic_distinct_and_bounded(self):
        chosen = select_gate_starts("a", 200)
        self.assertEqual(chosen, select_gate_starts("a", 200))
        self.assertEqual(len(chosen), 64)
        self.assertEqual(chosen, sorted(set(chosen)))
        self.assertNotEqual(chosen, select_gate_starts("b", 200))
        self.assertEqual(select_gate_starts("a", 4), [0, 1, 2, 3])
        self.assertEqual(select_gate_starts("a", 0), [])

    def test_features_use_unknown_proposal_groups(self):
        rows = np.arange(15)[:, None] + np.zeros((15, 32))
        features = pooled_features(rows, 15)
        np.testing.assert_allclose(features[:32], 1)
        np.testing.assert_allclose(features[32:64], 4.5)
        np.testing.assert_allclose(features[64:96], 10.5)
        self.assertEqual(features[-1], np.log(16))

    def test_weighted_ridge_normalization_intercept_and_fold(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(101, 97))
        x[:, 0] = 17
        y, w = rng.normal(size=101) - 4, np.arange(1, 102)
        fitted = weighted_ridge(x, y, w)
        scaled = weighted_ridge(x, y, w * 100)
        np.testing.assert_allclose(fitted["folded_weight"], scaled["folded_weight"], atol=1e-10)
        self.assertEqual(fitted["feature_scale"][0], 1e-6)
        z = (x - fitted["feature_mean"]) / fitted["feature_scale"]
        prediction = z @ fitted["standardized_weight"] + fitted["standardized_intercept"]
        np.testing.assert_allclose(prediction, x @ fitted["folded_weight"] + fitted["folded_intercept"], atol=1e-10)
        mean = np.dot(w / w.sum(), y)
        self.assertAlmostEqual(fitted["standardized_intercept"], mean)
        gradient = z.T @ ((w / w.sum()) * (prediction - y)) + .001 * np.array(fitted["standardized_weight"])
        np.testing.assert_allclose(gradient, 0, atol=1e-11)

    def test_intercept_not_penalized_and_negative_targets_retained(self):
        fit = weighted_ridge(np.ones((4, 97)), [-3] * 4, [1, 3, 5, 7])
        self.assertAlmostEqual(fit["folded_intercept"], -3)
        np.testing.assert_allclose(fit["folded_weight"], 0)

    def test_calibration_uses_only_five_decode_cost(self):
        behaviors, rows = [], []
        for i in range(128):
            behavior = "FIVE" if i % 2 == 0 else "THREE"
            behaviors.append(dict(request_id=f"r{i}", behavior=behavior, observed_rounds=1,
                committed_progress=8, decode_round_wall_s=.008 if behavior == "FIVE" else 100))
            rows.append(dict(request_id=f"r{i}", round_index=0, observed_rounds=1, sampled_rounds=1,
                             p3=3, p5=1, features=[0] * 97))
        fit = fit_gate(rows, behaviors, .5)
        self.assertAlmostEqual(fit["kappa_ms_per_token"], 1)
        self.assertEqual(fit["negative_targets"], 128)
        self.assertEqual(fit["folded_intercept"], -2)
        rows[0]["sampled_rounds"] = 2
        with self.assertRaisesRegex(ValueError, "expansion weight"):
            fit_gate(rows, behaviors, .5)


class EndpointTests(unittest.TestCase):
    def test_fixed_budget_balanced_block_order_and_reversed_second_sweep(self):
        schedule = endpoint_schedule(panel())
        self.assertEqual(len(schedule), 2304)
        self.assertEqual(len({(r["request_id"], r["arm"], r["repeat"]) for r in schedule}), 2304)
        for block in range(24):
            order = [[r["arm"] for r in schedule if r["block"] == block and r["repeat"] == repeat][::16] for repeat in (0, 1)]
            self.assertEqual(order[0], list(reversed(order[1])))

    def test_exact_output_termination_and_capture_contract(self):
        request = panel()[0]
        record = execution(request)
        self.assertEqual(validate_execution(record, request, canonical(request)), [])
        record["response_ids"][1] = 88
        self.assertIn("canonical_parity", validate_execution(record, request, canonical(request)))
        record = execution(request)
        record["metadata"]["trace"] = [{}]
        self.assertIn("timed_capture_or_trace", validate_execution(record, request, canonical(request)))

    def test_counts_reject_accepted_exceeding_output_and_bad_route(self):
        request = panel()[0]
        record = execution(request)
        record["metadata"]["accepted_draft_tokens"] = 4
        record["metadata"]["shallow_rounds"] = 1
        self.assertEqual(set(validate_execution(record, request, canonical(request))), {"native_count_consistency", "route_counts"})

    def test_pair_bootstrap_keeps_questions_paired(self):
        comparisons = paired_bootstrap([[i, i * .8, i * .7] for i in range(1, 9)])
        self.assertAlmostEqual(comparisons["ADAPTIVE_vs_FIVE"]["point_pct"], 30)
        np.testing.assert_allclose(comparisons["ADAPTIVE_vs_FIVE"]["ci95_pct"], [30, 30])

    def test_complete_analysis_uses_one_trajectory_counts_and_both_repeat_times(self):
        requests = panel()
        refs = {r["request_id"]: canonical(r) for r in requests}
        records = []
        for cell in endpoint_schedule(requests):
            latency = {"FIVE": 1, "THREE": .9, "ADAPTIVE": .8}[cell["arm"]] * (1 + cell["repeat"])
            records.append(execution(requests[cell["request_ordinal"]], cell["arm"], cell["repeat"], latency))
        result = analyze_endpoint(records, requests, refs)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["arm_totals"]["FIVE"]["verification_calls"], 384)
        self.assertAlmostEqual(result["arm_totals"]["FIVE"]["request_latency_s"], 576)
        self.assertEqual(result["decision"], "advance_adaptive")
        self.assertTrue(result["practical_two_pct_vs_FIVE_point_reached"])

    def test_partial_work_stays_partial_and_preserves_unknown(self):
        requests = panel()
        result = analyze_endpoint([execution(requests[0])], requests, {requests[0]["request_id"]: canonical(requests[0])}, unknown_executions=1)
        self.assertEqual(result["status"], "incomplete_or_invalid")
        self.assertIsNone(result["comparison"])
        self.assertEqual(result["accounting"]["unattempted"], 2302)
        self.assertEqual(len(result["per_request"]), 384)

    def test_repeat_count_mismatch_invalidates_both_rows(self):
        requests = panel()
        first, second = execution(requests[0]), execution(requests[0], repeat=1)
        second["metadata"].update(verification_calls=2, deep_rounds=2)
        result = analyze_endpoint([first, second], requests, {requests[0]["request_id"]: canonical(requests[0])})
        self.assertEqual(result["accounting"]["invalid"], 2)
        self.assertIn("repeat_consistency:verification_calls", result["measurements"][0]["errors"])

    def test_correct_measurements_in_wrong_order_are_not_a_complete_endpoint(self):
        requests = panel()
        refs = {r["request_id"]: canonical(r) for r in requests}
        records = [execution(requests[c["request_ordinal"]], c["arm"], c["repeat"])
                   for c in endpoint_schedule(requests)]
        records[0], records[1] = records[1], records[0]
        result = analyze_endpoint(records, requests, refs)
        self.assertEqual(result["execution_order_errors"], [0, 1])
        self.assertEqual(result["status"], "incomplete_or_invalid")
        self.assertIsNone(result["comparison"])

    def test_prefill_eos_has_zero_decode_progress(self):
        request = panel()[0]
        record = execution(request)
        record["response_ids"] = [99]
        record["metadata"].update(verification_calls=0,accepted_draft_tokens=0,committed_progress=0,
                                   shallow_rounds=0,deep_rounds=0,decode_round_wall_s=0)
        ref = dict(prompt_ids=request["prompt_ids"], response_ids=[99], finish_reason="eos")
        self.assertEqual(validate_execution(record, request, ref), [])


if __name__ == "__main__":
    unittest.main()
