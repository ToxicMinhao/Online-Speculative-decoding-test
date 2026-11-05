"""CPU contracts for sealed inputs and the bounded native route-cost screen."""

from collections import Counter
from copy import deepcopy
import hashlib
import math
import unittest

from experiments.block_depth import data as D, pricing as P
from experiments.block_depth.prepare import historical_overlap


def catalog(split, count):
    offset = 10000 if split == "test" else 0
    return [dict(
        source_row=i, source_id=f"gsm8k:{split}:{i}", question=f"{split} question {i}",
        user_message=f"{split} question {i}", rendered_prompt=f"user: {split} question {i}",
        prompt_ids=[offset + i + 1, 151644],
    ) for i in range(count)]


def prompts(rounds=16):
    return [dict(request_id=f"readout-{i}", observed_rounds=rounds, request_latency_s=0.1)
            for i in range(16)]


def measurements(cohort, times=None):
    times = times or dict(FIVE=5.0, THREE=3.0, ADAPTIVE3=3.5, ADAPTIVE5=5.5)
    return [dict(row, elapsed_ms=times[row["route"]], valid=True)
            for row in P.pricing_schedule(cohort)]


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.train, self.test = catalog("train", 530), catalog("test", 400)

    def test_sizes_order_and_exact_hash(self):
        manifest = D.build_manifests(self.train, self.test)
        parts = manifest["manifests"]
        self.assertEqual({k: v["count"] for k, v in parts.items()},
                         dict(READOUT=384, GATE=128, FINAL=384))
        train = parts["READOUT"]["requests"] + parts["GATE"]["requests"]
        expected = sorted(self.train, key=lambda r: D.selection_key("train", r["source_id"]))[:512]
        self.assertEqual([r["source_id"] for r in train], [r["source_id"] for r in expected])
        self.assertEqual(D.selection_key("train", "gsm8k:train:1"), hashlib.sha256(
            b"vfd_depth35_gsm8k_v1|train|gsm8k:train:1").hexdigest())
        ids = [r["source_id"] for p in parts.values() for r in p["requests"]]
        self.assertEqual(len(set(ids)), 896)

    def test_input_order_and_history_do_not_change_selection(self):
        clean = D.build_manifests(self.train, self.test)
        history = {r["source_id"]: [dict(path="old-results.json")] for r in self.test}
        with_history = D.build_manifests(list(reversed(self.train)), list(reversed(self.test)),
                                        historical_overlap=history)
        for role in clean["manifests"]:
            self.assertEqual(
                [r["source_id"] for r in clean["manifests"][role]["requests"]],
                [r["source_id"] for r in with_history["manifests"][role]["requests"]],
            )
        self.assertEqual(with_history["manifests"]["FINAL"]["prior_project_use_count"], 384)
        self.assertFalse(with_history["selection"]["globally_fresh_claim"])

    def test_selection_is_independent_of_answers_and_old_performance(self):
        clean = D.build_manifests(self.train, self.test)
        for row in self.train + self.test:
            row.update(answer="unused", generated_length=999999, verification_calls=1)
        enriched = D.build_manifests(self.train, self.test)
        self.assertEqual(clean, enriched)

    def test_native_context_eligibility_reserves_full_output_cap(self):
        self.train[0]["prompt_ids"] = [1] * 225
        self.train[1]["prompt_ids"] = [2] * 224
        self.test[0]["prompt_ids"] = [3] * 2049
        self.test[1]["prompt_ids"] = [4] * 2048
        manifest = D.build_manifests(self.train, self.test)
        excluded = {row["source_id"]: row["reason"] for row in manifest["exclusions"]}
        self.assertEqual(excluded["gsm8k:train:0"], "training_context_overflow")
        self.assertEqual(excluded["gsm8k:test:0"], "serving_context_overflow")
        self.assertNotIn("gsm8k:train:1", excluded)
        self.assertNotIn("gsm8k:test:1", excluded)

    def test_text_and_token_deduplication_and_cross_split_isolation(self):
        self.train[1]["question"] = "  TRAIN\tquestion 0  "
        self.train[2]["prompt_ids"] = list(self.train[0]["prompt_ids"])
        chosen = D.build_manifests(self.train, self.test)["manifests"]["READOUT"]["requests"][0]
        self.test[0]["question"] = "  " + chosen["question"].upper() + "\n"
        manifest = D.build_manifests(self.train, self.test)
        excluded = {row["source_id"]: row["reason"] for row in manifest["exclusions"]}
        self.assertEqual(excluded["gsm8k:train:1"], "duplicate_normalized_prompt")
        self.assertEqual(excluded["gsm8k:train:2"], "duplicate_normalized_prompt")
        self.assertEqual(excluded["gsm8k:test:0"], "auxiliary_fit_overlap")

    def test_deduplicate_identical_id_but_reject_conflicting_identity(self):
        self.train.append(deepcopy(self.train[0]))
        manifest = D.build_manifests(self.train, self.test)
        self.assertIn("duplicate_source_identity", [r["reason"] for r in manifest["exclusions"]])
        self.train[-1]["question"] = "Conflicting source"
        with self.assertRaisesRegex(ValueError, "conflicting"):
            D.build_manifests(self.train, self.test)

    def test_source_row_token_integrity_and_shortfall_fail_visibly(self):
        with self.assertRaisesRegex(ValueError, "512"):
            D.build_manifests(self.train[:511], self.test)
        with self.assertRaisesRegex(ValueError, "384"):
            D.build_manifests(self.train, self.test[:383])
        self.train[0]["prompt_ids_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "token binding"):
            D.build_manifests(self.train, self.test)
        self.train[0].pop("prompt_ids_sha256")
        self.train[0]["source_id"] = "gsm8k:test:0"
        with self.assertRaisesRegex(ValueError, "identity"):
            D.build_manifests(self.train, self.test)

    def test_manifest_keeps_exact_text_ids_and_no_mutable_aliases(self):
        original = deepcopy(self.train)
        manifest = D.build_manifests(self.train, self.test)
        first = manifest["manifests"]["READOUT"]["requests"][0]
        source = original[first["source_row"]]
        self.assertEqual(first["question"], source["question"])
        self.assertEqual(first["prompt_ids"], source["prompt_ids"])
        first["prompt_ids"].append(123)
        self.assertEqual(self.train, original)

    def test_rendering_preserves_pr25_split_specific_template(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **options):
                self.messages, self.options = messages, options
                return "wrapped:" + messages[0]["content"]

            def encode(self, rendered, **options):
                self.encode_options = options
                return [1, 2, 3]

        tokenizer = Tokenizer()
        train = D.render_rows(["  Keep\nwording  "], "train", tokenizer)
        self.assertEqual(train[0]["user_message"], "  Keep\nwording  ")
        self.assertFalse(tokenizer.options["enable_thinking"])
        self.assertTrue(tokenizer.options["add_generation_prompt"])
        self.assertFalse(tokenizer.encode_options["add_special_tokens"])
        test = D.render_rows(["Question?"], "test", tokenizer)
        self.assertEqual(test[0]["user_message"], "Question?" + D.SUFFIX)


class HistoricalInputTests(unittest.TestCase):
    def setUp(self):
        self.train, self.test = catalog("train", 2), catalog("test", 2)
        for row in self.train + self.test:
            row["prompt_ids_sha256"] = D.token_digest(row["prompt_ids"])
        self.old = dict(train_requests=[dict(dataset_id="old-id", prompt_ids=self.train[0]["prompt_ids"])])
        self.inventory = dict(
            scan_receipts=[dict(path="retained-inputs.json", bytes=12, sha256="a" * 64, matched_source_rows=[0])],
            used_source_rows=[0], distinct_scanned_files=1, unreadable_or_unavailable=[],
        )
        self.newer = dict(requests=[self.test[1]])

    def test_history_is_bound_to_exact_inputs_without_assuming_train_row_order(self):
        overlap, summary = historical_overlap(self.train, self.test, self.old, self.inventory, self.newer)
        self.assertEqual(summary["current_train_questions_matching_original_prompts"], 1)
        self.assertEqual(set(overlap), {"gsm8k:train:0", "gsm8k:test:0", "gsm8k:test:1"})
        self.assertEqual(overlap["gsm8k:train:0"][0]["original_dataset_ids"], ["old-id"])
        self.assertEqual(overlap["gsm8k:test:0"][0]["inventory_receipt_index"], 0)
        self.assertFalse(summary["historical_exposure_affects_selection"])

    def test_inconsistent_inventory_summary_and_newer_prompt_binding_fail(self):
        self.inventory["used_source_rows"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "summary"):
            historical_overlap(self.train, self.test, self.old, self.inventory, self.newer)
        self.inventory["used_source_rows"] = [0]
        self.newer = deepcopy(self.newer)
        self.newer["requests"][0]["prompt_ids"].append(999)
        with self.assertRaisesRegex(ValueError, "prompt tokens"):
            historical_overlap(self.train, self.test, self.old, self.inventory, self.newer)


class PricingTests(unittest.TestCase):
    def test_distinct_evenly_spaced_starts_cover_boundaries(self):
        self.assertEqual(P.select_round_starts(1), [0])
        self.assertEqual(P.select_round_starts(4), [0, 1, 2, 3])
        self.assertEqual(P.select_round_starts(16), [0, 2, 4, 6, 8, 10, 12, 15])
        for count in (8, 9, 101, 500):
            starts = P.select_round_starts(count)
            self.assertEqual(len(starts), 8)
            self.assertEqual(len(set(starts)), 8)
            self.assertEqual((starts[0], starts[-1]), (0, count - 1))
        with self.assertRaises(ValueError):
            P.select_round_starts(0)

    def test_exact_budget_and_balanced_route_order_at_each_start(self):
        schedule = P.pricing_schedule(prompts())
        self.assertEqual(len(schedule), 2048)
        self.assertEqual(len({(r["request_id"], r["round_index"], r["repeat"], r["route"])
                              for r in schedule}), 2048)
        positions = Counter((r["request_id"], r["round_index"], r["route"], r["route_position"])
                            for r in schedule)
        self.assertEqual(set(positions.values()), {1})
        with self.assertRaisesRegex(ValueError, "first 16"):
            P.pricing_schedule(prompts()[:15])
        cohort = prompts()
        cohort[-1]["request_id"] = cohort[0]["request_id"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            P.pricing_schedule(cohort)

    def test_full_pricing_ratios_prompt_weights_and_continuation_units(self):
        cohort = prompts(16)
        result = P.analyze_pricing(measurements(cohort), cohort)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["accounting"]["valid_unique_measurements"], 2048)
        self.assertEqual(result["optimistic_savings"]["point_pct"], 24.0)
        self.assertEqual(result["optimistic_savings"]["ci95_pct"], [24.0, 24.0])
        self.assertEqual(result["continuation_cost_ms"], 2.0)
        self.assertEqual(result["per_prompt"][0]["expansion_weight"], 2.0)
        self.assertEqual(result["target_saved_ms_per_existing_round"], 0.125)
        self.assertEqual(result["decision"], "continue_to_readout_fit")

    def test_zero_saving_allowed_when_both_adaptive_routes_are_slower(self):
        cohort = prompts()
        times = dict(FIVE=5.0, THREE=1.0, ADAPTIVE3=5.2, ADAPTIVE5=5.8)
        result = P.analyze_pricing(measurements(cohort, times), cohort)
        self.assertEqual(result["optimistic_savings"]["point_pct"], 0.0)
        self.assertEqual(result["decision"], "stop_pricing_only")
        # A fast static-three route cannot be silently substituted for the
        # adaptive route with its real gate/branch overhead.
        self.assertEqual(result["per_prompt"][0]["route_mean_ms"]["THREE"], 1.0)

    def test_economic_gate_is_strict_at_upper_two_percent(self):
        cohort = prompts(8)
        times = dict(FIVE=5.0, THREE=4.0, ADAPTIVE3=4.75, ADAPTIVE5=5.1)
        result = P.analyze_pricing(measurements(cohort, times), cohort)
        self.assertEqual(result["optimistic_savings"]["ci95_pct"], [2.0, 2.0])
        self.assertEqual(result["decision"], "continue_to_readout_fit")

    def test_short_requests_are_not_duplicated_to_fill_eight_starts(self):
        cohort = prompts(1)
        result = P.analyze_pricing(measurements(cohort), cohort)
        self.assertEqual(result["accounting"]["planned_measurements"], 256)
        self.assertEqual(result["per_prompt"][0]["sampled_rounds"], 1)
        self.assertEqual(result["per_prompt"][0]["expansion_weight"], 1.0)

    def test_prompt_bootstrap_preserves_numerator_denominator_pairing(self):
        cohort = prompts(16)
        for ordinal, row in enumerate(cohort):
            row["observed_rounds"] = 8 * (ordinal + 1)
            row["request_latency_s"] = 0.05 * (ordinal + 1)
        result = P.analyze_pricing(measurements(cohort), cohort)
        self.assertEqual(result["optimistic_savings"]["point_pct"], 24.0)
        self.assertEqual(result["optimistic_savings"]["ci95_pct"], [24.0, 24.0])
        self.assertEqual(result, P.analyze_pricing(measurements(cohort), cohort))

    def test_missing_measurements_remain_unattempted_and_block_inference(self):
        cohort = prompts()
        rows = measurements(cohort)[:-1]
        result = P.analyze_pricing(rows, cohort)
        self.assertEqual(result["decision"], "blocked")
        self.assertIsNone(result["optimistic_savings"])
        self.assertEqual(result["accounting"]["unattempted_measurements"], 1)
        self.assertEqual(result["accounting"]["invalid_measurements"], 0)

    def test_failed_native_validation_is_not_reclassified_as_valid(self):
        cohort, rows = prompts(), measurements(prompts())
        rows[12]["valid"] = False
        result = P.analyze_pricing(rows, cohort)
        self.assertEqual(result["decision"], "blocked")
        self.assertEqual(result["accounting"]["invalid_measurements"], 1)
        self.assertEqual(result["accounting"]["unattempted_measurements"], 0)

    def test_duplicates_extra_routes_and_changed_order_fail_visibly(self):
        cohort = prompts()
        original = measurements(cohort)
        cases = [original + [original[0]], list(reversed(original))]
        extra = deepcopy(original)
        extra[0]["route"] = "FAKE"
        cases.append(extra)
        for rows in cases:
            with self.subTest(case=rows[0]["route"]):
                self.assertEqual(P.analyze_pricing(rows, cohort)["decision"], "blocked")

    def test_nonfinite_and_invalid_time_does_not_enter_statistics(self):
        cohort = prompts()
        for value in (0, -1, math.inf, math.nan, True):
            rows = measurements(cohort)
            rows[0]["elapsed_ms"] = value
            self.assertEqual(P.analyze_pricing(rows, cohort)["decision"], "blocked")
        cohort[0]["request_latency_s"] = 0
        with self.assertRaisesRegex(ValueError, "denominator"):
            P.analyze_pricing(measurements(cohort), cohort)


if __name__ == "__main__":
    unittest.main()
