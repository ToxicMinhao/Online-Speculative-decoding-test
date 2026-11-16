"""CPU contracts for the prespecified scalar-control attribution reduction."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.native_update_size import analysis


def make_grid(requests=2, starts=4, windows=(8, 16), prefixes=None):
    sealed, rows = [], []
    for request in range(requests):
        for start in range(starts):
            key = dict(request_id=f"r{request:02d}", canonical_start=start * 10, round_ordinal=start)
            sealed.append(key)
            for window in windows:
                values = prefixes(request, start, window) if prefixes else dict(P=3, N=4, E=5, S_low=4, S_high=4)
                for state in analysis.STATES:
                    value = values[state]
                    rows.append(dict(key, window=window, state=state, accepted_prefix=value,
                        valid_proposals=15, first_error=value + 1 if value < 15 else None,
                        matched_eos=False, censored_at_cap=False, valid=True, reused_from=None))
    return sealed, rows


def reduce_fixture(rows, sealed, windows=(8, 16)):
    return analysis.reduce_probes(rows, sealed, windows=windows,
        require_full_population=False, draws=1000)


class ReductionTests(unittest.TestCase):
    def test_complete_production_population_and_historical_count(self):
        sealed, rows = make_grid(64, windows=analysis.WINDOWS)
        result, requests, windows, blocks = analysis.reduce_probes(rows, sealed)
        self.assertEqual((len(requests), len(windows), len(blocks)), (64, 16, 4096))
        self.assertEqual(result["historical_observations"], 12288)
        self.assertEqual(result["new_logical_control_observations"], 8192)
        self.assertEqual(result["contrasts"]["E_N"]["estimate"], 1)
        self.assertEqual(result["contrasts"]["E_N"]["measurement_origin"], "historical_PR47_recomputed")
        self.assertEqual(result["bootstrap"]["seed"], 1707)

    def test_request_means_do_not_weight_by_valid_length(self):
        def values(request, start, window):
            return dict(P=1, N=1, E=1 + int(request == 0 and start == 0 and window == 8), S_low=1, S_high=1)
        sealed, rows = make_grid(prefixes=values)
        for row in rows:
            if row["request_id"] == "r00":
                row["valid_proposals"] = 3
        result, requests, _, _ = reduce_fixture(rows, sealed)
        self.assertEqual(requests[0]["E_N"], 1 / 8)
        self.assertEqual(requests[1]["E_N"], 0)
        self.assertEqual(result["contrasts"]["E_N"]["exact"], {"numerator": 1, "denominator": 16})

    def test_bootstrap_uses_shared_whole_request_draws(self):
        rows = []
        names = [f"{a}_{b}" for a, b in analysis.CONTRASTS]
        for i in (2, 0, 3, 1):
            rows.append(dict(request_id=f"r{i}", **{name: float((j + 1) * (i - 1)) for j, name in enumerate(names)}))
        result = analysis.bootstrap_requests(rows, draws=1000)
        rng = np.random.Generator(np.random.PCG64(1707))
        draws = rng.integers(0, 4, size=(1000, 4))
        base = np.asarray([-1., 0., 1., 2.])[draws].mean(1)
        for j, name in enumerate(names):
            expected = np.quantile(base * (j + 1), [.025, .975], method="linear")
            self.assertEqual(result["intervals"][name], dict(lower=expected[0], upper=expected[1]))

    def test_blocker_break_and_displaced_repair_denominators_are_distinct(self):
        def values(request, start, window):
            return (dict(P=15, N=14, E=13, S_low=15, S_high=15) if start == 0 else
                    dict(P=0, N=2, E=0, S_low=1, S_high=0))
        sealed, rows = make_grid(requests=1, starts=2, windows=(8,), prefixes=values)
        result, _, _, blocks = reduce_fixture(rows, sealed, (8,))
        self.assertEqual(result["denominators"]["pre_first_blocker_opportunities"], 1)
        self.assertEqual(result["denominators"]["pre_prefix_break_opportunities"], 1)
        self.assertEqual(result["denominators"]["native_repair_displacement_opportunities"], 1)
        self.assertEqual(result["gross_counts"]["E_displaced_native_repair"], 1)
        self.assertEqual(result["gross_counts"]["E_earlier_break"], 1)
        self.assertEqual(blocks[0]["E_N_net"], -1)
        self.assertEqual(blocks[1]["E_N_net"], -2)

    def test_exact_alias_counted_once_and_equal_outcomes_not_assumed_alias(self):
        sealed, rows = make_grid()
        result, *_ = reduce_fixture(rows, sealed)
        self.assertEqual(result["distinct_primary_comparisons"], 2)
        for row in rows:
            if row["state"] == "S_high":
                row["reused_from"] = "S_low"
        result, *_ = reduce_fixture(rows, sealed)
        self.assertEqual(result["distinct_primary_comparisons"], 1)
        self.assertEqual(result["contrasts"]["S_high_S_low"]["estimate"], 0)

    def test_bad_alias_and_missing_duplicate_invalid_boundaries_fail(self):
        sealed, rows = make_grid()
        cases = [rows[:-1], rows + [rows[0]]]
        for field, value in (("valid", False), ("first_error", None), ("valid_proposals", 14)):
            changed = deepcopy(rows)
            changed[0][field] = value
            cases.append(changed)
        bad_alias = deepcopy(rows)
        bad_alias[0]["reused_from"] = "S_low"
        cases.append(bad_alias)
        for changed in cases:
            with self.subTest(change=changed[0]), self.assertRaises(ValueError):
                reduce_fixture(changed, sealed)

    def test_production_population_or_seed_cannot_drift(self):
        sealed, rows = make_grid()
        with self.assertRaises(ValueError):
            analysis.reduce_probes(rows, sealed)
        sealed, rows = make_grid(64, windows=analysis.WINDOWS)
        for kwargs in (dict(seed=17), dict(draws=100)):
            with self.assertRaises(ValueError):
                analysis.reduce_probes(rows, sealed, **kwargs)


class DecisionTests(unittest.TestCase):
    def summary(self):
        return {"contrasts": {f"{a}_{b}": dict(estimate=1, lower=.1, upper=2) for a, b in analysis.CONTRASTS}}

    def test_joint_success_requires_both_primary_bounds(self):
        summary = self.summary()
        valid = dict(complete=True, all_required_checks_passed=True)
        self.assertEqual(analysis.decide(summary, valid), "LOCAL_ROLE_ADVANTAGE_BEYOND_BOTH_NEARBY_SIZE_CONTROLS")
        summary["contrasts"]["E_S_high"]["lower"] = 0
        self.assertNotEqual(analysis.decide(summary, valid), "LOCAL_ROLE_ADVANTAGE_BEYOND_BOTH_NEARBY_SIZE_CONTROLS")
        self.assertEqual(analysis.decide(summary, {}), "INCOMPLETE")
        self.assertEqual(analysis.decide(summary, dict(complete=True)), "INVALID_OR_UNQUALIFIED")

    def test_negative_and_unresolved_results_are_not_equivalence(self):
        summary = self.summary()
        valid = dict(complete=True, all_required_checks_passed=True)
        for name in analysis.PRIMARY:
            summary["contrasts"][name].update(estimate=-1, lower=-2, upper=-.1)
        self.assertEqual(analysis.decide(summary, valid), "BOTH_SIZE_CONTROLS_OUTPERFORM_HISTORICAL_E")
        for value in summary["contrasts"].values():
            value.update(estimate=0, lower=-1, upper=1)
        self.assertEqual(analysis.decide(summary, valid), "UNRESOLVED_STOP_THIS_SCREEN")

    def test_incomplete_report_preserves_unknown_and_unattempted_without_effect(self):
        sealed, rows = make_grid()
        retained = rows[:2] + [dict(rows[2], valid=False, status="unknown", error="worker unavailable")]
        with tempfile.TemporaryDirectory() as folder:
            result = analysis.write_report(folder, probe_rows=retained, sealed_probes=sealed,
                validity=dict(complete=False), work={}, execution_commit="test",
                windows=(8, 16), require_full_population=False)
            self.assertEqual(result["grid"]["unknown"], 1)
            self.assertEqual(result["grid"]["unattempted"], len(rows) - 3)
            self.assertNotIn("comparison", result)
            self.assertIn("worker unavailable", (Path(folder) / "PROBE_OUTCOMES.csv").read_text())
            self.assertEqual(json.loads((Path(folder) / "SUMMARY.json").read_text())["status"], "incomplete")

    def test_complete_claim_cannot_override_bad_grid(self):
        sealed, rows = make_grid()
        with tempfile.TemporaryDirectory() as folder:
            result = analysis.write_report(folder, probe_rows=rows[:-1], sealed_probes=sealed,
                validity=dict(complete=True, all_required_checks_passed=True), work={}, execution_commit="test",
                windows=(8, 16), require_full_population=False)
            self.assertEqual(result["status"], "invalid")
            self.assertNotIn("comparison", result)


if __name__ == "__main__":
    unittest.main()
