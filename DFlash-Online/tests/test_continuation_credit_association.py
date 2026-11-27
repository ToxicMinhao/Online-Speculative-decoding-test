"""Synthetic-only fixed-effects, bootstrap and outcome-label checks."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from experiments.continuation_credit_cpu import association
from experiments.continuation_value_diagnostic import statistics as native


def fixture(responses=10, blocks=24, delta_coefficient=1.75):
    random = np.random.Generator(np.random.PCG64(815))
    rows = []
    for response in range(responses):
        for block in range(blocks):
            e, r, n = 4 + block % 2, 2 + (block // 2) % 2, 8
            h1 = random.uniform(.2, .9)
            error_tie, prefix_tie = block % 11 == 0, block % 13 == 0
            difficulty = 0. if error_tie else random.uniform(.1, 3)
            prefix = 0. if prefix_tie else random.uniform(.1, 3)
            old = r * h1 / 14 - random.uniform(.005, .015)
            later_errors = int(random.integers(1, 4))
            delta = random.uniform(.01, .08) / later_errors
            base = (.2 * response + .3 * np.log1p(difficulty) - .2 * np.log1p(prefix)
                    + .4 * h1 + .6 * old + .2 * later_errors + .15 * r)
            rows.append(dict(
                record_id=f"private_response_{response:03d}", response_ordinal=response,
                anchor=block * 3, high=True, e=e, r=r, n=n,
                beta=np.exp(-(e - 1) / 7), u_old=old, delta_u=delta,
                later_error_count=later_errors, h1=h1, d=difficulty,
                minimum_original_prefix_margin=prefix, empty_prefix=False,
                error_tie=error_tie, prefix_tie=prefix_tie,
                Y_full=base + delta_coefficient * delta,
                beyond_K=.3 + .6 * old + 2.5 * delta,
            ))
    return rows


class SharedCreditAssociationTests(unittest.TestCase):
    def test_design_retains_controls_weights_and_fixed_effects(self):
        rows = fixture(responses=4)
        rows = [row for row in rows if row["response_ordinal"] != 3 or row["anchor"] < 15]
        excluded = dict(rows[0], high=False, anchor=999, Y_full=-99)
        design = association.make_design(rows + [excluded])
        self.assertEqual(len(design.rows), len(rows))
        self.assertEqual(design.columns[-1], "delta_u")
        for name in (*native.CONTINUOUS, *native.INDICATORS, "u_old", "later_error_count"):
            self.assertIn(name, design.columns[:-1])
        for cluster in range(4):
            chosen = design.clusters == cluster
            self.assertAlmostEqual(design.weights[chosen].sum(), 1)
            np.testing.assert_allclose(np.sqrt(design.weights[chosen]) @ design.matrix[chosen], 0, atol=1e-14)
            np.testing.assert_allclose(np.sqrt(design.weights[chosen]) @ design.outcomes[chosen], 0, atol=1e-14)

    def test_two_real_outcomes_recover_known_coefficients(self):
        design = association.make_design(fixture())
        self.assertEqual(design.outcomes.shape[1], 2)
        fit = native.direct_fit(design)
        self.assertEqual(fit["status"], "identified")
        coefficients = association._coefficients(fit)
        self.assertAlmostEqual(coefficients["Y_full"], 1.75, places=9)
        self.assertAlmostEqual(coefficients["beyond_K"], 2.5, places=9)
        self.assertLess(fit["fwl_max_absolute_discrepancy"], 1e-10)

    def test_old_score_and_error_count_explain_their_own_effects(self):
        fitted = native.direct_fit(association.make_design(fixture(delta_coefficient=0)))
        self.assertEqual(fitted["status"], "identified")
        self.assertAlmostEqual(association._coefficients(fitted)["Y_full"], 0, places=9)

    def test_accelerated_direct_and_physical_cluster_copies_agree(self):
        design = association.make_design(fixture(responses=8))
        counts = np.array([2, 0, 1, 2, 0, 1, 1, 1])
        fast = native.ClusterSolver(design).fit(counts)
        direct = native.direct_fit(design, counts)
        native.check_fit_equivalence(fast, direct)
        copies = np.repeat(np.arange(len(design.rows)), counts[design.clusters])
        physical = native.factor_fit(design.matrix[copies], design.outcomes[copies], design.u_scale)
        native.check_fit_equivalence(direct, physical)

    def test_fixed_bootstrap_preserves_failed_draw_and_original_random_sequence(self):
        design = association.make_design(fixture(responses=6))
        solver = native.ClusterSolver(design)
        original_fit = solver.fit
        calls = []
        def fail_once(counts):
            calls.append(counts.copy())
            if len(calls) == 3:
                raise np.linalg.LinAlgError("synthetic failure")
            return original_fit(counts)
        with patch.object(solver, "fit", side_effect=fail_once):
            report = association._bootstrap(design, solver, draws=20)
        expected = np.random.Generator(np.random.PCG64(20261127)).integers(0, 6, (20, 6))
        self.assertEqual(len(calls), 20)
        self.assertEqual(report["attempted"], 20)
        self.assertEqual(report["draw_records"][2]["status"], "invalid_numerical")
        self.assertIsNone(report["draw_records"][2]["coefficients"]["Y_full"])
        for counts, sample, record in zip(calls, expected, report["draw_records"]):
            np.testing.assert_array_equal(counts, np.bincount(sample, minlength=6))
            self.assertEqual(record["response_multiplicity"], counts.tolist())
        self.assertEqual(report["direct_checked_draws"], 4)

    def test_draws_losing_delta_information_remain_unidentifiable(self):
        rows = fixture(responses=2, blocks=40)
        for row in rows:
            if row["response_ordinal"] == 1:
                row["delta_u"] = 0.
        design = association.make_design(rows)
        report = association._bootstrap(design, native.ClusterSolver(design), draws=40)
        self.assertEqual(report["attempted"], 40)
        self.assertGreater(report["status_counts"].get("unidentifiable", 0), 0)
        self.assertEqual(sum(report["status_counts"].values()), 40)

    def test_parent_scaling_and_columns_do_not_depend_on_child_outcomes(self):
        rows = fixture(responses=4)
        altered = deepcopy(rows)
        for row in altered:
            row["Y_full"] *= -7
            row["beyond_K"] += 12
        first, second = association.make_design(rows), association.make_design(altered)
        self.assertEqual(first.scaling, second.scaling)
        self.assertEqual(first.columns, second.columns)
        np.testing.assert_array_equal(first.matrix, second.matrix)
        self.assertFalse(np.array_equal(first.outcomes, second.outcomes))

    def test_public_run_uses_exactly_2000_draws_and_leaves_legacy_constants_intact(self):
        report = association.run_association(fixture(responses=5, blocks=20))
        self.assertEqual(report["settings"]["bootstrap"]["draws"], 2000)
        self.assertEqual(report["settings"]["bootstrap"]["seed"], 20261127)
        self.assertEqual(len(report["bootstrap"]["draw_records"]), 2000)
        self.assertEqual(report["bootstrap"]["unattempted"], 0)
        self.assertAlmostEqual(report["primary"]["coefficient_per_0_1_delta_u"], .175, places=9)
        self.assertAlmostEqual(report["secondary"]["coefficient_per_0_1_delta_u"], .25, places=9)
        self.assertNotIn("u", report["model"]["parent_scaling"])
        self.assertEqual(set(report["model"]["coefficients"]), {"Y_full", "beyond_K"})
        self.assertNotIn("private_response", json.dumps(report, allow_nan=False))
        self.assertEqual(native.BOOTSTRAP_DRAWS, 10_000)
        self.assertEqual(native.BOOTSTRAP_SEED, 20261126)
        self.assertEqual(native.OUTCOMES, ("Y", "useful_repair", "prefix_damaged"))

    def test_empty_population_is_reported_without_inventing_draws(self):
        report = association.run_association([])
        self.assertEqual(report["primary"]["status"], "unidentifiable")
        self.assertEqual(report["bootstrap"]["attempted"], 0)
        self.assertEqual(report["bootstrap"]["unattempted"], 2000)
        json.dumps(report, allow_nan=False)

    def test_alignment_and_nonfinite_inputs_fail_without_row_deletion(self):
        rows = fixture(responses=2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            association.make_design(rows + [rows[0]])
        rows[0]["Y_full"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            association.make_design(rows)


if __name__ == "__main__":
    unittest.main()
