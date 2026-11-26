"""Synthetic confounding, numerical rank and whole-response sampling checks."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from experiments.continuation_value_diagnostic.statistics import (
    ClusterSolver, analyze, bootstrap, check_fit_equivalence, decision, direct_fit,
    factor_fit, make_design, numerical_contract,
)


def fixture(kind="later", responses=36, blocks=24, epsilon=None):
    rng = np.random.Generator(np.random.PCG64(731))
    rows = []
    for response in range(responses):
        level = response / max(1, responses - 1)
        for block in range(blocks):
            e, r, n = 4 + block % 2, 2 + (block // 2) % 2, 8
            h1 = rng.uniform(.2, .9)
            error_tie, prefix_tie = block % 11 == 0, block % 13 == 0
            difficulty = 0. if error_tie else rng.uniform(0, 3)
            prefix = 0. if prefix_tie else rng.uniform(0, 3)
            deficit = rng.uniform(.05, .6) * (r - 1) * h1 / 14
            if epsilon is not None:
                deficit *= epsilon
            u = r * h1 / 14 - deficit
            baseline = .2 * level + .3 * np.log1p(difficulty) - .2 * np.log1p(prefix)
            if kind == "response":
                outcome = .5 * level
            elif kind == "run":
                outcome = .5 * r - .25 * e
            elif kind == "immediate":
                outcome = baseline + h1 + .4 * h1 ** 2 + .7 * r * h1 / 14
            elif kind == "later":
                outcome = baseline + .4 * h1 + 1.5 * u
            else:
                raise ValueError(kind)
            rows.append(dict(record_id=f"response_{response:03d}", anchor=block * 2,
                             high=True, e=e, r=r, n=n, m=e+n, beta=np.exp(-(e - 1) / 7),
                             u=u, D=deficit, h1=h1, d=difficulty, error_margin=-difficulty,
                             minimum_original_prefix_margin=prefix, empty_prefix=False,
                             error_tie=error_tie, prefix_tie=prefix_tie,
                             Y=outcome, useful_repair=outcome > 0, prefix_damaged=outcome < 0))
    return rows


class ContinuationStatisticsTests(unittest.TestCase):
    def test_contract_is_fixed_before_outcomes(self):
        contract = numerical_contract()
        self.assertEqual(contract["rank_rcond"], 1e-10)
        self.assertEqual(contract["bootstrap"]["draws"], 10000)
        self.assertEqual(contract["bootstrap"]["seed"], 20261126)
        self.assertEqual(contract["support"]["minimum_responses"], 30)

    def test_response_weights_and_centering(self):
        rows = fixture(responses=5, blocks=20)
        rows = [r for r in rows if r["record_id"] != "response_004" or r["anchor"] < 10]
        design = make_design(rows)
        for response in range(5):
            selected = design.clusters == response
            self.assertAlmostEqual(design.weights[selected].sum(), 1)
            np.testing.assert_allclose(np.sqrt(design.weights[selected]) @ design.matrix[selected], 0, atol=1e-14)

    def test_response_and_joint_run_confounding_removed(self):
        for kind in ("response", "run"):
            fitted = direct_fit(make_design(fixture(kind)))
            self.assertEqual(fitted["status"], "identified")
            self.assertAlmostEqual(fitted["theta"], 0, places=10)

    def test_immediate_confidence_controls_are_required(self):
        rows = fixture("immediate")
        full = direct_fit(make_design(rows))
        reduced = direct_fit(make_design(rows, include_immediate=False))
        self.assertAlmostEqual(full["theta"], 0, places=9)
        self.assertGreater(abs(reduced["theta"]), 1)

    def test_added_later_information_and_FWL_agree(self):
        fitted = direct_fit(make_design(fixture("later")))
        self.assertEqual(fitted["status"], "identified")
        self.assertAlmostEqual(fitted["theta"], 1.5, places=9)
        self.assertLess(fitted["fwl_max_absolute_discrepancy"], 1e-10)

    def test_crossproducts_match_direct_for_cluster_multiplicity(self):
        design = make_design(fixture(responses=12))
        solver = ClusterSolver(design)
        counts = np.array([2, 0, 1, 2, 0, 0, 3, 1, 1, 0, 1, 1])
        accelerated = solver.fit(counts)
        reference = direct_fit(design, counts)
        check_fit_equivalence(accelerated, reference)
        repeated = np.repeat(np.arange(len(design.rows)), counts[design.clusters])
        physical_copies = factor_fit(design.matrix[repeated], design.outcomes[repeated], design.u_scale)
        check_fit_equivalence(reference, physical_copies)
        self.assertEqual(accelerated["solver"], "whitened_crossproduct_svd")

    def test_disappearing_stratum_can_reduce_rank_without_invalidating_theta(self):
        rows = fixture(responses=12)
        for row in rows:
            if row["record_id"] == "response_000" and row["anchor"] % 4 == 0:
                row["r"] = 5
                row["u"] = row["r"] * row["h1"] / 14 - row["D"]
        design = make_design(rows)
        solver = ClusterSolver(design)
        counts = np.ones(12, dtype=int); counts[0] = 0
        dropped = solver.fit(counts)
        check_fit_equivalence(dropped, direct_fit(design, counts))
        self.assertEqual(dropped["status"], "identified")
        self.assertLess(dropped["baseline_rank"], direct_fit(design)["baseline_rank"])

    def test_nearly_collinear_design_uses_direct_fallback_and_same_rank(self):
        design = make_design(fixture(responses=12, epsilon=1e-8))
        solver = ClusterSolver(design)
        counts = np.ones(12, dtype=int)
        result = solver.fit(counts)
        self.assertEqual(result["solver"], "direct_svd")
        self.assertEqual(result["fallback_reason"], "global_condition")
        check_fit_equivalence(result, direct_fit(design, counts))
        weaker = make_design(fixture(responses=12, epsilon=1e-13))
        self.assertEqual(direct_fit(weaker)["status"], "unidentifiable")

    def test_fixed_bootstrap_keeps_invalid_draws_and_does_not_resample(self):
        design = make_design(fixture(responses=6, blocks=12))
        solver = ClusterSolver(design)
        original = solver.fit
        calls = []
        def once_failed(counts):
            calls.append(counts.copy())
            if len(calls) == 2:
                raise np.linalg.LinAlgError("synthetic numerical failure")
            return original(counts)
        with patch.object(solver, "fit", side_effect=once_failed):
            result = bootstrap(design, solver, .01, draws=25)
        self.assertEqual(len(calls), 25)
        self.assertIsNone(result["coefficients"][1])
        self.assertEqual(result["statuses"][1], "invalid_numerical")
        self.assertEqual(result["numerical_failures"][0]["draw"], 1)
        self.assertTrue(all(c.sum() == 6 for c in calls))

    def test_bootstrap_retains_draws_without_identifiable_score_variation(self):
        rows = fixture(responses=2, blocks=24)
        for row in rows:
            if row["record_id"] == "response_001":
                row["D"] = 0.
                row["u"] = row["r"] * row["h1"] / 14
        design = make_design(rows)
        result = bootstrap(design, ClusterSolver(design), .01, draws=100)
        self.assertEqual(result["planned"], 100)
        self.assertEqual(len(result["coefficients"]), 100)
        self.assertGreater(result["status_counts"].get("unidentifiable", 0), 0)
        self.assertLess(result["valid_fraction"], .95)

    def test_constant_D_does_not_pass_support_via_raw_u_variation(self):
        rows = fixture(responses=6, blocks=12)
        for row in rows:
            row["D"] = 0.
            row["u"] = row["r"] * row["h1"] / 14
        with patch("experiments.continuation_value_diagnostic.statistics.BOOTSTRAP_DRAWS", 25):
            result = analyze(rows)
        self.assertTrue(all(r["D_range"] == 0 for r in result["per_response"]))
        self.assertTrue(all(not r["supports_later_information"] for r in result["per_response"]))
        self.assertEqual(result["decision"]["label"], "NOT_SUPPORTED_IN_THIS_DIAGNOSTIC")

    def test_decision_checks_support_before_positive_interval(self):
        primary = dict(status="identified", residual_df=100)
        boot = dict(valid_fraction=1., theta_ci95=[1., 2.])
        influence = [dict(theta=1.)] * 40
        self.assertEqual(decision(primary, 29, boot, influence, correctness_passed=True,
                                  structural_absence=False)["label"], "INCONCLUSIVE")
        self.assertEqual(decision(primary, 30, boot, influence, correctness_passed=True,
                                  structural_absence=False)["label"], "SUPPORTED_FOR_A_SEPARATE_PILOT")
        boot["valid_fraction"] = .94
        self.assertEqual(decision(primary, 40, boot, influence, correctness_passed=True,
                                  structural_absence=False)["label"], "INCONCLUSIVE")
        self.assertEqual(decision(primary, 40, boot, influence, correctness_passed=False,
                                  structural_absence=True)["label"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
