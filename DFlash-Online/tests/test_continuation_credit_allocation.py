"""Literal CPU allocation identities and fixed one-at-a-time sensitivity."""
import math
import unittest

import numpy as np

from experiments.continuation_credit_cpu.allocation import (
    allocate, attach_oof_scores, compare_allocations, sensitivity,
)


def row(anchor, *, e=1, high=True, old=.1, shared=.2, ordinal=0, m=15):
    has_error = e > 0
    return dict(response_ordinal=ordinal, anchor=anchor, e=e, m=m,
                has_error=has_error, high=high if has_error else False,
                beta=math.exp(-(e - 1) / 7) if has_error else 0.,
                u_old=old if has_error else None, u_shared=shared if has_error else None,
                u_const=(old + shared) / 2 if has_error and high else None,
                u_adapt=old + .75 * (shared - old) if has_error and high else None)


def mixed_pool():
    return [row(0, e=1, old=.05, shared=.2),
            row(16, e=3, old=.1, shared=.12),
            row(32, e=5, high=False, old=0., shared=.25),
            row(48, e=-1, m=6)]


def literal_pool(rows, score, gamma=7., eta=.5, tail=True):
    """Scalar implementation of the handoff, independent of array reductions."""
    repairs = [item for item in rows if item["has_error"]]
    beta = [math.exp(-(item["e"] - 1) / gamma) for item in repairs]
    mass = sum(beta)
    high_mass = sum(weight for weight, item in zip(beta, repairs) if item["high"])
    fraction = high_mass / mass if mass else 0.
    c = eta * (1 - fraction) / (1 + fraction)
    kappa = min(c, eta - c)
    mean = (sum(weight * item[score] for weight, item in zip(beta, repairs) if item["high"])
            / high_mass) if high_mass else 0.
    coefficients = []
    for weight, item in zip(beta, repairs):
        if item["high"]:
            value = weight * (1 + c)
            if tail:
                value += kappa * weight * (item[score] - mean)
        else:
            value = weight * (1 - eta * fraction / (1 + fraction))
        coefficients.append(value)
    denominator = sum(sum(math.exp(-(slot - 1) / gamma)
                          for slot in range(1, (item["e"] if item["has_error"] else item["m"]) + 1))
                      for item in rows) + 1e-6
    return coefficients, denominator


class SharedCreditAllocationTests(unittest.TestCase):
    def test_literal_coefficients_and_response_denominator(self):
        rows = mixed_pool()
        for score in ("u_old", "u_shared", "u_const", "u_adapt"):
            coefficients, pools = allocate(rows, score)
            expected, denominator = literal_pool(rows, score)
            np.testing.assert_allclose([item["coefficient"] for item in coefficients], expected,
                                       atol=1e-15, rtol=1e-15)
            self.assertAlmostEqual(pools[0]["denominator"], denominator, places=13)
            np.testing.assert_allclose([item["effective_coefficient"] for item in coefficients],
                                       np.array(expected) / denominator, atol=1e-15, rtol=1e-15)

    def test_empty_all_high_single_high_and_equal_score_noops(self):
        cases = [
            [row(0, e=-1, m=0)],
            [row(0, e=1, old=.1), row(1, e=3, old=.4)],
            [row(0, e=1), row(1, e=3, high=False)],
            [row(0, e=1, old=.2), row(1, e=3, old=.2), row(2, e=5, high=False)],
            [row(0, high=False), row(1, e=3, high=False)],
        ]
        for rows in cases:
            with self.subTest(rows=rows):
                coefficients, pools = allocate(rows)
                self.assertFalse(pools[0]["active"])
                self.assertEqual(pools[0]["moved_from_native"], 0.)
                for item in coefficients:
                    self.assertEqual(item["coefficient"], item["native_coefficient"])
        self.assertEqual(allocate([]), ([], []))

    def test_high_mass_and_bounds_low_unchanged_for_varied_pools(self):
        rng = np.random.Generator(np.random.PCG64(20261127))
        for count in (3, 8, 64):
            rows = [row(i, e=1 + i % 14, high=i % 3 != 0, old=float(rng.random()))
                    for i in range(count)]
            coefficients, pools = allocate(rows)
            self.assertTrue(pools[0]["active"])
            high = [item for item in coefficients if item["high"]]
            self.assertAlmostEqual(sum(item["coefficient"] for item in high),
                                   sum(item["native_coefficient"] for item in high), places=12)
            for item in coefficients:
                if item["high"]:
                    self.assertGreaterEqual(item["coefficient"] + 1e-13, item["beta"])
                    self.assertLessEqual(item["coefficient"], 1.5 * item["beta"] + 1e-13)
                else:
                    self.assertEqual(item["coefficient"], item["native_coefficient"])

    def test_positive_shared_score_never_changes_low_coefficient(self):
        rows = mixed_pool()
        old, _ = allocate(rows, "u_old")
        shared, _ = allocate(rows, "u_shared")
        self.assertGreater(rows[2]["u_shared"], rows[2]["u_old"])
        self.assertEqual(old[2]["coefficient"], shared[2]["coefficient"])

    def test_response_pools_never_mix(self):
        first = mixed_pool()
        second = [row(0, ordinal=1), row(1, ordinal=1, e=3, high=False)]
        together, pools = allocate(first + second)
        separate_first, _ = allocate(first)
        separate_second, _ = allocate(second)
        self.assertEqual(together, separate_first + separate_second)
        self.assertEqual(len(pools), 2)

    def test_uniform_added_score_disappears_after_high_centering(self):
        rows = mixed_pool()
        for item in rows:
            if item["high"]:
                item["u_shared"] = item["u_old"] + .1
        changes, summaries = compare_allocations(rows, scores=("u_old", "u_shared"))
        shared = next(item for item in summaries if item["score"] == "u_shared")
        self.assertAlmostEqual(shared["added_high_score_mean_removed"], .1, places=14)
        self.assertAlmostEqual(shared["increment_energy_fraction_removed"], 1, places=14)
        self.assertLess(shared["moved_vs_old"], 1e-14)
        self.assertTrue(all(abs(item["coefficient_change_vs_old"]) < 1e-14 for item in changes))

    def test_movement_denominators_use_actual_coefficient_mass(self):
        changes, summaries = compare_allocations(mixed_pool(), scores=("u_old", "u_shared"))
        summary = next(item for item in summaries if item["score"] == "u_shared")
        moved = .5 * sum(abs(item["coefficient_change_vs_old"])
                         for item in changes if item["score"] == "u_shared")
        self.assertGreater(moved, 0)
        self.assertAlmostEqual(summary["moved_vs_old"], moved)
        self.assertAlmostEqual(summary["moved_fraction_of_all_repair_mass"], moved / summary["repair_mass"])
        self.assertAlmostEqual(summary["moved_fraction_of_native_high_mass"], moved / summary["native_high_mass"])
        self.assertAlmostEqual(summary["effective_half_l1_vs_old"], moved / summary["denominator"])

    def test_gamma_changes_selected_denominator_including_noerror_blocks(self):
        rows = mixed_pool()
        denominators = []
        for gamma in (5., 7., 10.):
            coefficients, pools = allocate(rows, "u_shared", gamma=gamma)
            expected, denominator = literal_pool(rows, "u_shared", gamma=gamma)
            denominators.append(pools[0]["denominator"])
            self.assertAlmostEqual(pools[0]["denominator"], denominator, places=13)
            np.testing.assert_allclose([item["coefficient"] for item in coefficients], expected, atol=1e-14)
            self.assertAlmostEqual(coefficients[1]["beta"], math.exp(-2 / gamma))
        self.assertLess(denominators[0], denominators[1])
        self.assertLess(denominators[1], denominators[2])

    def test_sensitivity_is_one_at_a_time_and_eta_is_component_only(self):
        rows = mixed_pool()
        outputs, summaries = sensitivity(rows, scores=("u_old", "u_shared"))
        settings = {(item["parameter"], item["value"], item["score"]) for item in summaries}
        self.assertEqual(len(settings), 9)  # Six gamma settings and three eta settings.
        for eta in (.25, .5, .75):
            candidates = [item for item in outputs if item["parameter"] == "eta" and item["value"] == eta]
            expected, denominator = literal_pool(rows, "u_old", eta=eta, tail=False)
            self.assertTrue(all(not item["tail"] and item["gamma"] == 7. for item in candidates))
            np.testing.assert_allclose([item["coefficient"] for item in candidates], expected, atol=1e-14)
            self.assertTrue(all(item["denominator"] == denominator for item in candidates))
            if eta == .5:
                self.assertTrue(all(item["coefficient_change"] == item["effective_change"] == 0 for item in candidates))
        for item in outputs:
            if item["parameter"] == "gamma":
                self.assertEqual(item["eta"], .5)

    def test_oof_probabilities_recover_old_shared_and_intermediate_scores(self):
        rows = [row(0, old=.1, shared=.4), row(1, e=3, old=.2, shared=.2),
                row(2, high=False, old=0, shared=.3)]
        followers = [dict(response_ordinal=0, anchor=0, position=4, credit=.1),
                     dict(response_ordinal=0, anchor=0, position=6, credit=.2)]
        for probability in (0., .25, 1.):
            predictions = [dict(item, rho_CONST=probability, rho_SEVERITY=probability)
                           for item in followers]
            result = attach_oof_scores(rows, followers, predictions)
            for name in ("u_const", "u_adapt"):
                self.assertAlmostEqual(result[0][name], .1 + probability * .3)
                self.assertEqual(result[1][name], rows[1]["u_old"])
                self.assertIsNone(result[2][name])
                self.assertLessEqual(rows[0]["u_old"], result[0][name])
                self.assertLessEqual(result[0][name], rows[0]["u_shared"])
        self.assertNotEqual(rows[0]["u_const"], .4)  # Original records are not mutated.

    def test_missing_oof_fit_and_out_of_bound_score_are_explicit(self):
        rows = [row(0, old=.1, shared=.2)]
        follower = dict(response_ordinal=0, anchor=0, position=4, credit=.1)
        for probability in (None, 2., float("nan")):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                attach_oof_scores(rows, [follower], [dict(follower, rho_CONST=probability, rho_SEVERITY=.5)])


if __name__ == "__main__":
    unittest.main()
