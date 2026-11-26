"""Synthetic CPU checks for the independent result auditor."""

import unittest

import numpy as np

from audit_statistics import construct, explicit_response_fit, fit


def rows_fixture():
    rng = np.random.default_rng(98371)
    rows = []
    for response in range(8):
        for anchor in range(30 - response):
            error, run, available = 2 + anchor % 2, 2 + anchor % 3, 8
            confidence = rng.uniform(.2, .8)
            difficulty = rng.uniform(.1, 3)
            prefix = rng.uniform(.1, 2)
            deficit = rng.uniform(.02, .08) * confidence
            score = run * confidence / 14 - deficit
            value = response * .4 + .2 * np.log1p(difficulty) + .4 * confidence + 1.75 * score
            rows.append(dict(
                record_id=str(response), anchor=anchor, high=True, e=error, r=run, n=available,
                beta=np.exp(-(error - 1) / 7), d=difficulty, minimum_original_prefix_margin=prefix,
                h1=confidence, u=score, error_tie=False, prefix_tie=False, empty_prefix=False,
                Y=value, useful_repair=value > 0, prefix_damaged=value < 0,
            ))
    return rows


class IndependentStatisticsTests(unittest.TestCase):
    def test_construct_weights_and_known_conditional_coefficient(self):
        design = construct(rows_fixture())
        for group in range(8):
            self.assertAlmostEqual(design['weights'][design['groups'] == group].sum(), 1)
        self.assertAlmostEqual(fit(design)['theta'], 1.75, places=9)

    def test_explicit_response_fixed_effects_matches_centered_design(self):
        design = construct(rows_fixture())
        np.testing.assert_allclose(explicit_response_fit(design), fit(design)['coefficients'],
                                   atol=1e-10, rtol=1e-10)

    def test_whole_response_multiplicity_and_omission(self):
        design = construct(rows_fixture())
        for counts in (np.array([2, 0, 1, 1, 1, 0, 2, 1]), np.array([0, 1, 1, 1, 1, 1, 1, 1])):
            self.assertAlmostEqual(fit(design, counts)['theta'], 1.75, places=9)

    def test_full_baseline_identifies_exact_redundancy(self):
        rows = rows_fixture()
        for row in rows:
            row['u'] = row['r'] * row['h1'] / 14
        result = fit(construct(rows))
        self.assertFalse(result['identified'])
        self.assertIsNone(result['theta'])


if __name__ == '__main__':
    unittest.main()
