"""Paired three-arm latency inference and failure-visible scalar accounting."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mathqa_transfer import statistics as S


def rows():
    result = []
    for ordinal in range(1024):
        for repeat in range(3):
            for method in S.METHODS:
                calls = 10 + ordinal % 7 + (method == 'KPLUS')
                multiplier = dict(O=1.1, K=1., KPLUS=.9)[method]
                latency = multiplier * (2 + ordinal % 11) + .1 * repeat
                result.append(dict(seed=17, event=58, phase='final', panel_ordinal=ordinal,
                    request_id=f'mathqa:{ordinal}', repeat=repeat, method=method,
                    verification_calls=calls, accepted_draft_tokens=50, committed_progress=99,
                    output_tokens=100, checkpoint_id='/export/' + method,
                    finish_reason='length' if ordinal == 0 else 'eos', cap_hit=ordinal == 0,
                    valid_measurement=True, validation_status='valid', errors=[],
                    request_latency_s=latency, native_generate_s=latency-.05,
                    cache_maintenance_s=.02))
    return result


class StatisticsTests(unittest.TestCase):
    def test_same_paired_bootstrap_draws_and_three_repeat_means(self):
        values = rows()
        with patch.object(S, 'RESAMPLES', 80):
            result = S.summarize(values)
        basis = 2 + np.arange(1024) % 11
        indices = np.random.Generator(np.random.PCG64(20261103)).integers(0, 1024, (80, 1024))
        for reference, scale in [('O', 1.1), ('K', 1.)]:
            draws = 100 * (1 - (.9*basis+.1)[indices].sum(1)/(scale*basis+.1)[indices].sum(1))
            expected = np.quantile(draws, [.025, .975], method='linear').tolist()
            actual = result['comparisons']['KPLUS_vs_' + reference]['request_latency_s_reduction']
            np.testing.assert_allclose(actual['ci95_pct'], expected, rtol=1e-13)
        self.assertEqual(result['bootstrap']['seed'], 20261103)
        self.assertEqual(result['method_totals']['K']['verification_calls'], int((10+np.arange(1024)%7).sum()))
        self.assertAlmostEqual(result['method_totals']['K']['request_latency_s'], float(basis.sum()+102.4))
        self.assertAlmostEqual(result['method_totals']['K']['mean_request_ms'], (basis.mean()+.1)*1000)
        self.assertEqual(result['method_totals']['K']['cap_hits'], 1)
        self.assertEqual(result['method_totals']['K']['eos_responses'], 1023)
        self.assertTrue(result['interpretation']['primary']['passed'])
        # Calls intentionally worsen: latency remains the prespecified primary.
        self.assertLess(result['comparisons']['KPLUS_vs_K']['call_reduction']['point_pct'], 0)

    def test_fixed_order_secondary_descriptive_when_primary_does_not_pass(self):
        comparisons = {key: {'request_latency_s_reduction': {'point_pct': 2., 'ci95_pct': interval}}
            for key, interval in [('KPLUS_vs_O', [0., 2.]), ('KPLUS_vs_K', [1., 3.])]}
        result = S.ordered_interpretation(comparisons)
        self.assertFalse(result['primary']['passed'])
        self.assertEqual(result['secondary']['role'], 'descriptive')
        self.assertIsNone(result['secondary']['passed'])
        self.assertTrue(result['secondary']['interval_above_zero'])
        comparisons['KPLUS_vs_O']['request_latency_s_reduction']['ci95_pct'] = [.001, 2.]
        result = S.ordered_interpretation(comparisons)
        self.assertEqual(result['secondary']['role'], 'confirmatory')
        self.assertTrue(result['secondary']['passed'])
        self.assertFalse(result['automatic_followup_authorized'])

    def test_missing_or_duplicate_execution_cannot_form_subset(self):
        values = rows()
        for changed in [values[:-1], values+[values[0]], values[:-9]]:
            with self.assertRaises(ValueError):
                S.reduce_final(changed)
        values[-1]['repeat'] = 0
        with self.assertRaises(ValueError):
            S.reduce_final(values)

    def test_unvalidated_rows_and_control_rows_rejected(self):
        for field, value in [('validation_status', 'pending_canonical'), ('valid_measurement', False),
                             ('errors', ['output_mismatch']), ('phase', 'control')]:
            values = rows()
            values[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                S.reduce_final(values)

    def test_repeated_counts_checkpoint_and_stop_must_be_stable(self):
        for field, value in [('verification_calls', 88), ('checkpoint_id', '/later'),
                             ('finish_reason', 'eos'), ('cap_hit', False)]:
            values = rows()
            values[3][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                S.reduce_final(values)

    def test_nonfinite_and_impossible_accounting_rejected_even_if_marked_valid(self):
        for field, value in [('request_latency_s', float('nan')), ('native_generate_s', float('inf')),
            ('cache_maintenance_s', -1.), ('request_latency_s', 0.),
            ('accepted_draft_tokens', 101), ('committed_progress', 101), ('verification_calls', 0)]:
            values = rows()
            for row in values[:9]:
                if row['method'] == 'O':
                    row[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                S.reduce_final(values)

    def test_sealed_identity_and_cross_method_outputs_match(self):
        values = rows()
        requests = [dict(panel_ordinal=i, request_id=f'mathqa:{i}') for i in range(1024)]
        self.assertEqual(len(S.reduce_final(values, requests)), 1024)
        requests[0]['request_id'] = 'wrong-source'
        with self.assertRaises(ValueError):
            S.reduce_final(values, requests)
        for row in values[:9]:
            if row['method'] == 'KPLUS':
                row['output_tokens'] = 101
        with self.assertRaises(ValueError):
            S.reduce_final(values)

    def test_zero_cache_time_is_allowed_and_component_ratio_undefined(self):
        values = rows()
        for row in values:
            row['cache_maintenance_s'] = 0.
        with patch.object(S, 'RESAMPLES', 10):
            result = S.summarize(values)
        cache = result['comparisons']['KPLUS_vs_O']['cache_maintenance_s_reduction']
        self.assertIsNone(cache['point_pct'])
        self.assertIsNone(cache['ci95_pct'])
        self.assertEqual(cache['undefined_bootstrap_draws'], 10)
        self.assertTrue(result['interpretation']['primary']['passed'])


if __name__ == '__main__':
    unittest.main()
