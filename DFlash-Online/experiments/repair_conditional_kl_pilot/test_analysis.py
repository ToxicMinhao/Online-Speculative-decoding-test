"""Independent scalar reference and strict repeated-count/identity fixtures."""
from copy import deepcopy
import hashlib

import numpy as np
import unittest

from .analysis import reduce_requests, summarize


def fixture():
    rows = []
    for request_id, base_calls, pair_calls, base_times, pair_times in (
            ('b', 0, 0, [1., 2., 3.], [2., 2., 2.]),
            ('a', 10, 8, [2., 4., 6.], [1., 2., 3.])):
        output = 1 if base_calls == 0 else 41
        for method, calls, times in (('BASE', base_calls, base_times), ('PAIR', pair_calls, pair_times)):
            for repeat in range(3):
                rows.append(dict(request_id=request_id, source_id=request_id, method=method,
                    phase='scored', repeat=repeat, valid_measurement=True, errors=[],
                    checkpoint_id=method, physical_gpu_uuid='GPUa', finish_reason='eos',
                    output_tokens=output, verification_calls=calls,
                    native_accepted_draft_tokens=output - 1 - calls,
                    accepted_draft_tokens=output - 1 - calls, committed_progress=output - 1,
                    request_latency_s=times[repeat], native_generate_s=times[repeat] * .9,
                    cache_maintenance_s=.01))
    return rows


class AnalysisTests(unittest.TestCase):
    def test_primary_averages_repeats_then_uses_paired_ratio_of_sums(self):
        report = summarize(fixture(), expected_requests=2)
        result = report['contrasts']['PAIR_vs_BASE']
        self.assertEqual(report['bootstrap']['input_order'], ['a', 'b'])
        self.assertEqual(report['bootstrap']['seed'], 20261124)
        self.assertEqual(report['totals']['BASE']['verification_calls'], 10)
        self.assertEqual(report['totals']['PAIR']['verification_calls'], 8)
        self.assertEqual(report['totals']['BASE']['request_latency_s'], 6)
        self.assertEqual(report['totals']['PAIR']['request_latency_s'], 4)
        self.assertAlmostEqual(result['request_latency_s_reduction_pct']['value'], 100 / 3)
        draws = np.random.Generator(np.random.PCG64(20261124)).integers(0, 2, size=(10000, 2))
        current, reference = np.array([2., 2.]), np.array([4., 2.])
        expected = np.percentile(100 * (1 - current[draws].sum(1) / reference[draws].sum(1)),
                                 [2.5, 97.5], method='linear')
        self.assertEqual(result['request_latency_s_reduction_pct']['ci95'], expected.tolist())
        self.assertEqual(report['bootstrap']['draw_sha256'], hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest())
        self.assertEqual((result['wins'], result['regressions'], result['ties'], result['net_calls_saved']), (1, 0, 1, 2))

    def test_incomplete_inconsistent_or_invalid_evidence_is_never_averaged(self):
        for mutation in ('missing', 'duplicate', 'invalid', 'count', 'gpu', 'endpoint',
                         'source', 'method', 'nonfinite', 'negative', 'output', 'native_time'):
            rows = deepcopy(fixture())
            if mutation == 'missing':
                rows.pop()
            elif mutation == 'duplicate':
                rows.append(deepcopy(rows[-1]))
            elif mutation == 'invalid':
                rows[0]['valid_measurement'] = False
            elif mutation == 'count':
                rows[-1]['verification_calls'] += 1
            elif mutation == 'gpu':
                rows[0]['physical_gpu_uuid'] = 'GPUb'
            elif mutation == 'endpoint':
                for row in rows:
                    if row['request_id'] == 'a' and row['method'] == 'PAIR':
                        row['checkpoint_id'] = 'another-export'
            elif mutation == 'source':
                for row in rows:
                    if row['method'] == 'PAIR':
                        row['source_id'] += '-other'
            elif mutation == 'method':
                rows[0]['method'] = 'F'
            elif mutation == 'nonfinite':
                rows[0]['request_latency_s'] = float('nan')
            elif mutation == 'negative':
                rows[0]['cache_maintenance_s'] = -1.
            elif mutation == 'output':
                rows[0]['output_tokens'] = 0
            else:
                rows[0]['native_generate_s'] = 2 * rows[0]['request_latency_s']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                reduce_requests(rows, expected_requests=2)


if __name__ == '__main__':
    unittest.main()
