"""Full-panel pairing, bootstrap and conservative endpoint decision."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from experiments.adaflash_c1 import statistics as S


def rows():
    result = []
    for i in range(1024):
        for repeat in range(3):
            for method in S.METHODS:
                calls = 10 + i % 7 - (method == 'KPLUS')
                result.append(dict(seed=17, event=58, phase='final', panel_ordinal=i,
                    request_id=f'eval{i}', repeat=repeat, method=method, verification_calls=calls,
                    accepted_draft_tokens=50, committed_progress=99, output_tokens=100,
                    checkpoint_id='/' + method, finish_reason='eos', cap_hit=False,
                    validation_status='valid', valid_measurement=True, errors=[],
                    request_latency_s=calls + repeat, native_generate_s=calls - .1 + repeat,
                    cache_maintenance_s=0.))
    return result


class StatisticsTests(unittest.TestCase):
    def test_complete_counts_once_timing_means_and_paired_bootstrap(self):
        values = rows()
        with patch.object(S, 'RESAMPLES', 100):
            summary = S.summarize(values)
        before = np.array([10 + i % 7 for i in range(1024)])
        indices = np.random.Generator(np.random.PCG64(20261102)).integers(0, 1024, (100, 1024))
        draws = 100 * (1 - (before - 1)[indices].sum(1) / before[indices].sum(1))
        effect = summary['comparisons']['KPLUS_vs_A']
        self.assertEqual(effect['call_reduction']['ci95_pct'], np.quantile(draws, [.025, .975], method='linear').tolist())
        self.assertEqual(summary['method_totals']['A']['verification_calls'], int(before.sum()))
        self.assertEqual(summary['method_totals']['A']['request_latency_s'], float(before.sum() + 1024))
        self.assertEqual(effect['gross_calls_saved'], 1024)

    def test_missing_request_or_duplicate_repeat_rejected(self):
        with self.assertRaises(ValueError):
            S.reduce_final(rows()[:-6])
        values = rows()
        values[-1]['repeat'] = 0
        with self.assertRaises(ValueError):
            S.reduce_final(values)

    def test_primary_is_latency_not_call_count(self):
        effect = dict(call_reduction=dict(ci95_pct=[-.1, 1.]),
                      request_latency_s_reduction=dict(ci95_pct=[.1, 1.]))
        self.assertEqual(S.interpret(effect)['endpoint_label'], 'favors_KPLUS')
        effect['request_latency_s_reduction']['ci95_pct'] = [-1., -.1]
        self.assertEqual(S.interpret(effect)['endpoint_label'], 'favors_A')
        effect['request_latency_s_reduction']['ci95_pct'] = [-.1, .2]
        self.assertEqual(S.interpret(effect)['endpoint_label'], 'inconclusive')
        self.assertFalse(S.interpret(effect)['automatic_followup_authorized'])


if __name__ == '__main__':
    unittest.main()
