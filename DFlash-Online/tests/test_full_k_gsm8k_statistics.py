"""Count/timing pairing, ratio-of-sums and fixed bootstrap CPU checks."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from experiments.full_k_gsm8k import statistics as S


def rows():
    result = []
    calls = {'F':[10,100], 'O':[10,100], 'K':[11,85], 'K_HALF':[9,80]}
    for ordinal in range(2):
        for repeat in range(3):
            for method in S.METHODS:
                result.append(dict(seed=17,event=58,phase='final',panel_ordinal=ordinal,
                    request_id=f'request{ordinal}',repeat=repeat,method=method,
                    verification_calls=calls[method][ordinal],accepted_draft_tokens=90,
                    committed_progress=199,output_tokens=200,checkpoint_id='/'+method,
                    finish_reason='eos',cap_hit=False,validation_status='valid',valid_measurement=True,errors=[],
                    request_latency_s=calls[method][ordinal]+repeat,
                    native_generate_s=calls[method][ordinal]-.1+repeat,cache_maintenance_s=0.))
    return result


class StatisticsTests(unittest.TestCase):
    def test_counts_once_timing_means_and_ratio_of_sums(self):
        with patch.object(S,'RESAMPLES',100):
            summary = S.summarize(rows())
        effect = summary['comparisons']['K_HALF_vs_O']
        self.assertAlmostEqual(effect['call_reduction']['point_pct'],100*(1-89/110))
        self.assertEqual(summary['method_totals']['O']['verification_calls'],110)
        self.assertEqual(summary['method_totals']['O']['request_latency_s'],112)
        self.assertAlmostEqual(effect['request_latency_s_reduction']['point_pct'],100*(1-91/112))
        self.assertEqual((effect['request_wins'],effect['request_regressions'],effect['request_ties']), (2,0,0))
        self.assertEqual(effect['request_denominator'],2)
        self.assertEqual(effect['cache_maintenance_s_reduction']['point_pct'],None)
        self.assertEqual(effect['cache_maintenance_s_reduction']['undefined_bootstrap_draws'],100)

    def test_paired_bootstrap_exact_pcg64_and_seed(self):
        summary = S.summarize(rows())
        indices = np.random.Generator(np.random.PCG64(20261102)).integers(0,2,(10000,2))
        c,b=np.array([9,80]),np.array([10,100])
        expected=np.quantile(100*(1-c[indices].sum(1)/b[indices].sum(1)),[.025,.975],method='linear')
        self.assertEqual(summary['comparisons']['K_HALF_vs_O']['call_reduction']['ci95_pct'],expected.tolist())
        self.assertEqual(summary['bootstrap']['resamples'],10000)

    def test_invalid_missing_and_inconsistent_repeats_rejected(self):
        bad=rows();bad[0]['valid_measurement']=False
        with self.assertRaises(ValueError):S.reduce_final(bad)
        with self.assertRaises(ValueError):S.reduce_final(rows()[:-1])
        bad=rows();bad[-1]['verification_calls']+=1
        with self.assertRaises(ValueError):S.reduce_final(bad)

    def test_no_old_priority_or_equivalence_rule_added(self):
        effect={'call_reduction':{'point_pct':.1,'ci95_pct':[.01,.2]}}
        self.assertTrue(S.interpret(effect)['supported_endpoint_gain'])
        effect['call_reduction']['ci95_pct']=[-.1,.2]
        self.assertEqual(S.interpret(effect)['endpoint_label'],'inconclusive endpoint difference')
        self.assertFalse(S.interpret(effect)['automatic_followup_authorized'])

    def test_intermediate_is_count_only_against_fixed_F(self):
        selected=[dict(row,phase='intermediate',event=14) for row in rows()
                  if row['method']=='K' and row['repeat']==0]
        references={f'request{i}':{'F_counts':{'verification_calls':v}} for i,v in enumerate([10,100])}
        summary=S.intermediate_summary(selected,references)
        self.assertEqual(summary['verification_calls'],96)
        self.assertEqual((summary['request_wins'],summary['request_regressions']),(1,1))
        self.assertFalse(any('latency' in key for key in summary))
        self.assertAlmostEqual(summary['call_reduction_vs_F_pct'],100*(1-96/110))


if __name__ == '__main__':
    unittest.main()
