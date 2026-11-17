"""CPU-only pooled-count, common-bootstrap and fixed-decision checks."""
from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from experiments.unchanged_c_mathqa_transfer import analysis as A, evaluation as E

def scalar_rows():
    result=[]
    for ordinal in range(256):
        for repeat in range(3):
            for method in E.METHODS:
                calls=12+(ordinal%7)-{'N':0,'S':1,'C':2}[method]
                native=99-calls
                # A legal block-16 histogram whose native acceptance sums exactly.
                q,remainder=divmod(native,calls)
                histogram={str(q):calls-remainder}
                if remainder:histogram[str(q+1)]=remainder
                result.append(dict(phase='scored',event=6,scope=E.SCOPE,panel_ordinal=ordinal,
                    block_ordinal=ordinal//64,request_id=f'eval{ordinal}',source_id=f'mathqa:validation:{ordinal}',
                    method=method,repeat=repeat,checkpoint_id='/'+method,physical_gpu_uuid='GPU-fixed',
                    verification_calls=calls,native_accepted_draft_tokens=native,accepted_draft_tokens=native,
                    committed_progress=99,output_tokens=100,native_acceptance_histogram=histogram,
                    finish_reason='eos',cap_hit=False,validation_status='valid',valid_measurement=True,errors=[],
                    request_latency_s=float(calls+repeat),native_generate_s=calls+repeat-.1,cache_maintenance_s=.1))
    return result


class AnalysisTests(unittest.TestCase):
    def test_common_bootstrap_identity_and_pooled_counts(self):
        rows=scalar_rows();result=A.summarize(rows)
        paired=result['paired_requests']
        self.assertEqual([r['source_id'] for r in paired],sorted({r['source_id'] for r in rows}))
        expected=np.random.Generator(np.random.PCG64(1717)).integers(0,256,size=(10000,256))
        self.assertEqual(result['bootstrap']['draw_matrix_sha256'],hashlib.sha256(expected.astype('<i8').tobytes()).hexdigest())
        n=sum(r['verification_calls'] for r in rows if r['method']=='N' and r['repeat']==0)
        s=sum(r['verification_calls'] for r in rows if r['method']=='S' and r['repeat']==0)
        self.assertEqual(result['method_totals']['N']['verification_calls'],n)
        self.assertEqual(result['comparisons']['S_vs_N']['call_reduction']['point'],1-s/n)
        self.assertEqual(result['comparisons']['C_vs_N']['net_calls_saved'],512)
        self.assertEqual(result['decision']['outcome'],'TRANSFER_GATE_PASSED')

    def test_zero_call_requests_preserved_and_undefined_draws_explicit(self):
        rows=scalar_rows()
        for row in rows:
            row.update(verification_calls=0,native_accepted_draft_tokens=0,accepted_draft_tokens=0,
                committed_progress=0,output_tokens=1,native_acceptance_histogram={})
        result=A.summarize(rows)
        self.assertEqual(result['request_count'],256)
        self.assertEqual(result['method_totals']['S']['zero_call_requests'],256)
        effect=result['comparisons']['S_vs_N']
        for k in ('acceptance_improvement','acceptance_difference','call_reduction'):
            self.assertIsNone(effect[k]['point']);self.assertIsNone(effect[k]['ci95'])
            self.assertEqual(effect[k]['undefined_draws'],10000)
        self.assertFalse(result['decision']['checks']['acceptance_lower_bound_positive'])

    def test_invalid_rows_and_counts_cannot_enter_summary(self):
        rows=scalar_rows();rows[0]['valid_measurement']=False
        with self.assertRaisesRegex(ValueError,'Only valid'):A.summarize(rows)
        rows=scalar_rows();rows[2]['verification_calls']+=1
        with self.assertRaisesRegex(ValueError,'inconsistent'):A.summarize(rows)

    def test_cap_requests_are_retained(self):
        rows=scalar_rows()
        for row in rows:
            if row['panel_ordinal']==0:
                row.update(verification_calls=1024,native_accepted_draft_tokens=1023,accepted_draft_tokens=1023,
                    committed_progress=2047,output_tokens=2048,native_acceptance_histogram={'0':1,'1':1023},
                    finish_reason='length',cap_hit=True)
        result=A.summarize(rows)
        self.assertEqual(result['method_totals']['N']['cap_hit_requests'],1)
        self.assertEqual(result['request_count'],256)

    def test_decision_categories_and_strict_comparator_gate(self):
        def comparison(time_point=.006, time_ci=(.001,.012), acceptance_ci=(.001,.02),
                       repeats=(.01,.01,.01), scalar_time=.001, scalar_acceptance=.001):
            return dict(C_vs_N=dict(time_reduction={'point':time_point,'ci95':time_ci},
                acceptance_improvement={'point':.01,'ci95':acceptance_ci},
                repeat_time_reductions=repeats), C_vs_S=dict(
                time_reduction={'point':scalar_time,'ci95':[-.01,.01]},
                acceptance_improvement={'point':scalar_acceptance,'ci95':[-.01,.01]}))
        cases = [
            (comparison(time_point=.005), 'TRANSFER_GATE_PASSED'),
            (comparison(scalar_time=0), 'NATIVE_GAIN_BUT_COMPARATOR_GATE_FAILED'),
            (comparison(scalar_acceptance=0), 'NATIVE_GAIN_BUT_COMPARATOR_GATE_FAILED'),
            (comparison(scalar_time=-.001), 'NATIVE_GAIN_BUT_COMPARATOR_GATE_FAILED'),
            (comparison(scalar_acceptance=None), 'NATIVE_GAIN_BUT_COMPARATOR_GATE_FAILED'),
            (comparison(time_point=.001,time_ci=(-.001,.004),acceptance_ci=(-.02,-.001)), 'CLEAR_HARM_ON_THIS_COMPARISON'),
            (comparison(time_point=-.01,time_ci=(-.02,-.001)), 'CLEAR_HARM_ON_THIS_COMPARISON'),
            (comparison(time_point=.001,time_ci=(-.001,.004),acceptance_ci=(-.01,.01)), 'PRACTICAL_TIME_GAIN_NOT_SUPPORTED'),
            (comparison(time_point=.004), 'UNRESOLVED_OR_MIXED'),
            (comparison(repeats=(.01,0,.01)), 'UNRESOLVED_OR_MIXED'),
            (comparison(repeats=(.01,-.001,.01)), 'UNRESOLVED_OR_MIXED'),
            (comparison(acceptance_ci=None), 'UNRESOLVED_OR_MIXED'),
            (comparison(time_point=.004,time_ci=(.001,.005)), 'UNRESOLVED_OR_MIXED'),
        ]
        for value, expected in cases:
            with self.subTest(expected=expected,value=value):
                self.assertEqual(A.decision(value)['outcome'], expected)
        self.assertFalse(A.decision(comparison())['automatic_followup_authorized'])

    def test_all_comparisons_share_identical_draw_object_and_request_pairs(self):
        original = A.compare
        with patch.object(A, 'compare', wraps=original) as spy:
            result = A.summarize(list(reversed(scalar_rows())))
        calls = spy.call_args_list
        self.assertEqual([(call.args[1],call.args[2]) for call in calls], [('C','N'),('C','S'),('S','N')])
        self.assertTrue(all(call.args[0] is calls[0].args[0] and call.args[3] is calls[0].args[3] for call in calls))
        self.assertEqual(set(result['comparisons']), {'C_vs_N','C_vs_S','S_vs_N'})
        self.assertTrue(result['bootstrap']['common_draws_all_comparisons'])
        self.assertEqual(result['primary_comparison'], 'C_vs_N')

    def test_pooled_effect_uses_summed_means_not_mean_request_percentages(self):
        rows = scalar_rows()
        for row in rows:
            if row['method'] == 'C':
                row['request_latency_s'] *= 0.1 if row['panel_ordinal'] == 0 else 2
        result = A.summarize(rows)
        expected = 1 - result['method_totals']['C']['request_latency_s'] / result['method_totals']['N']['request_latency_s']
        self.assertEqual(result['comparisons']['C_vs_N']['time_reduction']['point'], expected)



if __name__=='__main__':unittest.main()
