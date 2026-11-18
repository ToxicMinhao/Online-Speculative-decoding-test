"""Paired ratio arithmetic, distinct dataset draws, and three-way evidence."""
import hashlib
import unittest

import numpy as np

from experiments.c_paired_confirmation import analysis as A, evaluation as E


def rows(dataset):
    result=[];contract=E.dataset_contract(dataset)
    for ordinal in range(256):
        for repeat in range(3):
            for method in E.METHODS:
                calls=12+ordinal%7-(method=='C');native=99-calls;q,remainder=divmod(native,calls)
                histogram={str(q):calls-remainder}
                if remainder:histogram[str(q+1)]=remainder
                result.append(dict(phase='scored',**contract,panel_ordinal=ordinal,block_ordinal=ordinal//64,
                    request_id=f'eval{ordinal}',source_id=f'{dataset}:test:{ordinal}',method=method,repeat=repeat,
                    checkpoint_id='/'+method,physical_gpu_uuid='GPU-fixed',verification_calls=calls,
                    native_accepted_draft_tokens=native,accepted_draft_tokens=native,committed_progress=99,output_tokens=100,
                    native_acceptance_histogram=histogram,finish_reason='eos',cap_hit=False,validation_status='valid',
                    valid_measurement=True,errors=[],request_latency_s=float(calls+repeat),native_generate_s=calls+repeat-.1,cache_maintenance_s=.1))
    return result


class AnalysisTests(unittest.TestCase):
    def test_exact_dataset_draw_seeds_and_pooled_totals(self):
        hashes=[]
        for dataset,seed in [('gsm8k',1729),('mathqa',1730)]:
            data=rows(dataset);result=A.summarize(data,dataset=dataset)
            expected=np.random.Generator(np.random.PCG64(seed)).integers(0,256,size=(10000,256))
            self.assertEqual(result['bootstrap']['draw_matrix_sha256'],hashlib.sha256(expected.astype('<i8').tobytes()).hexdigest())
            self.assertEqual(result['comparisons']['C_vs_N']['net_calls_saved'],256)
            self.assertEqual(result['method_totals']['N']['verification_calls'],sum(r['verification_calls'] for r in data if r['method']=='N' and r['repeat']==0))
            self.assertEqual(result['evidence']['acceptance_improvement'],'SUPPORTED_GAIN')
            self.assertEqual([x['source_id'] for x in result['paired_requests']],sorted({r['source_id'] for r in data}))
            self.assertEqual(set(result['comparisons']),{'C_vs_N'});self.assertNotIn('decision',result)
            hashes.append(result['bootstrap']['draw_matrix_sha256'])
        self.assertNotEqual(*hashes)

    def test_three_way_labels_no_minimum_gain_or_repeat_gate(self):
        cases=[([1e-12,2e-12],'SUPPORTED_GAIN'),([-2e-12,-1e-12],'SUPPORTED_LOSS'),
            ([0,.2],'DIRECTION_UNRESOLVED'),([-.2,0],'DIRECTION_UNRESOLVED'),([-.1,.1],'DIRECTION_UNRESOLVED'),
            (None,'DIRECTION_UNRESOLVED')]
        for interval,expected in cases:
            with self.subTest(interval=interval):self.assertEqual(A.evidence_label({'ci95':interval}),expected)

    def test_undefined_zero_call_draws_preserved(self):
        data=rows('mathqa')
        for row in data:row.update(verification_calls=0,native_accepted_draft_tokens=0,accepted_draft_tokens=0,
            committed_progress=0,output_tokens=1,native_acceptance_histogram={})
        result=A.summarize(data,dataset='mathqa')
        for key in ('acceptance_improvement','acceptance_difference','call_reduction'):
            effect=result['comparisons']['C_vs_N'][key]
            self.assertEqual(effect,dict(point=None,ci95=None,undefined_draws=10000))
            self.assertEqual(result['evidence'][key],'DIRECTION_UNRESOLVED')
        self.assertEqual(result['method_totals']['N']['zero_call_requests'],256)

    def test_no_missing_mixed_or_invalid_pairs(self):
        data=rows('gsm8k');data[0]['valid_measurement']=False
        with self.assertRaisesRegex(ValueError,'Only valid'):A.summarize(data,dataset='gsm8k')
        with self.assertRaisesRegex(ValueError,'Only valid'):A.summarize(rows('gsm8k'),dataset='mathqa')
        with self.assertRaisesRegex(ValueError,'exactly three'):A.summarize(rows('gsm8k')[1:],dataset='gsm8k')
        data=rows('gsm8k');data[0]['verification_calls']+=1
        with self.assertRaisesRegex(ValueError,'inconsistent'):A.summarize(data,dataset='gsm8k')

    def test_caps_preserved_and_pooled_latency_not_percentage_average(self):
        data=rows('gsm8k')
        for row in data:
            if row['panel_ordinal']==0:
                row.update(verification_calls=1024,native_accepted_draft_tokens=1023,accepted_draft_tokens=1023,
                    committed_progress=2047,output_tokens=2048,native_acceptance_histogram={'0':1,'1':1023},finish_reason='length',cap_hit=True)
            if row['method']=='C':row['request_latency_s']*=.1 if row['panel_ordinal']==0 else 2
        result=A.summarize(data,dataset='gsm8k')
        self.assertEqual(result['method_totals']['C']['cap_hit_requests'],1)
        expected=1-result['method_totals']['C']['request_latency_s']/result['method_totals']['N']['request_latency_s']
        self.assertEqual(result['comparisons']['C_vs_N']['time_reduction']['point'],expected)
        self.assertEqual(result['evidence']['time_reduction'],'SUPPORTED_LOSS')


if __name__=='__main__':unittest.main()
