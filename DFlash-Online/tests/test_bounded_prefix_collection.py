import copy
import unittest

from test_pr49_collection_cpu import fixture
from experiments.bounded_prefix_allocation import collection_analysis as module


def inputs(kind='bonus'):
    requests=[];records={a:[] for a in module.ARMS};captures={a:{} for a in module.ARMS}
    checkpoints={a:dict(export=a+'7',weights_identity=dict(bytes=100,sha256={'N':'a','C':'b','S':'c'}[a]*64)) for a in module.ARMS}
    for i in range(128):
        original,request,_=fixture(kind)
        request.update(request_id=f'r{i:03d}',source_id=f's{i:03d}',request_index=897+i,prompt_ids=[7,1000+i])
        requests.append(request)
        for arm in module.ARMS:
            record=copy.deepcopy(original)
            record.update(record_id=request['request_id'],request_index=request['request_index'],
                prompt_ids=request['prompt_ids'],session_id=module.SESSIONS[arm],checkpoint_id=arm+'7')
            capture=record['metadata']['capture']
            capture.update(prompt_ids=request['prompt_ids'],covered_token_ids=(request['prompt_ids']+record['response_ids'])[:-1])
            records[arm].append(record);captures[arm][request['request_id']]=copy.deepcopy(capture)
    provenance=dict(status='passed',checkpoints=copy.deepcopy(checkpoints),consumed_inputs=[dict(path='authenticated_fixture',sha256='a'*64)])
    return requests,records,captures,checkpoints,provenance


class SecondaryTests(unittest.TestCase):
    def test_complete_ties_keep128_and_shared_bootstrap_no_latency(self):
        data=inputs()
        for records in data[1].values():
            for r in records:r['metadata']['request_latency_s']=float('nan')
        result=module.analyze(*data)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['totals']['C']['V'],128)
        self.assertEqual(result['totals']['C']['A'],256)
        self.assertEqual(result['totals']['C']['P'],384)
        self.assertEqual(result['comparisons']['C_vs_N']['call_reduction']['point'],0)
        self.assertEqual(result['comparisons']['C_vs_N']['request_call_ties'],128)
        self.assertTrue(result['latency_excluded'])
        self.assertFalse(result['primary_endpoint_decision_affected'])
        self.assertEqual(result['bootstrap']['draws'],10000)
        self.assertNotIn('latency',repr(result['totals']))

    def test_native_count_improvement_direction_and_common_intervals(self):
        data=inputs()
        for arm in ('N','S'):
            for record in data[1][arm]:
                first=copy.deepcopy(record['metadata']['trace'][0])
                first.update(draft_tokens=[20,888],target_argmax=[20,21,99],committed_progress=2,
                    committed_tokens=[20,21],raw_accepted_draft_tokens_before_stop=1,
                    accepted_draft_tokens=1,native_accepted_draft_tokens=1,stop_clipped=False,finished=False)
                second=copy.deepcopy(first)
                second.update(output_start=2,anchor_token=21,draft_tokens=[888,777],target_argmax=[99,777,666],
                    committed_progress=1,committed_tokens=[99],positions=[4,5,6],
                    raw_accepted_draft_tokens_before_stop=0,accepted_draft_tokens=0,native_accepted_draft_tokens=0,finished=True)
                trace=[first,second]
                meta=record['metadata'];meta.update(trace=trace,verification_calls=2,accepted_draft_tokens=1,
                    committed_progress_per_call=1.5,calls_per_1000_output_tokens=500)
                meta['capture']['blocks']=trace
                data[2][arm][record['record_id']]=copy.deepcopy(meta['capture'])
        result=module.analyze(*data)
        self.assertEqual(result['status'],'completed')
        effect=result['comparisons']['C_vs_N']
        self.assertEqual(effect['call_reduction']['point'],.5)
        self.assertEqual(effect['call_reduction']['interval'],[.5,.5])
        self.assertEqual(effect['acceptance_improvement']['point'],3.)
        self.assertEqual(effect['gross_calls_saved'],128)
        self.assertEqual(effect['gross_calls_added'],0)
        self.assertEqual(effect['requests_fewer_calls'],128)

    def test_matching_eos_is_not_native_accepted_bonus(self):
        result=module.analyze(*inputs('clipped'))
        self.assertEqual(result['status'],'completed')
        self.assertEqual((result['totals']['C']['V'],result['totals']['C']['A'],result['totals']['C']['M']),(128,0,128))
        self.assertIsNone(result['comparisons']['C_vs_N']['acceptance_improvement']['point'])

    def test_extra_missing_and_duplicate_records_are_invalid_not_subsets(self):
        data=inputs();data[1]['C'][-1]=copy.deepcopy(data[1]['C'][0])
        result=module.analyze(*data)
        self.assertEqual(result['status'],'INVALID_OR_INCOMPLETE_SECONDARY')
        self.assertNotIn('comparisons',result)
        self.assertFalse(result['primary_endpoint_decision_affected'])
        self.assertLess(result['validation']['valid_pairs'],128)

    def test_wrong_actual_c_session_is_not_silently_normalized(self):
        data=inputs();data[1]['C'][0]['session_id']='persistent_C_event8'
        result=module.analyze(*data)
        self.assertEqual(result['status'],'INVALID_OR_INCOMPLETE_SECONDARY')
        self.assertTrue(any(x.endswith(':actual_session') for x in result['validation']['errors']))

    def test_capture_discrepancy_is_preserved(self):
        data=inputs();data[2]['C']['r000']['covered_rows']+=1
        result=module.analyze(*data)
        self.assertNotIn('comparisons',result)
        self.assertTrue(any('independent_json_capture_equality' in x for x in result['validation']['errors']))

    def test_wrong_checkpoint_or_provenance_is_invalid(self):
        data=inputs();data[1]['C'][0]['checkpoint_id']='C8'
        result=module.analyze(*data)
        self.assertNotIn('comparisons',result)
        data=inputs();data[4]['status']='incomplete'
        self.assertNotIn('comparisons',module.analyze(*data))

    def test_zero_call_requests_and_undefined_draws_remain(self):
        result=module.analyze(*inputs('zero'))
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['totals']['C']['zero_calls'],128)
        effect=result['comparisons']['C_vs_N']['call_reduction']
        self.assertIsNone(effect['point']);self.assertIsNone(effect['interval'])
        self.assertEqual(effect['undefined_draws'],10000)

    def test_valid_individual_outputs_but_cross_arm_mismatch_suppresses_comparison(self):
        data=inputs()
        other,request,capture=fixture('middle_error')
        original=data[1]['C'][0]
        other.update(record_id=original['record_id'],request_index=original['request_index'],
            session_id=original['session_id'],checkpoint_id='C7',prompt_ids=original['prompt_ids'])
        other['metadata']['capture'].update(prompt_ids=other['prompt_ids'],
            covered_token_ids=(other['prompt_ids']+other['response_ids'])[:-1])
        data[1]['C'][0]=other;data[2]['C']['r000']=copy.deepcopy(other['metadata']['capture'])
        result=module.analyze(*data)
        self.assertNotIn('comparisons',result)
        self.assertIn('r000:C:paired_response_ids',result['validation']['errors'])
        self.assertFalse(result['paired_requests'][0]['valid'])


if __name__=='__main__':unittest.main()
