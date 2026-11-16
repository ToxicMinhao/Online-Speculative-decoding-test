from copy import deepcopy
import unittest
from experiments.pr49_collection_cpu import independent as A

class MemoryReader:
    def __init__(self,capture):self.errors=[];self.checks=0;self.capture=capture
    def check(self,value,name):
        self.checks+=1
        if not value:self.errors.append(name)
    def read(self,path):return self.capture


def fixture(kind='clipped'):
    stops=[151645,151643];prompt=[7]
    if kind=='zero':response=[stops[0]];blocks=[]
    elif kind=='bonus':
        response=[8,9,10,stops[0]]
        blocks=[dict(output_start=0,anchor_token=8,draft_tokens=[9,10],target_argmax=[9,10,stops[0]],verify_size=3,committed_progress=3,committed_tokens=response[1:],positions=[1,2,3],raw_accepted_draft_tokens_before_stop=2,accepted_draft_tokens=2,native_accepted_draft_tokens=2,stop_clipped=False,finished=True)]
    else:
        response=[8,9,stops[0]]
        blocks=[dict(output_start=0,anchor_token=8,draft_tokens=[9,stops[0],4],target_argmax=[9,stops[0],4,5],verify_size=4,committed_progress=2,committed_tokens=response[1:],positions=[1,2,3,4],raw_accepted_draft_tokens_before_stop=3,accepted_draft_tokens=2,native_accepted_draft_tokens=1,stop_clipped=True,finished=True)]
    capture=dict(prompt_ids=prompt,response_ids=response,target_argmax_ids=response,covered_token_ids=(prompt+response)[:-1],covered_rows=len(prompt)+len(response)-1,blocks=blocks,collect_features=True,native_capture_validated=True,status='passed',label_binding_scope='native_prefill_and_committed_verification',block_size=16)
    calls=len(blocks);P=len(response)-1;matched=sum(b['accepted_draft_tokens'] for b in blocks)
    meta=dict(capture=capture,trace=blocks,block_size=16,context_limit=4096,runtime='pinned_SGLang_FA3',feedback_capture_in_timed_request=True,native_trace_path='memory',verification_calls=calls,committed_progress=P,accepted_draft_tokens=matched,committed_progress_per_call=P/calls if calls else None,calls_per_1000_output_tokens=1000*calls/len(response))
    request=dict(source_id='fixture',request_id='r',request_index=1,prompt_ids=prompt,stop_token_ids=stops,max_new_tokens=2048)
    record=dict(record_id='r',request_index=1,session_id='persistent_N_event8',checkpoint_id='N7',purpose='serving',prompt_ids=prompt,response_ids=response,stop_token_ids=stops,max_new_tokens=2048,finish_reason='eos',metadata=meta)
    return MemoryReader(capture),record,request


class AuditTests(unittest.TestCase):
    def test_terminal_clipping_keeps_native_and_matching_counts_distinct(self):
        reader,record,request=fixture()
        row,_=A.count_record(reader,record,request,'N','N7')
        self.assertFalse(reader.errors,reader.errors);self.assertEqual((row['V'],row['P'],row['A'],row['M']),(1,2,1,2))

    def test_full_match_bonus(self):
        reader,record,request=fixture('bonus');row,_=A.count_record(reader,record,request,'N','N7')
        self.assertFalse(reader.errors,reader.errors);self.assertEqual((row['V'],row['P'],row['A'],row['M']),(1,3,2,2))

    def test_first_token_zero_calls_stays_valid(self):
        reader,record,request=fixture('zero');row,_=A.count_record(reader,record,request,'N','N7')
        self.assertFalse(reader.errors,reader.errors);self.assertEqual((row['V'],row['P'],row['A'],row['M']),(0,0,0,0))
        self.assertTrue(row['zero_call'])

    def test_float_count_rejected_even_if_numerically_equal(self):
        reader,record,request=fixture();record['metadata']['trace'][0]['native_accepted_draft_tokens']=1.0
        row,_=A.count_record(reader,record,request,'N','N7');self.assertFalse(row['valid']);self.assertTrue(reader.errors)

    def test_undefined_bootstrap_denominators_are_not_imputed(self):
        reader,record,request=fixture('zero');row,_=A.count_record(reader,record,request,'N','N7')
        rows=[]
        for i in range(128):
            for arm in A.ARMS:rows.append(dict(row,source_id=f's{i:03}',request_id=f'r{i:03}',arm=arm))
        result,paired=A.compute(rows)
        self.assertEqual(len(paired),128)
        for contrast in result['contrasts'].values():
            for key in ('absolute_acceptance','relative_acceptance_pct','call_reduction_pct'):
                self.assertIsNone(contrast[key]['point']);self.assertIsNone(contrast[key]['ci95']);self.assertEqual(contrast[key]['undefined_draws'],10000)

if __name__=='__main__':unittest.main()
