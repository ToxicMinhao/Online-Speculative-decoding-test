"""Small native-trace fixtures; no models, Torch or backend imports."""
import copy
import argparse
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import numpy as np

from experiments.pr49_collection_cpu.reducer import (
    Checks, identity, interval, pairing_errors, pooled_statistics, population_errors, reduce, validate_record,
)


def fixture(kind="bonus"):
    prompt, stop, response = [7, 8], [99], [10, 20, 21, 99]
    draft, target, c, a = [20, 21], [20, 21, 99], 3, 2
    if kind == "first_error":
        response, target, c, a = [10, 99], [99, 21, 98], 1, 0
    elif kind == "middle_error":
        response, target, c, a = [10, 20, 99], [20, 99, 98], 2, 1
    elif kind == "clipped":
        stop, response, target, c = [20], [10, 20], [20, 21, 99], 1
    elif kind == "zero":
        response = [99]
    block = {"output_start":0, "anchor_token":10, "draft_tokens":draft,
             "target_argmax":target, "verify_size":3, "committed_progress":c,
             "committed_tokens":response[1:], "positions":[2,3,4],
             "raw_accepted_draft_tokens_before_stop":a, "accepted_draft_tokens":min(a,c),
             "native_accepted_draft_tokens":c-1, "stop_clipped":c<a+1, "finished":True}
    trace = [] if kind == "zero" else [block]
    capture = {"status":"passed", "prompt_ids":prompt, "response_ids":response,
               "target_argmax_ids":response, "blocks":trace,
               "covered_token_ids":(prompt+response)[:-1], "covered_rows":len(prompt)+len(response)-1,
               "prefill_chunks":1, "block_size":16, "native_capture_validated":True,
               "label_binding_scope":"native_prefill_and_committed_verification",
               "final_feature_logits_used_for_labels":False,
               "final_emitted_token_feature_present":False, "collect_features":True}
    record = {"record_id":"r", "request_index":897, "session_id":"persistent_N_event8",
              "checkpoint_id":"N7", "purpose":"serving", "prompt_ids":prompt,
              "response_ids":response, "finish_reason":"eos", "stop_token_ids":stop,
              "max_new_tokens":2048,
              "metadata":{"block_size":16, "context_limit":4096, "trace":trace, "capture":capture,
                          "feedback_capture_in_timed_request":True, "trace_instrumentation_in_timed_request":True,
                          "runtime":"pinned_SGLang_FA3", "cache_policy":"idle_full_flush_before_every_request_and_publication",
                          "verification_calls":len(trace), "accepted_draft_tokens":sum(b['accepted_draft_tokens'] for b in trace),
                          "committed_progress":len(response)-1,
                          "committed_progress_per_call":len(response)-1 if trace else None,
                          "calls_per_1000_output_tokens":1000*len(trace)/len(response)}}
    request = {"source_id":"gsm8k:main:train:896", "request_id":"r", "request_index":897,
               "prompt_ids":prompt, "stop_token_ids":stop}
    return record, request, copy.deepcopy(capture)


class CollectionFixtures(unittest.TestCase):
    def validate(self, kind):
        record, request, saved = fixture(kind)
        checks = Checks()
        result = validate_record(record, request, "N", "N7", saved, checks)
        self.assertEqual(checks.errors, [])
        return result

    def test_first_rejection_and_bonus(self):
        for kind, expected in [("first_error",(1,1,0,0)),("middle_error",(1,2,1,1)),("bonus",(1,3,2,2))]:
            with self.subTest(kind=kind):
                row=self.validate(kind)
                self.assertEqual(tuple(row[k] for k in ("V","P","A","M")),expected)

    def test_terminal_matching_and_native_are_different(self):
        row=self.validate("clipped")
        self.assertEqual((row['V'],row['P'],row['A'],row['M']), (1,1,0,1))
        self.assertEqual(row['stop_clipped_blocks'],1)

    def test_one_token_zero_call_retained(self):
        row=self.validate("zero")
        self.assertEqual((row['V'],row['P'],row['A'],row['M']), (0,0,0,0))
        self.assertIsNone(row['native_acceptance'])
        self.assertTrue(row['zero_calls'])

    def test_legitimate_length_cap(self):
        record,request,_=fixture()
        response=[10]+[20]*2047
        trace=[]
        progress=0
        while progress<2047:
            c=min(16,2047-progress)
            trace.append(dict(output_start=progress,anchor_token=response[progress],
                draft_tokens=[20]*15,target_argmax=[20]*16,verify_size=16,
                committed_progress=c,committed_tokens=[20]*c,
                positions=list(range(2+progress,2+progress+16)),
                raw_accepted_draft_tokens_before_stop=15,accepted_draft_tokens=min(15,c),
                native_accepted_draft_tokens=c-1,stop_clipped=c<16,finished=progress+c==2047))
            progress+=c
        record.update(response_ids=response,finish_reason='length')
        meta=record['metadata'];capture=meta['capture']
        capture.update(response_ids=response,target_argmax_ids=response,blocks=trace,
                       covered_token_ids=(request['prompt_ids']+response)[:-1],covered_rows=2049)
        meta.update(trace=trace,verification_calls=128,accepted_draft_tokens=1920,
                    committed_progress=2047,committed_progress_per_call=2047/128,
                    calls_per_1000_output_tokens=128000/2048)
        checks=Checks()
        row=validate_record(record,request,'N','N7',copy.deepcopy(capture),checks)
        self.assertEqual(checks.errors,[])
        self.assertTrue(row['cap_hit'])
        self.assertEqual((row['V'],row['P'],row['A'],row['M']),(128,2047,1919,1920))

    def test_duplicate_missing_and_extra_identity(self):
        requests=[dict(request_id='r1',source_id='s1'),dict(request_id='r2',source_id='s2')]
        ids={a:['r1','r2'] for a in 'NES'}
        self.assertEqual(population_errors(requests,ids,2),[])
        ids['E']=['r1','r1']
        self.assertIn('E:missing_duplicate_or_extra_records',population_errors(requests,ids,2))
        ids['E']=['r1']
        self.assertTrue(population_errors(requests,ids,2))
        ids['E']=['r1','r2','r3']
        self.assertTrue(population_errors(requests,ids,2))
        requests[1]['source_id']='s1'
        self.assertIn('sealed_source_duplicates',population_errors(requests,ids,2))

    def test_cross_arm_output_mismatch_is_rejected(self):
        record,_,_=fixture()
        arms={a:copy.deepcopy(record) for a in 'NES'}
        self.assertEqual(pairing_errors(arms),[])
        arms['E']['response_ids'][1]=300
        self.assertEqual(pairing_errors(arms),['E:paired_response_ids'])

    def test_persisted_json_and_counter_corruption(self):
        record,request,saved=fixture('clipped')
        record['metadata']['trace'][0]['native_accepted_draft_tokens']=1
        checks=Checks()
        result=validate_record(record,request,'N','N7',saved,checks)
        self.assertFalse(result['valid'])
        self.assertTrue(any('independent_json_capture_equality' in e for e in checks.errors))
        self.assertTrue(any('native_acceptance' in e for e in checks.errors))

    def test_booleans_are_not_count_integers(self):
        record,request,saved=fixture()
        record['metadata']['trace'][0]['verify_size']=True
        checks=Checks()
        result=validate_record(record,request,'N','N7',saved,checks)
        self.assertFalse(result['valid'])
        self.assertTrue(any('integer_geometry' in e for e in checks.errors))

    def test_bad_termination_is_preserved_as_invalid(self):
        record,request,saved=fixture()
        record['finish_reason']='length'
        checks=Checks()
        result=validate_record(record,request,'N','N7',saved,checks)
        self.assertFalse(result['valid'])
        self.assertTrue(any(e.endswith(':termination') for e in checks.errors))

    def test_undefined_draws_are_not_dropped(self):
        receipt=interval(np.array([1.,np.inf,2.]))
        self.assertIsNone(receipt['interval'])
        self.assertEqual(receipt['undefined_draws'],1)

    def test_missing_top_level_metadata_retains_identifiable_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prod=root/'prod';qual=root/'qual';out=root/'output'
            files={}
            def save(base,name,payload):
                path=base/name;path.parent.mkdir(parents=True,exist_ok=True)
                data=(json.dumps(payload)+'\n').encode();path.write_bytes(data)
                files.setdefault(base,{})[name]=identity(data)
            record,request,_=fixture()
            plan=dict(execution_commit='db6e1221d8d1b234d9d07e246025d1afd5dc30d5',root='/original',event8_inputs='/qualified/sealed/EVENT8_INPUTS.json')
            save(prod,'sealed/PLAN.json',plan)
            save(qual,'sealed/EVENT8_INPUTS.json',dict(requests=[request]))
            for arm in 'NES':
                row=copy.deepcopy(record);row['checkpoint_id']=arm+'7';row['session_id']=f'persistent_{arm}_event8'
                row['metadata']['native_trace_path']=f'/original/tasks/{arm}_collect/native/request_000001.json'
                save(prod,f'tasks/{arm}_collect/records/r.json',row)
                save(prod,f'tasks/{arm}_collect/native/request_000001.json',row['metadata']['capture'])
                if arm!='E':save(prod,f'tasks/{arm}_collect/RESULT.json',dict(status='completed',generated_responses=128,own_checkpoint=arm+'7'))
            for base,manifest in files.items():(base/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest))
            args=argparse.Namespace(delivery=str(prod),qualification_delivery=str(qual),provenance=str(root/'missing-provenance.json'),analysis_commit='fixture',output=str(out))
            with patch.dict(os.environ,dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')):
                summary,_=reduce(args)
            self.assertEqual(summary['status'],'incomplete')
            self.assertNotIn('totals',summary)
            validation=json.loads((out/'VALIDATION.json').read_text())
            self.assertEqual(validation['identifiable_rows'],3)
            self.assertTrue(any('E_collect/RESULT.json' in e for e in validation['errors']))
            self.assertTrue(any('provenance_unavailable' in e for e in validation['errors']))
            self.assertTrue((out/'PER_REQUEST.csv').exists())

    def test_pooled_not_mean_request_ratios_and_common_draws(self):
        rows=[]
        for arm,calls in [('N',[2,8]),('E',[1,8]),('S',[2,7])]:
            for i,v in enumerate(calls):
                rows.append(dict(arm=arm,source_id=f's{i}',request_id=f'r{i}',V=v,A=10-v,P=10,M=10-v,
                                 output_tokens=11,finish_reason='eos',cap_hit=False,zero_calls=False))
        x=pooled_statistics(rows,draws_count=20)
        self.assertEqual(x['totals']['N']['native_acceptance'],1)
        self.assertAlmostEqual(x['totals']['E']['native_acceptance'],11/9)
        self.assertEqual(x['comparisons']['E_vs_N']['net_calls_saved'],1)
        self.assertEqual(x['comparisons']['E_vs_N']['gross_saved'],1)
        self.assertEqual(x['comparisons']['E_vs_N']['gross_added'],0)
        self.assertEqual(x['bootstrap']['draw_matrix_sha256'],pooled_statistics(list(reversed(rows)),draws_count=20)['bootstrap']['draw_matrix_sha256'])


if __name__ == '__main__':
    unittest.main()
