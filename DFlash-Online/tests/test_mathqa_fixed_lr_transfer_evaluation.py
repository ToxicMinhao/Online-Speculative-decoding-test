"""CPU checks for the genuine two-arm schedule, raw validity and fixed decision."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mathqa_fixed_lr_transfer import analysis as A, evaluation as E
from test_full_k_gsm8k_evaluation import FullBackend, requests, harmful
from test_persistent_prefix_evaluation import references, quiet


def panel():
    result=requests(256)
    for i,row in enumerate(result):
        row['source_id']=f'mathqa:validation:{i}'
    return result


def origin_references(rows):
    result = {k:dict(v,reference_checkpoint='/KPLUS4') for k,v in references(rows).items()}
    for reference in result.values():
        payload = {k:v for k,v in reference.items() if k != 'binding_digest'}
        reference['binding_digest'] = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


def scalar_rows():
    result=[]
    for ordinal in range(256):
        for repeat in range(3):
            for method in E.METHODS:
                calls=10+(ordinal%7)-(method=='S')
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


class ScheduleTests(unittest.TestCase):
    def test_fixed_parity_budget_and_within_block_order(self):
        schedule=E.evaluation_schedule(panel())
        self.assertEqual(len(schedule),24)
        self.assertEqual(sum(len(r['request_ordinals']) for r in schedule),1536)
        self.assertEqual(2*len(schedule),48)
        self.assertEqual(Counter(r['method'] for r in schedule if r['position']==0),{'N':6,'S':6})
        for i in range(0,24,2):
            first,second=schedule[i:i+2]
            expected=('N','S') if (first['repeat']+first['block_ordinal'])%2==0 else ('S','N')
            self.assertEqual((first['method'],second['method']),expected)
            start=first['block_ordinal']*64
            self.assertEqual(first['request_ordinals'],list(range(start,start+64)))
            self.assertEqual(first['request_ordinals'],second['request_ordinals'])

    def test_reference_must_be_separate_origin_and_warmups_disjoint(self):
        inputs=panel();warmups=deepcopy(inputs[:2])
        for i,row in enumerate(warmups):row.update(request_id=f'warm{i}',source_id=f'warm:{i}',prompt_ids=[1,8000+i])
        refs=origin_references(inputs);warmrefs=origin_references(warmups)
        E.validate_inputs(inputs,refs,warmups,warmrefs,'/KPLUS4')
        refs['eval0']['reference_checkpoint']='/S'
        with self.assertRaisesRegex(ValueError,'KPLUS4'):
            E.validate_inputs(inputs,refs,warmups,warmrefs,'/KPLUS4')
        refs=origin_references(inputs);warmups[0]['source_id']=inputs[0]['source_id']
        with self.assertRaisesRegex(ValueError,'separate fixed warmups'):
            E.validate_inputs(inputs,refs,warmups,warmrefs,'/KPLUS4')

    def test_canonical_contents_must_match_binding_digest(self):
        inputs=panel();refs=origin_references(inputs)
        refs['eval0']['source_record_path']='/substituted.json'
        with self.assertRaisesRegex(ValueError,'binding digest'):
            E.validate_references(inputs,refs,origin_checkpoint='/KPLUS4')


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.inputs=panel();self.refs=origin_references(self.inputs)
        self.warmups=deepcopy(self.inputs[:2])
        for i,row in enumerate(self.warmups):row.update(request_id=f'warm{i}',source_id=f'warm:{i}',prompt_ids=[1,8000+i])
        self.warmrefs=origin_references(self.warmups)

    def run_final(self,backend,folder):
        return E.evaluate_final(backend,self.inputs,{'N':'/N6','S':'/S6'},self.root/folder,
            canonicals=self.refs,warmup_requests=self.warmups,warmup_canonicals=self.warmrefs,
            schedule=E.evaluation_schedule(self.inputs),origin_checkpoint='/KPLUS4',
            physical_gpu_uuid='GPU-fixed',boundary_check=quiet,execution_commit='c'*40)

    def test_complete_native_consumer_and_machine_tables(self):
        backend=FullBackend(self.inputs)
        status=self.run_final(backend,'complete')
        self.assertEqual((status['scored_generations'],status['warmup_generations']),(1536,48))
        self.assertEqual(len(backend.switches),24)
        self.assertEqual(backend.weight_checks,['/N6','/S6'])
        self.assertTrue(all(x[2]['session_id']==E.SESSION for x in backend.executions))
        root=self.root/'complete'
        result=A.write_report(root)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(set(result['method_totals']),{'N','S'})
        self.assertEqual(result['method_totals']['N']['verification_calls'],512)
        self.assertEqual(result['decision']['outcome'],'PRACTICAL_TIME_GAIN_NOT_SUPPORTED')
        self.assertEqual(len((root/'ALL_REPEATS.csv').read_text().splitlines()),1537)
        self.assertEqual(len((root/'WARMUPS.csv').read_text().splitlines()),49)
        self.assertEqual(len((root/'PER_REQUEST.csv').read_text().splitlines()),513)
        self.assertEqual(len((root/'TRACE_COUNTS.csv').read_text().splitlines()),3169)
        costs=json.loads((root/'EVALUATION_COSTS.json').read_text())
        self.assertEqual(costs['measured_time_by_phase']['scored']['request_latency_s'],1536)
        self.assertEqual(costs['measured_time_by_phase']['warmup']['request_latency_s'],48)
        protocol_path=root/'PROTOCOL.json'
        protocol=json.loads(protocol_path.read_text())
        protocol_path.write_text(json.dumps(dict(protocol,execution_commit='unverified')))
        self.assertIn('execution_identity',E.audit_raw_records(root)['errors'])
        protocol_path.write_text(json.dumps(protocol))
        # Independent native capture comparison rejects changed raw proposals.
        rawpath=root/'0000_r0_N.json';raw=json.loads(rawpath.read_text())
        raw['metadata']['trace'][0]['draft_tokens'][0]+=1
        rawpath.write_text(json.dumps(raw))
        self.assertEqual(A.write_report(root)['status'],'INVALID_OR_INCOMPLETE')

    def test_invalid_return_saved_before_failure(self):
        def mutation(record,options):record['response_ids'][0]+=1
        with self.assertRaisesRegex(RuntimeError,'correctness'):
            self.run_final(FullBackend(self.inputs,mutation),'invalid')
        root=self.root/'invalid';status=json.loads((root/'STATUS.json').read_text())
        self.assertEqual((status['invalid_generations'],status['unknown_generation_attempts'],status['unattempted_generations']),(1,0,1583))
        self.assertTrue((root/'warmup_a00_0_N.json').is_file())
        self.assertEqual(A.write_report(root)['status'],'INVALID_OR_INCOMPLETE')

    def test_unknown_attempt_not_retried_or_counted_unattempted(self):
        backend=FullBackend(self.inputs)
        with patch.object(backend,'generate',side_effect=RuntimeError('worker lost')):
            with self.assertRaisesRegex(RuntimeError,'worker lost'):self.run_final(backend,'unknown')
        status=json.loads((self.root/'unknown/STATUS.json').read_text())
        self.assertEqual((status['returned_generations'],status['unknown_generation_attempts'],status['unattempted_generations']),(0,1,1583))

    def test_repeat_count_mismatch_stops_even_with_identical_tokens(self):
        def mutation(record,options):
            if options['request_id']=='0000_r1_N':harmful(record)
        with self.assertRaisesRegex(RuntimeError,'repeat'):
            self.run_final(FullBackend(self.inputs,mutation),'repeat')
        rows=[json.loads(s) for s in (self.root/'repeat/MEASUREMENTS.jsonl').read_text().splitlines()]
        self.assertFalse(rows[-1]['valid_measurement'])
        self.assertEqual(rows[-1]['label'],'0000_r1_N')

    def test_zero_call_eos_valid_and_no_historical_count_gate(self):
        def mutation(record,options):
            record['response_ids']=[151645]
            record['metadata'].update(trace=[],verification_calls=0,accepted_draft_tokens=0,committed_progress=0)
        backend=FullBackend(self.inputs,mutation)
        slot=dict(phase='scored',method='N',repeat=0,event=6,panel_ordinal=0,activation=0,block_ordinal=0,position=0,request_id='eval0')
        record=backend.generate(self.inputs[0]['prompt_ids'],request_id=E.label_for(slot),session_id=E.SESSION,
            request_index=0,purpose='evaluation',save_to=self.root/'zero.json')
        ref=dict(self.refs['eval0'],response_ids=[151645],F_counts={'verification_calls':999})
        row,errors=E.measure(record,self.inputs[0],'/F',ref,slot,'GPU-fixed')
        self.assertEqual(errors,[])
        self.assertEqual(row['native_accepted_draft_tokens'],0)


class AnalysisTests(unittest.TestCase):
    def test_common_bootstrap_identity_and_pooled_counts(self):
        rows=scalar_rows();result=A.summarize(rows)
        paired=result['paired_requests']
        self.assertEqual([r['source_id'] for r in paired],sorted({r['source_id'] for r in rows}))
        expected=np.random.Generator(np.random.PCG64(1711)).integers(0,256,size=(10000,256))
        self.assertEqual(result['bootstrap']['draw_matrix_sha256'],hashlib.sha256(expected.astype('<i8').tobytes()).hexdigest())
        n=sum(r['verification_calls'] for r in rows if r['method']=='N' and r['repeat']==0)
        s=sum(r['verification_calls'] for r in rows if r['method']=='S' and r['repeat']==0)
        self.assertEqual(result['method_totals']['N']['verification_calls'],n)
        self.assertEqual(result['comparisons']['S_vs_N']['call_reduction']['point'],1-s/n)
        self.assertEqual(result['comparisons']['S_vs_N']['net_calls_saved'],256)
        self.assertEqual(result['decision']['outcome'],'PRACTICALLY_SUPPORTED_TRANSFER')

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

    def test_decision_categories_and_clear_harm_precedence(self):
        def effect(time_point,time_ci,accept_ci,repeats=(.01,.01,.01)):
            return dict(time_reduction={'point':time_point,'ci95':time_ci},
                acceptance_improvement={'point':.01,'ci95':accept_ci},repeat_time_reductions=repeats)
        cases=[(effect(.005,[.001,.009],[.001,.01]),'PRACTICALLY_SUPPORTED_TRANSFER'),
               (effect(.001,[-.001,.004],[-.02,-.001]),'CLEAR_HARM_ON_THIS_COMPARISON'),
               (effect(-.01,[-.02,-.001],[.001,.01]),'CLEAR_HARM_ON_THIS_COMPARISON'),
               (effect(.001,[-.001,.004],[-.01,.01]),'PRACTICAL_TIME_GAIN_NOT_SUPPORTED'),
               (effect(.004,[.001,.009],[.001,.01]),'UNRESOLVED_OR_MIXED'),
               (effect(.006,[.001,.01],[.001,.01],(.01,-.001,.01)),'UNRESOLVED_OR_MIXED'),
               (effect(.006,[.001,.01],None),'UNRESOLVED_OR_MIXED')]
        for value,expected in cases:self.assertEqual(A.decision(value)['outcome'],expected)


if __name__=='__main__':unittest.main()
