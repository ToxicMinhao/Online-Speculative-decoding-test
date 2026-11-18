"""CPU qualification of genuine two-arm, dataset-bound native evaluation."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.c_paired_confirmation import analysis as A, evaluation as E
from test_full_k_gsm8k_evaluation import FullBackend, requests, harmful
from test_persistent_prefix_evaluation import references, quiet


def panel(dataset):
    result=requests(256)
    for i,row in enumerate(result):row.update(dataset=dataset,source_id=f'{dataset}:test:{i}')
    return result


def origin_references(rows,origin):
    result={k:dict(v,reference_checkpoint=origin) for k,v in references(rows).items()}
    for value in result.values():
        payload={k:v for k,v in value.items() if k!='binding_digest'}
        value['binding_digest']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.dataset='gsm8k';self.inputs=panel(self.dataset);self.origin='/KPLUS6'
        self.warmups=deepcopy(self.inputs[:2])
        for i,row in enumerate(self.warmups):row.update(request_id=f'warm{i}',source_id=f'gsm8k:warm:{i}',prompt_ids=[1,8000+i])
        self.refs=origin_references(self.inputs,self.origin);self.warmrefs=origin_references(self.warmups,self.origin)

    def run_final(self,backend,name):
        return E.evaluate_final(backend,self.inputs,{'N':'/N29','C':'/C29'},self.root/name,dataset=self.dataset,
            canonicals=self.refs,warmup_requests=self.warmups,warmup_canonicals=self.warmrefs,
            schedule=E.evaluation_schedule(self.inputs),origin_checkpoint=self.origin,physical_gpu_uuid='GPU-fixed',
            boundary_check=quiet,execution_commit='c'*40)

    def test_budget_balance_and_dataset_event_bindings(self):
        for dataset,event in [('gsm8k',8),('mathqa',6)]:
            inputs=panel(dataset);schedule=E.evaluation_schedule(inputs)
            self.assertEqual(len(schedule),24)
            self.assertEqual(sum(len(x['request_ordinals']) for x in schedule),1536)
            self.assertEqual(Counter((x['method'],x['position']) for x in schedule),{(m,p):6 for m in E.METHODS for p in range(2)})
            self.assertEqual(Counter(x['method'] for x in schedule if x['position']==0),{'N':6,'C':6})
            # Each arm starts six of the twelve repeat/block groups.
            self.assertEqual(Counter(schedule[i]['method'] for i in range(0,24,2)),{'N':6,'C':6})
            for i in range(0,24,2):
                entry=schedule[i];order=['N','C'] if (entry['repeat']+entry['block_ordinal'])%2==0 else ['C','N']
                self.assertEqual([x['method'] for x in schedule[i:i+2]],order)
            slots=list(E.execution_slots(schedule,inputs,inputs[:2],dataset=dataset))
            self.assertEqual(Counter(x['phase'] for x,_ in slots),{'scored':1536,'warmup':48})
            self.assertTrue(all(x['event']==event and x['dataset']==dataset for x,_ in slots))

    def test_complete_consumer_report_and_native_trace_reaudit(self):
        backend=FullBackend(self.inputs);status=self.run_final(backend,'complete');root=self.root/'complete'
        self.assertEqual((status['scored_generations'],status['warmup_generations']),(1536,48))
        self.assertEqual(len(backend.switches),24)
        self.assertEqual(backend.weight_checks,['/N29','/C29'])
        self.assertTrue(all(x[2]['session_id']==E.dataset_contract('gsm8k')['session'] for x in backend.executions))
        summary=A.write_report(root)
        self.assertEqual(summary['status'],'completed')
        self.assertEqual(summary['dataset'],'gsm8k');self.assertEqual(summary['event'],8)
        self.assertEqual(summary['evidence']['time_reduction'],'DIRECTION_UNRESOLVED')
        self.assertNotIn('decision',summary)
        self.assertEqual(len((root/'ALL_REPEATS.csv').read_text().splitlines()),1537)
        self.assertEqual(len((root/'WARMUPS.csv').read_text().splitlines()),49)
        rawpath=root/'0000_r0_N.json';raw=json.loads(rawpath.read_text());raw['metadata']['trace'][0]['draft_tokens'][0]+=1
        rawpath.write_text(json.dumps(raw))
        self.assertEqual(A.write_report(root)['status'],'INVALID_OR_INCOMPLETE')

    def test_invalid_and_unknown_returns_preserved(self):
        def mutation(record,options):record['response_ids'][0]+=1
        with self.assertRaisesRegex(RuntimeError,'correctness'):self.run_final(FullBackend(self.inputs,mutation),'invalid')
        status=json.loads((self.root/'invalid/STATUS.json').read_text())
        self.assertEqual((status['invalid_generations'],status['unknown_generation_attempts'],status['unattempted_generations']),(1,0,1583))
        self.assertTrue((self.root/'invalid/warmup_a00_0_N.json').is_file())
        backend=FullBackend(self.inputs)
        with patch.object(backend,'generate',side_effect=RuntimeError('lost')):
            with self.assertRaisesRegex(RuntimeError,'lost'):self.run_final(backend,'unknown')
        status=json.loads((self.root/'unknown/STATUS.json').read_text())
        self.assertEqual((status['returned_generations'],status['unknown_generation_attempts'],status['unattempted_generations']),(0,1,1583))

    def test_mixed_dataset_or_missing_arm_rejected_before_calls(self):
        backend=FullBackend(self.inputs);self.warmups[0]['dataset']='mathqa'
        with self.assertRaisesRegex(ValueError,'declared dataset'):self.run_final(backend,'mixed')
        self.assertFalse(backend.executions)
        self.warmups[0]['dataset']='gsm8k'
        with self.assertRaisesRegex(ValueError,'two distinct'):
            E.evaluate_final(backend,self.inputs,{'N':'/N29','S':'/S'},self.root/'missing',dataset='gsm8k',
                canonicals=self.refs,warmup_requests=self.warmups,warmup_canonicals=self.warmrefs,
                schedule=E.evaluation_schedule(self.inputs),origin_checkpoint=self.origin,physical_gpu_uuid='GPU-fixed',
                boundary_check=quiet,execution_commit='c'*40)
        self.assertFalse(backend.switches)

    def test_repeat_mismatch_stops_without_historical_call_gate(self):
        def mutation(record,options):
            if options['request_id']=='0000_r1_N':harmful(record)
        with self.assertRaisesRegex(RuntimeError,'repeat'):self.run_final(FullBackend(self.inputs,mutation),'repeat')
        rows=[json.loads(s) for s in (self.root/'repeat/MEASUREMENTS.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['label'],'0000_r1_N');self.assertFalse(rows[-1]['valid_measurement'])

    def test_zero_call_eos_and_prior_contract_unchanged(self):
        def mutation(record,options):
            record['response_ids']=[151645]
            record['metadata'].update(trace=[],verification_calls=0,accepted_draft_tokens=0,committed_progress=0)
        backend=FullBackend(self.inputs,mutation)
        slot=dict(phase='scored',method='N',repeat=0,event=8,dataset='gsm8k',panel_ordinal=0,activation=0,block_ordinal=0,position=0,request_id='eval0')
        record=backend.generate(self.inputs[0]['prompt_ids'],request_id=E.label_for(slot),session_id=E.dataset_contract('gsm8k')['session'],request_index=0,purpose='evaluation',save_to=self.root/'zero.json')
        reference=dict(self.refs['eval0'],response_ids=[151645],F_counts={'verification_calls':999})
        row,errors=E.measure(record,self.inputs[0],'/F',reference,slot,'GPU-fixed',dataset='gsm8k')
        self.assertEqual(errors,[]);self.assertEqual(row['native_accepted_draft_tokens'],0)
        from experiments.mathqa_fixed_lr_transfer import evaluation as old
        self.assertEqual(old.METHODS,('N','S'));self.assertEqual(old.SESSION,'mathqa_fixed_lr_transfer_event6')


if __name__=='__main__':unittest.main()
