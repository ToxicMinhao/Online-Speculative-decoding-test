"""Fresh-panel CPU qualification: synthetic native records, no model imports."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.fresh_mathqa_endpoints import evaluation as E, analysis as A
from test_full_k_gsm8k_evaluation import FullBackend, requests, harmful
from test_persistent_prefix_evaluation import references, quiet

IDENTITIES = {m:dict(bytes=123,sha256=str(i+1)*64) for i,m in enumerate(E.METHODS)}
PROVENANCE = {k:{'sha256':str(i+4)*64} for i,k in enumerate(
    ('target_identity','tokenizer_identity','runtime_identity'))}


def seal(value):
    payload={k:v for k,v in value.items() if k!='binding_digest'}
    value['binding_digest']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return value


def panel():
    rows=requests(1024)
    for i,row in enumerate(rows):row.update(dataset='mathqa',source_id=f'mathqa:test:{i}')
    return rows


def bound_refs(rows,producer,*,fresh=True):
    values=references(rows)
    for request in rows:
        value=values[request['request_id']]
        value.update({k:request[k] for k in ('request_id','source_id','stop_token_ids','max_new_tokens')})
        value['reference_checkpoint']=producer
        if fresh:value.update(native_checkpoint=producer,producer_weight_identity=deepcopy(IDENTITIES['O58']),**deepcopy(PROVENANCE))
        seal(value)
    return values


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);self.root=Path(directory.name)
        self.rows=panel();self.warm=deepcopy(self.rows[:2])
        for i,row in enumerate(self.warm):row.update(request_id=f'warm{i}',source_id=f'warm:{i}',prompt_ids=[1,9000+i])
        self.refs=bound_refs(self.rows,'/O58');self.wrefs=bound_refs(self.warm,'/guard',fresh=False)
        self.checkpoints={m:'/'+m for m in E.METHODS}

    def arguments(self):
        return dict(dataset='mathqa',canonicals=self.refs,warmup_requests=self.warm,
            warmup_canonicals=self.wrefs,schedule=E.evaluation_schedule(self.rows),
            reference_checkpoints={'scored':'/O58','warmup':'/guard'},physical_gpu_uuid='GPU-fixed',
            boundary_check=quiet,execution_commit='a'*40,checkpoint_identities=deepcopy(IDENTITIES),
            canonical_provenance=deepcopy(PROVENANCE))

    def run_final(self,backend,name,**changes):
        return E.evaluate_final(backend,self.rows,self.checkpoints,self.root/name,**(self.arguments()|changes))

    def test_exact_aggregate_balanced_schedule_and_complete_coverage(self):
        schedule=E.evaluation_schedule(self.rows)
        orders=[tuple(x['method'] for x in schedule[i:i+3]) for i in range(0,144,3)]
        self.assertEqual(Counter(orders),{p:8 for p in E.ORDER_PERMUTATIONS})
        self.assertEqual(orders,[E.ORDER_PERMUTATIONS[(16*r+b)%6] for r in range(3) for b in range(16)])
        self.assertEqual(Counter((x['method'],x['position']) for x in schedule),{(m,p):16 for m in E.METHODS for p in range(3)})
        slots=list(E.execution_slots(schedule,self.rows,self.warm,dataset='mathqa'))
        self.assertEqual(Counter(s['phase'] for s,_ in slots),{'scored':9216,'warmup':288})
        self.assertEqual(len({E.label_for(s) for s,_ in slots}),9504)
        self.assertEqual(Counter((s['method'],s['repeat'],s['request_id']) for s,_ in slots if s['phase']=='scored'),
            {(m,r,q['request_id']):1 for m in E.METHODS for r in range(3) for q in self.rows})
        with self.assertRaises(ValueError):E.dataset_contract('gsm8k')

    def test_complete_fresh_panel_without_old_counts_and_raw_reaudit(self):
        backend=FullBackend(self.rows);result=self.run_final(backend,'complete');root=self.root/'complete'
        self.assertEqual((len(backend.executions),len(backend.switches)),(9504,144))
        self.assertEqual((result['scored_generations'],result['warmup_generations']),(9216,288))
        self.assertEqual(set(backend.weight_checks),set(self.checkpoints.values()))
        with patch.object(A,'BOOTSTRAP_DRAWS',32):summary=A.write_report(root)
        self.assertEqual(summary['status'],'completed')
        self.assertEqual(summary['primary_comparison'],'KPLUS_C58_vs_AREPO58')
        self.assertEqual(set(summary['comparisons']),{'KPLUS_C58_vs_AREPO58','KPLUS_C58_vs_O58','AREPO58_vs_O58'})
        self.assertEqual(len((root/'ALL_REPEATS.csv').read_text().splitlines()),9217)
        self.assertEqual(len((root/'PER_REQUEST.csv').read_text().splitlines()),3073)
        self.assertEqual(len((root/'WARMUPS.csv').read_text().splitlines()),289)
        self.assertFalse((root/'HISTORICAL_COUNTS.json').exists())
        measurements=root/'MEASUREMENTS.jsonl';original=measurements.read_text();lines=original.splitlines()
        for changed in ('\n'.join(lines[:-1])+'\n', original+lines[0]+'\n'):
            measurements.write_text(changed);audit=E.audit_raw_records(root)
            self.assertIn('measurement_count',audit['errors'])
        measurements.write_text(original)
        with self.assertRaises(FileExistsError):self.run_final(FullBackend(self.rows),'complete')
        # Saved valid flags never replace replay of the native decisions.
        record=root/'0000_r0_AREPO58.json';raw=json.loads(record.read_text());raw['response_ids'][0]+=1
        record.write_text(json.dumps(raw));audit=E.audit_raw_records(root)
        self.assertEqual(audit['status'],'invalid_or_incomplete')
        self.assertTrue(any('0000_r0_AREPO58' in x for x in audit['errors']))

    def test_canonical_binding_request_and_producer_identity_fail_before_serving(self):
        with self.assertRaisesRegex(ValueError,'canonical provenance'):
            self.run_final(FullBackend(self.rows),'missingprovenance',canonical_provenance=None)
        original=deepcopy(self.refs)
        for mutation,reseal in [
            (lambda ref:ref['response_ids'].__setitem__(0,999),False),
            (lambda ref:ref.update(request_id='other'),True),
            (lambda ref:ref.update(native_checkpoint='/AREPO58'),True),
            (lambda ref:ref.update(producer_weight_identity={'bytes':123,'sha256':'f'*64}),True),
            (lambda ref:ref.update(target_identity={'sha256':'f'*64}),True),
            (lambda ref:ref.pop('runtime_identity'),True),
        ]:
            self.refs=deepcopy(original);ref=self.refs['eval0'];mutation(ref)
            if reseal:seal(ref)
            backend=FullBackend(self.rows)
            with self.assertRaises(ValueError):self.run_final(backend,'badbinding')
            self.assertFalse(backend.executions);self.assertFalse(backend.switches)
        self.refs=original
        wrong=deepcopy(IDENTITIES);wrong['O58']['sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'checkpoint identity'):self.run_final(FullBackend(self.rows),'wrongweights',checkpoint_identities=wrong)

    def test_missing_duplicate_inputs_and_schedule_tampering_rejected(self):
        bad=deepcopy(self.refs);bad.pop('eval0')
        with self.assertRaises(ValueError):self.run_final(FullBackend(self.rows),'missingref',canonicals=bad)
        self.rows[1]['source_id']=self.rows[0]['source_id']
        self.refs=bound_refs(self.rows,'/O58')
        with self.assertRaisesRegex(ValueError,'distinct'):self.run_final(FullBackend(self.rows),'duplicate')
        self.rows=panel();self.refs=bound_refs(self.rows,'/O58')
        schedule=E.evaluation_schedule(self.rows);schedule[0]['method']='O58'
        with self.assertRaisesRegex(ValueError,'schedule changed'):self.run_final(FullBackend(self.rows),'schedule',schedule=schedule)

    def test_invalid_unknown_and_wrong_weight_activation_stop_preserving_work(self):
        def corrupt(record,options):record['response_ids'][0]+=1
        with self.assertRaisesRegex(RuntimeError,'correctness'):self.run_final(FullBackend(self.rows,corrupt),'invalid')
        status=json.loads((self.root/'invalid/STATUS.json').read_text())
        self.assertEqual((status['invalid_generations'],status['unknown_generation_attempts'],status['unattempted_generations']),(1,0,9503))
        self.assertEqual(len(list((self.root/'invalid').glob('*_VALIDATION.json'))),1)
        backend=FullBackend(self.rows)
        with patch.object(backend,'generate',side_effect=RuntimeError('lost native return')):
            with self.assertRaisesRegex(RuntimeError,'lost native'):self.run_final(backend,'unknown')
        status=json.loads((self.root/'unknown/STATUS.json').read_text());self.assertEqual(status['unknown_generation_attempts'],1)
        backend=FullBackend(self.rows)
        wrong_reply=backend.check_weights('/AREPO58')
        wrong_reply['all_equal']=False
        with patch.object(backend,'check_weights',return_value=wrong_reply):
            with self.assertRaisesRegex(RuntimeError,'weights differ'):self.run_final(backend,'wrongactivation')
        self.assertFalse(backend.executions)

    def test_native_termination_checkpoint_and_repeat_consistency(self):
        request=self.rows[0];slot=next(E.execution_slots(E.evaluation_schedule(self.rows),self.rows,self.warm,dataset='mathqa'))[0]
        slot.update(phase='scored',panel_ordinal=0,request_id=request['request_id'])
        def one(mutation=None):
            backend=FullBackend(self.rows,mutation);backend.publish('/AREPO58')
            return backend.generate(request['prompt_ids'],request_index=0,request_id=E.label_for(slot),
                session_id=E.dataset_contract('mathqa')['session'],purpose='evaluation',save_to=self.root/'single.json')
        first,errors=E.measure(one(),request,'/AREPO58',self.refs['eval0'],slot,'GPU-fixed',dataset='mathqa')
        self.assertFalse(errors)
        for mutate in [lambda r,o:r.update(finish_reason='length'),lambda r,o:r.update(checkpoint_id='/wrong')]:
            _,errors=E.measure(one(mutate),request,'/AREPO58',self.refs['eval0'],slot,'GPU-fixed',dataset='mathqa')
            self.assertTrue(errors)
        second,errors=E.measure(one(lambda r,o:harmful(r)),request,'/AREPO58',self.refs['eval0'],slot,'GPU-fixed',dataset='mathqa')
        self.assertFalse(errors)
        self.assertIn('repeat_count_mismatch',E._repeat_errors([first],second))

if __name__=='__main__':unittest.main()
