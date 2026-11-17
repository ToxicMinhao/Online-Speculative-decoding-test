"""Synthetic package consumer tests; no real candidate results or model imports."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import run_secondary_collection as D
sys.path.insert(0,str(D.SRC/'tests'))
from test_pr49_collection_cpu import fixture


def inputs():
    requests=[];records={a:[] for a in ('N','C','S')};captures={a:{} for a in records}
    checkpoints={a:dict(export=a+'7',weights_identity=dict(bytes=100,sha256={'N':'a','C':'b','S':'c'}[a]*64)) for a in records}
    for i in range(128):
        original,request,_=fixture('bonus')
        request.update(request_id=f'r{i:03d}',source_id=f's{i:03d}',request_index=897+i,prompt_ids=[7,1000+i])
        requests.append(request)
        for arm in records:
            record=deepcopy(original)
            record.update(record_id=request['request_id'],request_index=request['request_index'],
                prompt_ids=request['prompt_ids'],session_id='bounded_prefix_allocation_C_event8' if arm=='C' else f'persistent_{arm}_event8',checkpoint_id=arm+'7')
            captured=record['metadata']['capture']
            captured.update(prompt_ids=request['prompt_ids'],covered_token_ids=(request['prompt_ids']+record['response_ids'])[:-1])
            records[arm].append(record);captures[arm][request['request_id']]=deepcopy(captured)
    return requests,records,captures,checkpoints,None


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'new';self.root.mkdir();self.old=self.base/'old';self.old.mkdir()
        self.requests,self.records,self.captures,self.checkpoints,_=inputs()
        self.remote=Path('/remote/new');self.commit='a'*40;self.binding='b'*64
        self.payload=b'qualified source fixture\n';pin=dict(bytes=len(self.payload),sha256=D.hashlib.sha256(self.payload).hexdigest())
        source_names=('experiments/bounded_prefix_allocation/task_runner.py','experiments/bounded_prefix_allocation/training.py','experiments/persistent_prefix_withholding/worker.py','online_feedback/sglang_worker.py','online_feedback/sglang_backend.py','online_feedback/native.py')
        self.save('SOURCE.json',dict(commit=self.commit,dirty=False,files={n:pin for n in source_names}))
        runtime=dict(native_runtime_sha256={'speculative/dflash_worker.py':'d'*64},training={},serving={})
        sealed=dict(requests=self.requests);self.save('sealed/EVENT8_INPUTS.json',sealed)
        self.plan=dict(root=str(self.remote),source_manifest=str(self.remote/'SOURCE.json'),execution_commit=self.commit,
            binding=self.binding,target='/target',runtime_contract=runtime,event8_inputs=str(self.remote/'sealed/EVENT8_INPUTS.json'),
            origin_state='/K6.pt',origin_export='/K6',input_identities={str(self.remote/'sealed/EVENT8_INPUTS.json'):D.identity(self.root/'sealed/EVENT8_INPUTS.json')})
        self.save('sealed/PLAN.json',self.plan);self.save('LAUNCH.json',dict(status='completed'))
        self.save('CPU_PREFLIGHT.json',dict(status='passed',execution_commit=self.commit,runtime=dict(installed_sources={'speculative/dflash_worker.py':dict(sha256='d'*64)}),inputs={}))
        boundary=dict(status='passed',export_exact=True,global_step=896,live=dict(exact_resume=True,mismatches=[]))
        endpoint=dict(state=str(self.remote/'tasks/C7/trainer/training_state.pt'),export=str(self.remote/'tasks/C7/trainer/events/event_007/export'),global_step=896,presentations=1792,
            state_identity=dict(bytes=200,sha256='f'*64),weights_identity=self.checkpoints['C']['weights_identity'])
        self.checkpoints['C']=endpoint
        self.save('tasks/C7/RESULT.json',dict(status='completed',arm='C',event=7,execution_commit=self.commit,binding=self.binding,endpoint=endpoint,boundary=boundary))
        self.save('tasks/C7/BOUNDARY.json',boundary)
        self.save('tasks/C7/trainer/RESUME_CHECK.json',dict(status='passed',exact_resume=True,mismatches=[],global_step=768,origin_state='/K6.pt',origin_export='/K6'))
        collection=dict(status='completed',arm='C',execution_commit=self.commit,binding=self.binding,
            own_checkpoint=endpoint['export'],own_weights_identity=endpoint['weights_identity'],generated_responses=128,teacher_preparations=128,
            feature_paths=[str(self.remote/'tasks/C_collect/loop/features'/f"{r['request_id']}.pt") for r in self.requests])
        for name in ('RESULT','FEATURES'):self.save('tasks/C_collect/'+name+'.json',collection)
        self.save('tasks/C_collect/native/engine_settings.json',dict(model_path='/target',speculative_draft_model_path=endpoint['export']))
        self.save('tasks/C_collect/native/ready.json',dict(status=dict(status='passed',publication=None)))
        for ordinal,name in enumerate(('C7','C_collect','C8')):
            self.save('tasks/'+name+'/STATUS.json',dict(status='completed',execution_commit=self.commit,binding=self.binding,started_epoch=2*ordinal,wall_s=1.))
        mappings=[];consumed=[]
        for arm in ('N','C','S'):
            for ordinal,request in enumerate(self.requests):
                record=deepcopy(self.records[arm][ordinal]);rid=request['request_id']
                native=f'tasks/{arm}_collect/native/request_{ordinal+1:06d}.json'
                rel=f'tasks/{arm}_collect/records/{rid}.json'
                if arm=='C':
                    record['checkpoint_id']=endpoint['export'];record['metadata']['native_trace_path']=str(self.remote/native)
                    self.save(rel,record);self.save(native,self.captures[arm][rid])
                else:
                    self.save(rel,record,root=self.old);self.save(native,self.captures[arm][rid],root=self.old)
                    mappings.append(dict(arm=arm,ordinal=ordinal,request_id=rid,source_id=request['source_id'],request_index=request['request_index'],record_path=str(self.old/rel),native_path=str(self.old/native)))
                    for path in (self.old/rel,self.old/native):consumed.append(dict(local_path=str(path),identity=D.identity(path)))
        self.oldproof=dict(status='passed',checkpoints={a:self.checkpoints[a] for a in ('N','S')},
            runtime=dict(target='/target',runtime_contract=runtime,inherited_tokenizer_identities={},engine=dict(model_path='/target')),
            chronology={a:dict(own_collection_precedes_training=True) for a in ('N','S')},request_mappings=mappings,consumed_inputs=consumed)
        self.refresh_manifest()

    def save(self,relative,value,root=None):
        path=(root or self.root)/relative;path.parent.mkdir(parents=True,exist_ok=True);D.write(path,value)

    def refresh_manifest(self):
        D.write(self.root/'RETRIEVAL_MANIFEST.json',{str(p.relative_to(self.root)):D.identity(p) for p in self.root.rglob('*.json') if p.name!='RETRIEVAL_MANIFEST.json'})

    def run_driver(self):
        def prior_recheck(*args):
            destination=args[-1];destination.mkdir();D.write(destination/'PROVENANCE.json',self.oldproof)
            return deepcopy(self.oldproof)
        with patch.object(D.historical,'run',side_effect=prior_recheck),patch.object(D.subprocess,'check_output',return_value=self.payload):
            return D.run(self.root,self.old,self.base/'q',self.base/'r',self.base/'out')

    def test_source_discovery_supports_task_and_published_locations(self):
        task=self.base/'workspace';source=task/'implementation_checkout/DFlash-Online'
        (source/'experiments').mkdir(parents=True);(source/'online_feedback').mkdir()
        self.assertEqual(D.discover_source(start=task),source)
        published=source/'reports/study/analysis';published.mkdir(parents=True)
        self.assertEqual(D.discover_source(start=published),source)
        self.assertEqual(D.discover_source(source),source)
        with self.assertRaises(FileNotFoundError):D.discover_source(self.base/'missing')

    def test_complete_package_binds_source_and_counts_without_latency(self):
        result=self.run_driver()
        self.assertEqual(result['status'],'completed',result.get('error',result.get('validation')))
        self.assertEqual(result['totals']['C']['V'],128)
        self.assertTrue(result['latency_excluded']);self.assertFalse(result['primary_endpoint_decision_affected'])
        self.assertTrue((self.base/'out/PER_REQUEST.csv').is_file())
        proof=json.loads((self.base/'out/PROVENANCE.json').read_text());self.assertEqual(proof['status'],'passed')

    def test_wrong_own_weights_receipt_invalidates_secondary(self):
        for name in ('RESULT','FEATURES'):
            path=f'tasks/C_collect/{name}.json';value=json.loads((self.root/path).read_text());value['own_weights_identity']['sha256']='0'*64;self.save(path,value)
        self.refresh_manifest();result=self.run_driver()
        self.assertEqual(result['status'],'INVALID_OR_INCOMPLETE_SECONDARY');self.assertNotIn('comparisons',result)

    def test_changed_old_raw_file_cannot_rely_on_prior_pass_string(self):
        path=self.old/'tasks/N_collect/records/r000.json';value=json.loads(path.read_text());value['purpose']='evaluation';D.write(path,value)
        result=self.run_driver();self.assertEqual(result['status'],'INVALID_OR_INCOMPLETE_SECONDARY');self.assertNotIn('comparisons',result)

    def test_missing_capture_preserves_incomplete_status(self):
        (self.root/'tasks/C_collect/native/request_000001.json').unlink()
        result=self.run_driver();self.assertEqual(result['status'],'INVALID_OR_INCOMPLETE_SECONDARY');self.assertNotIn('comparisons',result)
        self.assertTrue((self.base/'out/PROVENANCE.json').is_file())

    def test_collection_after_own_training_is_not_admitted(self):
        path='tasks/C8/STATUS.json';value=json.loads((self.root/path).read_text());value['started_epoch']=2.;self.save(path,value);self.refresh_manifest()
        result=self.run_driver();self.assertEqual(result['status'],'INVALID_OR_INCOMPLETE_SECONDARY');self.assertNotIn('comparisons',result)


if __name__=='__main__':unittest.main()
