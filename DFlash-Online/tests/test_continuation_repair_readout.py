"""Fixed-parent mechanism reduction and explicit read-only endpoint identity."""
from collections import Counter
from copy import deepcopy
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from experiments.continuation_repair_pilot.analysis import mechanism_summary
from experiments.continuation_repair_pilot.readout import paired_rows, run_readout, validate_saved_readout, resume_completed_readouts
from experiments.continuation_value_diagnostic.metrics import paired_block, roles_from_predictions
from experiments.continuation_value_diagnostic.native import ReadOnlyEndpoint
from test_continuation_value_native import DraftFixture


def block(matches):
    labels = list(range(16)); proposals = labels.copy()
    for i, matched in enumerate(matches,1):
        if not matched: proposals[i]=31
    valid=[False]+[True]*len(matches)+[False]*(15-len(matches))
    roles=roles_from_predictions(labels,valid,proposals)
    return dict(labels=labels,valid=valid,proposals=proposals,roles=roles,m=len(matches),
        positions=[dict(position=i,margin=1. if matched else -1., support_fp32=.5 if matched else 0.)
                   for i,matched in enumerate(matches,1)])


class ReadoutTests(unittest.TestCase):
    def test_explicit_tail_label_leaves_legacy_restriction_unchanged(self):
        with self.assertRaisesRegex(ValueError,'explicit read-only contract'):
            ReadOnlyEndpoint(DraftFixture(),endpoint='TAIL',maximum_length=100)
        endpoint=ReadOnlyEndpoint(DraftFixture(),endpoint='TAIL',maximum_length=100,allowed_endpoints=('TAIL',))
        self.assertEqual(endpoint.endpoint,'TAIL')
        self.assertEqual(endpoint.close()['endpoint'],'TAIL')
        legacy=ReadOnlyEndpoint(DraftFixture(),endpoint='child',maximum_length=100)
        self.assertEqual(legacy.close()['endpoint'],'child')

    def test_raw_repair_damage_and_fixed_parent_horizon(self):
        origin=block([True,True,False,True,True,False])
        base=paired_block(origin,block([True,True,True,True,False,True]),record_id='r',anchor=3)
        tail=paired_block(origin,block([False,True,True,True,True,True]),record_id='r',anchor=3)
        result=mechanism_summary([base],[tail],cohort='primary')
        self.assertEqual(result['totals']['BASE']['Y'],2)
        self.assertEqual(result['totals']['TAIL']['raw_repair'],1)
        self.assertEqual(result['totals']['TAIL']['useful_repair'],0)
        self.assertEqual(result['totals']['TAIL']['lost_prefix'],2)
        self.assertEqual(result['differences']['Y'],-4)
        self.assertEqual(result['responses'],1)

    def test_parent_change_duplicate_or_coverage_change_rejected(self):
        origin=block([True,False,True])
        row=paired_block(origin,origin,record_id='r',anchor=3)
        changed=deepcopy(row);changed['K']=2
        with self.assertRaisesRegex(ValueError,'origin changed'):
            mechanism_summary([row],[changed],cohort='primary')
        with self.assertRaisesRegex(ValueError,'unique matched'):
            mechanism_summary([row],[row,row],cohort='primary')
        changed=deepcopy(row);changed['Y']=1
        with self.assertRaisesRegex(ValueError,'identity failed'):
            mechanism_summary([row],[changed],cohort='primary')

    def test_readout_phase_budget_and_failure_finalization(self):
        origin=block([True,False,True])
        bindings=[dict(record_id=f'r{i}',ordinal=i,anchors=[3]) for i in range(128)]
        parent={b['record_id']:dict(record_id=b['record_id'],anchors=b['anchors'],blocks=[origin]) for b in bindings}
        legacy_bindings=[dict(record_id=f'old{i}',anchors=list(range(8))) for i in range(8)]
        legacy_rows=[paired_block(origin,origin,record_id=b['record_id'],anchor=a)
                     for b in legacy_bindings for a in b['anchors']]
        contract=dict(features=bindings,legacy_binding={'sets':{'parent_training':legacy_bindings}},
                      endpoints={'child':{'identity':{'sha256':'base'}}})
        class Adapter:
            loading={'mapping':{'status':'passed'}}
            def __init__(self, fail_close):
                self.fail_close=fail_close
                self.work=dict(response_reads_attempted=0,response_reads_completed=0)
            def read_response(self,binding,path):
                self.work['response_reads_attempted']+=1
                self.work['response_reads_completed']+=1
                return dict(record_id=binding['record_id'],anchors=binding['anchors'],
                            geometry={'block_size':16},blocks=[origin]*len(binding['anchors']))
            def close(self):
                if self.fail_close: raise ValueError('fixture state mutation on close')
                return dict(status='passed',work=dict(self.work))
        for failure in (False,True):
            with self.subTest(failure=failure), TemporaryDirectory() as folder:
                root=Path(folder); qualification=root/'qualification.json'
                qualification.write_text(json.dumps({'status':'passed','binding':'fixed','base_disabled_exact':True,'retained_BASE_reusable':True}))
                plan=dict(root=str(root),qualification_result=str(qualification),binding='fixed',original_config='config',training_gpu='fixtureGPU')
                receipts=dict(BASE={'checkpoint_identity':{'sha256':'base'}},
                              TAIL={'checkpoint':'tail','checkpoint_identity':{'sha256':'tail'}})
                adapter=Adapter(failure)
                with patch.dict('os.environ',CUDA_VISIBLE_DEVICES='fixtureGPU'), \
                     patch('experiments.continuation_repair_pilot.launch.boundary',return_value={}), \
                     patch('experiments.continuation_repair_pilot.serving.endpoint_receipts',return_value=receipts), \
                     patch('experiments.continuation_repair_pilot.readout.authenticated_retained',
                           return_value=(contract,parent,parent,{},{})), \
                     patch('experiments.continuation_repair_pilot.readout.legacy_pairs',return_value=legacy_rows), \
                     patch('experiments.continuation_repair_pilot.readout.ReadOnlyEndpoint.load',return_value=adapter):
                    if failure:
                        with self.assertRaisesRegex(ValueError,'mutation on close'): run_readout(plan)
                    else:
                        self.assertEqual(run_readout(plan)['status'],'completed')
                status=json.loads((root/'mechanism/STATUS.json').read_text())
                self.assertEqual(status['work']['response_reads_completed'],137)
                self.assertEqual(status['unattempted_response_forwards'],0)
                self.assertEqual(status['status'],'failed' if failure else 'completed')
                if failure: self.assertIn('mutation on close',status['closure_error'])

    def test_resume_only63_unexecuted_reads_preserves_first74(self):
        origin=block([True,False,True])
        bindings=[dict(record_id=f'r{i}',ordinal=i,anchors=list(range(64)),
                       feature_identity={'sha256':str(i)}) for i in range(128)]
        old=[dict(record_id=f'old{i}',anchors=list(range(8)),feature_identity={'sha256':'old'+str(i)}) for i in range(8)]
        parent={b['record_id']:dict(record_id=b['record_id'],anchors=b['anchors'],blocks=[origin]*64) for b in bindings}
        legacy_rows=[paired_block(origin,origin,record_id=b['record_id'],anchor=a) for b in old for a in b['anchors']]
        contract=dict(features=bindings,legacy_binding={'sets':{'parent_training':old}},endpoints={'child':{'identity':{'sha256':'base'}}})
        loading=dict(mapping={'export_identity':{'sha256':'tail'},'exact_copy':True,'tensors':58})
        def record(binding):
            n=len(binding['anchors'])
            return dict(status='passed',endpoint='TAIL',record_id=binding['record_id'],anchors=binding['anchors'],
                feature_identity=binding['feature_identity'],geometry={'block_size':16},blocks=[origin]*n,
                rng_before='rng',rng_after='rng',parameters_unchanged=True,gradients_unchanged=True,
                runtime_restored=True,rng_preserved=True,
                work=dict(response_reads_attempted=1,response_reads_completed=1,draft_forwards_attempted=1,
                    draft_forwards_completed=1,head_calls_attempted=n//8,head_calls_completed=n//8,
                    blocks_completed=n,backwards=0,optimizer_steps=0,teacher_forwards=0,
                    feedback_generations=0,serving_generations=0))
        class Adapter:
            def __init__(self):
                self.loading=loading;self.work=Counter();self.called=[]
            def read_response(self,binding,path):
                self.called.append(binding['record_id'])
                self_record=record(binding);self.work.update(self_record['work'])
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(self_record))
                return self_record
            def close(self):
                return dict(status='passed',work=dict(self.work),loading=loading,parameters_unchanged=True,
                            modes_unchanged=True,optimizer_created=False)
        with TemporaryDirectory() as folder:
            root=Path(folder);mechanism=root/'mechanism';(mechanism/'primary').mkdir(parents=True)
            prior_work=Counter()
            for ordinal,binding in enumerate(bindings[:73]):
                actual=record(binding);prior_work.update(actual['work'])
                (mechanism/'primary'/f'{ordinal:03d}.json').write_text(json.dumps(actual))
            actual=record(bindings[0]);prior_work.update(actual['work'])
            (mechanism/'qualification_repeat.json').write_text(json.dumps(actual))
            previous=dict(status='failed',error=repr(FileNotFoundError(2,'No such file or directory',str(root/'STORAGE.json.tmp'))),completed_primary_responses=73,
                          completed_retention_responses=0,work=dict(prior_work),wall_s=89.5)
            self.assertEqual(previous['error'], "FileNotFoundError(2, 'No such file or directory')")
            failure_log=root/'readout.log'
            failure_log.write_text(f"FileNotFoundError: [Errno 2] No such file or directory: '{root/'STORAGE.json.tmp'}'\n")
            from experiments.ok_coding.storage import identity
            repair=root/'resume_001';repair.mkdir()
            (repair/'REPAIR.json').write_text(json.dumps({'original_plan':str(root/'sealed/PLAN.json'),'preserved_failures':{'readout.log':identity(failure_log)}}))
            (mechanism/'STATUS.json').write_text(json.dumps(previous))
            (mechanism/'LOADING.json').write_text(json.dumps(loading))
            (mechanism/'ENDPOINT_STATE.json').write_text(json.dumps(dict(status='passed',work=dict(prior_work),loading=loading,
                parameters_unchanged=True,modes_unchanged=True,optimizer_created=False)))
            (mechanism/'ADAPTER_QUALIFICATION.json').write_text(json.dumps({'status':'passed'}))
            original={str(p.relative_to(mechanism)):p.read_bytes() for p in mechanism.rglob('*.json')}
            q=root/'qualification.json';q.write_text(json.dumps({'status':'passed','binding':'fixed','base_disabled_exact':True,'retained_BASE_reusable':True}))
            plan=dict(root=str(root),qualification_result=str(q),binding='fixed',original_config='config',training_gpu='fixtureGPU')
            receipts=dict(BASE={'checkpoint_identity':{'sha256':'base'}},TAIL={'checkpoint':'tail','checkpoint_identity':{'sha256':'tail'}})
            adapter=Adapter()
            preflight=resume_completed_readouts(mechanism,contract,receipts['TAIL'],preserve=False)
            self.assertFalse(preflight[2]['preservation_written'])
            self.assertFalse((mechanism/'attempts').exists())
            self.assertFalse((mechanism/'RESUME_BINDING.json').exists())
            self.assertEqual((mechanism/'STATUS.json').read_bytes(),original['STATUS.json'])
            original_trace=failure_log.read_bytes()
            failure_log.write_text('changed traceback')
            with self.assertRaisesRegex(ValueError,'traceback changed'):
                resume_completed_readouts(mechanism,contract,receipts['TAIL'],preserve=False)
            failure_log.write_bytes(original_trace)
            with patch.dict('os.environ',CUDA_VISIBLE_DEVICES='fixtureGPU'), \
                 patch('experiments.continuation_repair_pilot.launch.boundary',return_value={}), \
                 patch('experiments.continuation_repair_pilot.serving.endpoint_receipts',return_value=receipts), \
                 patch('experiments.continuation_repair_pilot.readout.authenticated_retained',return_value=(contract,parent,parent,{},{})), \
                 patch('experiments.continuation_repair_pilot.readout.legacy_pairs',return_value=legacy_rows), \
                 patch('experiments.continuation_repair_pilot.readout.ReadOnlyEndpoint.load',return_value=adapter):
                result=run_readout(plan,resume_unexecuted=True)
            self.assertEqual(adapter.called,[b['record_id'] for b in bindings[73:]]+[b['record_id'] for b in old])
            self.assertEqual(len(adapter.called),63)
            self.assertEqual(result['attempt_work']['response_reads_completed'],63)
            self.assertEqual(result['work']['response_reads_completed'],137)
            self.assertEqual(result['work']['head_calls_completed'],1040)
            self.assertEqual(result['work']['blocks_completed'],8320)
            self.assertEqual(result['unattempted_response_forwards'],0)
            self.assertEqual(result['completed_primary_responses'],128)
            self.assertEqual(result['completed_retention_responses'],8)
            self.assertEqual(result['status'],'completed')
            for name,data in original.items():
                destination=mechanism/name if name.startswith('primary/') or name=='qualification_repeat.json' else mechanism/'attempts/001'/name
                self.assertEqual(destination.read_bytes(),data)
            self.assertEqual(json.loads((mechanism/'attempts/002/ENDPOINT_STATE.json').read_text())['work']['response_reads_completed'],63)

    def test_saved_readout_rejects_incomplete_or_changed_binding(self):
        binding=dict(record_id='r',anchors=[1],feature_identity={'sha256':'f'})
        incomplete=dict(status='failed',endpoint='TAIL')
        with self.assertRaisesRegex(ValueError,'incomplete'):
            validate_saved_readout(incomplete,binding)

    def test_binding_order_and_missing_readout_rejected(self):
        origin=block([True,False,True])
        before={'r':dict(record_id='r',anchors=[3],blocks=[origin])}
        binding=[dict(record_id='r',anchors=[3])]
        self.assertEqual(len(paired_rows(before,before,binding,cohort='primary')),1)
        after=deepcopy(before);after['r']['anchors']=[4]
        with self.assertRaisesRegex(ValueError,'bindings'):
            paired_rows(before,after,binding,cohort='primary')


if __name__ == '__main__': unittest.main()
