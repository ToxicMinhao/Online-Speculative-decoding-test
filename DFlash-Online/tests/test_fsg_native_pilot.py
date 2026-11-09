"""CPU evidence for native pilot wiring, immutable pools and R1 continuation."""
from copy import deepcopy
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.functional_step_geometry import contract as C
from experiments.functional_step_geometry import training as T
from experiments.functional_step_geometry import native_pilot as N
from experiments.functional_step_geometry import pilot_storage as S
from online_feedback.rng import capture_rng
from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from test_fsg_continuation import continuation, NEW_BINDING
from test_fsg_pilot import presentations


class RevisionTests(unittest.TestCase):
    def test_r1_is_explicit_and_legacy_contract_is_not_rewritten(self):
        old = C.geometry_contract()
        self.assertNotIn('realization_revision',old)
        self.assertEqual(old['scalar_target'],'own_raw_fsg_realized_bfloat16_norm')
        new = C.geometry_contract(C.REALIZATION_R1)
        self.assertEqual(new['feasibility_bisection_iterations'],24)
        self.assertEqual(new['feasibility_endpoint'],'lower_feasible_bracket')
        self.assertEqual(new['scalar_target'],'own_norm_feasible_fsg_realized_bfloat16_norm')
        for key in old.keys()-{'scalar_target'}:
            self.assertEqual(new[key],old[key])
        with self.assertRaises(ValueError):C.geometry_contract('unsealed')

    def test_revision_migration_and_snapshot_bind_all_arms_without_tensor_change(self):
        for arm in C.ARMS:
            origin,state,args = continuation(arm=arm)
            saved=deepcopy(origin)
            current=T.migrate_origin(origin,binding=NEW_BINDING,arm=arm,dataset='gsm8k',
                config_text=state['config_text'],run_id='r1',
                original_config_text=args['original_config_text'],expected_recipe=args['expected_recipe'],
                expected_cuda_rng_states=0,realization_revision=C.REALIZATION_R1)
            self.assertFalse(T.differences(origin,saved))
            self.assertIs(current['backend'],origin['backend'])
            T.validate_snapshot(current,**args,realization_revision=C.REALIZATION_R1)
            with self.assertRaisesRegex(ValueError,'contract differs'):
                T.validate_snapshot(current,**args)
            self.assertEqual(current['algorithm_contract'][T.CONTRACT_KEY]['realization_revision'],C.REALIZATION_R1)


class NativeBindingTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def test_requests_are_exact_existing_events_not_resampled(self):
        rows=[dict(request_id=f'q{i}',event_id=f'event_{5+i//128:03d}') for i in range(256)]
        write_json(self.root/'training.json',dict(train_requests=rows))
        plan=dict(datasets=dict(gsm8k=dict(training_manifest=str(self.root/'training.json'))))
        self.assertEqual(N.feedback_requests(plan,'gsm8k',5),rows[:128])
        self.assertEqual(N.feedback_requests(plan,'gsm8k',6),rows[128:])
        with self.assertRaises(ValueError):N.feedback_requests(plan,'gsm8k',4)

    def test_qualification_feedback_reuses_exact_first_two_packets(self):
        rows=[]
        for i in range(2):
            record=self.root/f'r{i}.json';feature=self.root/f'f{i}.pt'
            record.write_text('{}');feature.write_bytes(bytes([i]))
            rows.append(dict(request_id=f'q{i}',status='completed',record_path=str(record),
                feature_path=str(feature),record_identity=identity(record),feature_identity=identity(feature)))
        preparation=self.root/'preparation.json'
        write_json(preparation,dict(status='completed',datasets=dict(gsm8k=rows)))
        plan=dict(shared_feedback_preparation=str(preparation))
        requests=[dict(request_id=f'q{i}') for i in range(128)]
        self.assertEqual(N.retained_packets(plan,'gsm8k',requests),rows)
        (self.root/'f0.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'packet changed'):N.retained_packets(plan,'gsm8k',requests)

    def test_diagnostic_callback_loads_fixed_a_and_d_and_restores_rng(self):
        baseline=dict(records=presentations())
        pool=dict(records=[])
        for index in range(128):
            record_id=f'train_{index:05d}'
            path=self.root/f'{record_id}.pt'
            torch.save(dict(record_id=record_id,input_ids=torch.tensor([[index]])),path)
            pool['records'].append(dict(record_id=record_id,feature=dict(path=str(path))))
        trainer=SimpleNamespace(global_step=512,response_records=[dict(record_id='train_00000'),dict(record_id='train_00001')],
            _batch=lambda raw:SimpleNamespace(tensors=dict(input_ids=raw['input_ids'])))
        calls=[]
        def diagnostic(hook,a,d,*,output,binding):
            calls.append((a,d,binding))
            torch.rand(3)
            return dict(status='completed',dataset=binding['dataset'],event=binding['event'],window=binding['window'],
                        restoration_exact=True,response_forwards=36,backwards=4)
        with patch.object(N.importlib,'import_module',return_value=SimpleNamespace(run=diagnostic)):
            observer,receipts=N.diagnostic_observer(trainer,pool,baseline,dataset='gsm8k',event=5,
                output=self.root/'diagnostics',factory='fake:run')
        before=capture_rng()
        observer('native',None)
        self.assertFalse(calls)
        observer('committed',None)
        self.assertEqual([row['record_id'] for row in calls[0][0]],['train_00000','train_00001'])
        self.assertEqual([row['record_id'] for row in calls[0][1]],['train_00064','train_00065'])
        self.assertFalse(T.differences(before,capture_rng()))
        self.assertEqual(receipts[0]['window'],1)
        trainer.global_step=513
        observer('committed',None)
        self.assertEqual(len(calls),1)
        trainer.global_step=512;trainer.response_records[0]['record_id']='wrong'
        with self.assertRaisesRegex(ValueError,'fitting pair'):observer('committed',None)

    def test_native_collection_reuses_two_and_generates_only_remaining126(self):
        requests=[dict(request_id=f'q{i}',event_id='event_005',request_index=i,
                       prompt_ids=[1,i+3],max_new_tokens=2048) for i in range(128)]
        write_json(self.root/'training.json',dict(train_requests=requests))
        retained=[]
        def record(request):
            return dict(record_id=request['request_id'],checkpoint_id='KPLUS4',
                prompt_ids=request['prompt_ids'],response_ids=[9],finish_reason='eos')
        for request in requests[:2]:
            path=self.root/(request['request_id']+'.json');feature=self.root/(request['request_id']+'.pt')
            write_json(path,record(request));feature.write_bytes(b'feature')
            retained.append(dict(request_id=request['request_id'],status='completed',
                record_path=str(path),record_identity=identity(path),
                feature_path=str(feature),feature_identity=identity(feature)))
        write_json(self.root/'preparation.json',dict(status='completed',datasets=dict(gsm8k=retained)))
        plan=dict(root=str(self.root),shared_feedback_preparation=str(self.root/'preparation.json'),
            target='target',worker_python='python',datasets=dict(gsm8k=dict(
                training_manifest=str(self.root/'training.json'),origin_export='KPLUS4')))
        calls=[]
        class Backend:
            loading_s=0.
            def __init__(self,*args,**kwargs):pass
            def generate(self,prompt_ids,**kwargs):
                calls.append(kwargs['request_id'])
                value=record(requests[kwargs['request_index']])
                write_json(kwargs['save_to'],value)
                return value
            def prepare_feedback(self,value,path):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'feature')
                return dict(status='prepared',eligible=True,feature_path=str(path))
            def close(self):pass
        def check(value,request,checkpoint,*,capture):
            self.assertEqual(value,record(request));self.assertEqual(checkpoint,'KPLUS4')
            self.assertTrue(capture)
        with patch('online_feedback.sglang_backend.SGLangDFlashBackend',Backend), \
             patch('experiments.full_k_gsm8k.campaign.online_record_check',check), \
             patch('experiments.ok_events_5_8.campaign.publish_checked',return_value={}), \
             patch('experiments.kplus_full_gsm8k.storage.retire_redundant_features',return_value={}), \
             patch.object(N,'require_capacity'):
            pool=N.collect_shared(plan,'gsm8k',5)
        self.assertEqual(calls,[row['request_id'] for row in requests[2:]])
        self.assertEqual([row['record_id'] for row in pool['records']],[r['request_id'] for r in requests])
        status=json.loads((N.event_root(self.root,'gsm8k',5)/'shared/COLLECTION.json').read_text())
        self.assertEqual(status['work']['completed'],dict(feedback_generations=126,teacher_preparations=126))
        self.assertEqual(status['reused_feedback'],2)

    def test_native_training_launch_is_a_separate_gpu_worker_without_retry(self):
        path=N.event_root(self.root,'gsm8k',5);path.mkdir(parents=True)
        result=dict(status='completed')
        write_json(N.result_path(self.root,'gsm8k',5,'KPLUS'),result)
        process=SimpleNamespace(pid=123,returncode=0,wait=lambda:0,poll=lambda:0)
        plan=dict(root=str(self.root),training_python='/python',training_gpu=1,
                  execution_source=dict(source_path='/code'))
        with patch.object(N.subprocess,'Popen',return_value=process) as launch,patch('experiments.role_loss_pilot.campaign.install_signal_handlers'),patch.object(N,'require_capacity'):
            self.assertEqual(N.launch_training(plan,'gsm8k',5,'KPLUS',{},None),result)
        self.assertEqual(launch.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'],'1')
        self.assertEqual(launch.call_args.args[0][-4:],['--train','gsm8k','5','KPLUS'])
        self.assertEqual(launch.call_count,1)


class StorageTests(unittest.TestCase):
    def test_peak_includes_all_exact_diagnostic_endpoints_and_atomic_overlap(self):
        plan=S.capacity_plan(2097259104,6297245019)
        self.assertEqual(plan['diagnostic_endpoint_pairs'],16)
        self.assertEqual(plan['diagnostic_endpoint_bytes'],32*2097259104)
        self.assertGreater(plan['required_initial_free_bytes'],160_000_000_000)
        self.assertEqual(plan['retained_final_boundaries'],6)
        self.assertFalse(plan['offload_assumed'])

    def test_retirement_preserves_original_or_changed_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);keep=root/'raw.pt';keep.write_bytes(b'proof')
            with self.assertRaisesRegex(ValueError,'changed'):
                S.retire_exact(root,[(keep,dict(bytes=1,sha256='x'),'wrong')],root/'receipt.json',retained={},lost='none')
            self.assertTrue(keep.exists())
            receipt=S.retire_exact(root,[(keep,identity(keep),'consumed feature')],root/'receipt.json',retained={},lost='replay retired')
            self.assertEqual(receipt['removed_bytes'],5)
            self.assertFalse(keep.exists())
            with self.assertRaisesRegex(ValueError,'retried'):
                S.retire_exact(root,[],root/'receipt.json',retained={},lost='none')


class NativeStartupTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('FSG_NATIVE_CPU_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('FSG_NATIVE_CPU_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('FSG_NATIVE_CPU_CHILD') == '1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)

    def test_actual_startup_then_event_keeps_rng_state_and_native_accounting(self):
        if os.environ.get('FSG_NATIVE_CPU_CHILD') != '1':
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', FSG_NATIVE_CPU_CHILD='1')
            result = subprocess.run([sys.executable, '-m', 'unittest',
                __name__+'.NativeStartupTinyTests.'+self._testMethodName],
                env=environment,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            return
        from test_fsg_continuation import TinyContinuationTests
        from experiments.functional_step_geometry.qualify import scientific_state
        state,kwargs = TinyContinuationTests.build_origin(self)
        trainer = T.FunctionalGeometryTrainer(self.config,self.root/'native',
            **kwargs,realization_revision=C.REALIZATION_R1)
        self.addCleanup(trainer.close)
        before = scientific_state(trainer)
        result = N.discarded_startup(trainer,self.features,self.root/'startup')
        self.assertFalse(T.differences(scientific_state(trainer),before))
        self.assertEqual(result['work']['completed'],N.STARTUP_WORK)
        self.assertEqual(trainer.global_step,512)
        self.assertFalse(trainer.response_records)
        receipt=trainer.train_event('event_005',self.features)
        self.assertEqual(receipt['optimizer_steps'],128)
        self.assertEqual(receipt['response_presentations'],256)
        trainer.verify_boundary(receipt['export_path'])
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
