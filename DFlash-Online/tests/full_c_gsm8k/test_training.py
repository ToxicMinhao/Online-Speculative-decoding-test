"""Full-horizon C metadata and real tiny native-training continuity."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from experiments.full_c_gsm8k import training as T
from experiments.full_k_gsm8k import training as native
from experiments.kplus_full_gsm8k import training as kplus
from online_feedback.rng import capture_rng
from test_kplus_full_training import plus_state
from test_full_k_training import BINDING
import test_full_k_training as native_fixtures


def allocation_summary():
    return dict(changed_positions=2, changed_blocks=1, weight_moved=.25,
        prefix_weight_before=3., prefix_weight_after=3., mass_error_abs_sum=0., delta_numerator=0.,
        mass_violations=0, bounds_violations=0, support_violations=0, nonfinite_values=0)


def candidate_state(event=0):
    state, args = plus_state(event)
    del state['algorithm_contract'][kplus.CONTRACT_KEY]
    state['algorithm_contract'][T.CONTRACT_KEY] = T.experiment_contract(BINDING)
    args['expected_contract'] = deepcopy(state['algorithm_contract'])
    for entry in state['events']:
        entry.update(method=T.ARM, dataset='gsm8k', **{T.CONTRACT_KEY:T.experiment_contract(BINDING)})
        for row in entry['records']:
            row['response_observation'].update(sampled_anchors=deepcopy(row['sampled_anchors']),
                                               prefix_allocation=allocation_summary())
    return state, args


class ContractTests(unittest.TestCase):
    def test_released_and_full_endpoint_preserve_native_horizon_without_fork(self):
        for event in (0, 1, 58):
            state, args = candidate_state(event)
            before = capture_rng()
            result = T.validate_snapshot(state, **args)
            self.assertEqual(result['global_step'],128*event)
            self.assertEqual(result['arm'],T.ARM)
            self.assertIsNone(result['continuation_rng_fork'])
            self.assertEqual(native.differences(before,capture_rng()),[])
            self.assertNotIn(kplus.CONTRACT_KEY,state['algorithm_contract'])

    def test_changed_formula_fork_or_adam_is_rejected(self):
        state, args = candidate_state(1)
        for mutate in (
            lambda s:s['algorithm_contract'][T.CONTRACT_KEY].update(continuation_rng_fork=29),
            lambda s:s['algorithm_contract'][T.CONTRACT_KEY]['allocation'].update(relative_cap=.5),
            lambda s:s['backend']['optimizer']['optimizer_state_dict']['param_groups'][0].update(lr=1.5e-4),
            lambda s:s['events'][0]['records'][0]['response_observation']['prefix_allocation'].update(bounds_violations=1),
        ):
            bad=deepcopy(state);mutate(bad)
            with self.assertRaises(ValueError):T.validate_snapshot(bad,**args)

    def test_old_native_validators_are_not_weakened(self):
        state,args=candidate_state()
        with self.assertRaises(ValueError):kplus.validate_snapshot(state,**args)
        with self.assertRaises(ValueError):native.experiment_contract(T.ARM,BINDING)

    def test_production_does_not_accept_disposable_reset_or_forced_sampling(self):
        trainer=object.__new__(T.FullCTrainer);trainer.qualification_only=False
        with self.assertRaises(ValueError):trainer.reset_qualification()
        with self.assertRaises(ValueError):trainer.run_pair([])
        with self.assertRaises(ValueError):trainer.train_event('event_001',[],presentation_schedule={})

    def test_load_isolates_optimizer_payload_and_restores_own_c_controller(self):
        trainer=object.__new__(T.FullCTrainer)
        model=torch.nn.Linear(2,1,bias=False).to(torch.bfloat16)
        backend={};state={'draft_state_dict':deepcopy(model.state_dict()),
            'backend':{'optimizer':{'counter':torch.tensor(5.)}},'global_step':128,
            'presentations':256,'events':[{'event_id':'event_001'}],'rng':capture_rng()}
        from types import SimpleNamespace
        trainer.bundle=SimpleNamespace(draft_model=model)
        trainer.backend=SimpleNamespace(load_state_dict=lambda value:backend.update(value))
        trainer.core=SimpleNamespace(_micro=0)
        installed=[];trainer._install_allocation=lambda:installed.append(True)
        trainer._load_scientific_state(state)
        backend['optimizer']['counter'].add_(1)
        self.assertEqual(state['backend']['optimizer']['counter'].item(),5)
        self.assertEqual((trainer.global_step,trainer.presentations,trainer.core._micro),(128,256,256))
        self.assertEqual(installed,[True])
        self.assertEqual(native.differences(capture_rng(),state['rng']),[])


class TinyNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('FULL_C_TINY_CPU_CHILD')=='1':
            native_fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('FULL_C_TINY_CPU_CHILD')=='1':
            native_fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('FULL_C_TINY_CPU_CHILD')=='1':native_fixtures.FullKTinyIntegrationTests.setUp(self)

    def build(self,name,**kwargs):
        trainer=T.FullCTrainer(self.config,self.root/name,binding=BINDING,
            original_config_text=self.original_config_text,expected_cuda_rng_states=0,**kwargs)
        self.addCleanup(trainer.close);return trainer

    def test_released_identity_two_events_and_own_resume_without_reseed(self):
        if os.environ.get('FULL_C_TINY_CPU_CHILD')!='1':
            root=Path(__file__).resolve().parents[2]
            env=dict(os.environ,FULL_C_TINY_CPU_CHILD='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            env['PYTHONPATH']=os.pathsep.join(str(root/p) for p in ('.','tests','training','inference'))
            result=subprocess.run([sys.executable,'-m','unittest','-v',
                'full_c_gsm8k.test_training.TinyNativeTests.'+self._testMethodName],cwd=root,env=env,
                text=True,capture_output=True,timeout=240)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            return
        from safetensors.torch import save_file
        baseline=native_fixtures.FullKTinyIntegrationTests.build(self,'original')
        released=self.root/'released';released.mkdir()
        save_file(dict(baseline.bundle.draft_model.state_dict()),released/'model.safetensors')
        initial=baseline.save_initial()['state_path'];baseline.close()
        trainer=self.build('C')
        initial_rng=capture_rng();receipt=trainer.save_initial(initial)
        self.assertTrue(receipt['original_K_initialization']['exact_released_weights_empty_adam_rng'])
        self.assertEqual(native.differences(initial_rng,capture_rng()),[])
        first=trainer.train_event('event_001',self.features)
        trainer.verify_boundary(first['export_path'])
        self.assertEqual(first['objective_work']['work']['response_forwards'],256)
        self.assertTrue(all('prefix_allocation' in r['response_observation'] for r in first['records']))
        boundary=self.root/'boundary.pt';boundary.write_bytes(trainer.latest.read_bytes())
        own=[]
        for index,path in enumerate(self.features):
            payload=torch.load(path,weights_only=True);payload['record_id']=f'own_C1_{index}'
            payload['label_binding']['record_id']=payload['record_id']
            destination=self.root/f'own_{index}.pt';torch.save(payload,destination);own.append(destination)
        second=trainer.train_event('event_002',own)
        expected=torch.load(trainer.latest,weights_only=False);trainer.close()
        restored=self.build('restored',resume_state=boundary,completed_event=1)
        repeated=restored.train_event('event_002',own)
        restored.verify_boundary(repeated['export_path'])
        actual=torch.load(restored.latest,weights_only=False)
        for key in ('draft_state_dict','backend','rng','global_step','presentations'):
            self.assertEqual(native.differences(actual[key],expected[key]),[],key)
        for key in ('record_id','pass','sampled_anchors','metrics'):
            self.assertEqual([r[key] for r in repeated['records']],[r[key] for r in second['records']],key)
        self.assertEqual((restored.global_step,restored.presentations),(256,512))
        self.assertIsNone(actual['algorithm_contract'][T.CONTRACT_KEY]['continuation_rng_fork'])
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
