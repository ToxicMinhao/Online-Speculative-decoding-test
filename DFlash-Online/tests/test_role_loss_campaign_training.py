"""Scientific state isolation and native full-event pilot boundaries."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from experiments.role_loss_pilot import training as T
from experiments.kplus_full_gsm8k import training as old_plus
from experiments.full_k_gsm8k import training as native
from online_feedback.rng import capture_rng
from test_kplus_full_training import plus_state
from test_full_k_training import BINDING
import test_full_k_training as fixtures


def pilot_state(event=0,arm='ROLE',dataset='mathqa'):
    state,kwargs=plus_state(event)
    state['algorithm_contract'].pop(old_plus.CONTRACT_KEY)
    state['algorithm_contract'][T.CONTRACT_KEY]=T.experiment_contract(BINDING,arm,dataset)
    for row in state['events']:
        row.pop('redistribution')
        row.update(method=arm,dataset=dataset,pilot_loss=T.loss_contract(arm))
    kwargs.update(arm=arm,dataset=dataset,expected_contract=deepcopy(state['algorithm_contract']))
    return state,kwargs


class PilotStateTests(unittest.TestCase):
    def test_all_cells_validate_read_only_with_unchanged_native_horizon(self):
        for arm in T.ARMS:
            for dataset in T.DATASETS:
                for event in (0,1,4):
                    state,kwargs=pilot_state(event,arm,dataset);saved=deepcopy(state);rng=capture_rng()
                    result=T.validate_snapshot(state,**kwargs)
                    self.assertEqual(result['global_step'],event*128)
                    self.assertEqual(state['horizon'],7424)
                    self.assertFalse(T.differences(state,saved))
                    self.assertFalse(T.differences(rng,capture_rng()))

    def test_cross_arm_cross_dataset_and_fifth_event_fail_before_native(self):
        state,kwargs=pilot_state(1)
        for edit in ({'arm':'MIX'},{'dataset':'gsm8k'},{'completed_event':5}):
            with patch.object(native,'validate_snapshot') as gate,self.assertRaises(ValueError):
                T.validate_snapshot(state,**dict(kwargs,**edit))
            gate.assert_not_called()

    def test_historical_guard_does_not_accept_pilot(self):
        state,kwargs=pilot_state()
        kwargs.pop('arm');kwargs.pop('dataset')
        with self.assertRaises(ValueError):old_plus.validate_snapshot(state,**kwargs)
        with self.assertRaises(ValueError):native.validate_snapshot(state,arm='K',**kwargs)

    def test_nonfinite_diagnostics_and_lost_mass_fail(self):
        for field,value in [('kplus_response_count',2),('kplus_W1',float('nan')),
                            ('kplus_first_error_weight_after',9)]:
            state,kwargs=pilot_state(1)
            state['events'][0]['records'][0]['response_observation']['sum_metrics'][field]=value
            with self.assertRaises(ValueError):T.validate_snapshot(state,**kwargs)


class PilotTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('ROLELOSS_TINY_CHILD')=='1': fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls):
        if os.environ.get('ROLELOSS_TINY_CHILD')=='1': fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)
    def setUp(self):
        if os.environ.get('ROLELOSS_TINY_CHILD')=='1': fixtures.FullKTinyIntegrationTests.setUp(self)
    def test_native_initial_event_and_resume(self):
        if os.environ.get('ROLELOSS_TINY_CHILD')!='1':
            root=Path(__file__).resolve().parents[1]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',ROLELOSS_TINY_CHILD='1')
            env['PYTHONPATH']=os.pathsep.join(str(root/p) for p in ('.','tests','training','inference'))
            result=subprocess.run([sys.executable,'-m','unittest','-v',__name__+'.PilotTinyTests.'+self._testMethodName],
                cwd=root,env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);return
        original=fixtures.FullKTinyIntegrationTests.build(self,'original')
        released=self.root/'released';released.mkdir();save_file(dict(original.bundle.draft_model.state_dict()),released/'model.safetensors')
        initial=original.save_initial()['state_path'];original.close()
        for arm in T.ARMS:
            config=self.root/(arm+'.yaml');config.write_text(self.original_config_text)
            kwargs=dict(binding=BINDING,arm=arm,dataset='mathqa',original_config_text=self.original_config_text,
                expected_cuda_rng_states=0)
            trainer=T.RoleLossTrainer(config,self.root/arm,**kwargs);self.addCleanup(trainer.close)
            trainer.save_initial(initial);result=trainer.train_event('event_001',self.features)
            self.assertEqual(result['method'],arm);self.assertEqual(result['dataset'],'mathqa')
            trainer.verify_boundary(result['export_path'])
            self.assertEqual(len(trainer.window_records),128)
            for row in trainer.response_records:
                self.assertIn('role_loss_full_denominator',row['sum_metrics'])
                self.assertIn('role_loss_prefix_weight',row['sum_metrics'])
            retained=self.root/(arm+'_state.pt');retained.write_bytes(trainer.latest.read_bytes())
            trainer.close()
            resumed=T.RoleLossTrainer(config,self.root/(arm+'_resumed'),resume_state=retained,completed_event=1,**kwargs)
            self.addCleanup(resumed.close)
            self.assertEqual(resumed.global_step,128);resumed.close()
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
