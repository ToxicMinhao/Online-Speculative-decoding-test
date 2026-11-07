"""Released-loss state guards and actual tiny native full-event continuity."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from experiments.adaflash_arepo import training as T
from experiments.adaflash_c1 import training as paper
from experiments.full_k_gsm8k import training as native
from online_feedback.rng import capture_rng
from test_adaflash_c1_training import a_state
import test_full_k_training as fixtures
from test_full_k_training import BINDING


def repo_state(event=0):
    state, kwargs = a_state(event)
    state['algorithm_contract'].pop(paper.CONTRACT_KEY)
    state['algorithm_contract'][T.CONTRACT_KEY] = T.experiment_contract(BINDING)
    for item in state['events']:
        item.pop('paper_loss')
        item.update(method='AREPO',released_loss=T.adaflash_repo_contract())
    kwargs['expected_contract'] = deepcopy(state['algorithm_contract'])
    return state,kwargs


class RepoStateTests(unittest.TestCase):
    def test_complete_state_validates_without_mutation(self):
        for event in (0,1,58):
            state,kwargs = repo_state(event)
            saved = deepcopy(state)
            rng = capture_rng()
            result = T.validate_snapshot(state,**kwargs)
            self.assertEqual(result['arm'],'AREPO')
            self.assertEqual(result['global_step'],128*event)
            self.assertFalse(T.differences(state,saved))
            self.assertFalse(T.differences(rng,capture_rng()))

    def test_paper_and_k_guards_do_not_admit_repo(self):
        state,kwargs = repo_state()
        with self.assertRaises(ValueError):
            paper.validate_snapshot(state,**kwargs)
        with self.assertRaises(ValueError):
            native.validate_snapshot(state,arm='K',**kwargs)

    def test_changed_loss_is_rejected_before_native_structure(self):
        state,kwargs = repo_state()
        state['algorithm_contract'][T.CONTRACT_KEY]['loss']={'changed':True}
        with patch.object(native,'validate_snapshot') as gate, self.assertRaisesRegex(ValueError,'released'):
            T.validate_snapshot(state,**kwargs)
        gate.assert_not_called()

    def test_nonfinite_or_duplicate_response_diagnostics_fail(self):
        for value in (2.,float('nan')):
            state,kwargs = repo_state(1)
            state['events'][0]['records'][0]['response_observation']['sum_metrics']['adaflash_response_count']=value
            with self.assertRaises(ValueError):
                T.validate_snapshot(state,**kwargs)


class RepoTinyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('AREPO_TINY_CPU_CHILD')=='1':
            fixtures.FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('AREPO_TINY_CPU_CHILD')=='1':
            fixtures.FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('AREPO_TINY_CPU_CHILD')=='1':
            fixtures.FullKTinyIntegrationTests.setUp(self)

    def build_repo(self,name,**kwargs):
        config=self.root/(name+'.yaml')
        config.write_text(self.original_config_text)
        trainer=T.AdaFlashRepoTrainer(config,self.root/name,binding=BINDING,
            original_config_text=self.original_config_text,expected_cuda_rng_states=0,**kwargs)
        self.addCleanup(trainer.close)
        return trainer

    def test_fresh_initialization_complete_event_and_exact_resume(self):
        if os.environ.get('AREPO_TINY_CPU_CHILD')!='1':
            root=Path(__file__).resolve().parents[1]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',AREPO_TINY_CPU_CHILD='1')
            env['PYTHONPATH']=os.pathsep.join(str(root/p) for p in ('.','tests','training','inference'))
            result=subprocess.run([sys.executable,'-m','unittest','-v',
                'test_adaflash_arepo_training.RepoTinyTests.test_fresh_initialization_complete_event_and_exact_resume'],
                cwd=root,env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            return
        original=fixtures.FullKTinyIntegrationTests.build(self,'original')
        released=self.root/'released'
        released.mkdir()
        save_file(dict(original.bundle.draft_model.state_dict()),released/'model.safetensors')
        initial=original.save_initial()['state_path']
        original.close()
        trainer=self.build_repo('repo')
        checked=trainer.save_initial(initial)
        self.assertTrue(checked['original_K_initialization']['exact_released_weights_empty_adam_rng'])
        result=trainer.train_event('event_001',self.features)
        self.assertEqual(result['method'],'AREPO')
        self.assertNotIn('paper_loss',result)
        self.assertEqual(result['objective_work']['work']['response_forwards'],256)
        self.assertEqual(sum(r['response_observation']['sum_metrics']['adaflash_response_count'] for r in result['records']),256)
        trainer.verify_boundary(result['export_path'])
        state=torch.load(trainer.latest,weights_only=False)
        retained=self.root/'retained.pt'
        retained.write_bytes(trainer.latest.read_bytes())
        trainer.close()
        resumed=self.build_repo('resumed',resume_state=retained,completed_event=1)
        self.assertTrue(T.verify_live(resumed,state)['exact_resume'])
        self.assertEqual(resumed.global_step,128)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main()
