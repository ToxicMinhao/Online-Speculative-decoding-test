"""Exact disposable ceilings, retained failure evidence and actual native updates."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from experiments.role_loss_pilot import qualify as Q
import test_adaflash_arepo_qualify as retained_fixtures


class BoundedTests(unittest.TestCase):
    def test_no_startup_work_and_reject_before_seventh_step(self):
        self.assertEqual(Q.BUDGET['startup_response_forwards'],0)
        with tempfile.TemporaryDirectory() as temp:
            log=Q.BoundedLog(Path(temp))
            for _ in range(6):
                with log.work('adam_step'):pass
            with self.assertRaises(RuntimeError):
                with log.work('adam_step'):self.fail('Unbounded step ran')
            self.assertEqual(log.counts['adam_step_attempted'],6)
    def test_invalid_inputs_preserve_zero_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=root/'prepared.json'
            p.write_text(json.dumps(dict(status='completed',feature_paths=[],records=[])))
            with self.assertRaises(ValueError):Q.run('unused','unused',p,root/'qualification')
            result=json.loads((root/'qualification/QUALIFICATION.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertEqual(result['work'],{})
            self.assertEqual(result['uncompleted_cases'],6)


class NativeQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('ROLELOSS_Q_CHILD')=='1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls):
        if os.environ.get('ROLELOSS_Q_CHILD')=='1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)
    def setUp(self):
        if os.environ.get('ROLELOSS_Q_CHILD')=='1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)
    def child(self):
        if os.environ.get('ROLELOSS_Q_CHILD')=='1':return False
        root=Path(__file__).resolve().parents[1]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',ROLELOSS_Q_CHILD='1')
        env['PYTHONPATH']=os.pathsep.join(str(root/p) for p in ('.','tests','training','inference'))
        result=subprocess.run([sys.executable,'-m','unittest','-v',__name__+'.NativeQualificationTests.'+self._testMethodName],
            cwd=root,env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr);return True
    def test_six_updates_twelve_presentations_exact_restore(self):
        if self.child():return
        initial,path,prepared=retained_fixtures.TinyQualificationTests.saved_inputs(self)
        with patch.object(Q.torch.cuda,'reset_peak_memory_stats'):
            result=Q.run(self.config,path,prepared,self.root/'qualification',execution_commit='a'*40)
        self.assertEqual(result['status'],'passed');self.assertEqual(len(result['cases']),6)
        for kind,n in (('adam_step',6),('training_response_forward',12),('core_backward',12)):
            self.assertEqual(result['work'][kind+'_attempted'],n)
            self.assertEqual(result['work'][kind+'_completed'],n)
        self.assertEqual(result['work']['response_model_forwards'],12)
        self.assertTrue(all(c['decoded_state_exact'] for c in result['cases']))
        self.assertEqual(list((self.root/'qualification').rglob('*.pt')),[])
        self.assertFalse(Q.differences(initial,torch.load(path,weights_only=False)))
        self.assertFalse(torch.cuda.is_initialized())
    def test_nonfinite_first_backward_stops_before_adam(self):
        if self.child():return
        initial,path,prepared=retained_fixtures.TinyQualificationTests.saved_inputs(self);factory=Q.SpecForgeEventTrainer
        def poison(*args,**kwargs):
            trainer=factory(*args,**kwargs)
            p=next(p for p in trainer.bundle.draft_model.parameters() if p.requires_grad)
            p.register_hook(lambda grad:torch.full_like(grad,float('nan')))
            return trainer
        with patch.object(Q.torch.cuda,'reset_peak_memory_stats'),patch.object(Q,'SpecForgeEventTrainer',side_effect=poison):
            with self.assertRaisesRegex(ValueError,'no optimizer'):
                Q.run(self.config,path,prepared,self.root/'qualification')
        result=json.loads((self.root/'qualification/QUALIFICATION.json').read_text())
        self.assertEqual(result['work'].get('adam_step_attempted',0),0)
        self.assertEqual(result['work']['core_backward_completed'],1)
        self.assertEqual(result['uncompleted_cases'],6)
        self.assertFalse(Q.differences(initial,torch.load(path,weights_only=False)))


if __name__=='__main__':unittest.main()
