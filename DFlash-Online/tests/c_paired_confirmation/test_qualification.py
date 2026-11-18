from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from experiments.c_paired_confirmation import qualification as Q, training as T
from experiments.unchanged_c_mathqa_transfer.runtime import WorkLedger, initialize_work
from online_feedback.rng import capture_rng, restore_rng
import unchanged_c_mathqa_transfer.test_qualification as earlier


class TinyTrainer(earlier.TinyTrainer):
    def reset_qualification(self,**kwargs):
        result=super().reset_qualification(**kwargs);self._fork_applied=False;return result

    def apply_continuation_fork(self):
        if self._fork_applied:return dict(reseeded=False,status='already_applied')
        state=capture_rng();state['torch']=torch.Generator().manual_seed(29).get_state();restore_rng(state)
        self._fork_applied=True
        return dict(reseeded=True,status='passed',seed=29)

    def restore_qualification(self,*args,**kwargs):
        result=super().restore_qualification(*args,**kwargs);self._fork_applied=True;return result


class QualificationTests(unittest.TestCase):
    def setUp(self):
        earlier.QualificationTests.setUp(self)
        self.plan['datasets']={'mathqa':{key:self.plan[key] for key in ('configs','origin_state','origin_export','origin_config')}}
        self.plan['datasets']['mathqa'].update(common_feature_paths=['unused']*4,origin_state_identity={'sha256':'a'*64,'bytes':1})
        TinyTrainer.corrupt_cap_zero=False;TinyTrainer.fail_adam=False
        self.ledger=WorkLedger(self.root,'cpu_qualification')

    def execute(self,startup_override=None):
        def startup(trainer,features,output):
            for _ in range(2):
                loss,_=trainer.strategy.forward_loss(None);trainer.backend.backward(loss/2)
            trainer.backend.optimizer.zero_grad();return dict(wall_s=.1,optimizer_steps=0)
        def export(_state,_config,output,**kwargs):
            folder=Path(output);folder.mkdir();(folder/'model.safetensors').write_bytes(b'disposable fixture')
        def reconstructed(_state,_config,_expected,output,**kwargs):
            export(_state,_config,output)
            return {'status':'passed','optimizer_steps':0}
        with ExitStack() as stack:
            stack.enter_context(patch.object(T,'PairedContinuationTrainer',TinyTrainer))
            stack.enter_context(patch('online_feedback.training.SpecForgeEventTrainer._prepare_event',return_value=self.features))
            stack.enter_context(patch('experiments.prefix_guard.qualify.pin_retained_graph_runtime',return_value={'status':'passed'}))
            stack.enter_context(patch('experiments.functional_step_geometry.native_pilot.discarded_startup',side_effect=startup_override or startup))
            stack.enter_context(patch('specforge.export.to_hf.export_to_hf',side_effect=export))
            stack.enter_context(patch('experiments.c_paired_confirmation.export.qualify_cpu_export',side_effect=reconstructed))
            return Q.run_training_qualification(self.plan,'mathqa','A',ledger=self.ledger)

    def test_six_steps_native_parity_active_reload_and_exact_work(self):
        result=self.execute()
        self.assertEqual(result['status'],'passed')
        self.assertEqual([r['case'] for r in result['cases']],list(Q.CASES))
        self.assertTrue(all(r['checks']=='passed' for r in result['cases']))
        self.assertTrue(result['cpu_reconstruction_verified'])
        self.assertEqual(result['cpu_export'],result['cpu_export_qualification'])
        self.assertEqual(len(result['disposable_payload_files']),3)
        self.assertTrue(all(Path(path).is_file() for path in result['disposable_payload_files']))
        self.assertTrue(all(row['pair_wall_excluding_observers_s']>0 for row in result['cases']))
        self.assertGreaterEqual(result['loading_s'],0)
        self.assertEqual(result['startup_s'],.1)
        for key in ('draft_config','embedding_source','embedding_key'):
            self.assertIn(key,result)
        self.assertEqual(result['origin_before'],result['origin_after'])
        work=json.loads((self.root/'WORK.json').read_text())
        self.assertEqual(work['completed']['adam'],6)
        self.assertEqual(work['completed']['response_forward'],14)
        self.assertEqual(work['completed']['response_backward'],14)
        self.assertEqual(work['completed']['training_initialization'],1)
        self.assertEqual(result['sampling']['bare_native']['order'],result['sampling']['active']['order'])

    def test_failure_saved_before_raise_and_no_later_model_work(self):
        TinyTrainer.corrupt_cap_zero=True
        with self.assertRaisesRegex(ValueError,'cap_zero native equivalence'):self.execute()
        result=json.loads((self.root/'qualification/mathqa_training_A/DETAILS.json').read_text())
        self.assertEqual(result['status'],'failed');self.assertEqual(len(result['cases']),3)
        self.assertIn('preclip',result['cases'][-1]);self.assertIn('endpoint',result['cases'][-1])
        self.assertEqual(json.loads((self.root/'WORK.json').read_text())['completed']['adam'],3)
        with self.assertRaises(FileExistsError):self.execute()

if __name__=='__main__':unittest.main()
