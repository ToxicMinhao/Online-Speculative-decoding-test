"""Bounded qualification preserves failures, startup work and exact restoration."""
from contextlib import ExitStack
from pathlib import Path
import json
import unittest
from unittest.mock import patch

import torch
from experiments.full_c_gsm8k import qualification as Q, training as T
from experiments.unchanged_c_mathqa_transfer.runtime import WorkLedger
import unchanged_c_mathqa_transfer.test_qualification as earlier


class TinyTrainer(earlier.TinyTrainer):
    def save_initial(self, comparator_path):
        destination=self.output_root/'initial_state.pt'
        torch.save(self.origin,destination)
        return dict(status='passed',state_path=str(destination),export_exact=True,
                    original_K_initialization={'exact_released_weights_empty_adam_rng':True})


class QualificationTests(unittest.TestCase):
    def setUp(self):
        earlier.QualificationTests.setUp(self)
        source=torch.load(self.root/'origin.pt',weights_only=False)
        source['global_step']=source['presentations']=0
        source['backend']['optimizer']['optimizer_state_dict']['state'].clear()
        torch.save(source,self.root/'origin.pt')
        self.plan.update(config_path=self.plan['configs']['C'],original_config=self.plan['origin_config'],
            initial_witness=self.plan['origin_state'],qualification_feature_paths=[f'feature{i}' for i in range(4)])
        self.ledger=WorkLedger(self.root,'cpu_qualification')
        TinyTrainer.corrupt_cap_zero=False;TinyTrainer.fail_adam=False

    def execute(self,startup_override=None):
        def build(config,output,**options):
            return TinyTrainer(config,output,origin_state=self.plan['initial_witness'],**options)
        def startup(trainer,features,output):
            for _ in range(2):
                loss,_=trainer.strategy.forward_loss(None);trainer.backend.backward(loss/2)
            trainer.backend.optimizer.zero_grad();return dict(wall_s=.1,optimizer_steps=0)
        def export(_state,_config,output,**kwargs):
            folder=Path(output);folder.mkdir();(folder/'model.safetensors').write_bytes(b'CPU fixture')
        def reconstruct(_state,_config,_expected,output,**kwargs):
            export(_state,_config,output)
            return dict(status='passed',optimizer_steps=0,exact_weights_file_identity=True)
        with ExitStack() as stack:
            stack.enter_context(patch.object(T,'FullCTrainer',side_effect=build))
            stack.enter_context(patch('online_feedback.training.SpecForgeEventTrainer._prepare_event',return_value=self.features))
            stack.enter_context(patch('experiments.prefix_guard.qualify.pin_retained_graph_runtime',return_value={'status':'passed'}))
            stack.enter_context(patch('experiments.functional_step_geometry.native_pilot.discarded_startup',side_effect=startup_override or startup))
            stack.enter_context(patch('specforge.export.to_hf.export_to_hf',side_effect=export))
            stack.enter_context(patch('experiments.c_paired_confirmation.export.qualify_cpu_export',side_effect=reconstruct))
            return Q.run_training_qualification(self.plan,'B',ledger=self.ledger)

    def test_six_steps_native_capzero_active_and_own_resume_no_fork(self):
        result=self.execute()
        self.assertEqual(result['status'],'passed')
        self.assertEqual([r['case'] for r in result['cases']],list(Q.CASES))
        self.assertTrue(all(r['checks']=='passed' for r in result['cases']))
        self.assertEqual(result['cases'][-1]['endpoint']['global_step'],2)
        self.assertEqual(result['origin_before'],result['origin_after'])
        self.assertIsNone(result['stochastic_consumers']['explicit_continuation_seed'])
        self.assertTrue(all(not row['reseeded'] for row in result['sampling'].values()))
        self.assertTrue(result['cpu_reconstruction_verified'])
        self.assertTrue(result['cpu_export']['exact_weights_file_identity'])
        self.assertEqual(len(result['disposable_payload_files']),4)
        self.assertTrue(all(Path(path).is_file() for path in result['disposable_payload_files']))
        work=json.loads((self.root/'WORK.json').read_text())
        self.assertEqual(work['completed']['adam'],6)
        self.assertEqual(work['completed']['response_forward'],14)
        self.assertEqual(work['completed']['response_backward'],14)
        self.assertEqual(work['completed']['training_initialization'],1)

    def test_failed_parity_is_saved_before_raise_without_later_work(self):
        TinyTrainer.corrupt_cap_zero=True
        with self.assertRaisesRegex(ValueError,'cap_zero native equivalence'):self.execute()
        result=json.loads((self.root/'qualification/training_B/DETAILS.json').read_text())
        self.assertEqual(result['status'],'failed');self.assertEqual(len(result['cases']),3)
        self.assertIn('preclip',result['cases'][-1]);self.assertIn('endpoint',result['cases'][-1])
        self.assertEqual(json.loads((self.root/'WORK.json').read_text())['completed']['adam'],3)
        with self.assertRaises(FileExistsError):self.execute()

    def test_startup_failure_records_entered_work_and_no_optimizer_step(self):
        def broken(trainer,features,output):
            trainer.strategy.forward_loss(None)
            raise RuntimeError('startup fails before backward')
        with self.assertRaisesRegex(RuntimeError,'startup fails'):self.execute(startup_override=broken)
        result=json.loads((self.root/'qualification/training_B/DETAILS.json').read_text())
        self.assertEqual(result['status'],'failed');self.assertEqual(result['cases'],[])
        counts=result['operation_counts']['startup']
        self.assertEqual(counts['response_forward_completed'],1)
        self.assertNotIn('response_backward_attempted',counts)
        self.assertNotIn('adam_attempted',counts)

    def test_duplicate_or_unsealed_fixture_count_stops_before_loading(self):
        self.plan['qualification_feature_paths']=['same']*4
        with self.assertRaisesRegex(ValueError,'four prespecified'):self.execute()
        self.assertFalse((self.root/'qualification/training_B/DETAILS.json').exists())


if __name__=='__main__':unittest.main()
