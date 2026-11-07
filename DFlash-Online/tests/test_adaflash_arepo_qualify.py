"""A_repo qualification accounting, nonfinite evidence and native CPU path."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.adaflash_arepo import qualify as Q
from experiments.prefix_ranking.qualify import WorkLog, cpu_copy
from online_feedback.rng import capture_rng
from online_feedback.storage import save_torch
from specforge.core.chunking import checkpointed_chunk_reduce


class QualificationUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_fixed_fixtures_cover_caps_and_tiny_probabilities_without_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            log=WorkLog(Path(directory))
            rows=Q.fixed_loss_checks(log,'cpu')
            self.assertEqual(len(rows),3)
            self.assertTrue(rows[0]['clipped'])
            self.assertFalse(rows[1]['clipped'])
            self.assertGreater(rows[2]['smallest_probability'],0)
            self.assertLess(rows[2]['smallest_probability'],1e-30)
            self.assertEqual(log.counts['fixture_loss_forward_completed'],6)
            self.assertEqual(log.counts['fixture_loss_backward_completed'],6)
            self.assertEqual(log.counts['adam_step_completed'],0)

    def test_nonfinite_gradient_evidence_precedes_any_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            p=torch.nn.Parameter(torch.zeros(4,dtype=torch.bfloat16))
            p.grad=torch.tensor([0.,float('nan'),float('inf'),-float('inf')],dtype=torch.bfloat16)
            with self.assertRaisesRegex(ValueError,'no optimizer'):
                Q.finite_gradient_check({'draft.weight':p},root,0)
            evidence=json.loads((root/'NONFINITE_GRADIENTS.json').read_text())
            self.assertFalse(evidence['optimizer_called'])
            self.assertEqual(evidence['action'],'stop_without_stabilization')
            self.assertEqual(evidence['parameters'][0]['nonfinite'],3)
            self.assertEqual(evidence['parameters'][0]['sample_values'],['nan','inf','-inf'])

    def test_missing_gradient_and_finite_gradient_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=torch.nn.Parameter(torch.zeros(1))
            with self.assertRaises(ValueError):Q.finite_gradient_check({'weight':p},root,0)
            self.assertTrue(json.loads((root/'NONFINITE_GRADIENTS.json').read_text())['parameters'][0]['missing'])
            p.grad=torch.ones(1);Q.finite_gradient_check({'weight':p},root,1)

    def test_zero_probability_inputs_saved_and_checkpoint_recompute_not_recounted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);observer=Q.QualificationLoss('literal_source',root)
            logits=torch.tensor([[[[0.,-100.,-200.],[.1,.2,.3]],[[.2,.3,.4],[.1,-.2,.3]]]],dtype=torch.bfloat16,requires_grad=True)
            teacher=torch.zeros_like(logits);weights=torch.ones(1,2,2)
            labels=teacher.argmax(-1)
            observer.prepare_response(weights)
            terms=checkpointed_chunk_reduce(observer.chunk_terms,logits,teacher,labels,weights,chunk_size=1,dim=1,checkpoint=True)
            loss=observer.reduce_response(terms,weights.sum()+1e-6)
            observer.response_metrics(terms,weights.sum()+1e-6)
            loss.backward()
            self.assertEqual(len(observer.responses),1)
            self.assertEqual(len(observer.responses[0]['chunks']),2)
            self.assertEqual(observer.chunks,[])
            saved=torch.load(root/'ZERO_PROBABILITY_INPUTS.pt',weights_only=True)
            self.assertEqual(tuple(saved['logits'].shape),(1,3))
            self.assertEqual(float(saved['full_response_denominator']),float(weights.sum()+1e-6))
            evidence=json.loads((root/'RESPONSES.json').read_text())
            self.assertEqual(evidence[0]['chunks'][0]['zero_rows'],1)
            self.assertEqual(evidence[0]['chunks'][0]['zero_probabilities'],2)
            self.assertTrue(torch.isnan(logits.grad[0,0,0]).all())

    def test_invalid_plan_preserves_failure_and_zero_executed_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prepared=root/'prepared.json'
            prepared.write_text(json.dumps(dict(status='completed',feature_paths=[],records=[])))
            with self.assertRaisesRegex(ValueError,'six retained'):
                Q.run('unused','unused',prepared,root/'qualification')
            result=json.loads((root/'qualification/RESULT.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertFalse(result['main_training_authorized'])
            self.assertEqual(result['uncompleted_branches'],6)
            self.assertEqual(result['work'],{})
            self.assertEqual(result['budget']['maximum_optimizer_steps'],6)


class TinyQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('AREPO_QUAL_CPU_CHILD')=='1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        if os.environ.get('AREPO_QUAL_CPU_CHILD')=='1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.tearDownClass.__func__(cls)

    def setUp(self):
        if os.environ.get('AREPO_QUAL_CPU_CHILD')=='1':
            from test_full_k_training import FullKTinyIntegrationTests
            FullKTinyIntegrationTests.setUp(self)

    def child_process(self, name):
        if os.environ.get('AREPO_QUAL_CPU_CHILD')!='1':
            root=Path(__file__).resolve().parents[1]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',AREPO_QUAL_CPU_CHILD='1')
            env['PYTHONPATH']=os.pathsep.join(str(root/p) for p in ('.','tests','training','inference'))
            result=subprocess.run([sys.executable,'-m','unittest','-v',
                'test_adaflash_arepo_qualify.TinyQualificationTests.'+name],
                cwd=root,env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            return True
        return False

    def saved_inputs(self):
        from safetensors.torch import save_file
        source=Q.SpecForgeEventTrainer(self.config,self.root/'source',objective='osd')
        initial=dict(draft_state_dict=cpu_copy(dict(source.bundle.draft_model.state_dict())),
            backend=cpu_copy(dict(optimizer=source.backend.optimizer.state_dict())),rng=capture_rng(),
            global_step=0,presentations=0,events=[],config_text=self.config.read_text())
        source.close();initial_path=self.root/'initial.pt';save_torch(initial_path,initial)
        released=self.root/'released';released.mkdir(exist_ok=True)
        save_file(initial['draft_state_dict'],released/'model.safetensors')
        prepared=dict(status='completed',feature_paths=[str(p) for p in self.features[:6]],records=[])
        for path in prepared['feature_paths']:
            raw=torch.load(path,weights_only=True)
            anchors=(raw['loss_mask'][0]>0).nonzero().flatten().tolist()[:2]
            prepared['records'].append(dict(record_id=raw['record_id'],**{'pass':0},
                sampled_anchors=[dict(positions=[anchors],keep_mask=[[True,True]])]))
        prepared_path=self.root/'prepared.json';prepared_path.write_text(json.dumps(prepared))
        return initial,initial_path,prepared_path

    def test_six_native_steps_are_exact_and_stay_disposable(self):
        if self.child_process(self._testMethodName):return
        initial,initial_path,prepared_path=self.saved_inputs()
        native_checks=Q.fixed_loss_checks
        native_factory=Q.SpecForgeEventTrainer
        runtimes=[]
        def factory(*args,**kwargs):
            instance=native_factory(*args,**kwargs);runtimes.append(instance);return instance
        with patch.object(Q,'fixed_loss_checks',side_effect=lambda log,device:native_checks(log,'cpu')), \
                patch.object(Q.torch.cuda,'reset_peak_memory_stats'), \
                patch.object(Q,'SpecForgeEventTrainer',side_effect=factory):
            result=Q.run(self.config,initial_path,prepared_path,self.root/'qualification')
        self.assertEqual(result['status'],'passed')
        self.assertEqual(len(runtimes),7)
        self.assertTrue(all(runtime.closed for runtime in runtimes))
        self.assertEqual(result['work']['adam_step_completed'],6)
        self.assertEqual(result['work']['core_backward_completed'],12)
        self.assertEqual(result['work']['response_model_forwards'],14)
        self.assertEqual(result['work']['startup_backward_completed'],2)
        self.assertEqual(result['work']['fixture_loss_backward_completed'],6)
        self.assertEqual(result['uncompleted_branches'],0)
        self.assertTrue(result['main_training_authorized'])
        for row in result['comparisons']:
            self.assertEqual(row['gradients']['global_statistics']['different'],0)
            self.assertEqual(row['displacements']['global_statistics']['different'],0)
            self.assertFalse(row['optimizer_differences'])
            self.assertFalse(row['rng_differences'])
            self.assertFalse(row['response_differences'])
        self.assertEqual(list((self.root/'qualification').rglob('*.pt')),[])
        self.assertFalse(Q.differences(initial,torch.load(initial_path,weights_only=False)))
        self.assertFalse(torch.cuda.is_initialized())

    def test_first_nonfinite_backward_stops_before_any_adam_and_preserves_evidence(self):
        if self.child_process(self._testMethodName):return
        initial,initial_path,prepared_path=self.saved_inputs()
        native_checks=Q.fixed_loss_checks
        native_factory=Q.SpecForgeEventTrainer
        runtimes=[]
        def factory(*args,**kwargs):
            instance=native_factory(*args,**kwargs)
            runtimes.append(instance)
            if len(runtimes)==2:  # disposable startup is separate
                parameter=next(p for p in instance.bundle.draft_model.parameters() if p.requires_grad)
                parameter.register_hook(lambda grad:torch.full_like(grad,float('nan')))
            return instance
        from online_feedback.optimizer import DirectBF16AdamW
        with patch.object(Q,'fixed_loss_checks',side_effect=lambda log,device:native_checks(log,'cpu')), \
                patch.object(Q.torch.cuda,'reset_peak_memory_stats'), \
                patch.object(Q,'SpecForgeEventTrainer',side_effect=factory), \
                patch.object(DirectBF16AdamW,'step',side_effect=AssertionError('optimizer must not run')):
            with self.assertRaisesRegex(ValueError,'no optimizer'):
                Q.run(self.config,initial_path,prepared_path,self.root/'qualification')
        result=json.loads((self.root/'qualification/RESULT.json').read_text())
        evidence=json.loads((self.root/'qualification/batch_01_literal_source/NONFINITE_GRADIENTS.json').read_text())
        self.assertEqual(result['status'],'failed')
        self.assertFalse(result['main_training_authorized'])
        self.assertEqual(result['uncompleted_branches'],6)
        self.assertEqual(result['work']['core_backward_completed'],1)
        self.assertEqual(result['work'].get('adam_step_attempted',0),0)
        self.assertEqual(result['work']['startup_backward_completed'],2)
        self.assertFalse(evidence['optimizer_called'])
        self.assertTrue(all(runtime.closed for runtime in runtimes))
        self.assertFalse(Q.differences(initial,torch.load(initial_path,weights_only=False)))
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
