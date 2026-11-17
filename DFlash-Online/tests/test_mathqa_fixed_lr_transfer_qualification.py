"""CPU-only qualification orchestration, durable failures and shared accounting."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.mathqa_fixed_lr_transfer import qualification as Q
from experiments.mathqa_fixed_lr_transfer import training as T
from experiments.mathqa_fixed_lr_transfer.runtime import initialize_work, read
from experiments.functional_step_geometry.qualify import restore_runtime
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng, restore_rng


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor([.125,-.125],dtype=torch.bfloat16))
        self.register_buffer('cache_marker',torch.tensor([0.]))
        self._cached_seq_len=0
        self.costs={}

    def begin_event(self):
        self.costs.clear()


class TinyTrainer:
    instances=[]
    corrupt_factor_one=False
    fail_adam=False

    def __init__(self,config,output,**options):
        self.output_root=Path(output);self.events_root=self.output_root/'events'
        self.events_root.mkdir(parents=True)
        self.bundle=SimpleNamespace(model=TinyModel());self.bundle.draft_model=self.bundle.model
        optimizer=DirectBF16AdamW(self.bundle.model)
        self.backend=SimpleNamespace(optimizer=optimizer,backward=self.backward)
        self.core=SimpleNamespace(_micro=1024,accumulation_remainder=0)
        self.cfg=SimpleNamespace(model=SimpleNamespace(draft_model_config=str(config),target_model_path='unused',embedding_key='unused'))
        self.origin=torch.load(options['origin_state'],weights_only=False)
        self.response_records,self.window_records=[],[]
        self.work,self.failed,self.closed={},False,False
        self.step_observer=None;self.arm='N';self._presentation_schedule=None
        self._ordinary_forward=self.forward
        self._native_forward=self.checked_forward
        self.strategy=SimpleNamespace(forward_loss=lambda feature:self._native_forward(feature))
        self.reset_qualification()
        self.instances.append(self)

    def state(self):
        return deepcopy(dict(draft_state_dict={name:p for name,p in self.bundle.model.named_parameters()},
            backend={'optimizer':self.backend.optimizer.state_dict()},rng=capture_rng(),
            global_step=self.global_step,presentations=self.presentations))

    def load(self,state):
        self.bundle.model.load_state_dict(state['draft_state_dict'],strict=False)
        self.backend.optimizer.load_state_dict(deepcopy(state['backend']['optimizer']))
        self.global_step,self.presentations=state['global_step'],state['presentations']
        self.core._micro=self.presentations;restore_rng(state['rng'])

    def reset_qualification(self,*,arm=None,runtime=None):
        self.arm=arm or self.arm
        self.backend.optimizer.zero_grad()
        self.backend.optimizer.recipe['optimizer_options']['lr']=Q.arm_lr(self.arm)
        own=deepcopy(self.origin)
        own['backend']['optimizer']['recipe']['optimizer_options']['lr']=Q.arm_lr(self.arm)
        own['backend']['optimizer']['optimizer_state_dict']['param_groups'][0]['lr']=Q.arm_lr(self.arm)
        self.load(own);self.response_records.clear();self.window_records.clear()
        if runtime is not None:restore_runtime(self.bundle.model,runtime)
        return dict(status='passed',arm=self.arm)

    def checked_forward(self,feature):
        loss,row=self.forward(feature)
        if self.corrupt_factor_one and getattr(self,'_active_event',None)=='factor_one':
            loss=loss*2
        return loss,row

    def forward(self,feature):
        if feature is None:return self.bundle.model.weight.sum(),None
        draw=dict(position=int(torch.randint(100,(1,)).item()))
        loss=(self.bundle.model.weight*self.bundle.model.weight.new_tensor([.5,-.25])).sum()
        self.bundle.model.cache_marker.add_(1);self.bundle.model._cached_seq_len+=1
        self.work['response_forwards']=self.work.get('response_forwards',0)+1
        observation=dict(loss=float(loss.detach()),sum_metrics={'weight':2.},ratio_metrics={})
        row=dict(record_id=feature['record_id'],**{'pass':0},sampled_anchors=[draw],
                 response_observation=observation,metrics={'loss':float(loss.detach())})
        self.response_records.append(observation)
        return loss,row

    def backward(self,loss,**unused):
        loss.backward()

    def run_pair(self,pair,pass_index,ordinal):
        rows=[]
        for feature in pair:
            loss,row=self.strategy.forward_loss(feature)
            self.backend.backward(loss/2);rows.append(row)
            self.presentations+=1;self.core._micro+=1
        self.step_observer(self)
        if self.fail_adam:
            # Preserve the ledger wrapper by making its original Adam fail at
            # the optimizer's lower-level implementation rather than bypassing it.
            with patch.object(torch.optim.AdamW,'_init_group',side_effect=RuntimeError('injected Adam failure')):
                self.backend.optimizer.step()
        else:grad_norm=self.backend.optimizer.step()
        self.global_step+=1
        self.window_records.append(dict(gradient_norm=float(grad_norm)))
        return rows

    def save_qualification(self,path):
        torch.save(self.state(),path)
        return dict(global_step=self.global_step,presentations=self.presentations)

    def restore_qualification(self,path,*,runtime):
        self.load(torch.load(path,weights_only=False));restore_runtime(self.bundle.model,runtime)
        return dict(status='passed',exact_scientific_state=True)

    def close(self):
        self.closed=True


class QualificationTests(unittest.TestCase):
    def setUp(self):
        before=capture_rng();self.addCleanup(lambda:restore_rng(before))
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);initialize_work(self.root)
        TinyTrainer.instances.clear();TinyTrainer.corrupt_factor_one=False;TinyTrainer.fail_adam=False
        model=TinyModel();optimizer=DirectBF16AdamW(model)
        saved=optimizer.state_dict()
        for index,p in zip(saved['optimizer_state_dict']['param_groups'][0]['params'],model.parameters()):
            saved['optimizer_state_dict']['state'][index]=dict(step=torch.tensor(512.),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        source=dict(draft_state_dict={name:p.detach().clone() for name,p in model.named_parameters()},
                    backend={'optimizer':saved},rng=capture_rng(),global_step=512,presentations=1024)
        torch.save(source,self.root/'origin.pt')
        (self.root/'config.yaml').write_text('CPU orchestration fixture\n')
        self.features=[dict(record_id=f'r{i}',path=f'feature{i}') for i in range(4)]
        self.plan=dict(root=str(self.root),binding='b'*64,execution_commit='fixture',devices={'A':'GPU-A','B':'GPU-B'},
            configs={'N':str(self.root/'config.yaml')},origin_state=str(self.root/'origin.pt'),
            origin_export='unused',origin_config=str(self.root/'config.yaml'))

    def execute(self,slot='A',startup_override=None):
        def startup(trainer,features,output):
            for _ in range(2):
                loss,_=trainer.strategy.forward_loss(None);trainer.backend.backward(loss/2)
            trainer.backend.optimizer.zero_grad()
            return dict(wall_s=.1,optimizer_steps=0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(T,'MathQAFixedLRTrainer',TinyTrainer))
            stack.enter_context(patch('experiments.mathqa_fixed_lr_transfer.task_runner.feature_pool',return_value=['unused']*4))
            stack.enter_context(patch('online_feedback.training.SpecForgeEventTrainer._prepare_event',return_value=self.features))
            stack.enter_context(patch('experiments.prefix_guard.qualify.pin_retained_graph_runtime',return_value={'status':'passed'}))
            stack.enter_context(patch('experiments.functional_step_geometry.native_pilot.discarded_startup',side_effect=startup_override or startup))
            export=stack.enter_context(patch('specforge.export.to_hf.export_to_hf'))
            result=Q.run_training_qualification(self.plan,slot)
            self.assertEqual(export.call_count,1)
            return result

    def test_two_cards_native_scalar_resume_and_exact_cumulative_work(self):
        a=self.execute('A');b=self.execute('B')
        self.assertEqual(Q.compare_training_cards(a,b)['status'],'passed')
        for result in (a,b):
            self.assertEqual([r['case'] for r in result['cases']],list(Q.CASES))
            self.assertTrue(all(row['checks']=='passed' for row in result['cases']))
            self.assertTrue(all(row['pair_wall_excluding_observers_s']>=0 for row in result['cases']))
            self.assertTrue(result['origin_restored'])
            self.assertEqual(result['origin_before'],result['origin_after'])
            self.assertEqual(result['operation_counts']['loading']['training_initialization_completed'],1)
            self.assertEqual(result['operation_counts']['startup']['response_forward_completed'],2)
            self.assertEqual(result['operation_counts']['startup']['response_backward_completed'],2)
            self.assertEqual(result['cases'][2]['actual_learning_rate'],2.88e-4)
            self.assertTrue(Path(result['snapshot_path']).exists())
            self.assertEqual(result['cases'][4]['endpoint']['global_step'],514)
        work=read(self.root/'WORK.json')
        self.assertEqual(work['completed']['adam'],10)
        self.assertEqual(work['completed']['response_forward'],24)
        self.assertEqual(work['completed']['response_backward'],24)
        self.assertEqual(work['completed']['training_initialization'],2)
        self.assertEqual(work['pending'],{})
        self.assertTrue(all(t.closed for t in TinyTrainer.instances))
        self.assertFalse(torch.cuda.is_initialized())

    def test_failed_parity_preserves_endpoint_before_raising_and_stops(self):
        TinyTrainer.corrupt_factor_one=True
        with self.assertRaisesRegex(ValueError,'Native/factor-one'):self.execute()
        result=read(self.root/'qualification/training_A/DETAILS.json')
        self.assertEqual(result['status'],'failed')
        self.assertEqual(len(result['cases']),2)
        self.assertEqual(result['cases'][1]['status'],'completed')
        self.assertEqual(result['cases'][1]['checks'],'pending')
        self.assertIn('preclip',result['cases'][1]);self.assertIn('endpoint',result['cases'][1])
        self.assertEqual(read(self.root/'WORK.json')['completed']['adam'],2)
        self.assertTrue(TinyTrainer.instances[-1].closed)

    def test_cleanup_failure_does_not_replace_the_original_failure(self):
        TinyTrainer.corrupt_factor_one=True
        seen=[]
        def close(trainer):
            seen.append(read(self.root/'qualification/training_A/DETAILS.json'))
            raise RuntimeError('cleanup failed')
        with patch.object(TinyTrainer,'close',close):
            with self.assertRaisesRegex(ValueError,'Native/factor-one'):self.execute()
        self.assertEqual(seen[0]['status'],'failed')
        result=read(self.root/'qualification/training_A/DETAILS.json')
        self.assertIn('Native/factor-one',result['error']);self.assertIn('cleanup failed',result['cleanup_error'])

    def test_startup_failure_records_only_actual_work(self):
        def broken(trainer,features,output):
            trainer.strategy.forward_loss(None)
            raise RuntimeError('startup failed before backward')
        with self.assertRaisesRegex(RuntimeError,'startup failed'):self.execute(startup_override=broken)
        work=read(self.root/'WORK.json')
        self.assertEqual(work['completed']['response_forward'],1)
        self.assertEqual(work['attempted']['response_backward'],0)
        self.assertEqual(work['attempted']['adam'],0)
        self.assertEqual(read(self.root/'qualification/training_A/DETAILS.json')['cases'],[])

    def test_failed_adam_is_pending_and_preclip_receipt_survives(self):
        TinyTrainer.fail_adam=True
        with self.assertRaisesRegex(RuntimeError,'injected Adam'):self.execute()
        work=read(self.root/'WORK.json')
        self.assertEqual(work['attempted']['adam'],1)
        self.assertEqual(work['completed']['adam'],0)
        self.assertEqual(len(work['pending']),1)
        result=read(self.root/'qualification/training_A/DETAILS.json')
        self.assertEqual(result['cases'][0]['status'],'attempted')
        self.assertIn('preclip',result['cases'][0])

    def test_serving_failure_is_saved_before_cleanup_and_primary_error_survives(self):
        from experiments.mathqa_fixed_lr_transfer import qualification_entry as E
        from online_feedback.storage import write_json
        base=self.root/'qualification/serving_A';base.mkdir(parents=True)
        write_json(self.root/'qualification/training_A/STATUS.json',
                   dict(status='passed',result=dict(export='/temporary_S')))
        self.plan.update(warmups=str(self.root/'warmups.json'),warmup_references=str(self.root/'references.json'))
        write_json(self.plan['warmups'],{'requests':[]});write_json(self.plan['warmup_references'],{})
        backend=SimpleNamespace(loading_s=.1,generate=lambda *a,**k:None,check_weights=lambda checkpoint:{})
        observed=[]
        def close(*unused):
            observed.append(read(base/'RESULT.json'))
            raise RuntimeError('serving cleanup failed')
        with patch.object(E,'create_backend',return_value=backend), \
             patch.object(E,'weight_check_errors',return_value=[]), \
             patch('experiments.mathqa_fixed_lr_transfer.evaluation.validate_references'), \
             patch('experiments.mathqa_fixed_lr_transfer.task_runner.verify_reference_seal'), \
             patch.object(E,'evaluate',side_effect=ValueError('original serving mismatch')), \
             patch.object(E,'close_backend',side_effect=close):
            with self.assertRaisesRegex(ValueError,'original serving mismatch'):
                E.serving(self.plan,'A',base)
        self.assertEqual(observed[0]['status'],'failed')
        self.assertIn('original serving mismatch',observed[0]['error'])
        saved=read(base/'RESULT.json')
        self.assertIn('original serving mismatch',saved['error'])
        self.assertIn('serving cleanup failed',saved['cleanup_error'])
        self.assertEqual(read(self.root/'WORK.json')['attempted']['qualification_generation'],0)

    def test_crosscard_failure_is_reported_for_explicit_samecard_fallback(self):
        a=self.execute();b=deepcopy(a)
        b['cases'][2]['endpoint']['weights']['weight']['sha256']='different'
        result=Q.compare_training_cards(a,b)
        self.assertEqual(result['status'],'failed')
        self.assertTrue(result['same_card_checks_passed'])
        self.assertIn('scaled: cross-card endpoint',result['errors'])
        with self.assertRaises(FileExistsError):self.execute()


if __name__=='__main__':unittest.main()
