"""CPU controls for bounded native-LR calibration; no model downloads or GPU work."""
from pathlib import Path
from copy import deepcopy
import json
import math
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from experiments.native_update_size.calibration import (
    CalibrationUnresolved, Point, cached_trial, calibrate, ordinary_pair,
    gradient_identity, observe_adam, norm_agreement, tight_bracket,
)
from experiments.native_update_size.qualification import qualify
from experiments.native_update_size.runner import WindowController, update_geometry
from experiments.ok_continuation.resume import differences
from experiments.threatened_repair_replay.state import capture_state, copy_tree, restore_state, verify_state
from experiments.threatened_repair_replay.work import Ledger, TrainingInstrumentation
from test_threat_state import tiny_trainer, first_pair


class SearchTests(unittest.TestCase):
    def trial(self,function,calls):
        def call(alpha,index):
            calls.append(alpha)
            norm = function(alpha)
            return {'small_fixture':alpha},dict(status='completed',optimizer_applications=1,
                displacement=dict(difference_norm=norm))
        return call

    def test_tight_two_sided_bracket_is_attained_within_ten_calls(self):
        calls=[]
        lower,upper,rows=calibrate(.945,1.,{'native':1},self.trial(lambda alpha:alpha,calls))
        self.assertTrue(tight_bracket(lower,upper,.945))
        self.assertLessEqual(len(calls),10)
        self.assertTrue(all(0<alpha<1 for alpha in calls))
        self.assertEqual(len(rows),len(calls))

    def test_exact_match_alias_stops_immediately(self):
        calls=[]
        lower,upper,rows=calibrate(.75,1.,{},self.trial(lambda alpha:alpha,calls))
        self.assertIs(lower,upper)
        self.assertEqual(calls,[.5,.75])
        self.assertEqual(rows[-1]['selection_status'],'exact_match_both_controls')

    def test_native_endpoint_exact_match_needs_no_new_application(self):
        calls=[]
        lower,upper,rows=calibrate(1.,1.,{'w':1},self.trial(lambda alpha:alpha,calls))
        self.assertIs(lower,upper)
        self.assertEqual((lower.alpha,calls,rows),(1.,[],[]))

    def test_plateau_bisects_alpha_without_repeating_or_relaxing_tolerance(self):
        calls=[]
        with self.assertRaisesRegex(CalibrationUnresolved,'Ten native scalar'):
            calibrate(.8,1.,{},self.trial(lambda alpha:math.floor(alpha*4)/4,calls))
        self.assertEqual(len(calls),10)
        self.assertEqual(len(set(calls)),10)

    def test_observed_nonmonotonicity_stops_and_saves_failed_trial(self):
        calls=[]; receipts=[]
        with self.assertRaisesRegex(CalibrationUnresolved,'monotone'):
            calibrate(.95,1.,{},self.trial(lambda a:.9 if a==.5 else .8,calls),on_attempt=receipts.append)
        self.assertEqual(len(calls),2)
        self.assertEqual(receipts[-1]['status'],'failed')

    def test_nonfinite_norm_and_invalid_target_are_not_epsilon_repaired(self):
        receipts=[]
        with self.assertRaises(CalibrationUnresolved):
            calibrate(.9,1.,{},self.trial(lambda a:float('nan'),[]),on_attempt=receipts.append)
        self.assertEqual(receipts[-1]['actual_norm'],'nan')
        json.dumps(receipts,allow_nan=False)
        for target in (0.,-1.,float('nan'),1.1):
            with self.subTest(target=target),self.assertRaises(CalibrationUnresolved):
                calibrate(target,1.,{},lambda *_:None)

    def test_scalar_accounting_tolerance_is_fixed_and_tighter_than_bracket(self):
        self.assertTrue(norm_agreement(1.+5e-11,1.))
        self.assertFalse(norm_agreement(1.+2e-10,1.))
        self.assertFalse(norm_agreement(float('nan'),1.))


class NativeTrialTests(unittest.TestCase):
    def prepare(self,root):
        trainer,features=tiny_trainer(Path(root)/'trainer',count=4)
        pair=first_pair(trainer,features)
        pre=capture_state(trainer)
        ordinary=ordinary_pair(trainer,pair,0,0,category='native',label='N')
        native=capture_state(trainer)
        return trainer,pre,native,ordinary

    def test_cached_replays_keep_preclip_cache_and_native_rng_sampler_exact(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,pre,native,ordinary=self.prepare(root)
            gradients=ordinary['preclip_gradients']['native']
            immutable=copy_tree(gradients)
            one,first=cached_trial(trainer,pre,native,gradients,ordinary['update'],.5,
                window=1,trial_index=1,output=Path(root)/'trials')
            two,second=cached_trial(trainer,pre,native,gradients,ordinary['update'],.5,
                window=1,trial_index=2,output=Path(root)/'trials')
            self.assertEqual(differences(one,two),[])
            self.assertEqual(differences(gradients,immutable),[])
            self.assertEqual(first['requested_lrs'],[1.5e-4])
            self.assertEqual(first['restored_group_lrs'],[3e-4])
            self.assertEqual(first['cached_state_semantics']['status'],'passed')
            self.assertEqual(first['moment_clip_option_parity']['status'],'passed')
            self.assertEqual(first['response_forwards'],0)
            self.assertEqual(first['end_presentations'],pre['scientific']['presentations'])
            self.assertEqual(verify_state(trainer,native)['status'],'passed')
            self.assertEqual(second['adam_observation']['clipped_gradients'],first['adam_observation']['clipped_gradients'])

    def test_zero_lr_retains_weights_and_none_gradient_membership(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,features=tiny_trainer(Path(root)/'trainer',count=4)
            first_pair(trainer,features)
            pre=capture_state(trainer)
            gradient={'weight':torch.full_like(trainer.bundle.draft_model.weight,2.),'bias':None}
            trainer.bundle.draft_model.weight.grad=gradient['weight'].clone()
            trainer.bundle.draft_model.bias.grad=None
            with observe_adam(trainer,[3e-4]) as observation:
                norm=trainer._native_step()
            trainer.global_step+=1
            factor=torch.clamp(trainer.backend.optimizer.max_grad_norm/(norm.detach()+1e-6),max=1.)
            native=capture_state(trainer)
            update=dict(combined_native_clip_norm=float(norm),clip_factor=float(factor),
                clip_factor_dtype=str(factor.dtype),adam_observation=observation)
            weights,receipt=cached_trial(trainer,pre,native,gradient,update,0.,window=1,
                trial_index=1,output=Path(root)/'trials')
            self.assertEqual(differences(weights,pre['scientific']['draft_state_dict']),[])
            self.assertIsNone(receipt['adam_observation']['clipped_gradients']['bias'])
            self.assertEqual(receipt['displacement']['difference_norm'],0.)
            self.assertEqual(receipt['moment_clip_option_parity']['status'],'passed')

    def test_lr_overwrite_is_rejected_and_native_restored(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,pre,native,ordinary=self.prepare(root)
            original=trainer._native_step
            def overwritten():
                trainer.backend.optimizer.optimizer.param_groups[0]['lr']=.123
                return original()
            trainer._native_step=overwritten
            output=Path(root)/'trials'
            with self.assertRaisesRegex(ValueError,'overwrote'):
                cached_trial(trainer,pre,native,ordinary['preclip_gradients']['native'],ordinary['update'],.5,
                    window=1,trial_index=1,output=output)
            self.assertEqual(verify_state(trainer,native)['status'],'passed')
            row=json.loads((output/'TRIAL_01.json').read_text())
            self.assertEqual(row['status'],'failed')
            self.assertEqual(row['optimizer_applications'],0)

    def test_nonfinite_preclip_fails_without_hiding_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,pre,native,ordinary=self.prepare(root)
            gradients=copy_tree(ordinary['preclip_gradients']['native'])
            gradients['weight'][0,0]=torch.nan
            with self.assertRaises(RuntimeError):
                cached_trial(trainer,pre,native,gradients,ordinary['update'],.5,
                    window=1,trial_index=1,output=Path(root)/'trials')
            self.assertEqual(verify_state(trainer,native)['status'],'passed')
            row=json.loads((Path(root)/'trials/TRIAL_01.json').read_text())
            self.assertEqual(row['status'],'failed')

    def test_raw_gradient_identity_preserves_signed_zero(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,_=tiny_trainer(root,count=4)
            trainer.bundle.draft_model.weight.grad=torch.zeros_like(trainer.bundle.draft_model.weight)
            positive=gradient_identity(trainer)
            trainer.bundle.draft_model.weight.grad[0,0]=-0.
            negative=gradient_identity(trainer)
            self.assertNotEqual(positive['weight']['sha256'],negative['weight']['sha256'])


class QualificationTests(unittest.TestCase):
    def test_six_applications_use_only_one_full_pair_and_restore_origin(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,features=tiny_trainer(Path(root)/'trainer',count=128)
            before=capture_state(trainer)
            ledger=Ledger(Path(root)/'ledger.jsonl',limits={'optimizer_step':6})
            trainer.ledger=ledger
            instrumentation=TrainingInstrumentation(trainer,ledger)
            try:
                result=qualify(trainer,[row['path'] for row in features],Path(root)/'qualification')
                self.assertEqual(result['status'],'passed')
                self.assertEqual(result['optimizer_steps'],6)
                self.assertEqual(result['response_forwards'],2)
                self.assertEqual(result['response_backwards'],2)
                self.assertEqual(ledger.summary()['completed'],dict(response_forward=2,response_backward=2,optimizer_step=6))
                verify_state(trainer,before)
                self.assertFalse(list((Path(root)/'qualification').rglob('*.pt')))
            finally:
                instrumentation.close();ledger.close()

    def test_qualification_failure_restores_origin_and_retains_failed_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,features=tiny_trainer(Path(root)/'trainer',count=128)
            before=capture_state(trainer)
            def bad_trial(*args,**kwargs):
                weights,row=cached_trial(*args,**kwargs)
                weights['weight'][0,0]+=1
                return weights,row
            output=Path(root)/'qualification'
            with patch('experiments.native_update_size.qualification.cached_trial',side_effect=bad_trial):
                with self.assertRaisesRegex(ValueError,'alpha1_native_weights'):
                    qualify(trainer,[row['path'] for row in features],output)
            result=json.loads((output/'RESULT.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['optimizer_steps'],2)
            self.assertEqual(result['cases'][-1]['status'],'failed')
            verify_state(trainer,before)

    def test_failed_cached_gate_counts_completed_application_and_saves_case(self):
        with tempfile.TemporaryDirectory() as root:
            trainer,features=tiny_trainer(Path(root)/'trainer',count=128)
            before=capture_state(trainer)
            def wrong_clip(*args,**kwargs):
                values=list(args)
                values[4]=dict(values[4],combined_native_clip_norm=999.)
                return cached_trial(*values,**kwargs)
            output=Path(root)/'qualification'
            with patch('experiments.native_update_size.qualification.cached_trial',side_effect=wrong_clip):
                with self.assertRaisesRegex(ValueError,'changed native clipping'):
                    qualify(trainer,[row['path'] for row in features],output)
            result=json.loads((output/'RESULT.json').read_text())
            self.assertEqual(result['optimizer_steps'],2)
            self.assertEqual(result['cases'][-1]['status'],'failed')
            self.assertEqual(result['cases'][-1]['update']['optimizer_applications'],1)
            verify_state(trainer,before)


class ControllerTests(unittest.TestCase):
    def test_one_window_keeps_native_trajectory_and_exports_exact_alias_once(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            trainer,features=tiny_trainer(root/'trainer',count=4)
            for state in trainer.backend.optimizer.optimizer.state.values():
                state['exp_avg'].fill_(1.)
            pair=first_pair(trainer,features)
            before=capture_state(trainer)
            native_result=ordinary_pair(trainer,pair,0,0,category='native',label='N')
            expected=capture_state(trainer)
            from experiments.threatened_repair_replay.state import tensor_change
            target=tensor_change(before['scientific']['draft_state_dict'],expected['scientific']['draft_state_dict'])['difference_norm']
            restore_state(trainer,before)
            template=root/'template';template.mkdir()
            save_file(before['scientific']['draft_state_dict'],str(template/'model.safetensors'))
            e=root/'retained_e';e.mkdir()
            save_file(expected['scientific']['draft_state_dict'],str(e/'model.safetensors'))
            output=root/'run';output.mkdir()
            plan=dict(origin_export=str(template),historical_windows={'8':dict(E_norm=target,N_norm=target,
                E={'checkpoint':str(e)})},remaining_probe_estimate_s=1.,remaining_overhead_estimate_s=0.)
            qualification={'cases':[dict(update={'wall_s':.01})]+[dict(update={'wall_s':.01})]*5}
            observed=[]
            def authenticate(_plan,window,state,weights,_directory):
                observed.append((window,state))
                return {'status':'passed'}
            ledger=Ledger(root/'ledger.jsonl')
            try:
                controller=WindowController(plan,output,ledger,qualification,time.monotonic()+1000,
                    authenticate=authenticate,witness_check=lambda *_:{'status':'passed'},
                    storage_check=lambda *_:None)
                result=controller(trainer,8,pair,0,0)
                self.assertEqual(result['update']['optimizer_applications'],1)
                self.assertEqual(observed,[(8,'P'),(8,'N')])
                self.assertEqual(len(controller.trials),0)
                self.assertEqual(len(controller.exports),2)
                self.assertEqual(controller.exports[1]['reused_from'],'S_low')
                self.assertEqual(controller.exports[0]['checkpoint'],controller.exports[1]['checkpoint'])
                self.assertEqual(len(list((output/'exports').rglob('model.safetensors'))),1)
                self.assertFalse((output/'CONTROL_SEAL.json').exists())
                verify_state(trainer,expected)
            finally:
                ledger.close()


class GeometryTests(unittest.TestCase):
    def test_detached_update_geometry_uses_displacements_not_weight_cosines(self):
        before={'layers.0.weight':torch.tensor([1.,2.],dtype=torch.bfloat16)}
        native={'layers.0.weight':torch.tensor([3.,2.],dtype=torch.bfloat16)}
        scalar={'layers.0.weight':torch.tensor([2.,2.],dtype=torch.bfloat16)}
        row=update_geometry(before,scalar,native)
        self.assertEqual(row['global_geometry']['norm'],1.)
        self.assertEqual(row['global_geometry']['comparison_norm'],2.)
        self.assertEqual(row['global_geometry']['cosine'],1.)
        self.assertEqual(row['layers']['layers.0']['changed_entries'],1)


if __name__=='__main__':
    unittest.main()
