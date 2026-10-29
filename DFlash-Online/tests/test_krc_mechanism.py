"""Bounded interventions leave the original KRC trajectory unchanged."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch
from online_feedback.krc_mechanism import MechanismDiagnosticMixin, ContinuationSnapshot, apply_dose, direction_evidence, CANDIDATES
from online_feedback.reference_correction_event import ReferenceCorrectionEvent
from online_feedback.rng import capture_rng
from test_reference_correction_event import ReferenceWindowTrainer
from test_training_contract import trees_equal

class DiagnosticEvent(MechanismDiagnosticMixin, ReferenceCorrectionEvent):
    pass

def fixture(directory):
    trainer = ReferenceWindowTrainer(directory)
    for index in (4, 5):
        raw = torch.load(trainer.feature_paths['response_2'], weights_only=True)
        raw['record_id'] = f'response_{index}'
        path = Path(directory) / f'{index}.pt'
        torch.save(raw, path)
        trainer.feature_paths[raw['record_id']] = path
    plan = []
    for window in trainer.schedule['windows']:
        d = [dict(record_id=f'response_{i}', sampled_anchors=[dict(positions=[[0]], keep_mask=[[True]])]) for i in (4, 5)]
        plan.append(dict(step=window['step'], event_number=2, pass_index=0,
                         A=window['learning'], R=window['reference'], D=d))
    return trainer, plan

class MechanismTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_same_endpoint_two_windows_and_physical_final_witness(self):
        with tempfile.TemporaryDirectory() as directory:
            ordinary, _ = fixture(directory)
            with ReferenceCorrectionEvent(ordinary, schedule=ordinary.schedule, feature_paths=ordinary.feature_paths, probe_steps=()):
                for i in range(4): ordinary.present(i)
            expected = (deepcopy(ordinary.bundle.model.state_dict()), deepcopy(ordinary.backend.optimizer.state_dict()), capture_rng())
            replay, plan = fixture(directory)
            with DiagnosticEvent(replay, schedule=replay.schedule, feature_paths=replay.feature_paths,
                                 probe_steps=(), window_plan=plan, mechanism_dir=Path(directory)/'diagnostic') as event:
                for i in range(4): replay.present(i)
            self.assertTrue(trees_equal(expected, (replay.bundle.model.state_dict(), replay.backend.optimizer.state_dict(), capture_rng())))
            report = event.report()
            self.assertEqual(report['counts']['Adam_completed'], 2)
            for name in ('original_response_forwards','pre_reference_response_forwards','post_k_reference_response_forwards'):
                self.assertEqual(report['work'][name], 4)
            for name, expected_count in [('pre_capture_response_forwards',8),('candidate_response_forwards',48),('d_gradient_response_forwards',4)]:
                self.assertEqual(report['mechanism']['work'][name], expected_count)
            for row in report['mechanism']['rows']:
                self.assertEqual(row['candidate_order'], list(CANDIDATES))
                self.assertTrue(row['restoration_passed'])
                self.assertTrue(row['k_only_witness_passed'])
                self.assertEqual(len(row['candidates']), 4)
            self.assertEqual(report['mechanism']['rows'][1]['candidates']['R_FULL']['branch']['status'], 'empty_protection')
            self.assertEqual(len(list((Path(directory)/'diagnostic').glob('*_BLOCKS.jsonl'))),2)

    def test_snapshot_restores_buffer_rng_mode_cache_and_absent_attribute(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer, _ = fixture(directory)
            model=trainer.bundle.model
            model.register_buffer('diagnostic_buffer', torch.tensor([1.,2.]))
            model._cached_seq_len=torch.tensor([3])
            if hasattr(model,'_cached_bsz'): delattr(model,'_cached_bsz')
            snapshot=ContinuationSnapshot(trainer)
            with torch.no_grad():
                trainer.backend.optimizer.model_params[0].add_(1)
                model.diagnostic_buffer.add_(9)
                model._cached_seq_len.add_(1)
            model._cached_bsz=9
            model.eval(); torch.rand(5)
            snapshot.restore()
            self.assertTrue(all(snapshot.check().values()))
            self.assertFalse(hasattr(model,'_cached_bsz'))

    def test_half_scales_before_cast_not_rounded_full_change(self):
        p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.bfloat16))
        gradient=[torch.tensor([-1.])]; value=torch.tensor(.004)
        info=apply_dose([p],gradient,value,1.,.5)
        self.assertEqual(p.item(),1.)
        evidence=direction_evidence([p],[torch.tensor([1.],dtype=torch.bfloat16)],gradient,info,.5)
        self.assertGreater(evidence['proposed_norm'],0); self.assertEqual(evidence['realized_norm'],0)
        apply_dose([p],gradient,value,1.,1.)
        self.assertGreater(p.item(),1.)

    def test_empty_zero_gradient_and_nonfinite_boundaries(self):
        p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.bfloat16))
        for denominator,value,grad,branch in [(0.,0.,1.,'empty_protection'),(1.,0.,1.,'zero_violation'),(1.,1.,0.,'unresolved_zero_gradient')]:
            info=apply_dose([p],[torch.tensor([grad])],torch.tensor(value),denominator,.5)
            self.assertEqual(info['status'],branch); self.assertEqual(p.item(),1.)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            apply_dose([p],[torch.tensor([float('nan')])],torch.tensor(1.),1.,.5)

    def test_invalid_temporary_candidate_is_retained_and_original_continues(self):
        from online_feedback import krc_mechanism as mechanism
        with tempfile.TemporaryDirectory() as directory:
            trainer, plan = fixture(directory)
            original = mechanism.apply_dose
            def invalid(*args, **kwargs):
                raise ValueError('Nonfinite synthetic half candidate')
            with mock.patch.object(mechanism, 'apply_dose', side_effect=invalid):
                with DiagnosticEvent(trainer, schedule=trainer.schedule, feature_paths=trainer.feature_paths,
                        probe_steps=(), window_plan=plan[:1], mechanism_dir=Path(directory)/'diagnostic') as event:
                    trainer.present(0); trainer.present(1)
            row = event.report()['mechanism']['rows'][0]
            self.assertEqual(row['candidates']['R_HALF']['status'], 'invalid')
            self.assertTrue(row['restoration_passed'])
            self.assertTrue(row['k_only_witness_passed'])
            self.assertEqual(row['work']['candidate_response_forwards'],18)
            self.assertEqual(trainer.global_step,1)

    def test_optimizer_corruption_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer,_=fixture(directory); trainer.present(0);trainer.present(1)
            snapshot=ContinuationSnapshot(trainer)
            next(iter(trainer.backend.optimizer.optimizer.state.values()))['exp_avg'].add_(1)
            with self.assertRaisesRegex(ValueError,'Adam'): snapshot.restore()

if __name__=='__main__': unittest.main()
