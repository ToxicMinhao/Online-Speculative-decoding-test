"""CPU native-gradient isolation and explicit Point3 event-contract checks."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from online_feedback.objective import OSDTrainStrategy
from online_feedback.rng import capture_rng, restore_rng
from .allocation_integration import _endpoint_identity, _restore_parent, ordinary_control_window
from .allocation_qualify import vector_identity
from .allocation_scoring import NativeAllocationScorer
from .native import WorkReceipt, rng_digest
from .point1_native import ancillary_state
from .point3_native import (native_c_reference_gradient, score_native_C, execute_cases,
                            require_historical_coefficients)
from .point3_training import (
    CONTRACT_KEY, Point3Trainer, compare_schedule, production_contract, validate_boundary, validate_event,
    sparse_diagnostics,
)
from .test_allocation_integration import fixture
from .test_allocation_training import fixture as boundary_fixture
from experiments.functional_step_geometry.qualify import runtime_state
from .fork import FORK_CONTRACT_KEY


class NativeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def make(self, root):
        trainer, state, features, responses, snapshots, _, _ = fixture(root)
        trainer.reduction, trainer.accumulation = 'response_mean', 2
        trainer._native_forward = OSDTrainStrategy(trainer.bundle.model).forward_loss
        trainer.events = state['events'] = [{'event_id': 'event_001'}]
        return trainer, state, features, responses, snapshots

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_step_reference_matches_actual_native_preclip_gradient_bitwise(self):
        with TemporaryDirectory() as tmp:
            trainer, state, features, responses, _ = self.make(tmp)
            runtime, ancillary = runtime_state(trainer.bundle.model), ancillary_state(trainer)
            _restore_parent(trainer, state, runtime, ancillary)
            captured = []
            original_step = trainer.backend.optimizer.step
            def step():
                captured.extend(p.grad.detach().clone() for p in trainer.bundle.draft_model.parameters())
                return original_step()
            trainer.backend.optimizer.step = step
            ordinary_control_window(trainer, features)
            trainer.backend.optimizer.step = original_step
            _restore_parent(trainer, state, runtime, ancillary)
            before = _endpoint_identity(trainer)
            with patch.object(trainer.backend.optimizer, 'step', side_effect=AssertionError('reference Adam')):
                gradients, receipt = native_c_reference_gradient(trainer, features, responses,
                    WorkReceipt(Path(tmp)/'REFERENCE_WORK.json'))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(gradients, captured)))
            self.assertEqual(before, _endpoint_identity(trainer))
            self.assertEqual(receipt['native_adam_steps'], 0)
            self.assertEqual(receipt['committed_presentations'], 0)
            self.assertTrue(receipt['rng_restored'])
            self.assertEqual(ancillary, ancillary_state(trainer))

    def test_failed_reference_cleans_gradients_rng_and_observations(self):
        with TemporaryDirectory() as tmp:
            trainer, _, features, responses, _ = self.make(tmp)
            before = _endpoint_identity(trainer)
            original_forward = trainer._native_forward
            calls = []
            def fail_second(*args):
                calls.append(None)
                if len(calls) == 2:
                    raise RuntimeError('injected reference failure')
                return original_forward(*args)
            trainer._native_forward = fail_second
            with self.assertRaisesRegex(RuntimeError, 'injected reference'):
                native_c_reference_gradient(trainer, features, responses)
            self.assertEqual(before, _endpoint_identity(trainer))
            self.assertTrue(all(p.grad is None for p in trainer.bundle.draft_model.parameters()))

    def test_reference_preserves_post_sampling_rng_and_never_resamples(self):
        with TemporaryDirectory() as tmp:
            trainer, _, features, responses, _ = self.make(tmp)
            restore_rng(responses[-1].rng_after_sampling)
            expected = rng_digest(capture_rng())
            calls = trainer.sampler_calls
            gradient, _ = native_c_reference_gradient(trainer, features, responses)
            self.assertEqual(trainer.sampler_calls, calls)
            self.assertEqual(rng_digest(capture_rng()), expected)
            self.assertTrue(all(value.dtype == torch.bfloat16 for value in gradient))

    def test_sparse_score_readout_preserves_small_contrasts_above_common_offset(self):
        beta = torch.zeros(1, 1, 16)
        beta[..., 1:3] = torch.tensor([1., 3.])
        scores = torch.zeros_like(beta)
        scores[..., 1:3] = torch.tensor([100_000_000., 100_000_008.])
        scorer = SimpleNamespace(snapshots=[dict(
            roles=dict(beta=beta, prefix=beta > 0), old_C_coefficients=beta,
        )])
        row = sparse_diagnostics(scorer, dict(allocations=[beta], scores=[scores]))[0]
        self.assertEqual(row['centered_score_min'], -6.)
        self.assertEqual(row['centered_score_max'], 2.)
        self.assertEqual(row['zero_centered_positions'], 0)

    def test_SUM_and_MAX_share_native_reference_and_do_not_mutate_Adam(self):
        with TemporaryDirectory() as tmp:
            trainer, _, features, responses, _ = self.make(tmp)
            scorer = NativeAllocationScorer(trainer, responses, collect_diagnostics=True,
                collect_assessment_diagnostics=True)
            scorer.freeze_current()
            gradients, _ = native_c_reference_gradient(trainer, features, responses)
            before = _endpoint_identity(trainer)
            identity = vector_identity(scorer.parameter_names, gradients)
            for kind in ('SUM', 'MAX'):
                scorer.assessment = kind
                direction, preview = score_native_C(scorer, gradients)
                self.assertEqual(identity, vector_identity(scorer.parameter_names, gradients))
                self.assertEqual(preview['assessment_kind'], kind)
                self.assertEqual(len(preview['bottlenecks']), 2)
                self.assertTrue(all(torch.isfinite(value).all() for value in direction))
            self.assertEqual(before, _endpoint_identity(trainer))

    def test_three_case_qualification_uses_no_step_reference_and_restores_parent(self):
        from . import point2a_math, point2a_native
        from .test_allocation_integration import verify_parent
        from .test_point2a_native import Ledger
        with TemporaryDirectory() as tmp:
            trainer, state, features, responses, _ = self.make(tmp)
            trainer.response_records.clear()
            trainer._anchor_observations.clear()
            runtime, ancillary = runtime_state(trainer.bundle.model), ancillary_state(trainer)
            _restore_parent(trainer, state, runtime, ancillary)
            scorer = NativeAllocationScorer(trainer, responses, collect_diagnostics=True)
            parent = _endpoint_identity(trainer)
            output = Path(tmp)/'qualification'
            output.mkdir()
            ledger = Ledger()
            def qualify(scorer, direction, *args):
                return {'status':'passed'}, scorer.fast_scores(direction)
            with patch.object(point2a_native, 'verify_live',
                              side_effect=lambda t,s: verify_parent(t,s,None)), \
                 patch.object(point2a_math, 'select_reference_pools', return_value={}), \
                 patch.object(point2a_math, 'qualify_direction', side_effect=qualify):
                result = execute_cases(trainer, scorer, state, runtime, ancillary,
                    features, responses, ledger, WorkReceipt(output/'WORK.json'), {},
                    {'C':None,'F_C_SUM':None}, parent, output)
            self.assertEqual(list(result['cases']), ['C','F_C_SUM','F_C_MAX'])
            self.assertEqual([row['case'] for row in ledger.rows], ['C','F_C_SUM','F_C_MAX'])
            self.assertEqual(result['native_reference']['status'], 'passed')
            self.assertEqual(result['native_reference']['native_adam_steps'], 0)
            self.assertEqual(set(result['readouts']), {'parent','C','F_C_SUM','F_C_MAX'})
            self.assertTrue(all('MAX' in row['assessments'] for row in result['readouts'].values()))
            self.assertEqual(parent, _endpoint_identity(trainer))

    def test_sparse_bottleneck_diagnostics_leave_SUM_direction_bit_identical(self):
        with TemporaryDirectory() as tmp:
            trainer, _, features, responses, _ = self.make(tmp)
            scorer = NativeAllocationScorer(trainer, responses, collect_diagnostics=True)
            scorer.freeze_current()
            gradients, _ = native_c_reference_gradient(trainer, features, responses)
            ordinary, receipt = score_native_C(scorer, gradients)
            scorer.collect_assessment_diagnostics = True
            observed, diagnostic = score_native_C(scorer, gradients)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(ordinary, observed)))
            self.assertEqual(receipt['assessment'], diagnostic['assessment'])
            self.assertEqual(len(diagnostic['bottlenecks']), 2)


def point3_boundary_fixture(arm='F_C_MAX'):
    state, trainer = boundary_fixture()
    state['algorithm_contract'].pop(FORK_CONTRACT_KEY)
    state['algorithm_contract'][CONTRACT_KEY] = production_contract(arm, trainer.fork_binding)
    event = state['events'][1]
    event.pop(FORK_CONTRACT_KEY)
    event.update(method=arm, **{CONTRACT_KEY: production_contract(arm, trainer.fork_binding)})
    trainer.contract = deepcopy(state['algorithm_contract'])
    trainer.point3_arm, trainer.point3_binding = arm, trainer.fork_binding
    return state, trainer


class Point3EventTests(unittest.TestCase):
    def test_historical_coefficient_check_is_not_replaced_by_equal_endpoints(self):
        historical = {'coefficient_identity': {'sha256':'retained'}, 'endpoint': {'sha256':'same'}}
        require_historical_coefficients({'sha256':'retained'}, historical)
        with self.assertRaisesRegex(ValueError, 'coefficient identity'):
            require_historical_coefficients({'sha256':'rounded-away-difference'}, historical)

    def test_both_new_arms_are_named_and_parent_history_is_unchanged(self):
        for arm in ('F_C_SUM','F_C_MAX'):
            state, trainer = point3_boundary_fixture(arm)
            before = deepcopy(state['events'][0])
            result = validate_boundary(state, trainer=trainer)
            self.assertEqual(result['point3_arm'], arm)
            self.assertEqual(result['global_step'], 256)
            self.assertEqual(state['events'][0], before)

    def test_legacy_arm_name_or_third_event_cannot_be_admitted(self):
        with self.assertRaisesRegex(ValueError, 'Unknown Point3'):
            production_contract('FRONTIER_P', 'a'*64)
        trainer = object.__new__(Point3Trainer)
        trainer.qualification_only = False
        trainer.global_step,trainer.presentations,trainer.events = 256,512,[{},{}]
        with self.assertRaisesRegex(ValueError, 'one warm continuation'):
            trainer._prepare_event('event_003', ['unused']*128, 2)

    def test_schedule_comparison_checks_all_draws_without_using_loss(self):
        row = dict(record_id='first', **{'pass':0}, sampled_anchors=[{'positions':[[3,6]],'keep_mask':[[True,True]]}],
                   optimizer_step_after=128,optimizer_stepped=False,loss=9.)
        expected = deepcopy(row)
        expected['loss'] = 2.
        self.assertEqual(compare_schedule([row],[expected])['status'], 'passed')
        for key in ('record_id','pass','sampled_anchors','optimizer_step_after','optimizer_stepped'):
            altered = deepcopy(row)
            altered[key] = 'changed'
            with self.assertRaisesRegex(ValueError, 'sampling differs'):
                compare_schedule([altered], [expected])

    def test_SUM_checkpoint_cannot_be_validated_as_MAX(self):
        state, trainer = point3_boundary_fixture('F_C_SUM')
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            validate_event(state['events'][1], arm='F_C_MAX', binding=trainer.point3_binding,
                           require_completed=False)


if __name__ == '__main__':
    unittest.main()
