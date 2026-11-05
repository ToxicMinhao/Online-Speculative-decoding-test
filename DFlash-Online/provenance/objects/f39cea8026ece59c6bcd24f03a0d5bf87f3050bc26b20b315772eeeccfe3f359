"""CPU coverage of the fixed-prefix post-K correction and bounded replay path."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from online_feedback import training as T
from online_feedback.prefix_correction import (
    PrefixCorrection, apply_correction, canonical_margin, correction_direction,
    repair_transitions, violation_terms,
)
from online_feedback.rng import capture_rng
from test_training_contract import trees_equal
import test_osd_recipe as legacy


class CorrectionMathTests(unittest.TestCase):
    def test_native_tie_can_be_wrong_with_zero_hinge_and_zero_gradient(self):
        logits = torch.tensor([[[[2., 2.]]]], dtype=torch.bfloat16, requires_grad=True)
        labels = torch.tensor([[[1]]])
        loss, margins = violation_terms(logits, labels, torch.ones_like(labels).float())
        self.assertEqual(int(logits.argmax(-1)), 0)
        self.assertEqual(float(margins.detach()), 0)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_rival_selection_is_deterministic_and_logits_remain_differentiable(self):
        logits = torch.tensor([[[[3., 3., 1.]]]], dtype=torch.bfloat16, requires_grad=True)
        labels = torch.tensor([[[2]]])
        loss, margin = violation_terms(logits, labels, torch.ones_like(labels).float())
        self.assertEqual(float(margin.detach()), -2)
        loss.backward()
        self.assertEqual(logits.grad.tolist(), [[[[1., 0., -1.]]]])
        self.assertEqual(margin.dtype, torch.float32)

    def test_one_global_denominator_differs_from_response_averaging(self):
        a = torch.tensor([[[[0., 2.]]]], requires_grad=True)
        b = torch.tensor([[[[0., 4.]]]], requires_grad=True)
        labels = torch.zeros((1, 1, 1), dtype=torch.long)
        wa, wb = torch.tensor([[[3.]]]), torch.tensor([[[1.]]])
        na, _ = violation_terms(a, labels, wa)
        nb, _ = violation_terms(b, labels, wb)
        value = (na + nb)/(wa.sum() + wb.sum())
        self.assertEqual(float(value.detach()), 2.5)
        value.backward()
        self.assertEqual(a.grad.tolist(), [[[[-.75, .75]]]])
        self.assertEqual(b.grad.tolist(), [[[[-.25, .25]]]])
        self.assertNotEqual(float(value.detach()), 3.0)

    def test_inactive_positions_have_no_direct_loss_gradient(self):
        logits = torch.tensor([[[[0., 3.], [1., 2.]]]], requires_grad=True)
        labels = torch.zeros((1, 1, 2), dtype=torch.long)
        value, _ = violation_terms(logits, labels, torch.tensor([[[0., 2.]]]))
        value.backward()
        self.assertTrue(torch.equal(logits.grad[0, 0, 0], torch.zeros(2)))

    def test_one_exact_fp32_coefficient_then_bf16_cast(self):
        parameter = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.bfloat16))
        original = parameter.detach().clone()
        gradient = torch.tensor([2., -1.])
        value = torch.tensor(.5)
        result = apply_correction([parameter], [gradient], value,
                                  protected_weight_sum=3., k_update_norm=.2)
        expected = (original.float() - .1*gradient).bfloat16()
        self.assertTrue(torch.equal(parameter, expected))
        self.assertEqual(result['gradient_norm_squared_fp32'], 5.)
        self.assertEqual(result['status'], 'applied')
        self.assertAlmostEqual(result['realized_correction_norm'], float((expected.float()-original.float()).double().norm()))
        self.assertAlmostEqual(result['proposed_correction_to_K_norm_ratio'], (.05**.5)/.2, places=6)

    def test_zero_empty_and_positive_zero_gradient_leave_native_weights_untouched(self):
        for value, weight, gradient, status in (
            (0., 1., [1., 2.], 'zero_violation'),
            (0., 0., [0., 0.], 'empty_protection'),
            (1., 1., [0., 0.], 'unresolved_zero_gradient'),
        ):
            parameter = torch.nn.Parameter(torch.tensor([.05, .05], dtype=torch.bfloat16))
            original = parameter.detach().clone(); version = parameter._version
            result = apply_correction([parameter], [torch.tensor(gradient)], torch.tensor(value),
                                      protected_weight_sum=weight, k_update_norm=1.)
            self.assertEqual(result['status'], status)
            self.assertTrue(torch.equal(parameter, original))
            self.assertEqual(parameter._version, version)

    def test_gradient_underflow_is_unresolved_without_epsilon(self):
        status, squared, coefficient = correction_direction(torch.tensor(1.), [torch.tensor([1e-30])])
        self.assertEqual(status, 'unresolved_zero_gradient')
        self.assertEqual(float(squared), 0)
        self.assertIsNone(coefficient)

    def test_nonfinite_coefficient_stops_before_parameter_mutation(self):
        parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        version = parameter._version
        with self.assertRaisesRegex(ValueError, 'Nonfinite correction coefficient'):
            apply_correction([parameter], [torch.tensor([1e-20])], torch.tensor(1.),
                             protected_weight_sum=1., k_update_norm=1.)
        self.assertEqual(parameter._version, version)

    def test_finite_deltas_that_overflow_bf16_are_rejected_before_any_parameter_changes(self):
        parameters = [torch.nn.Parameter(torch.tensor([0.], dtype=torch.bfloat16)),
                      torch.nn.Parameter(torch.tensor([-3e38], dtype=torch.bfloat16))]
        before = [p.detach().clone() for p in parameters]
        with self.assertRaisesRegex(ValueError, 'Nonfinite proposed or BF16'):
            apply_correction(parameters, [torch.tensor([1.]), torch.tensor([1.])], torch.tensor(3e38),
                             protected_weight_sum=1., k_update_norm=1.)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(parameters,before)))

    def test_repairs_are_matched_by_block_identity_not_aggregate_count(self):
        labels = torch.tensor([[[0, 1, 2], [0, 1, 2]]])
        original = labels.clone(); original[..., 2] = 4
        ordinary = labels.clone(); ordinary[0, 1, 2] = 4
        corrected = labels.clone(); corrected[0, 0, 2] = 4
        result = repair_transitions(original, ordinary, corrected, labels, torch.ones_like(labels).float())
        self.assertEqual(result['K_useful_repairs'], 1)
        self.assertEqual(result['C_useful_repairs'], 1)
        self.assertEqual(result['useful_repairs_retained'], 0)
        self.assertEqual(result['useful_repairs_lost'], 1)
        self.assertEqual(result['useful_repairs_newly_obtained'], 1)


class ToyBlockModel(torch.nn.Module):
    """Two BF16 linear logits, used only to exercise replay and V gradients."""
    block_size = 3
    objective_chunk_blocks = 1
    eos_token_ids = ()

    def __init__(self):
        super().__init__()
        self.draft_model = torch.nn.Linear(2, 2, bias=False).bfloat16()
        self.draft_model.weight.data.copy_(torch.eye(2).bfloat16())
        self.lm_head = torch.nn.Identity()
        self._sample_anchor_positions = lambda *args, **kwargs: (torch.tensor([[0]]), torch.tensor([[True]]))

    def _forward_draft_blocks(self, input_ids, hidden_states, loss_mask, *args):
        anchors, keep = self._sample_anchor_positions()
        return anchors, keep, self.draft_model(hidden_states)


def toy_recorder():
    model = ToyBlockModel()
    trainer = SimpleNamespace(objective='osd_first_error', accumulation=2,
                              bundle=SimpleNamespace(model=model),
                              backend=SimpleNamespace(optimizer=SimpleNamespace(model_params=list(model.draft_model.parameters()))))
    correction = PrefixCorrection(trainer, record_ids=('first', 'second'))
    with correction.capture_k():
        for scale in (1., 2.):
            inputs = torch.zeros((1, 3), dtype=torch.long)
            features = torch.tensor([[[scale, 0.], [scale, 0.], [scale, 0.]]], dtype=torch.bfloat16)
            anchors = torch.tensor([[0]]); keep = torch.tensor([[True]])
            labels = inputs.unsqueeze(1); validity = torch.tensor([[[0., 1., 1.]]])
            logits = model.draft_model(features).unsqueeze(1)
            correction.begin_response(inputs, features, torch.ones_like(inputs).float(), anchors, keep, labels, validity)
            correction.chunk(logits, labels, validity, 0)
            correction.finish_response(torch.tensor(0.), torch.tensor(0.))
        # A synthetic post-step state that breaks both original prefixes.
        with torch.no_grad():
            model.draft_model.weight[0, 0] = -1
    return model, correction


class ReplayTests(unittest.TestCase):
    def test_global_gradient_and_single_correction_use_original_prefixes(self):
        model, correction = toy_recorder()
        rng = capture_rng(); before = model.draft_model.weight.detach().clone()
        post = correction.measure_post_k()
        self.assertEqual(post['denominator'], 6.)
        self.assertEqual(post['numerator'], 9.)
        self.assertEqual(post['V'], 1.5)
        self.assertTrue(torch.equal(model.draft_model.weight, before))
        self.assertTrue(trees_equal(capture_rng(), rng))
        self.assertEqual(correction.gradients[0].dtype, torch.float32)
        torch.testing.assert_close(correction.gradients[0], torch.tensor([[-1.5, 0.], [1.5, 0.]]), rtol=0, atol=0)
        result = correction.apply_once()
        self.assertEqual(result['status'], 'applied')
        corrected = correction.measure_post_c()
        self.assertEqual(corrected['V'], 0.)
        self.assertEqual(corrected['totals']['old_prefix_tokens_lost'], 0)
        self.assertEqual(correction.work['violation_backward_completed'], 2)
        self.assertEqual(correction.work['direct_corrections'], 1)
        self.assertEqual(correction.work['post_k_response_forwards'], 2)
        self.assertEqual(correction.work['post_c_response_forwards'], 2)
        self.assertTrue(trees_equal(capture_rng(), rng))
        correction.close()
        self.assertIsNone(correction.model)
        self.assertEqual(correction.report()['post_c']['V'], 0.)

    def test_finite_bf16_logits_with_overflowing_fp32_margin_stop(self):
        model, correction = toy_recorder()
        correction.phase = 'post_c'
        with self.assertRaisesRegex(ValueError, 'Nonfinite post_c margin'):
            correction._objective_chunk(
                torch.tensor([[[[-3e38, 3e38]]]], dtype=torch.bfloat16),
                torch.tensor([[[0]]]), torch.ones((1,1,1)),
            )
        correction.close()


class KIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(legacy.OSDRecipeTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.OSDRecipeTests.tearDownClass.__func__)
    trainer = legacy.OSDRecipeTests.trainer
    saved = legacy.OSDRecipeTests.saved

    def setUp(self):
        legacy.OSDRecipeTests.setUp(self)
        patch = mock.patch.object(T, 'ranking_eos_token_ids', return_value=(28,))
        patch.start(); self.addCleanup(patch.stop)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_capture_preserves_K_weights_gradients_optimizer_rng_and_counts(self):
        gradient_snapshots = {}
        def record_preclip(trainer, name):
            native_step = trainer.backend.optimizer.step
            def step():
                gradient_snapshots[name] = [parameter.grad.detach().clone()
                    for parameter in trainer.backend.optimizer.model_params]
                return native_step()
            trainer.backend.optimizer.step = step
        ordinary = self.trainer('ordinary', objective='osd_first_error')
        record_preclip(ordinary, 'ordinary')
        expected = ordinary.train_event('e1', self.features[:2], passes=1)
        expected_state = self.saved(ordinary)
        observed = self.trainer('captured', objective='osd_first_error')
        record_preclip(observed, 'captured')
        correction = PrefixCorrection(observed)
        with correction.capture_k():
            result = observed.train_event('e1', self.features[:2], passes=1, presentation_schedule=expected)
        actual_state = self.saved(observed)
        for expected_gradient, actual_gradient in zip(gradient_snapshots['ordinary'], gradient_snapshots['captured']):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        for name in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertTrue(trees_equal(expected_state[name], actual_state[name]), name)
        self.assertEqual(result['optimizer_steps'], 1)
        self.assertEqual(correction.work['original_response_forwards'], 2)
        before = deepcopy(observed.backend.optimizer.state_dict())
        correction.measure_post_k(); correction.apply_once(); correction.measure_post_c()
        self.assertTrue(trees_equal(before, observed.backend.optimizer.state_dict()))
        self.assertEqual(correction.work['post_k_response_forwards'], 2)
        self.assertEqual(correction.work['post_c_response_forwards'], 2)
        self.assertLessEqual(correction.work['violation_backward_completed'], 2)
        self.assertEqual(observed.global_step, 1)
        self.assertTrue(observed.bundle.model.training)
        correction.close()


if __name__ == '__main__':
    unittest.main()
