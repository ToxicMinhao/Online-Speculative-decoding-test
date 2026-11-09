"""Independent CPU formula checks for the same-physical-head-layout oracle."""
import json
import unittest
from unittest.mock import patch

import torch

from experiments.functional_step_geometry.same_layout_oracle import (
    reference_weights, same_layout_reference, sensitivity_terms,
)


class RecordingHead(torch.nn.Linear):
    def __init__(self, dtype):
        super().__init__(2, 3, bias=False, dtype=dtype)
        self.requires_grad_(False)
        self.shapes = []
        with torch.no_grad():
            self.weight.copy_(torch.tensor([[.01, -.02], [.02, .03], [-.01, .015]], dtype=dtype))

    def forward(self, hidden):
        self.shapes.append(tuple(hidden.shape))
        return super().forward(hidden)


def fixture(dtype=torch.bfloat16, blocks=9):
    head = RecordingHead(dtype)
    parameter = torch.nn.Parameter(torch.tensor([[.4, .1], [.2, .3]], dtype=dtype))
    x = (torch.arange(blocks*4*2).reshape(1, blocks, 4, 2).to(dtype)%7)/16
    hidden = x @ parameter
    native = hidden.detach()+torch.tensor([.05, -.04], dtype=dtype)
    labels = head(hidden.detach()).argmax(-1)
    validity = torch.ones_like(labels, dtype=torch.float32)
    validity[..., 0] = 0.
    head.shapes.clear()
    return head, parameter, x, hidden, native, labels, validity


class IndependentFormulaTests(unittest.TestCase):
    def test_prefix_recurrence_seed_errors_gap_and_eos(self):
        labels = torch.tensor([[[0, 1, 2, 9, 4], [0, 1, 2, 3, 4],
                                [9, 1, 2, 3, 4], [0, 1, 2, 3, 4]]])
        proposed = labels.clone(); proposed[0, 1, 1] = 7
        validity = torch.ones_like(labels).float(); validity[0, 3, 2] = 0.
        qy = torch.full_like(validity, .3); qj = torch.full_like(validity, .29)
        result = reference_weights(proposed, labels, validity, qy, qj, (9,))
        self.assertEqual(result['prefix_mask'].tolist(),
            [[[False, True, True, True, False], [False, False, False, False, False],
              [False, False, False, False, False], [False, True, False, False, False]]])
        expected = torch.tensor([[[0., 3., 2., 1., 0.], [0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]]])/7
        self.assertTrue(torch.equal(result['weights'], expected))
        self.assertAlmostEqual(result['denominator'], 7/.05**2)

    def test_tied_native_argmax_is_not_replaced_by_strict_positive_margin(self):
        logits = torch.tensor([[[[1., 1.], [1., 1.], [1., 1.]]]], dtype=torch.bfloat16)
        labels = torch.tensor([[[0, 0, 1]]])
        proposals = logits.argmax(-1)
        probability = logits.float().softmax(-1)
        canonical = probability.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        result = reference_weights(proposals, labels, torch.ones_like(labels).float(),
                                   canonical, torch.full_like(canonical, .5))
        self.assertEqual(result['prefix_mask'].tolist(), [[[False, True, False]]])
        self.assertEqual(result['weights'].tolist(), [[[0., 1., 0.]]])
        self.assertEqual(float(result['margins'][0, 0, 1]), 0.)

    def test_probability_margin_weighting_uses_one_response_denominator(self):
        labels = torch.zeros(1, 9, 3, dtype=torch.long)
        qj = torch.zeros_like(labels).float()
        qy = torch.full_like(qj, .1); qy[:, -1] = .05
        result = reference_weights(labels, labels, torch.ones_like(qj), qy, qj)
        raw = torch.tensor([[[0., 2., 1.]]]*9).reshape(1, 9, 3).double()
        margins = qy.double().clamp_min(.05)
        raw = raw/margins.square()
        expected = (raw/raw.sum()).float()
        self.assertTrue(torch.equal(result['weights'], expected))
        self.assertGreater(float(result['weights'][:, -1].sum()), float(result['weights'][:, 0].sum()))
        self.assertFalse(result['weights'].requires_grad)

    def test_centered_sensitivity_matches_independent_fp64_covariance(self):
        z0 = torch.tensor([[[[.2, -.1, .4], [.3, .8, -.2]]]], dtype=torch.float64,
                          requires_grad=True)
        zn = z0.detach()+torch.tensor([[[[.4, -.2, .1], [-.1, .2, .5]]]], dtype=torch.float64)
        w = torch.tensor([[[.3, .7]]], dtype=torch.float32)
        sensitivity, energy, risk, absolute, _ = sensitivity_terms(z0, zn, w)
        q = z0.detach().softmax(-1)
        v = zn-z0.detach()
        covariance = torch.diag_embed(q)-q.unsqueeze(-1)*q.unsqueeze(-2)
        expected = (covariance @ v.unsqueeze(-1)).squeeze(-1)*w.double().unsqueeze(-1)
        expected_c = (v*expected).sum()
        torch.testing.assert_close(sensitivity.double(), expected, rtol=2e-6, atol=2e-9)
        torch.testing.assert_close(energy, expected_c, rtol=2e-6, atol=2e-9)
        self.assertFalse(sensitivity.requires_grad)
        self.assertFalse(energy.requires_grad)
        self.assertGreater(float(risk), 0.)
        self.assertGreater(float(absolute), 0.)

    def test_native_bf16_layout_and_buffers_remain_unchanged(self):
        head, parameter, _, hidden, native, labels, validity = fixture()
        head.train()
        parameter.grad = torch.ones_like(parameter)
        old_gradient = parameter.grad
        old_weight = head.weight.clone()
        rng = torch.get_rng_state().clone()
        work = []
        with patch('online_feedback.functional_geometry_math.geometry_terms', side_effect=AssertionError('production formula called')), \
             patch('online_feedback.functional_geometry_math.prefix_statistics', side_effect=AssertionError('production prefix called')):
            result = same_layout_reference(head, hidden, native, labels, validity, (parameter,),
                work_callback=lambda kind, status: work.append((kind, status)))
        self.assertEqual(result.statistics['chunks'], [8, 1])
        self.assertEqual(set(shape[1] for shape in head.shapes), {8, 1})
        self.assertTrue(all(shape[0] == 1 and shape[2:] == (4, 2) for shape in head.shapes))
        self.assertEqual(result.gradients[0].dtype, torch.bfloat16)
        self.assertEqual(work, [('additional_backwards', 'attempted'), ('additional_backwards', 'completed')])
        self.assertIs(parameter.grad, old_gradient)
        self.assertTrue(head.training)
        self.assertTrue(torch.equal(head.weight, old_weight))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertGreater(result.c, 0.)
        self.assertEqual(len(result.statistics['chunk_statistics']), 2)
        json.dumps(result.statistics, allow_nan=False)

    def test_parameter_gradient_matches_separate_small_fp64_matrix_oracle(self):
        head, parameter, x, hidden, native, labels, validity = fixture(torch.float64)
        result = same_layout_reference(head, hidden, native, labels, validity, (parameter,))
        # All logits here are small: canonical probability margins lie below .05.
        # Every valid proposal is correct, so independently known weights are
        # [0,3,2,1] / (9*6), regardless of projection chunk boundaries.
        expected_weights = torch.tensor([0., 3., 2., 1.], dtype=torch.float64).repeat(1, 9, 1)/54
        torch.testing.assert_close(result.weights.double(), expected_weights, rtol=1e-7, atol=1e-9)
        before_logits = (x @ parameter.detach()) @ head.weight.T
        after_logits = native @ head.weight.T
        q = before_logits.softmax(-1)
        v = after_logits-before_logits
        centered = v-(q*v).sum(-1, keepdim=True)
        s = expected_weights.unsqueeze(-1)*q*centered
        expected_gradient = x.reshape(-1, 2).T @ (s.reshape(-1, 3) @ head.weight)
        torch.testing.assert_close(result.gradients[0], expected_gradient, rtol=2e-5, atol=1e-10)

    def test_empty_protection_and_zero_energy_do_not_invent_backwards(self):
        for empty in (True, False):
            head, parameter, _, hidden, native, labels, validity = fixture()
            if empty:
                labels = (labels+1)%3
            else:
                native = hidden.detach().clone()
            work = []
            result = same_layout_reference(head, hidden, native, labels, validity, (parameter,),
                work_callback=lambda *args: work.append(args))
            self.assertEqual(result.c, 0.)
            self.assertEqual(result.gradients, (None,))
            self.assertEqual(work, [])
            if empty:
                self.assertEqual(float(result.weights.sum()), 0.)
                self.assertEqual(result.statistics['work']['proxy_calls'], 0)

    def test_nonfinite_and_wrong_physical_partition_fail(self):
        head, parameter, _, hidden, native, labels, validity = fixture()
        with self.assertRaisesRegex(ValueError, 'eight-block'):
            same_layout_reference(head, hidden, native, labels, validity, (parameter,), chunk_blocks=4)
        with torch.no_grad():
            native[0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            same_layout_reference(head, hidden, native, labels, validity, (parameter,))


if __name__ == '__main__':
    unittest.main()
