"""CPU numerical checks independent of DFlash's forward implementation."""
import unittest

import torch
import torch.nn.functional as F

from online_feedback.ranking import (
    allocation_coefficients, matching_prefix_mask, rival_softplus,
)


class PrefixTests(unittest.TestCase):
    def test_seed_mismatch_does_not_count_and_later_matches_do_not_reopen(self):
        labels = torch.tensor([[9, 1, 2, 3, 4], [9, 1, 2, 3, 4]])
        proposals = torch.tensor([[0, 1, 8, 3, 4], [9, 7, 2, 3, 4]])
        weights = torch.tensor([[0., 1., 1., 1., 1.]]).expand_as(labels)
        actual = matching_prefix_mask(proposals, labels, weights)
        self.assertEqual(actual.tolist(), [[False, True, False, False, False], [False] * 5])

    def test_invalid_gap_padding_and_cap_stop_the_prefix(self):
        labels = torch.tensor([[9, 1, 2, 3, 4]]).expand(3, -1)
        weights = torch.tensor([[0., 1., 0., 1., 1.], [0., 1., 1., 0., 0.], [0.] * 5])
        self.assertEqual(matching_prefix_mask(labels, labels, weights).tolist(), [
            [False, True, False, False, False],
            [False, True, True, False, False],
            [False] * 5,
        ])

    def test_matching_eos_counts_once_and_seed_eos_has_no_continuation(self):
        labels = torch.tensor([[9, 1, 2, 7, 4], [7, 1, 2, 3, 4], [9, 7, 2, 3, 4]])
        weights = torch.tensor([[0., 1., 1., 1., 1.]]).expand_as(labels)
        actual = matching_prefix_mask(labels, labels, weights, (7, 8))
        self.assertEqual(actual.tolist(), [
            [False, True, True, True, False], [False] * 5, [False, True, False, False, False],
        ])
        proposals = labels.clone()
        proposals[0, 3] = 6
        self.assertFalse(matching_prefix_mask(proposals, labels, weights, (7,))[0, 3])

    def test_unsupervised_eos_is_not_protected(self):
        labels = torch.tensor([[1, 2, 7, 3]])
        weights = torch.tensor([[0., 1., 0., 1.]])
        self.assertEqual(matching_prefix_mask(labels, labels, weights, (7,)).tolist(),
                         [[False, True, False, False]])


class RivalAndAllocationTests(unittest.TestCase):
    def test_exact_full_vocabulary_rival_and_ties_have_correct_gradients(self):
        logits = torch.tensor([[1., 5., 4., 4., -2.], [8., 8., 7., 9., 9.]], requires_grad=True)
        labels = torch.tensor([1, 3])
        loss = rival_softplus(logits, labels)
        torch.testing.assert_close(loss, F.softplus(torch.tensor([-1., 0.])))
        loss.sum().backward()
        expected = torch.zeros_like(logits)
        expected[0, 1], expected[0, 2] = -torch.sigmoid(torch.tensor(-1.)), torch.sigmoid(torch.tensor(-1.))
        expected[1, 3], expected[1, 4] = -.5, .5
        torch.testing.assert_close(logits.grad, expected)

    def test_bf16_projection_values_use_fp32_subtraction(self):
        logits = torch.tensor([[1000., -1000., 9.], [-1000., 1000., 9.]], dtype=torch.bfloat16,
                              requires_grad=True)
        labels = torch.tensor([0, 0])
        loss = rival_softplus(logits, labels)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(float(loss[1].detach()), 2000.)
        loss.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[1, 0]), -1.)
        self.assertEqual(float(logits.grad[1, 1]), 1.)

    def test_uniform_mass_matches_prefix_over_the_whole_response(self):
        weights = torch.tensor([[[0., 3., 2., 1.], [0., 4., 2., 1.]]], requires_grad=True)
        prefix = torch.tensor([[[False, True, True, False], [False, False, False, False]]])
        uniform, rho = allocation_coefficients(prefix, weights, 'osd_uniform')
        selective, other_rho = allocation_coefficients(prefix, weights, 'osd_prefix')
        torch.testing.assert_close(rho, torch.tensor(5. / 13.))
        torch.testing.assert_close(rho, other_rho)
        torch.testing.assert_close((uniform * weights).sum(), (selective * weights).sum())
        self.assertFalse(uniform.requires_grad)
        self.assertFalse(selective.requires_grad)
        self.assertFalse(rho.requires_grad)
        # A single block's ratio would be 5/6, which is deliberately not U.
        self.assertNotEqual(float(rho), 5. / 6.)

    def test_zero_mass_has_zero_rho_without_epsilon_in_nonzero_ratio(self):
        for weight_value, expected in ((0., 0.), (1e-8, 1.)):
            weights = torch.tensor([[[0., weight_value]]])
            prefix = torch.tensor([[[False, True]]])
            coefficients, rho = allocation_coefficients(prefix, weights, 'osd_uniform')
            self.assertEqual(float(rho), expected)
            self.assertTrue(torch.isfinite(coefficients).all())

    def test_auxiliary_is_not_divided_by_protected_count_or_one_plus_lambda(self):
        logits = torch.tensor([[[[0., 1., 2.], [0., 1., 2.], [0., 1., 2.]]]], requires_grad=True)
        labels = torch.tensor([[[0, 1, 2]]])
        weights = torch.tensor([[[0., 2., 1.]]])
        prefix = torch.tensor([[[False, True, False]]])
        coefficients, _ = allocation_coefficients(prefix, weights, 'osd_prefix')
        actual = (weights * coefficients * rival_softplus(logits, labels)).sum() / (weights.sum() + 1e-6)
        expected = 2 * F.softplus(torch.tensor(1.)) / (3 + 1e-6)
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
