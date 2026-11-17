"""CPU checks for the fixed labels, masking, normalization and head state."""

import unittest

import torch
import torch.nn.functional as F

from online_feedback.survival_head import (
    ScalarLinearHead, bce_terms, proposal_targets, response_bce,
)


class SurvivalHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_example_and_seed_do_not_enter_survival_product(self):
        labels = torch.tensor([[[9, 1, 2, 3]]])
        proposals = torch.tensor([[[8, 1, 7, 3]]])
        weights = torch.tensor([[[0.0, 1.0, 0.5, 0.25]]])
        local, valid = proposal_targets(proposals, labels, weights, arm="LOCAL")
        survival, same_valid = proposal_targets(proposals, labels, weights, arm="SURVIVAL")
        self.assertEqual(local.tolist(), [[[0.0, 1.0, 0.0, 1.0]]])
        self.assertEqual(survival.tolist(), [[[0.0, 1.0, 0.0, 0.0]]])
        self.assertEqual(valid.tolist(), [[[False, True, True, True]]])
        self.assertTrue(torch.equal(valid, same_valid))
        self.assertFalse(local.requires_grad)
        self.assertFalse(survival.requires_grad)

    def test_seed_eos_inclusive_terminal_padding_and_gap(self):
        labels = torch.tensor([[9, 1, 2, 3, 4], [0, 1, 9, 3, 4],
                               [0, 1, 2, 3, 4], [0, 1, 2, 3, 4],
                               [0, 1, 2, 3, 4]])
        weights = torch.tensor([[0, 1, 1, 1, 1], [0, 1, 1, 1, 1],
                                [0, 1, 0, 1, 1], [0, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0]], dtype=torch.float32)
        expected = [[False] * 5, [False, True, True, False, False],
                    [False, True, False, False, False],
                    [False, True, True, False, False], [False] * 5]
        for arm in ("LOCAL", "SURVIVAL"):
            targets, valid = proposal_targets(labels, labels, weights, arm=arm, eos_token_ids=(9,))
            self.assertEqual(valid.tolist(), expected)
            self.assertTrue(torch.equal(targets, valid.float()))

    def test_bce_is_all_valid_unweighted_proposals_and_fp32(self):
        logits = torch.tensor([[0.2, -1.0, 0.5, 2.0]], dtype=torch.bfloat16,
                              requires_grad=True)
        labels = torch.tensor([[0, 1, 2, 3]])
        proposals = torch.tensor([[7, 1, 5, 3]])
        weights = torch.tensor([[0.0, 1.0, 0.02, 0.001]])
        targets, valid = proposal_targets(proposals, labels, weights, arm="SURVIVAL")
        terms = bce_terms(logits, targets, valid)
        actual = response_bce(terms)
        expected = F.binary_cross_entropy_with_logits(
            logits.float()[..., 1:], torch.tensor([[1.0, 0.0, 0.0]]), reduction="mean")
        self.assertEqual(terms[1].item(), 3)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        self.assertEqual(logits.grad[0, 0].item(), 0)
        self.assertNotEqual(logits.grad[0, 3].item(), 0)

    def test_chunk_sum_divides_once_and_preserves_logits_gradient(self):
        generator = torch.Generator().manual_seed(11)
        original = torch.randn((1, 7, 4), generator=generator)
        labels = torch.randint(0, 5, original.shape, generator=generator)
        proposals = torch.randint(0, 5, original.shape, generator=generator)
        weights = torch.ones_like(original)
        weights[..., 0] = 0
        weights[:, 2, 2:] = 0
        weights[:, 5, 1:] = 0
        targets, valid = proposal_targets(proposals, labels, weights, arm="SURVIVAL")
        reference = None
        for width in (1, 2, 4, 7):
            logits = original.clone().requires_grad_()
            chunks = [bce_terms(logits[:, start:start+width], targets[:, start:start+width],
                                valid[:, start:start+width])
                      for start in range(0, 7, width)]
            terms = tuple(sum(chunk[index] for chunk in chunks) for index in (0, 1))
            loss = response_bce(terms)
            loss.backward()
            if reference is None:
                reference = (loss.detach(), logits.grad.clone())
            torch.testing.assert_close(loss, reference[0], rtol=2e-7, atol=1e-7)
            torch.testing.assert_close(logits.grad, reference[1], rtol=2e-7, atol=1e-8)
        expected = F.binary_cross_entropy_with_logits(original[valid], targets[valid])
        torch.testing.assert_close(reference[0], expected, rtol=2e-7, atol=1e-7)

    def test_empty_mask_has_differentiable_zero(self):
        logits = torch.randn((1, 2, 4), requires_grad=True)
        loss = response_bce(bce_terms(logits, torch.zeros_like(logits),
                                     torch.zeros_like(logits, dtype=torch.bool)))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_initialization_is_nonzero_identical_bf16_and_rng_isolated(self):
        before = torch.get_rng_state().clone()
        local = ScalarLinearHead(16, device="cpu")
        survival = ScalarLinearHead(16, device="cpu")
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertEqual(local.weight.dtype, torch.bfloat16)
        self.assertEqual(local.bias.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(local.weight, survival.weight))
        self.assertTrue(torch.equal(local.bias, survival.bias))
        self.assertGreater(torch.count_nonzero(local.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(local.bias).item(), 0)
        hidden = torch.ones((1, 2, 4, 16), dtype=torch.bfloat16, requires_grad=True)
        local(hidden).float().sum().backward()
        self.assertGreater(hidden.grad.float().norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
