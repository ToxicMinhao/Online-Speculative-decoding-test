"""CPU-only synthetic checks. Run with the pinned native checkout on PYTHONPATH."""

import random
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

if __package__:
    from .direct_support_score import (
        allocation_rows, chunk_terms, direct_coefficients, operation_contract, reduce_response,
    )
    from .scalar_oracle import response_oracle
else:
    from direct_support_score import (
        allocation_rows, chunk_terms, direct_coefficients, operation_contract, reduce_response,
    )
    from scalar_oracle import response_oracle


torch.set_num_threads(1)


def native_rows(student_logits, teacher_logits):
    """Literal row arithmetic from online_feedback/objective.py:145-164."""
    logits = student_logits.to(torch.bfloat16)
    with torch.no_grad():
        teacher_probability = F.softmax(teacher_logits.to(torch.bfloat16), dim=-1)
    log_probability = F.log_softmax(logits, dim=-1)
    losses = F.kl_div(log_probability, teacher_probability, reduction="none").sum(-1)
    return log_probability, losses, logits.detach().argmax(-1)


def fixture(dtype=torch.float32):
    labels = torch.tensor([[1, 2, 3, 4, 5, 0], [0, 2, 3, 4, 1, 2], [2, 3, 0, 1, 2, 3]])
    proposals = torch.tensor([[1, 2, 1, 4, 5, 0], [0, 2, 3, 4, 1, 2], [2, 4, 0, 1, 2, 3]])
    beta = torch.tensor([[0, 1, .8, .64, .512, 0], [0, 1, .8, 0, .512, .4], [0, 1, .8, .64, .512, .4]])
    q = torch.tensor([[.1, .7, .2, .8, .9, .3], [.1, .8, .7, .2, .3, .9], [.1, .1, .3, .7, .8, .6]])
    logp = q.log().unsqueeze(-1).expand(-1, -1, 6).to(dtype).clone()
    losses = torch.arange(1, 19, dtype=dtype).reshape(3, 6) / 7
    return logp, losses, proposals, labels, beta


class DirectSupportScoreTests(unittest.TestCase):
    def assert_close(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype), **kwargs)

    def test_native_masks_first_error_eos_seed_gap_and_empty(self):
        labels = torch.tensor([
            [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4],
            [0, 1, 5, 3, 4], [5, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4],
        ])
        proposals = labels.clone()
        proposals[0, 1] = 4
        proposals[1, 3] = 4
        proposals[3, 2] = 2  # First error is canonical EOS: EOS itself remains supervised.
        beta = torch.ones(7, 5)
        beta[5, 2] = 0  # A positive weight after a gap cannot reopen the block.
        beta[6] = 0
        rows = allocation_rows(torch.zeros(7, 5, 6), proposals, labels, beta, (5,))
        expected = [[0,1,0,0,0], [0,1,1,1,0], [0,1,1,1,1],
                    [0,1,1,0,0], [0,0,0,0,0], [0,1,0,0,0], [0,0,0,0,0]]
        self.assert_close(rows.support, expected)
        self.assertFalse(rows.valid[:, 0].any())
        self.assertTrue(rows.valid[3, 2])
        self.assertFalse(rows.valid[3, 3:].any())

    def test_scalar_oracle_unequal_lengths_and_beta(self):
        args = fixture()
        rows = allocation_rows(args[0], args[2], args[3], args[4], (0,))
        result = reduce_response([chunk_terms(*args, eos_token_ids=(0,))])
        q = args[0].gather(-1, args[3].unsqueeze(-1)).squeeze(-1).float().exp()
        oracle = response_oracle(q.tolist(), args[2].tolist(), args[3].tolist(),
                                 args[4].tolist(), args[1].tolist(), (0,))
        for name in ("valid", "support"):
            self.assert_close(getattr(rows, name), oracle[name])
        self.assert_close(rows.score, oracle["scores"])
        self.assert_close(result.beta_mass, oracle["B"])
        self.assert_close(result.score_mass, oracle["R"])
        self.assert_close(direct_coefficients(rows, result), oracle["coefficients"])
        self.assert_close(result.loss, oracle["loss"])

    def test_randomized_scalar_oracle_100_fixed_cases(self):
        rng = random.Random(20261119)
        for case in range(100):
            blocks, slots = rng.randrange(1, 8), rng.randrange(2, 17)
            labels = torch.tensor([[rng.randrange(6) for _ in range(slots)] for _ in range(blocks)])
            proposals = torch.tensor([[rng.randrange(6) for _ in range(slots)] for _ in range(blocks)])
            beta = torch.tensor([[0.0 if rng.random() < .17 else .8 ** max(j-1, 0)
                                  for j in range(slots)] for _ in range(blocks)])
            q = torch.tensor([[rng.random() for _ in range(slots)] for _ in range(blocks)])
            logp = q.log().unsqueeze(-1).expand(-1, -1, 6).to(torch.bfloat16)
            losses = torch.tensor([[rng.random() * 3 for _ in range(slots)] for _ in range(blocks)])
            args = (logp, losses, proposals, labels, beta)
            rows = allocation_rows(logp, proposals, labels, beta, (5,))
            result = reduce_response([chunk_terms(*args, eos_token_ids=(5,))])
            rounded_q = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1).float().exp()
            oracle = response_oracle(rounded_q.tolist(), proposals.tolist(), labels.tolist(),
                                     beta.tolist(), losses.tolist(), (5,))
            with self.subTest(case=case):
                self.assert_close(rows.support, oracle["support"])
                self.assert_close(rows.score, oracle["scores"])
                self.assert_close(direct_coefficients(rows, result), oracle["coefficients"])
                self.assert_close(result.loss, oracle["loss"])

    def test_padding_and_invalid_content_do_not_change_scores_or_loss(self):
        logp, losses, proposals, labels, beta = fixture()
        base = reduce_response([chunk_terms(logp, losses, proposals, labels, beta)])
        rows = allocation_rows(logp, proposals, labels, beta)
        for pad_width in (1, 7, 19):
            padded_logp = F.pad(logp, (0, 0, 0, pad_width), value=float("nan"))
            padded_labels = F.pad(labels, (0, pad_width), value=3)
            padded_proposals = F.pad(proposals, (0, pad_width), value=4)
            padded_beta = F.pad(beta, (0, pad_width))
            padded_losses = F.pad(losses, (0, pad_width), value=123)
            got_rows = allocation_rows(padded_logp, padded_proposals, padded_labels, padded_beta)
            got = reduce_response([chunk_terms(padded_logp, padded_losses, padded_proposals,
                                              padded_labels, padded_beta)])
            self.assert_close(got_rows.score[:, :labels.shape[1]], rows.score, rtol=0, atol=0)
            self.assert_close(got.loss, base.loss)
            self.assert_close(got.beta_mass, base.beta_mass)
            self.assert_close(got.score_mass, base.score_mass)

    def test_full_valid_suffix_changes_scores_but_has_zero_direct_loss(self):
        labels = torch.zeros(2, 5, dtype=torch.long)
        proposals = torch.ones_like(labels)  # Only slot one in each block is in S.
        beta = torch.tensor([[0., 1., .8, .64, .512]]).expand(2, -1)
        losses = torch.tensor([[7., 1., 3., 4., 5.], [7., 2., 3., 4., 5.]])
        logp = torch.full((2, 5, 2), -.7)
        before_rows = allocation_rows(logp, proposals, labels, beta)
        before = reduce_response([chunk_terms(logp, losses, proposals, labels, beta)])
        changed = logp.clone()
        changed[0, 2:, 0] = -.1
        after_rows = allocation_rows(changed, proposals, labels, beta)
        after = reduce_response([chunk_terms(changed, losses, proposals, labels, beta)])
        self.assertGreater(float(after_rows.score[0, 1]), float(before_rows.score[0, 1]))
        self.assertNotEqual(float(before.loss), float(after.loss))
        changed_losses = losses.clone()
        changed_losses[:, 2:] = 1e6
        suffix_loss = reduce_response([chunk_terms(changed, changed_losses, proposals, labels, beta)])
        self.assert_close(suffix_loss.loss, after.loss, rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(direct_coefficients(after_rows, after)[:, 2:])), 0)

    def test_detached_probability_weights_and_direct_suffix_gradient(self):
        logp, losses, proposals, labels, beta = fixture()
        logp.requires_grad_()
        beta.requires_grad_()
        losses.requires_grad_()
        rows = allocation_rows(logp, proposals, labels, beta)
        result = reduce_response([chunk_terms(logp, losses, proposals, labels, beta)])
        coefficient = direct_coefficients(rows, result)
        result.loss.backward()
        self.assertIsNone(logp.grad)
        self.assertIsNone(beta.grad)
        self.assertFalse(coefficient.requires_grad)
        self.assertFalse(result.coefficient_scale.requires_grad)
        self.assert_close(losses.grad, coefficient / (result.beta_mass + 1e-6))
        self.assertEqual(int(torch.count_nonzero(losses.grad[~rows.support])), 0)
        self.assertTrue(torch.isfinite(losses.grad).all())

    def test_native_bf16_kl_rows_and_finite_logit_gradients(self):
        generator = torch.Generator().manual_seed(71)
        student = torch.randn(4, 16, 11, generator=generator, requires_grad=True)
        teacher = torch.randn(4, 16, 11, generator=generator, requires_grad=True)
        logp, losses, proposals = native_rows(student, teacher)
        labels = proposals.clone()
        labels[0, 3] = (labels[0, 3] + 1) % 11
        labels[2, 1] = (labels[2, 1] + 1) % 11
        beta = torch.exp(-torch.arange(16).clamp(min=1).sub(1).float() / 7).expand(4, -1).clone()
        beta[:, 0] = 0
        beta[3, 6:] = 0
        rows = allocation_rows(logp, proposals, labels, beta)
        result = reduce_response([chunk_terms(logp, losses, proposals, labels, beta)])
        self.assertEqual(losses.dtype, torch.bfloat16)
        expected_loss = (direct_coefficients(rows, result) * losses).sum() / (result.beta_mass + 1e-6)
        self.assert_close(result.loss, expected_loss)
        result.loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertIsNone(teacher.grad)
        self.assertEqual(int(torch.count_nonzero(student.grad[~rows.support])), 0)
        self.assertGreater(int(torch.count_nonzero(student.grad[rows.support])), 0)

    def test_response_wide_pool_across_chunks_value_and_gradient(self):
        args = list(fixture())
        args[1] = args[1].clone().requires_grad_()
        full = reduce_response([chunk_terms(*args)])
        full.loss.backward()
        gradient = args[1].grad.clone()
        args[1].grad = None
        split = reduce_response([chunk_terms(*(x[:1] for x in args)),
                                 chunk_terms(*(x[1:] for x in args))])
        split.loss.backward()
        self.assert_close(split.loss, full.loss)
        self.assert_close(split.beta_mass, full.beta_mass)
        self.assert_close(split.score_mass, full.score_mass)
        self.assert_close(args[1].grad, gradient)
        wrong = sum(reduce_response([chunk_terms(*(x[i:i+1] for x in args))]).loss for i in range(3)) / 3
        self.assertGreater(abs(float(wrong.detach() - full.loss.detach())), .01)

    def test_unequal_width_chunks_share_one_response_pool(self):
        args = fixture()
        terms = [chunk_terms(*(x[:1] for x in args)),
                 chunk_terms(*(x[1:2, :3] for x in args)),
                 chunk_terms(*(x[2:] for x in args))]
        ragged = reduce_response(terms)
        full = reduce_response([chunk_terms(*args)])
        self.assert_close(ragged.loss, full.loss)
        self.assert_close(ragged.beta_mass, full.beta_mass)
        self.assert_close(ragged.score_mass, full.score_mass)

    def test_conservation_native_beta_only_sets_total_mass(self):
        args = fixture()
        rows = allocation_rows(args[0], args[2], args[3], args[4])
        result = reduce_response([chunk_terms(*args)])
        coefficients = direct_coefficients(rows, result)
        self.assert_close(coefficients.sum(), result.beta_mass, rtol=2e-6, atol=2e-7)
        # Move beta within support, retaining its sum and positive validity.
        beta = args[4].clone()
        beta[0, 1] -= .2
        beta[0, 2] += .2
        changed = reduce_response([chunk_terms(*args[:4], beta)])
        self.assert_close(changed.loss, result.loss)
        self.assert_close(direct_coefficients(rows, changed), coefficients)
        selected = rows.support
        self.assert_close(coefficients[selected] / rows.score[selected],
                          result.coefficient_scale.expand(int(selected.sum())))

    def test_empty_support_matches_native_zero_supervision(self):
        logp, losses, proposals, labels, beta = fixture()
        losses.requires_grad_()
        beta.zero_()
        result = reduce_response([chunk_terms(logp, losses, proposals, labels, beta)])
        native_zero = (losses * beta).sum() / (beta.sum() + 1e-6)
        self.assert_close(result.loss, native_zero, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        self.assertEqual(int(torch.count_nonzero(losses.grad)), 0)
        self.assertEqual(float(result.coefficient_scale), 0)

    def test_q_boundary_values_and_smoothing_have_positive_scores(self):
        labels = torch.zeros(2, 16, dtype=torch.long)
        proposals = labels.clone()
        beta = torch.ones(2, 16)
        beta[:, 0] = 0
        logp = torch.zeros(2, 16, 2)
        logp[0] = -float("inf")  # q=0 still has factor 0.5.
        rows = allocation_rows(logp, proposals, labels, beta)
        self.assertGreater(float(rows.score[0, 15]), 0)
        self.assert_close(rows.score[0, 15], .5 ** 15, rtol=0, atol=0)
        self.assert_close(rows.score[1, 1:], list(range(15, 0, -1)), rtol=0, atol=0)

    def test_source_score_agrees_on_valid_positions(self):
        from online_feedback import prefix_allocation as native
        self.assertEqual(native.SMOOTHING, operation_contract()["smoothing"])
        args = fixture(torch.bfloat16)
        rows = allocation_rows(args[0], args[2], args[3], args[4])
        q = args[0].gather(-1, args[3].unsqueeze(-1)).squeeze(-1).float().exp()
        # Capture the actual pinned production score before its different
        # bounded-prefix allocator. This process-local patch is restored here.
        with patch.object(native, "coefficients_from_scores", side_effect=lambda scores, *a, **k: scores):
            production_score = native.bounded_prefix_coefficients(q, args[4], rows.valid, rows.support)
        self.assert_close(rows.score[rows.valid], production_score[rows.valid], rtol=0, atol=0)

    def test_zero_direct_suffix_loss_does_not_isolate_context_parameter_gradients(self):
        # A synthetic shared-context dependency demonstrates the distinction;
        # this is not a claimed native-attention reproduction.
        hidden = torch.tensor([[[.1, .2], [.3, -.2], [.4, .7], [.8, -.5]]], requires_grad=True)
        mixed = hidden + hidden.mean(dim=1, keepdim=True)
        student = mixed @ torch.tensor([[1., -.5, .2], [.3, .7, -.8]])
        logp, losses, proposals = native_rows(student, torch.zeros_like(student))
        labels = proposals.clone()
        labels[:, 1] = (labels[:, 1] + 1) % 3  # Support stops at slot one.
        beta = torch.tensor([[0., 1., .8, .64]])
        rows = allocation_rows(logp, proposals, labels, beta)
        result = reduce_response([chunk_terms(logp, losses, proposals, labels, beta)])
        self.assertEqual(int(torch.count_nonzero(direct_coefficients(rows, result)[:, 2:])), 0)
        result.loss.backward()
        self.assertGreater(int(torch.count_nonzero(hidden.grad[:, 2:])), 0)
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_native_two_response_average_is_not_a_cross_response_pool(self):
        args = fixture()
        first = chunk_terms(*(x[:1] for x in args))
        second = chunk_terms(*(x[1:] for x in args))
        average = (reduce_response([first]).loss + reduce_response([second]).loss) / 2
        pooled = reduce_response([first, second]).loss
        self.assertGreater(abs(float(average - pooled)), .01)

    def test_response_dimension_and_missing_chunks_fail_explicitly(self):
        args = fixture()
        with self.assertRaisesRegex(ValueError, "exactly one response"):
            allocation_rows(args[0].unsqueeze(0), args[2].unsqueeze(0),
                            args[3].unsqueeze(0), args[4].unsqueeze(0))
        with self.assertRaisesRegex(ValueError, "at least one native chunk"):
            reduce_response([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
