"""CPU checks of support, allocation, continuous Adam and matrix-free scores."""

from copy import deepcopy
import itertools
import math
import unittest

import torch
import torch.nn.functional as F

from . import math as M


def beta_for(labels):
    beta = torch.exp(-torch.arange(labels.shape[-1], dtype=torch.float32) / 7 + 1 / 7)
    beta[0] = 0
    return beta.expand(labels.shape).clone()


def roles_for(matches):
    labels = torch.zeros((1, 1, len(matches) + 1), dtype=torch.long)
    proposals = labels.clone()
    proposals[..., 1:] = torch.tensor([not value for value in matches]).long()
    return M.current_roles(proposals, labels, beta_for(labels))


def independent_adam(parameters, gradients, means, squares, steps, options, max_norm):
    """Separate differentiable FP64 formula, including global clipping."""
    norm = torch.cat([gradient.reshape(-1) for gradient in gradients]).norm()
    coefficient = torch.minimum(norm.new_tensor(1.0), max_norm / (norm + 1e-6))
    outputs = []
    beta1, beta2 = options['betas']
    for parameter, gradient, mean, square, step in zip(
        parameters, gradients, means, squares, steps
    ):
        gradient = gradient * coefficient
        corrected_mean = (beta1 * mean + (1 - beta1) * gradient) / (1 - beta1 ** (step + 1))
        corrected_square = (beta2 * square + (1 - beta2) * gradient.square()) / (
            1 - beta2 ** (step + 1)
        )
        outputs.append(
            parameter * (1 - options['lr'] * options['weight_decay'])
            - options['lr'] * corrected_mean / (corrected_square.sqrt() + options['eps'])
        )
    return outputs


class RoleAssessmentTests(unittest.TestCase):
    def test_frontier_example_and_native_validity_edges(self):
        roles = roles_for([1, 1, 1, 0, 1, 1, 0, 1])
        self.assertEqual(
            [int(roles[key]) for key in ('L', 'e', 'H', 'K', 'm')], [3, 4, 3, 6, 8]
        )
        self.assertEqual(
            roles['assessment_weight'].flatten().tolist(), [0, 6, 5, 4, 3, 2, 1, 0, 0]
        )
        for matches, expected in [
            ([1, 1, 1], [3, -1, 0, 3]),
            ([0, 1, 1], [0, 1, 3, 3]),
            ([0, 0, 1], [0, 1, 1, 1]),
        ]:
            role = roles_for(matches)
            self.assertEqual([int(role[key]) for key in ('L', 'e', 'H', 'K')], expected)
        labels = torch.tensor([[[5, 1, 9, 2, 3], [5, 1, 2, 3, 4], [9, 1, 2, 3, 4]]])
        proposals = labels.clone()
        weights = beta_for(labels)
        weights[0, 1, 2] = 0
        roles = M.current_roles(proposals, labels, weights, (9,))
        self.assertEqual(roles['m'].tolist(), [[2, 1, 0]])
        self.assertTrue(roles['valid'][0, 0, 2])
        self.assertFalse(roles['valid'][0, 0, 3])
        self.assertEqual(roles['K'].tolist(), [[2, 1, 0]])

    def test_shortfall_bound_includes_simultaneous_prefix_and_follower_changes(self):
        roles = roles_for([1, 1, 0, 1, 1, 0])
        endpoint = int(roles['K'])
        self.assertEqual(endpoint, 5)
        for after in itertools.product((False, True), repeat=6):
            logits = torch.zeros((1, 1, 7, 2), dtype=torch.bfloat16)
            for position, correct in enumerate(after, 1):
                logits[0, 0, position, 0 if correct else 1] = 2
            assessment, _ = M.assessment(logits, roles)
            length = next((i for i, correct in enumerate(after) if not correct), len(after))
            weighted_errors = sum(
                (endpoint - j + 1) * (not after[j - 1]) for j in range(1, endpoint + 1)
            )
            self.assertLessEqual(max(endpoint - length, 0), weighted_errors)
            self.assertLessEqual(weighted_errors, float(assessment))

    def test_fresh_wrong_rival_native_ties_and_active_branch(self):
        roles = roles_for([0])
        logits = torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, -2.0, 3.0]]]], dtype=torch.bfloat16, requires_grad=True
        )
        assessment, details = M.assessment(logits, roles)
        self.assertEqual(details['rivals'].tolist(), [2])
        self.assertGreater(float(assessment.detach()), 1.0)
        self.assertLess(float(F.softplus(torch.tensor(-2.0)) / math.log(2.0)), 1.0)
        gradient = torch.autograd.grad(assessment, logits)[0]
        self.assertEqual(float(gradient[0, 0, 1, 1]), 0.0)
        self.assertGreater(float(gradient[0, 0, 1, 2]), 0.0)
        tied = torch.tensor(
            [[[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]]], dtype=torch.bfloat16, requires_grad=True
        )
        assessment, details = M.assessment(tied, roles)
        self.assertEqual(details['rivals'].tolist(), [1])
        self.assertEqual(details['margin_ties'], 1)
        self.assertEqual(details['wrong_rival_ties'], 1)
        self.assertTrue(bool(details['correct'][0]))
        gradient = torch.autograd.grad(assessment, tied)[0]
        self.assertGreater(float(gradient[0, 0, 1, 1]), 0.0)
        self.assertEqual(float(gradient[0, 0, 1, 2]), 0.0)
        gathered = torch.tensor([[[[0.0, 0.0], [256.0, -1.0]]]], dtype=torch.bfloat16)
        _, details = M.assessment(gathered, roles)
        self.assertEqual(float(details['margins'][0]), 257.0)

    def test_empty_response_keeps_the_other_half_and_empty_blocks_do_not_dilute(self):
        roles = roles_for([0, 1])
        logits = torch.zeros((1, 1, 3, 2), requires_grad=True)
        value, _ = M.assessment(logits, roles)
        empty = M.current_roles(
            torch.zeros((1, 2, 3), dtype=torch.long),
            torch.zeros((1, 2, 3), dtype=torch.long),
            torch.zeros((1, 2, 3)),
        )
        zero, _ = M.assessment(torch.ones((1, 2, 3, 2), requires_grad=True), empty)
        self.assertEqual(float(zero.detach()), 0.0)
        self.assertEqual(
            float(M.window_mean((value, zero)).detach()), float(value.detach()) / 2
        )
        with self.assertRaisesRegex(ValueError, 'two-response'):
            M.window_mean((value,))

    def test_bf16_token_direction_uses_fp32_coefficient_normalization_once(self):
        roles = roles_for([1, 1, 0])
        direction = torch.tensor([[[0.0, 0.123, -0.245, 0.5]]], dtype=torch.bfloat16)
        scores = M.coefficient_sensitivities(direction, roles)
        self.assertEqual(scores.dtype, torch.float32)
        expected = -direction.float() * roles['prefix'] / (2 * roles['denominator'])
        torch.testing.assert_close(scores, expected, rtol=0, atol=0)

    def test_final_direct_support_detaches_coefficients_but_reference_x_is_live(self):
        roles = roles_for([1, 1, 0, 1, 1])
        logits = torch.randn(
            (1, 1, 6, 3),
            generator=torch.Generator().manual_seed(4),
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        teacher = torch.tensor([0.6, 0.3, 0.1], dtype=torch.bfloat16)
        token_losses = (teacher * (teacher.log() - logits.log_softmax(-1))).sum(-1)
        coefficients = roles['beta'].clone().requires_grad_()
        repair = roles['beta'] * 1.2
        final = M.response_loss(token_losses, coefficients, repair, roles)
        gradient, coefficient_gradient = torch.autograd.grad(
            final, (logits, coefficients), allow_unused=True, retain_graph=True
        )
        self.assertIsNone(coefficient_gradient)
        self.assertEqual(
            int(torch.count_nonzero(gradient[~(roles['prefix'] | roles['repair'])])), 0
        )
        reference = M.response_loss(token_losses, coefficients, repair, roles, scoring=True)
        score_gradient = torch.autograd.grad(reference, coefficients)[0]
        self.assertTrue(bool((score_gradient[roles['prefix']] != 0).all()))
        self.assertEqual(int(torch.count_nonzero(score_gradient[~roles['prefix']])), 0)


class AllocationTests(unittest.TestCase):
    def test_signed_block_local_budget_bounds_variance_and_reference_branches(self):
        scores = torch.tensor(
            [
                [-5.0, -2.0, -3.0, 99.0],
                [2.0, 2.0, 2.0, 0.0],
                [-8.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        beta = torch.tensor([[1.0, 0.8, 0.6, 0.4]]).expand(4, -1).clone()
        prefix = torch.tensor(
            [[1, 1, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.bool
        )
        candidate, diagnostic = M.allocate_prefix(scores.requires_grad_(), beta, prefix)
        self.assertFalse(candidate.requires_grad)
        self.assertGreater(float(candidate[0, 1]), float(beta[0, 1]))
        self.assertLess(float(candidate[0, 0]), float(beta[0, 0]))
        torch.testing.assert_close(candidate[1:], beta[1:], rtol=0, atol=0)
        torch.testing.assert_close(
            (candidate * prefix).sum(-1),
            (beta * prefix).sum(-1),
            rtol=M.MASS_RTOL,
            atol=M.MASS_ATOL,
        )
        torch.testing.assert_close(
            diagnostic['signed_benefit'],
            diagnostic['variance_identity_rhs'],
            rtol=M.MASS_RTOL,
            atol=M.MASS_ATOL,
        )
        self.assertGreater(float(diagnostic['signed_benefit'][0]), 0.0)
        unchanged, _ = M.allocate_prefix(scores, beta, prefix, relative_cap=0.0)
        torch.testing.assert_close(unchanged, beta, rtol=0, atol=0)
        small, _ = M.allocate_prefix(scores.detach() * 1e-9, beta, prefix)
        torch.testing.assert_close(small, candidate, rtol=M.MASS_RTOL, atol=M.MASS_ATOL)

    def test_near_constant_representable_scores_keep_the_cap_without_losing_mass(self):
        unit = torch.tensor(1.0, dtype=torch.float32)
        next_unit = torch.nextafter(unit, torch.tensor(torch.inf))
        next_next = torch.nextafter(next_unit, torch.tensor(torch.inf))
        scores = torch.stack((unit, next_unit, next_next)).reshape(1, 3)
        beta = torch.tensor([[1.0, 0.8, 0.6]])
        candidate, details = M.allocate_prefix(
            scores, beta, torch.ones_like(beta, dtype=torch.bool)
        )
        torch.testing.assert_close(
            candidate.sum(), beta.sum(), rtol=M.MASS_RTOL, atol=M.MASS_ATOL
        )
        self.assertFalse(bool(details['reference_branch']))
        self.assertGreater(float((candidate - beta).abs().sum()), 0.1)
        self.assertGreater(float(details['centered_signed_benefit']), 0.0)
        torch.testing.assert_close(
            details['centered_signed_benefit'],
            details['variance_identity_rhs'],
            rtol=M.MASS_RTOL,
            atol=M.MASS_ATOL,
        )

    def test_nonfinite_inputs_fail_instead_of_becoming_uniform(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                M.allocate_prefix(
                    torch.tensor([[value, 0.0]]),
                    torch.ones(1, 2),
                    torch.ones(1, 2, dtype=torch.bool),
                )

    def test_common_score_offset_preserves_allocation_and_centered_identity(self):
        scores = torch.tensor([[0.0, 0.125, 0.25]])
        beta = torch.tensor([[1.0, 0.8, 0.6]])
        prefix = torch.ones_like(beta, dtype=torch.bool)
        reference, _ = M.allocate_prefix(scores, beta, prefix)
        shifted, details = M.allocate_prefix(scores + 1e6, beta, prefix)
        torch.testing.assert_close(shifted, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            details['centered_signed_benefit'],
            details['variance_identity_rhs'],
            rtol=M.MASS_RTOL,
            atol=M.MASS_ATOL,
        )
        self.assertGreater(float(details['centered_signed_benefit']), 0.0)
        self.assertGreater(
            float((details['signed_benefit'] - details['centered_signed_benefit']).abs()),
            M.MASS_ATOL,
        )


class AdamShadowTests(unittest.TestCase):
    def test_preview_and_vjp_match_independent_fp64_and_native_adam_both_clip_branches(self):
        generator = torch.Generator().manual_seed(81)
        parameters = [
            torch.randn(shape, generator=generator, dtype=torch.float64)
            for shape in ((3,), (2, 2))
        ]
        means = [
            torch.randn(p.shape, generator=generator, dtype=torch.float64) * 0.1
            for p in parameters
        ]
        squares = [
            torch.rand(p.shape, generator=generator, dtype=torch.float64) + 0.2
            for p in parameters
        ]
        steps = [3, 7]
        options = dict(lr=0.03, weight_decay=0.07, betas=(0.8, 0.91), eps=0.02)
        for scale in (0.01, 4.0):
            with self.subTest(scale=scale):
                gradients = [
                    torch.randn(p.shape, generator=generator, dtype=torch.float64) * scale
                    for p in parameters
                ]
                cotangents = [
                    torch.randn(p.shape, generator=generator, dtype=torch.float64)
                    for p in parameters
                ]
                originals = [
                    [x.clone() for x in values]
                    for values in (parameters, gradients, means, squares)
                ]
                preview, context = M.adamw_shadow(
                    parameters,
                    gradients,
                    means,
                    squares,
                    steps,
                    options,
                    max_grad_norm=1.0,
                    dtype=torch.float64,
                )
                differentiable_gradients = [g.clone().requires_grad_() for g in gradients]
                independent = independent_adam(
                    parameters, differentiable_gradients, means, squares, steps, options, 1.0
                )
                expected = torch.autograd.grad(
                    sum(
                        (value * cotangent).sum()
                        for value, cotangent in zip(independent, cotangents)
                    ),
                    differentiable_gradients,
                )
                actual = M.optimizer_vjp(context, cotangents)
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=2e-12, atol=2e-12)
                native_params = [torch.nn.Parameter(p.clone()) for p in parameters]
                optimizer = torch.optim.AdamW(native_params, **options, foreach=False)
                for p, gradient, mean, square, step in zip(
                    native_params, gradients, means, squares, steps
                ):
                    p.grad = gradient.clone()
                    optimizer.state[p].update(
                        step=torch.tensor(float(step)),
                        exp_avg=mean.clone(),
                        exp_avg_sq=square.clone(),
                    )
                torch.nn.utils.clip_grad_norm_(native_params, 1.0)
                optimizer.step()
                for actual, expected in zip(preview, native_params):
                    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
                for before, after in zip(originals, (parameters, gradients, means, squares)):
                    for left, right in zip(before, after):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertEqual(context['clipping_active'], scale == 4.0)

    def test_fp32_promotion_and_global_clipping_couple_parameter_groups(self):
        parameters = [
            torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            torch.tensor([3.0], dtype=torch.bfloat16),
        ]
        gradients = [
            torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
            torch.tensor([2.0], dtype=torch.bfloat16),
        ]
        means = [torch.full_like(p, 0.2) for p in parameters]
        squares = [torch.full_like(p, 0.4) for p in parameters]
        options = dict(lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        preview, context = M.adamw_shadow(
            parameters, gradients, means, squares, [128, 128], options, max_grad_norm=1.0
        )
        self.assertTrue(all(p.dtype == torch.float32 and p.requires_grad for p in preview))
        direction = M.optimizer_vjp(context, [torch.ones(2), torch.zeros(1)])
        self.assertGreater(float(direction[1].abs().sum()), 0.0)

    def test_finite_gradient_norm_overflow_rejects_like_native_clipping(self):
        zero = torch.zeros(3)
        options = dict(lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        with self.assertRaisesRegex(ValueError, 'global gradient norm'):
            M.adamw_shadow(
                [zero],
                [torch.full_like(zero, 1e30)],
                [zero],
                [torch.ones(3)],
                [128],
                options,
                max_grad_norm=1.0,
            )
        _, context = M.adamw_shadow(
            [zero], [zero], [zero], [torch.ones(3)], [128], options, max_grad_norm=1.0
        )
        with self.assertRaisesRegex(ValueError, 'cotangent shape'):
            M.optimizer_vjp(context, [torch.ones(1)])

    def test_exact_zero_coordinate_limit_and_nondifferentiable_state_rejection(self):
        zero = torch.zeros(2)
        options = dict(lr=0.03, weight_decay=0.01, betas=(0.8, 0.91), eps=0.02)
        _, context = M.adamw_shadow(
            [zero], [zero], [zero], [zero], [5], options, max_grad_norm=1.0
        )
        derivative = M.optimizer_vjp(context, [torch.ones(2)])[0]
        expected = -0.03 / (1 - 0.8**6) * (1 - 0.8) / 0.02
        torch.testing.assert_close(derivative, torch.full_like(derivative, expected))
        with self.assertRaisesRegex(ValueError, 'no unique derivative'):
            M.adamw_shadow(
                [zero], [zero], [torch.ones(2)], [zero], [5], options, max_grad_norm=1.0
            )


class SharedParameterSensitivityTests(unittest.TestCase):
    def test_matrix_free_scores_equal_direct_coefficient_ad_and_include_continuation(self):
        theta = torch.tensor(
            [[1.1, -0.7, 0.2], [-0.2, 0.8, -0.5]], dtype=torch.float64, requires_grad=True
        )
        features = [
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.5], [1.0, -0.4], [-0.5, 1.0], [0.8, -0.2], [0.3, 0.7]],
                dtype=torch.float64,
            ),
            torch.tensor(
                [[0.0, 0.0], [0.7, 0.2], [-0.4, 0.9], [0.9, 0.1], [0.2, 0.8], [0.4, 0.5]],
                dtype=torch.float64,
            ),
        ]
        roles, teachers = [], []
        for index, feature in enumerate(features):
            proposals = (feature @ theta).argmax(-1).reshape(1, 1, -1)
            labels = proposals.clone()
            labels[..., 3 if index == 0 else 2] = (labels[..., 3 if index == 0 else 2] + 1) % 3
            labels[..., 5] = (labels[..., 5] + 1) % 3
            roles.append(M.current_roles(proposals, labels, beta_for(labels)))
            teachers.append(F.one_hot(labels, 3).double() * 0.7 + 0.1)
        self.assertNotEqual(float(roles[0]['denominator']), float(roles[1]['denominator']))

        def losses(parameter, index):
            logits = (features[index] @ parameter).reshape(1, 1, 6, 3)
            teacher = teachers[index]
            return (teacher * (teacher.log() - logits.log_softmax(-1))).sum(-1)

        def risk(parameter, continuation_only=False):
            values = []
            for index in (0, 1):
                logits = (features[index] @ parameter).reshape(1, 1, 6, 3)
                frozen = roles[index]
                if continuation_only:
                    frozen = {
                        **frozen,
                        'assessment_mask': frozen['assessment_mask']
                        & ~(frozen['prefix'] | frozen['repair']),
                    }
                value, _ = M.assessment(logits, frozen)
                values.append(value)
            return M.window_mean(values)

        coefficients = [role['beta'].double().requires_grad_() for role in roles]
        terms = [
            M.response_loss(
                losses(theta, index),
                coefficients[index],
                roles[index]['beta'] * 1.1,
                roles[index],
                scoring=True,
            )
            for index in (0, 1)
        ]
        gradient = torch.autograd.grad(M.window_mean(terms), theta, create_graph=True)[0]
        mean = torch.full_like(theta, 0.04)
        square = torch.full_like(theta, 0.2)
        options = dict(lr=0.04, weight_decay=0.03, betas=(0.8, 0.93), eps=0.01)
        preview_direct = independent_adam(
            [theta.detach()], [gradient], [mean], [square], [9], options, 0.2
        )[0]
        direct = torch.autograd.grad(risk(preview_direct), coefficients, retain_graph=True)
        preview, context = M.adamw_shadow(
            [theta],
            [gradient],
            [mean],
            [square],
            [9],
            options,
            max_grad_norm=0.2,
            dtype=torch.float64,
        )
        outer = torch.autograd.grad(risk(preview[0]), preview)[0]
        direction = M.optimizer_vjp(context, [outer])[0]
        for index in (0, 1):
            _, directional = torch.func.jvp(
                lambda value: losses(value, index), (theta.detach(),), (direction,)
            )
            score = M.coefficient_sensitivities(directional, roles[index])
            torch.testing.assert_close(score, -direct[index], rtol=2e-6, atol=2e-8)
            self.assertGreater(float(score[roles[index]['prefix']].abs().sum()), 0.0)
        continuation_outer = torch.autograd.grad(
            risk(preview[0], continuation_only=True), preview
        )[0]
        continuation_direction = M.optimizer_vjp(context, [continuation_outer])[0]
        _, continuation_effect = torch.func.jvp(
            lambda value: losses(value, 0), (theta.detach(),), (continuation_direction,)
        )
        # Assessment at OTHER positions changes prefix coefficient sensitivities
        # because all positions share theta; independent-logit scoring misses it.
        self.assertGreater(float(continuation_effect[roles[0]['prefix']].abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
