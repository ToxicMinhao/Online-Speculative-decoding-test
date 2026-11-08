"""Small CPU oracles for FSG; these do not substitute for native qualification."""
from decimal import Decimal, localcontext
import math
import unittest

import torch

from online_feedback.functional_geometry_math import (
    Risk, endpoint_statistics, expanded_terms, gate, geometry_terms,
    make_expanded_sketch, make_sketch, normalize_weights, pooled_risk,
    prefix_statistics, risk_terms, scalar_match, vector_dot, write_candidate,
)


class PrefixWeightsTests(unittest.TestCase):
    def test_correct_prefix_excludes_seed_first_error_and_later_matches(self):
        labels = torch.tensor([[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]])
        proposal = labels.clone(); proposal[0, 0, 3] = 0; proposal[0, 1, 1] = 0
        probability = torch.full_like(labels, .6, dtype=torch.float32)
        result = prefix_statistics(proposal, labels, torch.ones_like(probability), probability, probability-.2)
        self.assertEqual(result['triangular'].tolist(), [[[0, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 4, 3, 2, 1]]])
        self.assertEqual(result['mask'].sum().item(), 6)

    def test_eos_seed_matching_eos_and_padding_boundaries(self):
        labels = torch.tensor([[[0, 1, 9, 3, 4], [9, 1, 2, 3, 4], [0, 1, 2, 3, 4]]])
        valid = torch.ones_like(labels).float(); valid[0, 2, 2] = 0
        result = prefix_statistics(labels, labels, valid, valid*.5, valid*.25, (9,))
        self.assertEqual(result['triangular'].tolist(), [[[0, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]])

    def test_native_ties_can_be_correct_and_use_margin_floor(self):
        logits = torch.tensor([[[[0., 1.], [1., 1.], [1., 1.]]]], dtype=torch.bfloat16)
        labels = torch.tensor([[[0, 0, 1]]]); prob = logits.float().softmax(-1)
        result = prefix_statistics(logits.argmax(-1), labels, torch.ones_like(labels).float(),
            prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1), torch.full_like(labels, .5, dtype=torch.float32))
        self.assertEqual(result['mask'].tolist(), [[[False, True, False]]])
        self.assertAlmostEqual(result['denominator'].item(), 400.)

    def test_margin_weight_formula_and_response_wide_chunk_normalization(self):
        labels = torch.tensor([[[0, 1, 2], [0, 1, 2]]])
        margin = torch.tensor([[[.2, .1, .05], [.2, .02, .4]]])
        parts = [prefix_statistics(labels[:, i:i+1], labels[:, i:i+1], torch.ones_like(margin[:, i:i+1]),
                 margin[:, i:i+1]+.2, torch.full_like(margin[:, i:i+1], .2)) for i in range(2)]
        denominator = sum(p['denominator'] for p in parts)
        weights = torch.cat([normalize_weights(p['raw_weights'], denominator) for p in parts], 1)
        expected = torch.tensor([[[0., 200., 400.], [0., 800., 6.25]]])
        expected /= expected.sum()
        torch.testing.assert_close(weights, expected, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(float(weights.sum()), 1., places=6)
        self.assertNotAlmostEqual(float(weights[:, 0].sum()), .5)

    def test_empty_response_stays_zero_and_detached(self):
        labels = torch.tensor([[[0, 1, 2]]]); proposal = torch.zeros_like(labels)
        prob = torch.ones_like(labels).float().requires_grad_()
        result = prefix_statistics(proposal, labels, prob, prob, prob*.5)
        weights = normalize_weights(result['raw_weights'], result['denominator'])
        self.assertEqual(float(weights.sum()), 0.)
        self.assertFalse(weights.requires_grad)


class GeometryTests(unittest.TestCase):
    def test_cotangent_and_vjp_match_dense_softmax_covariance(self):
        generator = torch.Generator().manual_seed(17)
        jacobian = torch.randn(3, 5, 7, generator=generator)
        theta = torch.randn(7, generator=generator, requires_grad=True)
        native = theta.detach()+torch.randn(7, generator=generator)*.03
        z0 = torch.einsum('ivp,p->iv', jacobian, theta)
        z1 = torch.einsum('ivp,p->iv', jacobian, native)
        weights = torch.tensor([.2, .0, .8])
        s, c = geometry_terms(z0, z1, weights)
        self.assertFalse(s.requires_grad); self.assertFalse(c.requires_grad)
        actual = torch.autograd.grad((z0*s).sum(), theta)[0]
        covariance = []
        for q, w in zip(z0.detach().softmax(-1), weights):
            covariance.append(w*(q.diag()-q[:, None]*q[None, :]))
        expected_s = torch.stack([matrix@(right-left).detach() for matrix, left, right in zip(covariance, z0, z1)])
        expected_gradient = torch.einsum('ivp,iv->p', jacobian, expected_s)
        expected_c = sum((right-left).detach()@matrix@(right-left).detach()
                         for matrix, left, right in zip(covariance, z0, z1))
        torch.testing.assert_close(actual, expected_gradient, rtol=2e-5, atol=2e-7)
        self.assertAlmostEqual(float(c), float(expected_c), places=7)

    def test_chunk_reduction_matches_full_weighted_response(self):
        torch.manual_seed(2)
        z0 = torch.randn(1, 5, 4, 9, dtype=torch.bfloat16)
        z1 = z0+torch.randn_like(z0)*.125
        weights = torch.rand(1, 5, 4); weights /= weights.sum()
        full_s, full_c = geometry_terms(z0, z1, weights)
        chunks = [geometry_terms(z0[:, k:k+2], z1[:, k:k+2], weights[:, k:k+2]) for k in (0, 2, 4)]
        torch.testing.assert_close(torch.cat([s for s, _ in chunks], 1), full_s, rtol=0, atol=0)
        self.assertAlmostEqual(sum(float(c) for _, c in chunks), float(full_c), places=14)

    def test_constant_logit_shift_and_empty_weights_have_zero_energy(self):
        z = torch.zeros(3, 4); weights = torch.ones(3)/3
        s, c = geometry_terms(z, z+2, weights)
        self.assertEqual(float(c), 0.); self.assertEqual(float(s.abs().sum()), 0.)
        s, c = geometry_terms(z, torch.randn_like(z), weights*0)
        self.assertEqual(float(c), 0.); self.assertEqual(float(s.abs().sum()), 0.)

    def test_nonfinite_logits_stop(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            geometry_terms(torch.zeros(1, 2), torch.tensor([[0., float('nan')]]), torch.ones(1))


class RiskTests(unittest.TestCase):
    @staticmethod
    def decimal_kl(a, b):
        with localcontext() as context:
            context.prec = 80
            a = [Decimal(str(float(x))) for x in a]
            b = [Decimal(str(float(x))) for x in b]
            za = sum(x.exp() for x in a); zb = sum(x.exp() for x in b)
            return float(sum((x.exp()/za)*((x-za.ln())-(y-zb.ln())) for x, y in zip(a, b)))

    def test_risk_matches_80_digit_oracle_inside_fixed_deadband(self):
        for offset in (0., 1e-7, .1, 5.):
            z0 = torch.tensor([[1.25, -.5, 3.]], dtype=torch.float64)
            z1 = z0+torch.tensor([[offset, -offset, offset/3]], dtype=torch.float64)
            risk = pooled_risk([risk_terms(z0, z1, torch.ones(1))], 3)
            expected = .5*self.decimal_kl(z0[0], z1[0])
            self.assertLessEqual(abs(risk.raw-expected), risk.eta)

    def test_64_native_bf16_small_vocab_risks_match_independent_oracle(self):
        generator = torch.Generator().manual_seed(20261108)
        for case in range(64):
            scale = (1., 8., 32., 128.)[case % 4]
            delta_scale = (1e-4, 1e-2, 1., 20.)[(case // 4) % 4]
            z0 = (torch.randn(3, 7, generator=generator)*scale).bfloat16()
            z1 = (z0.float()+torch.randn(3, 7, generator=generator)*delta_scale).bfloat16()
            weights = torch.rand(3, generator=generator); weights /= weights.sum()
            risk = pooled_risk([risk_terms(z0, z1, weights)], 7)
            expected = .5*math.fsum(float(weight)*self.decimal_kl(a, b)
                                   for weight, a, b in zip(weights, z0, z1))
            self.assertLessEqual(abs(risk.raw-expected), risk.eta, 'fixed case '+str(case))

    def test_empty_second_response_does_not_renormalize_first(self):
        z0 = torch.tensor([[1., 0.]]); z1 = torch.tensor([[0., 1.]])
        part = risk_terms(z0, z1, torch.ones(1))
        zero = risk_terms(z0, z1, torch.zeros(1))
        risk = pooled_risk([part, zero], 2)
        self.assertAlmostEqual(risk.raw, float(part[0])*.5)

    def test_chunk_and_full_risk_agree_and_identical_endpoint_is_zero(self):
        torch.manual_seed(4)
        z0 = torch.randn(5, 7, dtype=torch.bfloat16); z1 = z0+torch.randn_like(z0)
        weights = torch.rand(5); weights /= weights.sum()
        full = pooled_risk([risk_terms(z0, z1, weights)], 7)
        chunked = pooled_risk([risk_terms(z0[k:k+2], z1[k:k+2], weights[k:k+2]) for k in (0, 2, 4)], 7)
        self.assertAlmostEqual(full.raw, chunked.raw, places=15)
        self.assertEqual(pooled_risk([risk_terms(z0, z0, weights)], 7).value, 0.)

    def test_negative_roundoff_logged_but_large_negative_stops(self):
        small = pooled_risk([(-1e-16, 1.)], 3)
        self.assertEqual(small.value, 0.); self.assertLess(small.raw, 0.)
        with self.assertRaisesRegex(ValueError, 'Negative'):
            pooled_risk([(-1., 1.)], 3)

    def test_actual_gate_branches_and_identical_nonzero_risk(self):
        native = Risk(.2, .2, 1e-10, 1.)
        good = Risk(.1, .1, 1e-10, 1.)
        self.assertTrue(gate(good, native, 1., 2.)['accepted'])
        self.assertEqual(gate(good, native, 3., 2.)['reason'], 'rounded_norm_exceeds_native')
        self.assertEqual(gate(native, native, 1., 2.)['reason'], 'numerical_ambiguity')
        self.assertTrue(gate(native, native, 2., 2., identical=True)['accepted'])
        self.assertEqual(gate(native, good, 1., 2.)['reason'], 'actual_risk_increased')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            gate(good, native, float('nan'), 2.)


class SketchTests(unittest.TestCase):
    def setUp(self):
        self.old = [torch.tensor([.1, .2], dtype=torch.bfloat16), torch.tensor([.3], dtype=torch.bfloat16)]
        self.native = [self.old[0]+torch.tensor([.01, .02], dtype=torch.bfloat16), self.old[1]-.015]
        self.gradients = [(torch.tensor([1., 2.], dtype=torch.bfloat16), torch.tensor([3.], dtype=torch.bfloat16)),
                          (torch.tensor([-1., 1.], dtype=torch.bfloat16), torch.tensor([.5], dtype=torch.bfloat16))]

    def test_streamed_gram_and_solve_match_dense_oracle(self):
        sketch = make_sketch(self.gradients, [2., 3.], self.old, self.native, chunk_size=1)
        basis = torch.stack([torch.cat([x.double() for x in g])/math.sqrt(2*c)
                             for g, c in zip(self.gradients, [2., 3.])], 1)
        u = torch.cat([(new.float()-old.float()).double() for old, new in zip(self.old, self.native)])
        dense_gram = basis.T@basis
        sigma = float(torch.linalg.eigvalsh(dense_gram)[-1])
        exact = torch.linalg.solve(torch.eye(3, dtype=torch.float64)+basis@basis.T/sigma, u)
        actual = u-basis@sketch.coefficients
        torch.testing.assert_close(sketch.gram, dense_gram, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(actual, exact, rtol=1e-13, atol=1e-15)
        self.assertLessEqual(sketch.statistics['condition_number'], 2.+1e-14)
        self.assertLessEqual(float(actual.norm()), float(u.norm())+1e-15)

    def test_round_candidate_preserves_parameter_identity(self):
        sketch = make_sketch(self.gradients, [2., 3.], self.old, self.native)
        params = [torch.nn.Parameter(x.clone()) for x in self.native]; objects = [id(p) for p in params]
        expected = []
        for index, (old, native) in enumerate(zip(self.old, self.native)):
            delta = native.float()-old.float()
            for gradient, scale, coef in zip(sketch.gradients, sketch.normalizers, sketch.coefficients):
                delta -= gradient[index].float()*float(scale*coef)
            expected.append((old.float()+delta).bfloat16())
        stats = write_candidate(params, self.old, self.native, sketch, chunk_size=1)
        self.assertEqual(objects, [id(p) for p in params])
        for p, e in zip(params, expected): self.assertTrue(torch.equal(p, e))
        self.assertEqual(stats['norm'], math.sqrt(sum(float((p.detach().float()-o.float()).double().square().sum())
                                                     for p, o in zip(params, self.old))))

    def test_zero_lambda_copies_exact_native_without_interpolation(self):
        sketch = make_sketch(self.gradients, [2., 3.], self.old, self.native, strength=0.)
        params = [torch.nn.Parameter(x.clone()*0) for x in self.native]
        stats = write_candidate(params, self.old, self.native, sketch)
        self.assertEqual(stats['bypass'], 'lambda_zero')
        for p, n in zip(params, self.native): self.assertTrue(torch.equal(p, n))

    def test_zero_energy_gradient_and_empty_sketch_are_logged_bypasses(self):
        zero = tuple(torch.zeros_like(x) for x in self.old)
        sketch = make_sketch([self.gradients[0], zero], [0., 1.], self.old, self.native)
        self.assertEqual(sketch.sigma, 0.)
        self.assertEqual([x['bypass'] for x in sketch.statistics['responses']], ['zero_output_energy', 'zero_gradient'])
        params = [x.clone() for x in self.old]
        self.assertEqual(write_candidate(params, self.old, self.native, sketch)['bypass'], 'empty_sketch')

    def test_none_gradient_is_zero_and_native_zero_coordinates_can_move(self):
        old = [torch.zeros(2, dtype=torch.bfloat16), torch.zeros(1, dtype=torch.bfloat16)]
        native = [torch.tensor([1., 0.], dtype=torch.bfloat16), old[1].clone()]
        sketch = make_sketch([(torch.tensor([1., 1.], dtype=torch.bfloat16), None)], [1.], old, native)
        params = [x.clone() for x in native]
        stats = write_candidate(params, old, native, sketch)
        self.assertEqual(stats['changed_zero_native_coordinates'], 1)
        self.assertTrue(torch.equal(params[1], native[1]))
        self.assertEqual(params[0].tolist(), [.75, -.25])

    def test_native_gradient_storage_is_not_promoted_or_mutated(self):
        before = [[x.clone() for x in g] for g in self.gradients]
        sketch = make_sketch(self.gradients, [2., 3.], self.old, self.native)
        write_candidate([x.clone() for x in self.old], self.old, self.native, sketch)
        for actual, saved in zip(self.gradients, before):
            for a, s in zip(actual, saved):
                self.assertEqual(a.dtype, torch.bfloat16); self.assertTrue(torch.equal(a, s))

    def test_nonfinite_vjp_is_failure_even_with_zero_energy(self):
        bad = [(torch.full_like(self.old[0], float('nan')), None)]
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            make_sketch(bad, [0.], self.old, self.native)

    def test_expanded_covariance_uses_cross_terms(self):
        z0 = torch.tensor([[0., 1., -1.], [1., 0., 2.]])
        z1 = z0+torch.tensor([[.5, 0., -.5], [.25, -.25, .5]])
        z2 = z0+2*(z1-z0)
        (s1, s2), c = expanded_terms(z0, z1, z2, torch.tensor([.25, .75]))
        torch.testing.assert_close(s2, 2*s1)
        self.assertAlmostEqual(float(c[0, 1]), float(c[0, 0])*2)
        self.assertAlmostEqual(float(c[1, 1]), float(c[0, 0])*4)

    def test_expanded_whitening_matches_joint_pseudoinverse_not_outer_sum(self):
        grads = [self.gradients]
        covariance = [torch.tensor([[2., 1.], [1., 3.]], dtype=torch.float64)]
        production = make_sketch([self.gradients[0]], [2.], self.old, self.native)
        expanded = make_expanded_sketch(grads, covariance, self.old, self.native, production_sigma=production.sigma)
        dense_g = torch.stack([torch.cat([x.double() for x in g]) for g in self.gradients], 1)
        matrix = .5*dense_g@torch.linalg.pinv(covariance[0])@dense_g.T
        u = torch.cat([(n.float()-o.float()).double() for o, n in zip(self.old, self.native)])
        expected = torch.linalg.solve(torch.eye(3, dtype=torch.float64)+matrix/production.sigma, u)
        actual = u-dense_g@expanded.coefficients
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
        self.assertEqual(expanded.sigma, production.sigma)

    def test_expanded_duplicate_direction_is_discarded_and_zero_scale_bypasses(self):
        gradients = [[self.gradients[0], tuple(x*2 for x in self.gradients[0])]]
        covariance = [torch.tensor([[1., 2.], [2., 4.]])]
        result = make_expanded_sketch(gradients, covariance, self.old, self.native, production_sigma=0.)
        self.assertEqual(result.statistics['usable_columns'], 1)
        self.assertEqual(result.sigma, 0.)
        self.assertEqual(float(result.coefficients.abs().sum()), 0.)


class ScalarTests(unittest.TestCase):
    def test_rounding_mismatch_is_reported_not_hidden(self):
        old = [torch.tensor([.05, .05], dtype=torch.bfloat16)]
        native = [(old[0].float()+3e-4).bfloat16()]
        params = [native[0].clone()]
        target = float((native[0].float()-old[0].float())[0])
        stats = scalar_match(params, old, native, target)
        self.assertEqual(stats['iterations'], 24)
        self.assertGreater(stats['relative_norm_mismatch'], .4)
        self.assertEqual(stats['norm'], stats['native_norm'])
        self.assertLessEqual(stats['bracket'][1]-stats['bracket'][0], 2**-24)

    def test_scalar_zero_and_native_norm_targets_use_exact_endpoints(self):
        old = [torch.tensor([.01, -.03, .05], dtype=torch.bfloat16)]
        native = [(old[0].float()+torch.tensor([1e-4, -1e-3, .02])).bfloat16()]
        norm = endpoint_statistics(native, old, native)['native_norm']
        for target, expected in ((0., old[0]), (norm, native[0])):
            params = [native[0].clone()]
            result = scalar_match(params, old, native, target)
            self.assertTrue(torch.equal(params[0], expected))
            self.assertEqual(result['absolute_norm_mismatch'], 0.)

    def test_scalar_final_bracket_chooses_closer_norm_and_tie_lower_alpha(self):
        old = [torch.zeros(2, dtype=torch.bfloat16)]; native = [torch.ones(2, dtype=torch.bfloat16)]
        params = [native[0].clone()]
        result = scalar_match(params, old, native, .7)
        errors = [abs(norm-.7) for norm in result['bracket_norms']]
        chosen = 0 if errors[0] <= errors[1] else 1
        self.assertEqual(result['alpha'], result['bracket'][chosen])
        self.assertEqual(result['absolute_norm_mismatch'], errors[chosen])

    def test_scalar_invalid_target_or_changed_iteration_count_stops(self):
        old = [torch.zeros(1, dtype=torch.bfloat16)]; native = [torch.ones(1, dtype=torch.bfloat16)]
        with self.assertRaises(ValueError): scalar_match([native[0].clone()], old, native, 2.)
        with self.assertRaises(ValueError): scalar_match([native[0].clone()], old, native, .5, iterations=12)
        with self.assertRaises(ValueError): scalar_match([native[0].clone()], old, native, float('nan'))


if __name__ == '__main__':
    unittest.main()
