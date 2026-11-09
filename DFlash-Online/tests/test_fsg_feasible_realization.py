"""Small BF16 realization oracles; these do not replace native qualification."""
import math
import unittest
from unittest.mock import patch

import torch

from online_feedback import functional_geometry_math as geometry


def fixed_direction(old, native, delta):
    """A prescribed continuous sketch direction for rounding-boundary fixtures."""
    gradient = tuple(n.float()-o.float()-d.float() for o, n, d in zip(old, native, delta))
    return geometry.Sketch((gradient,), (1.,), torch.ones(1, 1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64), 1., 1., {})


def dense_endpoint(old, native, sketch, alpha):
    """Direct small-vector oracle, independent of the streamed construction."""
    result = []
    for index, (o, n) in enumerate(zip(old, native)):
        displacement = n.float()-o.float()
        for gradients, normalizer, coefficient in zip(
                sketch.gradients, sketch.normalizers, sketch.coefficients):
            if gradients[index] is not None:
                displacement = displacement-gradients[index].float()*float(normalizer*coefficient)
        result.append(o.clone() if alpha == 0 else (o.float()+alpha*displacement).to(o.dtype))
    return result


def norm(endpoint, old):
    return math.sqrt(sum(float((p.float()-o.float()).double().square().sum())
                         for p, o in zip(endpoint, old)))


class FeasibleRealizationTests(unittest.TestCase):
    def setUp(self):
        self.old = [torch.tensor([1., -1., 1., -1.], dtype=torch.bfloat16)]
        self.native = [torch.tensor([1.0078125, -1.0078125, 1., -1.], dtype=torch.bfloat16)]
        # Continuous contraction with new coordinates crossing BF16 half-ULPs.
        self.delta = [torch.tensor([.006, -.006, .004, -.004])]
        self.sketch = fixed_direction(self.old, self.native, self.delta)

    def parameters(self):
        return [torch.nn.Parameter(value.clone()) for value in self.native]

    def test_raw_candidate_preserves_original_arithmetic_and_parameters(self):
        parameters = self.parameters()
        identities = [id(p) for p in parameters]
        gradients = [torch.ones_like(p) for p in parameters]
        for p, g in zip(parameters, gradients):
            p.grad = g
        raw = geometry.write_candidate(parameters, self.old, self.native, self.sketch, chunk_size=2)
        expected = dense_endpoint(self.old, self.native, self.sketch, 1.)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters, expected)))
        self.assertEqual([id(p) for p in parameters], identities)
        self.assertTrue(all(p.grad is g for p, g in zip(parameters, gradients)))
        self.assertEqual(raw['norm'], norm(expected, self.old))

    def test_probes_and_writes_share_original_not_rounded_direction(self):
        alpha = .75
        expected = dense_endpoint(self.old, self.native, self.sketch, alpha)
        raw = dense_endpoint(self.old, self.native, self.sketch, 1.)
        wrong = [(o.float()+alpha*(r.float()-o.float())).bfloat16()
                 for o, r in zip(self.old, raw)]
        self.assertFalse(torch.equal(expected[0], wrong[0]))
        measured = geometry.candidate_norm(self.old, self.native, self.sketch, alpha, chunk_size=1)
        self.assertEqual(measured, norm(expected, self.old))
        parameters = self.parameters()
        written = geometry.write_scaled_candidate(parameters, self.old, self.native,
                                                  self.sketch, alpha, 1)
        self.assertEqual(written['norm'], measured)
        self.assertTrue(torch.equal(parameters[0], expected[0]))

    def test_exact_bf16_half_ulp_boundary_uses_native_rounding(self):
        old = [torch.tensor([1., -1.], dtype=torch.bfloat16)]
        native = [torch.tensor([1.0078125, -1.0078125], dtype=torch.bfloat16)]
        sketch = fixed_direction(old, native, [native[0].float()-old[0].float()])
        self.assertEqual(geometry.candidate_norm(old, native, sketch, .5), 0.)
        self.assertGreater(geometry.candidate_norm(old, native, sketch, .5001), 0.)
        parameters = [native[0].clone()]
        geometry.write_scaled_candidate(parameters, old, native, sketch, .5)
        self.assertTrue(torch.equal(parameters[0], old[0]))

    def test_identity_uses_actual_weights_despite_same_fp32_displacements(self):
        old = [torch.tensor([1e30], dtype=torch.bfloat16)]
        native = [torch.tensor([1.], dtype=torch.bfloat16)]
        candidate = [torch.tensor([2.], dtype=torch.bfloat16)]
        self.assertTrue(torch.equal(candidate[0].float()-old[0].float(),
                                    native[0].float()-old[0].float()))
        identity = geometry.endpoint_identities(candidate, old, native)
        self.assertFalse(identity['identical_native'])
        self.assertFalse(identity['identical_prestate'])
        self.assertEqual(identity['different_native_coordinates'], 1)

    def test_positive_negative_coordinates_give_monotone_rounded_norm(self):
        scales = [0., .25, .5, .75, .9765625, 1.]
        norms = [geometry.candidate_norm(self.old, self.native, self.sketch, a, chunk_size=2)
                 for a in scales]
        self.assertEqual(norms, sorted(norms))
        self.assertEqual(norms[0], 0.)
        for alpha, actual in zip(scales, norms):
            self.assertEqual(actual, norm(dense_endpoint(self.old, self.native, self.sketch, alpha), self.old))

    def test_infeasible_raw_uses_exactly_24_probes_and_feasible_lower_endpoint(self):
        parameters = self.parameters()
        with patch.object(geometry, 'candidate_norm', wraps=geometry.candidate_norm) as measured:
            result = geometry.realize_feasible_candidate(parameters, self.old, self.native,
                                                        self.sketch, chunk_size=2)
        self.assertTrue(result['search_ran'])
        self.assertEqual(measured.call_count, 24)
        self.assertEqual(result['iterations'], 24)
        self.assertEqual(result['probe_count'], 24)
        self.assertEqual(result['norm_scan_passes'], 26)
        self.assertEqual(result['write_passes'], 2)
        lo, hi = result['bracket']
        self.assertEqual(result['alpha'], lo)
        self.assertEqual(hi-lo, 2.**-24)
        self.assertLessEqual(result['bracket_norms'][0], result['raw_candidate']['native_norm'])
        self.assertGreater(result['bracket_norms'][1], result['raw_candidate']['native_norm'])
        self.assertEqual(result['candidate']['norm'], result['bracket_norms'][0])
        self.assertTrue(torch.equal(parameters[0], dense_endpoint(self.old, self.native, self.sketch, lo)[0]))
        self.assertTrue(all(math.isfinite(v) and v >= 0 for v in result['costs'].values()))

    def test_already_feasible_raw_candidate_is_exact_and_never_searched(self):
        sketch = fixed_direction(self.old, self.native, [d*.5 for d in self.delta])
        parameters = self.parameters()
        with patch.object(geometry, 'candidate_norm', side_effect=AssertionError('unexpected probe')):
            result = geometry.realize_feasible_candidate(parameters, self.old, self.native, sketch)
        self.assertFalse(result['search_ran'])
        self.assertEqual(result['alpha'], 1.)
        self.assertIs(result['candidate'], result['raw_candidate'])
        self.assertEqual(result['write_passes'], 1)
        self.assertTrue(torch.equal(parameters[0], dense_endpoint(self.old, self.native, sketch, 1.)[0]))

    def test_zero_scale_is_exact_prestate_with_nonzero_native_update(self):
        old = [torch.zeros(1, dtype=torch.bfloat16)]
        native = [torch.tensor([1e-10], dtype=torch.bfloat16)]
        sketch = fixed_direction(old, native, [torch.ones(1)])
        parameters = [value.clone() for value in native]
        result = geometry.realize_feasible_candidate(parameters, old, native, sketch)
        self.assertEqual(result['alpha'], 0.)
        self.assertTrue(result['candidate']['identical_prestate'])
        self.assertFalse(result['candidate']['identical_native'])
        self.assertTrue(torch.equal(parameters[0], old[0]))
        # Final-arm policy is intentionally the caller's responsibility.

    def test_lambda_zero_and_empty_sketch_keep_exact_native_without_probes(self):
        for strength, sigma, reason in [(0., 1., 'lambda_zero'), (1., 0., 'empty_sketch')]:
            sketch = geometry.Sketch((), (), torch.zeros(0, 0), torch.zeros(0), sigma, strength, {})
            parameters = self.parameters()
            with patch.object(geometry, 'candidate_norm', side_effect=AssertionError('unexpected probe')):
                result = geometry.realize_feasible_candidate(parameters, self.old, self.native, sketch)
            self.assertEqual(result['bypass'], reason)
            self.assertTrue(result['candidate']['identical_native'])
            self.assertTrue(torch.equal(parameters[0], self.native[0]))

    def test_zero_native_displacement_keeps_exact_native(self):
        sketch = geometry.make_sketch([(torch.ones_like(self.old[0]),)], [1.], self.old, self.old)
        parameters = self.parameters()
        result = geometry.realize_feasible_candidate(parameters, self.old, self.old, sketch)
        self.assertFalse(result['search_ran'])
        self.assertEqual(result['bypass'], 'zero_native_displacement')
        self.assertTrue(result['candidate']['identical_prestate'])
        self.assertTrue(result['candidate']['identical_native'])
        self.assertTrue(torch.equal(parameters[0], self.old[0]))

    def test_parameter_partition_and_streaming_do_not_change_endpoint(self):
        old = [self.old[0].repeat(65)]
        native = [self.native[0].repeat(65)]
        delta = [self.delta[0].repeat(65)]
        endpoints, results = [], []
        for chunk_size in (1, 7, 262144):
            parameters = [native[0].clone()]
            results.append(geometry.realize_feasible_candidate(parameters, old, native,
                           fixed_direction(old, native, delta), chunk_size=chunk_size))
            endpoints.append(parameters[0])
        self.assertTrue(all(torch.equal(endpoints[0], p) for p in endpoints))
        self.assertTrue(all(r['alpha'] == results[0]['alpha'] for r in results))
        self.assertTrue(all(r['candidate']['norm'] == results[0]['candidate']['norm'] for r in results))

    def test_actual_two_response_sketch_realization_keeps_bf16_gradient_storage(self):
        generator = torch.Generator().manual_seed(20261109)
        for _ in range(8):
            old = [torch.randn(7, generator=generator).bfloat16(),
                   torch.randn(3, generator=generator).bfloat16()]
            native = [(o.float()+.02*torch.randn(o.shape, generator=generator)).bfloat16()
                      for o in old]
            gradients = [tuple(torch.randn(o.shape, generator=generator).bfloat16() for o in old)
                         for _ in range(2)]
            saved = [[g.clone() for g in response] for response in gradients]
            sketch = geometry.make_sketch(gradients, [.4, .7], old, native, chunk_size=2)
            parameters = [value.clone() for value in native]
            result = geometry.realize_feasible_candidate(parameters, old, native, sketch, chunk_size=2)
            self.assertLessEqual(result['candidate']['norm'], result['candidate']['native_norm'])
            for response, reference in zip(gradients, saved):
                for value, initial in zip(response, reference):
                    self.assertEqual(value.dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(value, initial))

    def test_nonfinite_coefficients_probes_and_endpoints_stop(self):
        for bad in (float('nan'), float('inf')):
            sketch = fixed_direction(self.old, self.native, self.delta)
            sketch.coefficients[0] = bad
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                geometry.realize_feasible_candidate(self.parameters(), self.old, self.native, sketch)
        with patch.object(geometry, 'candidate_norm', return_value=float('nan')) as probe:
            with self.assertRaisesRegex(ValueError, 'Nonfinite|contradicts'):
                geometry.realize_feasible_candidate(self.parameters(), self.old, self.native, self.sketch)
        self.assertEqual(probe.call_count, 1)
        bad_native = [self.native[0].clone()]; bad_native[0][0] = float('inf')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            geometry.realize_feasible_candidate(self.parameters(), self.old, bad_native, self.sketch)

    def test_final_write_disagreement_is_failure_without_another_search(self):
        original = geometry._write_scaled_candidate
        calls = []

        def altered(*args, **kwargs):
            value = original(*args, **kwargs)
            calls.append(args[4])
            if len(calls) == 2:
                value['norm'] = value['native_norm']+1.
            return value

        with patch.object(geometry, '_write_scaled_candidate', side_effect=altered), \
                patch.object(geometry, 'candidate_norm', wraps=geometry.candidate_norm) as probes:
            with self.assertRaisesRegex(ValueError, 'contradicts'):
                geometry.realize_feasible_candidate(self.parameters(), self.old, self.native, self.sketch)
        self.assertEqual(probes.call_count, 24)
        self.assertEqual(len(calls), 2)

    def test_interrupted_probe_retains_mutable_progress_and_original_error(self):
        progress = {}
        original = geometry.candidate_norm
        calls = 0

        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('deliberate probe failure')
            return original(*args, **kwargs)

        with patch.object(geometry, 'candidate_norm', side_effect=interrupted), \
                self.assertRaisesRegex(RuntimeError, 'deliberate probe failure'):
            geometry.realize_feasible_candidate(self.parameters(), self.old, self.native,
                                                self.sketch, progress=progress)
        self.assertEqual(progress['status'], 'failed')
        self.assertEqual(progress['phase'], 'probe')
        self.assertEqual(progress['probe_count'], 3)
        self.assertEqual(progress['completed_probes'], 2)
        self.assertEqual(progress['iterations'], 2)
        self.assertEqual(progress['write_attempts'], 1)
        self.assertEqual(progress['write_passes'], 1)
        self.assertIsNotNone(progress['raw_candidate'])
        self.assertGreaterEqual(progress['costs']['probe_s'], 0.)
        self.assertGreaterEqual(progress['costs']['total_s'], progress['costs']['probe_s'])


if __name__ == '__main__':
    unittest.main()
