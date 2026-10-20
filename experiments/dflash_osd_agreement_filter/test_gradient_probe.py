"""CPU fixtures for one-graph backward order and descriptive comparisons."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

_spec = importlib.util.spec_from_file_location('_gradient_probe_tests', Path(__file__).with_name('gradient_probe.py'))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def fixture(dtype=torch.float32):
    parameters = {'x': torch.nn.Parameter(torch.tensor([.5, -1.5], dtype=dtype)),
                  'y': torch.nn.Parameter(torch.tensor([2., .75, -.25], dtype=dtype))}
    calls = []
    def forward():
        calls.append(1)
        x, y = parameters['x'], parameters['y']
        shared = x.square().sum() + y.square().sum()
        a = shared * .25 + (x.sum() * y.sum())
        b = shared * .75 - x.mean() * y.mean()
        return a+b, a, b, {'fixed_anchor_ids': [4, 7], 'forward_marker': len(calls)}
    return parameters, forward, calls


class BackwardProbeTests(unittest.TestCase):
    def test_group_and_full_backward_orders_have_one_forward_each(self):
        for dtype in (torch.float32, torch.bfloat16):
            p, factory, calls = fixture(dtype)
            abr = probe.run_backward_probe(factory, p, order=('A', 'B', 'F'))
            fab = probe.run_backward_probe(factory, p, order=('F', 'A', 'B'))
            self.assertEqual(len(calls), 2)
            self.assertEqual(abr['forward_calls'], 1)
            self.assertEqual(fab['forward_calls'], 1)
            self.assertEqual(abr['backward_calls'], 3)
            for label in ('A', 'B', 'F'):
                for name in p:
                    self.assertTrue(torch.equal(abr['gradients'][label][name], fab['gradients'][label][name]))
                    self.assertEqual(abr['gradients'][label][name].device.type, 'cpu')
                    self.assertEqual(abr['gradients'][label][name].dtype, torch.float32)
            self.assertEqual(abr['losses']['F'], fab['losses']['F'])
            self.assertEqual(abr['metadata']['fixed_anchor_ids'], [4, 7])

    def test_repeated_full_backward_reuses_the_same_loss_graph(self):
        p, factory, calls = fixture()
        result = probe.run_backward_probe(factory, p, order=('F1', 'F2'))
        self.assertEqual(calls, [1])
        self.assertEqual(result['selected_loss_order'], ['F', 'F'])
        self.assertEqual(result['backward_order'], ['F1', 'F2'])
        for name in p:
            self.assertTrue(torch.equal(result['gradients']['F1'][name], result['gradients']['F2'][name]))
            self.assertNotEqual(result['gradients']['F1'][name].data_ptr(), result['gradients']['F2'][name].data_ptr())

    def test_no_sampling_or_rng_consumption_between_backwards(self):
        p, _, _ = fixture()
        generator = torch.Generator().manual_seed(17)
        expected = torch.Generator().manual_seed(17)
        torch.rand(3, generator=expected)
        calls = []
        def factory():
            calls.append(1)
            weights = torch.rand(3, generator=generator)
            a = p['x'].square().sum() * weights[0] + p['y'].sum() * weights[1]
            b = (p['x'].sum() + p['y'].square().sum()) * weights[2]
            return a+b, a, b, {'draws': weights.tolist()}
        result = probe.run_backward_probe(factory, p)
        self.assertEqual(calls, [1])
        self.assertTrue(torch.equal(generator.get_state(), expected.get_state()))
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertEqual(result['file_writes'], 0)

    def test_existing_parameter_grad_buffers_and_weights_remain_unchanged(self):
        p, factory, _ = fixture()
        before = {name: value.detach().clone() for name, value in p.items()}
        sentinel = {name: torch.full_like(value, 17.) for name, value in p.items()}
        for name, value in p.items():
            value.grad = sentinel[name]
        probe.run_backward_probe(factory, p)
        for name, value in p.items():
            self.assertTrue(torch.equal(value, before[name]))
            self.assertIs(value.grad, sentinel[name])
            self.assertTrue(bool((value.grad == 17).all()))

    def test_invalid_order_fails_without_forward(self):
        p, factory, calls = fixture()
        for order in ((), ('F', 'F'), ('A', 'B', 'F1', 'F2'), ('Z',), ('F0',)):
            with self.assertRaises(ValueError):
                probe.run_backward_probe(factory, p, order=order)
        self.assertFalse(calls)

    def test_unused_parameter_is_not_silently_replaced_with_zero(self):
        p, _, _ = fixture()
        def factory():
            a, b = p['x'].sum(), p['x'].square().sum()
            return a+b, a, b, {}
        with self.assertRaises(RuntimeError):
            probe.run_backward_probe(factory, p)

    def test_mutating_factory_is_detected(self):
        p, original, _ = fixture()
        def mutated():
            with torch.no_grad():
                p['x'].add_(1)
            return original()
        with self.assertRaisesRegex(ValueError, 'mutated'):
            probe.run_backward_probe(mutated, p)


class ComparisonTests(unittest.TestCase):
    def test_global_fp64_norms_equal_flat_calculation_not_mean_of_tensor_ratios(self):
        p, _, _ = fixture()
        left = {'x': torch.tensor([1., 2.]), 'y': torch.tensor([3., 4., 5.])}
        right = {'x': torch.tensor([-1., 2.]), 'y': torch.tensor([3., 0., 5.])}
        with patch.object(probe, 'CHUNK_ELEMENTS', 2):
            result = probe.compare_gradients(left, right, p)
        l, r = torch.cat(list(left.values())).double(), torch.cat(list(right.values())).double()
        row = result['global']
        self.assertAlmostEqual(row['left_norm'], float(l.norm()))
        self.assertAlmostEqual(row['right_norm'], float(r.norm()))
        self.assertAlmostEqual(row['difference_norm'], float((l-r).norm()))
        self.assertAlmostEqual(row['relative_l2_error_vs_right'], float((l-r).norm()/r.norm()))
        self.assertAlmostEqual(row['cosine_similarity'], float(l.dot(r)/(l.norm()*r.norm())))
        self.assertEqual(row['different_coordinates'], 2)
        self.assertEqual(row['elements'], 5)
        self.assertEqual(row['max_abs_difference'], 4.)
        self.assertEqual(result['per_parameter']['x']['different_coordinates'], 1)
        self.assertFalse(result['acceptance_tolerance_applied'])
        self.assertEqual(result['chunk_elements'], 2)

    def test_exact_repeat_returns_zero_error(self):
        p, factory, _ = fixture()
        result = probe.run_backward_probe(factory, p, order=('F1', 'F2'))
        report = probe.compare_gradients(result['gradients']['F1'], result['gradients']['F2'], p)
        self.assertTrue(report['global']['exactly_equal'])
        self.assertEqual(report['global']['difference_norm'], 0.)
        self.assertEqual(report['global']['different_coordinates'], 0)

    def test_sum_is_measured_without_an_acceptance_gate(self):
        p, factory, _ = fixture()
        result = probe.run_backward_probe(factory, p)
        g = result['gradients']
        summed = {name: g['A'][name] + g['B'][name] for name in p}
        report = probe.compare_gradients(summed, g['F'], p)
        self.assertFalse(report['acceptance_tolerance_applied'])
        self.assertNotIn('passed', report)
        self.assertNotIn('tolerance', report)
        self.assertLess(report['global']['difference_norm'], 1e-5)

    def test_nonfinite_values_remain_explicit_not_zero_error(self):
        p, _, _ = fixture()
        left = {'x': torch.tensor([float('nan'), 2.]), 'y': torch.tensor([3., float('inf'), 5.])}
        right = {'x': torch.tensor([1., 2.]), 'y': torch.tensor([3., 4., 0.])}
        report = probe.compare_gradients(left, right, p)['global']
        self.assertEqual(report['nonfinite_left'], 2)
        self.assertEqual(report['nonfinite_right'], 0)
        self.assertEqual(report['finite_pairs'], 3)
        self.assertFalse(report['exactly_equal'])
        self.assertFalse(report['all_values_finite'])
        self.assertIsNone(report['difference_norm'])
        self.assertIsNone(report['max_abs_difference'])
        self.assertEqual(report['finite_pairs_difference_norm'], 5.)
        self.assertEqual(report['finite_pairs_max_abs_difference'], 5.)

    def test_zero_denominator_is_explicit(self):
        p, _, _ = fixture()
        zero = {name: torch.zeros_like(value) for name, value in p.items()}
        same = probe.compare_gradients(zero, zero, p)['global']
        self.assertEqual(same['relative_l2_error_vs_right'], 0.)
        nonzero = {name: torch.ones_like(value) for name, value in p.items()}
        report = probe.compare_gradients(nonzero, zero, p)['global']
        self.assertIsNone(report['relative_l2_error_vs_right'])
        self.assertFalse(report['relative_l2_error_defined'])
        self.assertIsNone(report['cosine_similarity'])

    def test_inventory_or_shape_mismatch_rejected(self):
        p, _, _ = fixture()
        grads = {name: torch.zeros_like(value) for name, value in p.items()}
        with self.assertRaisesRegex(ValueError, 'inventories'):
            probe.compare_gradients(grads, {'x': grads['x']}, p)
        changed = dict(grads)
        changed['x'] = torch.ones(3)
        with self.assertRaisesRegex(ValueError, 'shape'):
            probe.compare_gradients(grads, changed, p)


if __name__ == '__main__':
    unittest.main()
