"""Focused displacement/numerics tests. No model, GPU or experimental data."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

spec = importlib.util.spec_from_file_location('agreement_core_fixture', Path(__file__).with_name('core.py'))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def setup(values, *, dtype=torch.float32, lr=3e-4, weight_decay=.01, betas=(.9, .999)):
    parameters = {name: torch.nn.Parameter(torch.tensor(value, dtype=dtype)) for name, value in values.items()}
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=lr, betas=betas, eps=1e-8,
                                  weight_decay=weight_decay, foreach=None, fused=None)
    return parameters, optimizer


def gradients(parameters, values):
    return {name: torch.tensor(values[name], dtype=torch.float32).reshape(parameter.shape)
            for name, parameter in parameters.items()}


def ordinary(parameters, optimizer, a, b):
    optimizer.zero_grad(set_to_none=True)
    for name, p in parameters.items():
        p.grad = (a[name] + b[name]).to(p.dtype)
    norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1., error_if_nonfinite=True)
    optimizer.step()
    return norm


def assert_states_equal(test, left, right):
    l, r = left.state_dict(), right.state_dict()
    test.assertEqual(l['param_groups'], r['param_groups'])
    test.assertEqual(set(l['state']), set(r['state']))
    for index in l['state']:
        test.assertEqual(set(l['state'][index]), set(r['state'][index]))
        for key, value in l['state'][index].items():
            if isinstance(value, torch.Tensor):
                test.assertTrue(torch.equal(value, r['state'][index][key]), (index, key))
            else:
                test.assertEqual(value, r['state'][index][key])


class AgreementSignTests(unittest.TestCase):
    def test_tiny_positive_product_underflow_does_not_admit_coordinate(self):
        a = torch.tensor([1e-30, -1e-30, 1e-30, -1e-30])
        b = torch.tensor([1e-30, -1e-30, -1e-30, 1e-30])
        d = torch.full((4,), 1e-30)
        self.assertTrue(bool((a*d == 0).all()))
        self.assertEqual(core.agreement_mask(a, b, d).tolist(), [False, True, False, False])

    def test_zeros_and_opposite_signs_obey_non_strict_inequality(self):
        a = torch.tensor([0., -0., 1., -1., 0., 1., -1., 1.])
        b = torch.tensor([1., -1., 0., 0., 0., 1., -1., -1.])
        d = torch.tensor([-1., 1., 0., -0., 1., -1., 1., 1.])
        self.assertEqual(core.agreement_mask(a, b, d).tolist(), [True]*7 + [False])

    def test_nonfinite_sign_input_fails(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                core.agreement_mask(torch.tensor([value]), torch.tensor([0.]), torch.tensor([1.]))

    def test_precision_not_silently_promoted(self):
        with self.assertRaisesRegex(ValueError, 'FP32'):
            core.agreement_mask(torch.ones(1, dtype=torch.bfloat16), torch.ones(1), torch.ones(1))


class AdamDisplacementTests(unittest.TestCase):
    def test_unfiltered_matches_ordinary_with_existing_momentum_for_both_dtypes(self):
        for dtype in (torch.float32, torch.bfloat16):
            p, opt = setup({'a': [1., -2., .5], 'b': [3., -4.]}, dtype=dtype)
            a, b = gradients(p, {'a': [2., -1., .5], 'b': [5., -3.]}), gradients(p, {'a': [1., 2., -.1], 'b': [-2., 1.]})
            ordinary(p, opt, a, b)
            q, control = setup({n: x.detach().float().tolist() for n, x in p.items()}, dtype=dtype)
            control.load_state_dict(copy.deepcopy(opt.state_dict()))
            a2, b2 = {n: -2*x for n, x in a.items()}, {n: .25*x for n, x in b.items()}
            ordinary(q, control, a2, b2)
            record = core.apply_adamw_step(p, opt, a2, b2, mode='unfiltered')
            for n in p:
                self.assertTrue(torch.equal(p[n], q[n]), (dtype, n))
            assert_states_equal(self, opt, control)
            self.assertEqual(record['optimizer_steps_before'], 1)
            self.assertEqual(record['optimizer_steps_after'], 2)
            self.assertEqual(record['optimizer_step_calls'], 1)

    def test_momentum_can_reverse_shared_gradient_direction(self):
        p, opt = setup({'x': [0.]}, lr=.1, weight_decay=0)
        ordinary(p, opt, gradients(p, {'x': [-1.]}), gradients(p, {'x': [-1.]}))
        previous = p['x'].detach().clone()
        a, b = gradients(p, {'x': [.01]}), gradients(p, {'x': [.01]})
        record = core.apply_adamw_step(p, opt, a, b, mode='G')
        self.assertTrue(torch.equal(p['x'], previous))
        self.assertGreater(record['proposed_step_norm'], 0)
        self.assertEqual(record['kept_nonzero_coordinates'], 0)
        self.assertEqual(int(opt.state[p['x']]['step']), 2)
        self.assertLess(float(opt.state[p['x']]['exp_avg']), 0)

    def test_weight_decay_participates_in_the_actual_direction(self):
        p, opt = setup({'x': [1000.]}, lr=.1, weight_decay=.1)
        record = core.apply_adamw_step(p, opt, gradients(p, {'x': [-1e-12]}),
                                     gradients(p, {'x': [-1e-12]}), mode='G')
        self.assertEqual(float(p['x']), 1000.)
        self.assertGreater(record['proposed_step_norm'], 9.)
        self.assertEqual(record['applied_step_norm'], 0.)
        self.assertLess(float(opt.state[p['x']]['exp_avg']), 0.)
        self.assertEqual(int(opt.state[p['x']]['step']), 1)

    def test_global_n_scale_is_shared_across_tensors(self):
        p, opt = setup({'x': [0.], 'y': [0.]}, lr=.1, weight_decay=0, betas=(0., 0.))
        a, b = gradients(p, {'x': [1.], 'y': [1.]}), gradients(p, {'x': [1.], 'y': [-2.]})
        before_a, before_b = copy.deepcopy(a), copy.deepcopy(b)
        record = core.apply_adamw_step(p, opt, a, b, mode='N')
        self.assertAlmostEqual(record['uniform_scale_alpha'], 2**-.5, places=6)
        self.assertAlmostEqual(float(p['x']), -.1*(2**-.5), places=6)
        self.assertAlmostEqual(float(p['y']), .1*(2**-.5), places=6)
        self.assertAlmostEqual(record['applied_step_norm'], record['hypothetical_filtered_step_norm'], places=7)
        for n in p:
            self.assertTrue(torch.equal(a[n], before_a[n]))
            self.assertTrue(torch.equal(b[n], before_b[n]))

    def test_g_and_n_have_different_changes_but_identical_ordinary_moments(self):
        p, opt = setup({'x': [0., 0.]}, lr=.1, weight_decay=0, betas=(0., 0.))
        q, other = setup({'x': [0., 0.]}, lr=.1, weight_decay=0, betas=(0., 0.))
        a, b = gradients(p, {'x': [1., 1.]}), gradients(p, {'x': [1., -2.]})
        rg = core.apply_adamw_step(p, opt, a, b, mode='G')
        rn = core.apply_adamw_step(q, other, a, b, mode='N')
        self.assertEqual(float(p['x'][1]), 0.)
        self.assertGreater(float(q['x'][1]), 0.)
        assert_states_equal(self, opt, other)
        self.assertLessEqual(rg['group_directional_derivatives']['applied']['A'], 0.)
        self.assertLessEqual(rg['group_directional_derivatives']['applied']['B'], 0.)
        self.assertEqual(rn['own_hypothetical_uniform_scale'], rg['own_hypothetical_uniform_scale'])

    def test_everything_rejected_still_advances_moments_once(self):
        p, opt = setup({'x': [0., 0.]}, lr=.1, weight_decay=0)
        a, b = gradients(p, {'x': [1., 1.]}), gradients(p, {'x': [-2., -2.]})
        for mode in ('G', 'N'):
            p['x'].data.zero_()
            before = int(opt.state[p['x']]['step']) if p['x'] in opt.state else 0
            record = core.apply_adamw_step(p, opt, a, b, mode=mode)
            self.assertTrue(record['applied_zero'])
            self.assertTrue(bool((p['x'] == 0).all()))
            self.assertEqual(int(opt.state[p['x']]['step']), before+1)
            self.assertGreater(float(opt.state[p['x']]['exp_avg_sq'].sum()), 0.)

    def test_zero_proposal_has_zero_alpha_and_advances_optimizer(self):
        p, opt = setup({'x': [0.]}, weight_decay=0)
        a, b = gradients(p, {'x': [0.]}), gradients(p, {'x': [0.]})
        record = core.apply_adamw_step(p, opt, a, b, mode='N')
        self.assertEqual(record['uniform_scale_alpha'], 0.)
        self.assertEqual(record['proposed_step_norm'], 0.)
        self.assertEqual(record['nonzero_keep_fraction'], 0.)
        self.assertEqual(record['mask_keep_fraction'], 1.)
        self.assertEqual(int(opt.state[p['x']]['step']), 1)

    def test_raw_gradients_are_not_clipped_for_sign_tests(self):
        p, opt = setup({'x': [0., 0.]}, weight_decay=0)
        a, b = gradients(p, {'x': [100., 100.]}), gradients(p, {'x': [100., -200.]})
        original_clip = torch.nn.utils.clip_grad_norm_
        with patch.object(torch.nn.utils, 'clip_grad_norm_', wraps=original_clip) as clip:
            record = core.apply_adamw_step(p, opt, a, b, mode='G')
        self.assertEqual(clip.call_count, 1)
        self.assertEqual(a['x'].tolist(), [100., 100.])
        self.assertEqual(b['x'].tolist(), [100., -200.])
        self.assertGreater(record['gradient_norm_before_clip'], 200.)
        self.assertLessEqual(record['gradient_norm_after_clip_fp64'], 1.000001)
        self.assertEqual(record['kept_nonzero_coordinates'], 1)

    def test_bf16_n_reports_applied_rounding_separately(self):
        p, opt = setup({'x': [1., 1.]}, dtype=torch.bfloat16, lr=.01, weight_decay=0, betas=(0., 0.))
        a, b = gradients(p, {'x': [1., 1.]}), gradients(p, {'x': [1., -2.]})
        record = core.apply_adamw_step(p, opt, a, b, mode='N')
        self.assertNotEqual(record['native_rounding_norm_difference'], 0.)
        self.assertEqual(record['actual_parameter_dtype'], 'torch.bfloat16')
        self.assertEqual(opt.state[p['x']]['exp_avg'].dtype, torch.bfloat16)
        self.assertAlmostEqual(record['applied_step_norm'], float(torch.linalg.vector_norm((p['x'].float()-1).double())))

    def test_bf16_g_preserves_exact_ordinary_native_proposal_on_kept_coordinates(self):
        p, opt = setup({'x': [1., 1.]}, dtype=torch.bfloat16, lr=.01, weight_decay=0)
        q, control = setup({'x': [1., 1.]}, dtype=torch.bfloat16, lr=.01, weight_decay=0)
        a, b = gradients(p, {'x': [1., 1.]}), gradients(p, {'x': [1., -2.]})
        ordinary(q, control, a, b)
        record = core.apply_adamw_step(p, opt, a, b, mode='G')
        self.assertEqual(p['x'][0], q['x'][0])
        self.assertEqual(float(p['x'][1]), 1.)
        self.assertEqual(record['native_rounding_norm_difference'], 0.)
        assert_states_equal(self, opt, control)

    def test_nonfinite_gradient_fails_before_step(self):
        p, opt = setup({'x': [1.]})
        a, b = gradients(p, {'x': [float('nan')]}), gradients(p, {'x': [1.]})
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            core.apply_adamw_step(p, opt, a, b)
        self.assertFalse(opt.state)
        self.assertEqual(float(p['x']), 1.)

    def test_nonfinite_existing_moment_fails_before_step(self):
        p, opt = setup({'x': [1.]})
        a, b = gradients(p, {'x': [1.]}), gradients(p, {'x': [1.]})
        ordinary(p, opt, a, b)
        opt.state[p['x']]['exp_avg'].fill_(float('inf'))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            core.apply_adamw_step(p, opt, a, b)
        self.assertEqual(int(opt.state[p['x']]['step']), 1)

    def test_incomplete_inventory_and_changed_clip_are_rejected(self):
        p, opt = setup({'x': [1.], 'y': [2.]})
        a = gradients(p, {'x': [1.], 'y': [1.]})
        with self.assertRaisesRegex(ValueError, 'inventory'):
            core.apply_adamw_step(p, opt, a, {'x': a['x']})
        with self.assertRaisesRegex(ValueError, 'clip'):
            core.apply_adamw_step(p, opt, a, a, clip_norm=2.)
        self.assertFalse(opt.state)

    def test_two_steps_recompute_from_filtered_state_and_keep_history(self):
        p, opt = setup({'x': [0., 0.]}, weight_decay=0)
        a, b = gradients(p, {'x': [1., 1.]}), gradients(p, {'x': [1., -2.]})
        records = [core.apply_adamw_step(p, opt, a, b, mode='G') for _ in range(2)]
        self.assertEqual([r['optimizer_steps_after'] for r in records], [1, 2])
        self.assertEqual(float(p['x'][1]), 0.)
        self.assertLess(float(opt.state[p['x']]['exp_avg'][1]), 0.)
        self.assertEqual(sum(r['optimizer_step_calls'] for r in records), 2)


if __name__ == '__main__':
    unittest.main()
