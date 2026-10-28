"""CPU checks for G's scalar objective, displacement guard and native K wiring."""
from copy import deepcopy
import math
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from online_feedback import training as T
from online_feedback.prefix_guard import (
    install_guard, normalized_preservation, prefix_changes, preservation_terms,
    preservation_weights, project_native_displacement,
)
from online_feedback.rng import capture_rng
from test_training_contract import trees_equal
import test_osd_recipe as legacy


class PreservationMathTests(unittest.TestCase):
    def test_strict_prefix_triangular_weights_seed_and_later_matches(self):
        labels = torch.tensor([[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]])
        proposals = labels.clone()
        proposals[0, 0, 3] = 0
        proposals[0, 1, 1] = 0
        valid = torch.ones_like(labels, dtype=torch.float32)
        actual = preservation_weights(proposals, labels, valid)
        self.assertEqual(actual.tolist(), [[[0, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 4, 3, 2, 1]]])

    def test_eos_is_included_but_padding_gap_or_eos_seed_ends_prefix(self):
        labels = torch.tensor([[[0, 1, 9, 2, 3], [9, 1, 2, 3, 4], [0, 1, 2, 3, 4]]])
        valid = torch.ones_like(labels, dtype=torch.float32)
        valid[0, 2, 2] = 0
        actual = preservation_weights(labels, labels, valid, (9,))
        self.assertEqual(actual.tolist(), [[[0, 2, 1, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]]])

    def test_native_argmax_tie_rule_decides_prefix_membership(self):
        logits = torch.tensor([[[[1., 0.], [1., 1.], [1., 1.]]]], dtype=torch.bfloat16)
        labels = torch.tensor([[[0, 0, 1]]])
        weights = preservation_weights(logits.argmax(-1), labels, torch.ones_like(labels).float())
        self.assertEqual(weights.tolist(), [[[0, 1, 0]]])
        numerator, denominator = preservation_terms(logits, labels, weights)
        self.assertAlmostEqual(float(normalized_preservation(numerator, denominator)), 1, places=6)

    def test_response_wide_normalization_is_not_mean_of_chunk_means(self):
        logits = torch.tensor([[[[0., 0.], [4., 0.], [2., 0.]],
                                [[0., 0.], [0., 0.], [0., 0.]]]], requires_grad=True)
        labels = torch.zeros((1, 2, 3), dtype=torch.long)
        weights = torch.tensor([[[0., 2., 1.], [0., 1., 0.]]])
        pieces = [preservation_terms(logits[:, i:i+1], labels[:, i:i+1], weights[:, i:i+1]) for i in range(2)]
        actual = normalized_preservation(sum(x[0] for x in pieces), sum(x[1] for x in pieces))
        expected = (2*F.softplus(torch.tensor(-4.)) + F.softplus(torch.tensor(-2.)) + math.log(2))/(4*math.log(2))
        torch.testing.assert_close(actual, expected)
        wrong = sum(normalized_preservation(*piece) for piece in pieces)/2
        self.assertNotAlmostEqual(float(actual), float(wrong), places=4)
        actual.backward()
        self.assertEqual(float(logits.grad[0, 1, 2].abs().sum()), 0)

    def test_empty_set_has_zero_loss_and_gradient(self):
        logits = torch.randn(1, 2, 4, 7, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.zeros((1, 2, 4), dtype=torch.long)
        value = normalized_preservation(*preservation_terms(logits, labels, torch.zeros_like(labels).float()))
        value.backward()
        self.assertEqual(float(value), 0)
        self.assertEqual(float(logits.grad.abs().sum()), 0)

    def test_zero_response_still_receives_its_half_in_response_average(self):
        a = torch.tensor(2., requires_grad=True)
        b = torch.tensor(0., requires_grad=True)
        ha = normalized_preservation(a, torch.tensor(2.))
        hb = normalized_preservation(b * 0, torch.tensor(0.))
        ((ha + hb)/2).backward()
        self.assertAlmostEqual(float(a.grad), 1/(4*math.log(2)), places=6)
        self.assertEqual(float(b.grad), 0)

    def test_margin_selects_noncanonical_rival_but_gathered_logits_are_differentiable(self):
        logits = torch.tensor([[[[1., 2., 3.]]]], requires_grad=True)
        labels = torch.tensor([[[2]]])
        value = normalized_preservation(*preservation_terms(logits, labels, torch.ones_like(labels).float()))
        value.backward()
        self.assertEqual(float(logits.grad[0, 0, 0, 0]), 0)
        self.assertGreater(float(logits.grad[0, 0, 0, 1]), 0)
        self.assertLess(float(logits.grad[0, 0, 0, 2]), 0)

    def test_prefix_repairs_do_not_count_later_fix_behind_new_earlier_error(self):
        labels = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]])
        before = labels.clone(); before[..., 3] = 4
        after = labels.clone(); after[0, 0, 1] = 4
        changes = prefix_changes(before, after, labels, torch.ones_like(labels).float())
        self.assertEqual(changes['first_errors_repaired_with_prefix_intact'], 1)
        self.assertEqual(changes['new_earlier_errors'], 1)
        self.assertEqual(changes['old_prefix_tokens_lost'], 2)
        self.assertEqual(changes['net_matching_prefix_change'], -1)


class ProjectionTests(unittest.TestCase):
    def test_global_projection_across_parameter_boundaries(self):
        before = [torch.tensor([0.], dtype=torch.bfloat16) for _ in range(2)]
        params = [torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16)) for _ in range(2)]
        h = [torch.tensor([1.]), torch.tensor([2.])]
        row = project_native_displacement(params, before, h)
        self.assertTrue(row['active'])
        self.assertAlmostEqual(row['projection_coefficient'], .3)
        torch.testing.assert_close(params[0], torch.tensor([.2], dtype=torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(params[1], torch.tensor([-.1], dtype=torch.bfloat16), rtol=0, atol=0)
        self.assertEqual(row['analytic_projected_directional_derivative'], 0)
        fp32_delta = torch.tensor([.5, .5]) - .3 * torch.tensor([1., 2.])
        self.assertEqual(row['ideal_directional_derivative'], float((fp32_delta.double() * torch.tensor([1., 2.])).sum()))
        expected_dot = float(params[0].float() + 2*params[1].float())
        self.assertEqual(row['realized_directional_derivative'], expected_dot)

    def test_inactive_disabled_zero_and_tangent_branches_leave_native_storage_untouched(self):
        for h, enabled in (([0., 0.], True), ([-1., 0.], True), ([1., -1.], True), ([1., 1.], False)):
            parameter = torch.nn.Parameter(torch.tensor([.05, .05], dtype=torch.bfloat16))
            old = torch.tensor([0., 0.], dtype=torch.bfloat16)
            exact = parameter.detach().clone(); version = parameter._version
            row = project_native_displacement([parameter], [old], [torch.tensor(h)], enabled=enabled)
            self.assertFalse(row['active'])
            self.assertTrue(torch.equal(parameter, exact))
            self.assertEqual(parameter._version, version, 'Inactive path wrote native weights')

    def test_realized_bf16_norm_and_direction_are_reported_not_idealized(self):
        old = torch.tensor([.05, .05], dtype=torch.bfloat16)
        parameter = torch.nn.Parameter(old + torch.tensor([.0003, .0003], dtype=torch.bfloat16))
        h = torch.tensor([1., 2.])
        row = project_native_displacement([parameter], [old], [h])
        delta = parameter.float() - old.float()
        self.assertAlmostEqual(row['realized_update_norm'], float(delta.double().norm()), places=12)
        self.assertAlmostEqual(row['realized_directional_derivative'], float((delta.double()*h.double()).sum()))

    def test_nonfinite_gradient_rejects_before_correction(self):
        parameter = torch.nn.Parameter(torch.tensor([1.], dtype=torch.bfloat16))
        before = parameter.clone()
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            project_native_displacement([parameter], [torch.zeros_like(parameter)], [torch.tensor([float('nan')])])
        self.assertTrue(torch.equal(parameter, before))


class GuardTrainerTests(unittest.TestCase):
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

    def test_disabled_guard_exact_native_K_state_and_rng(self):
        ordinary = self.trainer('K', objective='osd_first_error')
        expected = ordinary.train_event('e1', self.features[:2])
        expected_state = self.saved(ordinary)
        guarded = self.trainer('Gdisabled', objective='osd_first_error')
        with install_guard(guarded, enabled=False, probe_steps=()) as guard:
            actual = guarded.train_event('e1', self.features[:2], presentation_schedule=expected)
        actual_state = self.saved(guarded)
        for key in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertTrue(trees_equal(expected_state[key], actual_state[key]), key)
        self.assertEqual(guard.work['h_backward_calls'], 0)
        self.assertEqual(guard.work['k_backward_calls'], 4)

    def test_enabled_guard_preserves_K_gradients_and_Adam_before_correction(self):
        # A deterministic synthetic response of token 0 and a frozen head bias
        # guarantee nonempty prefixes, so this checks nonzero H gradients too.
        for path in self.features[:2]:
            raw = torch.load(path, weights_only=True)
            raw['input_ids'].zero_()
            raw['label_binding']['target_argmax_ids'] = raw['input_ids'][0, 4:].tolist()
            torch.save(raw, path)
        def expose_correct_prefixes(trainer):
            head = trainer.bundle.model.lm_head
            original = head.forward
            def biased(hidden):
                logits = original(hidden)
                bias = torch.zeros(logits.shape[-1], dtype=logits.dtype, device=logits.device)
                bias[0] = 8
                return logits + bias
            head.forward = biased
        expected = {}
        ordinary = self.trainer('K', objective='osd_first_error')
        expose_correct_prefixes(ordinary)
        native_step = ordinary.backend.optimizer.step
        def save_native():
            expected['gradients'] = [p.grad.detach().clone() for p in ordinary.backend.optimizer.model_params]
            result = native_step()
            expected['weights'] = deepcopy(ordinary.bundle.draft_model.state_dict())
            expected['optimizer'] = deepcopy(ordinary.backend.optimizer.state_dict())
            return result
        ordinary.backend.optimizer.step = save_native
        receipt = ordinary.train_event('e1', self.features[:2], passes=1)
        guarded = self.trainer('G', objective='osd_first_error')
        expose_correct_prefixes(guarded)
        observations = []
        def check(phase, trainer, guard):
            if phase == 'before_native_step':
                for actual, saved in zip(guard.parameters, expected['gradients']):
                    torch.testing.assert_close(actual.grad, saved, rtol=0, atol=0)
                self.assertTrue(all(x.dtype == torch.float32 for x in guard.h))
            if phase == 'after_native_step':
                self.assertTrue(trees_equal(expected['weights'], trainer.bundle.draft_model.state_dict()))
                self.assertTrue(trees_equal(expected['optimizer'], trainer.backend.optimizer.state_dict()))
            observations.append(phase)
        with install_guard(guarded, probe_steps=(), observer=check) as guard:
            result = guarded.train_event('e1', self.features[:2], passes=1, presentation_schedule=receipt)
        self.assertIn('after_native_step', observations)
        self.assertEqual(guard.work['h_backward_calls'], 2)
        for kind in ('h', 'k'):
            self.assertEqual(guard.work[f'{kind}_backward_attempted'], 2)
            self.assertEqual(guard.work[f'{kind}_backward_completed'], 2)
            self.assertEqual(guard.work[f'{kind}_backward_failed'], 0)
        self.assertEqual(guard.work['current_response_forwards'], 2)
        self.assertEqual(len(guard.step_records), 1)
        self.assertGreater(sum(row['protected_positions'] for row in guard.response_records), 0)
        self.assertGreater(guard.step_records[0]['h_norm'], 0)
        self.assertEqual(result['response_presentations'], 2)
        self.assertTrue(trees_equal(expected['optimizer'], guarded.backend.optimizer.state_dict()))

    def test_probe_evaluations_restore_exact_weights_optimizer_rng_and_anchor_trajectory(self):
        plain = self.trainer('Gplain', objective='osd_first_error')
        with install_guard(plain, probe_steps=()) as plain_guard:
            expected = plain.train_event('e1', self.features[:2])
        expected_state = self.saved(plain)
        probed = self.trainer('Gprobed', objective='osd_first_error')
        with install_guard(probed, probe_steps=(1, 2)) as guard:
            result = probed.train_event('e1', self.features[:2], presentation_schedule=expected)
        actual_state = self.saved(probed)
        for key in ('draft_state_dict', 'backend', 'rng', 'global_step', 'presentations'):
            self.assertTrue(trees_equal(expected_state[key], actual_state[key]), key)
        self.assertEqual(guard.work['probe_response_forwards'], 12)
        self.assertEqual(guard.work['h_backward_calls'], 4)
        self.assertEqual(len(guard.probe_records), 6)
        self.assertEqual(len(guard.response_records), 4)
        self.assertEqual(guard.work['current_response_forwards'], 4)
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertTrue(probed.bundle.model.training)
        for row in guard.probe_records:
            self.assertEqual(len(row['responses']), 2)
            self.assertTrue(all(math.isfinite(response['h_after']) for response in row['responses']))

    def test_nonfinite_H_gradient_stops_before_K_backward_or_optimizer(self):
        trainer = self.trainer('Gnonfinite', objective='osd_first_error')
        initial = deepcopy(trainer.bundle.draft_model.state_dict())
        original_grad = torch.autograd.grad
        def injected(*args, **kwargs):
            gradients = list(original_grad(*args, **kwargs))
            gradients[0] = gradients[0].clone()
            gradients[0].view(-1)[0] = float('nan')
            return tuple(gradients)
        directory = self.root / 'guard_failure'
        with install_guard(trainer, probe_steps=(), output_dir=directory) as guard:
            with mock.patch('torch.autograd.grad', side_effect=injected):
                with self.assertRaisesRegex(ValueError, 'Nonfinite H gradient before K'):
                    trainer.train_event('e1', self.features[:2], passes=1)
            self.assertEqual(guard.work['k_backward_calls'], 0)
            self.assertEqual(guard.work['h_backward_attempted'], 1)
            self.assertEqual(guard.work['h_backward_completed'], 1)
            self.assertEqual(guard.work['h_backward_failed'], 0)
            self.assertEqual(guard.work['h_backward_calls'], 1)
            self.assertEqual(len(trainer.backend.optimizer.optimizer.state), 0)
            self.assertTrue(trees_equal(initial, trainer.bundle.draft_model.state_dict()))
        self.assertTrue((directory / 'NONFINITE_H_GRADIENT.json').exists())

    def test_failed_H_backward_counts_attempt_and_preserves_exception_and_time(self):
        trainer = self.trainer('Hfailure', objective='osd_first_error')
        failure = RuntimeError('compiled backward failure fixture')
        with install_guard(trainer, probe_steps=()) as guard:
            with mock.patch('torch.autograd.grad', side_effect=failure):
                with self.assertRaises(RuntimeError) as caught:
                    trainer.train_event('e1', self.features[:2], passes=1)
            self.assertIs(caught.exception, failure)
            self.assertEqual(guard.work['h_backward_attempted'], 1)
            self.assertEqual(guard.work['h_backward_completed'], 0)
            self.assertEqual(guard.work['h_backward_failed'], 1)
            self.assertEqual(guard.work['h_backward_calls'], 0)
            self.assertEqual(guard.work['k_backward_attempted'], 0)
            self.assertEqual(guard.work['k_backward_completed'], 0)
            self.assertEqual(guard.work['k_backward_failed'], 0)
            self.assertGreater(guard.costs['h_backward_s'], 0)
            self.assertEqual(len(trainer.backend.optimizer.optimizer.state), 0)

    def test_failed_K_backward_counts_attempt_after_successful_H(self):
        trainer = self.trainer('Kfailure', objective='osd_first_error')
        failure = RuntimeError('native K backward failure fixture')
        trainer.backend.backward = mock.Mock(side_effect=failure)
        with install_guard(trainer, probe_steps=()) as guard:
            with self.assertRaises(RuntimeError) as caught:
                trainer.train_event('e1', self.features[:2], passes=1)
            self.assertIs(caught.exception, failure)
            self.assertEqual(guard.work['h_backward_attempted'], 1)
            self.assertEqual(guard.work['h_backward_completed'], 1)
            self.assertEqual(guard.work['h_backward_failed'], 0)
            self.assertEqual(guard.work['h_backward_calls'], 1)
            self.assertEqual(guard.work['k_backward_attempted'], 1)
            self.assertEqual(guard.work['k_backward_completed'], 0)
            self.assertEqual(guard.work['k_backward_failed'], 1)
            self.assertEqual(guard.work['k_backward_calls'], 0)
            self.assertEqual(len(trainer.backend.optimizer.optimizer.state), 0)

    def test_guard_close_releases_runtime_owners_and_is_idempotent(self):
        trainer = self.trainer('Gclose', objective='osd_first_error')
        guard = install_guard(trainer, probe_steps=())
        guard.close()
        guard.close()
        self.assertIsNone(guard.model)
        self.assertIsNone(guard.trainer)
        self.assertEqual(guard.parameters, ())
        self.assertIsNone(guard.original_step)
        self.assertFalse(hasattr(trainer, 'prefix_guard'))
        self.assertFalse(hasattr(trainer.bundle.model, 'prefix_guard'))
        self.assertEqual(guard.report()['steps'], [])

    def test_original_K_contract_unchanged_when_no_guard_installed(self):
        trainer = self.trainer('Kplain', objective='osd_first_error')
        self.assertNotIn('prefix_preservation_guard', trainer.contract)
        self.assertFalse(hasattr(trainer.bundle.model, 'prefix_guard'))


if __name__ == '__main__':
    unittest.main()
