"""R1 policy routing uses one candidate check and preserves the native update."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from online_feedback import functional_geometry_math as geometry
from online_feedback.functional_geometry import (
    FEASIBLE_REALIZATION, install_functional_geometry,
)
from online_feedback.rng import capture_rng, restore_rng
from test_functional_geometry_runtime import equal_tree, run_window, state, tiny_trainer


class FeasibleRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def policy_fixture(self, arm='FSG'):
        trainer = tiny_trainer()
        hook = install_functional_geometry(trainer, arm=arm,
                                           realization_revision=FEASIBLE_REALIZATION)
        self.addCleanup(hook.close)
        parameter = trainer.bundle.draft_model.weight
        hook.active_parameters = (parameter,)
        hook.theta0 = (torch.zeros_like(parameter),)
        hook.theta_native = (torch.full_like(parameter, .5),)
        hook.window_context = {}
        hook.restore_endpoint(hook.theta_native)
        references = [SimpleNamespace(statistics=dict(protected_occurrences=2),
            native_risk_terms=(1., 2.)), SimpleNamespace(
            statistics=dict(protected_occurrences=3), native_risk_terms=(1., 3.))]
        sketch = SimpleNamespace(statistics={'sigma': 1.})
        native_risk = geometry.Risk(1., 1., 1e-12, 2.5)
        return hook, parameter, references, sketch, native_risk

    def realize(self, hook, value):
        hook.restore_endpoint((torch.full_like(hook.active_parameters[0], value),))
        candidate = hook._endpoint_statistics()
        raw = dict(candidate, norm=candidate['native_norm'] * 1.001, continuous_norm=.9)
        return dict(raw_candidate=raw, candidate=candidate, search_ran=True,
                    alpha=.9, bracket=[.9, .900001], bracket_norms=[candidate['norm'], 1.51],
                    iterations=24, probe_count=24, norm_scan_passes=27,
                    construction_scan_passes=26, identity_scan_passes=2, write_passes=2, costs={}, bypass=None)

    def test_lambda_zero_r1_is_exact_native_and_does_not_search(self):
        ordinary, revised = tiny_trainer(), tiny_trainer()
        revised.bundle.model.load_state_dict(ordinary.bundle.model.state_dict())
        rng = capture_rng()
        run_window(ordinary)
        expected = state(ordinary)
        restore_rng(rng)
        hook = install_functional_geometry(revised, strength=0.,
                                           realization_revision=FEASIBLE_REALIZATION)
        self.addCleanup(hook.close)
        with patch.object(geometry, 'realize_feasible_candidate') as search:
            run_window(revised)
        search.assert_not_called()
        self.assertTrue(equal_tree(expected, state(revised)))
        self.assertEqual(hook.steps[0]['decision']['reason'], 'lambda_zero')
        self.assertIsNone(hook.window_context)

    def test_empty_sketch_and_zero_native_bypass_without_search(self):
        for empty in (True, False):
            with self.subTest(empty=empty):
                trainer = tiny_trainer()
                if empty:
                    trainer.bundle.model.eos_token_ids = (0,)
                else:
                    trainer.backend.optimizer.optimizer.param_groups[0]['lr'] = 0.
                hook = install_functional_geometry(trainer, realization_revision=FEASIBLE_REALIZATION)
                with patch.object(geometry, 'realize_feasible_candidate') as search:
                    run_window(trainer)
                search.assert_not_called()
                self.assertTrue(hook.steps[0]['committed']['identical_native'])
                self.assertEqual(hook.steps[0]['decision']['reason'],
                                 'empty_sketch' if empty else 'zero_native_displacement')
                hook.close()

    def test_feasible_candidate_is_scored_even_when_original_rounded_norm_failed(self):
        hook, parameter, refs, sketch, native = self.policy_fixture()
        realized = self.realize(hook, .25)
        with patch.object(geometry, 'realize_feasible_candidate', return_value=realized), \
             patch.object(hook, 'score_candidate', return_value=(geometry.Risk(.4, .4, 1e-12, 2.),
                          [{'risk_unaveraged': .3}, {'risk_unaveraged': .5}])) as score:
            row = hook._filter_feasible(refs, sketch, native)
        score.assert_called_once_with(refs)
        self.assertGreater(row['raw_fsg']['norm'], row['raw_fsg']['native_norm'])
        self.assertTrue(row['decision']['accepted'])
        self.assertFalse(row['committed']['identical_native'])
        self.assertFalse(row['committed']['identical_prestate'])
        self.assertEqual(row['native_response_risks'][1]['risk_unaveraged'], 1.)
        self.assertEqual(hook.work['feasibility_probe_count'], 24)
        self.assertEqual(hook.work['feasibility_write_passes'], 2)

    def test_scalar_targets_own_realization_and_scores_only_scalar_endpoint(self):
        hook, parameter, refs, sketch, native = self.policy_fixture('SCALAR')
        realized = self.realize(hook, .25)
        target = realized['candidate']['norm']
        seen = []
        def scalar(parameters, theta0, theta_native, target_norm):
            seen.append(target_norm)
            hook.restore_endpoint((torch.full_like(parameter, .125),))
            return dict(geometry.endpoint_statistics(parameters, theta0, theta_native),
                target_norm=target_norm, absolute_norm_mismatch=.375,
                relative_norm_mismatch=.5, norm_scan_passes=25, iterations=24)
        def score(references):
            self.assertTrue(torch.equal(parameter, torch.full_like(parameter, .125)))
            return geometry.Risk(.4, .4, 1e-12, 2.), [{}, {}]
        with patch.object(geometry, 'realize_feasible_candidate', return_value=realized), \
             patch.object(geometry, 'scalar_match', side_effect=scalar) as matcher, \
             patch.object(hook, 'score_candidate', side_effect=score) as scorer:
            row = hook._filter_feasible(refs, sketch, native)
        self.assertEqual(seen, [target])
        matcher.assert_called_once()
        scorer.assert_called_once()
        self.assertTrue(row['scalar_matching_exercised'])
        self.assertEqual(row['scalar_target_norm'], target)
        self.assertEqual(row['scalar']['absolute_norm_mismatch'], .375)

    def test_scalar_hypothetical_native_still_runs_matching(self):
        hook, parameter, refs, sketch, native = self.policy_fixture('SCALAR')
        realized = self.realize(hook, .5)
        scalar = dict(realized['candidate'], target_norm=1.5, norm_scan_passes=25, iterations=24)
        with patch.object(geometry, 'realize_feasible_candidate', return_value=realized), \
             patch.object(geometry, 'scalar_match', return_value=scalar) as matcher, \
             patch.object(hook, 'score_candidate') as scorer:
            row = hook._filter_feasible(refs, sketch, native)
        matcher.assert_called_once()
        self.assertEqual(matcher.call_args.args[-1], realized['candidate']['native_norm'])
        scorer.assert_not_called()
        self.assertFalse(row['decision']['accepted'])
        self.assertEqual(row['final_candidate_check'], 'exact_native_risk_reuse')

    def test_zero_hypothetical_target_falls_back_before_scalar_matching(self):
        hook, parameter, refs, sketch, native = self.policy_fixture('SCALAR')
        realized = self.realize(hook, 0.)
        with patch.object(geometry, 'realize_feasible_candidate', return_value=realized), \
             patch.object(geometry, 'scalar_match') as matcher, \
             patch.object(hook, 'score_candidate') as scorer:
            row = hook._filter_feasible(refs, sketch, native)
        matcher.assert_not_called()
        scorer.assert_not_called()
        self.assertTrue(torch.equal(parameter, hook.theta_native[0]))
        self.assertEqual(row['decision']['reason'], 'zero_hypothetical_displacement')
        self.assertFalse(row['scalar_matching_exercised'])

    def test_exact_final_native_and_prestate_are_not_nontrivial_activity(self):
        for value, reason in ((.5, 'identical_native_endpoint'), (0., 'rounded_to_prestate')):
            with self.subTest(value=value):
                hook, parameter, refs, sketch, native = self.policy_fixture()
                realized = self.realize(hook, value)
                with patch.object(geometry, 'realize_feasible_candidate', return_value=realized), \
                     patch.object(hook, 'score_candidate') as score:
                    row = hook._filter_feasible(refs, sketch, native)
                score.assert_not_called()
                self.assertFalse(row['decision']['accepted'])
                self.assertEqual(row['decision']['reason'], reason)
                self.assertTrue(torch.equal(parameter, hook.theta_native[0]))

    def test_finite_kl_rejection_restores_exact_native_once(self):
        hook, parameter, refs, sketch, native = self.policy_fixture()
        realized = self.realize(hook, .25)
        with patch.object(geometry, 'realize_feasible_candidate', return_value=realized), \
             patch.object(hook, 'score_candidate', return_value=(geometry.Risk(2., 2., 1e-12, 2.),
                                                               [{}, {}])) as score:
            row = hook._filter_feasible(refs, sketch, native)
        score.assert_called_once()
        self.assertEqual(row['decision']['reason'], 'actual_risk_increased')
        self.assertTrue(torch.equal(parameter, hook.theta_native[0]))

    def test_real_r1_preserves_adam_rng_and_refreshes_context_between_windows(self):
        native, revised = tiny_trainer(), tiny_trainer()
        revised.bundle.model.load_state_dict(native.bundle.model.state_dict())
        rng = capture_rng()
        run_window(native)
        expected = state(native)
        restore_rng(rng)
        seen = []
        def observer(phase, hook):
            if phase == 'committed':
                seen.append([row.response['presentation'] for row in hook.window_context['references']])
        hook = install_functional_geometry(revised, realization_revision=FEASIBLE_REALIZATION,
                                           observer=observer)
        self.addCleanup(hook.close)
        run_window(revised)
        actual = state(revised)
        for key in ('optimizer', 'rng', 'modes', 'buffers', 'cache', 'grads', 'steps', 'presentations'):
            self.assertTrue(equal_tree(actual[key], expected[key]), key)
        run_window(revised)
        self.assertEqual(seen, [[0, 1], [2, 3]])
        self.assertEqual(revised.global_step, 2)
        self.assertIsNone(hook.window_context)
        self.assertLessEqual(hook.work['additional_response_forwards_completed'], 12)
        self.assertLessEqual(hook.work['additional_backwards_completed'], 4)

    def test_failed_search_preserves_partial_receipt_and_restores_native(self):
        trainer = tiny_trainer()
        native = []
        def observe(phase, hook):
            if phase == 'native':
                native.extend(parameter.detach().clone() for parameter in hook.active_parameters)
        hook = install_functional_geometry(trainer, realization_revision=FEASIBLE_REALIZATION,
                                           observer=observe)
        self.addCleanup(hook.close)
        def fail(*args, progress, **kwargs):
            progress.update(status='failed', phase='probe', probe_count=2,
                            completed_probes=1, write_passes=1, norm_scan_passes=2)
            raise ValueError('nonfinite feasibility probe')
        with patch.object(geometry, 'realize_feasible_candidate', side_effect=fail):
            with self.assertRaisesRegex(ValueError, 'nonfinite feasibility probe'):
                run_window(trainer)
        self.assertEqual(hook.steps[0]['realization']['completed_probes'], 1)
        self.assertEqual(hook.work['feasibility_probe_count'], 2)
        self.assertEqual(hook.steps[0]['status'], 'failed')
        self.assertTrue(torch.equal(trainer.bundle.draft_model.weight, native[0]))


if __name__ == '__main__':
    unittest.main()
