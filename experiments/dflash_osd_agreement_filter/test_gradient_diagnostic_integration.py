"""Direct-full-gradient training retains numerical evidence without a sum gate.

The historical helper's default enforcement remains tested for diagnostic replay.
Current training explicitly observes finite non-additivity and still rejects
nonfinite gradients before any optimizer step.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.dflash_osd_agreement_filter import gradient_diagnostics as diagnostics
from experiments.dflash_osd_agreement_filter import training
from experiments.dflash_osd_agreement_filter.test_training import trainer, response


class GradientDiagnosticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_gradients(self, *, invalid=1.):
        parameters = {'weight': torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))}
        a = {'weight': torch.tensor([invalid, 0.], dtype=torch.float32)}
        b = {'weight': torch.zeros(2, dtype=torch.float32)}
        reference = {'weight': torch.zeros(2, dtype=torch.bfloat16)}
        return a, b, reference, parameters

    def test_training_check_writes_before_observer_and_original_raise(self):
        args = self.make_gradients()
        context = {'event': 1, 'pass_index': 0, 'trial_type': 'disabled_filter',
                   'record_id': 'private-identity-only', 'state_identity_sha256': '0' * 64,
                   'sampling': {'anchors': [21, 28], 'shared_denominator': 17.5},
                   'groups': {'a': [21, 22], 'b': [23, 24]},
                   'losses': {'a': .2, 'b': .3, 'full': .5}}
        observed = []
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'checked_pass.json'
            def observer(summary, raw_a, raw_b, full):
                self.assertTrue(report.is_file())
                self.assertEqual(json.loads(report.read_text()), summary)
                self.assertEqual(summary['context'], context)
                self.assertIs(raw_a, args[0])
                self.assertIs(raw_b, args[1])
                self.assertIs(full, args[2])
                observed.append(summary['status'])
            with self.assertRaisesRegex(ValueError, '^Grouped raw gradients differ beyond declared sum tolerance$'):
                training.check_gradient_sum(*args, context=context, report_path=report, observer=observer)
            retained = json.loads(report.read_text())
        self.assertEqual(observed, ['failed'])
        self.assertEqual(retained['tolerance_violation_coordinates'], 1)
        self.assertEqual(retained['coordinates'], 2)
        self.assertEqual(retained['context'], context)

    def test_nonfinite_check_still_fails_after_durable_explicit_receipt(self):
        args = self.make_gradients(invalid=float('nan'))
        for enforce in (True, False):
            with self.subTest(enforce_tolerance=enforce), tempfile.TemporaryDirectory() as folder:
                report = Path(folder) / 'nonfinite.json'
                with self.assertRaisesRegex(ValueError, '^Nonfinite gradient qualification$'):
                    training.check_gradient_sum(*args, context={'losses': {'a': float('inf')}},
                                                report_path=report, enforce_tolerance=enforce)
                retained = json.loads(report.read_text())
            self.assertEqual(retained['status'], 'failed')
            self.assertEqual(retained['nonfinite_counts']['a'], 1)
            self.assertEqual(retained['samples'][0]['g_a']['nonfinite_float'], 'nan')
            self.assertEqual(retained['context']['losses']['a']['nonfinite_float'], 'inf')
            json.dumps(retained, allow_nan=False)

    def test_explicit_observation_mode_retains_finite_violation_without_admission_failure(self):
        args = self.make_gradients()
        observed = []
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'observation.json'
            def observer(summary, *_):
                self.assertEqual(json.loads(report.read_text()), summary)
                observed.append(summary['status'])
            summary = training.check_gradient_sum(*args, report_path=report, observer=observer,
                                                  enforce_tolerance=False)
            retained = json.loads(report.read_text())
        self.assertEqual(observed, ['failed'])
        self.assertEqual(summary, retained)
        self.assertEqual(summary['tolerance_violation_coordinates'], 1)
        self.assertEqual(summary['nonfinite_input_coordinates'], 0)

    def test_success_is_also_durable_and_tolerance_definition_is_unchanged(self):
        a, b, reference, parameters = self.make_gradients(invalid=0.)
        expected = 'abs(sum-reference) <= 8*eps_native*(abs(A)+abs(B)+abs(reference))+1e-8'
        self.assertEqual(training.GROUP_PROTOCOL['gradient_sum_tolerance'], expected)
        self.assertEqual(diagnostics.SUM_RULE, expected)
        before = copy.deepcopy(training.EQUIVALENCE_TOLERANCES)
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'passed.json'
            summary = training.check_gradient_sum(a, b, reference, parameters, report_path=report)
            self.assertEqual(json.loads(report.read_text()), summary)
        self.assertEqual(summary['status'], 'passed')
        self.assertEqual(summary['rule'], expected)
        self.assertEqual(training.EQUIVALENCE_TOLERANCES, before)
        self.assertEqual(before['parameter'], {'native_ulp_multiplier': 1, 'absolute': 1e-7})
        self.assertEqual(before['exp_avg'], {'native_epsilon_multiplier': 8, 'absolute': 1e-8})
        self.assertEqual(before['exp_avg_sq'], {'native_epsilon_multiplier': 16, 'absolute': 1e-12})
        self.assertEqual(before['displacement_relative_error_max'], .10)
        self.assertIs(training.COMPILER_POLICY['donated_buffer'], False)

    def test_toy_trainer_records_finite_nonadditivity_but_uses_unchanged_full_gradient(self):
        model = trainer('G', qualification=True, filter_enabled=False)
        first_response = response(1)
        original_check = training.check_gradient_sum
        observed = []
        with tempfile.TemporaryDirectory() as folder:
            model.gradient_diagnostics_dir = Path(folder)
            report = Path(folder) / 'event_01_pass_00_disabled_filter.json'
            def observer(summary, raw_a, raw_b, full):
                index = summary['context']['pass_index']
                pass_report = Path(folder) / ('event_01_pass_%02d_disabled_filter.json' % index)
                self.assertEqual(json.loads(pass_report.read_text()), summary)
                observed.append(summary)
            model.gradient_diagnostics_observer = observer
            def deliberate_numerical_failure(a, b, full, parameters, **kwargs):
                # Only the test's copied reference is perturbed. No forward,
                # anchor draw, model parameter, optimizer or RNG is replaced.
                changed = dict(full)
                name = next(iter(changed))
                changed[name] = changed[name].detach().float().clone()
                changed[name].view(-1)[0] += 100.
                return original_check(a, b, changed, parameters, **kwargs)
            with patch.object(training, 'check_gradient_sum', side_effect=deliberate_numerical_failure):
                result = model.train_event(1, [first_response])
            self.assertTrue(report.is_file())
            retained = json.loads(report.read_text())
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(item['status'] == 'failed' for item in observed))
        self.assertFalse(model.failed)
        self.assertEqual(model.optimizer_steps, 2)
        self.assertEqual(model.completed_events, 1)
        for batch in result['batches']:
            presentation = batch['presentations'][0]
            self.assertFalse(presentation['sum_tolerance_is_admission_gate'])
            self.assertEqual(presentation['gradient_backward_calls'], 3)
            self.assertEqual(batch['agreement']['proposal_gradient_source'], 'direct_full_loss_gradient')
        # Perturbing only a copied diagnostic reference cannot affect the actual
        # independently differentiated full gradient used for AdamW.
        control = trainer('G', qualification=True, filter_enabled=False)
        control.train_event(1, [first_response])
        for name, parameter in model.parameters.items():
            self.assertTrue(torch.equal(parameter, control.parameters[name]), name)
        self.assertEqual(training.state_digest(model.optimizer.state_dict()),
                         training.state_digest(control.optimizer.state_dict()))
        context = retained['context']
        self.assertEqual(context['event'], 1)
        self.assertEqual(context['pass_index'], 0)
        self.assertEqual(context['trial_type'], 'disabled_filter')
        self.assertEqual(context['record_id'], first_response['record_id'])
        self.assertEqual(context['optimizer_steps_before'], 0)
        self.assertEqual(context['response_audit'], training.audit_response(first_response))
        self.assertEqual(context['groups'], training.response_groups(training.audit_response(first_response)))
        for key in ('state_identity_sha256', 'anchor_rng_before_sha256', 'global_rng_before_sha256',
                    'anchor_rng_after_sha256', 'global_rng_after_sha256'):
            self.assertEqual(len(context[key]), 64, key)
        self.assertGreater(context['sampling']['normalization_denominator'], 0)
        self.assertTrue(context['sampling']['anchor_positions'])
        self.assertAlmostEqual(context['losses']['full'], context['losses']['a'] + context['losses']['b'], places=6)
        self.assertEqual(retained['rule'], training.GROUP_PROTOCOL['gradient_sum_tolerance'])

    def test_toy_trainer_nonfinite_observation_still_fails_before_optimizer(self):
        model = trainer('G', qualification=True, filter_enabled=False)
        original_check = training.check_gradient_sum
        def inject_nonfinite(a, b, full, parameters, **kwargs):
            changed = dict(a)
            name = next(iter(changed))
            changed[name] = changed[name].detach().clone()
            changed[name].view(-1)[0] = float('nan')
            return original_check(changed, b, full, parameters, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            model.gradient_diagnostics_dir = Path(folder)
            with patch.object(training, 'check_gradient_sum', side_effect=inject_nonfinite):
                with self.assertRaisesRegex(ValueError, '^Nonfinite gradient qualification$'):
                    model.train_event(1, [response(1)])
            retained = json.loads((Path(folder) / 'event_01_pass_00_disabled_filter.json').read_text())
        self.assertTrue(model.failed)
        self.assertEqual(model.optimizer_steps, 0)
        self.assertFalse(model.optimizer.state)
        self.assertEqual(retained['nonfinite_input_coordinates'], 1)


if __name__ == '__main__':
    unittest.main()
