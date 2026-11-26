"""Focused CPU checks for native capture, exact controls, and C-first plumbing."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.ok_coding.storage import identity
from online_feedback.storage import write_json
from . import point2a_native as native
from .allocation_qualify import vector_identity
from .allocation_scoring import NativeAllocationScorer
from .native import WorkReceipt
from .test_allocation_integration import fixture, verify_parent as tiny_verify_parent


class Ledger:
    def __init__(self):
        self.rows = []

    def reserve(self, case):
        self.rows.append(dict(case=case, status='reserved'))
        return len(self.rows) - 1

    def complete(self, reservation):
        self.rows[reservation]['status'] = 'completed'


def control_case():
    return dict(status='completed', adam_completed=1, native_preclip_norm=0.125,
        gradient_identity={'sha256': 'gradient'},
        endpoint=dict(parameters={'sha256': 'weights'}, exp_avg={'sha256': 'mean'},
            exp_avg_sq={'sha256': 'square'}, steps=[129.], options={'betas': [0.9, 0.999]},
            global_step=129, presentations=258, micro=258, rng_sha256='rng', gradients_cleared=True),
        rows=[dict(training_s=1., record_id='same', sampled_anchors=[1],
                   response_observation=dict(event_id='point1_c', loss=1.5, ratio_metrics={'loss': [3, 2]}))])


class ControlIdentityTests(unittest.TestCase):
    def test_only_labels_and_elapsed_time_are_excluded(self):
        historical = control_case()
        current = deepcopy(historical)
        current['rows'][0]['training_s'] = 999.
        current['rows'][0]['response_observation']['event_id'] = 'point2a_c'
        current['endpoint']['options']['betas'] = (0.9, 0.999)
        self.assertEqual(native.reproduce_control(current, historical)['status'], 'passed')

    def test_weights_alone_do_not_establish_scientific_equality(self):
        historical = control_case()
        for field in ('exp_avg', 'exp_avg_sq', 'steps', 'options', 'global_step',
                      'presentations', 'micro', 'rng_sha256', 'gradients_cleared'):
            with self.subTest(field=field):
                current = deepcopy(historical)
                current['endpoint'][field] = 'changed'
                with self.assertRaisesRegex(ValueError, 'reproduction failed'):
                    native.reproduce_control(current, historical)
                self.assertEqual(current['historical_reproduction']['status'], 'failed')

    def test_original_response_observation_is_scientific(self):
        current, historical = control_case(), control_case()
        current['rows'][0]['response_observation']['loss'] = 2.
        with self.assertRaisesRegex(ValueError, 'reproduction failed'):
            native.reproduce_control(current, historical)

    def test_authentication_rejects_modified_historical_receipt(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'case.json'
            write_json(path, control_case())
            binding = dict(path=str(path), identity=identity(path))
            contract = dict(historical_controls={'C': binding, 'F_BETA': binding})
            self.assertEqual(set(native.authenticate_controls(contract)), {'C', 'F_BETA'})
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                native.authenticate_controls(contract)


class NativeCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def make(self, root):
        trainer, state, features, responses, _, _, allocated = fixture(root)
        trainer.events = state['events'] = [dict(event_id='event_001')]
        trainer._anchor_observations.clear()
        trainer.response_records.clear()
        trainer.window_records.clear()
        scorer = NativeAllocationScorer(trainer, responses, collect_diagnostics=True)
        scorer.freeze_current()
        runtime, ancillary = native.runtime_state(trainer.bundle.model), native.ancillary_state(trainer)
        allocations = dict(C=native.native_c_coefficients(scorer), F_BETA=allocated, F_C=allocated)
        return trainer, scorer, state, runtime, ancillary, features, responses, allocations

    def invoke(self, values, name, root, ledger, retained, historical=None):
        allocations = values[-1]
        output = Path(root) / ('run_' + name)
        output.mkdir()
        work = WorkReceipt(output / 'WORK.json')
        with patch.object(native, 'verify_live',
                          side_effect=lambda trainer, state: tiny_verify_parent(trainer, state, None)):
            result = native.run_case(name, *values[:-1], allocations, retained, ledger, work,
                output, vector_identity(['response_0', 'response_1'], allocations[name]), historical)
        return result, work

    def test_native_C_capture_is_before_original_step_and_restores_all_state(self):
        with TemporaryDirectory() as tmp:
            values = self.make(tmp)
            trainer, scorer = values[:2]
            observed = {}
            original = trainer.backend.optimizer.step
            def step():
                observed['preclip'] = tuple(p.grad.detach().clone() for p in scorer.parameters)
                return original()
            trainer.backend.optimizer.step = step
            retained, ledger = dict(gradients={}, native={}), Ledger()
            result, work = self.invoke(values, 'C', tmp, ledger, retained)
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(ledger.rows, [dict(case='C', status='completed')])
            self.assertEqual(work.value['native_adam_steps'], 1)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(
                observed['preclip'], retained['gradients']['C'])))
            self.assertTrue(result['restoration']['runtime_exact'])
            self.assertTrue(result['restoration']['ancillary_exact'])
            self.assertEqual((trainer.global_step, trainer.presentations, len(trainer.events)), (128, 256, 1))
            self.assertIs(trainer.backend.optimizer.step, step)

    def test_failed_adam_remains_charged_and_parent_is_restored(self):
        with TemporaryDirectory() as tmp:
            values = self.make(tmp)
            trainer = values[0]
            trainer.backend.optimizer.step = lambda: (_ for _ in ()).throw(RuntimeError('injected Adam failure'))
            ledger = Ledger()
            with self.assertRaisesRegex(RuntimeError, 'injected Adam failure'):
                self.invoke(values, 'C', tmp, ledger, dict(gradients={}, native={}))
            self.assertEqual(ledger.rows, [dict(case='C', status='reserved')])
            case = json.loads((Path(tmp) / 'run_C/C.json').read_text())
            self.assertEqual(case['status'], 'failed')
            self.assertTrue(case['restoration']['parent_restored'])
            self.assertEqual(trainer.global_step, 128)

    def test_cached_candidate_starts_at_original_parent_and_preserves_C(self):
        with TemporaryDirectory() as tmp:
            values = self.make(tmp)
            retained, ledger = dict(gradients={}, native={}), Ledger()
            control, _ = self.invoke(values, 'C', tmp, ledger, retained)
            c_identity = vector_identity(values[1].parameter_names, retained['native']['C'])
            candidate, _ = self.invoke(values, 'F_BETA', tmp, ledger, retained)
            self.assertEqual([r['case'] for r in ledger.rows], ['C', 'F_BETA'])
            self.assertEqual(candidate['endpoint']['global_step'], 129)
            self.assertEqual(control['endpoint']['global_step'], 129)
            self.assertEqual(c_identity, vector_identity(values[1].parameter_names, retained['native']['C']))

    def test_modified_sealed_coefficients_stop_before_any_native_call(self):
        with TemporaryDirectory() as tmp:
            values = self.make(tmp)
            ledger, retained = Ledger(), dict(gradients={}, native={})
            with self.assertRaisesRegex(ValueError, 'changed after sealing'):
                native.run_case('C', *values[:-1], values[-1], retained, ledger,
                    WorkReceipt(Path(tmp) / 'WORK.json'), Path(tmp), {'sha256': 'wrong'})
            self.assertEqual(ledger.rows, [])

    def test_complete_small_native_schedule_emits_all_three_cases_and_readouts(self):
        """Exercise cross-module plumbing; the separately tested oracle is stubbed."""
        from . import point2a_math
        with TemporaryDirectory() as tmp:
            values = self.make(tmp)
            trainer, _, state, runtime, ancillary, features, responses, _ = values
            scorer = NativeAllocationScorer(trainer, responses, collect_diagnostics=True)
            output = Path(tmp) / 'complete'
            output.mkdir()
            def qualified(scorer, direction, output, *args):
                fast = scorer.fast_scores(direction)
                receipt = dict(status='passed', direction_metadata=dict(effective=
                    vector_identity(scorer.parameter_names, fast['effective_direction'])))
                del fast['effective_direction']
                write_json(output / 'QUALIFICATION.json', receipt)
                return receipt, fast
            with patch.object(native, 'verify_live',
                              side_effect=lambda trainer, state: tiny_verify_parent(trainer, state, None)), \
                 patch.object(point2a_math, 'select_reference_pools', return_value={'fixture': True}), \
                 patch.object(point2a_math, 'qualify_direction', side_effect=qualified):
                native.restore_parent(trainer, state, runtime, ancillary, output / 'START.json')
                parent = native._endpoint_identity(trainer)
                ledger = Ledger()
                result = native.execute_cases(trainer, scorer, state, runtime, ancillary,
                    features, responses, ledger, WorkReceipt(output / 'WORK.json'),
                    {'witness': {'status': 'unavailable'}}, {'C': None, 'F_BETA': None}, parent, output)
            self.assertEqual(list(result['cases']), ['C', 'F_BETA', 'F_C'])
            self.assertEqual([row['case'] for row in ledger.rows], ['C', 'F_BETA', 'F_C'])
            self.assertEqual(set(result['contrasts']), {'F_C-F_BETA', 'F_C-C', 'F_BETA-C'})
            for name in native.CASES:
                self.assertTrue((output / (name + '.json')).exists())
                self.assertTrue((output / ('window_N_' + name + '.json')).exists())
            self.assertEqual(result['contrasts']['F_C-C']['native_readouts']['new_buffer'], None)
            self.assertEqual(json.loads((output / 'SCORING_STATE_PRESERVATION.json').read_text())['status'], 'passed')


class ScheduleTests(unittest.TestCase):
    def test_failed_new_direction_stops_after_C_before_candidates(self):
        from . import point2a_math
        events = []
        scorer = SimpleNamespace(snapshots=[], freeze_current=lambda: events.append('freeze'))
        trainer = SimpleNamespace(work={})
        def one_case(name, *args, **kwargs):
            events.extend([name, 'restored'])
            return {'gradient_identity': {'sha256': 'C'}}
        def direction(*args):
            self.assertEqual(events[-1], 'restored')
            events.append('C preview at parent')
            return (), {}
        with TemporaryDirectory() as tmp, \
             patch.object(point2a_math, 'select_reference_pools', side_effect=lambda *a, **k: events.append('seal pools') or {}), \
             patch.object(native, 'seal_beta', side_effect=lambda *a: events.append('seal beta') or ({}, (), {'coefficients': {'C': {}}})), \
             patch.object(native, 'run_case', side_effect=one_case), \
             patch.object(native, 'c_direction', side_effect=direction), \
             patch.object(point2a_math, 'qualify_direction', return_value=({'status': 'failed'}, {})):
            with self.assertRaisesRegex(ValueError, 'failed qualification'):
                native.execute_cases(trainer, scorer, None, None, {'work': {}}, [], [],
                    None, None, {}, {'C': {}}, {}, Path(tmp))
        self.assertEqual(events, ['freeze', 'seal pools', 'seal beta', 'C', 'restored', 'C preview at parent'])

    def test_C_preview_uses_captured_gradient_and_Q_C_not_native_endpoint(self):
        from . import point2a_math
        parent = {'parameters': {'sha256': 'parent'}, 'steps': [128.]}
        gradient = (torch.tensor([0.25], dtype=torch.bfloat16),)
        scorer = SimpleNamespace(parameter_names=('weight',), parameters=(torch.tensor([0.]),),
            trainer=object(), _operation=lambda name: __import__('contextlib').nullcontext())
        retained = dict(gradients={'C': gradient}, native={'C': (torch.tensor([999.]),)})
        preview = (torch.tensor([3.], requires_grad=True),)
        captured = vector_identity(scorer.parameter_names, gradient)
        with patch.object(native, '_endpoint_identity', return_value=parent), \
             patch.object(native, 'shadow_at_parent', return_value=(preview, {'clip': 1.})) as shadow, \
             patch.object(native, 'assessment_cotangent', return_value=((torch.tensor([4.]),), {})) as assess, \
             patch.object(native, 'optimizer_vjp', return_value=(torch.tensor([5.]),)):
            direction, receipt = native.c_direction(scorer, retained, parent, captured)
        self.assertTrue(torch.equal(shadow.call_args.args[1][0], gradient[0]))
        self.assertIs(assess.call_args.args[1], preview)
        self.assertEqual(assess.call_args.args[2], 'shadow_C')
        self.assertEqual(float(direction[0]), 5.)
        self.assertTrue(receipt['binding']['parent_restored_before_shadow'])

    def test_post_C_moments_are_rejected_before_shadow(self):
        scorer = SimpleNamespace(parameter_names=('weight',), parameters=(torch.tensor([0.]),), trainer=object())
        gradient = (torch.tensor([0.25], dtype=torch.bfloat16),)
        retained = {'gradients': {'C': gradient}}
        with patch.object(native, '_endpoint_identity', return_value={'steps': [129.]}), \
             patch.object(native, 'shadow_at_parent') as shadow:
            with self.assertRaisesRegex(ValueError, 'fully restored parent'):
                native.c_direction(scorer, retained, {'steps': [128.]},
                    vector_identity(scorer.parameter_names, gradient))
            shadow.assert_not_called()


if __name__ == '__main__':
    unittest.main()
