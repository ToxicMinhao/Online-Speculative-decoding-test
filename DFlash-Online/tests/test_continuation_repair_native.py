"""Warm-state, bounded-work and native qualification plumbing without CUDA."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.continuation_repair_pilot import native, qualification
from experiments.continuation_repair_pilot.configuration import build_config, validate_config, PARENT_BINDING
from online_feedback.continuation_repair import high_pool_coefficients
from test_continuation_repair import native_pool, reduce_fixture


class NativeContinuationTests(unittest.TestCase):
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def fixture(self):
        beta = torch.ones(6)
        high = torch.tensor([True, True, False, False, False, False])
        scores = torch.tensor([0., 1., 0., 0., 0., 0.])
        candidate, ordinary, pool = high_pool_coefficients(
            beta, high, scores, native_pool(beta, high), (beta[high]*scores[high]).sum())
        return dict(beta=beta, high=high, scores=scores, candidate=candidate, native=ordinary)

    def test_literal_oracle_and_wrong_native_rejected(self):
        payload = self.fixture()
        result = qualification.coefficient_check(payload)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['changed_high_positions'], 2)
        self.assertTrue(result['structurally_active'])
        # Moving both reported native and candidate together must not fool the oracle.
        bad = deepcopy(payload)
        bad['candidate'] += .01
        bad['native'] += .01
        result = qualification.coefficient_check(bad)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['checks']['native_coefficient_oracle'])

    def test_neutral_empty_and_wrong_low(self):
        empty = dict(beta=torch.empty(0), high=torch.empty(0, dtype=torch.bool),
                     scores=torch.empty(0), native=torch.empty(0), candidate=torch.empty(0))
        result = qualification.coefficient_check(empty)
        self.assertEqual(result['status'], 'passed')
        self.assertFalse(result['structurally_active'])
        payload = self.fixture()
        payload['candidate'][-1] += .001
        self.assertFalse(qualification.coefficient_check(payload)['checks']['low_exact'])

    def test_real_summary_serializes_and_activity_counts_original_responses(self):
        beta = torch.ones(6)
        high = torch.tensor([True, True, False, False, False, False])
        _, _, summary = reduce_fixture(beta, high, torch.tensor([0., 1., 0., 0., 0., 0.]),
                                       torch.arange(6.).requires_grad_(), torch.ones(6))
        trainer = SimpleNamespace(bundle=SimpleNamespace(model=SimpleNamespace(
            prefix_allocation=SimpleNamespace(last_summary={'checked': True}),
            continuation_repair=SimpleNamespace(last_summary=summary))))
        row = {'sum_metrics': {}}
        with mock.patch.object(native.native.FullKTrainer, '_append_observation'):
            native.WarmContinuationTrainer._append_observation(trainer, 'RESPONSE_OBSERVATIONS.jsonl', row)
        json.dumps(row, allow_nan=False)
        observed = native.response_activity([row, deepcopy(row)])
        self.assertEqual(observed['original_responses'], 2)
        self.assertEqual(observed['active_response_count'], 2)
        self.assertEqual(observed['continuation_changed_high_positions'], 4)
        trainer.bundle.model.continuation_repair = None
        disabled = {}
        with mock.patch.object(native.native.FullKTrainer, '_append_observation'):
            native.WarmContinuationTrainer._append_observation(trainer, 'RESPONSE_OBSERVATIONS.jsonl', disabled)
        self.assertNotIn('continuation_repair', disabled)

    def test_backward_attempt_is_preserved_on_failure(self):
        backend = SimpleNamespace(backward=mock.Mock(side_effect=[7, ValueError('fixture')]))
        trainer = SimpleNamespace(backend=backend, work={})
        native.WarmContinuationTrainer._install_counted_backward(trainer)
        self.assertEqual(backend.backward(torch.tensor(1.), is_boundary=False), 7)
        with self.assertRaisesRegex(ValueError, 'fixture'):
            backend.backward(torch.tensor(2.), is_boundary=True)
        self.assertEqual(trainer.work, {'backwards_attempted': 2, 'backwards_completed': 1})

    def test_configuration_only_changes_metadata(self):
        original = 'run_id: old\noutput_dir: /old\ntraining:\n  learning_rate: 0.0003\n'
        current = build_config(original, 'TAIL', '/new')
        self.assertEqual(validate_config(current, original)['changed_settings'], [])
        with self.assertRaises(ValueError):
            validate_config(current.replace('0.0003', '0.0001'), original)

    def test_restore_precedes_installation_without_foreign_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root/'parent.json'
            config = root/'parent.yaml'
            config.write_text('run_id: native\n')
            parent.write_text(json.dumps(dict(configs={'PAIR': str(config)}, original_config=str(config), root=str(root))))
            order = []
            def construct(trainer, config_path, output_root, **kwargs):
                trainer.output_root = Path(output_root)
                trainer.output_root.mkdir()
                trainer.bundle = SimpleNamespace(model=SimpleNamespace())
                trainer.contract = {'native_parent': True}
                trainer.parent_validation = {'exact': True}
                order.append('restore')
            def install(model, *, enabled):
                order.append('install')
                self.assertEqual(order, ['restore', 'install'])
                self.assertTrue(enabled)
            with mock.patch.object(native.SpecForgeEventTrainer, '__init__', construct), \
                 mock.patch.object(native.WarmContinuationTrainer, '_install_counted_step'), \
                 mock.patch.object(native.WarmContinuationTrainer, '_install_counted_backward'), \
                 mock.patch('online_feedback.continuation_repair.install_continuation_repair', install):
                trainer = native.WarmContinuationTrainer(config, root/'runtime', arm='TAIL', binding='a'*64,
                    parent_plan=parent, retained_event={})
            self.assertEqual(trainer.arm, 'TAIL')
            self.assertEqual(trainer.contract[native.SCHEMA]['parent_binding'], PARENT_BINDING)

    def test_preservation_capture_ignores_backward_recomputation(self):
        allocation = SimpleNamespace(coefficient_probe=None)
        coefficients = torch.arange(32.).reshape(2,16)
        prefix = coefficients < 9
        def forward():
            allocation.coefficient_probe(dict(candidate=coefficients, prefix=prefix))
            return 3
        trainer = SimpleNamespace(bundle=SimpleNamespace(model=SimpleNamespace(prefix_allocation=allocation)),
                                  author_strategy=SimpleNamespace(forward_loss=forward))
        case = {}
        qualification.install_preservation_capture(trainer, case)
        self.assertEqual(trainer.author_strategy.forward_loss(), 3)
        allocation.coefficient_probe(dict(candidate=coefficients+1, prefix=prefix))
        self.assertEqual(len(case['preservation_coefficients']), 1)
        trainer.author_strategy.forward_loss()
        self.assertEqual(case['preservation_coefficients'][0], case['preservation_coefficients'][1])

    def test_scientific_comparison_keeps_gradient_rng_and_loss(self):
        case = dict(rows=[dict(training_s=1., response_observation=dict(event_id='a', loss=2.))],
                    gradient={'x': 'a'}, endpoint={'rng': 'a'}, gradient_norm=4.)
        alternate = deepcopy(case)
        alternate['rows'][0]['training_s'] = 8.
        alternate['rows'][0]['response_observation']['event_id'] = 'b'
        self.assertEqual(qualification.scientific_case(case), qualification.scientific_case(alternate))
        for location, key in (('endpoint','rng'), ('gradient','x')):
            alternate = deepcopy(case)
            alternate[location][key] = 'changed'
            self.assertNotEqual(qualification.scientific_case(case), qualification.scientific_case(alternate))

    def test_qualifier_structural_activity_controls_second_fixture(self):
        for active, expected_calls in ((True, 3), (False, 6)):
            with self.subTest(active=active), tempfile.TemporaryDirectory() as directory:
                plan = dict(root=directory, binding='b'*64, execution_commit='c'*40)
                def run_case(plan, arm, fixture, output, ledger):
                    token = ledger.reserve(arm)
                    ledger.complete(token)
                    preserve = {'fixture': fixture, 'rows': 16}
                    return dict(rows=[], gradient='g', endpoint={'parameters': 'p'}, gradient_norm=1.,
                        preservation_coefficients=[preserve, preserve], retained_BASE_first_window_exact=True,
                        actual_work=dict(response_forwards=2, backwards_attempted=2, backwards_completed=2,
                                         vocabulary_head_calls=20),
                        coefficient_checks=[dict(changed_high_positions=0, structurally_active=active,
                                                 preservation_coefficients=preserve)]*2)
                with mock.patch('experiments.continuation_repair_pilot.launch.boundary'), \
                     mock.patch.object(qualification, 'pin_retained_graph_runtime'), \
                     mock.patch.object(qualification, 'run_case', run_case):
                    result = qualification.qualify(plan)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(len(result['qualification_work']['calls']), expected_calls)
                # Structural activity suffices even when observed rounded movement is zero.
                self.assertEqual(result['activity'][0]['changed_high_positions'], 0)

    def test_failed_training_and_close_still_persist_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'q.json').write_text(json.dumps(dict(status='passed', binding='d'*64,
                base_disabled_exact=True, retained_BASE_reusable=True)))
            (root/'event.json').write_text('{}')
            (root/'features.json').write_text('[]')
            plan = dict(root=directory, qualification_result=str(root/'q.json'), binding='d'*64,
                execution_commit='a'*40, retained_event=str(root/'event.json'),
                configs={'TAIL':'unused'}, parent_plan='unused', feature_manifest=str(root/'features.json'))
            trainer = SimpleNamespace(work={'response_forwards':1}, response_records=[],
                train_event=mock.Mock(side_effect=ValueError('original training failure')),
                close=mock.Mock(side_effect=RuntimeError('close failure')))
            with mock.patch('experiments.prefix_guard.qualify.pin_retained_graph_runtime'), \
                 mock.patch('experiments.continuation_repair_pilot.launch.boundary'), \
                 mock.patch.object(native, 'WarmContinuationTrainer', return_value=trainer), \
                 mock.patch.object(native.parent_training, 'discarded_startup', return_value={}):
                with self.assertRaisesRegex(ValueError, 'original training failure'):
                    native.train(plan)
            result = json.loads((root/'arms/TAIL/STATUS.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertIn('close failure', result['close_error'])
            self.assertEqual(result['actual_work']['response_forwards'], 1)


if __name__ == '__main__':
    unittest.main()
