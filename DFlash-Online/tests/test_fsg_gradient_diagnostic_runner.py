"""CPU contracts for the zero-Adam diagnostic; no native accuracy claim."""
import json
from contextlib import ExitStack
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.functional_step_geometry import diagnose_gradients as diagnostic
from experiments.ok_continuation.resume import differences
from online_feedback.rng import capture_rng, preserve_rng


class DiagnosticWorkTests(unittest.TestCase):
    def test_runtime_pin_receipt_names_this_diagnostic_without_changing_pin(self):
        native = dict(effective=False, requested=False, torch_version='qualified',
                      applied_before='old qualification label', scope='old cases')
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(diagnostic, 'pin_retained_graph_runtime', return_value=native):
                receipt = diagnostic.pin_diagnostic_runtime(directory)
            saved = json.loads((Path(directory) / 'RUNTIME_COMPATIBILITY.json').read_text())
            self.assertEqual(saved, receipt)
            self.assertIs(saved['effective'], False)
            self.assertIs(saved['requested'], False)
            self.assertEqual(saved['torch_version'], 'qualified')
            self.assertEqual(saved['applied_before'], 'one retained-origin trainer construction')
            self.assertIn('zero-Adam', saved['scope'])

    def test_attempt_is_durable_before_call_and_failure_remains_charged(self):
        with tempfile.TemporaryDirectory() as directory:
            work = diagnostic.DiagnosticWork(directory)
            failure = RuntimeError('draft evaluation failed')
            with self.assertRaisesRegex(RuntimeError, 'draft evaluation failed') as caught:
                with work.operation('response_forward'):
                    saved = json.loads(work.path.read_text())
                    self.assertEqual(saved['counts'], {'response_forward_attempted': 1})
                    raise failure
            self.assertIs(caught.exception, failure)
            self.assertEqual(json.loads(work.path.read_text())['counts'], {
                'response_forward_attempted': 1,
                'response_forward_failed': 1,
            })

    def test_completed_calls_hit_cap_without_executing_extra_work(self):
        with tempfile.TemporaryDirectory() as directory:
            work = diagnostic.DiagnosticWork(directory)
            completed = []
            for index in range(diagnostic.BUDGET['response_forward']):
                with work.operation('response_forward'):
                    completed.append(index)
            before = work.path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'work limit exceeded'):
                with work.operation('response_forward'):
                    completed.append('must not execute')
            self.assertEqual(len(completed), 3)
            self.assertEqual(work.path.read_bytes(), before)
            self.assertEqual(work.counts['response_forward_completed'], 3)

    def test_zero_budget_categories_cannot_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            work = diagnostic.DiagnosticWork(directory)
            for kind in ('optimizer_step', 'feedback_generation',
                         'teacher_preparation', 'serving_generation'):
                with self.subTest(kind=kind):
                    with self.assertRaisesRegex(ValueError, 'work limit exceeded'):
                        with work.operation(kind):
                            self.fail('Zero-budget operation entered its body')
            self.assertEqual(work.counts, {})

    def test_optimizer_tripwire_does_not_change_weights_or_adam_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1., -2.]))
        optimizer = torch.optim.AdamW([parameter], lr=3e-4)
        parameter.grad = torch.tensor([.4, -.2])
        weights = parameter.detach().clone()
        before = optimizer.state_dict()
        wrapper = SimpleNamespace(step=diagnostic.forbid_step, optimizer=optimizer)
        optimizer.step = diagnostic.forbid_step
        for entry in (wrapper.step, wrapper.optimizer.step):
            with self.assertRaisesRegex(RuntimeError, 'Optimizer applications are forbidden'):
                entry()
        self.assertTrue(torch.equal(parameter, weights))
        self.assertFalse(differences(before, optimizer.state_dict()))
        self.assertEqual(len(optimizer.state), 0)


class RetainedAnchorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = dict(record_id='retained-response', prompt_length=2,
            input_ids=torch.arange(14).reshape(1, 14),
            loss_mask=torch.tensor([[0., 0.] + [1.] * 12]),
            hidden_states=torch.ones(1, 14, 3))
        self.feature = self.root / 'feature.pt'
        torch.save(self.raw, self.feature)
        self.presentations = self.root / 'presentations.json'
        self.records = [dict(record_id='retained-response', sampled_anchors=[
            dict(positions=[[7, 2, 9, 5]], keep_mask=[[True, True, True, True]])])]
        self.write_records()
        labels = torch.tensor([[[7, 8, 9], [2, 3, 4]]])
        weights = torch.tensor([[[0., 1., 1.], [0., 1., 1.]]])
        self.model = SimpleNamespace(
            _block_labels_and_weights=mock.Mock(return_value=(labels, weights, None)),
            _sample_anchors=mock.Mock(side_effect=AssertionError('Do not sample')))
        self.batch = dict(input_ids=self.raw['input_ids'], loss_mask=self.raw['loss_mask'],
                          hidden_states=self.raw['hidden_states'])
        self.trainer = SimpleNamespace(max_length=2272, bundle=SimpleNamespace(model=self.model),
            _batch=mock.Mock(return_value=SimpleNamespace(tensors=self.batch)))

    def write_records(self):
        self.presentations.write_text(json.dumps(self.records))

    def test_keeps_first_two_original_anchors_in_original_order(self):
        state = capture_rng()
        response, receipt = diagnostic.load_response(
            self.trainer, self.feature, self.presentations)
        self.assertEqual(response['anchors'].tolist(), [[7, 2]])
        self.assertEqual(response['keep'].tolist(), [[True, True]])
        self.assertEqual(receipt['original_anchor_count'], 4)
        self.assertEqual(receipt['record_id'], 'retained-response')
        self.assertIn('no sampler call', receipt['selection'])
        self.assertIs(response['input_ids'], self.batch['input_ids'])
        self.assertIs(response['hidden_states'], self.batch['hidden_states'])
        self.assertEqual(response['anchors'].dtype, torch.int64)
        self.assertEqual(response['keep'].dtype, torch.bool)
        self.model._sample_anchors.assert_not_called()
        self.model._block_labels_and_weights.assert_called_once()
        self.assertFalse(differences(state, capture_rng()))

    def test_mismatched_record_stops_before_batch_or_anchor_processing(self):
        self.records[0]['record_id'] = 'other-response'
        self.write_records()
        with self.assertRaisesRegex(ValueError, 'original anchor identity differ'):
            diagnostic.load_response(self.trainer, self.feature, self.presentations)
        self.trainer._batch.assert_not_called()
        self.model._block_labels_and_weights.assert_not_called()

    def test_short_or_invalid_original_fixture_is_not_resampled(self):
        for draw in (
            dict(positions=[[7]], keep_mask=[[True]]),
            dict(positions=[[7, 2, 9]], keep_mask=[[True, False, True]]),
        ):
            with self.subTest(draw=draw):
                self.records[0]['sampled_anchors'] = [draw]
                self.write_records()
                with self.assertRaisesRegex(ValueError, 'two-anchor fixture is unavailable'):
                    diagnostic.load_response(self.trainer, self.feature, self.presentations)
        self.trainer._batch.assert_not_called()
        self.model._sample_anchors.assert_not_called()


class ParameterVJPTests(unittest.TestCase):
    def test_vjp_preserves_existing_grads_rng_and_reusable_graph(self):
        with preserve_rng(), tempfile.TemporaryDirectory() as directory:
            parameter = torch.nn.Parameter(torch.tensor([2., 3.]))
            unused = torch.nn.Parameter(torch.tensor([4.]))
            ordinary_grad = torch.tensor([5., 6.])
            parameter.grad = ordinary_grad
            ordinary_before = ordinary_grad.clone()
            hidden = parameter.square()
            adjoint = torch.tensor([.5, 2.])

            def consume_randomness(gradient):
                torch.rand(5)
                np.random.rand(3)
                random.random()
                return gradient

            hook = parameter.register_hook(consume_randomness)
            self.addCleanup(hook.remove)
            work = diagnostic.DiagnosticWork(directory)
            before = capture_rng()
            first = diagnostic.parameter_vjp(hidden, adjoint, (parameter, unused), work)
            second = diagnostic.parameter_vjp(hidden, adjoint, (parameter, unused), work)
            self.assertTrue(torch.equal(first[0], torch.tensor([2., 12.])))
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertIsNone(first[1])
            self.assertFalse(first[0].requires_grad)
            self.assertIs(parameter.grad, ordinary_grad)
            self.assertTrue(torch.equal(parameter.grad, ordinary_before))
            self.assertIsNone(unused.grad)
            self.assertFalse(differences(before, capture_rng()))
            self.assertEqual(work.counts, {
                'draft_backward_attempted': 2, 'draft_backward_completed': 2})

    def test_failed_backward_preserves_rng_and_grad_and_is_durable(self):
        with preserve_rng(), tempfile.TemporaryDirectory() as directory:
            parameter = torch.nn.Parameter(torch.tensor([2.]))
            parameter.grad = torch.tensor([7.])
            original_grad = parameter.grad
            hidden = parameter.square()

            def failing_hook(gradient):
                torch.rand(5)
                random.random()
                raise RuntimeError('backward fixture failure')

            handle = parameter.register_hook(failing_hook)
            self.addCleanup(handle.remove)
            before = capture_rng()
            work = diagnostic.DiagnosticWork(directory)
            with self.assertRaisesRegex(RuntimeError, 'backward fixture failure'):
                diagnostic.parameter_vjp(hidden, torch.ones_like(hidden), (parameter,), work)
            self.assertFalse(differences(before, capture_rng()))
            self.assertIs(parameter.grad, original_grad)
            self.assertTrue(torch.equal(parameter.grad, torch.tensor([7.])))
            self.assertEqual(json.loads(work.path.read_text())['counts'], {
                'draft_backward_attempted': 1, 'draft_backward_failed': 1})


class RunnerCleanupTests(unittest.TestCase):
    def test_primary_failure_survives_secondary_cleanup_and_is_reported(self):
        with preserve_rng(), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'diagnostic'
            config = root / 'config.yaml'
            config.write_text('test configuration')
            qualification = root / 'qualification.json'
            qualification.write_text(json.dumps(dict(binding='retained-binding',
                original_config=str(config), datasets=dict(gsm8k=dict(
                    configs=dict(KPLUS='unused-config'), origin_state='unused-origin',
                    origin_export='unused-export')))))
            failed_step = root / 'step.json'
            failed_step.write_text(json.dumps(dict(active_parameter_names=['weight'])))
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(dict(output=str(output),
                execution_commit='cpu-test', storage=dict(maximum_additional_bytes=0,
                reserve_bytes=0), dependencies=[], qualification_plan=str(qualification),
                failed_step=str(failed_step), feature='unused', presentations='unused')))

            model = torch.nn.Linear(2, 1, bias=False)
            optimizer = torch.optim.AdamW(model.parameters())
            wrapper = SimpleNamespace(optimizer=optimizer, step=optimizer.step)
            trainer = SimpleNamespace(bundle=SimpleNamespace(model=model, draft_model=model),
                backend=SimpleNamespace(optimizer=wrapper), close=mock.Mock())
            state = dict(global_step=512, weights=model.state_dict(), rng=capture_rng())
            primary = ValueError('original diagnostic failure')
            geometry = SimpleNamespace(work={}, forward_calls={},
                close=mock.Mock(side_effect=RuntimeError('secondary hook cleanup failure')))
            with ExitStack() as patches:
                for name, replacement in (
                    ('FunctionalGeometryTrainer', mock.Mock(return_value=trainer)),
                    ('FunctionalGeometry', mock.Mock(return_value=geometry)),
                    ('scientific_state', mock.Mock(return_value=state)),
                    ('runtime_state', mock.Mock(return_value={})),
                    ('restore_runtime', mock.Mock(side_effect=RuntimeError('secondary restore failure'))),
                    ('optimizer_versions', mock.Mock(return_value=[])),
                    ('load_response', mock.Mock(side_effect=primary)),
                    ('pin_diagnostic_runtime', mock.Mock()),
                ):
                    patches.enter_context(mock.patch.object(diagnostic, name, replacement))
                patches.enter_context(mock.patch.object(torch.cuda, 'reset_peak_memory_stats'))
                with self.assertRaisesRegex(ValueError, 'original diagnostic failure') as caught:
                    diagnostic.run(plan_path)
            self.assertIs(caught.exception, primary)
            trainer.close.assert_called_once()
            receipt = json.loads((output / 'RESULT.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertEqual(receipt['error'], str(primary))
            notes = '\n'.join(primary.__notes__)
            self.assertIn('secondary restore failure', notes)
            self.assertIn('secondary hook cleanup failure', notes)
            self.assertEqual(receipt['additional_adam_steps'], 0)
            self.assertEqual(receipt['work'], {})
            self.assertEqual(receipt['dependency_postcheck'], [])


if __name__ == '__main__':
    unittest.main()
