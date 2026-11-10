"""Zero-Adam interception tests; CPU parity does not qualify CUDA startup."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from experiments.functional_step_geometry import diagnose_ordinary_gradients as diagnostic
from experiments.functional_step_geometry import qualify
from experiments.functional_step_geometry.diagnose_gradients import DiagnosticWork
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import restore_rng
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepContext, StepOutput
from test_fsg_qualification import replay_fixture
from test_functional_geometry_runtime import TinyBlocks


def restore_fixture(trainer, origin, baseline):
    """Restore the fixture's complete origin without invoking any optimizer step."""
    trainer.bundle.draft_model.load_state_dict(origin['weights'])
    trainer.backend.optimizer.load_state_dict(deepcopy(origin['optimizer']))
    trainer.backend.optimizer.zero_grad()
    trainer.global_step = origin['global_step']
    trainer.presentations = origin['presentations']
    trainer.core._micro = origin['micro']
    trainer.core._ratio_totals.clear()
    trainer.core._sum_totals.clear()
    trainer._anchor_observations.clear()
    trainer.response_records.clear()
    trainer.window_records.clear()
    trainer.work.clear()
    qualify.restore_runtime(trainer.bundle.model, baseline)
    restore_rng(origin['rng'])


def manual_gradients(trainer, paths, origin, baseline):
    """The control has two actual ordinary backwards, no TrainerCore boundary."""
    trainer._active_event = 'manual_control'
    trainer.bundle.model.begin_event()
    for index, path in enumerate(paths):
        raw = torch.load(path, weights_only=True)
        trainer._anchor_observations.clear()
        result = trainer.author_strategy.forward_loss(raw, StepContext(global_step=512, total_steps=7424))
        trainer.backend.backward(result.loss / 2, is_boundary=index == 1)
        trainer.presentations += 1
    gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                 for name, parameter in trainer.bundle.draft_model.named_parameters()}
    observations = deepcopy(trainer.response_records)
    restore_fixture(trainer, origin, baseline)
    return gradients, observations


def scalar_fixture(root):
    """Reuse the real BF16 block/capture fixture with the real TrainerCore."""
    trainer, paths = replay_fixture(root)
    model = TinyBlocks()
    model.begin_event = lambda: None
    optimizer = DirectBF16AdamW(model.draft_model)
    for parameter in optimizer.model_params:
        optimizer.optimizer.state[parameter] = dict(step=torch.tensor(512.),
            exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
    trainer.bundle = SimpleNamespace(model=model, draft_model=model.draft_model)
    trainer.backend = SimpleNamespace(optimizer=optimizer,
        backward=lambda loss, **kwargs: loss.backward(), step=lambda: optimizer.step())
    trainer.objective = 'osd_first_error'
    trainer.reduction = 'response_mean'
    trainer.accumulation = 2

    def forward(raw, _context):
        anchors, keep, hidden = model._forward_draft_blocks(
            raw['input_ids'], raw['hidden_states'], raw['loss_mask'])
        trainer._anchor_observations.append(dict(positions=anchors.tolist(), keep=keep.tolist()))
        loss = model.lm_head(hidden).float().square().mean()
        trainer.response_records.append(dict(record_id=raw['record_id'],
            event_id=trainer._active_event, loss=float(loss.detach()),
            sampled_anchors=deepcopy(trainer._anchor_observations)))
        return StepOutput(loss=loss, metrics={}, ratio_metrics={},
                          sum_metrics={'response_presentations': torch.tensor(1.)})

    trainer.author_strategy = SimpleNamespace(forward_loss=forward)
    trainer.strategy = trainer.author_strategy
    trainer.core = TrainerCore(trainer.strategy, trainer.backend, accumulation_steps=2)
    trainer.core._micro = 1024
    for index, path in enumerate(paths):
        torch.save(dict(record_id=f'response_{index}', input_ids=torch.zeros(1, 6, dtype=torch.long),
            hidden_states=(torch.arange(18).reshape(1, 6, 3).bfloat16() / 10 + index),
            loss_mask=torch.ones(1, 6)), path)
    return trainer, paths


class OrdinaryStartupTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def check_case(self, trainer, paths, root, *, scalar=False):
        baseline = qualify.runtime_state(trainer.bundle.model)
        origin = qualify.scientific_state(trainer)
        expected, observations = manual_gradients(trainer, paths, origin, baseline)
        work = DiagnosticWork(root, diagnostic.BUDGET)

        def restore(_trainer, _baseline, _root):
            restore_fixture(trainer, origin, baseline)

        with mock.patch.object(diagnostic, 'restore_origin', side_effect=restore), \
             mock.patch.object(diagnostic, 'compiler_state', return_value={}), \
             mock.patch.object(trainer.backend.optimizer.optimizer, 'step',
                               side_effect=AssertionError('Adam must not execute')) as adam:
            actual, receipt = diagnostic.capture_window(trainer, paths, 'capture', work,
                baseline, root / 'capture', origin, observations, scalar=scalar)
        self.assertFalse(qualify.differences(actual, expected))
        self.assertFalse(qualify.differences(qualify.scientific_state(trainer), origin))
        self.assertEqual(receipt['status'], 'captured_before_adam')
        self.assertEqual(len(receipt['responses']), 2)
        self.assertEqual(work.counts['response_forward_completed'], 2)
        self.assertEqual(work.counts['response_backward_completed'], 2)
        self.assertNotIn('optimizer_step_attempted', work.counts)
        adam.assert_not_called()
        self.assertTrue(json.loads((root / 'capture/RESTORATION.json').read_text())['exact_equal'])
        return actual, receipt

    def test_plain_capture_equals_manual_unclipped_gradients_and_restores_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, paths = replay_fixture(root)
            gradients, _ = self.check_case(trainer, paths, root)
            # This norm exceeds clipping's limit: equality verifies pre-clip capture.
            self.assertGreater(sum(float(value.float().square().sum())
                for value in gradients.values() if value is not None) ** .5, 1.)

    def test_real_scalar_capture_equals_manual_without_auxiliary_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, paths = scalar_fixture(root)
            original_forward = trainer.bundle.model._forward_draft_blocks
            hook_counts = [len(module._forward_pre_hooks) for module in trainer.bundle.model.modules()]
            _, receipt = self.check_case(trainer, paths, root, scalar=True)
            work = receipt['capture_hook']['work']
            self.assertEqual(work['captured_ordinary_responses'], 2)
            self.assertEqual(work['additional_response_forwards_attempted'], 0)
            self.assertEqual(work['additional_backwards_attempted'], 0)
            self.assertEqual(work['optimizer_steps_attempted'], 0)
            self.assertTrue(trainer.functional_geometry.closed)
            self.assertEqual(trainer.bundle.model._forward_draft_blocks, original_forward)
            self.assertEqual([len(module._forward_pre_hooks) for module in trainer.bundle.model.modules()],
                             hook_counts)

    def test_second_forward_failure_propagates_and_restores_complete_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, paths = scalar_fixture(root)
            baseline = qualify.runtime_state(trainer.bundle.model)
            origin = qualify.scientific_state(trainer)
            _, observations = manual_gradients(trainer, paths, origin, baseline)
            forward = trainer.author_strategy.forward_loss
            calls = 0
            original_forward = trainer.bundle.model._forward_draft_blocks

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ValueError('original forward failure')
                return forward(*args, **kwargs)

            def restore(_trainer, _baseline, _root):
                restore_fixture(trainer, origin, baseline)

            work = DiagnosticWork(root, diagnostic.BUDGET)
            with mock.patch.object(diagnostic, 'restore_origin', side_effect=restore), \
                 mock.patch.object(diagnostic, 'compiler_state', return_value={}), \
                 mock.patch.object(trainer.author_strategy, 'forward_loss', side_effect=fail_second), \
                 self.assertRaisesRegex(ValueError, 'original forward failure'):
                diagnostic.capture_window(trainer, paths, 'failed', work, baseline,
                    root / 'failed', origin, observations, scalar=True)
            self.assertFalse(qualify.differences(qualify.scientific_state(trainer), origin))
            self.assertEqual(trainer.bundle.model._forward_draft_blocks, original_forward)
            self.assertTrue(trainer.functional_geometry.closed)
            self.assertEqual(work.counts['response_forward_failed'], 1)
            self.assertEqual(work.counts['response_backward_completed'], 1)
            receipt = json.loads((root / 'failed/RESULT.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertIn('original forward failure', receipt['error'])
            self.assertFalse(receipt.get('intended_boundary_stop', False))

    def test_original_boundary_tripwire_remains_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, paths = replay_fixture(root)
            baseline = qualify.runtime_state(trainer.bundle.model)
            origin = qualify.scientific_state(trainer)
            _, observations = manual_gradients(trainer, paths, origin, baseline)
            # An erroneous backend bypasses the adapter; the real Adam tripwire must fire.
            trainer.backend.step = lambda: trainer.backend.optimizer.optimizer.step()
            work = DiagnosticWork(root, diagnostic.BUDGET)
            with mock.patch.object(diagnostic, 'restore_origin',
                    side_effect=lambda *_: restore_fixture(trainer, origin, baseline)), \
                 mock.patch.object(diagnostic, 'compiler_state', return_value={}), \
                 self.assertRaisesRegex(RuntimeError, 'Optimizer applications are forbidden'):
                diagnostic.capture_window(trainer, paths, 'tripwire', work, baseline,
                    root / 'tripwire', origin, observations)
            self.assertFalse(qualify.differences(qualify.scientific_state(trainer), origin))
            self.assertEqual(json.loads((root / 'tripwire/RESULT.json').read_text())['status'], 'failed')

    def test_cleanup_failure_keeps_the_completed_capture_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, paths = replay_fixture(root)
            baseline = qualify.runtime_state(trainer.bundle.model)
            origin = qualify.scientific_state(trainer)
            _, observations = manual_gradients(trainer, paths, origin, baseline)
            calls = 0

            def restore(*_):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ValueError('restoration rejected pending metrics')
                restore_fixture(trainer, origin, baseline)

            work = DiagnosticWork(root, diagnostic.BUDGET)
            with mock.patch.object(diagnostic, 'restore_origin', side_effect=restore), \
                 mock.patch.object(diagnostic, 'compiler_state', return_value={}), \
                 self.assertRaisesRegex(ValueError, 'restoration rejected'):
                diagnostic.capture_window(trainer, paths, 'cleanup_failure', work, baseline,
                    root / 'cleanup_failure', origin, observations)
            receipt = json.loads((root / 'cleanup_failure/RESULT.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertTrue(receipt['intended_boundary_stop'])
            self.assertEqual(len(receipt['responses']), 2)
            self.assertEqual(receipt['cleanup_errors'][0]['operation'], 'restore origin')
            self.assertEqual(work.counts['response_backward_completed'], 2)
            self.assertNotIn('optimizer_step_attempted', work.counts)
            restore_fixture(trainer, origin, baseline)


    def test_nonfinite_backward_stops_before_capture_success_and_restores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer, paths = scalar_fixture(root)
            baseline = qualify.runtime_state(trainer.bundle.model)
            origin = qualify.scientific_state(trainer)
            _, observations = manual_gradients(trainer, paths, origin, baseline)
            backward = trainer.backend.backward

            def corrupt_boundary_gradient(loss, **kwargs):
                self.assertTrue(bool(torch.isfinite(loss)))
                backward(loss, **kwargs)
                if kwargs['is_boundary']:
                    trainer.bundle.draft_model.weight.grad.view(-1)[0] = torch.nan

            work = DiagnosticWork(root, diagnostic.BUDGET)
            with mock.patch.object(diagnostic, 'restore_origin',
                    side_effect=lambda *_: restore_fixture(trainer, origin, baseline)), \
                 mock.patch.object(diagnostic, 'compiler_state', return_value={}), \
                 mock.patch.object(trainer.backend, 'backward', side_effect=corrupt_boundary_gradient), \
                 self.assertRaisesRegex(ValueError, 'Nonfinite ordinary gradients'):
                diagnostic.capture_window(trainer, paths, 'nonfinite', work, baseline,
                    root / 'nonfinite', origin, observations, scalar=True)
            self.assertEqual(json.loads((root / 'nonfinite/NONFINITE_GRADIENTS.json').read_text()),
                             {'weight': 1})
            receipt = json.loads((root / 'nonfinite/RESULT.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertFalse(receipt.get('intended_boundary_stop', False))
            self.assertEqual(work.counts['response_backward_completed'], 2)
            self.assertEqual(receipt['capture_hook']['work']['optimizer_steps_attempted'], 0)
            self.assertTrue(trainer.functional_geometry.closed)
            self.assertFalse(qualify.differences(qualify.scientific_state(trainer), origin))


if __name__ == '__main__':
    unittest.main()
