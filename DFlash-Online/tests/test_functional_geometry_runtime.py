"""CPU wiring checks; native full-size equivalence is a separate qualification."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from online_feedback import functional_geometry
from online_feedback.functional_geometry import (
    install_functional_geometry, isolated_geometry_runtime,
)
from online_feedback.optimizer import DirectBF16AdamW
from online_feedback.rng import capture_rng, restore_rng


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal_tree(a, b) for a, b in zip(left, right))
    if hasattr(left, 'dtype') and hasattr(left, 'shape'):
        return bool((left == right).all())
    return left == right


class TinyBlocks(torch.nn.Module):
    """BF16 projection with cross-position coupling and real training RNG use."""
    block_size = 3
    objective_chunk_blocks = 1
    eos_token_ids = (4,)

    def __init__(self):
        super().__init__()
        self.draft_model = torch.nn.Linear(3, 3, bias=False).bfloat16()
        self.draft_model.register_parameter('inactive', torch.nn.Parameter(torch.ones(2).bfloat16()))
        self.draft_model.weight.data.copy_(torch.eye(3).bfloat16())
        self.lm_head = torch.nn.Linear(3, 5, bias=False).bfloat16().requires_grad_(False)
        self.lm_head.weight.data.copy_(torch.tensor([
            [2., 0., 0.], [0., 2., 0.], [0., 0., 2.], [-1., 0., 0.], [0., -1., 0.],
        ]).bfloat16())
        self.dropout = torch.nn.Dropout(0.1)
        self.first_error_redistribution = object()
        self.register_buffer('replay_count', torch.tensor(0), persistent=False)
        self._cached_seq_len = None
        self._cached_block_mask = None
        self._cached_bsz = None

    def _sample_anchor_positions(self, *args, **kwargs):
        # Consume ordinary RNG, including when both anchors fit the response.
        anchors = torch.randperm(2).reshape(1, 2) * 3
        return anchors, torch.ones_like(anchors, dtype=torch.bool)

    def _forward_draft_blocks(self, input_ids, hidden_states, loss_mask, *args):
        anchors, keep = self._sample_anchor_positions()
        indices = anchors[..., None] + torch.arange(self.block_size)
        features = hidden_states[:, indices[0]].reshape(1, -1, 3)
        coupled = features + features.mean(1, keepdim=True) * .1
        self.replay_count.add_(1)
        self._cached_seq_len = input_ids.shape[1]
        return anchors, keep, self.draft_model(self.dropout(coupled))

    def _block_labels_and_weights(self, input_ids, loss_mask, anchors, keep):
        indices = anchors[..., None] + torch.arange(self.block_size)
        labels = input_ids[:, indices[0]]
        weights = loss_mask[:, indices[0]].float()
        weights[..., 0] = 0
        return labels, weights, indices


def tiny_trainer():
    model = TinyBlocks()
    optimizer = DirectBF16AdamW(model.draft_model, learning_rate=.03)
    return SimpleNamespace(objective='osd_first_error', reduction='response_mean',
        accumulation=2, presentations=0, global_step=0,
        bundle=SimpleNamespace(model=model, draft_model=model.draft_model),
        backend=SimpleNamespace(optimizer=optimizer))


def run_window(trainer, *, change=1.):
    model = trainer.bundle.model
    for scale in (1., .7):
        tokens = torch.zeros((1, 6), dtype=torch.long)
        features = torch.tensor([[[2., .5, .1]] * 6], dtype=torch.bfloat16) * scale
        mask = torch.ones_like(tokens).float()
        _, _, hidden = model._forward_draft_blocks(tokens, features, mask)
        logits = model.lm_head(hidden).float()
        # This fixture tests hooks around an unchanged ordinary scalar loss.
        loss = -logits.log_softmax(-1)[..., 1].mean() * change / 2
        loss.backward()
        if trainer.presentations % 2:
            trainer.backend.optimizer.step()
            trainer.global_step += 1
        trainer.presentations += 1


def state(trainer):
    return dict(weights=deepcopy(trainer.bundle.draft_model.state_dict()),
                optimizer=deepcopy(trainer.backend.optimizer.state_dict()),
                rng=capture_rng(), modes=[m.training for m in trainer.bundle.model.modules()],
                buffers={name: value.clone() for name, value in trainer.bundle.model.named_buffers()},
                cache=trainer.bundle.model._cached_seq_len,
                grads=[p.grad for p in trainer.backend.optimizer.model_params],
                steps=trainer.global_step, presentations=trainer.presentations)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_traced_lambda_zero_matches_plain_step_weights_adam_rng_and_buffers(self):
        ordinary = tiny_trainer()
        traced = tiny_trainer()
        traced.bundle.model.load_state_dict(ordinary.bundle.model.state_dict())
        rng = capture_rng()
        run_window(ordinary)
        reference = state(ordinary)
        restore_rng(rng)
        hook = install_functional_geometry(traced, strength=0.)
        parameter_ids = [id(p) for p in traced.backend.optimizer.model_params]
        run_window(traced)
        self.assertTrue(equal_tree(reference, state(traced)))
        self.assertEqual(parameter_ids, [id(p) for p in traced.backend.optimizer.model_params])
        self.assertEqual(hook.steps[0]['decision']['reason'], 'lambda_zero')
        self.assertEqual(hook.work['additional_response_forwards_completed'], 4)
        self.assertEqual(hook.work['additional_backwards_completed'], 2)
        self.assertFalse(hook.responses)
        self.assertFalse(hook.theta0)
        hook.close()

    def test_enabled_filter_preserves_native_moments_and_counters(self):
        ordinary = tiny_trainer()
        filtered = tiny_trainer()
        filtered.bundle.model.load_state_dict(ordinary.bundle.model.state_dict())
        rng = capture_rng()
        run_window(ordinary)
        native = state(ordinary)
        observed = []
        def observer(phase, hook):
            if phase == 'native':
                observed.append(state(hook.trainer))
        restore_rng(rng)
        hook = install_functional_geometry(filtered, observer=observer)
        run_window(filtered)
        self.assertTrue(equal_tree(native['weights'], observed[0]['weights']))
        self.assertTrue(equal_tree(native['optimizer'], state(filtered)['optimizer']))
        self.assertTrue(equal_tree(native['rng'], state(filtered)['rng']))
        self.assertTrue(equal_tree(native['buffers'], state(filtered)['buffers']))
        self.assertEqual(filtered.global_step, 1)
        self.assertEqual(hook.steps[0]['active_tensors'], 1)
        self.assertTrue(torch.equal(filtered.bundle.draft_model.inactive, torch.ones(2).bfloat16()))
        self.assertLessEqual(hook.work['additional_response_forwards_completed'], 6)
        self.assertLessEqual(hook.work['additional_backwards_completed'], 2)
        self.assertTrue(all(p.grad is None for p in filtered.backend.optimizer.model_params))
        hook.close()

    def test_consecutive_windows_refresh_captures_and_advance_one_optimizer(self):
        trainer = tiny_trainer()
        hook = install_functional_geometry(trainer, arm='SCALAR')
        run_window(trainer)
        run_window(trainer)
        self.assertEqual(trainer.global_step, 2)
        self.assertEqual([row['global_step'] for row in hook.steps], [1, 2])
        self.assertEqual(hook.work['captured_ordinary_responses'], 4)
        self.assertFalse(hook.responses)
        adam = trainer.backend.optimizer.state_dict()['optimizer_state_dict']['state']
        self.assertEqual([float(row['step']) for row in adam.values()], [2.])
        self.assertLessEqual(hook.work['additional_response_forwards_completed'], 12)
        self.assertLessEqual(hook.work['additional_backwards_completed'], 4)
        hook.clear_event()
        self.assertFalse(hook.steps)
        self.assertEqual(trainer.global_step, 2)
        hook.close()

    def test_isolation_restores_registered_buffer_identity_contents_and_new_cache(self):
        trainer = tiny_trainer()
        model = trainer.bundle.model
        original = model.replay_count
        rng = capture_rng()
        with isolated_geometry_runtime(model):
            self.assertFalse(model.training)
            model.replay_count.add_(10)
            model.register_buffer('new_buffer', torch.ones(2), persistent=False)
            model._cached_seq_len = 999
            model.max_seq_len_cached = 999
            torch.rand(3)
        self.assertIs(original, model.replay_count)
        self.assertEqual(int(model.replay_count), 0)
        self.assertFalse(hasattr(model, 'new_buffer'))
        self.assertFalse(hasattr(model, 'max_seq_len_cached'))
        self.assertIsNone(model._cached_seq_len)
        self.assertTrue(model.training)
        self.assertTrue(equal_tree(rng, capture_rng()))

    def test_zero_native_displacement_bypasses_without_auxiliary_model_work(self):
        trainer = tiny_trainer()
        trainer.backend.optimizer.optimizer.param_groups[0]['lr'] = 0.
        hook = install_functional_geometry(trainer)
        run_window(trainer)
        self.assertEqual(hook.steps[0]['decision']['reason'], 'zero_native_displacement')
        self.assertEqual(hook.work['additional_response_forwards_attempted'], 0)
        self.assertEqual(hook.work['additional_backwards_attempted'], 0)
        self.assertEqual(trainer.global_step, 1)
        hook.close()

    def test_empty_protection_keeps_native_endpoint_without_vjp(self):
        trainer = tiny_trainer()
        trainer.bundle.model.eos_token_ids = (0,)
        hook = install_functional_geometry(trainer)
        run_window(trainer)
        self.assertEqual(hook.steps[0]['decision']['reason'], 'empty_sketch')
        self.assertEqual(hook.work['additional_response_forwards_completed'], 4)
        self.assertEqual(hook.work['additional_backwards_attempted'], 0)
        self.assertEqual([row['protected_occurrences'] for row in hook.steps[0]['responses']], [0, 0])
        self.assertTrue(hook.steps[0]['committed']['identical_native'])
        hook.close()

    def test_normalized_chunked_proxy_matches_small_unchunked_native_vjp(self):
        trainer = tiny_trainer()
        receipts = []
        def observer(phase, hook):
            if phase != 'native':
                return
            with isolated_geometry_runtime(hook.model):
                response = hook.responses[0]
                native_hidden = hook.forward_hidden(response, 'oracle_native', grad=False)
                hook.restore_endpoint(hook.theta0)
                chunked = hook.build_reference(response, native_hidden, hook.active_parameters)
                full = hook.build_reference(response, native_hidden, hook.active_parameters, chunk_blocks=0)
                receipts.append((chunked, full))
                hook.restore_endpoint(hook.theta_native)
        hook = install_functional_geometry(trainer, strength=0., observer=observer)
        run_window(trainer)
        chunked, full = receipts[0]
        self.assertTrue(torch.equal(chunked.weights, full.weights))
        self.assertTrue(torch.equal(chunked.proposals, full.proposals))
        self.assertAlmostEqual(chunked.c, full.c, places=8)
        for actual, expected in zip(chunked.gradients, full.gradients):
            if actual is not None:
                torch.testing.assert_close(actual, expected, rtol=.02, atol=2e-4)
        self.assertTrue(all(p.grad is None for p in trainer.backend.optimizer.model_params))
        hook.close()

    def test_failed_auxiliary_work_is_counted_and_restores_native_endpoint(self):
        trainer = tiny_trainer()
        native = []
        events = []
        def observe(phase, hook):
            if phase == 'native':
                native.extend(p.detach().clone() for p in hook.parameters)
        def work(kind, status):
            events.append((kind, status))
        hook = install_functional_geometry(trainer, observer=observe, work_callback=work)
        original = hook.original_forward
        def faulty(*args, **kwargs):
            if hook.phase == 'prestate':
                raise RuntimeError('fixture replay failure')
            return original(*args, **kwargs)
        hook.original_forward = faulty
        with self.assertRaisesRegex(RuntimeError, 'fixture replay failure'):
            run_window(trainer)
        self.assertEqual(hook.steps[0]['status'], 'failed')
        self.assertEqual(hook.work['additional_response_forwards_attempted'], 3)
        self.assertEqual(hook.work['additional_response_forwards_failed'], 1)
        self.assertIn(('additional_response_forwards', 'failed'), events)
        self.assertTrue(all(torch.equal(p, saved) for p, saved in zip(hook.parameters, native)))
        self.assertFalse(hook.responses)
        hook.close()

    def test_failed_step_reporting_keeps_the_original_model_error(self):
        trainer = tiny_trainer()
        with tempfile.TemporaryDirectory() as temporary:
            hook = install_functional_geometry(trainer, output_dir=temporary)
            original = hook.original_forward
            def faulty(*args, **kwargs):
                if hook.phase == 'prestate':
                    raise ValueError('original auxiliary failure')
                return original(*args, **kwargs)
            hook.original_forward = faulty
            write = functional_geometry.write_json
            def failing_report(path, value):
                if Path(path).name.startswith('step_'):
                    raise OSError('secondary report failure')
                return write(path, value)
            with mock.patch.object(functional_geometry, 'write_json', side_effect=failing_report):
                with self.assertRaisesRegex(ValueError, 'original auxiliary failure'):
                    run_window(trainer)
            self.assertIn('secondary report failure', hook.steps[0]['diagnostic_errors'][0])
            self.assertFalse(hook.responses)
            hook.close()


if __name__ == '__main__':
    unittest.main()
