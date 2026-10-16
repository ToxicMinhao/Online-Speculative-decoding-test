"""Focused independent-branch/exposure/native-accounting checks (CPU only)."""
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from . import native
from experiments.dflash_per_turn import test_training as fixtures


class Contract(unittest.TestCase):
    def test_budget_accounts_two_weights_and_staging_not_diagnostic_fleet(self):
        self.assertEqual(native.storage_bound(100), 300 + 512 * 2**20)
        for value in (0, -1, True, 3.5):
            with self.assertRaises(ValueError):
                native.storage_bound(value)
        self.assertEqual(native.base.OUTPUT_CAP, 1024)
        self.assertEqual(native.base.BLOCK_SIZE, 10)
        self.assertEqual(native.base.PREFILL_CHUNK_SIZE, 8192)

    def test_capture_and_out_of_scope_replays_rejected(self):
        backend = native.NativeBackend({})
        for label, turn, capture in (('D2', 2, False), ('F', 3, False), ('F', 1, True)):
            with self.assertRaises(ValueError):
                backend.serve([1, 2], label, capture, turn_index=turn, repeat=0)

    def test_replay_preserves_inference_only_timer_and_observed_tokens(self):
        backend = native.NativeBackend({})
        backend.checkpoints = {'D1': {}}
        backend._ensure_worker = lambda kind: 0.
        backend.publish = lambda _: {'activation_s': 7., 'native_activation_s': 2.}
        calls = []
        def call(action, body):
            calls.append((action, body))
            return ({'seconds': 3.} if action == 'flush' else
                    {'input_ids': [1, 2], 'input_ids_sha256': native.digest([1, 2]), 'output_ids': [3]})
        backend.worker = types.SimpleNamespace(call=call)
        with patch.object(native.base.time, 'perf_counter', side_effect=[30., 35.]):
            result = backend.serve([1, 2], 'D1', turn_index=2, repeat=0)
        self.assertEqual(result['request_s'], 5.)
        self.assertEqual(result['activation_s'], 7.)
        self.assertEqual(result['cache_maintenance_s'], 3.)
        self.assertEqual(calls[1][1]['input_ids'], [1, 2])
        self.assertFalse(calls[1][1]['capture'])

    def test_new_worker_for_each_branch_and_no_optimizer_persistence(self):
        created = []
        class Worker:
            def __init__(self, config, kind, out):
                self.config, self.out, self.startup_s = config, out, .1
                self.closed = False
                created.append(self)
            def call(self, action, body):
                label = self.config['first_update_branch']['checkpoint_id']
                return {'checkpoint': {'checkpoint_id': label}, 'accounting': {}}
            def close(self):
                self.closed = True
        with tempfile.TemporaryDirectory() as directory:
            backend = native.NativeBackend({})
            backend.episode_id = 'episode'; backend.out = Path(directory); backend.branches = {}
            backend.checkpoints = {'F': {'checkpoint_identity_sha256': 'f'}}
            support = {'turn': 1, 'capture_checkpoint': 'F', 'capture_checkpoint_sha256': 'f'}
            with patch.object(native.base, '_WorkerClient', Worker), \
                 patch.object(native.base, 'verify_checkpoint'), patch.object(native, 'verify_exposure'):
                backend.train_branch('D1', 3e-4, support, {'accounting': {}})
                backend.train_branch('L1', 1e-4, support, {'accounting': {}})
                with self.assertRaises(ValueError):
                    backend.train_branch('L1', 1e-4, support, {'accounting': {}})
            self.assertEqual(len(created), 2)
            self.assertTrue(all(worker.closed for worker in created))
            self.assertIsNone(backend.trainer)
            self.assertNotEqual(created[0].out, created[1].out)
            self.assertEqual([w.config['first_update_branch']['learning_rate'] for w in created], [3e-4, 1e-4])

    def test_failed_branch_closes_worker_and_does_not_register_weights(self):
        class Worker:
            startup_s = .1
            closed = False
            def call(self, *args):
                raise ValueError('injected exposure mismatch')
            def close(self):
                self.closed = True
        worker = Worker()
        with tempfile.TemporaryDirectory() as directory:
            backend = native.NativeBackend({})
            backend.episode_id = 'episode'; backend.out = Path(directory); backend.branches = {}
            backend.checkpoints = {'F': {'checkpoint_identity_sha256': 'f'}}
            support = {'turn': 1, 'capture_checkpoint': 'F', 'capture_checkpoint_sha256': 'f'}
            with patch.object(native.base, '_WorkerClient', return_value=worker):
                with self.assertRaisesRegex(ValueError, 'injected'):
                    backend.train_branch('D1', 3e-4, support, {'accounting': {}})
            self.assertTrue(worker.closed)
            self.assertIsNone(backend.trainer)
            self.assertNotIn('D1', backend.checkpoints)


@unittest.skipUnless(fixtures.HAS_TORCH, 'CPU torch required')
class Exposure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        first = fixtures.Numerical().trainer()
        cls.original = first.train_event(1, [fixtures.response(1)])

    def test_two_fresh_adams_have_identical_anchors_but_rate_dependent_weights(self):
        import torch
        high, low = fixtures.Numerical().trainer(), fixtures.Numerical().trainer()
        self.assertFalse(high.optimizer.state)
        self.assertFalse(low.optimizer.state)
        self.assertIsNot(high.optimizer, low.optimizer)
        low.learning_rate = 1e-4; low.optimizer.param_groups[0]['lr'] = 1e-4
        a = high.train_event(1, [fixtures.response(1)])
        b = low.train_event(1, [fixtures.response(1)])
        for accounting in (a, b):
            self.assertEqual(native.verify_exposure(accounting, self.original)['status'], 'passed')
        self.assertTrue(any(not torch.equal(value, low.draft.state_dict()[name])
                            for name, value in high.draft.state_dict().items()))
        self.assertEqual({int(state['step']) for state in high.optimizer.state.values()}, {2})
        self.assertEqual({int(state['step']) for state in low.optimizer.state.values()}, {2})

    def test_changed_anchor_rejected_even_when_old_summary_hash_copied(self):
        altered = copy.deepcopy(self.original)
        altered['batches'][0]['presentations'][0]['anchor_positions'][0] += 1
        with self.assertRaisesRegex(ValueError, 'anchors or supervised exposure'):
            native.verify_exposure(altered, self.original)

    def test_changed_supervised_position_or_weight_rejected(self):
        for field in ('label_positions', 'position_weights', 'binary_loss_mask'):
            altered = copy.deepcopy(self.original)
            altered['batches'][0]['presentations'][0][field][0][1] += 1
            with self.assertRaisesRegex(ValueError, 'anchors or supervised exposure'):
                native.verify_exposure(altered, self.original)

    def test_losses_and_timing_may_change_but_recipe_or_steps_cannot(self):
        altered = copy.deepcopy(self.original)
        altered['recipe']['learning_rate'] = 1e-4
        altered['batches'][0]['loss'] += 1
        altered['batches'][0]['presentations'][0]['loss'] += 1
        self.assertEqual(native.verify_exposure(altered, self.original)['status'], 'passed')
        for field in ('teacher_temperature', 'loss_decay_gamma'):
            wrong = copy.deepcopy(altered); wrong['recipe'][field] += 1
            with self.assertRaises(ValueError):
                native.verify_exposure(wrong, self.original)
        wrong = copy.deepcopy(altered); wrong['optimizer_steps_total'] = 3
        with self.assertRaises(ValueError):
            native.verify_exposure(wrong, self.original)


if __name__ == '__main__':
    unittest.main()
