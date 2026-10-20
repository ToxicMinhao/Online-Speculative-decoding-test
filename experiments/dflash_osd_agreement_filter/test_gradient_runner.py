"""Local fakes test diagnostic cleanup/proxy boundaries without loading models."""
import importlib
import sys
import types
import unittest
from unittest.mock import patch

import torch

from . import gradient_diagnostic as runner


class FakeTrainer:
    def __init__(self, failed):
        self.failed = failed
        self.state = {'weights': torch.tensor([1., 2.]), 'optimizer_steps': 0}
        self.snapshots = []
        self.raise_snapshot = False

    def snapshot(self):
        self.snapshots.append(self.failed)
        if self.raise_snapshot:
            raise RuntimeError('injected snapshot failure')
        if self.failed:
            raise AssertionError('Poisoned trainer cannot snapshot')
        return self.state


class RestorationTests(unittest.TestCase):
    def test_failed_flag_remains_poisoned_after_successful_identity_check(self):
        trainer = FakeTrainer(True)
        expected = runner.training.state_digest(trainer.state)
        result = runner.verify_restored(trainer, expected)
        self.assertTrue(trainer.failed)
        self.assertEqual(trainer.snapshots, [False])
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['failed_flag_preserved'])
        self.assertEqual(result['expected'], result['observed'])

    def test_initially_unpoisoned_flag_stays_false(self):
        trainer = FakeTrainer(False)
        result = runner.verify_restored(trainer, runner.training.state_digest(trainer.state))
        self.assertFalse(trainer.failed)
        self.assertTrue(result['failed_flag_preserved'])

    def test_mismatched_state_does_not_unpoison_failed_trainer(self):
        trainer = FakeTrainer(True)
        with self.assertRaisesRegex(ValueError, 'restore exact initial state'):
            runner.verify_restored(trainer, '0' * 64)
        self.assertTrue(trainer.failed)

    def test_snapshot_exception_still_restores_original_failure_flag(self):
        trainer = FakeTrainer(True)
        trainer.raise_snapshot = True
        with self.assertRaisesRegex(RuntimeError, 'injected snapshot'):
            runner.verify_restored(trainer, '0' * 64)
        self.assertTrue(trainer.failed)

    def test_hash_exception_still_restores_original_failure_flag(self):
        trainer = FakeTrainer(True)
        with patch.object(runner.training, 'state_digest', side_effect=RuntimeError('injected digest')):
            with self.assertRaisesRegex(RuntimeError, 'injected digest'):
                runner.verify_restored(trainer, '0' * 64)
        self.assertTrue(trainer.failed)


class EagerProxyTests(unittest.TestCase):
    def fixtures(self):
        transformers = types.ModuleType('transformers'); transformers.__path__ = []
        integrations = types.ModuleType('transformers.integrations'); integrations.__path__ = []
        module = types.ModuleType('transformers.integrations.flex_attention')
        original = lambda *args, **kwargs: ('original', args, kwargs)
        module.compile_friendly_flex_attention = original
        transformers.integrations = integrations
        integrations.flex_attention = module
        modules = {'transformers': transformers, 'transformers.integrations': integrations,
                   'transformers.integrations.flex_attention': module}
        native = importlib.import_module('torch.nn.attention.flex_attention')
        return modules, module, original, native

    def test_same_tensors_and_flex_keywords_are_forwarded_and_proxy_restores(self):
        modules, module, original, native = self.fixtures()
        q, k, v = [torch.zeros((1, 2, 3, 4)) for _ in range(3)]
        block_mask, score_mod = object(), object()
        returned = object()
        with patch.dict(sys.modules, modules), patch.object(native, 'flex_attention', return_value=returned) as target:
            with runner.eager_flex_backend() as calls:
                value = module.compile_friendly_flex_attention(q, k, v, block_mask=block_mask,
                    score_mod=score_mod, scale=.5, enable_gqa=True, return_lse=False)
                self.assertIs(value, returned)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]['q_shape'], list(q.shape))
                self.assertEqual(calls[0]['k_shape'], list(k.shape))
                self.assertEqual(calls[0]['dtype'], str(q.dtype))
                self.assertIsNot(module.compile_friendly_flex_attention, original)
            self.assertIs(module.compile_friendly_flex_attention, original)
        args, kwargs = target.call_args
        self.assertIs(args[0], q); self.assertIs(args[1], k); self.assertIs(args[2], v)
        self.assertIs(kwargs['block_mask'], block_mask)
        self.assertIs(kwargs['score_mod'], score_mod)
        self.assertEqual(kwargs['scale'], .5)
        self.assertTrue(kwargs['enable_gqa'])
        self.assertFalse(kwargs['return_lse'])

    def test_training_wrapper_only_kwarg_is_removed_without_changing_native_kwargs(self):
        modules, module, original, native = self.fixtures()
        q, k, v = [torch.zeros((1, 1, 2, 4)) for _ in range(3)]
        block_mask = object()
        with patch.dict(sys.modules, modules), patch.object(native, 'flex_attention', return_value='ok') as target:
            with runner.eager_flex_backend():
                result = module.compile_friendly_flex_attention(q, k, v, block_mask=block_mask,
                                                                training=True, enable_gqa=False)
            self.assertEqual(result, 'ok')
            self.assertIs(module.compile_friendly_flex_attention, original)
        self.assertNotIn('training', target.call_args.kwargs)
        self.assertIs(target.call_args.kwargs['block_mask'], block_mask)
        self.assertFalse(target.call_args.kwargs['enable_gqa'])

    def test_proxy_restores_after_backward_or_forward_exception(self):
        modules, module, original, native = self.fixtures()
        q, k, v = [torch.zeros((1, 1, 2, 4)) for _ in range(3)]
        with patch.dict(sys.modules, modules), patch.object(native, 'flex_attention', side_effect=RuntimeError('injected native')):
            with self.assertRaisesRegex(RuntimeError, 'injected native'):
                with runner.eager_flex_backend() as calls:
                    module.compile_friendly_flex_attention(q, k, v, block_mask=None)
            self.assertIs(module.compile_friendly_flex_attention, original)
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
