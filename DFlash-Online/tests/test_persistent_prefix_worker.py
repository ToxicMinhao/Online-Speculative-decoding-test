"""The startup observer counts native calls without changing their signature."""
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.persistent_prefix_withholding import worker


class GraphObserverTests(unittest.TestCase):
    def setUp(self):
        worker._INITIALIZATION.clear()
        self.graph = SimpleNamespace(model_runner=SimpleNamespace(is_draft_worker=True),
                                     num_tokens_per_bs=16)

    def observe(self, original):
        self.receipts = []
        return worker._observe_capture(original,
            lambda: self.receipts.append(deepcopy(worker._INITIALIZATION)))

    def test_native_signature_and_dflash_embeds_are_preserved(self):
        calls, embeddings, proxy = [], object(), object()
        def forward(tokens, positions, batch, *, input_embeds=None, pp_proxy_tensors=None):
            calls.append((tokens, positions, batch, input_embeds, pp_proxy_tensors))
            return embeddings
        def native(graph, bs, observed, stream=None, num_tokens_per_bs=None):
            self.assertEqual(inspect.signature(observed), inspect.signature(forward))
            options = {}
            if 'input_embeds' in inspect.signature(observed).parameters:
                options['input_embeds'] = embeddings
            if 'pp_proxy_tensors' in inspect.signature(observed).parameters:
                options['pp_proxy_tensors'] = proxy
            for _ in range(3):
                value = observed('tokens', 'positions', 'batch', **options)
            return value, stream, num_tokens_per_bs
        counted = self.observe(native)
        self.assertEqual(counted(self.graph, 1, forward, 'stream', num_tokens_per_bs=9),
                         (embeddings, 'stream', 9))
        self.assertEqual(calls, [('tokens', 'positions', 'batch', embeddings, proxy)] * 3)
        self.assertEqual(inspect.signature(counted), inspect.signature(native))
        final = self.receipts[-1][0]
        self.assertEqual((final['model_forward_calls'], final['model_forward_completions'],
                          final['model_forward_failures']), (3, 3, 0))
        self.assertEqual(final['status'], 'completed')
        self.assertEqual(final['tokens_per_batch'], 9)

    def test_native_failure_preserves_completed_work_before_rpc(self):
        def native(graph, bs, forward):
            forward()
            forward()
            raise RuntimeError('native graph failed')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'PHYSICAL_WORK.json'
            counted = worker._observe_capture(native, lambda: worker._save_work(path))
            with self.assertRaisesRegex(RuntimeError, 'native graph failed'):
                counted(self.graph, 1, lambda: None)
            row = json.loads(path.read_text())['initialization'][0]
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['model_forward_calls'], 2)
        self.assertEqual(row['model_forward_completions'], 2)
        self.assertIn('native graph failed', row['error'])

    def test_failed_forward_is_counted_separately(self):
        attempts = 0
        def forward():
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise ValueError('model failure')
        def native(graph, bs, observed):
            for _ in range(3):
                observed()
        with self.assertRaisesRegex(ValueError, 'model failure'):
            self.observe(native)(self.graph, 1, forward)
        row = self.receipts[-1][0]
        self.assertEqual((row['model_forward_calls'], row['model_forward_completions'],
                          row['model_forward_failures']), (2, 1, 1))
        self.assertEqual(row['status'], 'failed')

    def test_changed_forward_count_still_fails_after_recording(self):
        def native(graph, bs, forward):
            forward()
            forward()
        with self.assertRaisesRegex(RuntimeError, 'two graph warmup'):
            self.observe(native)(self.graph, 1, lambda: None)
        self.assertEqual(self.receipts[-1][0]['model_forward_calls'], 2)
        self.assertEqual(self.receipts[-1][0]['status'], 'failed')

    def test_out_of_bounds_shape_fails_before_model_call(self):
        calls = []
        def native(*args):
            calls.append(args)
        with self.assertRaisesRegex(RuntimeError, 'finite graph'):
            self.observe(native)(self.graph, 9, lambda: None)
        self.assertFalse(calls)
        self.assertEqual(self.receipts[-1][0]['model_forward_calls'], 0)
        self.assertEqual(self.receipts[-1][0]['status'], 'failed')

    def test_constructor_failure_keeps_child_receipt_and_restores_environment(self):
        def failed_constructor(instance, settings, root):
            path = Path(os.environ[worker._WORK_PATH_ENV])
            def native(graph, bs, forward):
                forward()
                raise RuntimeError('constructor graph failure')
            worker._observe_capture(native, lambda: worker._save_work(path))(
                self.graph, 1, lambda: None)
        with tempfile.TemporaryDirectory() as root, \
             patch.dict(os.environ, {worker._WORK_PATH_ENV: 'previous-value'}), \
             patch.object(worker.sglang_worker.NativeWorker, '__init__', failed_constructor), \
             patch.object(worker.Worker, 'work_receipt') as rpc, \
             patch.object(worker.sglang_hooks, 'scheduler_process'):
            with self.assertRaisesRegex(RuntimeError, 'constructor graph failure'):
                worker.Worker({}, Path(root))
            self.assertEqual(os.environ[worker._WORK_PATH_ENV], 'previous-value')
            rpc.assert_not_called()
            receipt = json.loads((Path(root) / 'PHYSICAL_WORK.json').read_text())
        self.assertEqual(receipt['initialization'][0]['model_forward_completions'], 1)
        self.assertEqual(receipt['initialization'][0]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
