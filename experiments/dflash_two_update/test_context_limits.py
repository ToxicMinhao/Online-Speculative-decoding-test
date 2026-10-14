"""CPU admission checks for the common 32K serving/capture/replay envelope."""
import ast
import copy
import hashlib
import inspect
import os
from pathlib import Path
import sys
import textwrap
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import experiment as E
import native as N
import training as T
from test_native import native_record


def long_record(size):
    doc = native_record()
    doc['input_ids'] = [1] * size
    doc['input_token_count'] = size
    doc['input_ids_sha256'] = N.object_hash(doc['input_ids'])
    doc['cache_namespace']['input_ids_sha256'] = doc['input_ids_sha256']
    doc['cache_namespace_sha256'] = N.object_hash(doc['cache_namespace'])
    doc['context_feature_rows'] = size + 1
    block = doc['blocks'][0]
    block['anchor'] = block['context_length'] = size
    block['positions'] = list(range(size, size + 10))
    block['teacher_prediction_positions'] = list(range(size + 1, size + 11))
    N.add_response_ownership(doc)
    return doc


class ContextLimitTests(unittest.TestCase):
    def test_all_public_limits_match_and_leave_proposal_headroom(self):
        self.assertEqual((N.INPUT_LIMIT, N.CONTEXT_LIMIT), (31744, 32768))
        self.assertEqual((E.INPUT_LIMIT, E.CONTEXT_LIMIT), (N.INPUT_LIMIT, N.CONTEXT_LIMIT))
        self.assertEqual((T.INPUT_LIMIT, T.CONTEXT_LIMIT), (N.INPUT_LIMIT, N.CONTEXT_LIMIT))
        self.assertLessEqual(N.INPUT_LIMIT + N.OUTPUT_CAP + N.BLOCK_SIZE - 1, N.CONTEXT_LIMIT)
        for size in (16791, 25930, N.INPUT_LIMIT):
            ids = [7] * size
            N.check_input(ids)
            self.assertEqual(len(ids), size)
            self.assertEqual(T.audit_response(long_record(size))['eligible_positions'], 1)
        with self.assertRaisesRegex(ValueError, '31744.*truncation'):
            N.check_input([7] * 31745)
        with self.assertRaisesRegex(ValueError, 'truncation forbidden'):
            T.audit_response(long_record(31745))

    def test_common_engine_limits_keep_other_runtime_settings(self):
        original = {'context_length': 17408, 'max_prefill_tokens': 16384,
                    'speculative_draft_model_path': '/draft',
                    'speculative_algorithm': 'DFLASH', 'speculative_num_draft_tokens': 10,
                    'mem_fraction_static': .8}
        native = SimpleNamespace(engine_kwargs=lambda _: copy.deepcopy(original))
        sd = N.serving_engine_kwargs(native, '/draft', is_ar=False)
        ar = N.serving_engine_kwargs(native, '/draft', is_ar=True)
        for value in (sd, ar):
            self.assertEqual(value['context_length'], 32768)
            self.assertEqual(value['chunked_prefill_size'], 8192)
            self.assertEqual(value['max_prefill_tokens'], 16384)
            self.assertEqual(value['mem_fraction_static'], .8)
        self.assertEqual(ar, {k: v for k, v in sd.items() if not k.startswith('speculative_')})
        self.assertEqual(original['context_length'], 17408)

    def test_capture_allowance_includes_serialization_overlap(self):
        context = (N.INPUT_LIMIT + N.OUTPUT_CAP - 1) * 20480 * 2
        teacher = (N.OUTPUT_CAP - 1) * 10 * 4096 * 2
        self.assertGreater(N.capture_memory_bound(), 2 * context + teacher)
        self.assertGreater(N.capture_memory_bound(), 2 * 2**30)
        self.assertLess(N.capture_memory_bound(), N.MAX_CAPTURE_BYTES)
        self.assertEqual(N.MAX_CAPTURE_BYTES, 4 * 2**30)
        self.assertEqual(N.episode_storage_bound(2097259104), 34928840256)
        self.assertGreater(N.episode_storage_bound(2097259104), 32 * 2**30)

    def test_controller_rejects_over_limit_before_backend_without_clipping(self):
        backend = mock.Mock()
        with self.assertRaisesRegex(ValueError, 'truncation prohibited'):
            E.call(backend, [7] * 31745, 'F', capture=False, turn=27)
        backend.serve.assert_not_called()

    def test_old_capture_budget_is_refused_before_worker_or_output_creation(self):
        backend = N.NativeBackend({'workspace_root': '/unused',
            'serving_cuda_visible_devices': '0', 'training_cuda_visible_devices': '1',
            'max_capture_bytes': 2 * 2**30})
        with mock.patch.object(N, 'check_environment'), mock.patch.object(N, 'check_dependencies'), \
                mock.patch.object(backend, '_ensure_worker') as start:
            with self.assertRaisesRegex(ValueError, 'serialization overlap'):
                backend.start_episode('R1_STABLE_CODE')
        start.assert_not_called()
        self.assertIsNone(backend.episode_id)


@unittest.skipUnless(os.environ.get('TWO_UPDATE_WORKSPACE'),
                     'Installed-source context checks need TWO_UPDATE_WORKSPACE')
class InstalledContextLimitTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ['TWO_UPDATE_WORKSPACE'])
        self.saved = {name: sys.modules.get(name) for name in ('records', 'cache', 'two_update_w8_replay')}

    def tearDown(self):
        for name, original in self.saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def test_real_capture_validator_accepts_long_prefix_and_preserves_ownership(self):
        records, cache = N.capture_modules(self.root)
        self.assertEqual(records.CONTEXT_LIMIT, 32768)
        self.assertEqual(cache.CONTEXT_LIMIT, 32768)
        for size in (16791, 25930, 31744):
            doc = long_record(size)
            records.validate(doc)
            self.assertEqual(doc['blocks'][0]['owned_eligible_slots'], [1])
            self.assertEqual(T.audit_response(doc)['eligible_positions'], 1)
        with self.assertRaisesRegex(ValueError, 'truncation'):
            records.validate(long_record(31745))

    def test_replay_default_and_explicit_limit_match_training_and_serving(self):
        losses = ModuleType('losses'); losses.require = N.require
        with mock.patch.dict(sys.modules, {'torch': ModuleType('torch'), 'losses': losses}):
            replay = N.replay_module(self.root)
        self.assertEqual(replay.SERVING_CONTEXT, 32768)
        self.assertEqual(inspect.signature(replay.NativeReplay).parameters['context_limit'].default, 32768)
        draft = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type='cpu'))]))
        for keywords in ({}, {'context_limit': 32768}):
            instance = replay.NativeReplay(draft, {}, None, identities={}, **keywords)
            self.assertEqual(instance.context_limit, 32768)
            with self.assertRaisesRegex(ValueError, 'native32768; truncation'):
                instance.replay_hidden({'context_length': 32759, 'anchor': 32759})
        with self.assertRaises(ValueError):
            replay.NativeReplay(draft, {}, None, identities={}, context_limit=17408)
        self.assertEqual(len(replay.two_update_compatibility['edits']), 2)

    def test_actual_shared_bridge_guard_handles_sd_and_ar_without_mutating_source(self):
        path = self.root / 'research/multiturn_online_setting_20261008/online_driver_d181/serving.py'
        before_hash = N.sha(path)
        source = path.read_text()
        tree = ast.parse(source)
        bridge = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Bridge')
        generate = next(node for node in bridge.body if isinstance(node, ast.FunctionDef) and node.name == 'generate')
        namespace = {'dependencies': lambda: (SimpleNamespace(guard=lambda: None), None),
                     'require': N.require, 'time': time, 'os': os}
        exec(compile(ast.Module(body=[generate], type_ignores=[]), str(path), 'exec'), namespace)
        base = type('Base', (), {'generate': namespace['generate']})
        sd, ar = type('SD', (base,), {}), type('AR', (base,), {})

        def exact_edit(function, before, after, globals_):
            original = textwrap.dedent(inspect.getsource(function))
            self.assertEqual(original.count(before), 1)
            updated = original.replace(before, after)
            local = {}; exec(compile(updated, function.__code__.co_filename, 'exec'), globals_, local)
            return local[function.__name__], {'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
                                             'effective_sha256': hashlib.sha256(updated.encode()).hexdigest()}

        entry = SimpleNamespace(_original_bridge=base, inherited=SimpleNamespace(**namespace),
                                exact_guard_edit=exact_edit)
        receipt = N.install_context_guard(entry)
        self.assertIs(N.install_context_guard(entry), receipt)
        self.assertNotEqual(receipt['original_sha256'], receipt['effective_sha256'])
        sampling = {'temperature': 0, 'top_p': 1, 'top_k': 1,
                    'max_new_tokens': 512, 'ignore_eos': False}
        for cls in (sd, ar):
            instance = cls(); instance.blocked = None; instance.capture = False
            instance.engine = SimpleNamespace(generate=mock.Mock(return_value={'output_ids': [123]}))
            for size in (16791, 25930, 31744):
                ids = [7] * size
                instance.generate({'input_ids': ids, 'sampling_params': sampling, 'stream': False})
                self.assertIs(instance.engine.generate.call_args.kwargs['input_ids'], ids)
            with self.assertRaisesRegex(ValueError, 'at most31744'):
                instance.generate({'input_ids': [7] * 31745, 'sampling_params': sampling})
            self.assertEqual(instance.engine.generate.call_count, 3)
        self.assertEqual(N.sha(path), before_hash)


if __name__ == '__main__':
    unittest.main()
