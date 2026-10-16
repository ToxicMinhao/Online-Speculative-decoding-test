"""Native-controller accounting and isolation tests; no GPU/SSH/model loading."""
import importlib.util
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    from . import native
    from . import test_training as fixtures
except ImportError:
    import native
    import test_training as fixtures


class Controller(unittest.TestCase):
    def test_outer_activation_includes_validation_and_publish(self):
        backend = native.NativeBackend({})
        checkpoint = {'checkpoint_id': 'D1'}
        backend.checkpoints = {'D1': checkpoint}; backend.active_checkpoint = 'F'
        backend._ensure_worker = lambda kind: 0.
        backend.worker = types.SimpleNamespace(call=lambda action, body: {'status': 'passed', 'activation_s': 2.})
        with patch.object(native.base.time, 'perf_counter', side_effect=[10., 17.]):
            result = backend.publish('D1')
        self.assertEqual(result['activation_s'], 7.)
        self.assertEqual(result['native_activation_s'], 2.)
        self.assertEqual(backend.totals['activation_s'], 7.)
        again = backend.publish('D1')
        self.assertEqual(again['activation_s'], 0.)
        self.assertEqual(backend.totals['activation_s'], 7.)

    def test_inference_timer_excludes_activation_and_cache_costs(self):
        backend = native.NativeBackend({})
        backend.checkpoints = {'D1': {}}
        backend._ensure_worker = lambda kind: 0.
        backend.publish = lambda label: {'activation_s': 7., 'native_activation_s': 2.}
        seen = []
        def call(action, body):
            seen.append((action, body))
            if action == 'flush':
                return {'seconds': 3.}
            return {'input_ids': [1, 2], 'input_ids_sha256': native.digest([1, 2]), 'output_ids': [3], 'status': 'passed'}
        backend.worker = types.SimpleNamespace(call=call)
        with patch.object(native.base.time, 'perf_counter', side_effect=[30., 35.]):
            result = backend.serve([1, 2], 'D1', turn_index=2, repeat=0)
        self.assertEqual(result['request_s'], 5.)
        self.assertEqual(result['activation_s'], 7.)
        self.assertEqual(result['cache_maintenance_s'], 3.)
        self.assertFalse(seen[1][1]['capture'])
        self.assertEqual(seen[1][1]['input_ids'], [1, 2])
        with self.assertRaises(ValueError):
            backend.serve([1, 2], 'D1', True, turn_index=2)

    def test_wrong_observed_input_rejected(self):
        backend = native.NativeBackend({})
        with patch.object(native.base.NativeBackend, 'serve', return_value={
            'input_ids': [1, 3], 'input_ids_sha256': native.digest([1, 2]), 'output_ids': [4]}):
            with self.assertRaisesRegex(ValueError, 'observation differs'):
                backend.serve([1, 2], 'F', turn_index=1)

    def test_physical_trace_alias_does_not_erase_logical_identity(self):
        service = native.ServingService.__new__(native.ServingService)
        service.logical_checkpoint = 'D23'
        def generate(obj, body):
            self.assertEqual(body['checkpoint_id'], 'U1')
            return {'input_ids_sha256': native.digest(body['input_ids']), 'output_ids': [3], 'trajectory': bound}
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory)/'trace.json'
            trace.write_text('{"input_token_ids": [1, 2]}')
            bound = native.base.binding(trace)
            with patch.object(native.base._ServingService, 'generate', generate):
                result = service.generate({'capture': False, 'checkpoint_id': 'D23', 'input_ids': [1, 2]})
        self.assertEqual(result['logical_checkpoint_id'], 'D23')
        self.assertEqual(result['input_ids'], [1, 2])

    def test_source_pins_and_budget_cover_new_bridge(self):
        pins = native.source_pins()
        for name in ('dflash_per_turn/native.py', 'dflash_per_turn/training.py',
                     'dflash_two_update_followup/author_training.py'):
            self.assertIn(name, pins)
        self.assertEqual(native.base.OUTPUT_CAP, 1024)
        self.assertEqual(native.base.PREFILL_CHUNK_SIZE, 8192)
        self.assertGreater(native.storage_bound(2_097_252_864), 60*2**30)
        self.assertEqual(native.storage_bound(100, 0), 1100+512*2**20)
        self.assertEqual(native.storage_bound(100, 3), 3200+512*2**20)
        for value in (-1, 4, True):
            with self.assertRaises(ValueError):
                native.storage_bound(100, value)


class Retention(unittest.TestCase):
    def backend(self, directory):
        backend = native.NativeBackend({})
        backend.out = Path(directory)/'episode'; backend.out.mkdir()
        checkpoints = backend.out/'checkpoints'; checkpoints.mkdir()
        states = backend.out/'training_worker'/'states'; states.mkdir(parents=True)
        def checkpoint(label):
            path = checkpoints/label; path.mkdir()
            (path/'config.json').write_text('{}')
            (path/'model.safetensors').write_bytes(label.encode())
            files = {p.name: {'bytes': p.stat().st_size, 'sha256': native.sha(p)} for p in path.iterdir()}
            return dict(path=str(path), files=files, checkpoint_id=label,
                        checkpoint_identity_sha256=native.digest(files))
        backend.checkpoints = {label: checkpoint(label) for label in ('F', 'D1', 'D2', 'D3', 'L1')}
        bindings = {}
        for label in ('initial', 'D1', 'D2', 'D3'):
            path = states/(label+'.pt'); path.write_bytes(label.encode())
            bindings[label] = native.base.binding(path)
        def state(label):
            return {'checkpoint': backend.checkpoints[label], 'state': bindings['initial' if label == 'F' else label]}
        backend.updates = {i: {'pre_state': state('F' if i == 1 else 'D'+str(i-1)),
                               'post_state': state('D'+str(i))} for i in range(1, 4)}
        return backend

    def test_final_retention_requires_complete_evidence_and_keeps_selected_state(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            backend = self.backend(directory)
            path = Path(directory)/'RESULT.json'
            evidence = dict(episode_id='episode', status='failed', implementation_checks={'status': 'passed'},
                            cleanup={'status': 'passed'}, diagnostics=[{'event': 1, 'status': 'complete'}])
            path.write_text(json.dumps(evidence))
            with self.assertRaises(ValueError):
                backend.finalize_retention(result_path=path)
            self.assertTrue((backend.out/'checkpoints/D3').is_dir())
            evidence['status'] = 'complete'; path.write_text(json.dumps(evidence))
            result = backend.finalize_retention(result_path=path)
            self.assertEqual(set(backend.checkpoints), {'F', 'D1', 'L1'})
            self.assertTrue((backend.out/'training_worker/states/initial.pt').is_file())
            self.assertTrue((backend.out/'training_worker/states/D1.pt').is_file())
            self.assertFalse((backend.out/'training_worker/states/D2.pt').exists())
            self.assertFalse((backend.out/'training_worker/states/D3.pt').exists())
            self.assertEqual(len(result['removed']), 4)
            self.assertIn('free_before_bytes', result)
            self.assertEqual(result['completion_evidence']['sha256'], native.sha(path))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU torch required')
class Diagnostic(unittest.TestCase):
    def service(self, out):
        trainer = fixtures.Numerical().trainer()
        pre = trainer.snapshot()
        feedback = fixtures.response(1)
        accounting = trainer.train_event(1, [feedback]); trainer.bind_checkpoint(1, 'b'*64)
        post = trainer.snapshot()
        service = native.TrainerService.__new__(native.TrainerService)
        service.out = Path(out); service.trainer = trainer; service.draft = trainer.draft
        service.current_state = {'key': 'post'}
        service._load_state = lambda reference: trainer.restore(pre if reference['key'] == 'pre' else post)
        service._export_label = lambda label: {'checkpoint_id': label, 'export_s': 0.,
                                             'checkpoint_identity_sha256': 'd'*64}
        update = dict(checkpoint={'checkpoint_id': 'D1'}, pre_state={'key': 'pre'},
                      accounting=accounting, responses=[feedback])
        service.updates = {1: update}
        return service, update, post

    def test_diagnostic_restores_full_state_and_exact_anchors(self):
        with tempfile.TemporaryDirectory() as out:
            service, update, post = self.service(out)
            result = service.diagnostic({'event': 1, 'original_update': update})
            self.assertTrue(result['main_state_restored'])
            self.assertTrue(result['same_anchor_exposure'])
            self.assertEqual(result['checkpoint']['checkpoint_id'], 'L1')
            self.assertEqual(native.training.state_digest(service.trainer.snapshot()), native.training.state_digest(post))
            self.assertTrue((Path(out)/'DIAGNOSTIC_1_ISOLATION.json').is_file())

    def test_failed_diagnostic_restores_weights_optimizer_and_all_rng(self):
        import torch
        with tempfile.TemporaryDirectory() as out:
            service, update, post = self.service(out)
            def fail(*args):
                with torch.no_grad():
                    next(service.draft.parameters()).add_(3.)
                random.random(); torch.rand(2)
                service.trainer.failed = True
                raise RuntimeError('injected branch failure')
            with patch.object(service.trainer, 'train_event', fail):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    service.diagnostic({'event': 1, 'original_update': update})
            self.assertEqual(native.training.state_digest(service.trainer.snapshot()), native.training.state_digest(post))
            import json
            receipt = json.loads((Path(out)/'DIAGNOSTIC_1_ISOLATION.json').read_text())
            self.assertTrue(receipt['main_state_restored'])
            self.assertEqual(receipt['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
