"""Storage rejection and partial-start shutdown without model execution."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import experiment, native


def stream():
    return dict(stream_id='wildchat_00', source='wildchat', synthetic=False,
                requests=[dict(turn=t, prompt=f'request {t}') for t in range(1, 21)])


class PartialStartCleanup(unittest.TestCase):
    def test_actual_episode_storage_rejection_preserves_primary_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'native'
            pins = dict.fromkeys(native.base.REQUIRED_RUNTIME, 'pinned')
            backend = native.NativeBackend(dict(output_root=str(output), workspace_root=directory,
                runtime_source_sha256=pins, serving_cuda_visible_devices='0',
                training_cuda_visible_devices='1', episode_storage_budget_bytes=1,
                reserve_bytes=8*2**30))
            # Skip the inherited tokenizer/source setup, but execute the actual
            # base admission check that failed before assigning self.out.
            with patch.object(native.per_turn.NativeBackend, 'start_episode',
                              native.base.NativeBackend.start_episode), \
                 patch.object(native.per_turn, 'storage_bound', native._original_storage_bound), \
                 patch.object(native.base, 'check_environment'), \
                 patch.object(native.base, 'check_dependencies', return_value={'runtime_source_sha256': pins}), \
                 patch.object(native.shutil, 'disk_usage', return_value=SimpleNamespace(free=8*2**30)), \
                 patch.object(native.base, '_WorkerClient') as worker:
                result = experiment.run_stream(stream(), backend)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['error'],
                'ValueError: Complete episode peak writes plus reserve exceed free space')
            self.assertEqual(result['cleanup']['status'], 'passed')
            self.assertEqual(result['cleanup']['episode_id'], 'wildchat_00')
            self.assertFalse(result['cleanup']['episode_initialized'])
            self.assertEqual(result['scores'], [])
            self.assertEqual(result['updates'], [])
            self.assertFalse(hasattr(backend, 'out'))
            self.assertFalse((output / 'wildchat_00').exists())
            worker.assert_not_called()

    def test_rejection_before_episode_initialization_also_cleans_up(self):
        backend = native.NativeBackend({})
        with patch.object(native, 'admit_campaign_storage', side_effect=ValueError('campaign budget rejected')), \
             patch.object(native.base, '_WorkerClient') as worker:
            result = experiment.run_stream(stream(), backend, qualification=True)
        self.assertEqual(result['error'], 'ValueError: campaign budget rejected')
        self.assertEqual(result['cleanup']['status'], 'passed')
        self.assertFalse(result['cleanup']['episode_initialized'])
        self.assertFalse(result['cleanup']['qualification_direct_features_retained'])
        worker.assert_not_called()

    def test_initialized_cleanup_closes_both_workers_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = native.NativeBackend({})
            backend.out = Path(directory)
            backend.episode_id = 'wildchat_00'
            backend.qualification = False
            worker, trainer = Mock(), Mock()
            backend.worker, backend.trainer = worker, trainer
            evidence = backend.out / 'retained_checkpoint'
            evidence.write_bytes(b'keep')
            receipt = backend.close_episode({'episode_id': 'wildchat_00', 'status': 'failed'})
            self.assertEqual(receipt['status'], 'passed')
            self.assertTrue(receipt['episode_initialized'])
            worker.close.assert_called_once_with()
            trainer.close.assert_called_once_with()
            self.assertIsNone(backend.worker)
            self.assertIsNone(backend.trainer)
            self.assertIsNone(backend.episode_id)
            self.assertEqual(evidence.read_bytes(), b'keep')
            self.assertTrue((backend.out / 'BANK_CLOSE.json').exists())
            self.assertTrue((backend.out / 'NATIVE_FINISH.json').exists())
            # A subsequent admission failure must not overwrite prior receipts.
            closed = (backend.out / 'BANK_CLOSE.json').read_bytes()
            receipt = backend.close_episode({'episode_id': 'next_stream', 'status': 'failed'})
            self.assertFalse(receipt['episode_initialized'])
            self.assertEqual(receipt['episode_id'], 'next_stream')
            self.assertEqual((backend.out / 'BANK_CLOSE.json').read_bytes(), closed)
            worker.close.assert_called_once_with()

    def test_shutdown_failure_is_reported_after_attempting_both_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = native.NativeBackend({})
            backend.out, backend.episode_id = Path(directory), 'wildchat_00'
            worker, trainer = Mock(), Mock()
            worker.close.side_effect = RuntimeError('shutdown failed')
            backend.worker, backend.trainer = worker, trainer
            with self.assertRaisesRegex(ValueError, 'Native shutdown failed'):
                backend.close_episode({'episode_id': 'wildchat_00'})
            trainer.close.assert_called_once_with()
            self.assertTrue((backend.out / 'NATIVE_FINISH.json').exists())
            self.assertFalse((backend.out / 'BANK_CLOSE.json').exists())
            self.assertIsNone(backend.worker)
            self.assertIsNone(backend.trainer)


class RetainedStateBudget(unittest.TestCase):
    def test_actual_retirement_and_close_keep_two_restart_bundles(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = native.NativeBackend({})
            backend.out, backend.episode_id = Path(directory), 'wildchat_00'
            checkpoints, states = backend.out/'checkpoints', backend.out/'training_worker'/'states'
            checkpoints.mkdir(); states.mkdir(parents=True)
            M = 100
            backend.checkpoints = {}
            bindings = {}
            for label in ('F', 'D1', 'D2', 'D3'):
                path = checkpoints/label; path.mkdir()
                (path/'config.json').write_text('{}')
                (path/'model.safetensors').write_bytes(b'm'*M)
                files = {p.name: {'bytes': p.stat().st_size, 'sha256': native.sha(p)} for p in path.iterdir()}
                backend.checkpoints[label] = dict(path=str(path), files=files, checkpoint_id=label,
                    checkpoint_identity_sha256=native.digest(files))
                state_path = states/(label+'.pt'); state_path.write_bytes(b'o'*(2*M))
                bindings[label] = native.base.binding(state_path)
            def state(label):
                return {'checkpoint': backend.checkpoints[label], 'state': bindings[label]}
            backend.updates = {i: {'pre_state': state('F' if i == 1 else 'D'+str(i-1)),
                                   'post_state': state('D'+str(i))} for i in range(1, 4)}
            # Use the production retention path, not a fixture of its result.
            backend.retire({'F', 'D3'}, set())
            self.assertEqual(set(backend.checkpoints), {'F', 'D2', 'D3'})
            self.assertEqual({p.name for p in states.iterdir()}, {'D2.pt', 'D3.pt'})
            retained = [checkpoints/label/'model.safetensors' for label in ('D2', 'D3')]+list(states.iterdir())
            self.assertEqual(sum(p.stat().st_size for p in retained), 6*M)
            before = {str(p): native.sha(p) for p in retained}
            backend.close_episode({'episode_id': 'wildchat_00', 'status': 'complete'})
            self.assertEqual({str(p): native.sha(p) for p in retained}, before)
            budget = native.campaign_storage_bound(M)
            self.assertEqual(budget['assumptions']['retained_osd_model_units_per_episode'], 6)


if __name__ == '__main__':
    unittest.main()
