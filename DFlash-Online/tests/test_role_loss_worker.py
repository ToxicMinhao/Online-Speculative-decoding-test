"""CPU-only worker transport qualification; native serving methods are inherited."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from experiments.role_loss_pilot import worker
from online_feedback.sglang_worker import NativeWorker


class RoleLossWorkerTests(unittest.TestCase):
    def test_native_methods_are_inherited_without_wrappers(self):
        for name in ('generate', 'teacher', 'publish', 'check_weights', 'status', 'close'):
            self.assertIs(getattr(worker.RoleLossWorker, name), getattr(NativeWorker, name))

    def test_constructor_selects_only_experiment_scheduler_before_native_startup(self):
        settings, root = {'model_path': 'target'}, Path('/unused')
        observed = []
        def native_init(instance, actual_settings, actual_root):
            observed.append(worker.sglang_hooks.scheduler_process)
            self.assertIs(actual_settings, settings)
            self.assertIs(actual_root, root)
        with mock.patch.object(worker.sglang_hooks, 'scheduler_process'), \
             mock.patch.object(NativeWorker, '__init__', native_init):
            worker.RoleLossWorker(settings, root)
        self.assertEqual(observed, [worker.scheduler_process])

    def test_scheduler_installs_online_hooks_and_single_rpc_without_depth_mutation(self):
        scheduler = ModuleType('sglang.srt.managers.scheduler')
        class Scheduler:
            existing_method = object()
        scheduler.Scheduler = Scheduler
        scheduler.run_scheduler_process = mock.Mock(return_value='native-loop')
        depth = ModuleType('sglang.srt.speculative.dflash_worker')
        native_forward = object()
        depth.DFlashWorker = SimpleNamespace(forward_batch_generation=native_forward)
        with mock.patch.dict(sys.modules, {'sglang.srt.managers.scheduler': scheduler,
                                          'sglang.srt.speculative.dflash_worker': depth}), \
             mock.patch.object(worker.sglang_hooks, 'install') as install:
            result = worker.scheduler_process('arg', config='value')
        self.assertEqual(result, 'native-loop')
        install.assert_called_once_with()
        scheduler.run_scheduler_process.assert_called_once_with('arg', config='value')
        self.assertIs(Scheduler.roleloss_reconstruct, worker.roleloss_reconstruct)
        self.assertEqual({name for name in vars(Scheduler) if not name.startswith('__')},
                         {'existing_method', 'roleloss_reconstruct'})
        self.assertIs(depth.DFlashWorker.forward_batch_generation, native_forward)

    def test_scheduler_rpc_requires_idle_and_calls_existing_reconstruction_directly(self):
        scheduler, record = object(), {'record_id': 'retained'}
        result = {'status': 'passed', 'record': record, 'new_generations': 0}
        with mock.patch.object(worker.sglang_hooks, 'strict_idle') as idle, \
             mock.patch('experiments.block_depth.reconstruct.reconstruct', return_value=result) as replay:
            actual = worker.roleloss_reconstruct(scheduler, record, '/receipt.json')
        idle.assert_called_once_with(scheduler)
        replay.assert_called_once_with(scheduler, record, '/receipt.json')
        self.assertIs(actual, result)
        with mock.patch.object(worker.sglang_hooks, 'strict_idle', side_effect=RuntimeError('busy')), \
             mock.patch('experiments.block_depth.reconstruct.reconstruct') as replay:
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                worker.roleloss_reconstruct(scheduler, record, '/receipt.json')
            replay.assert_not_called()

    def test_host_rpc_returns_record_without_nested_result_or_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipt.json'
            source = {'record_id': 'source'}
            expected = {'status': 'passed', 'record': {'record_id': 'source', 'restored': True},
                        'new_generations': 0}
            instance = worker.RoleLossWorker.__new__(worker.RoleLossWorker)
            instance.engine = mock.Mock()
            def collective(name, **kwargs):
                self.assertEqual(name, 'roleloss_reconstruct')
                self.assertIs(kwargs['record'], source)
                Path(kwargs['outputfile']).write_text(json.dumps(expected))
            instance.engine.collective_rpc.side_effect = collective
            result = instance.roleloss_reconstruct(dict(record=source, outputfile=str(path)))
            self.assertEqual(result, expected)
            self.assertNotIn('result', result)
            instance.engine.generate.assert_not_called()
            instance.engine.collective_rpc.assert_called_once()

    @staticmethod
    def fake_worker():
        instance = SimpleNamespace(blocked=False)
        for name in ('generate', 'teacher', 'publish', 'check_weights', 'roleloss_reconstruct'):
            setattr(instance, name, mock.Mock(return_value={'operation': name}))
        return instance

    def test_transport_dispatches_native_and_reconstruction_requests_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            instance = self.fake_worker()
            names = ('generate', 'teacher', 'publish', 'check_weights', 'roleloss_reconstruct')
            requests = [dict(operation=name, payload=index) for index, name in enumerate(names)]
            connection = mock.Mock()
            connection.recv.side_effect = [*requests, dict(operation='close')]
            worker.serve_connection(instance, connection, Path(root))
            for name, request, call in zip(names, requests, connection.send.call_args_list):
                getattr(instance, name).assert_called_once_with(request)
                self.assertEqual(call.args[0], dict(ok=True, result={'operation': name}))
            self.assertEqual(connection.send.call_args.args[0], dict(ok=True, result=None))
            self.assertFalse(instance.blocked)
            self.assertFalse((Path(root) / 'last_failure.json').exists())

    def test_failed_rpc_blocks_later_work_but_close_remains_available(self):
        with tempfile.TemporaryDirectory() as root:
            instance = self.fake_worker()
            instance.roleloss_reconstruct.side_effect = ValueError('posterior differs')
            connection = mock.Mock()
            connection.recv.side_effect = [dict(operation='roleloss_reconstruct'),
                                           dict(operation='generate'), dict(operation='close')]
            worker.serve_connection(instance, connection, root)
            first, second, last = [call.args[0] for call in connection.send.call_args_list]
            self.assertEqual(first['error']['type'], 'ValueError')
            self.assertEqual(first['error']['message'], 'posterior differs')
            self.assertIn('blocked', second['error']['message'])
            self.assertEqual(last, dict(ok=True, result=None))
            self.assertTrue(instance.blocked)
            instance.generate.assert_not_called()
            receipt = json.loads((Path(root) / 'last_failure.json').read_text())
            self.assertEqual(receipt['operation'], 'generate')

    def test_depth_and_unknown_operations_are_not_exposed(self):
        for operation in ('depth_rpc', 'generate_lean', 'unknown'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as root:
                instance, connection = self.fake_worker(), mock.Mock()
                connection.recv.side_effect = [dict(operation=operation), dict(operation='close')]
                worker.serve_connection(instance, connection, root)
                self.assertEqual(connection.send.call_args_list[0].args[0]['error']['type'], 'KeyError')
                self.assertTrue(instance.blocked)
                for name in ('generate', 'teacher', 'publish', 'check_weights', 'roleloss_reconstruct'):
                    getattr(instance, name).assert_not_called()

    def test_main_uses_native_handshake_and_closes_after_transport_failure(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            settings = root / 'settings.json'
            settings.write_text('{"model_path":"target"}')
            instance = self.fake_worker()
            instance.status = mock.Mock(return_value={'status': 'passed'})
            instance.close = mock.Mock()
            listener = mock.MagicMock()
            listener.address = ('127.0.0.1', 12345)
            listener.__enter__.return_value = listener
            with mock.patch.object(sys, 'argv', ['worker', '--settings', str(settings), '--output', str(root)]), \
                 mock.patch.object(worker, 'RoleLossWorker', return_value=instance) as constructor, \
                 mock.patch.object(worker, 'Listener', return_value=listener) as factory, \
                 mock.patch.object(worker, 'serve_connection', side_effect=EOFError('transport closed')):
                with self.assertRaisesRegex(EOFError, 'transport closed'):
                    worker.main()
            constructor.assert_called_once_with({'model_path': 'target'}, root)
            factory.assert_called_once_with(('127.0.0.1', 0), authkey=b'dflash-online-local')
            instance.close.assert_called_once_with()
            ready = json.loads((root / 'ready.json').read_text())
            self.assertEqual(ready['address'], ['127.0.0.1', 12345])
            self.assertEqual(ready['status'], {'status': 'passed'})


if __name__ == '__main__':
    unittest.main()
