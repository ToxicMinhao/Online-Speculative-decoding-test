"""CPU-only transfer/ACK fixtures; SSH is mocked, including tar streaming."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import courier
from .storage import GIB, identity, validate_ack


class CourierTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.destination = self.root/'offload'
        self.destination.mkdir()
        self.control = self.root/'control'
        self.control.mkdir()
        self.source = self.root/'source'
        self.source.mkdir()
        self.campaign = '/data/sources/minhao/runs/fixture/campaign_001'
        self.space = patch.object(courier.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*GIB))
        self.space.start()
        self.addCleanup(self.space.stop)

    def request(self, event=5):
        relative = f'event_{event:03d}/features/a b\nquote\'.pt'
        path = self.source/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture payload ' + str(event).encode())
        return dict(event=event, campaign_root=self.campaign, destination=str(self.destination),
                    phase='idle_after_training_and_evaluation', files={relative: identity(path)})

    def transfer(self, campaign, destination, names, control, event, **kwargs):
        for relative in names:
            path = destination/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.source/relative, path)

    def test_exact_hash_manifest_and_destination_precede_ack(self):
        request = self.request()
        courier.validate_request(request, event=5, campaign=self.campaign, destination=self.destination)
        def ack(campaign, event, result):
            validate_ack(request, result)
            self.assertEqual(result['destination'], str(self.destination))
            for name, expected in result['files'].items():
                self.assertEqual(identity(self.destination/name), expected)
        with patch.object(courier, 'transfer_files', side_effect=self.transfer) as transfer, \
                patch.object(courier, 'publish_ack', side_effect=ack) as publish:
            result = courier.copy_event(request, self.destination, self.control)
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(transfer.call_count, 1)
        self.assertEqual(publish.call_count, 1)
        self.assertTrue((self.control/'event_005_ACK_SENT.json').exists())

    def test_space_budget_charges_payload_reserve_and_workspace_before_transfer(self):
        request = self.request()
        payload = sum(row['bytes'] for row in request['files'].values())
        with patch.object(courier.shutil, 'disk_usage', return_value=SimpleNamespace(free=payload+9*GIB-1)), \
                patch.object(courier, 'transfer_files') as transfer, patch.object(courier, 'publish_ack') as ack:
            with self.assertRaisesRegex(RuntimeError, 'Insufficient local space'):
                courier.copy_event(request, self.destination, self.control)
        transfer.assert_not_called()
        ack.assert_not_called()

    def test_corrupt_partial_is_preserved_without_ack(self):
        request = self.request()
        def corrupt(*args, **kwargs):
            self.transfer(*args, **kwargs)
            (self.destination/next(iter(request['files']))).write_bytes(b'partial')
        with patch.object(courier, 'transfer_files', side_effect=corrupt), patch.object(courier, 'publish_ack') as ack:
            with self.assertRaisesRegex(ValueError, 'Transferred bytes differ'):
                courier.copy_event(request, self.destination, self.control)
        self.assertEqual((self.destination/next(iter(request['files']))).read_bytes(), b'partial')
        ack.assert_not_called()

    def test_existing_matching_files_skip_transfer_and_conflicts_are_not_overwritten(self):
        request = self.request()
        self.transfer(self.campaign, self.destination, request['files'], self.control, 5)
        with patch.object(courier, 'transfer_files') as transfer, patch.object(courier, 'publish_ack'):
            result = courier.copy_event(request, self.destination, self.control)
        transfer.assert_not_called()
        self.assertEqual(result['transferred_files'], 0)
        local = self.destination/next(iter(request['files']))
        local.write_bytes(b'keep conflicting evidence')
        with patch.object(courier, 'publish_ack') as ack:
            with self.assertRaisesRegex(ValueError, 'Existing local file differs'):
                courier.copy_event(request, self.destination, self.control)
        self.assertEqual(local.read_bytes(), b'keep conflicting evidence')
        ack.assert_not_called()

    def test_binding_traversal_and_symlink_fail_closed(self):
        request = self.request()
        for field, value in (('event', 6), ('campaign_root', '/wrong'), ('destination', '/wrong'),
                             ('phase', 'evaluation')):
            changed = dict(request, **{field: value})
            with self.assertRaises(ValueError):
                courier.validate_request(changed, event=5, campaign=self.campaign, destination=self.destination)
        for name in ('../outside', '/absolute', 'a/../outside', 'a//b', '_courier/FAILED.json'):
            changed = copy.deepcopy(request)
            changed['files'] = {name: next(iter(request['files'].values()))}
            with self.assertRaises(ValueError):
                courier.validate_request(changed, event=5, campaign=self.campaign, destination=self.destination)
        (self.destination/'escape').symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escaped'):
            courier.local_path(self.destination, 'escape/file')
        with self.assertRaises(ValueError):
            courier.remote_campaign('/tmp/unapproved')

    def test_sequential_wait_never_transfers_before_request_and_records_failure(self):
        replies = [None, {'status': 'running'}, self.request()]
        def failed_transfer(*args, **kwargs):
            self.transfer(*args, **kwargs)
            raise RuntimeError('transport interrupted')
        with patch.object(courier, 'read_remote_json', side_effect=replies) as read, \
                patch.object(courier.time, 'sleep') as sleep, \
                patch.object(courier, 'transfer_files', side_effect=failed_transfer) as transfer, \
                patch.object(courier, 'publish_ack') as ack:
            with self.assertRaisesRegex(RuntimeError, 'transport interrupted'):
                courier.run('/data/sources/minhao/runs/fixture', self.destination)
        sleep.assert_called_once_with(15)
        self.assertEqual(read.call_count, 3)
        self.assertEqual(transfer.call_count, 1)
        ack.assert_not_called()
        failure = json.loads((self.destination/'_courier/FAILED.json').read_text())
        self.assertEqual(failure['acknowledged_events'], [])
        self.assertFalse(failure['automatic_retry'])
        self.assertTrue((self.destination/next(iter(replies[-1]['files']))).exists())

    def test_four_events_acknowledged_in_order_and_stopped_campaign_no_transfer(self):
        replies = [self.request(event) for event in (5, 6, 7, 8)]
        with patch.object(courier, 'read_remote_json', side_effect=replies), \
                patch.object(courier, 'transfer_files', side_effect=self.transfer), \
                patch.object(courier, 'publish_ack') as ack:
            result = courier.run('/data/sources/minhao/runs/fixture', self.destination)
        self.assertEqual(result['acknowledged_events'], [5, 6, 7, 8])
        self.assertEqual([call.args[1] for call in ack.call_args_list], [5, 6, 7, 8])
        other = self.root/'stopped'
        with patch.object(courier, 'read_remote_json', side_effect=[None, {'status': 'failed'}]), \
                patch.object(courier, 'transfer_files') as transfer:
            with self.assertRaisesRegex(RuntimeError, 'Campaign stopped'):
                courier.run('/data/sources/minhao/runs/fixture', other)
        transfer.assert_not_called()

    def test_tar_stream_uses_exact_null_list_without_archive_or_remote_rsync(self):
        request = self.request()
        (self.source/'unlisted').write_bytes(b'not requested')
        real_popen = subprocess.Popen
        commands = []
        def local_ssh(arguments, **kwargs):
            commands.append(arguments)
            if arguments[0] == 'ssh':
                arguments = ['sh', '-c', arguments[-1]]
            return real_popen(arguments, **kwargs)
        with patch.object(courier.subprocess, 'Popen', side_effect=local_ssh):
            courier.transfer_files(str(self.source), self.destination, sorted(request['files']),
                                   self.control, 5, timeout=30)
        self.assertEqual(identity(self.destination/next(iter(request['files']))), next(iter(request['files'].values())))
        self.assertFalse((self.destination/'unlisted').exists())
        self.assertEqual(commands[0][:len(courier.SSH)], list(courier.SSH))
        self.assertIn('--keep-old-files', commands[1])
        self.assertTrue((self.control/'event_005_FILES.bin').read_bytes().endswith(b'\0'))
        self.assertFalse(list(self.root.rglob('*.tar')))

    def test_ack_shell_contains_no_payload_and_uses_stdin_atomic_move(self):
        ack = dict(status='verified', destination="/path with ' quote", files={'literal$(echo nope)': {}})
        with patch.object(courier.subprocess, 'run', return_value=SimpleNamespace(stdout='')) as run:
            courier.publish_ack(self.campaign, 5, ack)
        arguments, options = run.call_args.args[0], run.call_args.kwargs
        self.assertEqual(json.loads(options['input']), ack)
        self.assertNotIn('literal', arguments[-1])
        self.assertIn('set -C; cat >', arguments[-1])
        self.assertIn('mv --', arguments[-1])
        self.assertGreaterEqual(options['timeout'], 45)


if __name__ == '__main__':
    unittest.main()
