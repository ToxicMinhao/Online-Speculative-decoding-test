"""Mocked CPU launch tests: no subprocess, model, CUDA allocation or remote work."""

from contextlib import ExitStack
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from experiments.continuation_value_diagnostic import launch


class FakeProcess:
    def __init__(self, pid, completed=True):
        self.pid = pid
        self.returncode = 0 if completed else None
        self.waits = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class ContinuationLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = dict(bytes=10, sha256='fixture')
        self.contract = dict(
            root=str(self.root), source_root=str(self.root), devices=['GPU-A', 'GPU-B'],
            training_python='python', execution_commit='fixture', local_charged_bytes=100,
            limits=dict(reserve_bytes=400_000_000_000, allocated_bytes=2_000_000_000,
                        elapsed_s=5400, occupied_card_s=3600),
            endpoints={name: dict(export=str(self.root / name), identity=self.identity)
                       for name in ('parent', 'child')},
        )
        self.storage = dict(charged_bytes=200, free_bytes=405_000_000_000)

    def patches(self, **overrides):
        stack = ExitStack()
        values = dict(verify_source=None, gpu_inventory=dict(processes=[]),
                      storage_usage=self.storage, identity=self.identity)
        values.update(overrides)
        for name, value in values.items():
            stack.enter_context(patch.object(launch, name, return_value=value))
        return stack

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def completed_spawn(self, command, **kwargs):
        phase = command[command.index('--phase') + 1]
        shard = int(command[command.index('--shard') + 1])
        if phase == 'qualification':
            for endpoint in self.contract['endpoints']:
                path = self.root / phase / f'gpu{shard}' / endpoint / 'primary_1.json'
                path.parent.mkdir(parents=True)
                row = dict(record_id='fixture-response', anchors=[7], geometry=dict(slots=16), blocks=[])
                path.write_text(json.dumps(row))
        return FakeProcess(100 + shard)

    def test_peak_storage_reserve_is_checked_before_clock_or_child(self):
        with patch.object(launch, 'verify_source'), \
             patch.object(launch, 'storage_usage', return_value=dict(
                 charged_bytes=200, free_bytes=400_200_000_000)), \
             patch.object(launch.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'storage'):
                launch.execute(self.contract)
        spawn.assert_not_called()
        self.assertFalse((self.root / 'CLOCK.json').exists())

    def test_all_location_peak_includes_existing_remote_and_local(self):
        self.contract['local_charged_bytes'] = 900_000_000
        with patch.object(launch, 'storage_usage', return_value=dict(
                charged_bytes=900_000_001, free_bytes=405_000_000_000)):
            with self.assertRaisesRegex(RuntimeError, 'storage'):
                launch.check_storage(self.contract)

    def test_existing_attempt_is_never_overwritten_or_restarted(self):
        (self.root / 'CLOCK.json').write_text('retained-clock')
        with patch.object(launch.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(FileExistsError, 'preserved'):
                launch.execute(self.contract)
        self.assertEqual((self.root / 'CLOCK.json').read_text(), 'retained-clock')
        spawn.assert_not_called()

    def test_busy_gpu_prevents_start_without_allocating_clock(self):
        with self.patches(gpu_inventory=dict(processes=[dict(pid=99)])), \
             patch.object(launch.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                launch.execute(self.contract)
        spawn.assert_not_called()
        self.assertFalse((self.root / 'CLOCK.json').exists())

    def test_exit_race_during_termination_is_safe(self):
        process = FakeProcess(123, completed=False)
        with patch.object(launch.os, 'killpg', side_effect=ProcessLookupError) as terminate:
            launch.stop_process(process)
        terminate.assert_called_once_with(123, signal.SIGTERM)
        self.assertEqual(process.waits, [10])

    def test_second_spawn_failure_stops_first_and_preserves_cost(self):
        process = FakeProcess(123, completed=False)
        with self.patches(), \
             patch.object(launch.subprocess, 'Popen', side_effect=[process, OSError('spawn failed')]), \
             patch.object(launch.os, 'killpg') as terminate:
            with self.assertRaisesRegex(OSError, 'spawn failed'):
                launch.execute(self.contract)
        terminate.assert_called_once_with(123, signal.SIGTERM)
        clock = self.read('CLOCK.json')
        self.assertEqual(len(clock['processes']), 1)
        row = clock['processes'][0]
        self.assertIsNotNone(row['end_epoch'])
        self.assertEqual(row['exit_code'], -15)
        self.assertGreaterEqual(clock['occupied_card_s'], 0)
        self.assertEqual(self.read('STATUS.json')['status'], 'blocked')

    def test_cross_card_failure_is_saved_and_blocks_primary(self):
        calls = []

        def spawn(command, **kwargs):
            calls.append(command)
            process = self.completed_spawn(command, **kwargs)
            shard = command[command.index('--shard') + 1]
            if shard == '1':
                path = self.root / 'qualification/gpu1/parent/primary_1.json'
                row = json.loads(path.read_text())
                row['blocks'] = [dict(different=True)]
                path.write_text(json.dumps(row))
            return process

        with self.patches(), patch.object(launch.subprocess, 'Popen', side_effect=spawn):
            with self.assertRaisesRegex(ValueError, 'parity'):
                launch.execute(self.contract)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all('qualification' in call for call in calls))
        self.assertEqual(self.read('CROSS_CARD.json')['status'], 'failed')
        self.assertEqual(self.read('STATUS.json')['status'], 'blocked')

    def test_both_phases_use_pinned_devices_and_charge_all_children(self):
        with self.patches(), \
             patch.object(launch.subprocess, 'Popen', side_effect=self.completed_spawn) as spawn:
            launch.execute(self.contract)
        self.assertEqual(spawn.call_count, 4)
        self.assertEqual([call.kwargs['env']['CUDA_VISIBLE_DEVICES']
                          for call in spawn.call_args_list], ['GPU-A', 'GPU-B', 'GPU-A', 'GPU-B'])
        status = self.read('STATUS.json')
        self.assertEqual(status['status'], 'completed')
        self.assertEqual([p['phase'] for p in status['phases']], ['qualification', 'primary'])
        clock = self.read('CLOCK.json')
        self.assertEqual(len(clock['processes']), 4)
        self.assertAlmostEqual(clock['occupied_card_s'], sum(
            r['end_epoch'] - r['start_epoch'] for r in clock['processes']))
        self.assertEqual(self.read('CROSS_CARD.json')['status'], 'passed')
        self.assertEqual(sum(status['forbidden_work'].values()), 0)


if __name__ == '__main__':
    unittest.main()
