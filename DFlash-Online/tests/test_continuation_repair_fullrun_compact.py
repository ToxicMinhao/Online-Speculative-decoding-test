"""Completed-event storage repair never repeats scientific work."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments.continuation_repair_fullrun import compact


@unittest.skipUnless(shutil.which('zstd'), 'zstd is required for lossless archive verification')
class CompactRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root/'campaign_001/arms/BASE'
        self.event = self.base/'events/event_001'
        self.trainer = self.base/'trainer/events/event_001'
        self.observations = self.trainer/'RESPONSE_OBSERVATIONS.jsonl'
        self.put(self.root/'PROTECTED_INPUTS.json', dict(files=[]))
        self.put(self.base/'boundaries/event_001.json', dict(status='passed', event=1))
        self.put(self.base/'STATUS.json', dict(status='running', phase='idle_boundary', events_completed=1))
        self.records = []
        for i in range(128):
            path = self.base/'native'/f'{i:03d}.json'
            self.put(path, dict(response=[i]))
            self.records.append(dict(record_id=str(i), metadata=dict(native_trace_path=str(path))))
        self.put(self.event/'loop/feedback_buffer/state.json', dict(records=self.records))
        self.put(self.trainer/'RESULT.json', dict(status='completed'))
        self.put(self.trainer/'WINDOW_OBSERVATIONS.jsonl', dict(window=128))
        self.observations.write_text('{"loss": 0.1}\n')
        self.archive = self.root/'compact/BASE/event_001.tar.zst'
        self.receipt = Path(str(self.archive)+'.retirement.json')
        self.journal = Path(str(self.archive)+'.retirement.recovery_001.json')
        self.failure = Path(str(self.archive)+'.retirement.failed_001.json')
    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    def fluctuate(self):
        original = compact._identity
        calls = {}
        def sampled(path):
            value = original(path)
            if Path(path) == self.observations:
                calls[path] = calls.get(path, 0)+1
                if calls[path] == 1:
                    value['allocated_bytes'] += 8192
            return value
        return sampled
    def fail_like_old_checker(self):
        # Reproduce old full-dict comparison with identical content and only
        # the planned st_blocks snapshot differing at retirement.
        with patch.object(compact, '_identity', side_effect=self.fluctuate()), \
             patch.object(compact, '_same_file', side_effect=lambda a,b:a==b):
            with self.assertRaisesRegex(ValueError, 'Loose member changed'):
                compact.archive_completed_event(self.root, 'BASE', 1, self.records)
        self.before = self.receipt.read_bytes()
        self.before_archive = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.assertEqual(json.loads(self.before)['status'], 'failed')
        self.assertTrue(self.observations.exists())
    def test_allocation_change_is_not_content_change_and_actual_blocks_counted(self):
        with patch.object(compact, '_identity', side_effect=self.fluctuate()):
            result = compact.archive_completed_event(self.root, 'BASE', 1, self.records)
        self.assertEqual(result['status'], 'completed')
        index = json.loads(Path(str(self.archive)+'.index.json').read_text())
        expected = sum(v['allocated_bytes'] for v in index['members'].values())-8192
        self.assertEqual(result['removed_allocated_bytes'], expected)
        name = str(self.observations.relative_to(self.root))
        self.assertEqual(result['allocation_changes'][name]['planned_bytes']-
                         result['allocation_changes'][name]['before_unlink_bytes'], 8192)
        self.assertEqual(compact.read_member(self.archive, name), b'{"loss": 0.1}\n')
    def test_content_inode_and_mtime_changes_still_fail(self):
        original = dict(bytes=5, sha256='a', allocated_bytes=4096, device=1, inode=2, mtime_ns=3)
        self.assertTrue(compact._same_file(dict(original, allocated_bytes=8192), original))
        for key in ('bytes', 'sha256', 'device', 'inode', 'mtime_ns'):
            changed = dict(original); changed[key] = 'changed'
            self.assertFalse(compact._same_file(changed, original), key)
    def test_partial_verified_retirement_resumes_only_remaining_files(self):
        self.fail_like_old_checker()
        status_bytes = (self.base/'STATUS.json').read_bytes()
        previous = json.loads(self.before)
        result = compact.resume_verified_retirement(self.root, 'BASE', 1)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(self.failure.read_bytes(), self.before)
        self.assertEqual(hashlib.sha256(self.archive.read_bytes()).hexdigest(), self.before_archive)
        self.assertEqual((self.base/'STATUS.json').read_bytes(), status_bytes)
        self.assertEqual(result['recovery']['new_model_work'], 0)
        self.assertEqual(result['recovery']['previous_removed_count'], len(previous['removed_paths']))
        self.assertEqual(result['recovery']['remaining_count'], 3)
        self.assertEqual(set(result['recovery']['new_removed_paths']),
                         set(result['removed_paths'])-set(previous['removed_paths']))
        self.assertEqual(result['removed_allocated_bytes'],
                         previous['removed_allocated_bytes']+result['recovery']['new_removed_allocated_bytes'])
        with self.assertRaisesRegex(FileExistsError, 'already attempted'):
            compact.resume_verified_retirement(self.root, 'BASE', 1)
    def test_missing_unrecorded_member_stops_before_any_new_unlink(self):
        self.fail_like_old_checker(); self.observations.unlink()
        remaining = self.trainer/'RESULT.json'
        self.assertTrue(remaining.exists())
        with self.assertRaisesRegex(ValueError, 'Unrecorded missing'):
            compact.resume_verified_retirement(self.root, 'BASE', 1)
        self.assertTrue(remaining.exists())
        self.assertEqual(self.receipt.read_bytes(), self.before)
        self.assertEqual(json.loads(self.journal.read_text())['new_removed_paths'], [])
    def test_changed_remaining_content_keeps_original_failure(self):
        self.fail_like_old_checker(); self.observations.write_text('changed bytes\n')
        with self.assertRaisesRegex(ValueError, 'Remaining loose member differs'):
            compact.resume_verified_retirement(self.root, 'BASE', 1)
        self.assertEqual(self.receipt.read_bytes(), self.before)
        self.assertEqual(self.failure.read_bytes(), self.before)
        self.assertEqual(json.loads(self.journal.read_text())['new_removed_paths'], [])
    def test_all_remaining_members_checked_before_first_unlink(self):
        self.fail_like_old_checker()
        (self.trainer/'WINDOW_OBSERVATIONS.jsonl').write_text('changed last member\n')
        with self.assertRaisesRegex(ValueError, 'Remaining loose member differs'):
            compact.resume_verified_retirement(self.root, 'BASE', 1)
        self.assertTrue(self.observations.exists())
        self.assertEqual(json.loads(self.journal.read_text())['new_removed_paths'], [])
    def test_changed_archive_stops_before_retirement(self):
        self.fail_like_old_checker()
        with self.archive.open('ab') as stream: stream.write(b'bad')
        with self.assertRaisesRegex(ValueError, 'archive identity changed'):
            compact.resume_verified_retirement(self.root, 'BASE', 1)
        self.assertTrue(self.observations.exists())
        self.assertEqual(self.receipt.read_bytes(), self.before)
    def test_reappearing_recorded_removed_path_rejected(self):
        self.fail_like_old_checker()
        previous = json.loads(self.before)
        self.put(self.root/previous['removed_paths'][0], dict(unexpected=True))
        with self.assertRaisesRegex(ValueError, 'Previously retired path unexpectedly exists'):
            compact.resume_verified_retirement(self.root, 'BASE', 1)
        self.assertTrue(self.observations.exists())


if __name__ == '__main__': unittest.main()
