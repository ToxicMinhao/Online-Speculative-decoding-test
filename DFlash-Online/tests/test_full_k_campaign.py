"""CPU-only orchestration checks; these never construct a model or run an update."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.full_k_gsm8k import campaign as C
from experiments.full_k_gsm8k import storage as S


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CampaignBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def test_full_horizon_and_untrained_tail_accounting(self):
        self.assertEqual(C.expected_work(58, 49), C.ARM_LIMITS)
        self.assertEqual(C.expected_work(0), dict.fromkeys(C.ARM_LIMITS, 0))
        for event, intermediate in ((1, 0), (13, 0), (14, 128),
                                    (29, 256), (44, 384), (58, 384)):
            with self.subTest(event=event):
                work = C.expected_work(event)
                self.assertEqual(work['online_generations'], 128 * event)
                self.assertEqual(work['correctness_generations'], 4)
                self.assertEqual(work['intermediate_generations'], intermediate)
                self.assertEqual(work['optimizer_steps'], 128 * event)
                self.assertEqual(work['terminal_preparation_forwards'], 128 * event)
        before, after = C.expected_work(58), C.expected_work(58, 49)
        self.assertEqual({key for key in before if before[key] != after[key]},
                         {'online_generations'})

    def test_phase_wrappers_count_only_actual_actions(self):
        calls = []
        backend = SimpleNamespace(
            generate=lambda prompt, **kw: calls.append((prompt, kw)) or 'response',
            prepare_feedback=lambda record, path: (record, path))
        ledger = C.ArmLedger(self.root / 'WORK.jsonl')
        progress = {'generation_phase': 'online'}
        C.counted_backend(backend, ledger, progress)
        self.assertEqual(backend.generate([1], purpose='serving'), 'response')
        progress['generation_phase'] = 'correctness'
        for _ in range(4):
            backend.generate([2], purpose='evaluation')
        progress['generation_phase'] = 'intermediate'
        backend.generate([3], purpose='evaluation')
        self.assertEqual(backend.prepare_feedback('record', 'path'), ('record', 'path'))
        work = C.read_work(ledger.path, C.ARM_LIMITS)
        self.assertEqual(work['attempts'], work['completed'])
        self.assertEqual(work['completed']['online_generations'], 1)
        self.assertEqual(work['completed']['correctness_generations'], 4)
        self.assertEqual(work['completed']['intermediate_generations'], 1)
        self.assertEqual(work['completed']['terminal_preparation_forwards'], 1)
        self.assertEqual(work['completed']['optimizer_steps'], 0)
        self.assertEqual(len(calls), 6)

    def test_unknown_native_work_is_preserved_and_cannot_resume(self):
        ledger = C.ArmLedger(self.root / 'WORK.jsonl')
        def failed_generation():
            raise RuntimeError('native execution interrupted')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            ledger.call('online_generations', failed_generation)
        actual = C.read_work(ledger.path, C.ARM_LIMITS)
        self.assertEqual(actual['attempts']['online_generations'], 1)
        self.assertEqual(actual['completed']['online_generations'], 0)
        self.assertEqual(actual['unknown']['online_generations'], 1)
        preserved = ledger.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Unknown scientific work'):
            C.ArmLedger(ledger.path, resume=True)
        self.assertEqual(ledger.path.read_bytes(), preserved)

    def test_restart_rejects_partial_work_before_loading_tensors(self):
        arm_root = self.root / 'campaign_001/arms/K'
        save(arm_root / 'STATUS.json', dict(phase='idle_boundary', events_completed=1,
                                          tail_completed=0, arm='K', binding='binding'))
        rows = []
        for kind, count in C.expected_work(1).items():
            for ordinal in range(1, count + 1):
                rows.append(dict(kind=kind, phase='started', ordinal=ordinal))
                if not (kind == 'optimizer_steps' and ordinal == count):
                    rows.append(dict(kind=kind, phase='completed', ordinal=ordinal))
        (arm_root / 'WORK.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with patch('torch.load') as load, self.assertRaisesRegex(ValueError, 'repeat, omit or hide'):
            C.validate_restart(self.root, {'binding': 'binding'}, 'K')
        load.assert_not_called()

    def test_boundary_inventory_excludes_auxiliary_receipts_and_deduplicates_inodes(self):
        arm_root = self.root / 'campaign_001/arms/K'
        initial = arm_root / 'trainer/initial_state.pt'
        initial.parent.mkdir(parents=True)
        initial.write_bytes(b'initial')
        state = arm_root / 'trainer/events/event_001/training_state.pt'
        state.parent.mkdir(parents=True)
        state.write_bytes(b'saved state')
        weights = state.parent / 'export/model.safetensors'
        weights.parent.mkdir()
        weights.write_bytes(b'weights')
        save(arm_root / 'boundaries/event_001.json',
             dict(status='passed', state=str(state), weights=str(weights)))
        save(arm_root / 'boundaries/event_001_validation.json',
             dict(status='passed', live={'exact_resume': True}))
        save(arm_root / 'boundaries/event_001_retirement.json',
             dict(status='completed', retired_state='missing'))
        paths = S.boundary_files(self.root / 'campaign_001')
        self.assertEqual(set(paths), {initial, state, weights})
        latest = arm_root / 'trainer/training_state.pt'
        latest.hardlink_to(state)
        with patch.object(S.shutil, 'disk_usage', return_value=SimpleNamespace(free=10**12)):
            admission = S.admit(self.root, S.capacity_plan(100),
                                validated_tensor_paths=paths + [latest])
        self.assertEqual(admission['validated_existing_tensor_bytes'],
                         sum(path.stat().st_size for path in (initial, state, weights)))
        self.assertTrue(admission['no_future_deletion_credit'])

    def test_retirement_preserves_successor_and_compact_predecessor_evidence(self):
        arm_root = self.root / 'campaign_001/arms/K'
        old = arm_root / 'trainer/events/event_001/training_state.pt'
        successor = arm_root / 'trainer/events/event_002/training_state.pt'
        for path, data in ((old, b'old'), (successor, b'new')):
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
        previous = dict(status='passed', state=str(old), state_identity=C.identity(old))
        save(arm_root / 'boundaries/event_001.json', previous)
        save(arm_root / 'boundaries/event_002.json', dict(status='passed', state=str(successor)))
        C.retire_previous_state(self.root, 'K', 2)
        self.assertFalse(old.exists())
        self.assertEqual(successor.read_bytes(), b'new')
        self.assertEqual(C.read(arm_root / 'boundaries/event_001.json'), previous)
        retirement = C.read(arm_root / 'boundaries/event_001_retirement.json')
        self.assertEqual(retirement['status'], 'completed')
        self.assertEqual(retirement['successor'], str(successor))
        self.assertIn('not archived', retirement['lost_capability'])


if __name__ == '__main__':
    unittest.main()
