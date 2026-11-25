"""Capacity, alias and real-block accounting for the production-only route."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.repair_conditional_kl_pilot import production_storage as S


def disk(device=1, free=500_000_000_000):
    return dict(
        device=device,
        capacity_key=str(device),
        source='/dev/disk' + str(device),
        mount='/data',
        fstype='xfs',
        readonly=False,
        durable=True,
        total_bytes=2_000_000_000_000,
        free_bytes=free,
    )


class ProductionStorageTests(unittest.TestCase):
    def test_local_peak_is_maximum_phase_not_sum_of_nonconcurrent_overlap(self):
        requests = [{'prompt_ids': list(range(10))}] * 128
        single = S.phase_peaks(requests, 'local_single')
        f, s, w = [
            single[key]
            for key in (
                'feature_pool_bound_bytes',
                'recovery_bound_bytes',
                'export_bytes',
            )
        ]
        raw = max(row['main_bytes'] for row in single['phases'])
        self.assertEqual(raw, 2 * f + 2 * s + 2 * w)
        self.assertEqual(
            single['peaks']['main'], (raw * 105 + 99) // 100 + 4_000_000_000
        )
        split = S.phase_peaks(requests, 'local_split')
        self.assertEqual(
            max(row['main_bytes'] for row in split['phases']),
            2 * s + 2 * w + split['witness_copy_bound_bytes'],
        )
        self.assertEqual(
            max(row['features_bytes'] for row in split['phases']),
            2 * f + split['preparation_overlap_bytes'],
        )

    def test_hardlinks_and_nested_roots_count_once_and_symlinks_are_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sub = root / 'sub'
            sub.mkdir()
            data = sub / 'data'
            data.write_bytes(b'x' * 9000)
            os.link(data, root / 'link')
            os.symlink('/does/not/exist', root / 'symlink')
            expected = sum(
                p.lstat().st_blocks * 512 for p in (root, sub, data, root / 'symlink')
            )
            actual = S.allocated_by_device([root, sub, data, root / 'link'])
            self.assertEqual(sum(actual.values()), expected)

    def test_atomic_retirement_during_scan_does_not_abort_valid_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / 'pending.tmp'
            temporary.write_bytes(b'x' * 4096)
            original = Path.lstat

            def retiring(path):
                if path == temporary:
                    temporary.unlink()
                return original(path)

            with patch.object(Path, 'lstat', retiring):
                measured = S.allocated_by_device([root])
            self.assertEqual(sum(measured.values()), root.lstat().st_blocks * 512)

    def test_split_requires_actual_distinct_durable_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = dict(
                root=root / 'run',
                source_root=root / 'source/DFlash-Online',
                feature_roots={
                    'BASE': root / 'features/BASE',
                    'PAIR': root / 'features/PAIR',
                },
                strategy='local_split',
            )
            with patch.object(S, 'filesystem', return_value=disk()):
                result = S.prepare_storage([{'prompt_ids': [1]}] * 128, **arguments)
                self.assertFalse(result['admitted'])
                self.assertTrue(
                    any('aliases' in reason for reason in result['blockers'])
                )

            def independent(path):
                return disk(2, 40_000_000_000) if '/features/' in str(path) else disk()

            with patch.object(S, 'filesystem', side_effect=independent):
                result = S.prepare_storage([{'prompt_ids': [1]}] * 128, **arguments)
                self.assertTrue(result['admitted'])
            overlay = disk()
            overlay.update(fstype='overlay', durable=False)
            with patch.object(S, 'filesystem', return_value=overlay):
                result = S.prepare_storage([{'prompt_ids': [1]}] * 128, **arguments)
                self.assertFalse(result['admitted'])
                self.assertTrue(
                    any('durable' in reason for reason in result['blockers'])
                )

    def test_every_filesystem_reserve_and_growth_are_enforced(self):
        main, extra = disk(), disk(2, 2000)
        plan = dict(
            compiler_baseline={'directories': {'CACHE': '/main/cache'}},
            storage=dict(
                admitted=True,
                groups=[
                    dict(
                        name='main',
                        filesystem=main,
                        allocation_roots=['/main/new'],
                        projected_peak_bytes=1000,
                        reserve_bytes=400_000_000_000,
                    ),
                    dict(
                        name='features',
                        filesystem=extra,
                        allocation_roots=['/extra/new'],
                        projected_peak_bytes=1000,
                        reserve_bytes=1000,
                    ),
                ],
            ),
        )

        def identity(path):
            return main if str(path).startswith('/main') else extra

        with patch.object(S, 'filesystem', side_effect=identity), patch.object(
            S, 'allocated_by_device', return_value={1: 300, 2: 200}
        ):
            result = S.check_storage(plan, 100)
            self.assertEqual(result['allocated_bytes'], 600)
            self.assertEqual(result['groups'][0]['remaining_peak_bytes'], 600)
            self.assertEqual(result['groups'][1]['remaining_peak_bytes'], 800)
            extra['free_bytes'] = 1799
            with self.assertRaisesRegex(RuntimeError, 'crosses reserve'):
                S.check_storage(plan, 100)
        extra['free_bytes'] = 2000
        with patch.object(S, 'filesystem', side_effect=identity), patch.object(
            S, 'allocated_by_device', return_value={1: 200, 2: 1001}
        ):
            with self.assertRaisesRegex(RuntimeError, 'exceeded admitted peak'):
                S.check_storage(plan, 0)

    def test_sealed_path_cannot_redirect_through_same_filesystem_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / 'actual'
            destination.mkdir()
            redirected = root / 'sealed'
            redirected.symlink_to(destination, target_is_directory=True)
            plan = dict(
                compiler_baseline={'directories': {}},
                storage=dict(
                    admitted=True,
                    groups=[
                        dict(
                            name='main',
                            filesystem=disk(),
                            allocation_roots=[str(redirected)],
                            projected_peak_bytes=1000,
                            reserve_bytes=400_000_000_000,
                        )
                    ],
                ),
            )
            with self.assertRaisesRegex(RuntimeError, 'redirected'):
                S.check_storage(plan, 0)

    def test_mount_replacement_and_unbudgeted_files_are_rejected(self):
        original = disk()
        plan = dict(
            compiler_baseline={'directories': {}},
            storage=dict(
                admitted=True,
                groups=[
                    dict(
                        name='main',
                        filesystem=original,
                        allocation_roots=['/new'],
                        projected_peak_bytes=1000,
                        reserve_bytes=400_000_000_000,
                    )
                ],
            ),
        )
        changed = deepcopy(original)
        changed['source'] = '/dev/replacement'
        with patch.object(S, 'filesystem', return_value=changed), patch.object(
            S, 'allocated_by_device', return_value={}
        ):
            with self.assertRaisesRegex(RuntimeError, 'filesystem changed'):
                S.check_storage(plan, 0)
        with patch.object(S, 'filesystem', return_value=original), patch.object(
            S, 'allocated_by_device', return_value={1: 100, 2: 100}
        ):
            with self.assertRaisesRegex(RuntimeError, 'unbudgeted'):
                S.check_storage(plan, 0)


if __name__ == '__main__':
    unittest.main()
