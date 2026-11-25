"""Resource admission must stop before model work, with the refusal retained."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import prepare, qualify


class PrepareTests(unittest.TestCase):
    def admission(self, free, processes=''):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / 'run'
        parent = dict(binding='original', devices={'A': 'a', 'B': 'b'}, training_python='python')
        with patch.object(prepare, 'select_parent', return_value=(parent, Path('parent.pt'), [])), \
                patch.object(prepare, 'identity', return_value=prepare.INPUT_IDENTITY), \
                patch.object(prepare.subprocess, 'check_output', return_value=processes), \
                patch.object(prepare.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)), \
                patch.object(prepare, 'compiler_snapshot', return_value={'allocated_bytes': 0}):
            try:
                prepare.prepare('parent.json', root, 'source.json', 'commit')
            except RuntimeError:
                pass
        return json.loads((root / 'PLAN.json').read_text())

    def test_phase_is_derivative_only_with_no_optimizer_or_generations(self):
        plan = self.admission(432_000_000_000)
        self.assertEqual(plan['status'], 'admitted')
        self.assertFalse(plan['production_admitted'])
        self.assertEqual(plan['attempted_work_ceiling']['native_adam_steps'], 0)
        self.assertEqual(plan['attempted_work_ceiling']['complete_response_generations'], 0)

    def test_peak_writes_and_reserve_both_required(self):
        self.assertEqual(self.admission(405_000_000_000)['status'], 'blocked')

    def test_other_compute_jobs_block_launch(self):
        self.assertEqual(self.admission(432_000_000_000, 'another job')['status'], 'blocked')

    def test_actual_source_mutation_fails_despite_unchanged_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            code = source / 'module.py'
            code.write_text('original source\n')
            manifest = Path(directory) / 'SOURCE.json'
            manifest.write_text(json.dumps(dict(commit='pinned', files={
                'module.py': qualify.identity(code)})))
            plan = dict(source_manifest=str(manifest), source_identity=qualify.identity(manifest),
                        source_root=str(source), execution_commit='pinned')
            qualify.verify_source(plan)
            code.write_text('changed source\n')
            with self.assertRaisesRegex(ValueError, 'Staged source changed'):
                qualify.verify_source(plan)
            code.write_text('original source\n')
            with self.assertRaisesRegex(ValueError, 'Execution commit'):
                qualify.verify_source(dict(plan, execution_commit='other'))

    def test_compiler_growth_is_charged_with_run_and_source_allocations(self):
        plan = dict(root='/new/run', source_manifest='/new/staging/SOURCE.json',
                    compiler_baseline=dict(directories={'cache': '/native/cache'}, allocated_bytes=100))
        with patch.object(qualify, 'allocated_bytes', return_value=700) as allocated, \
                patch.object(qualify.shutil, 'disk_usage', return_value=SimpleNamespace(free=432_000_000_000)), \
                patch.object(qualify, 'compiler_snapshot', return_value=dict(
                    directories={'cache': '/native/cache'}, allocated_bytes=400)) as compiler:
            measured = qualify.storage_usage(plan)
            self.assertEqual(measured['charged_bytes'], 1000)
            self.assertEqual(measured['compiler_growth_bytes'], 300)
            allocated.assert_called_once_with(['/new/run', '/new/staging'])
            compiler.return_value['allocated_bytes'] = 50
            self.assertEqual(qualify.storage_usage(plan)['charged_bytes'], 700)
            compiler.return_value['directories'] = {'cache': '/unexpected/cache'}
            with self.assertRaisesRegex(ValueError, 'Compiler cache locations'):
                qualify.storage_usage(plan)


if __name__ == '__main__':
    unittest.main()
