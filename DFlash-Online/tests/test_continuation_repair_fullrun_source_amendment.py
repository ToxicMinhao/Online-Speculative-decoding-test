import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.continuation_repair_fullrun import source_amendment as amendment
from experiments.continuation_repair_fullrun import entry, launch, runtime


class AmendmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = self.root / 'original_source'
        self.repaired = self.root / 'run/infrastructure_sources/repair_001/DFlash-Online'
        self.original.mkdir()
        self.repaired.mkdir(parents=True)
        for name in ('compact.py', 'entry.py', 'launch.py'):
            self.write_source(self.original, amendment.PREFIX + name, 'original = 1\n')
            self.write_source(self.repaired, amendment.PREFIX + name, 'repaired = 1\n')
        self.write_source(self.repaired, amendment.PREFIX + 'source_amendment.py', 'verifier = 1\n')
        for root in (self.original, self.repaired):
            self.write_source(root, 'online_feedback/objective.py', 'same_numerics = 1\n')
            self.write_source(root, 'training/native.py', 'same_native = 1\n')
        self.write_source(self.repaired, 'tests/test_repair.py', 'test_only = 1\n')
        self.write_source(self.repaired, 'reports/repair/audit.py', 'report_only = 1\n')
        self.original_manifest = self.root/'ORIGINAL_SOURCE.json'
        self.repaired_manifest = self.root/'run/infrastructure_sources/repair_001/SOURCE.json'
        self.save_manifest(self.original, self.original_manifest, '1'*40)
        self.save_manifest(self.repaired, self.repaired_manifest, '2'*40)
        self.plan_path = self.root/'run/sealed/PLAN.json'
        self.plan = dict(root=str(self.root/'run'), source_root=str(self.original),
            source_manifest=str(self.original_manifest), source_identity=amendment.identity(self.original_manifest),
            execution_commit='1'*40, scientific_setting={'learning_rate': .0003},
            devices={'A': 'GPU-A', 'B': 'GPU-B'}, training_python='/python', zstd='/bin/zstd')
        self.plan['binding'] = amendment.digest(self.plan)
        runtime.write_json(self.plan_path, self.plan)
        self.cpu = self.root/'run/CPU_RECOVERY.json'
        runtime.write_json(self.cpu, dict(status='completed', new_model_work=0,
            plan_binding=self.plan['binding'], execution_commit=self.plan['execution_commit']))
        self.path = self.root/'run/infrastructure_sources/repair_001/SOURCE_AMENDMENT.json'

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_source(root, name, data):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)

    @staticmethod
    def save_manifest(root, path, commit):
        files = {str(p.relative_to(root)): amendment.identity(p) for p in root.rglob('*') if p.is_file()}
        runtime.write_json(path, dict(commit=commit, clean=True, files=files))

    def seal(self):
        return amendment.seal_amendment(self.plan_path, source_root=self.repaired,
            source_manifest=self.repaired_manifest, resume_evidence=self.cpu, output=self.path)

    def verify(self, **kwargs):
        with patch.object(amendment, '__file__', str(self.repaired/amendment.PREFIX/'source_amendment.py')):
            return amendment.verify_amendment(self.plan_path, self.path, **kwargs)

    def test_plan_stays_identical_and_commits_are_distinct(self):
        before = self.plan_path.read_bytes()
        self.seal()
        result = self.verify(expected_sha256=amendment.identity(self.path)['sha256'])
        self.assertEqual(result['execution_commit'], '1'*40)
        self.assertEqual(result['actual_execution_commit'], '2'*40)
        self.assertEqual(result['source_amendment_identity'], amendment.identity(self.path))
        self.assertEqual(self.plan_path.read_bytes(), before)

    def test_numeric_module_change_is_rejected(self):
        self.write_source(self.repaired, 'online_feedback/objective.py', 'changed_loss = 1\n')
        self.save_manifest(self.repaired, self.repaired_manifest, '2'*40)
        with self.assertRaisesRegex(ValueError, 'unapproved production'):
            self.seal()

    def test_nonpython_production_config_change_is_rejected(self):
        self.write_source(self.repaired, 'training/config.yaml', 'learning_rate: 1\n')
        self.save_manifest(self.repaired, self.repaired_manifest, '2'*40)
        with self.assertRaisesRegex(ValueError, 'unapproved production'):
            self.seal()

    def test_stale_manifest_or_untracked_python_is_rejected(self):
        self.seal()
        self.write_source(self.repaired, 'training/injected.py', 'bad = 1\n')
        with self.assertRaisesRegex(ValueError, 'Unmanifested Python'):
            self.verify()
        (self.repaired/'training/injected.py').unlink()
        self.write_source(self.repaired, 'training/native.py', 'changed = 1\n')
        with self.assertRaisesRegex(ValueError, 'Staged source changed'):
            self.verify()

    def test_original_plan_modified_and_resealed_is_still_rejected(self):
        self.seal()
        changed = copy.deepcopy(self.plan)
        changed['scientific_setting']['learning_rate'] = 1.
        changed['binding'] = amendment.digest({k:v for k,v in changed.items() if k!='binding'})
        runtime.write_json(self.plan_path, changed)
        with self.assertRaisesRegex(ValueError, 'unchanged original plan'):
            self.verify()

    def test_amendment_identity_and_import_root_are_checked(self):
        self.seal()
        with self.assertRaisesRegex(ValueError, 'amendment identity'):
            self.verify(expected_sha256='0'*64)
        with self.assertRaisesRegex(ValueError, 'not imported'):
            amendment.verify_amendment(self.plan_path, self.path)

    def test_cpu_recovery_must_be_zero_model_work_and_unchanged(self):
        runtime.write_json(self.cpu, dict(status='completed', new_model_work=1))
        with self.assertRaisesRegex(ValueError, 'CPU-only'):
            self.seal()

    def test_cpu_recovery_must_bind_the_same_plan(self):
        receipt = runtime.read(self.cpu)
        receipt['plan_binding'] = '0'*64
        runtime.write_json(self.cpu, receipt)
        with self.assertRaisesRegex(ValueError, 'another scientific plan'):
            self.seal()

    def test_prior_failed_status_preserved_before_mutation(self):
        self.seal()
        source = self.verify()
        root = Path(self.plan['root'])
        runtime.initialize(root)
        failure = dict(status='blocked', error='metadata identity changed', execution_commit='1'*40)
        runtime.write_json(root/'STATUS.json', failure)
        runtime.write_json(root/'qualification/QUALIFICATION.json', dict(status='passed', execution_commit='1'*40))
        runtime.write_json(root/'campaign_001/arms/BASE/STATUS.json', dict(status='failed', phase='idle_boundary', events_completed=1))
        before = (root/'STATUS.json').read_bytes()
        saved = amendment.preserve_resume(self.plan, source)
        runtime.write_json(root/'STATUS.json', dict(status='running', **source))
        receipt = runtime.read(saved['path'])
        previous = Path(saved['path']).parent/'prior/STATUS.json'
        self.assertEqual(previous.read_bytes(), before)
        self.assertEqual(receipt['actual_execution_commit'], '2'*40)
        self.assertEqual(receipt['new_model_work'], 0)
        self.assertEqual(receipt['original_model_load_epoch'], runtime.read(root/'CLOCK.json')['first_model_load_epoch'])

    def test_child_uses_actual_source_and_persists_both_commits(self):
        self.seal()
        source = self.verify()
        root = Path(self.plan['root'])
        runtime.initialize(root)
        (root/'logs').mkdir()
        captured = {}
        def failed_start(command, **kwargs):
            captured.update(command=command, **kwargs)
            raise OSError('fixture start failure')
        environment = {'PATH':'/bin','PYTHONPATH':str(self.original)+':'+str(self.original/'training')+':/unchanged'}
        with patch.object(launch, 'gpu_inventory', return_value={'processes': []}), \
             patch.object(launch, 'child_environment', return_value=environment), \
             patch.object(launch.subprocess, 'Popen', side_effect=failed_start):
            with self.assertRaisesRegex(OSError, 'fixture'):
                launch.child(self.plan_path, self.plan, 'train', arm='BASE', resume=True, attempt=2, source=source)
        self.assertEqual(captured['cwd'], str(self.repaired))
        self.assertTrue(captured['env']['PYTHONPATH'].startswith(str(self.repaired)+':'))
        self.assertNotIn(str(self.original), captured['env']['PYTHONPATH'])
        self.assertIn('--source-amendment-sha256', captured['command'])
        record = runtime.read(root/'processes/train_BASE_002.json')
        self.assertEqual(record['execution_commit'], '1'*40)
        self.assertEqual(record['actual_execution_commit'], '2'*40)
        self.assertEqual(record['source_amendment_identity'], amendment.identity(self.path))
        self.assertFalse(runtime.read(root/'CLOCK.json')['active'])

    def test_snapshot_supports_legacy_unfinished_qualification(self):
        self.seal()
        source = self.verify()
        root = Path(self.plan['root'])
        runtime.initialize(root)
        runtime.write_json(root/'STATUS.json', dict(status='blocked'))
        saved = amendment.preserve_resume(self.plan, source)
        files = runtime.read(saved['path'])['prior_files']
        self.assertEqual(set(files), {'STATUS.json', 'CLOCK.json', 'WORK.json'})

    def test_entry_records_actual_commit_without_changing_plan(self):
        self.seal()
        source = self.verify()
        root = Path(self.plan['root'])
        original = self.plan_path.read_bytes()
        with patch.object(entry, 'execution_source', return_value=source), \
             patch.object(entry, 'boundary', return_value={}), \
             patch('experiments.continuation_repair_fullrun.serving.run', return_value={'status':'completed'}):
            result = entry.run(self.plan, 'evaluate', plan_path=self.plan_path,
                source_amendment=str(self.path), source_amendment_sha256=source['source_amendment_identity']['sha256'])
        self.assertEqual(result['actual_execution_commit'], '2'*40)
        self.assertEqual(result['execution_commit'], '1'*40)
        self.assertEqual(self.plan_path.read_bytes(), original)
        self.assertEqual(runtime.read(root/'phases/evaluate_001/STATUS.json')['source_amendment_identity'], amendment.identity(self.path))

    def test_amended_entry_cannot_repeat_qualification(self):
        self.seal()
        with patch.object(entry, 'execution_source', return_value=self.verify()):
            with self.assertRaisesRegex(ValueError, 'cannot repeat qualification'):
                entry.run(self.plan, 'qualification_training', plan_path=self.plan_path,
                    source_amendment=str(self.path))


if __name__ == '__main__':
    unittest.main()
