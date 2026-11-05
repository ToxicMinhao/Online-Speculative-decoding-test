"""Selection is fixed before reading historical outcomes; no prompt generation."""

import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from experiments.osd_components.prepare import select_panel, verify_artifacts
from experiments.osd_components.run import load_prepared
from test_osd_components_run import Backend, request


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.indices = sorted(random.Random(20261024).sample(range(1024), 64))
        candidates = [dict(request_id=f'q{i}', dataset_id=f'id{i}', request_index=i + 1,
                           prompt_ids=[1, 2], max_new_tokens=2048, collect_feedback=False)
                      for i in range(1024)]
        self.manifest = {'eval_requests': candidates}
        backend = Backend([request(0)])
        for index in self.indices:
            for repeat in range(3):
                for method in ('F', 'O'):
                    backend.checkpoint_id = '/' + method
                    backend.generate([1, 2], request_index=0, purpose='evaluation',
                                     save_to=self.root / f'{index:04d}_r{repeat}_{method}.json')

    def select(self, manifest=None):
        return select_panel(manifest or self.manifest, self.root, {'F': '/F', 'O': '/O'})

    def test_fixed_sorted_indices_and_no_global_rng_mutation(self):
        before = random.getstate()
        panel = self.select()
        self.assertEqual(before, random.getstate())
        self.assertEqual([r['original_index'] for r in panel], self.indices)
        self.assertEqual([r['panel_ordinal'] for r in panel], list(range(64)))
        self.assertEqual([r['request_id'] for r in panel], [f'q{i}' for i in self.indices])

    def test_missing_selected_reference_fails_without_replacement(self):
        (self.root / f'{self.indices[0]:04d}_r0_O.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.select()

    def test_historical_repeat_count_discrepancy_is_not_ignored(self):
        path = self.root / f'{self.indices[0]:04d}_r2_O.json'
        record = json.loads(path.read_text())
        record['metadata']['verification_calls'] += 1
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'baseline record'):
            self.select()

    def test_prompt_cap_or_feedback_policy_cannot_silently_change(self):
        for field, value in [('prompt_ids', [123]), ('max_new_tokens', 1024), ('collect_feedback', True)]:
            manifest = copy.deepcopy(self.manifest)
            manifest['eval_requests'][self.indices[0]][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Saved input/cap'):
                self.select(manifest)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.plan = {'arms': {}, 'target': str(self.root / 'target')}
        identities = {'arms': {}, 'target': {}, 'experiment_source': {}}
        for name in ('F', 'O', 'C-only', 'B-only', 'target', 'code'):
            directory = self.root / name
            directory.mkdir()
            file = directory / 'artifact.bin'
            file.write_bytes(name.encode())
            files = {'artifact.bin': {'bytes': file.stat().st_size,
                                     'sha256': hashlib.sha256(file.read_bytes()).hexdigest()}}
            if name == 'code':
                identities['experiment_source'] = files
            elif name == 'target':
                identities['target'] = {'path': str(directory), 'files': files}
            else:
                self.plan['arms'][name] = str(directory)
                identities['arms'][name] = {'path': str(directory), 'files': files}
        (self.root / 'ARTIFACT_IDENTITIES.json').write_text(json.dumps(identities))

    def check(self):
        verify_artifacts(self.root, self.plan, self.root / 'code')

    def test_checks_actual_files_for_every_arm_target_and_source(self):
        self.check()
        for name in ('F', 'O', 'C-only', 'B-only', 'target', 'code'):
            file = self.root / name / 'artifact.bin'
            original = file.read_bytes()
            file.write_bytes(b'x' * len(original))
            with self.subTest(artifact=name), self.assertRaisesRegex(ValueError, 'artifact changed'):
                self.check()
            file.write_bytes(original)

    def test_added_model_loader_file_is_rejected(self):
        (self.root / 'target' / 'generation_config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'file set changed'):
            self.check()

    def test_repointing_arm_does_not_bypass_identity_check(self):
        self.plan['arms']['C-only'] = self.plan['arms']['F']
        with self.assertRaisesRegex(ValueError, 'path differs'):
            self.check()

    def test_changed_construction_receipt_fails_input_seal(self):
        receipt = self.root / 'CHECKPOINT_QUALIFICATION.json'
        receipt.write_text('{"status":"passed"}')
        (self.root / 'INPUT_SEAL.json').write_text(json.dumps({
            receipt.name: hashlib.sha256(receipt.read_bytes()).hexdigest()}))
        receipt.write_text('{"status":"changed"}')
        with self.assertRaisesRegex(ValueError, 'Prepared input changed'):
            load_prepared(self.root)


if __name__ == '__main__':
    unittest.main()
