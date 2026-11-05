"""CPU checks for exact historical-source reconstruction."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import archive


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b'def preserved():\n    return "unchanged"\n'
        self.digest = hashlib.sha256(self.data).hexdigest()
        (self.root / 'objects').mkdir()
        (self.root / 'objects' / self.digest).write_bytes(self.data)
        self.pin = {'source_path': '/data/sources/minhao/DFlash-Online/online_feedback/test.py',
                    'sha256': self.digest, 'object': f'objects/{self.digest}'}
        self.contract = {'retained_manifest': 'original/SEAL.json', 'retained_manifest_sha256': '0' * 64,
                         'pins': [self.pin]}
        (self.root / 'run.json').write_text(json.dumps({'contracts': [self.contract]}))
        (self.root / 'INDEX.json').write_text(json.dumps({'runs': [{'run_id': 'run', 'manifest': 'run.json'}],
                                                       'missing_exact_bytes': []}))
        (self.root / 'OBJECTS.json').write_text(json.dumps({self.digest: {'bytes': len(self.data)}}))

    def test_exact_extraction_and_receipt(self):
        target = self.root / 'extracted'
        receipt = archive.extract(self.root, 'run', 'original/SEAL.json', target)
        self.assertEqual((target / 'online_feedback/test.py').read_bytes(), self.data)
        self.assertEqual(receipt['files'], {'online_feedback/test.py': self.digest})
        self.assertEqual(archive.verify(self.root)['verified_objects'], 1)

    def test_corrupt_object_prevents_extraction(self):
        (self.root / 'objects' / self.digest).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            archive.extract(self.root, 'run', 'original/SEAL.json', self.root / 'extracted')
        self.assertFalse((self.root / 'extracted').exists())

    def test_missing_source_is_explicit(self):
        self.pin['object'] = None
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            archive.selected_sources(self.contract, archive.DEFAULT_PREFIX)

    def test_different_versions_are_not_merged(self):
        self.contract['pins'].append({**self.pin, 'sha256': '1' * 64})
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            archive.selected_sources(self.contract, archive.DEFAULT_PREFIX)

    def test_parent_escape_is_rejected(self):
        self.pin['source_path'] = '../outside.py'
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            archive.selected_sources(self.contract, archive.DEFAULT_PREFIX)

    def test_unrelated_absolute_paths_are_not_extracted(self):
        self.contract['pins'].append({**self.pin, 'source_path': '/other/runtime.py'})
        self.assertEqual(len(archive.selected_sources(self.contract, archive.DEFAULT_PREFIX)), 1)

    def test_existing_destination_is_preserved(self):
        target = self.root / 'existing'
        target.mkdir()
        with self.assertRaises(FileExistsError):
            archive.extract(self.root, 'run', 'original/SEAL.json', target)

    def test_readable_copy_must_match_object(self):
        (self.root / 'readable.py').write_bytes(b'changed')
        manifest = {'contracts': [self.contract], 'run_tools': [
            {'published_path': 'readable.py', 'sha256': self.digest}]}
        (self.root / 'run.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Readable source copy differs'):
            archive.verify(self.root)


if __name__ == '__main__':
    unittest.main()
