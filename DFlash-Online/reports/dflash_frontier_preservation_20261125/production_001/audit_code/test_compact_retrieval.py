import json
from pathlib import Path
import tempfile
import unittest

import compact_retrieval as compact


class CompactRetrievalTests(unittest.TestCase):
    def test_fixed_coverage_excludes_all_tensor_suffixes(self):
        paths = compact.required_paths()
        self.assertEqual(
            len([p for p in paths if p.startswith('evaluation/raw/')]), 792
        )
        self.assertEqual(len([p for p in paths if '/allocation_' in p]), 256)
        self.assertIn('witness/BINDINGS.json', paths)
        for path in paths | compact.supplementary_paths():
            self.assertFalse(path.startswith('/'))
            self.assertNotIn('..', Path(path).parts)
            self.assertNotIn(Path(path).suffix, ('.pt', '.bin', '.safetensors'))

    def test_terminal_sparse_inventory_retains_missing_status_and_ignores_bulk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sealed').mkdir()
            (root / 'sealed/PLAN.json').write_text(
                json.dumps({'execution_commit': compact.COMMIT})
            )
            (root / 'LAUNCH.json').write_text('{"status":"running"}')
            (root / 'PROCESS_EXIT_CODE').write_text('1\n')
            (root / 'model.safetensors').write_bytes(b'not a model')
            with self.assertRaisesRegex(ValueError, 'closure'):
                compact.inventory(root)
            (root / 'LAUNCH.json').write_text('{"status":"failed"}')
            result = compact.inventory(root)
            self.assertEqual(result['status'], 'incomplete_file_coverage')
            self.assertIn('witness/parent.json', result['missing_required'])
            self.assertNotIn(
                'model.safetensors', [row['path'] for row in result['files']]
            )

    def test_symlink_and_oversized_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'real.json').write_text('{}')
            (root / 'link.json').symlink_to(root / 'real.json')
            with self.assertRaisesRegex(ValueError, 'Symlink'):
                compact.inspect_file(root, 'link.json')
            with (root / 'large.json').open('wb') as stream:
                stream.truncate(compact.MAX_FILE_BYTES + 1)
            with self.assertRaisesRegex(ValueError, 'bounded'):
                compact.inspect_file(root, 'large.json')


if __name__ == '__main__':
    unittest.main()
