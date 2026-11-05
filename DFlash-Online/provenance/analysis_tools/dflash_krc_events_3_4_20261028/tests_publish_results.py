"""CPU-only synthetic fixtures for the compact-publication boundary."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import publish_results as publish


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'run'
        self.source.mkdir()
        self.destination = self.root / 'staged'
        execution = {'execution_git_commit': publish.EXECUTION_COMMIT, 'dirty_files': [],
                     'repository': publish.REPOSITORY, 'working_source': '/private/host/source'}
        summary = {'source': execution, 'postprocessing': {'git_commit': publish.ANALYSIS_COMMIT,
                   'script_sha256': publish.BUILDER_SHA256}, 'completed_audited_endpoint': True,
                   'panels': {'development': {'O4': 2994, 'K4': 2942},
                              'additional': {'historical_O1': None}}, 'audit': {'issues': []}}
        for name in publish.ALLOWLIST:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith('.csv'):
                path.write_text('request_id,prompt_tokens,output_tokens,errors,record_path\n'
                                'fixture,183,2048,[],/private/host/record.json\n')
        (self.source / 'EXECUTION_SOURCE.json').write_text(json.dumps(execution))
        (self.source / 'SUMMARY.json').write_text(json.dumps(summary))
        (self.source / 'REPORT.md').write_text(
            '# Fixture only\n\nCalls 2994 / 2942; reduction +1.7368%; absent additional O1.\n'
            f'Execution commit: `{publish.EXECUTION_COMMIT}`. Postprocessing commit: `{publish.ANALYSIS_COMMIT}`; script hash retained.\n'
            + ', '.join(old for old, _ in publish.LINK_REPLACEMENTS)
            + ', [launch record](LAUNCH.json).\n')
        (self.source / 'raw').mkdir()
        (self.source / 'raw/PLAN.json').write_text(json.dumps({'prompt_ids': [101, 102], 'question': 'PRIVATE SENTINEL'}))

    def update_summary(self, update):
        path = self.source / 'SUMMARY.json'
        value = json.loads(path.read_text())
        update(value)
        path.write_text(json.dumps(value))

    def test_exact_allowlist_byte_preservation_and_reversible_report_changes(self):
        before = {name: (self.source / name).read_bytes() for name in publish.ALLOWLIST}
        manifest = publish.stage_results(self.source, self.destination)
        paths = {str(path.relative_to(self.destination)) for path in self.destination.rglob('*') if path.is_file()}
        self.assertEqual(paths, set(publish.ALLOWLIST) | {publish.MANIFEST})
        self.assertFalse((self.destination / 'raw').exists())
        self.assertEqual(manifest['missing_files'], [])
        for name, data in before.items():
            self.assertEqual((self.source / name).read_bytes(), data)
            public = (self.destination / name).read_bytes()
            self.assertEqual(manifest['files'][name]['original']['sha256'], hashlib.sha256(data).hexdigest())
            self.assertEqual(manifest['files'][name]['public']['sha256'], hashlib.sha256(public).hexdigest())
            self.assertNotIn(b'PRIVATE SENTINEL', public)
            if name != 'REPORT.md':
                self.assertEqual(public, data)
        recovered = (self.destination / 'REPORT.md').read_text()
        for transform in reversed(manifest['files']['REPORT.md']['transforms']):
            if transform['operation'] == 'append':
                self.assertTrue(recovered.endswith(transform['text']))
                recovered = recovered[:-len(transform['text'])]
            else:
                self.assertEqual(recovered.count(transform['new']), transform['count'])
                recovered = recovered.replace(transform['new'], transform['old'])
        self.assertEqual(recovered.encode(), before['REPORT.md'])
        self.assertEqual(manifest['execution_git_commit'], publish.EXECUTION_COMMIT)
        self.assertEqual(manifest['analysis_git_commit'], publish.ANALYSIS_COMMIT)
        self.assertIn(
            f'/blob/{publish.EXECUTION_COMMIT}/DFlash-Online/provenance/run_tools/'
            'dflash_krc_events_3_4_20261028/run_campaign.sh)',
            (self.destination / 'REPORT.md').read_text())

    def test_raw_fields_freeform_tokens_and_secrets_fail_before_staging(self):
        cases = ({'prompt_ids': [101]}, {'question': 'private question'},
                 {'error': "response_ids=[101, 102]"},
                 {'error': 'https://fixture:password@example.invalid'},
                 {'error': 'ghp_' + 'a' * 24})
        for payload in cases:
            with self.subTest(payload=tuple(payload)):
                self.update_summary(lambda value: value.update({'status': payload}))
                with self.assertRaises(ValueError):
                    publish.stage_results(self.source, self.destination)
                self.assertFalse(self.destination.exists())

    def test_mismatched_identities_and_unexpected_raw_link_fail(self):
        self.update_summary(lambda value: value['postprocessing'].update({'git_commit': publish.EXECUTION_COMMIT}))
        with self.assertRaisesRegex(ValueError, 'Analysis source'):
            publish.stage_results(self.source, self.destination)
        self.update_summary(lambda value: value['postprocessing'].update({'git_commit': publish.ANALYSIS_COMMIT}))
        report = self.source / 'REPORT.md'
        report.write_text(report.read_text() + '[extra](raw/private.json)\n')
        with self.assertRaisesRegex(ValueError, 'unrecognized private evidence link'):
            publish.stage_results(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_missing_partial_files_preserved_and_complete_claim_rejected(self):
        missing = 'analysis/additional/ALL_REPEATS.csv'
        (self.source / missing).unlink()
        with self.assertRaisesRegex(ValueError, 'completed endpoint'):
            publish.stage_results(self.source, self.destination)
        self.update_summary(lambda value: value.update({'completed_audited_endpoint': False}))
        manifest = publish.stage_results(self.source, self.destination)
        self.assertEqual(manifest['missing_files'], [missing])
        self.assertFalse((self.destination / missing).exists())
        self.assertFalse(json.loads((self.destination / 'SUMMARY.json').read_text())['completed_audited_endpoint'])

    def test_symlink_and_existing_destination_rejected_without_overwrite(self):
        target = self.source / 'TRAINING.csv'
        target.unlink()
        target.symlink_to(self.source / 'raw/PLAN.json')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            publish.stage_results(self.source, self.destination)
        self.destination.mkdir()
        sentinel = self.destination / 'keep.txt'
        sentinel.write_text('untouched')
        with self.assertRaises(FileExistsError):
            publish.stage_results(self.source, self.destination)
        self.assertEqual(sentinel.read_text(), 'untouched')


if __name__ == '__main__':
    unittest.main()
