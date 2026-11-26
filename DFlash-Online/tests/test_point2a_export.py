"""Offline Point2A arithmetic, partial failures and private-content exclusion."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.frontier_preservation_pilot.point2a_export import export, write_json
from test_point1_export import readout


def source(root, witness=True):
    native = root / 'attempt_001/native'
    native.mkdir(parents=True)
    plan = dict(schema='point2a_c_reference_v1', execution_commit='a' * 40, status='admitted',
        cases=['C', 'F_BETA', 'F_C'], witness=dict(
            status='authenticated' if witness else 'unavailable', set_name='new_buffer', responses=8,
            items=[dict(record_id='PRIVATE_RECORD_SENTINEL', anchors=[7654321],
                        feature_path='/PRIVATE_PATH_SENTINEL', feature_identity=dict(bytes=7, sha256='b' * 64))]),
        parent_features=[dict(record_id='PRIVATE_RECORD_SENTINEL', path='/PRIVATE_PATH_SENTINEL',
                              identity=dict(bytes=6, sha256='c' * 64))],
        parent_recovery_identity=dict(bytes=8, sha256='d' * 64), limits={'attempted_adam_calls': 6})
    write_json(root / 'CONTRACT.json', plan)
    write_json(native / 'RESULT.json', dict(status='completed', execution_commit='a' * 40,
        final_restoration=dict(status='passed', exact_resume=True), wall_s=3,
        startup=dict(status='completed', optimizer_steps=0, wall_s=1, record_id='PRIVATE_RECORD_SENTINEL')))
    write_json(root / 'attempt_001/PROCESS.json', dict(status='completed', elapsed_s=4))
    write_json(root / 'attempt_001/ADMISSION.json', dict(admitted=True))
    write_json(root / 'CLOCK.json', dict(completed_occupied_card_s=4))
    write_json(root / 'STEP_LEDGER.json', dict(schema='point2a_native_step_ledger_v1', calls=[],
        attempts=[dict(number=1, status='completed', correction_note='PRIVATE_NOTE_SENTINEL')],
        maximum_attempted_calls=6))
    write_json(native / 'WORK.json', dict(attempted={}, completed={}, pending=[], native_adam_steps=3))
    for name in ('BETA', 'C'):
        write_json(native / ('SCORING_' + name + '.json'), dict(
            reference_case=name, response_denominators=[2, 2],
            direction_identity=dict(sha256='e' * 64, tensors=[dict(name='PRIVATE_NAME_SENTINEL')]),
            score_arrays=[987654321], record_id='PRIVATE_RECORD_SENTINEL'))
    write_json(native / 'QUALIFICATION.json', dict(passed=True, selected_pool_receipts=[dict(
        coefficient_fidelity=True, coefficient_error=0.01, labels=[987654321],
        record_id='PRIVATE_RECORD_SENTINEL', feature_path='/PRIVATE_PATH_SENTINEL')]))
    write_json(native / 'REFERENCE_PROGRESS.json', dict(status='completed', completed=28,
        rows=[dict(response_index=int(i >= 13), block_index=0, position=i % 15 + 1,
                   primary_dot_fp64=-2.0, denominator=2, reference_score_fp64=0.5,
                   record_id='PRIVATE_RECORD_SENTINEL') for i in range(28)]))
    contrasts = {}
    for name, raw in (('F_C-F_BETA', 0.5), ('F_C-C', 0.75), ('F_BETA-C', 0.25)):
        contrasts[name] = dict(prediction=dict(
            raw=dict(total=raw, positive=raw + 1, negative=-1, absolute=raw+2),
            beta_centered=dict(total=raw, positive=raw+1, negative=-1, absolute=raw+2)),
            actual_gradient_contraction=dict(total=raw, positive=raw + 1, negative=-1),
            coefficient_equality=False, changed_positions=4,
            per_response=[dict(coefficient_movement=dict(changed_positions=4,
                per_pool=[dict(max_relative_movement=0.1, half_l1_mass_fraction=0.05)]))])
    write_json(native / 'CONTRASTS.json', contrasts)
    for name in ('C', 'F_BETA', 'F_C'):
        write_json(native / (name + '.json'), dict(name=name, status='completed', adam_attempts=1,
            adam_completed=1, rows=[dict(record_id='PRIVATE_RECORD_SENTINEL')],
            historical_reproduction=dict(status='passed', complete_recorded_scientific_identity=True),
            endpoint=dict(global_step=129)))
        for suffix in ('INITIAL_STATE', 'RESTORATION'):
            write_json(native / (name + '_' + suffix + '.json'),
                       dict(status='passed', exact_resume=True, mismatches=[]))
    for name in ('AFTER_STARTUP', 'AFTER_QUALIFICATION', 'FINAL_RESTORATION'):
        write_json(native / (name + '.json'), dict(status='passed', exact_resume=True, mismatches=[]))
    for endpoint, value in (('parent', 2), ('N_C', 1.5), ('N_F_BETA', 1.25), ('N_F_C', 1.0)):
        for scope, count in (('window', 2), ('witness', 8)):
            if scope == 'witness' and not witness:
                continue
            data = readout(endpoint, value)
            data['responses'] = [deepcopy(data['responses'][0]) for _ in range(count)]
            for index, row in enumerate(data['responses']):
                row['record_id'] = 'PRIVATE_RECORD_SENTINEL_' + str(index)
            write_json(native / (scope + '_' + endpoint + '.json'), data)
    return native


class Point2AExportTests(unittest.TestCase):
    def test_complete_export_all_three_contrasts_both_scopes_and_privacy(self):
        with TemporaryDirectory() as folder:
            root, output = Path(folder) / 'raw', Path(folder) / 'public'
            source(root)
            result = export(root, output)
            self.assertTrue(result['complete'])
            text = '\n'.join(path.read_text() for path in output.iterdir())
            for private in ('PRIVATE_RECORD_SENTINEL', 'PRIVATE_PATH_SENTINEL', 'PRIVATE_NOTE_SENTINEL',
                            'PRIVATE_NAME_SENTINEL', '987654321', '7654321'):
                self.assertNotIn(private, text)
            pairs = json.loads((output / 'READOUT_COMPARISONS.json').read_text())
            self.assertEqual(set(pairs), {'window', 'new_buffer'})
            for scope in pairs.values():
                self.assertEqual(set(scope), {'F_C-F_BETA', 'F_C-C', 'F_BETA-C'})
                self.assertEqual(scope['F_C-F_BETA']['assessment_benefit_fp64'], 0.375)
            with (output / 'CONTRASTS.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            score = next(row for row in rows if row['contrast'] == 'F_C-F_BETA' and row['stage'] == 'C_score_prediction')
            self.assertEqual(float(score['value']), 0.5)
            self.assertEqual(score['role'], 'primary')
            self.assertEqual(len([row for row in rows if row['stage'] == 'native_assessment_benefit']), 6)
            audit = json.loads((output / 'EXPORT_AUDIT.json').read_text())
            self.assertEqual(audit['readout_count'], 8)
            self.assertFalse(audit['missing_files'])
            manifest = json.loads((output / 'PUBLIC_FILES.json').read_text())
            for name, binding in manifest['files'].items():
                data = (output / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), binding['sha256'])
                self.assertEqual(len(data), binding['bytes'])

    def test_failed_attempt_exports_completed_work_and_missing_files(self):
        with TemporaryDirectory() as folder:
            root, output = Path(folder) / 'raw', Path(folder) / 'public'
            native = source(root)
            (native / 'window_N_F_C.json').unlink()
            (native / 'QUALIFICATION.json').unlink()
            write_json(native / 'RESULT.json', dict(status='failed', execution_commit='a' * 40,
                error='PRIVATE_RECORD_SENTINEL', traceback='/PRIVATE_PATH_SENTINEL'))
            write_json(root / 'attempt_001/PROCESS.json', dict(status='failed', elapsed_s=2))
            result = export(root, output)
            self.assertFalse(result['complete'])
            audit = json.loads((output / 'EXPORT_AUDIT.json').read_text())
            self.assertEqual(audit['status'], 'partial_receipts_exported')
            self.assertIn('attempt_001/native/window_N_F_C.json', audit['missing_files'])
            self.assertEqual(audit['contrasts_by_scope'], {'window': 1, 'new_buffer': 3})

    def test_missing_receipt_cannot_claim_complete_even_if_status_completed(self):
        with TemporaryDirectory() as folder:
            root, output = Path(folder) / 'raw', Path(folder) / 'public'
            native = source(root)
            (native / 'SCORING_C.json').unlink()
            self.assertFalse(export(root, output)['complete'])

    def test_missing_native_result_is_preserved_as_partial(self):
        with TemporaryDirectory() as folder:
            root, output = Path(folder) / 'raw', Path(folder) / 'public'
            native = source(root)
            (native / 'RESULT.json').unlink()
            self.assertFalse(export(root, output)['complete'])
            audit = json.loads((output / 'EXPORT_AUDIT.json').read_text())
            self.assertEqual(audit['native_status'], 'unavailable')

    def test_failed_gate_cannot_be_hidden_by_completed_status(self):
        with TemporaryDirectory() as folder:
            root, output = Path(folder) / 'raw', Path(folder) / 'public'
            native = source(root)
            write_json(native / 'QUALIFICATION.json', dict(status='failed', passed=False))
            self.assertFalse(export(root, output)['complete'])
            audit = json.loads((output / 'EXPORT_AUDIT.json').read_text())
            self.assertFalse(audit['required_gate_passes']['numerical'])

    def test_authenticated_witness_cannot_silently_change_response_count(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'raw'
            native = source(root)
            write_json(native / 'witness_N_F_C.json', readout('N_F_C'))
            with self.assertRaisesRegex(ValueError, 'response count'):
                export(root, Path(folder) / 'public')

    def test_unavailable_witness_stays_window_only_and_partial(self):
        with TemporaryDirectory() as folder:
            root, output = Path(folder) / 'raw', Path(folder) / 'public'
            source(root, witness=False)
            self.assertFalse(export(root, output)['complete'])
            audit = json.loads((output / 'EXPORT_AUDIT.json').read_text())
            self.assertEqual(audit['contrasts_by_scope'], {'window': 3, 'new_buffer': 0})
            self.assertTrue(audit['no_scientific_promotion'])

    def test_wrong_endpoint_label_is_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder) / 'raw'
            native = source(root)
            write_json(native / 'window_N_F_C.json', readout('N_F'))
            with self.assertRaisesRegex(ValueError, 'label'):
                export(root, Path(folder) / 'public')

    def test_publication_is_never_overwritten(self):
        with TemporaryDirectory() as folder:
            root, output = Path(folder) / 'raw', Path(folder) / 'public'
            source(root)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                export(root, output)


if __name__ == '__main__':
    unittest.main()
