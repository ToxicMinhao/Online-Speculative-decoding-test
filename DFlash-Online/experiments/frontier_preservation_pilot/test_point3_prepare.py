"""CPU admission and immutable-source checks; no GPU or large tensor loading."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import point3_prepare as P


def resources():
    return dict(
        filesystems=[dict(device='a800-disk', role='a800_original',
                          free_bytes=430_000_000_000, remaining_peak_bytes=20_000_000_000,
                          reserve_bytes=400_000_000_000),
                     dict(device='fenrir-disk', role='local_delivery',
                          free_bytes=70_000_000_000, remaining_peak_bytes=600_000_000,
                          reserve_bytes=20_000_000_000)],
        all_locations_peak_bytes=20_600_000_000,
        host_available_bytes=100_000_000_000,
        gpu_free_bytes=80_000_000_000,
        gpu_busy=False, forecast_s=14000., occupied_forecast_s=14000.,
    )


class Point3AdmissionTests(unittest.TestCase):
    def test_contract_has_full_event_and_three_arm_budgets(self):
        plan = P.prospective_contract()
        self.assertEqual(plan['cases'], ['C', 'F_C_SUM', 'F_C_MAX'])
        self.assertEqual(plan['committed_adam_calls'], 256)
        self.assertEqual(plan['committed_presentations'], 512)
        self.assertEqual(plan['limits']['attempted_adam_calls'], 262)
        self.assertEqual(plan['limits']['allocated_bytes'], 25_000_000_000)
        self.assertEqual(plan['limits']['elapsed_s'], 21600)
        self.assertEqual(plan['limits']['occupied_card_s'], 14400)
        self.assertEqual(plan['planned_qualification_adam_calls'], 3)
        self.assertEqual(plan['planned_oracle_backwards'], 28)
        self.assertEqual(plan['evaluation']['total_generations'], 1188)
        self.assertEqual(plan['evaluation']['warmup_generations'], 36)
        self.assertEqual(plan['evaluation']['scored_generations'], 1152)
        self.assertFalse(plan['inherited_deletion_allowed'])
        self.assertFalse(plan['point2b_executed'])

    def test_forecast_charges_both_complete_recoveries_and_exports(self):
        forecast = P.forecast()
        storage = forecast['storage']
        self.assertEqual(storage['endpoints_bytes'], 2 * (6_400_000_000 + 2_100_000_000))
        self.assertGreaterEqual(storage['remote_peak_bytes'], storage['endpoints_bytes'])
        self.assertEqual(storage['all_locations_peak_bytes'],
                         storage['remote_peak_bytes'] + storage['local_allowance_bytes'])
        self.assertLessEqual(storage['all_locations_peak_bytes'], P.LIMITS['allocated_bytes'])
        self.assertEqual(forecast['forecast_s'], forecast['raw_s'] * 1.25)
        self.assertLessEqual(forecast['forecast_s'], P.LIMITS['occupied_card_s'])

    def test_admission_requires_all_resources_and_full_peak(self):
        result = P.resource_admission(**resources())
        self.assertTrue(result['admitted'])
        self.assertEqual(result['filesystems'][0]['projected_free_bytes'], 410_000_000_000)
        self.assertEqual(result['filesystems'][0]['deficit_bytes'], 0)

    def test_current_a800_capacity_cannot_admit_full_campaign(self):
        values = resources()
        values['filesystems'][0]['free_bytes'] = 407_182_495_744
        result = P.resource_admission(**values)
        self.assertFalse(result['admitted'])
        self.assertFalse(result['checks']['filesystems'])
        self.assertEqual(result['filesystems'][0]['deficit_bytes'], 12_817_504_256)

    def test_duplicate_physical_devices_do_not_count_as_more_storage(self):
        values = resources()
        values['filesystems'][1]['device'] = values['filesystems'][0]['device']
        with self.assertRaisesRegex(ValueError, 'physical filesystem'):
            P.resource_admission(**values)

    def test_empty_or_negative_storage_inventory_is_rejected(self):
        values = resources()
        values['filesystems'] = []
        with self.assertRaises(ValueError):
            P.resource_admission(**values)
        values = resources()
        values['filesystems'][0]['remaining_peak_bytes'] = -1
        with self.assertRaisesRegex(ValueError, 'Negative'):
            P.resource_admission(**values)

    def test_cap_counts_all_locations_not_only_a800(self):
        values = resources()
        values['all_locations_peak_bytes'] = 25_000_000_001
        result = P.resource_admission(**values)
        self.assertFalse(result['admitted'])
        self.assertFalse(result['checks']['storage_cap'])

    def test_reserve_cannot_be_lowered_or_original_filesystem_omitted(self):
        for modification in ('reserve', 'role'):
            values = resources()
            if modification == 'reserve':
                values['filesystems'][0]['reserve_bytes'] -= 1
            else:
                values['filesystems'][0]['role'] = 'new_directory_same_disk'
            with self.assertRaisesRegex(ValueError, '400GB reserve'):
                P.resource_admission(**values)

    def test_total_peak_cannot_omit_other_filesystems_or_overlap(self):
        values = resources()
        values['all_locations_peak_bytes'] = 20_000_000_000
        with self.assertRaisesRegex(ValueError, 'omits planned filesystem writes'):
            P.resource_admission(**values)

    def test_host_reserve_and_training_vram_are_independent(self):
        for key, value, check in (
            ('host_available_bytes', 86_999_999_999, 'host_memory'),
            ('gpu_free_bytes', 59_999_999_999, 'gpu_memory'),
            ('gpu_busy', True, 'gpu_idle'),
        ):
            with self.subTest(key=key):
                values = resources()
                values[key] = value
                result = P.resource_admission(**values)
                self.assertFalse(result['admitted'])
                self.assertFalse(result['checks'][check])

    def test_elapsed_and_card_caps_are_not_interchangeable(self):
        for key, value, check in (
            ('forecast_s', 21601., 'elapsed'),
            ('occupied_forecast_s', 14401., 'card_time'),
            ('forecast_s', 0., 'elapsed'),
        ):
            with self.subTest(key=key):
                values = resources()
                values[key] = value
                result = P.resource_admission(**values)
                self.assertFalse(result['admitted'])
                self.assertFalse(result['checks'][check])

    def test_preparation_stops_before_any_input_or_output_when_not_admitted(self):
        with TemporaryDirectory() as folder:
            admission = Path(folder) / 'admission.json'
            values = resources()
            values['filesystems'][0]['free_bytes'] = 407_182_495_744
            admission.write_text(json.dumps(P.resource_admission(**values)))
            output = Path(folder) / 'never_created'
            with patch.object(P, 'authenticated_inputs') as authenticate:
                with self.assertRaisesRegex(RuntimeError, 'must be admitted'):
                    P.prepare(production_plan='unused', point2a_root='unused',
                              archive_manifest='unused', root=output, source_manifest='unused',
                              execution_commit='unused', admission_path=admission)
                authenticate.assert_not_called()
            self.assertFalse(output.exists())

    def test_preparation_recomputes_a_claimed_pass(self):
        with TemporaryDirectory() as folder:
            values = resources()
            values['filesystems'][0]['free_bytes'] = 407_182_495_744
            result = P.resource_admission(**values)
            result['admitted'] = True
            path = Path(folder) / 'admission.json'
            path.write_text(json.dumps(result))
            with patch.object(P, 'authenticated_inputs') as authenticate:
                with self.assertRaisesRegex(ValueError, 'does not match its inputs'):
                    P.prepare(production_plan='unused', point2a_root='unused',
                              archive_manifest='unused', root=Path(folder)/'output',
                              source_manifest='unused', execution_commit='unused', admission_path=path)
                authenticate.assert_not_called()


def authentication_fixture():
    """Small stand-ins test mandatory file checks, not historical correctness."""
    marker = dict(bytes=1, sha256='test-only')
    receipts = {name: marker for name in (
        'CONTRACT.json', 'attempt_001/native/C.json', 'attempt_001/native/F_C.json')}
    previous = dict(parent_recovery='/parent/recovery', original_window='/original/window',
                    parent_features=[])
    production = dict(root='/production', parent_export='/parent/export', sealed_files={})
    entries = {name: marker for name in ('shared/STATUS.json', 'arms/CONTROL/STATUS.json')}
    archive = dict(files=[dict(path=name, **value) for name, value in entries.items()])
    shared = dict(status='completed', completed=128, source_indices=list(range(128, 256)))
    features = [dict(source_index=index, path=f'/features/{index}', identity=marker)
                for index in range(128, 256)]
    control = dict(status='completed', global_step=256, presentations=512, completed_event=2,
                   recovery='/control/recovery', checkpoint='/control/export')
    bindings = dict(sets=dict(parent_training=[
        dict(feature_path=f'/witness/{index}', feature_identity=marker) for index in range(8)]))
    witness = dict(binding_path='/witness/bindings', binding_identity=marker)
    records = {
        '/point2a/CONTRACT.json': previous, '/production-plan': production,
        '/archive': archive, '/production/shared/STATUS.json': shared,
        '/production/shared/FEATURE_IDENTITIES.json': features,
        '/production/arms/CONTROL/STATUS.json': control, '/original/window': {},
        '/witness/bindings': bindings,
        '/point2a/attempt_001/native/C.json': {}, '/point2a/attempt_001/native/F_C.json': {},
    }
    files = {
        '/parent/recovery': P.RECOVERY_IDENTITY,
        '/parent/export/model.safetensors': P.EXPORT_IDENTITY,
        '/control/recovery': P.CONTROL_RECOVERY,
        '/control/export/model.safetensors': P.CONTROL_EXPORT,
        **{row['path']: row['identity'] for row in features},
        **{row['feature_path']: row['feature_identity'] for row in bindings['sets']['parent_training']},
    }
    return dict(public=dict(raw_input_receipts=receipts), records=records, files=files,
                witness=witness)


class Point3InputTests(unittest.TestCase):
    def authenticate(self, fixture):
        with patch.object(P.Path, 'read_text', return_value=json.dumps(fixture['public'])), \
             patch.object(P, 'read_bound_json', side_effect=lambda path, expected:
                          deepcopy(fixture['records'][str(path)])) as read, \
             patch.object(P, 'identity', side_effect=lambda path: fixture['files'][str(path)]) as identify, \
             patch.object(P, 'authenticate_witness', return_value=fixture['witness']):
            result = P.authenticated_inputs('/production-plan', '/point2a', '/archive')
            return result, read.call_args_list, identify.call_args_list

    def test_all_shared_features_both_endpoint_tensors_and_witnesses_are_checked(self):
        result, reads, identities = self.authenticate(authentication_fixture())
        checked = {str(call.args[0]) for call in identities}
        self.assertEqual(len(checked), 4 + 128 + 8)
        self.assertEqual(len(result['features']), 128)
        self.assertEqual(set(result['historical_controls']), {'C', 'F_C_SUM'})
        self.assertIn('/control/recovery', checked)
        self.assertIn('/control/export/model.safetensors', checked)
        self.assertIn('/features/255', checked)
        receipt_reads = {str(call.args[0]) for call in reads}
        self.assertIn('/point2a/attempt_001/native/F_C.json', receipt_reads)

    def test_wrong_buffer_range_or_incomplete_control_stops(self):
        for record, key, value in (
            ('/production/shared/STATUS.json', 'source_indices', list(range(127, 255))),
            ('/production/arms/CONTROL/STATUS.json', 'global_step', 255),
            ('/production/arms/CONTROL/STATUS.json', 'status', 'failed'),
        ):
            with self.subTest(record=record, key=key):
                fixture = authentication_fixture()
                fixture['records'][record][key] = value
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    self.authenticate(fixture)

    def test_changed_parent_control_feature_or_witness_is_not_substituted(self):
        for path in ('/parent/recovery', '/control/export/model.safetensors',
                     '/features/200', '/witness/0'):
            with self.subTest(path=path):
                fixture = authentication_fixture()
                fixture['files'][path] = dict(bytes=99, sha256='changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    self.authenticate(fixture)

    def test_sealed_config_canonical_or_manifest_changes_stop_before_loading(self):
        for path in ('/sealed/config.yaml', '/sealed/canonicals.json', '/sealed/INPUTS.json'):
            with self.subTest(path=path):
                fixture = authentication_fixture()
                fixture['records']['/production-plan']['sealed_files'][path] = dict(
                    bytes=42, sha256='expected')
                fixture['files'][path] = dict(bytes=43, sha256='modified')
                with self.assertRaisesRegex(ValueError, 'Retained production input changed'):
                    self.authenticate(fixture)


class Point3PreparationBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.receipt = dict(schema='point3_disposable_retirement_v1', status='completed',
            authorization='User authorized this scoped retirement',
            candidates=[dict(path='/old/disposable/recovery.pt')],
            removed=['/old/disposable/recovery.pt'],
            after_free_bytes=423_960_395_776, freed_allocated_bytes=16_778_248_192,
            lost_capability='Two old disposable qualification endpoints cannot be reloaded')
        self.receipt_path = self.root/'CLEANUP_RECEIPT.json'
        self.receipt_path.write_text(json.dumps(self.receipt))

    def test_cleanup_receipt_binds_only_completed_authorized_retirement(self):
        result = P.cleanup_binding(self.receipt_path)
        self.assertEqual(result['identity'], P.identity(self.receipt_path))
        self.assertEqual(result['removed_count'], 1)
        self.assertEqual(result['lost_capability'], self.receipt['lost_capability'])
        self.assertFalse(result['further_cleanup_authorized'])

    def test_incomplete_or_unaccounted_cleanup_cannot_enter_plan(self):
        for key, value in (('status', 'validated_before_unlink'), ('removed', []),
                           ('authorization', ''), ('lost_capability', ''),
                           ('after_free_bytes', 399_999_999_999)):
            with self.subTest(key=key):
                receipt = dict(self.receipt, **{key: value})
                self.receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, 'cleanup receipt'):
                    P.cleanup_binding(self.receipt_path)

    def test_prepare_seals_fresh_cache_baseline_after_cleanup(self):
        admission = self.root/'ADMISSION.json'
        admission.write_text(json.dumps(P.resource_admission(**resources())))
        config = self.root/'control.yaml'
        config.write_text('run_id: old\noutput_dir: old\n')
        previous = {key: 'old-'+key for key in (
            'parent_plan', 'parent_plan_identity', 'parent_recovery', 'parent_export',
            'parent_features', 'original_probe', 'original_window', 'devices', 'training_python')}
        previous['compiler_baseline'] = dict(directories={'cache':'/old-cache'}, allocated_bytes=9000)
        production = {key: 'retained-'+key for key in (
            'manifest', 'canonicals', 'warmup_canonicals', 'canonical_provenance',
            'target', 'draft', 'worker_python', 'serving', 'model_identities', 'original_config')}
        production.update(root='/retained-production', configs={'CONTROL':str(config)})
        inputs = dict(production=production, previous=previous, historical_controls={},
            witness={}, features=[], retained_C_status='/retained/status',
            retained_C_status_identity={}, control=dict(training={}, checkpoint='/retained/export'))
        source_manifest = self.root/'SOURCE.json'
        source_manifest.write_text('{}')
        current = dict(directories={'cache':'/current-cache'}, allocated_bytes=1000)
        with patch.object(P, 'authenticated_inputs', return_value=inputs), \
             patch.object(P, 'compiler_snapshot', return_value=current) as snapshot, \
             patch('experiments.frontier_preservation_pilot.qualify.verify_source'):
            plan = P.prepare(production_plan='/retained-plan', point2a_root='/point2a',
                archive_manifest='/archive', root=self.root/'output',
                source_manifest=source_manifest, execution_commit='published',
                admission_path=admission, cleanup_receipt_path=self.receipt_path)
        snapshot.assert_called_once_with()
        self.assertEqual(plan['compiler_baseline'], current)
        self.assertEqual(plan['historical_compiler_baseline'], previous['compiler_baseline'])
        self.assertEqual(plan['prelaunch_cleanup']['identity'], P.identity(self.receipt_path))
        self.assertFalse(plan['inherited_deletion_allowed'])
        self.assertEqual(json.loads((self.root/'output/CONTRACT.json').read_text()), plan)


if __name__ == '__main__':
    unittest.main()
