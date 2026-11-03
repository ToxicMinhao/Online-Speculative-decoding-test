"""Synthetic receipt tests; no model, endpoint or production-result claims."""
import json
from pathlib import Path
import tempfile
import unittest

from summarize_cost_retention import ARMS, EXECUTION, markdown, summarize


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.put('sealed/PLAN.json', dict(events=1, execution_source={'commit': EXECUTION}, storage={'reserve_bytes': 80}))
        self.put('campaign_001/STATUS.json', dict(status='completed', events_completed=1,
            costs=dict(training_worker_wall_s=60, resource_check_s=2), campaign_wall_s=100,
            adaptation_ready_wall_s=80, resource_checks=[dict(free_bytes=1)]))
        self.put('PREFLIGHT.json', dict(status='passed', host_available_bytes=1000))
        self.put('POSTFLIGHT.json', dict(status='passed', storage=dict(free_bytes=300)))
        self.put('campaign_001/STORAGE_ADMISSION.json', dict(status='passed', free_bytes=400))
        self.put('campaign_001/final_evaluation/STATUS.json', dict(status='completed', evaluation_wall_s=15))
        self.put('campaign_001/events/event_001/COMPLETE.json', dict(status='passed', event_wall_s=70))
        self.put('campaign_001/events/event_001/SELECTION.json', dict(cost=dict(
            scoring_seconds=0.25, greedy_seconds=0.5, total_wall_seconds=1,
            array_working_bytes=4096, working_bytes_scope='persistent numeric arrays')))
        self.put('campaign_001/MEMORY.json', dict(completed_event=1, slots=[]))
        (self.root / 'campaign_001/RETIREMENTS').mkdir()
        self.resource_jsonl([dict(free_bytes=200, compiler_growth_bytes=20, compact_allocated_bytes=10,
                                  resource_check_wall_s=1),
                             dict(free_bytes=300, compiler_growth_bytes=30, compact_allocated_bytes=15,
                                  resource_check_wall_s=1)])
        for arm in ARMS:
            self.arm(arm)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def resource_jsonl(self, rows):
        (self.root / 'campaign_001/RESOURCE_CHECKS.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def arm(self, arm):
        base = f'campaign_001/arms/{arm}'
        initial = '/remote/campaign_001/arms/K/trainer/initial_state.pt' if arm == 'K' else f'/remote/{base}/trainer/initial_reference.json'
        self.put(f'{base}/INITIAL_WORKER.json', dict(status='passed', worker_wall_s=2, state_path=initial))
        self.put(f'{base}/initial_PROCESS.json', dict(returncode=0, wall_s=3))
        self.put(f'{base}/trainer/INITIAL_CHECK.json', dict(status='passed', state_path=initial))
        if arm != 'K':
            self.put(f'{base}/trainer/initial_reference.json', dict(state_path='/remote/campaign_001/arms/K/trainer/initial_state.pt', sha256='initialhash'))
        event = f'{base}/trainer/events/event_001'
        exposure = dict(presentations=128, replay_presentations=0, prompt_tokens=200,
                        response_tokens=400, active_anchors=500,
                        response_forward_backward_presentations=128, optimizer_steps=64)
        self.put(f'{event}/RESULT.json', dict(status='completed', export_status='deferred',
            costs=dict(training_s=8, checkpoint_save_s=1, export_s=0, event_wall_s_before_commit=14),
            runtime_costs=dict(construction_s=3, restore_s=2, construction_and_restore_s=5,
                               feature_admission_io_and_validation_s=1),
            pass_loop_costs=[dict(total_loop_s=5, input_load_and_loop_bookkeeping_s=1)] * 2,
            pass_exposure=[dict(exposure, pass_index=i) for i in (0, 1)]))
        self.put(f'{event}/EXPORT_CHECK.json', dict(status='passed', export_exact=True, export_and_verify_s=4))
        self.put(f'{base}/events/event_001/WORKER.json', dict(status='passed', worker_wall_s=20))
        self.put(f'{base}/event_001_PROCESS.json', dict(returncode=0, wall_s=21))
        self.put(f'{base}/events/event_001/RESOURCE_COSTS.json', dict(
            training_peak_allocated_bytes=100, training_peak_reserved_bytes=150,
            training_compiler_timers={'outer': dict(total_s=3, calls=1), 'inner': dict(total_s=2, calls=1)}))
        self.put(f'{base}/boundaries/event_001.json', dict(status='passed', event=1,
            state=f'/remote/{event}/training_state.pt', state_identity={'sha256': 'statehash', 'bytes': 300},
            export_status='materialized', checkpoint_path=f'/remote/{event}/export',
            weights_identity={'sha256': 'weighthash', 'bytes': 100}))

    def test_scopes_and_deferred_export_are_not_combined(self):
        result = summarize(self.root)
        k = result['training']['arms']['K']
        totals = k['additive_across_events']
        self.assertEqual(totals['training_s']['sum'], 8)
        self.assertEqual(totals['total_loop_s']['sum'], 10)
        self.assertEqual(totals['export_and_verify_s']['sum'], 4)
        self.assertEqual(totals['presentations']['sum'], 256)
        self.assertEqual(totals['optimizer_steps']['sum'], 128)
        self.assertEqual(set(k['compiler_timers_by_name']), {'outer', 'inner'})
        self.assertNotIn('total_compiler_s', k)
        self.assertEqual(result['costs']['campaign_wall_s'], 100)
        self.assertFalse(result['evidence']['issues'])
        self.assertFalse(result['evidence']['missing'])
        self.assertEqual(result['selection_costs']['sequential_sums']['scoring_seconds']['sum'], 0.25)
        self.assertEqual(result['selection_costs']['array_working_bytes']['max'], 4096)

    def test_resource_jsonl_has_priority_and_peak_scope_is_explicit(self):
        result = summarize(self.root)
        free = result['resources']['sampled_extrema']['free_bytes']
        self.assertEqual(free['min'], 200)  # STATUS duplicate has 1; it is ignored.
        self.assertEqual(free['observations'], 2)
        self.assertEqual(result['resources']['resource_check_timer']['sum'], 2)
        self.assertIn('excludes later verify_boundary', result['training']['allocator_scope'])
        self.assertIn('not continuous peaks', result['resources']['scope'])

    def test_missing_exposure_or_resource_is_unavailable_not_zero(self):
        path = self.root / 'campaign_001/arms/U/trainer/events/event_001/RESULT.json'
        value = json.loads(path.read_text())
        del value['pass_exposure'][1]['response_tokens']
        path.write_text(json.dumps(value))
        (self.root / 'campaign_001/arms/U/events/event_001/RESOURCE_COSTS.json').unlink()
        result = summarize(self.root)
        u = result['training']['arms']['U']
        self.assertIsNone(u['additive_across_events']['response_tokens']['sum'])
        self.assertIsNone(u['allocator_peaks']['allocator_peak_allocated_bytes']['max'])
        self.assertEqual(result['evidence']['completeness'], 'partial_receipt_coverage')

    def test_retirement_is_by_path_and_retention_is_only_receipt_evidence(self):
        for i in (1, 2):
            self.put(f'campaign_001/RETIREMENTS/row{i}_prepared.json', dict(
                status='retired', path=f'/remote/row{i}.pt', bytes=100, allocated_bytes=128,
                inode=5, hardlinks=2, reason='evicted', retained_evidence=['/remote/row.json'], archived=False))
        self.put('campaign_001/RETIREMENTS/row3_prepared.json', dict(status='prepared', path='/remote/row3.pt', bytes=100))
        result = summarize(self.root)
        retention = result['retention']
        self.assertEqual(retention['retired_receipts'], 2)
        self.assertEqual(retention['distinct_retired_paths'], 2)
        group = retention['categories']['prepared_packet']
        self.assertEqual(group['nominal_allocated_bytes']['sum'], 256)
        self.assertEqual(group['hardlinks_greater_than_one'], 2)
        self.assertEqual(len(retention['incomplete_or_other_receipts']), 1)
        self.assertNotIn('reclaimed_bytes', retention)
        self.assertTrue(all('not a fresh' in row['existence_proof'] for row in retention['retained']))
        self.assertIn('deleted, not archived', markdown(result))

    def test_retained_path_conflict_is_reported(self):
        path = '/remote/campaign_001/arms/K/trainer/events/event_001/training_state.pt'
        self.put('campaign_001/RETIREMENTS/event_002_K_predecessor_state.json',
                 dict(status='retired', path=path, bytes=100, allocated_bytes=128))
        result = summarize(self.root)
        self.assertTrue(any(row['retired_path_conflict'] for row in result['retention']['retained']))
        self.assertTrue(result['evidence']['issues'])

    def test_preparation_only_is_rejected(self):
        (self.root / 'campaign_001/STATUS.json').unlink()
        with self.assertRaisesRegex(ValueError, 'preparation alone'):
            summarize(self.root)

    def test_partial_next_event_uses_latest_commit_per_arm(self):
        base = 'campaign_001/arms/K'
        source = f'{base}/boundaries/event_002.json'
        state = f'/remote/{base}/trainer/events/event_002/training_state.pt'
        self.put(source, dict(status='passed', event=2, state=state,
                             state_identity={'sha256': 'newstate'}, export_status='deferred'))
        self.put('campaign_001/RETIREMENTS/event_002_K_predecessor_state.json', dict(
            status='retired', path=f'/remote/{base}/trainer/events/event_001/training_state.pt', bytes=300))
        self.put('campaign_001/RETIREMENTS/event_002_K_preceding_export.json', dict(
            status='retired', path=f'/remote/{base}/trainer/events/event_001/export/model.safetensors', bytes=100))
        result = summarize(self.root)
        retained = [row for row in result['retention']['retained'] if row.get('arm') == 'K']
        full = [row for row in retained if row['kind'] == 'last_committed_full_state']
        self.assertEqual(full[0]['path'], state)
        self.assertEqual(full[0]['event'], 2)
        self.assertFalse(any(row['kind'] == 'last_materialized_export' for row in retained))
        self.assertFalse(result['evidence']['issues'])

    def test_missing_retirement_directory_and_final_category_shortfalls_are_flagged(self):
        (self.root / 'campaign_001/RETIREMENTS').rmdir()
        self.put('sealed/PLAN.json', dict(events=8, execution_source={'commit': EXECUTION}))
        self.put('campaign_001/STATUS.json', dict(status='completed', events_completed=8))
        result = summarize(self.root)
        self.assertIn('campaign_001/RETIREMENTS/', result['evidence']['missing'])
        issues = [row for row in result['evidence']['issues'] if row.startswith('Retirement coverage')]
        self.assertEqual(len(issues), 5)
        self.assertEqual(sum(result['retention']['complete_protocol_expected_distinct_paths'].values()), 2986)


if __name__ == '__main__':
    unittest.main()
