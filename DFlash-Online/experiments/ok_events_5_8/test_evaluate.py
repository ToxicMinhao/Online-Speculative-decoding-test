"""CPU fake-backend coverage of the 1,920-slot endpoint evaluation contract."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.fresh_confirmation.test_evaluate import Backend
from test_osd_one_event import request as original_request
from test_prefix_ranking_evaluation import harmful
from .evaluate import evaluate
from .evaluation_protocol import method_order, validate_panels, TOTAL_GENERATIONS
from .report import summarize, write_campaign_report, curve_direction, validate_execution_source, execution_history


def requests(panel='development', count=3):
    values = [original_request(index) for index in range(count)]
    for row in values:
        if panel == 'fresh':
            row.pop('response_ids')
            row.pop('finish_reason')
        else:
            row['canonical_source_method'] = 'F' if panel == 'development' else 'K1'
    return values


def panels():
    values = []
    for offset, name in enumerate(('development', 'additional', 'fresh')):
        rows = requests(name, 64)
        for row in rows:
            row['request_id'] = name + '_' + row['request_id']
            row['prompt_ids'] = [1, offset*100+row['panel_ordinal']]
        value = dict(name=name, requests=rows,
                     canonical_mode='fresh' if name == 'fresh' else 'retained')
        if name == 'fresh':
            value['selection_seed'] = 20261029
        else:
            value['historical_counts'] = {arm: {row['request_id']: 2 for row in rows} for arm in ('O4', 'K4')}
        values.append(value)
    return values


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoints = {'O': '/O', 'K': '/K'}

    def run_panel(self, *, panel='development', event=6, count=3, mutation=None, name='evaluation', gate=1, rows=None):
        rows = requests(panel, count) if rows is None else rows
        backend = Backend(rows, mutation)
        for native_request, provided in zip(backend.requests, rows):
            native_request['prompt_ids'] = list(provided['prompt_ids'])
        status = evaluate(backend, rows, self.checkpoints, self.root/name,
            event_number=event, panel_name=panel,
            canonical_mode='fresh' if panel == 'fresh' else 'retained', qualification_requests=gate)
        return backend, status

    def read(self, name):
        return json.loads((self.root/name).read_text())

    def test_exact_main_budget_balanced_order_and_only_included_event6_gate(self):
        total = 0
        manifest = panels()
        for event, panel in ((6, 'development'), (6, 'additional'), (8, 'development'),
                             (8, 'additional'), (8, 'fresh')):
            name = f'campaign_001/event_{event:03d}/evaluation/{panel}'
            gate = 8 if event == 6 and panel == 'development' else 0
            rows = next(item['requests'] for item in manifest if item['name'] == panel)
            backend, status = self.run_panel(panel=panel, event=event, count=64, name=name, gate=gate, rows=rows)
            total += status['returned_generations']
            self.assertEqual(status['completed_valid_generations'], 384)
            self.assertEqual([item[0][1:] for item in backend.executions],
                [arm for index in range(64) for repeat in range(3) for arm in method_order(index, repeat)])
            self.assertEqual(backend.weight_checks, ['/O', '/K'])
            if gate:
                self.assertEqual(self.read(name+'/QUALIFICATION.json')['generations'], 48)
            else:
                self.assertFalse((self.root/name/'QUALIFICATION.json').exists())
                self.assertEqual(status['qualification'], 'not_required_prior_panel_gate')
        self.assertEqual(total, TOTAL_GENERATIONS)
        self.assertEqual(total, 1920)
        (self.root/'sealed').mkdir()
        (self.root/'sealed/PLAN.json').write_text(json.dumps(dict(panels=manifest,
            execution_source=dict(execution_git_commit='a'*40, dirty_files=[]))))
        (self.root/'campaign_001/STATUS.json').write_text(json.dumps(dict(status='completed')))
        (self.root/'campaign_001/COSTS.json').write_text(json.dumps(dict(event_006_O=dict(training_s=3., peak_allocated_bytes=123))))
        for event in (6, 8):
            for arm in ('O', 'K'):
                endpoint = self.root/f'campaign_001/{arm}/events/event_{event:03d}'
                endpoint.mkdir(parents=True)
                (endpoint/'RESULT.json').write_text(json.dumps(dict(export_path=self.checkpoints[arm])))
        report = write_campaign_report(self.root, self.root/'analysis', execution_commit='a'*40)
        self.assertEqual(report['complete_valid_panels'], 5)
        self.assertEqual(report['reaudit_valid_timed_generations'], 1920)
        curve = (self.root/'analysis/CALL_CURVE.csv').read_text()
        self.assertNotIn('latency', curve)
        self.assertIn('retained_PR16_counts_only', curve)
        self.assertTrue((self.root/'analysis/ALL_REPEATS.csv').is_file())
        self.assertTrue((self.root/'analysis/PROVENANCE.json').is_file())
        (self.root/'campaign_001/K/events/event_008/RESULT.json').write_text(json.dumps(dict(export_path='/wrong')))
        bad = write_campaign_report(self.root, self.root/'bad_analysis', execution_commit='a'*40)
        self.assertEqual(bad['complete_valid_panels'], 2)
        self.assertIn('own_event_endpoint_binding_K', bad['panels']['8_fresh']['binding_errors'])

    def test_retained_canonicals_have_no_historical_O_count_gate(self):
        rows = requests()
        for row in rows:
            row['reference_calls']['O'] = 999
        def changed(record, options):
            if record['checkpoint_id'] == '/O':
                harmful(record)
        status = evaluate(Backend(rows, changed), rows, self.checkpoints, self.root/'evaluation',
            event_number=6, panel_name='development', qualification_requests=1)
        self.assertEqual(status['status'], 'completed')
        summary = self.read('evaluation/report/SUMMARY.json')
        self.assertEqual(summary['O']['verification_calls'], 9)
        self.assertEqual(summary['K']['verification_calls'], 6)
        self.assertFalse(summary['historical_count_gate'])
        self.assertTrue(all(value['source_method'] == 'F' for value in self.read('evaluation/CANONICAL.json').values()))

    def test_campaign_report_preserves_unattempted_panels_as_inconclusive(self):
        (self.root/'sealed').mkdir()
        (self.root/'sealed/PLAN.json').write_text(json.dumps(dict(panels=panels(),
            execution_source=dict(execution_git_commit='a'*40, dirty_files=[]))))
        report = write_campaign_report(self.root, self.root/'analysis', execution_commit='a'*40)
        self.assertEqual(report['complete_valid_panels'], 0)
        self.assertEqual(report['reaudit_returned_measurements'], 0)
        self.assertTrue(all(value.startswith('inconclusive:') for value in report['interpretation'].values()))
        self.assertTrue(all(value['invalid_or_incomplete_requests'] == 64 for value in report['panels'].values()))

    def test_execution_source_accepts_both_explicit_clean_receipt_formats(self):
        commit = '04fac901f7af5478b62ff730833c026637ddd2cd'
        first = dict(execution_git_commit=commit, dirty_files=[])
        sealed = dict(repository='repository', commit=commit, branch='codex/branch', pr=18, dirty=False)
        self.assertEqual(validate_execution_source(first, commit), ['execution_git_commit/dirty_files'])
        self.assertEqual(validate_execution_source(sealed, commit), ['commit/dirty'])
        self.assertEqual(len(validate_execution_source({**first, **sealed}, commit)), 2)
        (self.root/'sealed').mkdir()
        (self.root/'sealed/PLAN.json').write_text(json.dumps(dict(panels=panels(), execution_source=sealed)))
        report = write_campaign_report(self.root, self.root/'analysis', execution_commit=commit)
        self.assertEqual(report['execution_source'], sealed)
        self.assertEqual(report['execution_source_receipt_formats'], ['commit/dirty'])

    def test_execution_source_rejects_missing_dirty_malformed_and_conflicting_aliases(self):
        commit = 'a'*40
        invalid = [{}, dict(commit=commit), dict(dirty=False), dict(execution_git_commit=commit),
            dict(dirty_files=[]), dict(commit=commit, dirty=0), dict(commit=commit, dirty=None),
            dict(commit=commit, dirty=True), dict(execution_git_commit=commit, dirty_files=None),
            dict(execution_git_commit=commit, dirty_files=['changed.py']),
            dict(commit='b'*40, dirty=False),
            dict(commit=commit, dirty=False, execution_git_commit='b'*40, dirty_files=[]),
            dict(commit=commit, dirty=False, dirty_files=[]),
            dict(execution_git_commit=commit, dirty_files=[], dirty=False)]
        for source in invalid:
            with self.subTest(source=source), self.assertRaises(ValueError):
                validate_execution_source(source, commit)
        with self.assertRaises(ValueError):
            validate_execution_source(dict(commit=commit, dirty=False), commit[:8])

    def recovery_fixture(self):
        recovery = self.root/'recovery_001'
        recovery.mkdir()
        metadata = dict(scientific_execution_commit='a'*40, recovery_execution_commit='b'*40,
            start_event=6, end_event=8, prior_status_path='recovery_001/PRIOR_STATUS.json',
            prior_costs_path='recovery_001/PRIOR_COSTS.json', native_output='campaign_001/native_recovery_001')
        (recovery/'RECOVERY.json').write_text(json.dumps(metadata))
        (recovery/'PRIOR_STATUS.json').write_text(json.dumps(dict(status='failed', optimizer_steps=256)))
        (recovery/'PRIOR_COSTS.json').write_text(json.dumps(dict(campaign_wall_s=7)))
        segments = [dict(execution_commit='a'*40, phase='original_through_event5', wall_s=7., backend_loading_s=2.),
                    dict(execution_commit='b'*40, phase='manual_events6_8', wall_s=11., backend_loading_s=3.)]
        return dict(execution_segments=segments, campaign_wall_s=18., backend_loading_s=2.)

    def test_report_preserves_recovery_sources_segment_costs_and_local_retention(self):
        costs = self.recovery_fixture()
        (self.root/'sealed').mkdir()
        (self.root/'sealed/PLAN.json').write_text(json.dumps(dict(panels=panels(),
            execution_source=dict(commit='a'*40, dirty=False))))
        campaign = self.root/'campaign_001'
        (campaign/'retention').mkdir(parents=True)
        (campaign/'COSTS.json').write_text(json.dumps(costs))
        (campaign/'STATUS.json').write_text(json.dumps(dict(status='completed', feedback_returned=512,
            optimizer_steps=1024, presentations=2048, evaluation_returned=1920)))
        (campaign/'retention/FINAL.json').write_text(json.dumps(dict(status='passed', mode='retained_locally',
            required_events=[6,8], destination='/A800/campaign_001', verified_files={'O/state': {}, 'K/state': {}})))
        report = write_campaign_report(self.root, self.root/'analysis', execution_commit='a'*40)
        history = report['execution_history']
        self.assertEqual(report['execution_git_commit'], 'a'*40)
        self.assertEqual(history['operational_recoveries'][0]['recovery_execution_commit'], 'b'*40)
        self.assertEqual(history['summed_segment_wall_s'], 18)
        self.assertEqual(history['summed_segment_backend_loading_s'], 5)
        self.assertTrue(history['campaign_wall_matches_segment_sum'])
        self.assertEqual(report['actual_work']['optimizer_steps'], 1024)
        self.assertEqual(report['endpoint_retention']['mode'], 'retained_locally')
        cost_report = self.read('analysis/COSTS.json')
        self.assertEqual(cost_report['execution_history']['execution_segments'], costs['execution_segments'])
        self.assertTrue((self.root/'analysis/EXECUTION_SEGMENTS.csv').exists())
        paths = {row['path'] for row in self.read('analysis/PROVENANCE.json')['receipts']}
        self.assertIn('recovery_001/RECOVERY.json', paths)
        self.assertIn('recovery_001/PRIOR_STATUS.json', paths)
        self.assertEqual(self.read('recovery_001/PRIOR_STATUS.json')['optimizer_steps'], 256)

    def test_recovery_history_rejects_source_or_scope_substitution_and_reports_cost_difference(self):
        costs = self.recovery_fixture()
        costs['campaign_wall_s'] = 19.
        history = execution_history(self.root, costs, 'a'*40)
        self.assertFalse(history['campaign_wall_matches_segment_sum'])
        path = self.root/'recovery_001/RECOVERY.json'
        original = json.loads(path.read_text())
        for key, value in (('scientific_execution_commit', 'c'*40), ('recovery_execution_commit', None),
                           ('start_event', 5), ('end_event', 9)):
            with self.subTest(key=key):
                path.write_text(json.dumps(dict(original, **{key:value})))
                with self.assertRaises(ValueError):
                    execution_history(self.root, costs, 'a'*40)
        path.write_text(json.dumps(original))
        costs['execution_segments'][1]['execution_commit'] = 'd'*40
        with self.assertRaisesRegex(ValueError, 'not bound'):
            execution_history(self.root, costs, 'a'*40)

    def test_retained_token_failure_stops_and_keeps_first_raw_record(self):
        def changed(record, options):
            record['response_ids'][1] = 777
            record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.run_panel(mutation=changed)
        status = self.read('evaluation/STATUS.json')
        self.assertEqual(status['returned_generations'], 1)
        self.assertEqual(status['invalid_generations'], 1)
        self.assertTrue((self.root/'evaluation/0000_r0_O.json').is_file())
        self.assertEqual(self.read('evaluation/report/SUMMARY.json')['valid_requests'], 0)

    def test_fresh_K_first_defers_until_first_O8_canonical_without_extra_generation(self):
        _, status = self.run_panel(panel='fresh', event=8, gate=0)
        self.assertEqual(status['returned_generations'], 18)
        events = [json.loads(line) for line in (self.root/'evaluation/VALIDATION_EVENTS.jsonl').read_text().splitlines()]
        self.assertEqual([row['status'] for row in events if row['label'] == '0001_r0_K'],
                         ['canonical_pending', 'valid'])
        canonical = self.read('evaluation/CANONICAL.json')['q1']
        self.assertEqual((canonical['source_method'], canonical['source_repeat']), ('O', 0))
        self.assertTrue(canonical['source_record_path'].endswith('0001_r0_O.json'))

    def test_fresh_mismatch_and_unknown_attempt_keep_pending_and_invalid_evidence(self):
        def wrong(record, options):
            if options['request_id'] == '0001_r0_K':
                record['response_ids'][1] = 777
                record['metadata']['trace'][0]['committed_tokens'][0] = 777
        with self.assertRaisesRegex(RuntimeError, 'reference_response_ids'):
            self.run_panel(panel='fresh', event=8, gate=0, mutation=wrong)
        status = self.read('evaluation/STATUS.json')
        self.assertEqual((status['returned_generations'], status['invalid_generations']), (8, 1))
        def lost(record, options):
            if options['request_id'] == '0001_r0_O':
                raise RuntimeError('reply lost')
        with self.assertRaisesRegex(RuntimeError, 'reply lost'):
            self.run_panel(panel='fresh', event=8, gate=0, mutation=lost, name='unknown')
        status = self.read('unknown/STATUS.json')
        self.assertEqual((status['returned_generations'], status['pending_canonical_generations'],
                          status['unknown_generation_attempts'], status['unattempted_generations']), (7, 1, 1, 10))

    def test_repeat_count_failure_invalidates_earlier_same_arm_rows(self):
        def changed(record, options):
            if options['request_id'] == '0000_r1_K':
                harmful(record)
        with self.assertRaisesRegex(RuntimeError, 'repeat_count_mismatch'):
            self.run_panel(mutation=changed)
        status = self.read('evaluation/STATUS.json')
        self.assertEqual((status['returned_generations'], status['invalid_generations']), (3, 2))

    def test_harmful_K_and_native_cache_times_are_reported_without_early_stop(self):
        def changed(record, options):
            if record['checkpoint_id'] == '/K':
                harmful(record)
                record['metadata']['request_latency_s'] = 2.
                record['metadata']['native_generate_s'] = 1.7
                record['metadata']['capture']['cache_maintenance_s'] = .2
        _, status = self.run_panel(mutation=changed)
        self.assertEqual(status['status'], 'completed')
        summary = self.read('evaluation/report/SUMMARY.json')
        self.assertEqual(summary['comparisons']['K_vs_O']['call_reduction_pct'], -50)
        self.assertEqual(summary['K']['summed_mean_native_generate_s'], 5.1)
        self.assertAlmostEqual(summary['K']['summed_mean_cache_maintenance_s'], .6)
        self.assertEqual(summary['K']['summed_mean_request_latency_s'], 6.)

    def test_report_reaudits_native_cache_scalar_and_retained_canonical_binding(self):
        self.run_panel()
        rows = [json.loads(line) for line in (self.root/'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
        canonical = self.read('evaluation/CANONICAL.json')
        rows[0]['cache_maintenance_s'] = .5
        summary, _, _, _ = summarize(rows, requests(), checkpoints=self.checkpoints,
                                     canonicals=canonical, canonical_mode='retained')
        self.assertEqual(summary['valid_requests'], 2)
        canonical['q1']['response_ids'][1] = 777
        summary, per_request, _, _ = summarize(rows, requests(), checkpoints=self.checkpoints,
                                               canonicals=canonical, canonical_mode='retained')
        self.assertEqual(summary['valid_requests'], 1)
        self.assertIn('retained_canonical_binding', per_request[1]['errors'])

    def test_nonfinite_native_failure_is_retained_without_report_crash(self):
        def changed(record, options):
            record['metadata']['request_latency_s'] = float('nan')
        with self.assertRaises(RuntimeError):
            self.run_panel(mutation=changed)
        self.assertEqual(self.read('evaluation/STATUS.json')['invalid_generations'], 1)
        self.assertTrue((self.root/'evaluation/report/SUMMARY.json').is_file())
        self.assertFalse((self.root/'evaluation/REPORT_FAILURE.json').exists())

    def test_fixed_panel_contract_disallows_overlap_wrong_seed_and_extra_reference(self):
        manifest = panels()
        self.assertEqual(validate_panels(manifest)['main_generations'], 1920)
        bad = copy.deepcopy(manifest)
        bad[2]['selection_seed'] = 17
        with self.assertRaisesRegex(ValueError, '20261029'):
            validate_panels(bad)
        bad = copy.deepcopy(manifest)
        bad[2]['requests'][0]['prompt_ids'] = bad[0]['requests'][0]['prompt_ids']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_panels(bad)
        with self.assertRaisesRegex(ValueError, 'event 8'):
            self.run_panel(panel='fresh', event=6, gate=0)

    def test_import_has_no_training_GPU_or_worker_side_effect(self):
        source = "import sys; import experiments.ok_events_5_8.evaluate; assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules; assert 'online_feedback.training' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_call_curve_classification_does_not_treat_incomplete_or_zero_as_advantage(self):
        self.assertIn('shrinks', curve_direction(3, 1, middle=2, complete=True))
        self.assertIn('grows', curve_direction(1, 3, middle=2, complete=True))
        self.assertIn('reverses', curve_direction(1, -1, middle=2, complete=True))
        self.assertIn('parity', curve_direction(1, 0, middle=2, complete=True))
        self.assertIn('inconclusive', curve_direction(1, 3, middle=2, complete=False))
        self.assertIn('inconclusive', curve_direction(1, 3, middle=None, complete=True))

    def test_call_curve_labels_event6_reversal_as_recovery_and_net_change_only(self):
        recovered = curve_direction(2, 3, middle=-1, complete=True)
        self.assertIn('recovers', recovered)
        self.assertIn('event-6 reversal', recovered)
        self.assertNotIn('persists', recovered)
        self.assertIn('net call reduction grows', recovered)
        self.assertIn('no monotonicity claim', recovered)
        parity = curve_direction(2, 1, middle=0, complete=True)
        self.assertIn('event-6 parity', parity)
        self.assertNotIn('persists', parity)
        nonmonotone = curve_direction(2, 3, middle=5, complete=True)
        self.assertIn('persists at all three measured endpoints', nonmonotone)
        self.assertIn('no monotonicity claim', nonmonotone)


if __name__ == '__main__':
    unittest.main()
