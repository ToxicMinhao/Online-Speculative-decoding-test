"""Synthetic standard-library checks for the postrun analyzer; no experiment."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_results as analysis


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def requests(count=3):
    return [{'request_id': f'q{index}', 'panel_ordinal': index, 'original_index': 10 + index,
             'prompt_ids': [1, 2], 'response_ids': [10, 11, 12, 13, 99],
             'finish_reason': 'eos', 'max_new_tokens': 2048, 'stop_token_ids': [99],
             'reference_calls': {'F': 4 + index, 'O': 999}} for index in range(count)]


def write_plan(root, panel):
    path = root / 'sealed/PLAN.json'
    write(path, {'eval_requests': panel, 'initial': {
        arm: {'export': f'/event1/{arm}'} for arm in analysis.ARMS}})
    identity = analysis.fingerprint(path)
    write(root / 'sealed/SEAL.json', {'files': {
        '/remote/run/sealed/PLAN.json': {key: identity[key] for key in ('bytes', 'sha256')}}})


def write_evaluation(root, event, panel, customize=None):
    root.mkdir(parents=True)
    checkpoints = {arm: f'/event{event}/{arm}' for arm in analysis.ARMS}
    rows = []
    for request in panel:
        index = request['panel_ordinal']
        for repeat in range(3):
            order = ('O', 'K') if (index + repeat) % 2 == 0 else ('K', 'O')
            for arm in order:
                label = f'{index:04d}_r{repeat}_{arm}'
                response = request['response_ids']
                trace = [dict(output_start=0, anchor_token=10, committed_progress=2,
                              committed_tokens=[11, 12], accepted_draft_tokens=1),
                         dict(output_start=2, anchor_token=12, committed_progress=2,
                              committed_tokens=[13, 99], accepted_draft_tokens=1)]
                record = {key: copy.deepcopy(request[key]) for key in
                          ('prompt_ids', 'response_ids', 'finish_reason', 'max_new_tokens', 'stop_token_ids')}
                record.update(record_id=label, request_index=index,
                    session_id='prefix_ranking_evaluation' if event == 1 else 'ok_continuation_evaluation',
                    purpose='evaluation', checkpoint_id=checkpoints[arm], metadata=dict(
                        verification_calls=2, accepted_draft_tokens=2, committed_progress=4,
                        request_latency_s=1., native_generate_s=.9, trace=trace,
                        runtime='pinned_SGLang_FA3', block_size=16, context_limit=4096,
                        cache_policy='idle_full_flush_before_every_request_and_publication',
                        feedback_capture_in_timed_request=False,
                        capture=dict(collect_features=False, feature_bytes=0,
                                     native_capture_validated=True, cache_maintenance_s=.1)))
                if customize:
                    customize(record, arm, index, repeat)
                row = dict(request_id=request['request_id'], panel_ordinal=index,
                    original_index=request['original_index'], method=arm, repeat=repeat,
                    checkpoint_id=checkpoints[arm], output_tokens=len(response), prompt_tokens=2,
                    finish_reason=record['finish_reason'], cap_hit=record['finish_reason'] == 'length',
                    record_path='/remote/evaluation/' + label + '.json', valid_measurement=True,
                    errors=[], cache_maintenance_s=.1)
                row.update({key: record['metadata'][key] for key in analysis.COUNTS[:3] +
                            ('request_latency_s', 'native_generate_s')})
                write(root / (label + '.json'), record)
                rows.append(row)
    (root / 'MEASUREMENTS.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write(root / 'PROTOCOL.json', {'checkpoints': checkpoints})
    write(root / 'STATUS.json', dict(status='completed', returned_generations=len(rows),
          attempted_generations=len(rows), activation_s=1000., weight_validation_s=1., evaluation_wall_s=2000.))
    return rows


def three_calls(record):
    record['metadata'].update(verification_calls=3, accepted_draft_tokens=1,
        trace=[dict(output_start=0, anchor_token=10, committed_progress=1,
                    committed_tokens=[11], accepted_draft_tokens=0),
               dict(output_start=1, anchor_token=11, committed_progress=1,
                    committed_tokens=[12], accepted_draft_tokens=0),
               dict(output_start=2, anchor_token=12, committed_progress=2,
                    committed_tokens=[13, 99], accepted_draft_tokens=1)])


def training_receipt(event, arm):
    features = [dict(record_id=f'r{i}', prompt_length=1, length=4, path=f'/shared/{event}/r{i}.pt')
                for i in range(128)]
    rows = []
    weight = 1 + math.exp(-1 / 7.)
    for index in range(256):
        responses = 1 if index % 2 == 0 else 2
        metrics = dict(response_presentations=responses,
                       supervised_position_exposure=responses * (2 if arm == 'O' else 1))
        if arm == 'K':
            metrics.update(available_position_occurrences=2 * responses, selected_position_occurrences=responses,
                available_position_weight=weight * responses, selected_position_weight=float(responses),
                main_loss_numerator=.5 * responses, loss_denominator=responses * (1 + 1e-6),
                no_error_blocks=0, empty_blocks=0,
                **{key: responses if key == 'first_error_position_1' else 0 for key in analysis.HISTOGRAM})
        rows.append(dict(record_id=f'r{index % 128}', pass_=index // 128, optimizer_stepped=index % 2 == 1,
            optimizer_step_after=(event - 1) * 128 + (index + 1) // 2,
            sampled_anchors=[dict(positions=[[1]], keep_mask=[[True]])], metrics=metrics))
        rows[-1]['pass'] = rows[-1].pop('pass_')
    receipt = dict(status='completed', features=features, records=rows, start_step=(event - 1) * 128,
                   event_id=f'event_{event:03d}', objective='osd' if arm == 'O' else 'osd_first_error',
                   passes=2, response_presentations=256, export_path=f'/event{event}/{arm}',
                   end_step=event * 128, optimizer_steps=128, costs=dict(training_s=2., checkpoint_save_s=1.),
                   objective_work=dict(selection_s=.5 if arm == 'K' else 0.))
    if arm == 'K':
        receipt.update(selection_totals_source='optimizer_boundary_accumulation_windows_once',
            selection_totals={key: sum(row['metrics'][key] for row in rows if row['optimizer_stepped'])
                              for key in rows[0]['metrics']})
    return receipt


class AnalyzerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.panel = requests()
        self.frozen = {row['request_id']: row['reference_calls']['F'] for row in self.panel}

    def event(self, directory, event=2):
        issues = []
        result = analysis.analyze_event(directory, event, self.panel, self.frozen, issues)
        return result, issues

    def test_one_trajectory_counts_three_mean_times_and_reporting_only_F(self):
        def change(record, arm, index, repeat):
            means = {'O': (1., 9., 10.), 'K': (1., 3., 10.)}
            record['metadata']['request_latency_s'] = means[arm][index] * (.5, 1., 1.5)[repeat]
            record['metadata']['native_generate_s'] = record['metadata']['request_latency_s'] * .9
            if arm == 'O' and index == 1:
                three_calls(record)
        directory = self.root / 'evaluation'
        write_evaluation(directory, 2, self.panel, change)
        (summary, rows, repeats, _), issues = self.event(directory)
        self.assertEqual(issues, [])
        self.assertEqual(summary['O']['verification_calls'], 7)
        self.assertEqual(summary['K']['verification_calls'], 6)
        self.assertEqual(summary['F_verification_calls'], 15)
        self.assertEqual(summary['O']['summed_mean_request_latency_s'], 20)
        self.assertEqual(summary['K']['summed_mean_request_latency_s'], 14)
        self.assertAlmostEqual(summary['comparisons']['K_vs_O']['call_reduction_pct'], 100 / 7)
        self.assertAlmostEqual(summary['comparisons']['K_vs_O']['request_time_reduction_pct'], 30)
        self.assertNotIn('request_time_reduction_pct', summary['comparisons']['O_vs_F'])
        self.assertEqual(summary['comparisons']['K_vs_O']['call_request_directions'],
                         dict(improved=1, regressed=0, tied=2))
        self.assertEqual(len(repeats), 3)
        self.assertTrue(all(row['valid'] for row in rows))

    def test_native_mismatch_and_repeat_count_mismatch_remove_whole_paired_request(self):
        for mode in ('native', 'repeat'):
            directory = self.root / mode
            def change(record, arm, index, repeat):
                if arm == 'K' and index == 1 and repeat == 1:
                    if mode == 'native':
                        record['response_ids'][1] = 777
                        record['metadata']['trace'][0]['committed_tokens'][0] = 777
                    else:
                        three_calls(record)
            write_evaluation(directory, 2, self.panel, change)
            (summary, rows, _, _), issues = self.event(directory)
            self.assertEqual(summary['valid_requests'], 2)
            self.assertEqual(summary['F_verification_calls'], 10)
            self.assertEqual(summary['O']['verification_calls'], 4)
            self.assertFalse(rows[1]['valid'])
            self.assertTrue(issues)

    def test_event_one_uses_contemporaneous_O_K_and_ignores_C_A(self):
        directory = self.root / 'event_one'
        rows = write_evaluation(directory, 1, self.panel)
        foreign = {**rows[0], 'method': 'C', 'request_latency_s': 10000.}
        with (directory / 'MEASUREMENTS.jsonl').open('a') as stream:
            stream.write(json.dumps(foreign) + '\n')
        (summary, _, _, _), issues = self.event(directory, 1)
        self.assertEqual(issues, [])
        self.assertEqual(summary['returned_O_K_measurements'], 18)
        self.assertEqual(summary['O']['summed_mean_request_latency_s'], 3)
        self.assertEqual(summary['comparisons']['K_vs_O']['request_time_reduction_pct'], 0)

    def test_missing_and_partial_event_do_not_invent_full_results(self):
        (summary, rows, _, _), issues = self.event(self.root / 'missing')
        self.assertEqual(summary['measurement_status'], 'missing')
        self.assertIsNone(summary['O']['verification_calls'])
        self.assertIsNone(summary['comparisons']['K_vs_O']['request_time_reduction_pct'])
        self.assertEqual(len(rows), 3)
        directory = self.root / 'partial'
        returned = write_evaluation(directory, 2, self.panel)
        (directory / 'MEASUREMENTS.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in returned[:7]))
        write(directory / 'STATUS.json', dict(status='failed', returned_generations=7, attempted_generations=8,
                                             unknown_generation_attempts=1, unattempted_generations=10))
        (summary, rows, _, _), issues = self.event(directory)
        self.assertEqual(summary['valid_requests'], 1)
        self.assertEqual(summary['O']['verification_calls'], 2)
        self.assertEqual(summary['F_verification_calls'], 4)
        self.assertEqual(summary['execution_accounting']['unknown_generation_attempts'], 1)
        self.assertEqual(issues, [])

    def test_nonfinite_scalar_and_missing_native_file_are_invalid(self):
        directory = self.root / 'invalid'
        rows = write_evaluation(directory, 2, self.panel)
        rows[0]['request_latency_s'] = float('nan')
        (directory / '0001_r0_K.json').unlink()
        (directory / 'MEASUREMENTS.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (summary, _, _, _), _ = self.event(directory)
        self.assertEqual(summary['valid_requests'], 1)
        self.assertEqual(summary['native_rows_with_errors'], 2)

    def test_training_exposure_is_boundary_only_and_normalized_weights_reconstructed(self):
        for arm in analysis.ARMS:
            path = self.root / (arm + '.json')
            write(path, training_receipt(2, arm))
            issues = []
            result, schedule = analysis.training_summary(path, 2, arm, issues)
            self.assertEqual(issues, [])
            self.assertEqual(result['available_occurrences'], 512)
            self.assertEqual(result['response_presentations'], 256)
            self.assertEqual(result['sampled_block_slots'], 256)
            self.assertEqual(result['boundary_metrics']['response_presentations'], 256)
            self.assertEqual(result['selected_occurrences'], 512 if arm == 'O' else 256)
            self.assertEqual(len(schedule['schedule']), 256)
        malformed = training_receipt(2, 'K')
        malformed['selection_totals']['selected_position_occurrences'] *= 1.5
        write(self.root / 'wrong.json', malformed)
        issues = []
        analysis.training_summary(self.root / 'wrong.json', 2, 'K', issues)
        self.assertTrue(any('Selection total differs' in issue['message'] for issue in issues))

    def test_completed_training_identity_and_objective_are_checked(self):
        malformed = training_receipt(2, 'K')
        malformed.update(objective='osd', event_id='event_003', start_step=0)
        path = self.root / 'wrong_identity.json'
        write(path, malformed)
        issues = []
        analysis.training_summary(path, 2, 'K', issues)
        self.assertTrue(any('identity/objective/counters' in issue['message'] for issue in issues))

    def test_malformed_repeat_excludes_request_without_crashing(self):
        directory = self.root / 'invalid_slot'
        rows = write_evaluation(directory, 2, self.panel)
        rows[0]['repeat'] = []
        (directory / 'MEASUREMENTS.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (summary, requests_out, _, _), _ = self.event(directory)
        self.assertEqual(summary['valid_requests'], 2)
        self.assertIn('incomplete_or_duplicate_slots', requests_out[0]['errors'])

    def test_panel_and_plan_digest_and_event_checkpoint_are_bound(self):
        write_plan(self.root, self.panel)
        issues = []
        analysis.bind_panel(self.root / 'campaign', self.panel, issues)
        self.assertEqual(issues, [])
        changed = copy.deepcopy(self.panel)
        changed[0]['prompt_ids'] = [999]
        analysis.bind_panel(self.root / 'campaign', changed, issues)
        self.assertTrue(any('panel differs' in issue['message'] for issue in issues))
        with (self.root / 'sealed/PLAN.json').open('a') as stream:
            stream.write(' ')
        analysis.bind_panel(self.root / 'campaign', self.panel, issues)
        self.assertTrue(any('PLAN bytes differ' in issue['message'] for issue in issues))
        directory = self.root / 'wrong_checkpoint'
        write_evaluation(directory, 2, self.panel)
        issues = []
        analysis.analyze_event(directory, 2, self.panel, self.frozen, issues,
                               expected_checkpoints={'O': '/event1/O', 'K': '/event1/K'})
        self.assertTrue(any('checkpoints differ' in issue['message'] for issue in issues))

    def test_partial_training_unknown_is_not_classified_as_zero_work(self):
        campaign = self.root / 'campaign'
        (campaign / 'resume_002_K').mkdir(parents=True)
        write(campaign / 'resume_002_O/CLOSED.json', dict(global_step=256, presentations=512))
        events = {str(event): {'execution_accounting': None, 'returned_O_K_measurements': 0} for event in (2, 4)}
        result = analysis.work_accounting(campaign, dict(status='failed', phase='training_K'), events, [], [])
        self.assertEqual(result['observed_arm_steps'], dict(O=256, K=128))
        self.assertIsNone(result['optimizer_steps_exact'])
        self.assertEqual(result['unknown_training_counters'], [dict(event=2, arm='K')])

    def test_interrupted_optimizer_remains_unknown_even_with_closed_counter(self):
        campaign = self.root / 'campaign'
        write(campaign / 'resume_002_O/CLOSED.json', dict(global_step=256, presentations=512))
        write(campaign / 'resume_002_K/CLOSED.json', dict(global_step=150, presentations=301))
        events = {str(event): {'execution_accounting': None, 'returned_O_K_measurements': 0}
                  for event in (2, 4)}
        result = analysis.work_accounting(campaign, dict(status='failed', phase='training_K', event=2), events, [], [])
        self.assertEqual(result['reported_optimizer_steps'], 150)
        self.assertEqual(result['reported_presentations'], 301)
        self.assertIsNone(result['optimizer_steps_exact'])
        self.assertIsNone(result['presentations_exact'])
        self.assertEqual(result['unknown_training_counters'], [dict(event=2, arm='K')])

    def test_cli_returns_nonzero_for_incomplete_or_issues(self):
        for status, expected in (('complete', 0), ('incomplete_or_issues', 1)):
            with self.subTest(status=status), patch('sys.argv', ['analyze_results.py', '--campaign', 'campaign',
                    '--output', 'output']), patch.object(analysis, 'analyze', return_value={
                    'status': status, 'issues': [], 'work_accounting': {'campaign_status': {'status': 'failed'}}}), \
                    patch('builtins.print'):
                self.assertEqual(analysis.main(), expected)

    def test_integration_emits_partial_outputs_and_refuses_overwrite(self):
        event_one, campaign, output = self.root / 'old', self.root / 'campaign', self.root / 'analysis'
        panel = requests(64)
        write_plan(self.root, panel)
        write(event_one / 'MANIFEST.json', {'eval_requests': panel})
        write_evaluation(event_one / 'evaluation', 1, panel)
        write(campaign / 'STATUS.json', dict(status='failed', phase='loading_native'))
        report = analysis.analyze(campaign, event_one, self.root / 'missing_training.json', output)
        self.assertEqual(report['status'], 'incomplete_or_issues')
        self.assertEqual(report['events']['1']['valid_requests'], 64)
        self.assertEqual(report['events']['2']['measurement_status'], 'missing')
        self.assertIsNone(report['events']['4']['O']['verification_calls'])
        self.assertTrue((output / 'PER_REQUEST.csv').is_file())
        self.assertIn('No F latency or cross-event latency change', report['timing_scope'])
        before = (output / 'ANALYSIS.json').read_bytes()
        with self.assertRaises(FileExistsError):
            analysis.analyze(campaign, event_one, self.root / 'missing_training.json', output)
        self.assertEqual((output / 'ANALYSIS.json').read_bytes(), before)

    def test_no_raw_campaign_means_no_analysis_or_experiment(self):
        with self.assertRaises(FileNotFoundError):
            analysis.analyze(self.root / 'missing', self.root / 'old', self.root / 'receipt', self.root / 'output')
        self.assertFalse((self.root / 'output').exists())

    def test_complete_synthetic_campaign_accounts_for_768_steps_and_same_event_time_only(self):
        event_one, campaign, output = self.root / 'old', self.root / 'campaign', self.root / 'analysis'
        panel = requests(64)
        write_plan(self.root, panel)
        write(event_one / 'MANIFEST.json', {'eval_requests': panel})
        for event in (1, 2, 4):
            def timing(record, arm, index, repeat):
                scale = {1: 100., 2: 10., 4: 1.}[event]
                record['metadata']['request_latency_s'] = scale * (1. if arm == 'O' else .9)
                record['metadata']['native_generate_s'] = record['metadata']['request_latency_s'] * .9
            directory = event_one / 'evaluation' if event == 1 else campaign / f'evaluation_{event:03d}'
            write_evaluation(directory, event, panel, timing)
        ordinary = self.root / 'ordinary.json'
        write(ordinary, training_receipt(1, 'O'))
        write(event_one / 'K/events/event_001/RESULT.json', training_receipt(1, 'K'))
        for event in (2, 3, 4):
            for arm in analysis.ARMS:
                write(campaign / arm / 'events' / f'event_{event:03d}' / 'RESULT.json', training_receipt(event, arm))
                write(campaign / f'resume_{event:03d}_{arm}/CLOSED.json',
                      dict(global_step=event * 128, presentations=event * 256))
            for index in range(128):
                write(campaign / f'event_{event:03d}/responses/r{index}.json', {'record_id': f'r{index}'})
        write(campaign / 'STATUS.json', dict(status='completed', optimizer_steps=768,
              presentations=1536, feedback_returned=384, evaluation_returned=768))
        write(campaign / 'COSTS.json', dict(campaign_wall_s=300., nested_training_s=100.))
        report = analysis.analyze(campaign, event_one, ordinary, output)
        self.assertEqual(report['issues'], [])
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['work_accounting']['optimizer_steps_exact'], 768)
        self.assertEqual(report['work_accounting']['presentations_exact'], 1536)
        self.assertEqual(report['work_accounting']['returned_evaluation_rows'], 768)
        self.assertEqual(report['work_accounting']['durable_feedback_response_files'], 384)
        self.assertEqual(len(report['training']), 8)
        for event in ('1', '2', '4'):
            self.assertAlmostEqual(report['events'][event]['comparisons']['K_vs_O']['request_time_reduction_pct'], 10.)
            self.assertEqual(report['events'][event]['O']['verification_calls'], 128)
        self.assertEqual(len(report['fallacy_scan']), 11)
        self.assertNotIn('historical_time', json.dumps(report))
        # Durable valid rows cannot turn a failed evaluator into a completed run.
        failed_status = analysis.load(campaign / 'evaluation_002/STATUS.json')
        failed_status['status'] = 'failed'
        write(campaign / 'evaluation_002/STATUS.json', failed_status)
        failed = analysis.analyze(campaign, event_one, ordinary, self.root / 'failed_analysis')
        self.assertEqual(failed['events']['2']['valid_requests'], 64)
        self.assertEqual(failed['status'], 'incomplete_or_issues')
        self.assertTrue(any('failed evaluator status' in issue['message'] for issue in failed['issues']))
        # The original event1 receipt must identify the sealed parent export.
        changed = analysis.load(ordinary)
        changed['export_path'] = '/unrelated/O'
        write(ordinary, changed)
        wrong_parent = analysis.analyze(campaign, event_one, ordinary, self.root / 'wrong_parent_analysis')
        self.assertTrue(any('Event1 export differs' in issue['message'] for issue in wrong_parent['issues']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
