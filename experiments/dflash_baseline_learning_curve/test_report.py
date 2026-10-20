import copy
import json

import pytest

from .report import comparison, summarize, write_reports
from .test_experiment import result


def test_fresh_group_source_fit_and_harm_remain_separate():
    report = summarize([result(harmful=True)])
    assert report['status'] == 'partial_or_failed'
    group = next(row for row in report['pooled'] if row['scope'] == 'group' and row['candidate'] == 'S8' and row['control'] == 'F')
    assert group['verification_call_reduction_percent'] == -100
    assert group['regressed_requests'] == 4
    assert [row['loss'] for row in report['source_losses']] == [1., .95, .9]
    assert report['core_generations'] == 16 and report['explicit_warmups'] == 3


def test_pool_uses_summed_latencies_not_mean_request_percentages():
    data = result()
    for row in data['scores']:
        index = int(row['request_id'][-1])
        row['request_s'] = ([1., 9., 1., 9.] if row['checkpoint_id'] == 'F' else [.5, 9., .5, 9.])[index]
    report = summarize([data])
    pooled = next(row for row in report['pooled'] if row['scope'] == 'overall' and row['candidate'] == 'S2' and row['control'] == 'F')
    assert pooled['control_metrics']['request_s'] == 20
    assert pooled['candidate_metrics']['request_s'] == 19
    assert pooled['request_time_reduction_percent'] == pytest.approx(5)


def test_rejected_measurements_cannot_reenter_gain_tables():
    data = result()
    row = next(row for row in data['scores'] if row['checkpoint_id'] == 'S2')
    row['accepted_draft_tokens'] = 999
    value, _ = comparison(data, row['request_id'], 'S2', 'F')
    assert not value['valid'] and value['request_time_reduction_percent'] is None
    report = summarize([data])
    assert len(report['per_request']) == 16
    pooled = next(row for row in report['pooled'] if row['scope'] == 'overall' and row['candidate'] == 'S2' and row['control'] == 'F')
    assert pooled['valid_requests'] == 3 and pooled['invalid_requests'] == 1


def test_fabricated_valid_score_flags_do_not_hide_invalid_training():
    data = result()
    data['training']['accounting']['optimizer_steps'] = 9
    row = data['scores'][0]
    value, _ = comparison(data, row['request_id'], 'S2', 'F')
    assert not value['valid'] and value['request_time_reduction_percent'] is None
    assert any(message.startswith('training:') for message in value['invalid_reasons'])
    assert all(not row['valid'] for row in summarize([data])['comparisons'])


def test_nonfinite_training_evidence_invalidates_gains_even_with_positive_scores():
    data = result()
    data['training']['costs']['training_s'] = float('nan')
    assert all(not row['valid'] for row in summarize([data])['comparisons'])


def test_missing_quiescence_or_changed_ids_invalidates_comparison():
    data = result()
    row = next(row for row in data['scores'] if row['checkpoint_id'] == 'S8')
    row['training_quiescence']['trainer_processes'] = [123]
    value, _ = comparison(data, row['request_id'], 'S8', 'S2')
    assert not value['valid']
    assert 'S8:background_training_not_excluded' in value['invalid_reasons']


def test_cost_components_not_added_to_latency():
    report = summarize([result()])
    cost = report['costs'][0]
    assert cost['teacher_preparation_s'] == 2 and cost['training_s'] == 5
    assert cost['checkpoint_loading_s'] == pytest.approx(15 * .3)
    comparison_row = next(row for row in report['pooled'] if row['scope'] == 'overall' and row['candidate'] == 'S2' and row['control'] == 'F')
    assert comparison_row['candidate_metrics']['request_s'] == 4.4
    assert 'do not sum' in cost['accounting']


def test_domains_and_extra_training_comparison_preserved():
    report = summarize([result(0), result(2), result(4)])
    assert {row['scope_id'] for row in report['pooled'] if row['scope'] == 'domain'} == {'gsm8k', 'codealpaca', 'sharegpt'}
    assert any(row['candidate'] == 'S8' and row['control'] == 'S2' for row in report['comparisons'])


def test_not_executed_report_and_full_machine_readable_files(tmp_path):
    report = write_reports([], tmp_path)
    assert report['status'] == 'not_executed'
    text = (tmp_path / 'REPORT.md').read_text()
    assert 'No native measurements have been executed' in text
    assert 'not establish net online speedup' in text
    assert {'REPORT.md', 'SUMMARY.json', 'PER_REQUEST.csv', 'PER_ROUND.csv', 'COMPARISONS.csv',
            'POOLED.csv', 'SOURCE_LOSSES.csv', 'SUPERVISED_EXPOSURE.csv', 'COSTS.csv', 'TRAINING_ACCOUNTING.json'} <= {p.name for p in tmp_path.iterdir()}


def test_report_roundtrip_preserves_outputs_in_result_not_public_summary(tmp_path):
    data = json.loads(json.dumps(result()))
    report = write_reports([data], tmp_path)
    assert len(report['per_round']) == 32
    assert all('output_ids' not in row and 'text' not in row for row in report['per_request'])
    assert 'Source losses use a separate fixed anchor set' in (tmp_path / 'REPORT.md').read_text()


def test_zero_exposure_is_reported_without_dropping_presentation(tmp_path):
    data = result()
    row = data['training']['accounting']['batches'][0]['presentations'][0]
    row.update(unique_supervised_position_ids=[], unique_supervised_positions=0,
               eligible_positions=0, zero_supervision=True,
               anchor_positions=[row['response_end_exclusive'] - 1], normalization_denominator=1e-6)
    report = write_reports([data], tmp_path)
    assert report['zero_supervision_presentations'] == 1
    assert report['training_response_presentations'] == 16
    assert report['optimizer_steps'] == 8
    assert report['supervised_exposure'][0]['unique_supervised_position_ids'] == []
    assert report['supervised_exposure'][0]['anchor_positions'] == row['anchor_positions']
    assert all(row['valid'] for row in report['comparisons'])
    assert 'Zero-supervision presentations: 1' in (tmp_path / 'REPORT.md').read_text()
