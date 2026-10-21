"""Independent aggregation and invalid/partial evidence admission checks."""
import copy
import json

import pytest

from . import experiment as ex, report
from .test_experiment import FakeBackend, complete_result, groups_fixture, native_row


def overall(summary, subset='all_valid'):
    return next(r for r in summary['pooled'] if r['scope'] == 'overall' and r['subset'] == subset)


def replace_pair(result, group_index, pair_index, *, f_calls=4, s_calls=3, output=None):
    group = result['groups'][group_index]; pair = group['pairs'][pair_index]
    for label, calls in (('F', f_calls), ('S8', s_calls)):
        native = native_row(pair['child'], pair['checkpoint_bindings'][label], label, calls=calls, output=output)
        pair[label]['native'] = native
        attempt = next(a for a in group['native_attempts'] if a['request_id'] == pair['request_id'] and a['checkpoint_id'] == label)
        group['native_raw'][attempt['raw_index']] = copy.deepcopy(native)
    pair['repetition'] = ex.repeat_flags(pair['child'], pair['F']['native'], group['feedback_answers'])


def test_complete_summaries_preserve_twelve_children_and_scope_denominators():
    summary = report.summarize(complete_result())
    assert summary['status'] == 'complete' and summary['valid_requests'] == summary['selected_requests'] == 12
    assert len(summary['pooled']) == 20 and {r['scope'] for r in summary['pooled']} == {'overall', 'domain', 'group'}
    row = overall(summary)
    assert (row['F_verification_calls'], row['S8_verification_calls']) == (48, 60)
    assert row['S8_vs_F_verification_call_reduction_percent'] == -25
    assert row['S8_vs_F_native_worse_requests'] == 12
    assert overall(summary, 'non_exact_repeat')['valid_requests'] == 12
    assert summary['comparisons'][0]['construction']['why_required_answer_differs']
    assert summary['accounting']['returned_valid_native_generations'] == 24


def test_call_reduction_uses_sums_not_mean_request_percentages():
    result = complete_result()
    replace_pair(result, 0, 0, f_calls=4, s_calls=3)
    replace_pair(result, 0, 1, f_calls=12, s_calls=12)
    rows = report.summarize(result)['comparisons'][:2]
    pooled = report.pool(rows)
    assert (pooled['F_verification_calls'], pooled['S8_verification_calls']) == (16, 15)
    assert pooled['S8_vs_F_verification_call_reduction_percent'] == pytest.approx(6.25)
    assert pooled['S8_vs_F_verification_call_reduction_percent'] != (25 + 0) / 2
    assert pooled['S8_vs_F_native_better_requests'] == pooled['S8_vs_F_native_tied_requests'] == 1


def test_repeated_parent_and_other_feedback_answers_remain_primary():
    result = complete_result(outcome='better')
    group = result['groups'][0]
    first = group['pairs'][0]
    parent = next(r for r in group['feedback_answers'] if r['request_id'] == first['parent_request_id'])
    replace_pair(result, 0, 0, f_calls=1, s_calls=1, output=parent['output_ids'])
    second = group['pairs'][1]
    other = next(r for r in group['feedback_answers'] if r['request_id'] != second['parent_request_id'] and len(r['output_ids']) > 1)
    replace_pair(result, 0, 1, f_calls=1, s_calls=1, output=other['output_ids'])
    summary = report.summarize(result)
    assert summary['status'] == 'complete' and overall(summary)['valid_requests'] == 12
    subset = overall(summary, 'non_exact_repeat')
    assert subset['selected_requests'] == 12 and subset['subset_requests'] == subset['valid_requests'] == 10
    assert summary['comparisons'][0]['matches_parent_answer'] is True
    assert summary['comparisons'][1]['matches_parent_answer'] is False
    assert summary['comparisons'][1]['matches_any_feedback_answer'] is True
    assert overall(summary)['valid_exact_repeat_requests'] == 2


@pytest.mark.parametrize('mutation', ['counter', 'checkpoint', 'output', 'recorded_validation', 'repeat',
                                     'duplicate', 'raw', 'child', 'cleanup'])
def test_invalid_evidence_cannot_contribute_gains(mutation):
    result = complete_result(outcome='better'); group = result['groups'][0]; pair = group['pairs'][0]
    if mutation == 'counter': pair['S8']['native']['accepted_draft_tokens'] = 99999
    if mutation == 'checkpoint': pair['S8']['native']['checkpoint_identity_sha256'] = 'x'*64
    if mutation == 'output': pair['S8']['native']['output_ids'][0] += 1
    if mutation == 'recorded_validation': pair['validation'] = {'status': 'failed', 'errors': ['execution rejected']}
    if mutation == 'repeat': pair['repetition']['non_exact_repeat'] = False
    if mutation == 'duplicate': group['pairs'].append(copy.deepcopy(pair))
    if mutation == 'raw': group['native_raw'][0]['request_s'] += 1
    if mutation == 'child': pair['child']['text'] = 'Changed after outputs'
    if mutation == 'cleanup': group['cleanup']['checkpoint_weights_unchanged'] = False
    summary = report.summarize(result); row = summary['comparisons'][0]
    assert not row['valid'] and row['invalid_reasons']
    assert row['S8_vs_F_verification_calls_saved'] is None
    assert row['S8_vs_F_verification_call_reduction_percent'] is None
    assert summary['status'] == 'partial_or_failed'
    assert overall(summary)['valid_requests'] < 12
    assert all('incomplete' in p['patterns'][0].lower() for p in summary['interpretations'])


def test_first_pair_survives_later_request_failure_when_cleanup_passes():
    result = ex.run_campaign(groups_fixture(), lambda group: FakeBackend(fail_at=4, outcome='better'))
    summary = report.summarize(result)
    assert summary['status'] == 'partial_or_failed'
    assert summary['comparisons'][0]['valid'] and not summary['comparisons'][1]['valid']
    row = overall(summary)
    assert row['selected_requests'] == 12 and row['valid_requests'] == 1
    assert row['S8_vs_F_verification_calls_saved'] == 1
    assert summary['accounting']['actual_native_generations'] == 3
    assert summary['accounting']['unknown_native_work_receipts'] == 1
    assert summary['accounting']['unattempted_native_generations'] == 20
    assert 'incomplete' in summary['interpretations'][0]['patterns'][0].lower()


def test_complete_first_pair_survives_second_pair_output_mismatch():
    class SecondMismatch(FakeBackend):
        def serve_request(self, child, label):
            row = super().serve_request(child, label)
            if len(self.calls) == 4:
                row['output_ids'][0] += 1
            return row
    result = ex.run_campaign(groups_fixture(), lambda group: SecondMismatch(outcome='better'))
    summary = report.summarize(result)
    assert summary['comparisons'][0]['valid'] and not summary['comparisons'][1]['valid']
    assert overall(summary)['valid_requests'] == 1 and summary['accounting']['returned_invalid_native_generations'] == 1


def test_cleanup_failure_excludes_both_otherwise_valid_pairs():
    result = ex.run_campaign(groups_fixture(), lambda group: FakeBackend(cleanup='weights', outcome='better'))
    summary = report.summarize(result)
    assert overall(summary)['valid_requests'] == 0
    assert overall(summary)['F_verification_calls'] is None
    assert overall(summary)['S8_vs_F_verification_calls_saved'] is None
    assert summary['accounting']['returned_valid_native_generations'] == 4


@pytest.mark.parametrize('problem', ['count', 'persistence', 'status'])
def test_global_execution_failures_preclude_complete_setting_transfer_claim(problem):
    result = complete_result(outcome='better')
    if problem == 'count': result['actual_native_generations'] = 23
    if problem == 'persistence': result['persistence_errors'] = [{'message': 'write failed'}]
    if problem == 'status': result['status'] = 'failed'
    summary = report.summarize(result)
    assert summary['status'] == 'partial_or_failed'
    assert all('incomplete' in p['patterns'][0].lower() for p in summary['interpretations'])


def test_unknown_and_returned_invalid_are_distinct_and_invalid_values_preserved():
    result = ex.run_campaign(groups_fixture(), lambda group: FakeBackend(corrupt='nonfinite'))
    summary = report.summarize(result)
    assert summary['accounting']['returned_invalid_native_generations'] == 1
    assert summary['accounting']['unknown_native_work_receipts'] == 0
    assert summary['comparisons'][0]['S8_request_s']['nonfinite_float'] == 'nan'
    assert overall(summary)['valid_requests'] == 0


def test_retained_context_is_separate_from_new_child_aggregate():
    summary = report.summarize(complete_result())
    context = summary['historical_context_pooled']
    parent = next(r for r in context if r['scope'] == 'overall' and r['population'] == 'parent_source')
    heldout = next(r for r in context if r['scope'] == 'overall' and r['population'] == 'ordinary_heldout')
    assert parent['available_pairs'] == 12 and heldout['available_pairs'] == 24
    assert heldout['F_verification_calls'] == 96 and heldout['S8_verification_calls'] == 120
    assert overall(summary)['F_verification_calls'] == 48
    assert all(r['execution'] == 'not_rerun' for r in summary['historical_context'])


def test_reports_retain_notes_counts_costs_and_no_speedup_metric(tmp_path):
    result = complete_result(outcome='better'); summary = report.write_reports(result, tmp_path)
    assert json.loads((tmp_path/'SUMMARY.json').read_text()) == summary
    assert {p.name for p in tmp_path.iterdir()} == {'SUMMARY.json', 'PAIRED_REQUESTS.csv', 'POOLED.csv',
        'RETAINED_CONTEXT.csv', 'RETAINED_CONTEXT_POOLED.csv', 'COSTS.csv', 'REPORT.md'}
    text = (tmp_path/'REPORT.md').read_text()
    assert 'does not establish speedup' in text and 'zero training' in text and 'never a mean' in text
    assert 'speedup' not in overall(summary)
    assert 'construction_method' in (tmp_path/'PAIRED_REQUESTS.csv').read_text()
    assert summary['costs'][0]['components']['fresh_request_timing']['recorded_values'] == 4


def test_zero_verification_call_pairs_have_defined_counts_without_division_by_zero():
    result = complete_result()
    replace_pair(result, 0, 0, output=[128009], f_calls=0, s_calls=0)
    row = report.summarize(result)['comparisons'][0]
    assert row['valid'] and row['S8_vs_F_native_outcome'] == 'tied'
    assert row['S8_vs_F_verification_calls_saved'] == 0
    assert row['S8_vs_F_verification_call_reduction_percent'] is None


def test_malformed_return_is_raw_preserved_and_reportable():
    class Malformed(FakeBackend):
        def serve_request(self, child, label):
            self.calls.append((label, child['request_id']))
            return ['malformed native response']
    result = ex.run_campaign(groups_fixture(), lambda group: Malformed())
    summary = report.summarize(result)
    assert result['groups'][0]['native_raw'] == [['malformed native response']]
    assert summary['accounting']['returned_invalid_native_generations'] == 1
    assert summary['status'] == 'partial_or_failed' and summary['valid_requests'] == 0


def test_executed_group_order_is_revalidated_in_report():
    result = complete_result(outcome='better'); result['groups'].reverse()
    summary = report.summarize(result)
    assert summary['status'] == 'partial_or_failed'
    assert 'executed_group_order_differs' in summary['accounting_errors']


def test_cli_provenance_and_native_timing_fields_are_retained_without_inference():
    result = complete_result()
    result.update(source_manifest={'path': '/private/SOURCE_MANIFEST.json', 'sha256': 'a'*64},
                  request_manifest={'path': '/private/REQUEST_MANIFEST.json', 'sha256': 'b'*64},
                  plan_sha256='c'*64)
    group = result['groups'][0]; pair = group['pairs'][0]
    timing = dict(native_generation_s=.2, cache_maintenance_s=.01,
        prefill_accounting=dict(input_tokens=len(pair['child']['input_ids']), initial_emitted_tokens=1,
            isolated_duration_s=None, duration_available=False,
            timing_scope='Prefill included in native_generation_s and request_s; standalone duration unavailable.'))
    pair['F']['native'].update(timing); group['native_raw'][0].update(copy.deepcopy(timing))
    summary = report.summarize(result)
    assert summary['provenance']['source_manifest'] == result['source_manifest']
    assert summary['provenance']['request_manifest'] == result['request_manifest']
    assert summary['provenance']['plan_sha256'] == 'c'*64
    row = summary['comparisons'][0]
    assert row['F_native_generation_s'] == .2 and row['F_cache_maintenance_s'] == .01
    assert row['F_prefill_accounting'] == timing['prefill_accounting']
    assert row['F_prefill_accounting']['isolated_duration_s'] is None
