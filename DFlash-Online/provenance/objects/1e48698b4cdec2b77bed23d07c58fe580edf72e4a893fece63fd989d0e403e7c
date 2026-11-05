"""Matched KRC2/K2 reports for one fixed panel, without cross-panel pooling."""

import json
from pathlib import Path
from statistics import median

from online_feedback.storage import write_json
from experiments.osd_one_event.report import direction, reduction
from .evaluation_protocol import (
    ARMS, REPEATS, COMPARISONS, COUNT_FIELDS, repeat_errors, record_errors, measurement_errors, prepare_requests,
)
from experiments.fresh_confirmation.report import write_csv


def validated_row(row, request, checkpoints):
    """A report independently checks every native row, including rejected rows."""
    errors = list(row.get('errors', []))
    errors.extend(measurement_errors(row, request))
    if row.get('method') not in ARMS or type(row.get('repeat')) is not int or row['repeat'] not in range(REPEATS):
        errors.append('invalid_evaluation_slot')
    if row.get('valid_measurement') is not True:
        errors.append('execution_rejected_or_pending_measurement')
    path = Path(row['record_path'])
    if not path.is_file():
        errors.append('native_record_unavailable')
    else:
        try:
            record = json.loads(path.read_text())
            errors.extend(record_errors(record, request, checkpoints[row['method']]))
            metadata = record['metadata']
            for name in ('verification_calls', 'accepted_draft_tokens', 'committed_progress',
                         'request_latency_s', 'native_generate_s'):
                if row.get(name) != metadata[name]:
                    errors.append('row_native_disagreement_' + name)
            if row.get('output_tokens') != len(record['response_ids']):
                errors.append('row_native_disagreement_output_tokens')
            if row.get('cache_maintenance_s') != metadata['capture']['cache_maintenance_s']:
                errors.append('row_native_disagreement_cache_maintenance_s')
            if row.get('checkpoint_id') != checkpoints[row['method']]:
                errors.append('checkpoint_identity')
            if (record['record_id'] != row['label']
                    or record['request_index'] != request['panel_ordinal']
                    or record['session_id'] != 'krc_continuation_evaluation'):
                errors.append('native_identity')
        except (KeyError, TypeError, ValueError, IndexError) as error:
            errors.append('malformed_native_record: ' + str(error))
    return {**row, 'report_errors': sorted(set(errors)), 'report_valid': not errors}


def summarize(rows, requests, *, checkpoints, expected_control_counts=None, historical_counts=None):
    requests = prepare_requests(requests, expected_control_counts)
    arms, comparisons = ARMS, COMPARISONS
    per_request, checked_rows, valid_rows = [], [], []
    for request in requests:
        raw_rows = [row for row in rows if row['request_id'] == request['request_id']]
        selected = [validated_row(row, request, checkpoints) for row in raw_rows]
        checked_rows.extend(selected)
        expected = {(method, repeat) for method in arms for repeat in range(REPEATS)}
        complete = (len(selected) == len(expected)
                    and {(row['method'], row['repeat']) for row in selected} == expected)
        errors = set()
        errors.update(error for row in selected for error in row['report_errors'])
        for method in arms:
            errors.update(repeat_errors(selected, method))
        if not complete:
            errors.add('incomplete_or_duplicate_slots')
        valid = complete and not errors
        result = {'request_id': request['request_id'], 'panel_ordinal': request['panel_ordinal'],
                  'complete': complete, 'valid': valid, 'errors': '; '.join(sorted(errors))}
        if valid:
            valid_rows.extend(selected)
            if historical_counts is not None:
                for method in ('K1', 'KRC1'):
                    result[f'{method}_historical_verification_calls'] = historical_counts[method][request['request_id']]
            result['canonical_output_tokens'] = len(request['response_ids'])
            result['canonical_finish_reason'] = request['finish_reason']
            for method in arms:
                repeats = [row for row in selected if row['method'] == method]
                primary = next(row for row in repeats if row['repeat'] == 0)
                for key in COUNT_FIELDS:
                    result[f'{method}_{key}'] = primary[key]
                for timer in ('request_latency_s', 'native_generate_s', 'cache_maintenance_s'):
                    result[f'{method}_mean_{timer}'] = sum(row[timer] for row in repeats) / REPEATS
                calls, tokens = primary['verification_calls'], primary['output_tokens']
                result[f'{method}_calls_per_1000_output_tokens'] = 1000 * calls / tokens if tokens else None
                result[f'{method}_accepted_draft_tokens_per_call'] = primary['accepted_draft_tokens'] / calls if calls else None
                result[f'{method}_committed_progress_per_call'] = primary['committed_progress'] / calls if calls else None
            for candidate, reference in comparisons:
                label = f'{candidate}_vs_{reference}'
                result[label + '_call_reduction_pct'] = reduction(
                    result[f'{reference}_verification_calls'], result[f'{candidate}_verification_calls'])
                result[label + '_request_time_reduction_pct'] = reduction(
                    result[f'{reference}_mean_request_latency_s'], result[f'{candidate}_mean_request_latency_s'])
                result[label + '_call_direction'] = direction(
                    result[f'{reference}_verification_calls'], result[f'{candidate}_verification_calls'])
                result[label + '_latency_direction'] = direction(
                    result[f'{reference}_mean_request_latency_s'], result[f'{candidate}_mean_request_latency_s'])
        per_request.append(result)

    valid_requests = [row for row in per_request if row['valid']]
    summary = {
        'planned_requests': len(requests), 'valid_requests': len(valid_requests),
        'invalid_or_incomplete_requests': len(requests) - len(valid_requests),
        'returned_measurements': len(rows), 'valid_timed_generations': len(valid_rows),
        'primary_comparison': '_vs_'.join(comparisons[0]), 'timing_repeats_per_method_request': REPEATS,
        'cohort': 'One common cohort of complete, valid ' + '/'.join(arms) + ' requests for all comparisons.',
        'count_scope': 'One validated repeat-0 trajectory per method/request; timing repeats are not extra examples.',
        'latency_scope': 'Capture-disabled complete request with checkpoint available: prefill, draft, verification and request cache work; training and activation excluded.',
    }
    for method in arms:
        totals = {key: sum(row[f'{method}_{key}'] for row in valid_requests) for key in COUNT_FIELDS}
        for timer in ('request_latency_s', 'native_generate_s', 'cache_maintenance_s'):
            totals[f'summed_mean_{timer}'] = sum(row[f'{method}_mean_{timer}'] for row in valid_requests)
            totals[f'all_repeats_{timer}'] = sum(row[timer] for row in valid_rows if row['method'] == method)
        calls, tokens = totals['verification_calls'], totals['output_tokens']
        totals['calls_per_1000_output_tokens'] = 1000 * calls / tokens if tokens else None
        totals['accepted_draft_tokens_per_call'] = totals['accepted_draft_tokens'] / calls if calls else None
        totals['committed_progress_per_call'] = totals['committed_progress'] / calls if calls else None
        summary[method] = totals
    summary['comparisons'] = {}
    for candidate, reference in comparisons:
        label = f'{candidate}_vs_{reference}'
        comparison = {
            'candidate': candidate, 'reference': reference,
            'call_reduction_pct': reduction(summary[reference]['verification_calls'], summary[candidate]['verification_calls']),
            'request_time_reduction_pct': reduction(summary[reference]['summed_mean_request_latency_s'], summary[candidate]['summed_mean_request_latency_s']),
        }
        for metric, key in (('call', 'verification_calls'), ('latency', 'mean_request_latency_s')):
            differences = [row[f'{reference}_{key}'] - row[f'{candidate}_{key}'] for row in valid_requests]
            comparison[metric + '_request_directions'] = {
                sign: sum(row[label + '_' + metric + '_direction'] == sign for row in valid_requests)
                for sign in ('improved', 'regressed', 'tied')}
            comparison[metric + '_gross_saved'] = sum(max(value, 0) for value in differences)
            comparison[metric + '_gross_extra'] = sum(max(-value, 0) for value in differences)
            comparison[metric + '_net_saved'] = sum(differences)
        summary['comparisons'][label] = comparison
    if historical_counts is not None:
        summary['historical_counts'] = {
            method: {'verification_calls': sum(row[f'{method}_historical_verification_calls'] for row in valid_requests),
                     'requests': len(valid_requests), 'timing_scope': 'No historical latency comparison'}
            for method in ('K1', 'KRC1')}
        summary['count_learning_curve'] = {}
        for candidate, reference in (('KRC1', 'K1'), ('KRC2', 'KRC1'), ('K2', 'K1')):
            candidate_key = (candidate+'_historical_verification_calls' if candidate.endswith('1')
                             else candidate+'_verification_calls')
            reference_key = reference+'_historical_verification_calls'
            candidate_calls = sum(row[candidate_key] for row in valid_requests)
            reference_calls = sum(row[reference_key] for row in valid_requests)
            deltas = [row[reference_key]-row[candidate_key] for row in valid_requests]
            summary['count_learning_curve'][candidate+'_vs_'+reference] = {
                'candidate_calls': candidate_calls, 'reference_calls': reference_calls,
                'call_reduction_pct': reduction(reference_calls, candidate_calls),
                'request_directions': {label: sum(direction(row[reference_key], row[candidate_key]) == label
                                                  for row in valid_requests)
                                       for label in ('improved', 'regressed', 'tied')},
                'gross_saved': sum(max(value, 0) for value in deltas),
                'gross_extra': sum(max(-value, 0) for value in deltas)}
        old = summary['count_learning_curve']['KRC1_vs_K1']['call_reduction_pct']
        new = summary['comparisons']['KRC2_vs_K2']['call_reduction_pct']
        summary['relative_advantage_change_percentage_points'] = None if old is None or new is None else new-old
    lengths = [row['canonical_output_tokens'] for row in valid_requests]
    summary['canonical_outputs'] = {
        'requests': len(lengths),
        'output_tokens': sum(lengths),
        'minimum_length': min(lengths) if lengths else None,
        'maximum_length': max(lengths) if lengths else None,
        'mean_length': sum(lengths) / len(lengths) if lengths else None,
        'median_length': median(lengths) if lengths else None,
        'eos_requests': sum(row['canonical_finish_reason'] == 'eos' for row in valid_requests),
        'length_terminated_requests': sum(row['canonical_finish_reason'] == 'length' for row in valid_requests),
        'output_cap_hits': sum(length == 2048 for length in lengths),
    }
    return summary, per_request, checked_rows, valid_rows


def write_results(rows, requests, root, *, checkpoints, execution, expected_control_counts=None, historical_counts=None):
    arms, comparisons = ARMS, COMPARISONS
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary, per_request, checked_rows, valid_rows = summarize(
        rows, requests, checkpoints=checkpoints, expected_control_counts=expected_control_counts, historical_counts=historical_counts)
    summary['execution'] = execution
    summary['timer_nesting'] = 'native_generate_s and cache_maintenance_s are components of request_latency_s; do not add them again.'
    write_json(root / 'SUMMARY.json', summary)
    write_csv(root / 'ALL_REPEATS.csv', checked_rows)
    write_csv(root / 'PER_REQUEST.csv', per_request)
    repeat_totals = []
    for repeat in range(REPEATS):
        for method in arms:
            selected = [row for row in valid_rows if row['repeat'] == repeat and row['method'] == method]
            repeat_totals.append({'repeat': repeat, 'method': method, 'requests': len(selected),
                                  **{timer: sum(row[timer] for row in selected) for timer in
                                     ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')},
                                  'verification_calls': sum(row['verification_calls'] for row in selected)})
    write_csv(root / 'REPEAT_TOTALS.csv', repeat_totals)
    title = '# KRC2/K2 event-2 continuation evaluation'
    text = [
        title, '',
        f"Execution: **{execution['status']}**. Valid common requests: **{summary['valid_requests']}/{len(requests)}**.", '',
        'KRC2/K2 values use fresh capture-disabled measurements on this fixed panel. '
        'All returned tokens and termination are checked against this panel’s retained canonical outputs. '
        'Request latency includes prefill, drafting, target verification and request cache maintenance. '
        'Training and activation are excluded; these are results with an available checkpoint.', '',
    ]
    text.extend(['| Arm | Verification calls | Sum of mean request times (s) | Calls / 1,000 tokens |',
                 '|---|---:|---:|---:|'])
    for method in arms:
        totals = summary[method]
        rate = totals['calls_per_1000_output_tokens']
        shown_rate = 'unavailable' if rate is None else f'{rate:.4f}'
        text.append(f"| {method} | {totals['verification_calls']} | "
                    f"{totals['summed_mean_request_latency_s']:.6f} | {shown_rate} |")
    text.extend(['', '| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |',
                 '|---|---:|---:|---:|'])
    for name, comparison in summary['comparisons'].items():
        values = [comparison['call_reduction_pct'], comparison['request_time_reduction_pct']]
        shown = ['unavailable' if value is None else f'{value:+.4f}%' for value in values]
        signs = comparison['call_request_directions']
        text.append(f"| {name} | {shown[0]} | {shown[1]} | {signs['improved']} / {signs['regressed']} / {signs['tied']} |")
    if historical_counts is not None:
        text.extend(['', '| Count-only history | Reference calls | Candidate calls | Reduction |',
                     '|---|---:|---:|---:|'])
        for label, value in summary['count_learning_curve'].items():
            gain = value['call_reduction_pct']
            shown = 'unavailable' if gain is None else f'{gain:+.4f}%'
            text.append(f"| {label} | {value['reference_calls']} | {value['candidate_calls']} | {shown} |")
        text.extend(['', 'Historical event-1 values are call counts on the same valid requests. '
                     'Relative KRC/K advantage and KRC’s own improvement are distinct; '
                     'historical-versus-fresh timings are not a learning curve.'])
    outputs = summary['canonical_outputs']
    text.extend(['', f"Common-cohort output lengths: {outputs['minimum_length']}–{outputs['maximum_length']}; "
                 f"EOS requests: {outputs['eos_requests']}; output-cap hits: {outputs['output_cap_hits']}.", '',
                 'Native generation and cache-maintenance timers are components of complete request latency, not additive costs. See the separate timer columns in the CSV/JSON tables.', '',
                 '| Arm | Accepted draft tokens | Committed progress | Accepted / call | Progress / call |',
                 '|---|---:|---:|---:|---:|'])
    for method in arms:
        totals = summary[method]
        rates = [totals['accepted_draft_tokens_per_call'], totals['committed_progress_per_call']]
        shown = ['unavailable' if rate is None else f'{rate:.4f}' for rate in rates]
        text.append(f"| {method} | {totals['accepted_draft_tokens']} | {totals['committed_progress']} | {shown[0]} | {shown[1]} |")
    primary_note = '**KRC2 versus K2 is the primary comparison.** The development panel requires retained K2 counts; the additional panel has no historical K2 count gate. No K1 or F generation or fresh latency is included. '
    text.extend([
        '', primary_note +
        'This fixed panel is reported separately from the other panel. Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.', '',
        'Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, '
        'not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. '
        'Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.', '',
        f"Benchmark-only activation: {execution['activation_s']:.6f}s; native weight validation: "
        f"{execution['weight_validation_s']:.6f}s. These are not charged to inference-only latency. "
        'KRC2 training, initial backend loading and complete campaign costs are reported separately. Historical K training is not a fresh matched cost control.', '',
        'Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. '
        'Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.',
    ])
    (root / 'REPORT.md').write_text('\n'.join(text) + '\n')
    return summary
