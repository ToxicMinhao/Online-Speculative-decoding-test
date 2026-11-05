"""Two-arm reports revalidate raw records against the first fresh K1 output."""

import json
from pathlib import Path

from online_feedback.storage import write_json
from experiments.osd_one_event.report import direction, reduction
from experiments.fresh_confirmation.report import write_csv
from .protocol import (
    ARMS, REPEATS, COMPARISONS, COUNT_FIELDS, measurement_errors, record_errors, repeat_errors,
)


def validated_row(row, request, checkpoints, canonical):
    errors = list(row.get('errors', []))
    errors.extend(measurement_errors(row, request, canonical))
    if row.get('method') not in ARMS or row.get('repeat') not in range(REPEATS):
        errors.append('invalid_evaluation_slot')
    if row.get('valid_measurement') is not True:
        errors.append('execution_rejected_or_pending_measurement')
    if canonical is None:
        errors.append('canonical_unavailable')
    path = Path(row['record_path'])
    if not path.is_file():
        errors.append('native_record_unavailable')
    else:
        try:
            record = json.loads(path.read_text())
            errors.extend(record_errors(record, request, checkpoints[row['method']], canonical))
            metadata = record['metadata']
            for name in ('verification_calls', 'accepted_draft_tokens', 'committed_progress',
                         'request_latency_s', 'native_generate_s'):
                if row.get(name) != metadata[name]:
                    errors.append('row_native_disagreement_' + name)
            if row.get('output_tokens') != len(record['response_ids']):
                errors.append('row_native_disagreement_output_tokens')
            if row.get('finish_reason') != record['finish_reason']:
                errors.append('row_native_disagreement_finish_reason')
            if (record['record_id'] != row['label']
                    or record['request_index'] != request['panel_ordinal']
                    or record['session_id'] != 'krc_confirmation_evaluation'):
                errors.append('native_identity')
            if row.get('checkpoint_id') != checkpoints[row['method']]:
                errors.append('checkpoint_identity')
        except (KeyError, TypeError, ValueError, IndexError) as error:
            errors.append('malformed_native_record: ' + str(error))
    return {**row, 'report_errors': sorted(set(errors)), 'report_valid': not errors}


def canonical_errors(canonical, request, rows, checkpoints):
    if canonical is None:
        return ['canonical_unavailable']
    sources = [row for row in rows if row['method'] == 'K1' and row['repeat'] == 0]
    if len(sources) != 1 or canonical['source_record_path'] != sources[0]['record_path']:
        return ['canonical_first_K1_binding']
    try:
        record = json.loads(Path(canonical['source_record_path']).read_text())
        errors = record_errors(record, request, checkpoints['K1'])
        if any(canonical[key] != record[key] for key in ('response_ids', 'finish_reason')):
            errors.append('canonical_source_output')
        if canonical['source_method'] != 'K1' or canonical['source_repeat'] != 0:
            errors.append('canonical_source_slot')
        return errors
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        return ['canonical_source_unavailable_or_malformed: ' + str(error)]


def summarize(rows, requests, *, checkpoints, canonicals):
    arms, comparisons = ARMS, COMPARISONS
    per_request, checked_rows, valid_rows = [], [], []
    for request in requests:
        raw_rows = [row for row in rows if row['request_id'] == request['request_id']]
        canonical = canonicals.get(request['request_id'])
        canonical_issues = canonical_errors(canonical, request, raw_rows, checkpoints)
        selected = [validated_row(row, request, checkpoints, canonical) for row in raw_rows]
        checked_rows.extend(selected)
        expected = {(method, repeat) for method in arms for repeat in range(REPEATS)}
        complete = (len(selected) == len(expected)
                    and {(row['method'], row['repeat']) for row in selected} == expected)
        errors = set(canonical_issues)
        errors.update(error for row in selected for error in row['report_errors'])
        for method in arms:
            errors.update(repeat_errors(selected, method))
        if not complete:
            errors.add('incomplete_or_duplicate_slots')
        valid = complete and not errors
        result = {'request_id': request['request_id'], 'panel_ordinal': request['panel_ordinal'],
                  'original_index': request['original_index'],
                  'complete': complete, 'valid': valid, 'errors': '; '.join(sorted(errors))}
        if valid:
            valid_rows.extend(selected)
            for method in arms:
                repeats = [row for row in selected if row['method'] == method]
                primary = next(row for row in repeats if row['repeat'] == 0)
                for key in COUNT_FIELDS:
                    result[f'{method}_{key}'] = primary[key]
                result[f'{method}_finish_reason'] = primary['finish_reason']
                for row in repeats:
                    result[f"{method}_repeat_{row['repeat']}_request_latency_s"] = row['request_latency_s']
                result[f'{method}_mean_request_latency_s'] = sum(row['request_latency_s'] for row in repeats) / REPEATS
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
        'all_planned_requests_valid': len(valid_requests) == len(requests),
        'primary_comparison': '_vs_'.join(comparisons[0]), 'timing_repeats_per_method_request': REPEATS,
        'cohort': 'One common cohort of complete, valid ' + '/'.join(arms) + ' requests for all comparisons.',
        'count_scope': 'One validated repeat-0 trajectory per method/request; timing repeats are not extra examples.',
        'latency_scope': 'Capture-disabled complete request with checkpoint available: prefill, draft, verification and request cache work; training and activation excluded.',
    }
    for method in arms:
        totals = {key: sum(row[f'{method}_{key}'] for row in valid_requests) for key in COUNT_FIELDS}
        totals['summed_mean_request_latency_s'] = sum(row[f'{method}_mean_request_latency_s'] for row in valid_requests)
        totals['all_repeats_request_latency_s'] = sum(row['request_latency_s'] for row in valid_rows if row['method'] == method)
        lengths = [row[f'{method}_output_tokens'] for row in valid_requests]
        totals['output_length_min'] = min(lengths, default=None)
        totals['output_length_max'] = max(lengths, default=None)
        totals['output_length_mean'] = sum(lengths) / len(lengths) if lengths else None
        totals['terminations'] = {
            reason: sum(row[f'{method}_finish_reason'] == reason for row in valid_requests)
            for reason in ('eos', 'length')
        }
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
    omissions = leave_one_out(per_request)
    call_reductions = [row['call_reduction_pct'] for row in omissions if row['call_reduction_pct'] is not None]
    summary['leave_one_out'] = {
        'scope': 'Descriptive leave-one-request-out sensitivity; the primary result retains every valid request.',
        'valid_source_requests': len(valid_requests),
        'omissions': len(omissions),
        'minimum_call_reduction_pct': min(call_reductions, default=None),
        'maximum_call_reduction_pct': max(call_reductions, default=None),
        'nonpositive_call_reduction_omissions': sum(value <= 0 for value in call_reductions),
    }
    return summary, per_request, checked_rows, valid_rows


def leave_one_out(per_request):
    """Remove each request once for sensitivity, without altering the primary pool."""
    valid = [row for row in per_request if row['valid']]
    totals = {
        method: {
            key: sum(row[f'{method}_{key}'] for row in valid)
            for key in ('verification_calls', 'mean_request_latency_s')
        }
        for method in ARMS
    }
    rows = []
    for omitted in valid:
        row = {
            'omitted_request_id': omitted['request_id'],
            'omitted_panel_ordinal': omitted['panel_ordinal'],
            'remaining_requests': len(valid) - 1,
        }
        for method in ARMS:
            for key, value in totals[method].items():
                row[f'{method}_{key}'] = value - omitted[f'{method}_{key}']
        row['net_calls_saved'] = row['K1_verification_calls'] - row['KRC_verification_calls']
        row['call_reduction_pct'] = reduction(row['K1_verification_calls'], row['KRC_verification_calls'])
        row['request_time_reduction_pct'] = reduction(
            row['K1_mean_request_latency_s'], row['KRC_mean_request_latency_s'])
        rows.append(row)
    return rows


def write_results(rows, requests, root, *, checkpoints, canonicals, execution):
    arms, comparisons = ARMS, COMPARISONS
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary, per_request, checked_rows, valid_rows = summarize(
        rows, requests, checkpoints=checkpoints, canonicals=canonicals)
    summary['execution'] = execution
    write_json(root / 'SUMMARY.json', summary)
    write_csv(root / 'ALL_REPEATS.csv', checked_rows)
    write_csv(root / 'PER_REQUEST.csv', per_request)
    write_csv(root / 'LEAVE_ONE_OUT.csv', leave_one_out(per_request))
    repeat_totals = []
    for repeat in range(REPEATS):
        for method in arms:
            selected = [row for row in valid_rows if row['repeat'] == repeat and row['method'] == method]
            repeat_totals.append({'repeat': repeat, 'method': method, 'requests': len(selected),
                                  'request_latency_s': sum(row['request_latency_s'] for row in selected),
                                  'verification_calls': sum(row['verification_calls'] for row in selected)})
    write_csv(root / 'REPEAT_TOTALS.csv', repeat_totals)
    title = '# Fresh K1/KRC checkpoint confirmation'
    text = [
        title, '',
        f"Execution: **{execution['status']}**. Valid common requests: **{summary['valid_requests']}/{len(requests)}**.", '',
        'All values use fresh capture-disabled measurements on inputs new to the current K1/KRC study. '
        'Older project exposure is disclosed in the input manifest. '
        'Request latency includes prefill, drafting, target verification and request cache maintenance. '
        'Training and activation are excluded; these are results with an available checkpoint.', '',
    ]
    text.extend(['| Arm | Verification calls | Accepted draft tokens | Output tokens | EOS / capped | Sum of mean times (s) | Calls / 1,000 tokens |',
                 '|---|---:|---:|---:|---:|---:|---:|'])
    for method in arms:
        totals = summary[method]
        rate = totals['calls_per_1000_output_tokens']
        shown_rate = 'unavailable' if rate is None else f'{rate:.4f}'
        text.append(f"| {method} | {totals['verification_calls']} | {totals['accepted_draft_tokens']} | "
                    f"{totals['output_tokens']} | {totals['terminations']['eos']} / {totals['terminations']['length']} | "
                    f"{totals['summed_mean_request_latency_s']:.6f} | {shown_rate} |")
    text.extend(['', '| Comparison | Verification-call reduction | Request-time reduction | Calls improved / regressed / tied |',
                 '|---|---:|---:|---:|'])
    for name, comparison in summary['comparisons'].items():
        values = [comparison['call_reduction_pct'], comparison['request_time_reduction_pct']]
        shown = ['unavailable' if value is None else f'{value:+.4f}%' for value in values]
        signs = comparison['call_request_directions']
        text.append(f"| {name} | {shown[0]} | {shown[1]} | {signs['improved']} / {signs['regressed']} / {signs['tied']} |")
    primary_note = '**KRC versus K1 is the sole comparison.** First fresh K1 establishes canonical tokens and termination; no historical call count is a gate. '
    comparison = summary['comparisons']['KRC_vs_K1']
    sensitivity = summary['leave_one_out']
    low = sensitivity['minimum_call_reduction_pct']
    high = sensitivity['maximum_call_reduction_pct']
    sensitivity_text = 'Leave-one-out call reductions are unavailable.'
    if low is not None and high is not None:
        sensitivity_text = (f"Leaving each request out once gives call reductions from {low:+.4f}% to {high:+.4f}%. "
                            f"{sensitivity['nonpositive_call_reduction_omissions']}/{sensitivity['omissions']} omissions are nonpositive. "
                            'This sensitivity analysis does not replace the complete-panel primary result.')
    text.extend([
        '', primary_note +
        'Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.', '',
        'Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, '
        'not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. '
        'Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.', '',
        f"Gross calls saved: {comparison['call_gross_saved']}; gross calls added: {comparison['call_gross_extra']}; "
        f"net saved: {comparison['call_net_saved']}. {sensitivity_text}", '',
        f"Benchmark-only activation: {execution['activation_s']:.6f}s across {execution['activation_count']} publications; native weight validation: "
        f"{execution['weight_validation_s']:.6f}s. These are not charged to inference-only latency. "
        'There is no new adaptation. Initial backend loading and complete campaign costs are reported separately.', '',
        'The canonical agreement gate uses the native verifier and is not an independent target-only reference. '
        'This fixed-checkpoint panel tests input transfer; it does not establish robustness across training seeds or feedback windows.', '',
        'Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv, LEAVE_ONE_OUT.csv and SUMMARY.json. '
        'Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.',
    ])
    (root / 'REPORT.md').write_text('\n'.join(text) + '\n')
    return summary
