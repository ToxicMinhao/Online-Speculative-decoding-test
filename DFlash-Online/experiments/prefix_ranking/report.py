"""Four-arm paired reporting with independently revalidated durable evidence."""

import csv
import json
from pathlib import Path

from online_feedback.storage import write_json
from experiments.osd_one_event.report import direction, reduction
from .protocol import (
    ARMS, REPEATS, COMPARISONS, COUNT_FIELDS, measurement_errors, record_errors, repeat_errors,
)


def validated_row(row, request, checkpoints, ordinary_counts, arms=ARMS):
    errors = list(row.get('errors', []))
    errors.extend(measurement_errors(row, request, checkpoints.get(row['method'])))
    if row['method'] not in arms or row['repeat'] not in range(REPEATS):
        errors.append('invalid_evaluation_slot')
    if row.get('valid_measurement') is not True:
        errors.append('execution_rejected_measurement')
    path = Path(row['record_path'])
    if not path.is_file():
        errors.append('native_record_unavailable')
    else:
        try:
            record = json.loads(path.read_text())
            errors.extend(record_errors(record, request, row['method'], row['checkpoint_id'], ordinary_counts))
            metadata = record['metadata']
            for name in ('verification_calls', 'accepted_draft_tokens', 'committed_progress',
                         'request_latency_s', 'native_generate_s'):
                if row.get(name) != metadata[name]:
                    errors.append('row_native_disagreement_' + name)
            if row.get('output_tokens') != len(record['response_ids']):
                errors.append('row_native_disagreement_output_tokens')
        except (KeyError, TypeError, ValueError, IndexError) as error:
            errors.append('malformed_native_record: ' + str(error))
    return {**row, 'report_errors': sorted(set(errors)), 'report_valid': not errors}


def summarize(rows, requests, *, checkpoints, ordinary_counts, arms=ARMS, comparisons=COMPARISONS):
    per_request, checked_rows, valid_rows = [], [], []
    for request in requests:
        selected = [validated_row(row, request, checkpoints, ordinary_counts, arms=arms)
                    for row in rows if row['request_id'] == request['request_id']]
        checked_rows.extend(selected)
        expected = {(method, repeat) for method in arms for repeat in range(REPEATS)}
        complete = (len(selected) == len(expected)
                    and {(row['method'], row['repeat']) for row in selected} == expected)
        errors = {error for row in selected for error in row['report_errors']}
        for method in arms:
            errors.update(repeat_errors(selected, method))
        if not complete:
            errors.add('incomplete_or_duplicate_slots')
        valid = complete and not errors
        result = {'request_id': request['request_id'], 'panel_ordinal': request['panel_ordinal'],
                  'complete': complete, 'valid': valid, 'errors': '; '.join(sorted(errors))}
        if valid:
            valid_rows.extend(selected)
            for method in arms:
                repeats = [row for row in selected if row['method'] == method]
                primary = next(row for row in repeats if row['repeat'] == 0)
                for key in COUNT_FIELDS:
                    result[f'{method}_{key}'] = primary[key]
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
        'primary_comparison': '_vs_'.join(comparisons[0]), 'timing_repeats_per_method_request': REPEATS,
        'cohort': 'One common cohort of complete, valid ' + '/'.join(arms) + ' requests for all comparisons.',
        'count_scope': 'One validated repeat-0 trajectory per method/request; timing repeats are not extra examples.',
        'latency_scope': 'Capture-disabled complete request with checkpoint available: prefill, draft, verification and request cache work; training and activation excluded.',
    }
    for method in arms:
        totals = {key: sum(row[f'{method}_{key}'] for row in valid_requests) for key in COUNT_FIELDS}
        totals['summed_mean_request_latency_s'] = sum(row[f'{method}_mean_request_latency_s'] for row in valid_requests)
        totals['all_repeats_request_latency_s'] = sum(row['request_latency_s'] for row in valid_rows if row['method'] == method)
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
            'request_time_reduction_pct': reduction(summary[reference]['all_repeats_request_latency_s'], summary[candidate]['all_repeats_request_latency_s']),
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
    return summary, per_request, checked_rows, valid_rows


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_results(rows, requests, root, *, checkpoints, ordinary_counts, execution,
                  arms=ARMS, comparisons=COMPARISONS):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary, per_request, checked_rows, valid_rows = summarize(
        rows, requests, checkpoints=checkpoints, ordinary_counts=ordinary_counts,
        arms=arms, comparisons=comparisons)
    summary['execution'] = execution
    write_json(root / 'SUMMARY.json', summary)
    write_csv(root / 'ALL_REPEATS.csv', checked_rows)
    write_csv(root / 'PER_REQUEST.csv', per_request)
    repeat_totals = []
    for repeat in range(REPEATS):
        for method in arms:
            selected = [row for row in valid_rows if row['repeat'] == repeat and row['method'] == method]
            repeat_totals.append({'repeat': repeat, 'method': method, 'requests': len(selected),
                                  'request_latency_s': sum(row['request_latency_s'] for row in selected),
                                  'verification_calls': sum(row['verification_calls'] for row in selected)})
    write_csv(root / 'REPEAT_TOTALS.csv', repeat_totals)
    title = ('# One-event prefix-ranking checkpoint comparison' if tuple(arms) == ARMS
             else '# One-event ' + '/'.join(arms) + ' checkpoint comparison')
    text = [
        title, '',
        f"Execution: **{execution['status']}**. Valid common requests: **{summary['valid_requests']}/{len(requests)}**.", '',
        'All values use fresh capture-disabled measurements on the same saved development inputs. '
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
    primary_candidate, primary_reference = comparisons[0]
    primary_note = f'**{primary_candidate} versus {primary_reference} is primary.** '
    if tuple(arms) == ARMS and tuple(comparisons) == COMPARISONS:
        primary_note += 'Beating U alone does not establish improvement over ordinary OSD. '
    elif primary_reference == 'O':
        primary_note += 'Other comparisons do not replace the ordinary OSD control. '
    text.extend([
        '', primary_note +
        'Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.', '',
        'Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, '
        'not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. '
        'Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.', '',
        f"Benchmark-only activation: {execution['activation_s']:.6f}s; native weight validation: "
        f"{execution['weight_validation_s']:.6f}s. These are not charged to inference-only latency. "
        'Adaptation and complete campaign costs belong to the separate run report.', '',
        'Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. '
        'Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.',
    ])
    (root / 'REPORT.md').write_text('\n'.join(text) + '\n')
    return summary
