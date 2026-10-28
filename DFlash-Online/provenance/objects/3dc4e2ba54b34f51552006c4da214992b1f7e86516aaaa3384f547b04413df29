"""Paired descriptive results, retaining every repeat and failed comparison."""

import csv
from pathlib import Path

from online_feedback.storage import write_json
from .protocol import ARMS, REPEATS


def reduction(frozen, updated):
    return 100 * (1 - updated / frozen) if frozen else None


def direction(frozen, updated):
    return 'improved' if updated < frozen else 'regressed' if updated > frozen else 'tied'


def summarize(rows, requests):
    per_request = []
    valid_rows = []
    for request in requests:
        selected = [row for row in rows if row['request_id'] == request['request_id']]
        slots = {(row['method'], row['repeat']) for row in selected}
        expected = {(method, repeat) for method in ARMS for repeat in range(REPEATS)}
        complete = len(selected) == len(expected) and slots == expected
        repeated = [method for method in ARMS if len({
            row['verification_calls'] for row in selected if row['method'] == method}) > 1]
        errors = {error for row in selected for error in row['errors']}
        if repeated:
            errors.add('repeat_count_mismatch')
        valid = complete and not repeated and all(row['valid_measurement'] for row in selected)
        result = {'request_id': request['request_id'], 'panel_ordinal': request['panel_ordinal'],
                  'complete': complete, 'valid': valid,
                  'errors': '; '.join(sorted(errors))}
        if valid:
            valid_rows.extend(selected)
            for method in ARMS:
                repeats = [row for row in selected if row['method'] == method]
                primary = next(row for row in repeats if row['repeat'] == 0)
                for key in ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens'):
                    result[method + '_' + key] = primary[key]
                result[method + '_mean_request_latency_s'] = sum(row['request_latency_s'] for row in repeats) / REPEATS
            result['call_reduction_pct'] = reduction(result['F_verification_calls'], result['O_verification_calls'])
            result['request_time_reduction_pct'] = reduction(result['F_mean_request_latency_s'], result['O_mean_request_latency_s'])
            result['call_direction'] = direction(result['F_verification_calls'], result['O_verification_calls'])
            result['latency_direction'] = direction(result['F_mean_request_latency_s'], result['O_mean_request_latency_s'])
        per_request.append(result)

    valid_requests = [row for row in per_request if row['valid']]
    summary = {
        'planned_requests': len(requests), 'valid_requests': len(valid_requests),
        'invalid_or_incomplete_requests': len(requests) - len(valid_requests),
        'valid_timed_generations': len(valid_rows),
        'count_scope': 'One validated trajectory (repeat 0) per method/request; repeats are not extra examples.',
        'latency_scope': 'Complete capture-disabled request with checkpoint available; excludes training and activation.',
        'timing_repeats_per_method_request': REPEATS,
    }
    for method in ARMS:
        totals = {key: sum(row[method + '_' + key] for row in valid_requests) for key in (
            'verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')}
        totals['summed_mean_request_latency_s'] = sum(row[method + '_mean_request_latency_s'] for row in valid_requests)
        totals['all_repeats_request_latency_s'] = sum(row['request_latency_s'] for row in valid_rows if row['method'] == method)
        calls, tokens = totals['verification_calls'], totals['output_tokens']
        totals['calls_per_1000_output_tokens'] = 1000 * calls / tokens if tokens else None
        totals['accepted_draft_tokens_per_call'] = totals['accepted_draft_tokens'] / calls if calls else None
        totals['committed_progress_per_call'] = totals['committed_progress'] / calls if calls else None
        summary[method] = totals
    summary['call_reduction_pct'] = reduction(summary['F']['verification_calls'], summary['O']['verification_calls'])
    summary['request_time_reduction_pct'] = reduction(
        summary['F']['all_repeats_request_latency_s'], summary['O']['all_repeats_request_latency_s'])
    for metric in ('call', 'latency'):
        summary[metric + '_request_directions'] = {
            value: sum(row[metric + '_direction'] == value for row in valid_requests)
            for value in ('improved', 'regressed', 'tied')}
    return summary, per_request


def write_results(rows, requests, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary, per_request = summarize(rows, requests)
    write_json(root / 'SUMMARY.json', summary)
    for name, table in (('ALL_REPEATS.csv', rows), ('PER_REQUEST.csv', per_request)):
        if not table:
            continue
        fields = list(dict.fromkeys(key for row in table for key in row))
        with (root / name).open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    return summary
