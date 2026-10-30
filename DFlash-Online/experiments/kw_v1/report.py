"""KW-v1 raw-record audit, paired request analysis and full campaign reporting."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re

from online_feedback.storage import write_json
from experiments.osd_one_event.report import direction
from experiments.ok_events_5_8.report import validate_execution_source, _load, _local_records
from .evaluation_protocol import (
    ARMS, REPEATS, COMPARISONS, COUNT_FIELDS, measurement_errors, record_errors, repeat_errors,
    GENERATIONS, TOTAL_GENERATIONS, prepare_requests, evaluation_schedule, validate_panels,
    publication_errors, weight_check_errors, method_order,
)


def reduction(reference, candidate):
    """Use the sealed ratio of summed differences, including the timing ratios."""
    return 100 * (reference - candidate) / reference if reference else None


def validated_row(row, request, checkpoints, canonical):
    errors = list(row.get('errors', []))
    errors.extend(measurement_errors(row, request, canonical))
    if (row.get('method') not in ARMS or type(row.get('repeat')) is not int
            or row['repeat'] not in range(REPEATS)):
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
            if row.get('cache_maintenance_s') != metadata['capture']['cache_maintenance_s']:
                errors.append('row_native_disagreement_cache_maintenance_s')
            if (record['record_id'] != row['label'] or record['request_index'] != request['panel_ordinal']
                    or record['session_id'] != 'kw_v1_evaluation'):
                errors.append('native_identity')
            if row.get('output_tokens') != len(record['response_ids']):
                errors.append('row_native_disagreement_output_tokens')
            if row.get('checkpoint_id') != checkpoints[row['method']]:
                errors.append('checkpoint_identity')
        except (KeyError, TypeError, ValueError, IndexError) as error:
            errors.append('malformed_native_record: ' + str(error))
    return {**row, 'report_errors': sorted(set(errors)), 'report_valid': not errors}


def canonical_errors(canonical, request, rows, checkpoints, canonical_mode):
    if canonical is None:
        return ['canonical_unavailable']
    if canonical_mode == 'retained':
        if (canonical.get('source') != 'retained_panel_canonical'
                or canonical.get('source_method') != request['canonical_source_method']
                or any(canonical.get(key) != request[key] for key in ('response_ids', 'finish_reason'))):
            return ['retained_canonical_binding']
        return []
    sources = [row for row in rows if row['method'] == 'O' and row['repeat'] == 0]
    if len(sources) != 1 or canonical['source_record_path'] != sources[0]['record_path']:
        return ['canonical_first_O_binding']
    try:
        record = json.loads(Path(canonical['source_record_path']).read_text())
        errors = record_errors(record, request, checkpoints['O'])
        if any(canonical[key] != record[key] for key in ('response_ids', 'finish_reason')):
            errors.append('canonical_source_output')
        if canonical['source_method'] != 'O' or canonical['source_repeat'] != 0:
            errors.append('canonical_source_slot')
        return errors
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        return ['canonical_source_unavailable_or_malformed: ' + str(error)]


def summarize(rows, requests, *, checkpoints, canonicals, canonical_mode):
    arms, comparisons = ARMS, COMPARISONS
    per_request, checked_rows, valid_rows = [], [], []
    for request in requests:
        raw_rows = [row for row in rows if row['request_id'] == request['request_id']]
        canonical = canonicals.get(request['request_id'])
        canonical_issues = canonical_errors(canonical, request, raw_rows, checkpoints, canonical_mode)
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
                  'complete': complete, 'valid': valid, 'errors': '; '.join(sorted(errors))}
        if valid:
            valid_rows.extend(selected)
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
            totals['summed_mean_' + timer] = sum(row[f'{method}_mean_{timer}'] for row in valid_requests)
            totals['all_repeats_' + timer] = sum(row[timer] for row in valid_rows if row['method'] == method)
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
            'native_generate_time_reduction_pct': reduction(summary[reference]['summed_mean_native_generate_s'], summary[candidate]['summed_mean_native_generate_s']),
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


def write_results(rows, requests, root, *, checkpoints, canonicals, execution, event_number,
                  panel_name, canonical_mode):
    arms, comparisons = ARMS, COMPARISONS
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary, per_request, checked_rows, valid_rows = summarize(
        rows, requests, checkpoints=checkpoints, canonicals=canonicals, canonical_mode=canonical_mode)
    summary.update(execution=execution, event_number=event_number, panel_name=panel_name,
                   canonical_mode=canonical_mode, historical_count_gate=False)
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
    title = f'# O{event_number}/K{event_number}/KW{event_number}: {panel_name} panel'
    text = [
        title, '',
        f"Execution: **{execution['status']}**. Valid common requests: **{summary['valid_requests']}/{len(requests)}**.", '',
        'All values use fresh capture-disabled measurements at this checkpoint endpoint. '
        'Retained panels use their original canonical outputs; fresh uses its first O12 output. '
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
    primary_note = f'**KW versus K is the primary component comparison; O comparisons are shown too.** Canonical mode: {canonical_mode}; no historical call count is a gate. '
    text.extend([
        '', primary_note +
        'Null and harmful results remain in all valid totals. No outcome-dependent rerun or tuning is implied.', '',
        'Counts use one validated repeat-0 trajectory per request. Three timings are repeated measurements, '
        'not independent examples. Percent reductions are ratios of pooled sums, not averages of per-request ratios. '
        'Invalid and incomplete requests remain in PER_REQUEST.csv and are excluded from gain calculations.', '',
        f"Benchmark-only activation: {execution['activation_s']:.6f}s; native weight validation: "
        f"{execution['weight_validation_s']:.6f}s. These are not charged to inference-only latency. "
        'Update time and initial backend loading are reported separately from serving latency.', '',
        'Files: ALL_REPEATS.csv, PER_REQUEST.csv, REPEAT_TOTALS.csv and SUMMARY.json. '
        'Raw native records, verification trajectories and validation receipts remain in the parent evaluation directory.',
    ])
    (root / 'REPORT.md').write_text('\n'.join(text) + '\n')
    return summary


def paired_bootstrap(reference_calls, candidate_calls):
    """The sealed primary: 64 pairs, PCG64/20261030, 10,000 ratio-of-sums draws."""
    import numpy as np

    if (len(reference_calls) != 64 or len(candidate_calls) != 64
            or any(type(value) is not int or value < 0
                   for values in (reference_calls, candidate_calls) for value in values)):
        raise ValueError('Primary bootstrap requires exactly 64 nonnegative integer request-call pairs')
    reference = np.asarray(reference_calls, dtype=np.int64)
    candidate = np.asarray(candidate_calls, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(20261030))
    indices = rng.integers(0, 64, size=(10000, 64))
    reference_sums = reference[indices].sum(axis=1)
    candidate_sums = candidate[indices].sum(axis=1)
    if np.any(reference_sums <= 0):
        raise ValueError('A fixed bootstrap resample has zero reference calls; statistic is undefined')
    samples = 100.0 * (reference_sums - candidate_sums) / reference_sums
    interval = np.percentile(samples, [2.5, 97.5], method='linear')
    return dict(request_pairs=64, resamples=10000, bit_generator='PCG64', seed=20261030,
                statistic='100 * (sum(K calls) - sum(KW calls)) / sum(K calls)',
                estimate_pct=100.0 * (int(reference.sum()) - int(candidate.sum())) / int(reference.sum()),
                percentile_95_ci_pct=interval.tolist(), percentile_method='linear',
                scope='Conditional on this run/checkpoint and request panel; no training-seed inference.')


def outcome_label(*, valid, primary_reduction, pooled_reduction, interval=None,
                  serving_reductions=(), costs_complete=True, overhead_prevents_efficiency=False):
    """Negative primary precedence is fixed before looking at other panels."""
    if not valid:
        return 'Implementation invalid'
    if primary_reduction is None or pooled_reduction is None:
        return 'Mixed/inconclusive'
    if primary_reduction <= 0:
        return 'Unsupported v1 in this run'
    if pooled_reduction <= 0:
        return 'Mixed/inconclusive'
    if (interval is None or interval[0] <= 0 or not costs_complete
            or overhead_prevents_efficiency
            or not serving_reductions or any(value is None or value <= 0 for value in serving_reductions)):
        return 'Mixed/inconclusive'
    return 'Promising v1'


def aggregate_requests(per_request):
    """Pool disjoint final request pairs while retaining every panel separately."""
    selected = [row for row in per_request if row['valid']]
    result = dict(valid_requests=len(selected), comparisons={})
    for method in ARMS:
        totals = {key: sum(row[f'{method}_{key}'] for row in selected) for key in COUNT_FIELDS}
        for timer in ('request_latency_s', 'native_generate_s', 'cache_maintenance_s'):
            totals['summed_mean_' + timer] = sum(row[f'{method}_mean_{timer}'] for row in selected)
        calls, tokens = totals['verification_calls'], totals['output_tokens']
        totals['calls_per_1000_output_tokens'] = 1000 * calls / tokens if tokens else None
        totals['accepted_draft_tokens_per_call'] = totals['accepted_draft_tokens'] / calls if calls else None
        totals['committed_progress_per_call'] = totals['committed_progress'] / calls if calls else None
        result[method] = totals
    for candidate, reference in COMPARISONS:
        differences = [row[f'{reference}_verification_calls'] - row[f'{candidate}_verification_calls']
                       for row in selected]
        result['comparisons'][candidate + '_vs_' + reference] = dict(
            candidate=candidate, reference=reference,
            call_reduction_pct=reduction(result[reference]['verification_calls'], result[candidate]['verification_calls']),
            request_time_reduction_pct=reduction(result[reference]['summed_mean_request_latency_s'],
                                                result[candidate]['summed_mean_request_latency_s']),
            native_generate_time_reduction_pct=reduction(result[reference]['summed_mean_native_generate_s'],
                                                        result[candidate]['summed_mean_native_generate_s']),
            call_request_directions=dict(improved=sum(value > 0 for value in differences),
                                         regressed=sum(value < 0 for value in differences),
                                         tied=sum(value == 0 for value in differences)),
            call_gross_saved=sum(max(value, 0) for value in differences),
            call_gross_extra=sum(max(-value, 0) for value in differences),
            call_net_saved=sum(differences))
    return result


def panel_binding_errors(directory, campaign, requests, rows, progress, protocol, event, name, mode):
    """Recheck saved ordering, publication, endpoint and work identities."""
    errors = []
    checkpoints = protocol.get('checkpoints', {})
    for arm in ARMS:
        endpoint = _load(campaign / arm / 'events' / f'event_{event:03d}/RESULT.json', {})
        if endpoint.get('export_path') != checkpoints.get(arm) or not checkpoints.get(arm):
            errors.append('own_event_endpoint_binding_' + arm)
        receipt = _load(directory / ('WEIGHTS_' + arm + '.json'), {})
        try:
            if receipt.get('errors') != [] or weight_check_errors(receipt['native_reply'], checkpoints[arm]):
                errors.append('native_weight_equality_' + arm)
        except (KeyError, TypeError, ValueError):
            errors.append('native_weight_equality_' + arm)
    expected_schedule = evaluation_schedule(requests)
    if (protocol.get('event_number') != event or protocol.get('panel_name') != name
            or protocol.get('canonical_mode') != mode or protocol.get('historical_count_gate') is not False
            or protocol.get('arms') != list(ARMS) or protocol.get('repeats') != REPEATS
            or protocol.get('schedule') != expected_schedule
            or protocol.get('extra_warmup_generations') != 0 or protocol.get('capture') is not False):
        errors.append('evaluation_protocol_binding')
    if _load(directory / 'REQUESTS.json') != requests:
        errors.append('sealed_request_manifest_binding')
    actual_slots = [{key: row.get(key) for key in ('panel_ordinal', 'request_id', 'repeat', 'method')}
                    for row in rows]
    expected_slots = [{key: row[key] for key in ('panel_ordinal', 'request_id', 'repeat', 'method')}
                      for row in expected_schedule]
    if actual_slots != expected_slots[:len(actual_slots)]:
        errors.append('actual_slot_order')
    if len(rows) != progress.get('returned_generations', 0):
        errors.append('returned_measurement_count')
    if progress.get('status') == 'completed':
        for key in ('attempted_generations', 'returned_generations', 'completed_valid_generations', 'planned_generations'):
            if progress.get(key) != GENERATIONS:
                errors.append('completed_' + key)
        for key in ('invalid_generations', 'pending_canonical_generations', 'unknown_generation_attempts',
                    'unattempted_generations', 'optimizer_steps', 'feedback_collections'):
            if progress.get(key) != 0:
                errors.append('completed_' + key)
    activations = directory / 'ACTIVATIONS.jsonl'
    if activations.is_file():
        for line in activations.read_text().splitlines():
            try:
                receipt = json.loads(line)
                if receipt.get('errors') != [] or publication_errors(receipt, checkpoints[receipt['method']]):
                    errors.append('native_publication')
            except (KeyError, TypeError, ValueError):
                errors.append('native_publication')
    else:
        errors.append('native_publication_receipts_unavailable')
    if event == 10 and name == 'development':
        gate = _load(directory / 'QUALIFICATION.json', {})
        if (gate.get('status') != 'passed' or gate.get('requests') != 8
                or gate.get('generations') != 72 or gate.get('extra_generations') != 0
                or gate.get('arms_with_native_equality') != sorted(ARMS)
                or progress.get('qualification') != 'passed'):
            errors.append('included_event10_gate')
    elif (directory / 'QUALIFICATION.json').exists() or progress.get('qualification_requests', 0) != 0:
        errors.append('unexpected_generation_gate')
    return sorted(set(errors))


def campaign_costs(campaign, costs):
    """Preserve measured cost receipts, with no invented zero for unavailable work."""
    events, totals = [], {}
    for arm in ARMS:
        training = []
        for event in range(9, 13):
            receipt = _load(campaign / arm / 'events' / f'event_{event:03d}/RESULT.json', {})
            recorded = costs.get(f'event_{event:03d}_{arm}', {})
            item = dict(event=event, arm=arm, campaign_costs=recorded,
                        training_receipt_costs=receipt.get('costs'),
                        controller_work=receipt.get('kw'), objective_work=receipt.get('objective_work'))
            events.append(item)
            value = recorded.get('training_s')
            if type(value) in (int, float) and math.isfinite(value) and value > 0:
                training.append(value)
        totals[arm] = dict(recorded_training_events=len(training),
                           training_s=sum(training) if len(training) == 4 else None)
    k_time, kw_time = totals['K']['training_s'], totals['KW']['training_s']
    controller_costs = {}
    for key in ('ordinary_training_s', 'capture_host_s', 'credit_and_controller_s',
                'previews_s', 'weighting_s', 'live_update_s', 'window_observed_s'):
        values = [(event['controller_work'] or {}).get('costs', {}).get(key)
                  for event in events if event['arm'] == 'KW']
        controller_costs[key] = (sum(values) if all(type(value) in (int, float)
            and math.isfinite(value) and value >= 0 for value in values) else None)
    return dict(raw_campaign_costs=costs, events=events, arm_totals=totals,
                KW_controller_costs=controller_costs,
                KW_vs_K_training_overhead_pct=100 * (kw_time - k_time) / k_time if k_time and kw_time else None,
                complete_training_costs=all(totals[arm]['recorded_training_events'] == 4 for arm in ARMS),
                scope='Training/controller costs are separate from serving latency. Event/load/save/export timers overlap their components; no asynchronous deployment amortization is measured.')


def work_errors(status, summaries):
    expected = dict(feedback_attempted=512, feedback_returned=512, optimizer_steps=1536,
                    presentations=3072, evaluation_attempted=4032, evaluation_returned=4032,
                    evaluation_valid=4032, optimizer_previews=1024, credit_opportunities=504)
    errors = [key for key, value in expected.items() if type(status.get(key)) is not int or status[key] != value]
    if status.get('status') != 'completed':
        errors.append('campaign_not_completed')
    if sum(value['returned_measurements'] for value in summaries.values()) != TOTAL_GENERATIONS:
        errors.append('raw_evaluation_generation_total')
    if status.get('arm_counters') != {arm: 1536 for arm in ARMS}:
        errors.append('cumulative_arm_steps')
    if status.get('arm_presentations') != {arm: 3072 for arm in ARMS}:
        errors.append('cumulative_arm_presentations')
    for key in ('extra_optimizer_steps', 'extra_generations', 'additional_controller_model_passes'):
        if type(status.get(key)) is not int or status[key] != 0:
            errors.append(key)
    expected_kw_work = dict(previews=1024, live_steps=512, credits=504)
    for field in ('kw_work_attempts', 'kw_work_completed_lower_bound'):
        value = status.get(field)
        if (not isinstance(value, dict) or value != expected_kw_work
                or any(type(count) is not int for count in value.values())):
            errors.append(field)
    unknown = status.get('kw_work_unknown')
    if (not isinstance(unknown, dict) or unknown != dict.fromkeys(expected_kw_work, 0)
            or any(type(count) is not int for count in unknown.values())):
        errors.append('kw_work_unknown')
    return errors


def training_receipt_errors(campaign):
    """Cross-check counted production work against every completed native event."""
    errors = []
    for event in range(9, 13):
        for arm in ARMS:
            receipt = _load(campaign / arm / 'events' / f'event_{event:03d}/RESULT.json', {})
            expected = dict(event_id=f'event_{event:03d}', status='completed',
                start_step=(event - 1) * 128, end_step=event * 128,
                optimizer_steps=128, response_presentations=256)
            if any(receipt.get(key) != value for key, value in expected.items()):
                errors.append(f'event_{event:03d}_{arm}_work_receipt')
            if arm == 'KW':
                controller = receipt.get('kw', {})
                if (controller.get('windows') != 128 or controller.get('previews') != 256
                        or controller.get('credit_opportunities') != 126
                        or controller.get('controller_final', {}).get('n') != (event - 8) * 126):
                    errors.append(f'event_{event:03d}_KW_controller_work_receipt')
    return errors


def qualification_errors(qualification, *, expected_bindings=None):
    errors = []
    if qualification.get('status') != 'passed':
        errors.append('implementation_qualification_not_passed')
    work = qualification.get('work', {})
    for key, ceiling in (('response_forwards', 8), ('response_backwards', 8), ('disposable_optimizer_applications', 16)):
        value = work.get(key)
        if type(value) is not int or not 0 <= value <= ceiling:
            errors.append('qualification_budget_' + key)
    for key in ('serving_generations', 'persistent_optimizer_updates'):
        if type(work.get(key)) is not int or work[key] != 0:
            errors.append('qualification_budget_' + key)
    for key in ('native_neutral_exact', 'preview_isolation', 'full_model_memory_checked'):
        if qualification.get(key) is not True:
            errors.append('qualification_' + key)
    if expected_bindings is not None and qualification.get('bindings') != expected_bindings:
        errors.append('qualification_input_source_bindings')
    return errors


def sealed_schedule_errors(root, plan):
    expected = [dict(event=event, panel=panel['name'], request_id=request['request_id'],
                     ordinal=ordinal, repeat=repeat, order=list(method_order(ordinal, repeat)))
                for event in (10, 12) for panel in plan['panels']
                if event == 12 or panel['name'] != 'final'
                for ordinal, request in enumerate(panel['requests']) for repeat in range(REPEATS)]
    path = root / 'data_preparation/EVALUATION_SCHEDULE.json'
    schedule = _load(path)
    digest = hashlib.sha256(json.dumps(expected, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return ([] if schedule == expected and plan.get('evaluation_schedule_sha256') == digest
            else ['presealed_full_evaluation_schedule'])


def receipt_index(root):
    candidates = list(root.glob('*.json')) + list((root / 'sealed').glob('*.json'))
    candidates += list((root / 'data_preparation').glob('*.json'))
    candidates += list((root / 'campaign_001').rglob('*.json'))
    names = {'PLAN.json', 'SOURCE.json', 'STATUS.json', 'COSTS.json', 'RESULT.json',
             'QUALIFICATION.json', 'EVENT_CHECK.json', 'LIVE.json', 'LIVE_BOUNDARY.json',
             'SAVED_ENDPOINT.json', 'FINAL.json', 'IMPLEMENTATION_QUALIFICATION.json',
             'QUALIFICATION_INPUTS.json', 'EVALUATION_SCHEDULE.json'}
    paths = [path for path in candidates if path.name in names]
    return [dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(paths)]


def write_campaign_report(root, output, *, execution_commit):
    """Reconstruct every fixed panel from raw records, then the sealed primary."""
    root, output = Path(root), Path(output)
    plan = _load(root / 'sealed/PLAN.json')
    source = plan['execution_source']
    source_formats = validate_execution_source(source, execution_commit)
    validate_panels(plan['panels'])
    campaign = root / 'campaign_001'
    status = _load(campaign / 'STATUS.json', {})
    costs = _load(campaign / 'COSTS.json', {})
    output.mkdir(parents=True, exist_ok=True)
    summaries, panel_rows, request_rows, audit_rows, repeat_rows = {}, [], [], [], []
    for event in (10, 12):
        for panel in plan['panels'][:3 if event == 10 else 4]:
            name, mode = panel['name'], panel['canonical_mode']
            directory = campaign / f'event_{event:03d}/evaluation' / name
            progress = _load(directory / 'STATUS.json', {'status': 'unattempted'})
            protocol = _load(directory / 'PROTOCOL.json', {})
            requests = prepare_requests(panel['requests'])
            path = directory / 'MEASUREMENTS.jsonl'
            native_rows = [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []
            rows, canonicals = _local_records(directory, native_rows, _load(directory / 'CANONICAL.json', {}))
            summary, per_request, checked, valid_rows = summarize(rows, requests,
                checkpoints=protocol.get('checkpoints', {}), canonicals=canonicals, canonical_mode=mode)
            bindings = panel_binding_errors(directory, campaign, requests, rows, progress, protocol, event, name, mode)
            if len(checked) != len(rows):
                bindings.append('foreign_request_measurements')
                known = {row['request_id'] for row in requests}
                checked.extend(dict(row, report_valid=False, report_errors=['foreign_request_measurement'])
                               for row in rows if row['request_id'] not in known)
            complete = (not bindings and progress.get('status') == 'completed'
                        and summary['valid_requests'] == 64 and summary['valid_timed_generations'] == GENERATIONS)
            summary.update(event_number=event, panel_name=name, canonical_mode=mode,
                           complete_valid_panel=complete, binding_errors=bindings, execution=progress)
            summaries[f'{event}_{name}'] = summary
            request_rows.extend(dict(event=event, panel=name, **row) for row in per_request)
            audit_rows.extend(dict(event=event, panel=name, **row) for row in checked)
            for comparison_name, comparison in summary['comparisons'].items():
                row = dict(event=event, panel=name, comparison=comparison_name,
                           complete_valid_panel=complete, valid_requests=summary['valid_requests'],
                           binding_errors=';'.join(bindings), **comparison)
                for arm in ARMS:
                    row.update({arm + '_' + key: value for key, value in summary[arm].items()})
                panel_rows.append(row)
            for repeat in range(REPEATS):
                for arm in ARMS:
                    selected = [row for row in valid_rows if row['method'] == arm and row['repeat'] == repeat]
                    repeat_rows.append(dict(event=event, panel=name, repeat=repeat, method=arm, requests=len(selected),
                        verification_calls=sum(row['verification_calls'] for row in selected),
                        **{timer: sum(row[timer] for row in selected) for timer in
                           ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')}))
    final_rows = [row for row in request_rows if row['event'] == 12]
    pooled = aggregate_requests(final_rows)
    pooled['panels'] = {panel['name']: summaries['12_' + panel['name']] for panel in plan['panels']}
    pooled['complete_valid_pool'] = all(value['complete_valid_panel'] for value in pooled['panels'].values())
    primary_summary = summaries['12_final']
    primary_rows = [row for row in final_rows if row['panel'] == 'final']
    bootstrap = (paired_bootstrap([row['K_verification_calls'] for row in primary_rows],
                                  [row['KW_verification_calls'] for row in primary_rows])
                 if primary_summary['complete_valid_panel'] else None)
    cost_summary = campaign_costs(campaign, costs)
    invalid_work = work_errors(status, summaries)
    invalid_work.extend(training_receipt_errors(campaign))
    invalid_work.extend(sealed_schedule_errors(root, plan))
    qualification = _load(root / 'IMPLEMENTATION_QUALIFICATION.json', {})
    qualification_inputs = _load(root / 'sealed/QUALIFICATION_INPUTS.json')
    k8_hash = plan.get('initial', {}).get('K', {}).get('state_identity', {}).get('sha256')
    if qualification_inputs is None or not k8_hash:
        invalid_work.append('qualification_input_seal_missing')
    expected_bindings = dict(source_k8_sha256=k8_hash, execution_commit=execution_commit,
        qualification_inputs_sha256=hashlib.sha256(json.dumps(qualification_inputs, sort_keys=True,
                                                             separators=(',', ':')).encode()).hexdigest())
    invalid_work.extend(qualification_errors(qualification, expected_bindings=expected_bindings))
    cost_summary['implementation_qualification'] = qualification
    invalid_panels = [key for key, value in summaries.items() if not value['complete_valid_panel']]
    primary = primary_summary['comparisons']['KW_vs_K']
    pooled_comparison = pooled['comparisons']['KW_vs_K']
    outcome = outcome_label(valid=not invalid_work and not invalid_panels,
        primary_reduction=primary['call_reduction_pct'], pooled_reduction=pooled_comparison['call_reduction_pct'],
        interval=bootstrap['percentile_95_ci_pct'] if bootstrap else None,
        serving_reductions=[primary['request_time_reduction_pct'], pooled_comparison['request_time_reduction_pct']],
        costs_complete=cost_summary['complete_training_costs'])
    result = dict(execution_git_commit=execution_commit, execution_source=source,
        execution_source_receipt_formats=source_formats, campaign_status=status.get('status'),
        primary_comparison='KW12_vs_K12_on_new_final64', primary=primary, primary_bootstrap=bootstrap,
        outcome=outcome, correctness_errors=dict(work=invalid_work, panels=invalid_panels),
        implementation_qualification=qualification,
        panels=summaries, pooled_final=pooled, actual_work=status,
        reaudit_returned_measurements=len(audit_rows),
        reaudit_valid_timed_generations=sum(value['valid_timed_generations'] for value in summaries.values()),
        complete_valid_panels=sum(value['complete_valid_panel'] for value in summaries.values()),
        planned_evaluation_generations=TOTAL_GENERATIONS, no_seed_replication=True,
        no_integrated_asynchronous_speedup_claim=True,
        decision_rule='Correctness first; nonpositive new-final primary is unsupported; positive primary with nonpositive pooled is mixed. Positive calls with interval crossing zero, discordant request timing, or incomplete costs remain mixed.')
    write_json(output / 'SUMMARY.json', result)
    write_json(output / 'COSTS.json', cost_summary)
    write_json(output / 'PROVENANCE.json', dict(execution_git_commit=execution_commit, receipts=receipt_index(root),
        scope='Compact receipt hashes; no independent bulk-state reload or GPU replication is claimed.'))
    for name, values in (('EVENT_PANEL', panel_rows), ('PER_REQUEST', request_rows),
                         ('ALL_REPEATS', audit_rows), ('REPEAT_TOTALS', repeat_rows)):
        write_csv(output / (name + '.csv'), values)
    write_campaign_markdown(output / 'REPORT.md', result, cost_summary)
    return result


def write_campaign_markdown(path, result, costs):
    def shown(value):
        return 'unavailable' if value is None else f'{value:+.4f}%'

    lines = ['# KW-v1: learned response weights through event 12', '',
        '## Material Passport', '',
        'ARS experiment-agent run/validate. ANALYZED by raw-record reconstruction; no independent GPU replication. '
        'Stage 1 / project None unchanged.', '',
        f"Execution source: `{result['execution_git_commit']}`.", '',
        f"**{result['outcome']}**. Primary KW12 versus K12 new-final64 call reduction: "
        f"**{shown(result['primary']['call_reduction_pct'])}**. Final four-panel pooled reduction: "
        f"**{shown(result['pooled_final']['comparisons']['KW_vs_K']['call_reduction_pct'])}**.", '']
    bootstrap = result['primary_bootstrap']
    if bootstrap:
        low, high = bootstrap['percentile_95_ci_pct']
        lines += [f'Paired request bootstrap percentile 95% interval: [{low:+.4f}%, {high:+.4f}%]. '
                  '10,000 resamples of 64 request pairs, isolated NumPy PCG64 seed 20261030, '
                  'ratio of summed calls. This interval is conditional on this run/checkpoint and panel; '
                  'it does not establish training-seed robustness.', '']
    if result['correctness_errors']['work'] or result['correctness_errors']['panels']:
        lines += ['Correctness/execution blockers: ' + json.dumps(result['correctness_errors']) + '. '
                  'Partial descriptive totals below do not support a scientific performance conclusion.', '']
    lines += ['| Event | Panel | Comparison | Valid requests | Calls (reference → candidate) | Call reduction | '
              'Wins/losses/ties | Gross saved/added | Request-time reduction | Native-time reduction |',
              '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    panels = [(value['event_number'], value['panel_name'], value) for value in result['panels'].values()]
    panels.append((12, 'pooled four panels', result['pooled_final']))
    for event, name, summary in panels:
        for label, comparison in summary['comparisons'].items():
            candidate, reference = comparison['candidate'], comparison['reference']
            signs = comparison['call_request_directions']
            lines.append(f"| {event} | {name} | {label} | {summary['valid_requests']} | "
                f"{summary[reference]['verification_calls']} → {summary[candidate]['verification_calls']} | "
                f"{shown(comparison['call_reduction_pct'])} | {signs['improved']}/{signs['regressed']}/{signs['tied']} | "
                f"{comparison['call_gross_saved']}/{comparison['call_gross_extra']} | "
                f"{shown(comparison['request_time_reduction_pct'])} | {shown(comparison['native_generate_time_reduction_pct'])} |")
    lines += ['', '| Arm | Four-event measured training (s) |', '|---|---:|']
    for arm in ARMS:
        value = costs['arm_totals'][arm]['training_s']
        lines.append(f"| {arm} | {'unavailable' if value is None else f'{value:.6f}'} |")
    lines += ['', f"KW versus K measured training overhead: {shown(costs['KW_vs_K_training_overhead_pct'])}. "
        'COSTS.json retains every event cost and controller/objective-work receipt, including loading, '
        'checkpoint persistence, export and memory. Timers are nested; do not add them twice. '
        'Updater cost is separate from request latency. This shared-feedback campaign does not measure '
        'deployed asynchronous queue/lag behavior or training-inclusive speedup.', '',
        '| KW measured component | Four-event sum (s) |', '|---|---:|']
    for key, value in costs['KW_controller_costs'].items():
        lines.append(f"| {key} | {'unavailable' if value is None else f'{value:.6f}'} |")
    lines += ['', 'Ordinary training observation includes loading/metric/capture components. '
        'Capture host time records enqueue/clone time and is nested in that observation. '
        'Window-observed time overlaps all window components. These rows are not additive independent costs.', '',
        f"Raw audit: {result['reaudit_valid_timed_generations']}/4032 valid generations; "
        f"{result['complete_valid_panels']}/7 complete valid fixed panel endpoints. "
        'Every output/termination, native count trace and repeat count is revalidated. '
        'The event-10 first-eight-development gate uses 72 included generations. Retained F/K1/O8 canonicals '
        'remain fixed; new final canonicals come from first O12, deferring earlier K/KW rows. '
        'This is native-verifier agreement, not extra target-only verification.', '',
        'Counts use one validated repeat-0 trajectory per request; request and native times use each request’s '
        'three-repeat mean. Repeats are not independent samples. EVENT_PANEL.csv and SUMMARY.json include '
        'exact calls, calls per 1,000 output tokens, accepted-draft and committed progress per call, every '
        'panel/endpoint and O comparison. PER_REQUEST.csv and ALL_REPEATS.csv retain rejected/incomplete evidence. '
        'A gain over O without a gain over K does not support the added component.', '',
        'KW adds a bounded five-feature allocation controller to unchanged K. Two isolated actual BF16 Adam '
        'previews produce a finite secant, credited with the next ordinary unweighted training-window gradient. '
        'That next window is shuffled training data, not held-out or necessarily the next chronological user. '
        'This is an approximate first-order loss-transfer signal, not an exact derivative through BF16, '
        'a serving objective, or future Adam history. Generic reweighting and next-batch gradient credit have '
        'prior art; no established novelty or causal-mechanism claim is made.', '',
        'ARS fallacy scan, 11/11: Simpson (all panels visible beside pooled result); ecological (paired requests); '
        'Berkson (fixed selected population limits generalization); collider (no performance filtering); '
        'base-rate neglect (pooled denominators); regression to mean (no extreme-outcome selection); '
        'survivorship (invalid/missing evidence visible); look-elsewhere (all seven endpoints); '
        'forking paths (fixed primary and bootstrap); correlation/causation (no mechanism attribution); '
        'reverse causality (training credit is not evidence of the serving cause).', '']
    if result['outcome'] == 'Promising v1':
        lines += ['The fixed result supports a subsequent confirmation run for discussion; it does not establish '
                  'a completed conference contribution or mechanism. No further run is authorized by this label.', '']
    elif result['outcome'] == 'Unsupported v1 in this run':
        lines += ['Retain K. The primary does not support KW-v1 improvement in this run; this does not prove '
                  'all possible learned-weight methods fail. No automatic tuning or probes follow.', '']
    lines += ['Stop at event 12 and this report; no fourth arm, tuning or event 13.', '']
    path.write_text('\n'.join(lines))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execution-commit', required=True)
    arguments = parser.parse_args()
    write_campaign_report(arguments.root, arguments.output, execution_commit=arguments.execution_commit)
