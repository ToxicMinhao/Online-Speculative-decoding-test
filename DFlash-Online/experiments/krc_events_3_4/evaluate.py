"""Fresh three-arm evaluation, reusing qualified native checks and scoring."""

import json
from pathlib import Path
from statistics import median
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.osd_persistent.run import score_record
from experiments.osd_one_event.report import direction, reduction
from experiments.fresh_confirmation.evaluate import (
    append_json, safe_scalars, save_measurements, update_counts,
)
from experiments.fresh_confirmation.report import write_csv
from .evaluation_protocol import (
    ARMS, COMPARISONS, REPEATS, QUALIFICATION_REQUESTS, SESSION_ID, COUNT_FIELDS,
    HISTORICAL_ARMS, method_order, rotation_balance, validate_inputs,
    publication_errors, weight_check_errors, record_errors, measurement_errors,
    repeat_errors, prepare_requests, canonical_outputs, validate_count_maps,
)

TIMERS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')


def validate_request_rows(root, request, rows):
    for row in rows:
        errors = sorted(set(row['structural_errors'] + measurement_errors(row, request)
                            + repeat_errors(rows, row['method'])))
        status = 'invalid' if errors else 'valid'
        previous = (row.get('validation_status'), row.get('errors'))
        row.update(validation_status=status, valid_measurement=not errors, errors=errors)
        if previous != (status, errors):
            receipt = dict(label=row['label'], request_id=request['request_id'],
                method=row['method'], repeat=row['repeat'], status=status,
                valid=not errors, errors=errors, canonical_source='retained_panel_canonical',
                canonical_method=request['canonical_source_method'])
            append_json(root / 'VALIDATION_EVENTS.jsonl', receipt)
            write_json(root / (row['label'] + '_VALIDATION.json'), receipt)


def evaluate(backend, requests, checkpoints, root, *, expected_control_counts=None,
             qualification_requests=QUALIFICATION_REQUESTS, historical_counts=None):
    requests = prepare_requests(requests, expected_control_counts)
    validate_inputs(requests, checkpoints, expected_control_counts, qualification_requests)
    if historical_counts is not None:
        validate_count_maps(historical_counts, {r['request_id'] for r in requests},
                            allowed_arms=HISTORICAL_ARMS)
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows, checked_arms = [], set()
    published = False
    active_attempt = None
    started = time.perf_counter()
    progress = dict(status='running', planned_generations=len(requests) * len(ARMS) * REPEATS,
        attempted_generations=0, returned_generations=0,
        qualification='pending' if qualification_requests else 'not_required_prior_panel_gate',
        qualification_requests=qualification_requests, activation_s=0., weight_validation_s=0.,
        optimizer_steps=0, feedback_collections=0, automatic_retry=False)
    write_json(root / 'PROTOCOL.json', dict(arms=list(ARMS), comparisons=COMPARISONS,
        primary_comparison='KRC4_vs_O4', repeats=REPEATS, checkpoints=checkpoints,
        rotation='(panel_ordinal + repeat) % 3', rotation_balance=rotation_balance(len(requests)),
        count_trajectory_repeat=0, canonical_source='retained panel output tokens and termination',
        historical_count_gates=[] if expected_control_counts is None else ['O4', 'K4'],
        capture=False, extra_warmup_generations=0,
        timer_scope='native_generate_s and cache_maintenance_s are nested in complete request latency'))
    write_json(root / 'EXPECTED_CONTROL_COUNTS.json', expected_control_counts)
    write_json(root / 'HISTORICAL_COUNTS.json', historical_counts)
    write_json(root / 'REQUESTS.json', requests)
    write_json(root / 'CANONICAL.json', canonical_outputs(requests))
    try:
        for request in requests:
            ordinal = request['panel_ordinal']
            request_rows = []
            for repeat in range(REPEATS):
                for method in method_order(ordinal, repeat):
                    checkpoint = checkpoints[method]
                    if not published or backend.checkpoint_id != checkpoint:
                        receipt = backend.publish(checkpoint)
                        errors = publication_errors(receipt, checkpoint)
                        append_json(root / 'ACTIVATIONS.jsonl', dict(panel_ordinal=ordinal, repeat=repeat,
                            method=method, **receipt, errors=errors))
                        progress['activation_s'] += receipt['activation_s']
                        if errors:
                            raise RuntimeError('Native publication failed: ' + str(errors))
                        published = True
                    if method not in checked_arms:
                        validation_started = time.perf_counter()
                        equality = backend.check_weights(checkpoint)
                        elapsed = time.perf_counter() - validation_started
                        errors = weight_check_errors(equality, checkpoint)
                        write_json(root / ('WEIGHTS_' + method + '.json'), dict(
                            native_reply=equality, errors=errors, validation_s=elapsed))
                        progress['weight_validation_s'] += elapsed
                        if errors:
                            raise RuntimeError('Native tensor equality failed: ' + str(errors))
                        checked_arms.add(method)
                    label = f'{ordinal:04d}_r{repeat}_{method}'
                    active_attempt = dict(label=label, request_id=request['request_id'], method=method, repeat=repeat)
                    progress['attempted_generations'] += 1
                    update_counts(progress, rows, active_attempt)
                    write_json(root / 'STATUS.json', progress)
                    path = root / (label + '.json')
                    record = backend.generate(request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
                        request_id=label, session_id=SESSION_ID, request_index=ordinal,
                        purpose='evaluation', save_to=path)
                    progress['returned_generations'] += 1
                    active_attempt = None
                    # Preserve malformed/nonfinite native evidence before rejection.
                    with _temporary_file(path) as temporary:
                        temporary.write_text(json.dumps(record, allow_nan=True) + '\n')
                    try:
                        errors = record_errors(record, request, checkpoint)
                        if (record['record_id'] != label or record['request_index'] != ordinal
                                or record['session_id'] != SESSION_ID):
                            errors.append('native_identity')
                        row = score_record(record, request_id=request['request_id'], repeat=repeat, method=method)
                        row.update(finish_reason=record['finish_reason'],
                                   cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
                    except (KeyError, TypeError, ValueError, IndexError) as error:
                        errors = ['malformed_native_record: ' + str(error)]
                        row = dict(request_id=request['request_id'], repeat=repeat, method=method,
                                   checkpoint_id=checkpoint)
                    row = safe_scalars(row)
                    row.update(label=label, panel_ordinal=ordinal, original_index=request['original_index'],
                               record_path=str(path), structural_errors=errors,
                               validation_status='invalid', valid_measurement=False,
                               errors=['validation_not_completed'])
                    rows.append(row)
                    request_rows.append(row)
                    validate_request_rows(root, request, request_rows)
                    save_measurements(root, rows)
                    update_counts(progress, rows, active_attempt)
                    write_json(root / 'STATUS.json', progress)
                    invalid = [row for row in request_rows if row['validation_status'] == 'invalid']
                    if invalid:
                        raise RuntimeError('Evaluation correctness failed: ' + str(invalid[-1]['errors']))
            if ordinal + 1 == qualification_requests:
                progress['qualification'] = 'passed'
                write_json(root / 'QUALIFICATION.json', dict(status='passed', requests=qualification_requests,
                    generations=progress['returned_generations'], arms_with_native_equality=sorted(checked_arms),
                    performance_win_required=False, extra_generations=0))
            write_json(root / 'STATUS.json', progress)
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error))
        if progress['qualification'] == 'pending':
            progress['qualification'] = 'incomplete'
        raise
    finally:
        update_counts(progress, rows, active_attempt)
        progress['evaluation_wall_s'] = time.perf_counter() - started
        save_measurements(root, rows)
        write_json(root / 'STATUS.json', progress)
        try:
            summary = write_results(rows, requests, root / 'report', checkpoints=checkpoints,
                expected_control_counts=expected_control_counts, execution=progress,
                historical_counts=historical_counts)
            if progress['status'] == 'completed' and summary['valid_requests'] != len(requests):
                raise RuntimeError('Independent report revalidation rejected a completed request')
        except BaseException as report_error:
            write_json(root / 'REPORT_FAILURE.json', dict(error=repr(report_error)))
            if progress['status'] == 'completed':
                progress.update(status='failed', error_type=type(report_error).__name__, error=str(report_error))
                write_json(root / 'STATUS.json', progress)
                raise
    return progress


def validated_row(row, request, checkpoints):
    """Reopen each native record rather than trusting derived success flags."""
    errors = list(row.get('errors', [])) + measurement_errors(row, request)
    if row.get('method') not in ARMS or type(row.get('repeat')) is not int or row['repeat'] not in range(REPEATS):
        errors.append('invalid_evaluation_slot')
    if row.get('valid_measurement') is not True:
        errors.append('execution_rejected_or_pending_measurement')
    try:
        record = json.loads(Path(row['record_path']).read_text())
        errors.extend(record_errors(record, request, checkpoints[row['method']]))
        scored = score_record(record, request_id=request['request_id'],
                              repeat=row['repeat'], method=row['method'])
        scored.update(finish_reason=record['finish_reason'],
                      cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
        for name in COUNT_FIELDS + TIMERS + ('checkpoint_id', 'finish_reason'):
            if row.get(name) != scored[name]:
                errors.append('row_native_disagreement_' + name)
        if (record['record_id'] != row['label'] or record['request_index'] != request['panel_ordinal']
                or record['session_id'] != SESSION_ID):
            errors.append('native_identity')
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        errors.append('native_record_unavailable_or_malformed: ' + str(error))
    return {**row, 'report_errors': sorted(set(errors)), 'report_valid': not errors}


def count_comparison(candidate, reference):
    differences = [old - new for old, new in zip(reference, candidate)]
    return dict(candidate_calls=sum(candidate), reference_calls=sum(reference),
        call_reduction_pct=reduction(sum(reference), sum(candidate)),
        request_directions={label: sum(direction(old, new) == label for old, new in zip(reference, candidate))
                            for label in ('improved', 'regressed', 'tied')},
        gross_saved=sum(max(value, 0) for value in differences),
        gross_extra=sum(max(-value, 0) for value in differences), net_saved=sum(differences))


def summarize(rows, requests, *, checkpoints, expected_control_counts=None, historical_counts=None):
    requests = prepare_requests(requests, expected_control_counts)
    per_request, checked_rows, valid_rows = [], [], []
    slots = {(arm, repeat) for arm in ARMS for repeat in range(REPEATS)}
    for request in requests:
        selected = [validated_row(row, request, checkpoints) for row in rows
                    if row['request_id'] == request['request_id']]
        checked_rows.extend(selected)
        complete = len(selected) == len(slots) and {(r['method'], r['repeat']) for r in selected} == slots
        errors = {error for row in selected for error in row['report_errors']}
        for arm in ARMS:
            errors.update(repeat_errors(selected, arm))
        if not complete:
            errors.add('incomplete_or_duplicate_slots')
        result = dict(request_id=request['request_id'], panel_ordinal=request['panel_ordinal'],
                      complete=complete, valid=complete and not errors, errors='; '.join(sorted(errors)))
        if result['valid']:
            valid_rows.extend(selected)
            result.update(canonical_output_tokens=len(request['response_ids']),
                          canonical_finish_reason=request['finish_reason'])
            for arm in ARMS:
                repeats = [row for row in selected if row['method'] == arm]
                primary = next(row for row in repeats if row['repeat'] == 0)
                result.update({f'{arm}_{key}': primary[key] for key in COUNT_FIELDS})
                result.update({f'{arm}_mean_{timer}': sum(row[timer] for row in repeats) / REPEATS
                               for timer in TIMERS})
            for arm, counts in (historical_counts or {}).items():
                result[arm + '_historical_verification_calls'] = counts[request['request_id']]
            for candidate, reference in COMPARISONS:
                label = candidate + '_vs_' + reference
                for metric, key in (('call', 'verification_calls'), ('latency', 'mean_request_latency_s')):
                    result[label + '_' + metric + '_direction'] = direction(result[reference + '_' + key],
                                                                            result[candidate + '_' + key])
        per_request.append(result)
    valid_requests = [row for row in per_request if row['valid']]
    summary = dict(planned_requests=len(requests), valid_requests=len(valid_requests),
        invalid_or_incomplete_requests=len(requests) - len(valid_requests),
        returned_measurements=len(rows), valid_timed_generations=len(valid_rows),
        primary_comparison='KRC4_vs_O4', timing_repeats_per_method_request=REPEATS,
        cohort='One common cohort of complete valid O4/K4/KRC4 requests; panels remain separate.',
        count_scope='One validated repeat-0 trajectory per method/request; repeats are not independent examples.',
        latency_scope='Complete capture-disabled request; loading, activation, export and training excluded.',
        comparisons={})
    for arm in ARMS:
        totals = {key: sum(row[arm + '_' + key] for row in valid_requests) for key in COUNT_FIELDS}
        for timer in TIMERS:
            totals['summed_mean_' + timer] = sum(row[arm + '_mean_' + timer] for row in valid_requests)
            totals['all_repeats_' + timer] = sum(row[timer] for row in valid_rows if row['method'] == arm)
        calls, tokens = totals['verification_calls'], totals['output_tokens']
        totals['calls_per_1000_output_tokens'] = 1000 * calls / tokens if tokens else None
        totals['accepted_draft_tokens_per_call'] = totals['accepted_draft_tokens'] / calls if calls else None
        totals['committed_progress_per_call'] = totals['committed_progress'] / calls if calls else None
        summary[arm] = totals
    for candidate, reference in COMPARISONS:
        label = candidate + '_vs_' + reference
        value = dict(candidate=candidate, reference=reference,
            call_reduction_pct=reduction(summary[reference]['verification_calls'], summary[candidate]['verification_calls']),
            request_time_reduction_pct=reduction(summary[reference]['summed_mean_request_latency_s'],
                                                summary[candidate]['summed_mean_request_latency_s']))
        for metric, key in (('call', 'verification_calls'), ('latency', 'mean_request_latency_s')):
            delta = [row[reference + '_' + key] - row[candidate + '_' + key] for row in valid_requests]
            value[metric + '_request_directions'] = {
                sign: sum(row[label + '_' + metric + '_direction'] == sign for row in valid_requests)
                for sign in ('improved', 'regressed', 'tied')}
            value[metric + '_gross_saved'] = sum(max(d, 0) for d in delta)
            value[metric + '_gross_extra'] = sum(max(-d, 0) for d in delta)
            value[metric + '_net_saved'] = sum(delta)
        summary['comparisons'][label] = value
    summary['historical_counts'] = {
        arm: dict(verification_calls=sum(row[arm + '_historical_verification_calls'] for row in valid_requests),
                  requests=len(valid_requests), measurement='historical counts; no latency comparison')
        for arm in (historical_counts or {})}
    curves = []
    for event in (1, 2, 4):
        curves.append(dict(event=event, measurement='fresh' if event == 4 else 'historical',
            **{arm: (summary[arm + '4']['verification_calls'] if event == 4 else
                     summary['historical_counts'].get(arm + str(event), {}).get('verification_calls'))
               for arm in ('O', 'K', 'KRC')}))
    summary['event_count_curve'] = curves
    vectors = {arm: [row[arm + '_verification_calls'] for row in valid_requests] for arm in ARMS}
    vectors.update({arm: [row[arm + '_historical_verification_calls'] for row in valid_requests]
                    for arm in (historical_counts or {})})
    pairs = [(family + str(new), family + str(old)) for family in ('O', 'K', 'KRC')
             for old, new in ((1, 2), (2, 4), (1, 4))]
    pairs += [(candidate + str(event), reference + str(event)) for event in (1, 2)
              for candidate, reference in (('KRC', 'O'), ('K', 'O'), ('KRC', 'K'))]
    summary['count_learning_curve'] = {
        candidate + '_vs_' + reference: count_comparison(vectors[candidate], vectors[reference])
        for candidate, reference in pairs if candidate in vectors and reference in vectors}
    summary['relative_advantage_change_percentage_points'] = {}
    for candidate, reference in (('KRC', 'O'), ('KRC', 'K'), ('K', 'O')):
        new = summary['comparisons'][candidate + '4_vs_' + reference + '4']['call_reduction_pct']
        for event in (1, 2):
            old = summary['count_learning_curve'].get(candidate + str(event) + '_vs_' + reference + str(event), {}).get('call_reduction_pct')
            summary['relative_advantage_change_percentage_points'][f'{candidate}_vs_{reference}_event{event}_to4'] = (
                None if old is None or new is None else new - old)
    lengths = [row['canonical_output_tokens'] for row in valid_requests]
    summary['canonical_outputs'] = dict(requests=len(lengths), output_tokens=sum(lengths),
        minimum_length=min(lengths) if lengths else None, maximum_length=max(lengths) if lengths else None,
        mean_length=sum(lengths) / len(lengths) if lengths else None, median_length=median(lengths) if lengths else None,
        eos_requests=sum(row['canonical_finish_reason'] == 'eos' for row in valid_requests),
        length_terminated_requests=sum(row['canonical_finish_reason'] == 'length' for row in valid_requests),
        output_cap_hits=sum(length == 2048 for length in lengths))
    return summary, per_request, checked_rows, valid_rows


def write_results(rows, requests, root, *, checkpoints, execution,
                  expected_control_counts=None, historical_counts=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary, per_request, checked_rows, valid_rows = summarize(rows, requests, checkpoints=checkpoints,
        expected_control_counts=expected_control_counts, historical_counts=historical_counts)
    summary['execution'] = execution
    summary['timer_nesting'] = 'Native generation and cache maintenance are inside complete request latency; do not add them again.'
    write_json(root / 'SUMMARY.json', summary)
    write_csv(root / 'ALL_REPEATS.csv', checked_rows)
    write_csv(root / 'PER_REQUEST.csv', per_request)
    repeat_totals = []
    for repeat in range(REPEATS):
        for arm in ARMS:
            selected = [row for row in valid_rows if row['repeat'] == repeat and row['method'] == arm]
            repeat_totals.append(dict(repeat=repeat, method=arm, requests=len(selected),
                verification_calls=sum(row['verification_calls'] for row in selected),
                **{timer: sum(row[timer] for row in selected) for timer in TIMERS}))
    write_csv(root / 'REPEAT_TOTALS.csv', repeat_totals)
    write_csv(root / 'EVENT_COUNT_CURVE.csv', summary['event_count_curve'])
    text = [f"# O4/K4/KRC4 evaluation: {execution['status']}", '',
        f"Valid common requests: {summary['valid_requests']}/{len(requests)}. O is the main baseline; K is the correction ablation.", '',
        '| Arm | Calls | Sum of mean complete request times (s) | Native generation (s) | Cache maintenance (s) |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        value = summary[arm]
        text.append(f"| {arm} | {value['verification_calls']} | " + ' | '.join(
            f"{value['summed_mean_' + timer]:.6f}" for timer in TIMERS) + ' |')
    text.extend(['', '| Comparison | Call reduction | Request-time reduction | Calls improved/regressed/tied | Gross calls saved/added |',
                 '|---|---:|---:|---:|---:|'])
    for label, value in summary['comparisons'].items():
        shown = ['unavailable' if value[key] is None else f'{value[key]:+.4f}%'
                 for key in ('call_reduction_pct', 'request_time_reduction_pct')]
        signs = value['call_request_directions']
        text.append(f"| {label} | {shown[0]} | {shown[1]} | {signs['improved']}/{signs['regressed']}/{signs['tied']} | "
                    f"{value['call_gross_saved']}/{value['call_gross_extra']} |")
    text.extend(['', '| Event | Measurement | O calls | K calls | KRC calls |', '|---|---|---:|---:|---:|'])
    for row in summary['event_count_curve']:
        text.append(f"| {row['event']} | {row['measurement']} | " + ' | '.join(
            'absent' if row[arm] is None else str(row[arm]) for arm in ('O', 'K', 'KRC')) + ' |')
    text.extend(['', 'Counts use one validated repeat-0 trajectory per request. Timings sum each request’s three-repeat mean. '
        'All ratios use pooled totals. Historical timings are not a learning curve. Panels remain separate. '
        'Null and harmful valid results remain included; invalid and incomplete requests remain visible in PER_REQUEST.csv.', '',
        summary['timer_nesting'], '',
        f"Activation {execution['activation_s']:.6f}s and weight validation {execution['weight_validation_s']:.6f}s are outside request latency. "
        'Initial loading, training and export costs are reported separately by the campaign. '
        'This is available-checkpoint serving evidence, not training-inclusive deployment acceleration. '
        'KRC4 versus K4 tests their complete correction history, not corrections in events 3 and 4 alone.', '',
        'The campaign report supplies the execution source commit and exact source bindings. '
        'Per-request, per-repeat, event-count and raw validation evidence accompany this panel report.'])
    (root / 'REPORT.md').write_text('\n'.join(text) + '\n')
    return summary
