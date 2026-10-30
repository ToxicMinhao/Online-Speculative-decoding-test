"""Three-arm KW-v1 evaluation with retained or first-O12 canonical outputs."""

import json
import math
import os
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.osd_persistent.run import score_record
from .evaluation_protocol import (
    ARMS, REPEATS, QUALIFICATION_REQUESTS, method_order, validate_inputs,
    publication_errors, weight_check_errors, record_errors, measurement_errors,
    repeat_errors, prepare_requests, canonical_outputs, evaluation_schedule, COMPARISONS,
)
from .report import write_results


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def safe_scalars(value):
    """Keep invalid scalar evidence representable in strict measurement JSON."""
    if isinstance(value, dict):
        return {key: safe_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_scalars(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def save_measurements(root, rows):
    # A K/KW-first row can transition from pending to valid/invalid after O returns.
    # Keep one row per returned generation, preserving chronological order.
    with _temporary_file(root / 'MEASUREMENTS.jsonl') as temporary:
        temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def update_counts(progress, rows, active_attempt):
    progress['completed_valid_generations'] = sum(row['validation_status'] == 'valid' for row in rows)
    progress['invalid_generations'] = sum(row['validation_status'] == 'invalid' for row in rows)
    progress['pending_canonical_generations'] = sum(row['validation_status'] == 'canonical_pending' for row in rows)
    progress['unknown_generation_attempts'] = int(active_attempt is not None)
    progress['unattempted_generations'] = progress['planned_generations'] - progress['attempted_generations']
    progress['interrupted_attempt'] = active_attempt


def validate_request_rows(root, request, rows, records, canonical):
    """Resolve earlier K/KW outputs only after this request's first O is durable."""
    for row in rows:
        errors = list(row['structural_errors'])
        if canonical is not None and not errors:
            record = records[row['label']]
            for name in ('response_ids', 'finish_reason'):
                if record[name] != canonical[name]:
                    errors.append('reference_' + name)
        errors.extend(measurement_errors(row, request, canonical))
        errors.extend(repeat_errors(rows, row['method']))
        errors = sorted(set(errors))
        status = 'invalid' if errors else 'canonical_pending' if canonical is None else 'valid'
        previous = (row.get('validation_status'), row.get('errors'))
        row.update(validation_status=status, valid_measurement=True if status == 'valid' else False if status == 'invalid' else None,
                   errors=errors)
        if previous != (status, errors):
            receipt = dict(label=row['label'], request_id=request['request_id'],
                method=row['method'], repeat=row['repeat'], status=status,
                valid=row['valid_measurement'], errors=errors,
                canonical_source=canonical['source_record_path'] if canonical else None)
            append_json(root / 'VALIDATION_EVENTS.jsonl', receipt)
            write_json(root / (row['label'] + '_VALIDATION.json'), receipt)


def evaluate(backend, requests, checkpoints, root, *, event_number, panel_name,
             canonical_mode='retained', qualification_requests=0):
    requests = prepare_requests(requests)
    validate_inputs(requests, checkpoints, qualification_requests,
                    event_number=event_number, panel_name=panel_name, canonical_mode=canonical_mode)
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows, canonicals, checked_arms = [], canonical_outputs(requests, canonical_mode), set()
    published = False
    active_attempt = None
    started = time.perf_counter()
    progress = dict(status='running', planned_generations=len(requests) * len(ARMS) * REPEATS,
        attempted_generations=0, returned_generations=0,
        event_number=event_number, panel_name=panel_name, canonical_mode=canonical_mode,
        qualification='pending' if qualification_requests else 'not_required_prior_panel_gate',
        qualification_requests=qualification_requests, activation_s=0., weight_validation_s=0.,
        optimizer_steps=0, feedback_collections=0, automatic_retry=False)
    write_json(root / 'PROTOCOL.json', dict(arms=list(ARMS), comparisons=[list(pair) for pair in COMPARISONS], repeats=REPEATS,
        checkpoints=checkpoints, rotation='even base O,K,KW; odd base O,KW,K; shift=(ordinal+repeat)%3',
        schedule=evaluation_schedule(requests), count_trajectory_repeat=0,
        event_number=event_number, panel_name=panel_name, canonical_mode=canonical_mode,
        canonical_source=('retained exact panel tokens/termination' if canonical_mode == 'retained' else
                          'first O12 execution for each request; earlier K/KW validation is deferred'),
        historical_count_gate=False, capture=False, extra_warmup_generations=0,
        freshness_scope='new to the current O/K/KW study; older project exposure is disclosed in the input manifest'))
    write_json(root / 'CANONICAL.json', canonicals)
    write_json(root / 'REQUESTS.json', requests)
    write_json(root / 'STATUS.json', progress)
    try:
        for request in requests:
            ordinal = request['panel_ordinal']
            request_rows, records = [], {}
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
                        request_id=label, session_id='kw_v1_evaluation', request_index=ordinal,
                        purpose='evaluation', save_to=path)
                    progress['returned_generations'] += 1
                    active_attempt = None
                    # Preserve malformed/nonfinite native evidence before rejecting it.
                    # Derived measurements below encode nonfinite values as strings.
                    with _temporary_file(path) as temporary:
                        temporary.write_text(json.dumps(record, allow_nan=True) + '\n')
                    records[label] = record
                    try:
                        errors = record_errors(record, request, checkpoint, canonicals.get(request['request_id']))
                        if (record['record_id'] != label or record['request_index'] != ordinal
                                or record['session_id'] != 'kw_v1_evaluation'):
                            errors.append('native_identity')
                        row = score_record(record, request_id=request['request_id'], repeat=repeat, method=method)
                        row.update(finish_reason=record['finish_reason'],
                                   cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
                    except (KeyError, TypeError, ValueError, IndexError) as error:
                        errors = ['malformed_native_record: ' + str(error)]
                        row = dict(request_id=request['request_id'], repeat=repeat, method=method, checkpoint_id=checkpoint)
                    row = safe_scalars(row)
                    row.update(label=label, panel_ordinal=ordinal, original_index=request['original_index'],
                               record_path=str(path), structural_errors=errors,
                               validation_status='invalid', valid_measurement=False,
                               errors=['validation_not_completed'])
                    rows.append(row)
                    request_rows.append(row)
                    errors.extend(measurement_errors(row, request))
                    if method == 'O' and request['request_id'] not in canonicals and not errors:
                        canonicals[request['request_id']] = dict(response_ids=list(record['response_ids']),
                            finish_reason=record['finish_reason'], source_record_path=str(path),
                            source_method='O', source_repeat=repeat)
                        write_json(root / 'CANONICAL.json', canonicals)
                    validate_request_rows(root, request, request_rows, records, canonicals.get(request['request_id']))
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
            write_results(rows, requests, root / 'report', checkpoints=checkpoints,
                          canonicals=canonicals, execution=progress, event_number=event_number,
                          panel_name=panel_name, canonical_mode=canonical_mode)
        except BaseException as report_error:
            write_json(root / 'REPORT_FAILURE.json', dict(error=repr(report_error)))
            if progress['status'] == 'completed':
                progress.update(status='failed', error_type=type(report_error).__name__, error=str(report_error))
                write_json(root / 'STATUS.json', progress)
                raise
    return progress
