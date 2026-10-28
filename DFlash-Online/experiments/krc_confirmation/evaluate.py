"""Bounded K1/KRC serving confirmation; no trainer, warmup or feedback path."""

import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.osd_persistent.run import score_record
from .protocol import (
    ARMS, REPEATS, QUALIFICATION_REQUESTS, method_order, validate_inputs,
    publication_errors, weight_check_errors, record_errors, measurement_errors,
    repeat_errors,
)
from .report import write_results


from experiments.fresh_confirmation.evaluate import (
    append_json, safe_scalars, save_measurements, update_counts, validate_request_rows,
)


def check_weights(backend, checkpoint, method, phase, root, progress):
    """Validate loaded native tensors outside serving time, without a generation."""
    started = time.perf_counter()
    equality = backend.check_weights(checkpoint)
    elapsed = time.perf_counter() - started
    errors = weight_check_errors(equality, checkpoint)
    receipt = dict(method=method, phase=phase, native_reply=equality,
                   errors=errors, validation_s=elapsed)
    write_json(root / f'WEIGHTS_{phase.upper()}_{method}.json', receipt)
    progress['weight_validation_s'] += elapsed
    progress['weight_checks'][phase][method] = not errors
    if errors:
        raise RuntimeError('Native tensor equality failed: ' + str(errors))


def invalidate_checkpoint_rows(root, rows, method):
    """A failed final weight check removes that checkpoint from gain claims."""
    for row in rows:
        if row['method'] == method:
            row['errors'] = sorted(set(row['errors'] + ['checkpoint_weight_integrity']))
            row['validation_status'] = 'invalid'
            row['valid_measurement'] = False
            receipt = dict(label=row['label'], request_id=row['request_id'],
                           method=method, repeat=row['repeat'], status='invalid',
                           valid=False, errors=row['errors'])
            append_json(root / 'VALIDATION_EVENTS.jsonl', receipt)
            write_json(root / (row['label'] + '_VALIDATION.json'), receipt)


def evaluate(backend, requests, checkpoints, output, qualification_requests=QUALIFICATION_REQUESTS):
    validate_inputs(requests, checkpoints, qualification_requests)
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rows, canonicals, checked_arms = [], {}, set()
    published = False
    active_attempt = None
    started = time.perf_counter()
    progress = dict(status='running', planned_generations=len(requests) * 2 * REPEATS,
        attempted_generations=0, returned_generations=0, qualification='pending',
        qualification_requests=qualification_requests, activation_s=0., weight_validation_s=0.,
        optimizer_steps=0, feedback_collections=0, automatic_retry=False,
        weight_checks={'before': {}, 'after': {}}, activation_count=0)
    write_json(root / 'PROTOCOL.json', dict(arms=list(ARMS), comparisons=[['KRC', 'K1']], repeats=REPEATS,
        checkpoints=checkpoints, rotation='(panel_ordinal + repeat) % 2', count_trajectory_repeat=0,
        canonical_source='first K1 execution for each request; earlier KRC validation is deferred',
        historical_count_gate=False, capture=False, extra_warmup_generations=0,
        native_weight_checks='before first and after final generation for each arm',
        freshness_scope='new to the current K1/KRC study; older project exposure is disclosed in the input manifest'))
    write_json(root / 'CANONICAL.json', canonicals)
    write_json(root / 'REQUESTS.json', requests)
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
                        progress['activation_count'] += 1
                        if errors:
                            raise RuntimeError('Native publication failed: ' + str(errors))
                        published = True
                    if method not in checked_arms:
                        check_weights(backend, checkpoint, method, 'before', root, progress)
                        checked_arms.add(method)
                    label = f'{ordinal:04d}_r{repeat}_{method}'
                    active_attempt = dict(label=label, request_id=request['request_id'], method=method, repeat=repeat)
                    progress['attempted_generations'] += 1
                    update_counts(progress, rows, active_attempt)
                    write_json(root / 'STATUS.json', progress)
                    path = root / (label + '.json')
                    record = backend.generate(request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
                        request_id=label, session_id='krc_confirmation_evaluation', request_index=ordinal,
                        purpose='evaluation', save_to=path)
                    progress['returned_generations'] += 1
                    active_attempt = None
                    # Preserve malformed/nonfinite native evidence before rejecting it.
                    # Derived measurements below encode nonfinite values as strings.
                    with _temporary_file(path) as temporary:
                        temporary.write_text(json.dumps(record, allow_nan=True) + '\n')
                    records[label] = record
                    try:
                        errors = record_errors(record, request, checkpoint)
                        if (record['record_id'] != label or record['request_index'] != ordinal
                                or record['session_id'] != 'krc_confirmation_evaluation'):
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
                    if method == 'K1' and request['request_id'] not in canonicals and not errors:
                        canonicals[request['request_id']] = dict(response_ids=list(record['response_ids']),
                            finish_reason=record['finish_reason'], source_record_path=str(path),
                            source_method='K1', source_repeat=repeat)
                        write_json(root / 'CANONICAL.json', canonicals)
                    validate_request_rows(root, request, request_rows, records, canonicals.get(request['request_id']))
                    save_measurements(root, rows)
                    update_counts(progress, rows, active_attempt)
                    write_json(root / 'STATUS.json', progress)
                    invalid = [row for row in request_rows if row['validation_status'] == 'invalid']
                    if invalid:
                        raise RuntimeError('Evaluation correctness failed: ' + str(invalid[-1]['errors']))
                    if ordinal == len(requests) - 1 and repeat == REPEATS - 1:
                        try:
                            check_weights(backend, checkpoint, method, 'after', root, progress)
                        except BaseException:
                            invalidate_checkpoint_rows(root, rows, method)
                            raise
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
                          canonicals=canonicals, execution=progress)
        except BaseException as report_error:
            write_json(root / 'REPORT_FAILURE.json', dict(error=repr(report_error)))
            if progress['status'] == 'completed':
                progress.update(status='failed', error_type=type(report_error).__name__, error=str(report_error))
                write_json(root / 'STATUS.json', progress)
                raise
    return progress
