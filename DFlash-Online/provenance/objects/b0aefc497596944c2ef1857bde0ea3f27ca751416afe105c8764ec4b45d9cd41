"""Bounded K/KC correction evaluation against retained F; no training path."""

import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.osd_persistent.run import score_record
from .protocol import (
    ARMS, REPEATS, QUALIFICATION_REQUESTS, method_order, validate_inputs,
    publication_errors, weight_check_errors, record_errors, measurement_errors,
    repeat_errors, prepare_requests, canonical_outputs,
)
from .report import write_results
from experiments.fresh_confirmation.evaluate import (
    append_json, safe_scalars, save_measurements, update_counts,
)


def validate_request_rows(root, request, rows):
    """Retained F labels make every returned row immediately checkable."""
    for row in rows:
        errors = list(row['structural_errors'])
        errors.extend(measurement_errors(row, request))
        errors.extend(repeat_errors(rows, row['method']))
        errors = sorted(set(errors))
        status = 'invalid' if errors else 'valid'
        previous = (row.get('validation_status'), row.get('errors'))
        row.update(validation_status=status, valid_measurement=not errors, errors=errors)
        if previous != (status, errors):
            receipt = dict(label=row['label'], request_id=request['request_id'],
                method=row['method'], repeat=row['repeat'], status=status,
                valid=row['valid_measurement'], errors=errors,
                canonical_source='retained_manifest_eval_requests', canonical_method='F')
            append_json(root / 'VALIDATION_EVENTS.jsonl', receipt)
            write_json(root / (row['label'] + '_VALIDATION.json'), receipt)


def evaluate(backend, requests, checkpoints, root, *, frozen_counts=None,
             qualification_requests=QUALIFICATION_REQUESTS):
    requests = prepare_requests(requests)
    validate_inputs(requests, checkpoints, frozen_counts, qualification_requests)
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows, checked_arms = [], set()
    canonicals = canonical_outputs(requests)
    published = False
    active_attempt = None
    started = time.perf_counter()
    progress = dict(status='running', planned_generations=len(requests) * 2 * REPEATS,
        attempted_generations=0, returned_generations=0, qualification='pending',
        qualification_requests=qualification_requests, activation_s=0., weight_validation_s=0.,
        optimizer_steps=0, feedback_collections=0, automatic_retry=False)
    write_json(root / 'PROTOCOL.json', dict(arms=list(ARMS), comparisons=[['KC', 'K']], repeats=REPEATS,
        checkpoints=checkpoints, rotation='(panel_ordinal + repeat) % 2', count_trajectory_repeat=0,
        canonical_source='retained F outputs and termination in original manifest',
        historical_count_gate=False, capture=False, extra_warmup_generations=0,
        frozen_counts_scope='historical F call context only; no fresh F timing'))
    write_json(root / 'FROZEN_COUNTS.json', frozen_counts)
    write_json(root / 'REQUESTS.json', requests)
    write_json(root / 'CANONICAL.json', canonicals)
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
                        request_id=label, session_id='prefix_correction_evaluation', request_index=ordinal,
                        purpose='evaluation', save_to=path)
                    progress['returned_generations'] += 1
                    active_attempt = None
                    # Preserve malformed/nonfinite native evidence before rejecting it.
                    # Derived measurements below encode nonfinite values as strings.
                    with _temporary_file(path) as temporary:
                        temporary.write_text(json.dumps(record, allow_nan=True) + '\n')
                    try:
                        errors = record_errors(record, request, checkpoint)
                        if (record['record_id'] != label or record['request_index'] != ordinal
                                or record['session_id'] != 'prefix_correction_evaluation'):
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
            write_results(rows, requests, root / 'report', checkpoints=checkpoints,
                          frozen_counts=frozen_counts, execution=progress)
        except BaseException as report_error:
            write_json(root / 'REPORT_FAILURE.json', dict(error=repr(report_error)))
            if progress['status'] == 'completed':
                progress.update(status='failed', error_type=type(report_error).__name__, error=str(report_error))
                write_json(root / 'STATUS.json', progress)
                raise
    return progress
