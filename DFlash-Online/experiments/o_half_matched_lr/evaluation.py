"""Count-focused bridge and two-repeat evaluation with immutable PR21 outputs."""

import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.osd_persistent.run import score_record
from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, measurement_errors, publication_errors, repeat_errors, weight_check_errors,
)
from experiments.kw_v1.evaluate import append_json, safe_scalars, save_measurements, update_counts
from .archive import ARCHIVED_METHODS, ArchiveReader, validate_record


SEEDS = (17, 29)
NEW_METHODS = ('O', 'O_HALF')
SESSION = 'o_half_matched_lr_evaluation'


def _schedule(requests, seed, phase):
    if seed not in SEEDS or phase not in ('bridge', 'final'):
        raise ValueError('Only the sealed seed-17/29 bridge or final cells are allowed')
    methods = ARCHIVED_METHODS if phase == 'bridge' else NEW_METHODS
    selected = requests[:8] if phase == 'bridge' else requests
    repeats = 1 if phase == 'bridge' else 2
    return [dict(seed=seed, event=4, phase=phase, method=method,
                 request_id=request['request_id'], panel_ordinal=request['panel_ordinal'],
                 repeat=repeat, position=position)
            for position, method in enumerate(methods)
            for request in selected for repeat in range(repeats)]


def bridge_schedule(requests, seed):
    return _schedule(requests, seed, 'bridge')


def final_schedule(requests, seed):
    return _schedule(requests, seed, 'final')


def _validate_inputs(requests, checkpoints, seed, phase, archive):
    methods = ARCHIVED_METHODS if phase == 'bridge' else NEW_METHODS
    if (archive['status'] != 'passed' or requests != archive['requests']
            or len(requests) != 128 or [r['panel_ordinal'] for r in requests] != list(range(128))):
        raise ValueError('Reuse the entire exact audited 128-request PR21 panel')
    if (set(checkpoints) != set(methods)
            or len(set(map(str, checkpoints.values()))) != len(methods)):
        raise ValueError('Use only the distinct checkpoints assigned to this cell')
    if phase == 'bridge' and {k: str(v) for k, v in checkpoints.items()} != archive['checkpoints'][str(seed)]:
        raise ValueError('Compatibility bridge checkpoints differ from the archive')
    if set(archive['canonicals']) != {r['request_id'] for r in requests}:
        raise ValueError('Immutable canonical panel is incomplete')
    return _schedule(requests, seed, phase)


def _measurement(record, request, checkpoint, canonical, slot):
    """Validate raw evidence before it may enter the count comparison."""
    label = f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}"
    try:
        errors = validate_record(record, request, checkpoint, canonical, label=label, session=SESSION)
        row = score_record(record, request_id=request['request_id'],
                           repeat=slot['repeat'], method=slot['method'])
        row.update(finish_reason=record['finish_reason'],
                   cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
        errors.extend(measurement_errors(row, request, canonical))
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
        errors = ['malformed_native_record: ' + repr(error)]
        row = {'request_id': request['request_id'], 'repeat': slot['repeat'],
               'method': slot['method'], 'checkpoint_id': checkpoint}
    row = safe_scalars(row)
    row.update(**slot, label=label, original_index=request['original_index'])
    return row, errors


def _comparison_errors(row, rows, archive, seed, phase):
    if phase == 'bridge':
        expected = archive['counts'][str(seed)][row['method']][row['request_id']]
        return ['archived_count_' + name for name in COUNT_FIELDS if row.get(name) != expected[name]]
    return repeat_errors([value for value in rows if value['request_id'] == row['request_id']], row['method'])


def _gate(method, phase):
    return dict(status='passed', method=method, phase=phase, requests=8,
                generations=8 if phase == 'bridge' else 16,
                extra_generations=0, performance_win_required=False)


def _evaluate(backend, requests, checkpoints, root, *, seed, phase, archive):
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    planned = _validate_inputs(requests, checkpoints, seed, phase, archive)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows, qualified, active_attempt = [], [], None
    current_method = None
    started = time.perf_counter()
    methods = list(ARCHIVED_METHODS if phase == 'bridge' else NEW_METHODS)
    progress = dict(status='running', seed=seed, event=4, phase=phase,
                    planned_generations=len(planned), attempted_generations=0, returned_generations=0,
                    qualification='pending', qualified_methods=[], activation_s=0., weight_validation_s=0.,
                    extra_generations=0, automatic_retry=False)
    write_json(root / 'PROTOCOL.json', dict(seed=seed, event=4, phase=phase, methods=methods,
        repeats=1 if phase == 'bridge' else 2, checkpoints=checkpoints, schedule=planned,
        canonical_source='immutable_PR21_first_F', canonical_path=archive['canonical_path'],
        canonical_identity=archive['canonical_identity'], count_trajectory_repeat=0,
        capture=False, extra_warmup_generations=0,
        timing_scope='Operational accounting only; no serving-time comparison'))
    write_json(root / 'REQUESTS.json', requests)
    write_json(root / 'CANONICAL_INPUT.json', archive['canonicals'])
    try:
        for slot in planned:
            method, ordinal = slot['method'], slot['panel_ordinal']
            request = requests[ordinal]
            checkpoint = checkpoints[method]
            if method != current_method:
                activation = backend.publish(checkpoint)
                errors = publication_errors(activation, checkpoint)
                append_json(root / 'ACTIVATIONS.jsonl', {'method': method, **activation, 'errors': errors})
                progress['activation_s'] += activation['activation_s']
                if errors:
                    raise RuntimeError('Checkpoint publication failed: ' + str(errors))
                checking = time.perf_counter()
                weights = backend.check_weights(checkpoint)
                errors = weight_check_errors(weights, checkpoint)
                elapsed = time.perf_counter() - checking
                write_json(root / f'WEIGHTS_{method}.json',
                           dict(native_reply=weights, errors=errors, validation_s=elapsed))
                progress['weight_validation_s'] += elapsed
                if errors:
                    raise RuntimeError('Loaded/exported weights differ: ' + str(errors))
                current_method = method
            label = f"{ordinal:04d}_r{slot['repeat']}_{method}"
            path = root / (label + '.json')
            active_attempt = {**slot, 'label': label}
            progress['attempted_generations'] += 1
            update_counts(progress, rows, active_attempt)
            write_json(root / 'STATUS.json', progress)
            record = backend.generate(request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
                request_id=label, session_id=SESSION, request_index=ordinal,
                purpose='evaluation', save_to=path)
            progress['returned_generations'] += 1
            active_attempt = None
            with _temporary_file(path) as temporary:
                temporary.write_text(json.dumps(record, allow_nan=True) + '\n')
            row, errors = _measurement(record, request, checkpoint,
                                       archive['canonicals'][request['request_id']], slot)
            row.update(record_path=str(path), structural_errors=list(errors),
                       validation_status='invalid', valid_measurement=False,
                       errors=['validation_not_completed'])
            rows.append(row)
            errors.extend(_comparison_errors(row, rows, archive, seed, phase))
            errors = sorted(set(errors))
            row.update(errors=errors, validation_status='invalid' if errors else 'valid',
                       valid_measurement=not errors)
            write_json(root / (label + '_VALIDATION.json'), {
                'label': label, 'request_id': request['request_id'], 'method': method,
                'repeat': slot['repeat'], 'status': row['validation_status'], 'errors': errors,
                'canonical_source': archive['canonicals'][request['request_id']]['source_record_path']})
            save_measurements(root, rows)
            update_counts(progress, rows, active_attempt)
            write_json(root / 'STATUS.json', progress)
            if errors:
                raise RuntimeError('Native correctness failure: ' + str(errors))
            last_gate_repeat = 0 if phase == 'bridge' else 1
            if ordinal == 7 and slot['repeat'] == last_gate_repeat:
                write_json(root / f'QUALIFICATION_{method}.json', _gate(method, phase))
                qualified.append(method)
                progress['qualified_methods'] = list(qualified)
                if qualified == methods:
                    progress['qualification'] = 'passed'
                    write_json(root / 'QUALIFICATION.json', dict(status='passed', phase=phase,
                        methods=methods, generations=8 * len(methods) * (1 if phase == 'bridge' else 2),
                        extra_generations=0, performance_win_required=False))
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error=repr(error))
        if progress['qualification'] == 'pending':
            progress['qualification'] = 'incomplete'
        raise
    finally:
        update_counts(progress, rows, active_attempt)
        progress['evaluation_wall_s'] = time.perf_counter() - started
        save_measurements(root, rows)
        write_json(root / 'STATUS.json', progress)
    return progress


def evaluate_bridge(backend, requests, checkpoints, root, *, seed, archive):
    return _evaluate(backend, requests, checkpoints, root, seed=seed, phase='bridge', archive=archive)


def evaluate_final(backend, requests, checkpoints, root, *, seed, archive):
    return _evaluate(backend, requests, checkpoints, root, seed=seed, phase='final', archive=archive)


def audit_stage(folder, requests, checkpoints, *, seed, phase, archive):
    """Recheck saved native records for the report; never creates model work."""
    folder = Path(folder)
    errors, validated, progress = [], [], {}
    try:
        campaign = next((parent for parent in folder.parents if parent.name == 'campaign_001'), None)
        native_reader = ArchiveReader(campaign.parent if campaign is not None else folder)
        planned = _validate_inputs(requests, checkpoints, seed, phase, archive)
        protocol = json.loads((folder / 'PROTOCOL.json').read_text())
        progress = json.loads((folder / 'STATUS.json').read_text())
        rows = [json.loads(line) for line in (folder / 'MEASUREMENTS.jsonl').read_text().splitlines()]
        methods = list(ARCHIVED_METHODS if phase == 'bridge' else NEW_METHODS)
        if (protocol['schedule'] != planned or protocol['checkpoints'] != checkpoints
                or protocol['canonical_identity'] != archive['canonical_identity']
                or json.loads((folder / 'CANONICAL_INPUT.json').read_text()) != archive['canonicals']
                or json.loads((folder / 'REQUESTS.json').read_text()) != requests):
            errors.append('sealed_inputs_or_canonical_changed')
        if (len(rows) != len(planned)
                or [{key: row.get(key) for key in slot} for row, slot in zip(rows, planned)] != planned):
            errors.append('incomplete_or_changed_schedule')
        for key, expected in dict(status='completed', qualification='passed',
                planned_generations=len(planned), attempted_generations=len(planned),
                returned_generations=len(planned), completed_valid_generations=len(planned),
                invalid_generations=0, pending_canonical_generations=0,
                unknown_generation_attempts=0, unattempted_generations=0).items():
            if progress.get(key) != expected:
                errors.append('status_' + key)
        activations = [json.loads(line) for line in (folder / 'ACTIVATIONS.jsonl').read_text().splitlines()]
        if [row['method'] for row in activations] != methods:
            errors.append('checkpoint_batching_changed')
        for row in activations:
            if row['errors'] or publication_errors(row, checkpoints[row['method']]):
                errors.append('checkpoint_publication')
        for method in methods:
            weights = json.loads((folder / f'WEIGHTS_{method}.json').read_text())
            if weights['errors'] or weight_check_errors(weights['native_reply'], checkpoints[method]):
                errors.append('loaded_weights_' + method)
            if json.loads((folder / f'QUALIFICATION_{method}.json').read_text()) != _gate(method, phase):
                errors.append('included_gate_' + method)
        for saved, slot in zip(rows, planned):
            request = requests[slot['panel_ordinal']]
            label = f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}"
            raw = json.loads((folder / (label + '.json')).read_text())
            native_path = raw['metadata']['native_trace_path']
            if campaign is not None:
                native_reader.origin = Path(native_path.split('/campaign_001/', 1)[0])
            native = native_reader.read(native_reader.saved_path(native_path))
            if native != raw['metadata']['capture']:
                errors.append(label + ': original_native_capture_changed')
            row, invalid = _measurement(raw, request, checkpoints[slot['method']],
                                        archive['canonicals'][request['request_id']], slot)
            invalid.extend(_comparison_errors(row, [*validated, row], archive, seed, phase))
            if (invalid or not saved['valid_measurement'] or saved['validation_status'] != 'valid'
                    or saved['errors'] or saved['structural_errors']
                    or any(saved.get(key) != value for key, value in row.items())):
                errors.append(label + ': ' + repr(invalid or ['saved_measurement_disagreement']))
            validated.append({**saved, **row})
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        errors.append('missing_or_malformed_evidence: ' + repr(error))
    return dict(status='passed' if not errors else 'invalid_or_incomplete', rows=validated,
                errors=errors, progress=progress, latency_comparison=False)
