"""One native session, fixed order, and durable validation of every execution."""
from collections import defaultdict
import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.osd_persistent.run import score_record
from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, measurement_errors, publication_errors, repeat_errors, weight_check_errors,
)
from experiments.fresh_confirmation.evaluate import save_measurements, update_counts
from experiments.o_half_matched_lr.archive import ArchiveReader, validate_record
from experiments.kw_v1.evaluate import append_json, safe_scalars
from .contract import METHODS, final_schedule, validate_inputs

SESSION = 'mathqa_transfer_evaluation'


def label_for(slot):
    if slot['phase'] == 'control':
        return 'historical_control_' + slot['method']
    return f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}"


def measure(record, request, checkpoint, canonical, slot):
    label = label_for(slot)
    try:
        errors = validate_record(record, request, checkpoint, canonical or record,
                                 label=label, session=SESSION)
        row = score_record(record, request_id=request['request_id'], repeat=slot['repeat'], method=slot['method'])
        row.update(finish_reason=record['finish_reason'],
                   cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
        errors.extend(measurement_errors(row, request, canonical))
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
        errors = ['malformed_native_record: ' + repr(error)]
        row = dict(request_id=request['request_id'], repeat=slot['repeat'], method=slot['method'],
                   checkpoint_id=str(checkpoint))
    row = safe_scalars(row)
    row.update(**slot, label=label, original_index=request['original_index'])
    return row, sorted(set(errors))


def activate(backend, checkpoint, slot, root, progress, checked):
    if backend.checkpoint_id != checkpoint or not progress['publications']:
        receipt = backend.publish(checkpoint)
        errors = publication_errors(receipt, checkpoint)
        append_json(root / 'ACTIVATIONS.jsonl', {**slot, **receipt, 'errors': errors})
        progress['activation_s'] += receipt['activation_s']
        progress['publications'] += 1
        if errors:
            raise RuntimeError('Checkpoint publication failed: ' + repr(errors))
    method = slot['method']
    if method not in checked:
        started = time.perf_counter()
        reply = backend.check_weights(checkpoint)
        errors = weight_check_errors(reply, checkpoint)
        elapsed = time.perf_counter() - started
        write_json(root / f'WEIGHTS_{method}.json', dict(native_reply=reply, errors=errors, validation_s=elapsed))
        progress['weight_validation_s'] += elapsed
        if errors:
            raise RuntimeError('Loaded/exported weights differ: ' + repr(errors))
        checked.add(method)


def validate_rows(root, request, rows, canonical):
    """Resolve records generated before this request's O repeat-zero reference."""
    for row in rows:
        errors = list(row['structural_errors'])
        if canonical is not None:
            record = json.loads(Path(row['record_path']).read_text())
            if record['response_ids'] != canonical['response_ids']:
                errors.append('canonical_response_ids')
            if record['finish_reason'] != canonical['finish_reason']:
                errors.append('canonical_finish_reason')
        errors.extend(repeat_errors(rows, row['method']))
        errors = sorted(set(errors))
        status = 'invalid' if errors else 'canonical_pending' if canonical is None else 'valid'
        previous = (row.get('validation_status'), row.get('errors'))
        row.update(errors=errors, validation_status=status, valid_measurement=status == 'valid')
        if previous != (status, errors):
            receipt = dict(label=row['label'], status=status, errors=errors,
                canonical_source=canonical['source_record_path'] if canonical else None)
            append_json(root / 'VALIDATION_EVENTS.jsonl', receipt)
            write_json(root / (row['label'] + '_VALIDATION.json'), receipt)


def evaluate(backend, requests, checkpoints, root, *, control=None, boundary_check=None, production=True):
    """Controls and timing use the same backend; controls are never timing samples."""
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    planned = validate_inputs(requests, checkpoints, production=production) if control is None else [
        dict(seed=17, event=58, phase='control', block_ordinal=-1, panel_ordinal=0,
             request_id=requests[0]['request_id'], repeat=0, method=method, position=i)
        for i, method in enumerate(METHODS)]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows, grouped, canonicals, checked = [], defaultdict(list), {}, set()
    active_attempt, active_batch = None, None
    started = time.perf_counter()
    progress = dict(status='running', planned_generations=len(planned), attempted_generations=0,
        returned_generations=0, activation_s=0., weight_validation_s=0., publications=0,
        optimizer_steps=0, feedback_collections=0, teacher_preparations=0,
        extra_generations=0, automatic_retry=False)
    write_json(root / 'PROTOCOL.json', dict(schedule=planned, checkpoints=checkpoints, capture=False,
        extra_warmup_generations=0, canonical_source='retained historical F' if control else 'O58_repeat_0',
        historical_call_gate=control is not None, count_trajectory_repeat=0))
    write_json(root / 'REQUESTS.json', requests)
    write_json(root / 'CANONICAL.json', canonicals)
    try:
        for slot in planned:
            method, ordinal = slot['method'], slot['panel_ordinal']
            batch = (slot['block_ordinal'], slot['repeat'], slot['position'])
            if batch != active_batch and boundary_check is not None:
                append_json(root / 'BOUNDARY_CHECKS.jsonl', dict(batch=batch, check=boundary_check()))
            active_batch = batch
            checkpoint = checkpoints[method]
            activate(backend, checkpoint, slot, root, progress, checked)
            request = requests[ordinal]
            label = label_for(slot)
            path = root / (label + '.json')
            active_attempt = {**slot, 'label': label}
            progress['attempted_generations'] += 1
            update_counts(progress, rows, active_attempt)
            write_json(root / 'STATUS.json', progress)
            record = backend.generate(request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
                request_id=label, session_id=SESSION, request_index=ordinal, purpose='evaluation', save_to=path)
            progress['returned_generations'] += 1
            active_attempt = None
            with _temporary_file(path) as temporary:
                temporary.write_text(json.dumps(record, allow_nan=True) + '\n')
            canonical = control['canonical'] if control else canonicals.get(request['request_id'])
            row, errors = measure(record, request, checkpoint, canonical, slot)
            if control:
                errors.extend('historical_' + key for key in COUNT_FIELDS
                              if row.get(key) != control['counts'][method][key])
            row.update(record_path=str(path), structural_errors=sorted(set(errors)))
            rows.append(row)
            group = grouped[request['request_id']]
            group.append(row)
            if not control and method == 'O' and slot['repeat'] == 0 and not errors:
                canonical = dict(response_ids=record['response_ids'], finish_reason=record['finish_reason'],
                    source_record_path=str(path), source_method='O', source_repeat=0)
                canonicals[request['request_id']] = canonical
                write_json(root / 'CANONICAL.json', canonicals)
            validate_rows(root, request, group, canonical)
            save_measurements(root, rows)
            update_counts(progress, rows, active_attempt)
            write_json(root / 'STATUS.json', progress)
            if any(r['validation_status'] == 'invalid' for r in group):
                raise RuntimeError('Native comparison invalid: ' + repr([r['errors'] for r in group if r['errors']]))
        if any(row['validation_status'] != 'valid' for row in rows):
            raise RuntimeError('Canonical validation remains incomplete')
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error=repr(error))
        raise
    finally:
        update_counts(progress, rows, active_attempt)
        progress['evaluation_wall_s'] = time.perf_counter() - started
        save_measurements(root, rows)
        write_json(root / 'STATUS.json', progress)
    return progress


def audit_stage(folder, requests, checkpoints, *, evidence_root=None, control=None, production=True):
    """Rebuild validity from raw outputs and original native traces, including failures."""
    folder = Path(folder)
    errors, validated, progress = [], [], {}
    try:
        checkpoints = {method: str(path) for method, path in checkpoints.items()}
        protocol = json.loads((folder / 'PROTOCOL.json').read_text())
        planned = final_schedule(requests) if control is None else [
            dict(seed=17, event=58, phase='control', block_ordinal=-1, panel_ordinal=0,
                 request_id=requests[0]['request_id'], repeat=0, method=method, position=i)
            for i, method in enumerate(METHODS)]
        if control is None:
            validate_inputs(requests, checkpoints, production=production)
        progress = json.loads((folder / 'STATUS.json').read_text())
        rows = [json.loads(line) for line in (folder / 'MEASUREMENTS.jsonl').read_text().splitlines()]
        if (protocol['schedule'] != planned or protocol['checkpoints'] != checkpoints
                or protocol['capture'] is not False or protocol['extra_warmup_generations'] != 0
                or protocol['historical_call_gate'] != (control is not None)
                or protocol['canonical_source'] != ('retained historical F' if control else 'O58_repeat_0')
                or json.loads((folder / 'REQUESTS.json').read_text()) != requests):
            errors.append('sealed_protocol_changed')
        if len(rows) != len(planned) or [{k: r.get(k) for k in slot} for r, slot in zip(rows, planned)] != planned:
            errors.append('incomplete_or_changed_schedule')
        for key, expected in dict(status='completed', planned_generations=len(planned),
                attempted_generations=len(planned), returned_generations=len(planned),
                completed_valid_generations=len(planned), invalid_generations=0,
                unknown_generation_attempts=0, unattempted_generations=0,
                pending_canonical_generations=0, optimizer_steps=0, feedback_collections=0,
                teacher_preparations=0, extra_generations=0).items():
            if progress.get(key) != expected:
                errors.append('status_' + key)
        saved_canonicals = json.loads((folder / 'CANONICAL.json').read_text())
        if not control and set(saved_canonicals) != {r['request_id'] for r in requests}:
            errors.append('canonical_identity_set')
        for method in METHODS:
            weights = json.loads((folder / f'WEIGHTS_{method}.json').read_text())
            if weights['errors'] or weight_check_errors(weights['native_reply'], checkpoints[method]):
                errors.append('loaded_weights_' + method)
        activations = [json.loads(line) for line in (folder / 'ACTIVATIONS.jsonl').read_text().splitlines()]
        expected_slots, current = [], None
        for slot in planned:
            if slot['method'] != current:
                expected_slots.append(slot)
                current = slot['method']
        if len(activations) != len(expected_slots):
            errors.append('publication_count')
        for saved, slot in zip(activations, expected_slots):
            if (any(saved.get(k) != v for k, v in slot.items()) or saved['errors']
                    or publication_errors(saved, checkpoints[slot['method']])):
                errors.append('publication_identity')
        boundary_file = folder / 'BOUNDARY_CHECKS.jsonl'
        if production or boundary_file.exists():
            boundaries = [json.loads(line) for line in boundary_file.read_text().splitlines()]
            expected_batches = []
            for slot in planned:
                batch = [slot['block_ordinal'], slot['repeat'], slot['position']]
                if not expected_batches or batch != expected_batches[-1]:
                    expected_batches.append(batch)
            if [row['batch'] for row in boundaries] != expected_batches:
                errors.append('boundary_check_schedule')
            for row in boundaries:
                check = row['check']
                if (check['storage']['status'] != 'passed'
                        or check['storage']['free_bytes'] < check['storage']['required_bytes']
                        or check['quiescence']['status'] != 'passed'
                        or any(check['quiescence'][key] for key in ('training', 'transfers', 'foreign_gpu_processes'))):
                    errors.append('failed_storage_or_quiescence_boundary')
        reader = ArchiveReader(evidence_root or folder)
        grouped = defaultdict(list)
        for saved, slot in zip(rows, planned):
            request, method = requests[slot['panel_ordinal']], slot['method']
            label = label_for(slot)
            raw = json.loads((folder / (label + '.json')).read_text())
            native_path = raw['metadata']['native_trace_path']
            if '/campaign_001/' in native_path:
                reader.origin = Path(native_path.split('/campaign_001/', 1)[0])
            if reader.read(reader.saved_path(native_path)) != raw['metadata']['capture']:
                errors.append(label + ': native_capture_changed')
            if control:
                canonical = control['canonical']
            else:
                reference = json.loads((folder / f"{slot['panel_ordinal']:04d}_r0_O.json").read_text())
                canonical = saved_canonicals[request['request_id']]
                if (canonical['response_ids'] != reference['response_ids']
                        or canonical['finish_reason'] != reference['finish_reason']
                        or canonical['source_method'] != 'O' or canonical['source_repeat'] != 0):
                    errors.append(label + ': canonical_changed')
            row, invalid = measure(raw, request, checkpoints[method], canonical, slot)
            if control:
                invalid.extend('historical_' + key for key in COUNT_FIELDS
                               if row.get(key) != control['counts'][method][key])
            group = grouped[request['request_id'], method]
            invalid.extend(repeat_errors([*group, row], method))
            if (invalid or saved.get('validation_status') != 'valid' or saved.get('valid_measurement') is not True
                    or saved.get('errors') or saved.get('structural_errors')
                    or any(saved.get(k) != v for k, v in row.items())):
                errors.append(label + ': ' + repr(invalid or ['saved_measurement_disagreement']))
            group.append(row)
            validated.append({**saved, **row})
    except (OSError, KeyError, ValueError, TypeError, IndexError, RuntimeError) as error:
        errors.append('missing_or_malformed_evidence: ' + repr(error))
    return dict(status='passed' if not errors else 'invalid_or_incomplete', errors=errors,
                rows=validated, progress=progress, model_work=0)
