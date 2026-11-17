"""Fixed N/C/S MathQA serving: 36 activations, 72 warmups, 2,304 scores.

Reuse the qualified native record and state validators while keeping this
three-arm execution contract separate from the earlier two-arm experiment.
"""
from collections import defaultdict
import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.fresh_confirmation.evaluate import append_json
from experiments.fresh_confirmation.protocol import publication_errors, weight_check_errors
from experiments.persistent_prefix_withholding import evaluation as native
from experiments.mathqa_fixed_lr_transfer.evaluation import (
    validate_references, validate_inputs, execution_slots, _repeat_errors,
)
from .data import evaluation_schedule

METHODS = ("N", "C", "S")
PANEL_SIZE, BLOCK_SIZE, REPEATS = 256, 64, 3
ACTIVATIONS = 36
SCORED, WARMUPS, TOTAL = 2304, 72, 2376
SCOPE = "UNCHANGED_C_MATHQA_TRANSFER_AUDITED_UNEXPOSED_PANEL"
SESSION = "unchanged_c_mathqa_transfer_event6"
label_for = native.label_for
boundary_errors = native.boundary_errors


def measure(record, request, checkpoint, canonical, slot, physical_gpu_uuid):
    row, errors = native.measure(record, request, checkpoint, canonical, slot,
                                 physical_gpu_uuid, session=SESSION)
    row["scope"] = SCOPE
    return row, errors


def evaluate_final(backend, requests, checkpoints, root, *, canonicals,
                   warmup_requests, warmup_canonicals, schedule, origin_checkpoint,
                   physical_gpu_uuid, boundary_check, execution_commit):
    """Preserve each native return before checking it; stop without retries."""
    if schedule != evaluation_schedule(requests):
        raise ValueError("The sealed three-arm schedule changed")
    checkpoints = {m: str(path) for m, path in checkpoints.items()}
    if (set(checkpoints) != set(METHODS) or len(set(checkpoints.values())) != 3
            or str(origin_checkpoint) in checkpoints.values()):
        raise ValueError("Require three distinct N6/C6/S6 endpoints, separate from KPLUS4")
    validate_inputs(requests, canonicals, warmup_requests, warmup_canonicals, origin_checkpoint)
    if (not physical_gpu_uuid or boundary_check is None or len(execution_commit) != 40
            or any(c not in '0123456789abcdef' for c in execution_commit)):
        raise ValueError("Require the fixed physical GPU, boundary checks and execution commit")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'PROTOCOL.json', dict(schedule=schedule, checkpoints=checkpoints,
        methods=list(METHODS), event=6, scope=SCOPE, session=SESSION, capture=False,
        origin_checkpoint=str(origin_checkpoint), historical_count_gate=False,
        physical_gpu_uuid=physical_gpu_uuid, execution_commit=execution_commit,
        planned_scored=SCORED, planned_warmups=WARMUPS))
    write_json(root/'REQUESTS.json', requests)
    write_json(root/'CANONICAL.json', canonicals)
    write_json(root/'WARMUPS.json', dict(requests=warmup_requests, canonicals=warmup_canonicals))
    rows, groups, checked = [], defaultdict(list), set()
    progress = dict(status='running', planned_generations=TOTAL, attempted_generations=0,
        returned_generations=0, publications=0, logical_activations=0, activation_s=0.,
        weight_validation_s=0., optimizer_steps=0, feedback_collections=0,
        teacher_preparations=0, automatic_retry=False)
    pending, active, started = None, None, time.perf_counter()

    def persist():
        progress.update(completed_valid_generations=sum(r['valid_measurement'] for r in rows),
            invalid_generations=sum(not r['valid_measurement'] for r in rows),
            scored_generations=sum(r['phase'] == 'scored' for r in rows),
            warmup_generations=sum(r['phase'] == 'warmup' for r in rows),
            unknown_generation_attempts=int(pending is not None), interrupted_attempt=pending,
            unattempted_generations=TOTAL-progress['attempted_generations'])
        write_json(root/'STATUS.json', progress)

    def boundary(activation):
        receipt = boundary_check()
        errors = boundary_errors(receipt, physical_gpu_uuid)
        append_json(root/'BOUNDARY_CHECKS.jsonl', dict(activation=activation, check=receipt, errors=errors))
        if errors:
            raise RuntimeError('Evaluation boundary failed: ' + repr(errors))

    try:
        for slot, request in execution_slots(schedule, requests, warmup_requests):
            method, checkpoint = slot['method'], checkpoints[slot['method']]
            if active != slot['activation']:
                active = slot['activation']
                boundary(active)
                receipt = backend.publish(checkpoint)
                errors = publication_errors(receipt, checkpoint)
                progress['publications'] += 1
                progress['logical_activations'] += 1
                progress['activation_s'] += receipt['activation_s']
                append_json(root/'ACTIVATIONS.jsonl', {**schedule[active], **receipt, 'errors': errors})
                if errors:
                    raise RuntimeError('Checkpoint publication failed: ' + repr(errors))
                if method not in checked:
                    before = time.perf_counter()
                    reply = backend.check_weights(checkpoint)
                    errors = weight_check_errors(reply, checkpoint)
                    elapsed = time.perf_counter()-before
                    progress['weight_validation_s'] += elapsed
                    write_json(root/f'WEIGHTS_{method}.json', dict(native_reply=reply, errors=errors, validation_s=elapsed))
                    if errors:
                        raise RuntimeError('Published weights differ: ' + repr(errors))
                    checked.add(method)
            label = label_for(slot)
            path = root/(label+'.json')
            pending = dict(slot, label=label)
            progress['attempted_generations'] += 1
            persist()
            record = backend.generate(request['prompt_ids'], max_new_tokens=2048,
                request_id=label, session_id=SESSION, request_index=request['panel_ordinal'],
                purpose='evaluation', save_to=path)
            progress['returned_generations'] += 1
            with _temporary_file(path) as temporary:
                temporary.write_text(json.dumps(record, allow_nan=True)+'\n')
            canonical = (warmup_canonicals if slot['phase']=='warmup' else canonicals)[request['request_id']]
            row, errors = measure(record, request, checkpoint, canonical, slot, physical_gpu_uuid)
            previous = groups[slot['phase'], method, request['request_id']]
            errors.extend(_repeat_errors(previous, row))
            row.update(record_path=str(path), errors=sorted(set(errors)),
                validation_status='invalid' if errors else 'valid', valid_measurement=not errors)
            rows.append(row)
            previous.append(row)
            pending = None
            append_json(root/'MEASUREMENTS.jsonl', row)
            write_json(root/(label+'_VALIDATION.json'), dict(status=row['validation_status'], errors=row['errors']))
            persist()
            if errors:
                raise RuntimeError('Native correctness failure: ' + repr(errors))
        boundary('after_final')
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error=repr(error))
        raise
    finally:
        progress['evaluation_wall_s'] = time.perf_counter()-started
        persist()
    return progress


def _audit_raw_records(root, native_trace_resolver):
    read = lambda path: json.loads(Path(path).read_text())
    lines = lambda path: [json.loads(s) for s in Path(path).read_text().splitlines()]
    protocol, requests = read(root/'PROTOCOL.json'), read(root/'REQUESTS.json')
    canonicals, warmups = read(root/'CANONICAL.json'), read(root/'WARMUPS.json')
    planned = evaluation_schedule(requests)
    validate_inputs(requests, canonicals, warmups['requests'], warmups['canonicals'], protocol['origin_checkpoint'])
    errors, rebuilt, groups = [], [], defaultdict(list)
    expected_protocol = dict(schedule=planned, methods=list(METHODS), event=6, scope=SCOPE,
        session=SESSION, capture=False, planned_scored=SCORED, planned_warmups=WARMUPS,
        historical_count_gate=False)
    errors.extend('protocol_'+k for k,v in expected_protocol.items() if protocol.get(k)!=v)
    commit = protocol.get('execution_commit', '')
    if (not protocol.get('physical_gpu_uuid') or len(commit) != 40
            or any(c not in '0123456789abcdef' for c in commit)):
        errors.append('execution_identity')
    if (set(protocol['checkpoints'])!=set(METHODS)
            or len(set(protocol['checkpoints'].values()))!=3
            or protocol['origin_checkpoint'] in protocol['checkpoints'].values()):
        errors.append('endpoint_identity')
    status = read(root/'STATUS.json')
    expected_status = dict(status='completed', planned_generations=TOTAL, attempted_generations=TOTAL,
        returned_generations=TOTAL, completed_valid_generations=TOTAL, invalid_generations=0,
        scored_generations=SCORED, warmup_generations=WARMUPS, publications=ACTIVATIONS,
        logical_activations=ACTIVATIONS, unknown_generation_attempts=0, unattempted_generations=0,
        optimizer_steps=0, feedback_collections=0, teacher_preparations=0)
    errors.extend('status_'+k for k,v in expected_status.items() if status.get(k)!=v)
    activations = lines(root/'ACTIVATIONS.jsonl')
    if len(activations)!=ACTIVATIONS:
        errors.append('activation_count')
    for saved, entry in zip(activations, planned):
        if (any(saved.get(k)!=v for k,v in entry.items()) or saved.get('errors')
                or publication_errors(saved, protocol['checkpoints'][entry['method']])):
            errors.append('activation_identity')
    for method in METHODS:
        saved = read(root/f'WEIGHTS_{method}.json')
        if saved['errors'] or weight_check_errors(saved['native_reply'], protocol['checkpoints'][method]):
            errors.append('weight_identity_'+method)
    boundaries = lines(root/'BOUNDARY_CHECKS.jsonl')
    if [r['activation'] for r in boundaries] != [*range(ACTIVATIONS),'after_final']:
        errors.append('boundary_schedule')
    for row in boundaries:
        errors.extend(row['errors'])
        errors.extend(boundary_errors(row['check'], protocol['physical_gpu_uuid']))
    saved_rows = lines(root/'MEASUREMENTS.jsonl')
    expected = list(execution_slots(planned, requests, warmups['requests']))
    if len(saved_rows)!=TOTAL:
        errors.append('measurement_count')
    for saved, (slot, request) in zip(saved_rows, expected):
        label = label_for(slot)
        record = read(root/(label+'.json'))
        if read(native_trace_resolver(record['metadata']['native_trace_path'])) != record['metadata']['capture']:
            errors.append(label+':native_capture_changed')
        canonical = (warmups['canonicals'] if slot['phase']=='warmup' else canonicals)[request['request_id']]
        row, invalid = measure(record, request, protocol['checkpoints'][slot['method']],
                               canonical, slot, protocol['physical_gpu_uuid'])
        if any(saved.get(k)!=v for k,v in row.items()):
            invalid.append('saved_measurement_differs_from_raw')
        if saved.get('valid_measurement') is not True or saved.get('errors') or saved.get('validation_status')!='valid':
            invalid.append('saved_invalid')
        previous = groups[slot['phase'], slot['method'], request['request_id']]
        invalid.extend(_repeat_errors(previous, row))
        previous.append(row)
        errors.extend(label+':'+e for e in sorted(set(invalid)))
        row.update(valid_measurement=not invalid, errors=invalid,
                   validation_status='invalid' if invalid else 'valid', record_path=str(root/(label+'.json')))
        rebuilt.append(row)
    return dict(status='passed' if not errors else 'invalid_or_incomplete', errors=errors,
        rows=rebuilt, execution_commit=protocol['execution_commit'], extra_model_work=0)


def audit_raw_records(root, *, native_trace_resolver=None):
    """Reapply native validity to retained returns, including their separate traces."""
    try:
        return _audit_raw_records(Path(root), native_trace_resolver or Path)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        return dict(status='invalid_or_incomplete', errors=['evidence_read_or_schema:'+repr(error)],
                    rows=[], extra_model_work=0)
