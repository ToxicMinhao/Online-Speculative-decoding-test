"""Four-arm capture-disabled native evaluation against immutable canonical outputs."""
from collections import defaultdict
import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.osd_persistent.run import score_record
from experiments.fresh_confirmation.protocol import (
    measurement_errors, publication_errors, repeat_errors, weight_check_errors,
)
from experiments.o_half_matched_lr.archive import ArchiveReader, validate_record
from experiments.kw_v1.evaluate import append_json, safe_scalars, update_counts
from .contract import METHODS, final_schedule, schedule_balance, validate_inputs

REPEATS = 3
SESSION = 'adaflash_arepo_evaluation'


def measurement(record, request, checkpoint, canonical, slot):
    label = f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}"
    try:
        errors = validate_record(record, request, checkpoint, canonical, label=label, session=SESSION)
        row = score_record(record, request_id=request['request_id'], repeat=slot['repeat'], method=slot['method'])
        row.update(finish_reason=record['finish_reason'],
                   cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'],
                   native_accepted_draft_tokens=sum(
                       block['native_accepted_draft_tokens'] for block in record['metadata']['trace']))
        errors.extend(measurement_errors(row, request, canonical))
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
        errors = ['malformed_native_record: ' + repr(error)]
        row = dict(request_id=request['request_id'], repeat=slot['repeat'], method=slot['method'],
                   checkpoint_id=str(checkpoint))
    row = safe_scalars(row)
    row.update(**slot, label=label, original_index=request['original_index'])
    return row, errors


def evaluate_final(backend, requests, checkpoints, root, *, canonicals, boundary_check=None):
    phase, arm, event = "final", "AREPO", 58
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    planned = validate_inputs(requests, checkpoints, canonicals)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows, grouped, checked = [], defaultdict(list), set()
    active_attempt = None
    started = time.perf_counter()
    progress = dict(status='running', seed=17, event=event, phase=phase, arm=arm,
        planned_generations=len(planned), attempted_generations=0, returned_generations=0,
        activation_s=0., weight_validation_s=0., extra_generations=0, automatic_retry=False,
        optimizer_steps=0, feedback_collections=0, qualification='per_record_native_checks')
    write_json(root / 'PROTOCOL.json', dict(seed=17, event=event, phase=phase, arm=arm,
        checkpoints=checkpoints, schedule=planned, capture=False, extra_warmup_generations=0,
        canonical_source='immutable_original_full_O_run_F', count_trajectory_repeat=0,
        historical_call_gate=False, performance_win_required=False,
        timing_scope='contemporaneous_checkpoint_ready_serving' if phase == 'final' else 'count_only_development'))
    write_json(root / 'REQUESTS.json', requests)
    write_json(root / 'CANONICAL_INPUT.json', canonicals)
    try:
        current = None
        active_batch = None
        for slot in planned:
            method, ordinal = slot['method'], slot['panel_ordinal']
            batch = (slot['block_ordinal'], slot['repeat'], slot['position'])
            if batch != active_batch and boundary_check is not None:
                append_json(root / 'BOUNDARY_CHECKS.jsonl', dict(batch=batch, check=boundary_check()))
            active_batch = batch
            checkpoint = checkpoints[method]
            if current != checkpoint:
                activation = backend.publish(checkpoint)
                errors = publication_errors(activation, checkpoint)
                append_json(root / 'ACTIVATIONS.jsonl', {**slot, **activation, 'errors': errors})
                progress['activation_s'] += activation['activation_s']
                if errors:
                    raise RuntimeError('Native checkpoint publication failed: ' + repr(errors))
                current = checkpoint
            if method not in checked:
                checking = time.perf_counter()
                weights = backend.check_weights(checkpoint)
                errors = weight_check_errors(weights, checkpoint)
                elapsed = time.perf_counter() - checking
                write_json(root / f'WEIGHTS_{method}.json', dict(native_reply=weights, errors=errors, validation_s=elapsed))
                progress['weight_validation_s'] += elapsed
                if errors:
                    raise RuntimeError('Loaded/exported weight mismatch: ' + repr(errors))
                checked.add(method)
            request = requests[ordinal]
            label = f"{ordinal:04d}_r{slot['repeat']}_{method}"
            path = root / (label + '.json')
            active_attempt = {**slot, 'label': label}
            progress['attempted_generations'] += 1
            update_counts(progress, rows, active_attempt)
            write_json(root / 'STATUS.json', progress)
            record = backend.generate(request['prompt_ids'], max_new_tokens=request['max_new_tokens'],
                request_id=label, session_id=SESSION, request_index=ordinal, purpose='evaluation', save_to=path)
            progress['returned_generations'] += 1
            active_attempt = None
            # Raw native evidence is durable before any scientific validation.
            with _temporary_file(path) as temporary:
                temporary.write_text(json.dumps(record, allow_nan=True) + '\n')
            row, errors = measurement(record, request, checkpoint, canonicals[request['request_id']], slot)
            row.update(record_path=str(path), structural_errors=list(errors))
            group = grouped[request['request_id'], method]
            errors.extend(repeat_errors([*group, row], method))
            if group and row.get('native_accepted_draft_tokens') != group[0].get('native_accepted_draft_tokens'):
                errors.append('repeat_native_accepted_count_mismatch')
            errors = sorted(set(errors))
            row.update(errors=errors, valid_measurement=not errors, validation_status='invalid' if errors else 'valid')
            rows.append(row)
            group.append(row)
            append_json(root / 'MEASUREMENTS.jsonl', row)
            write_json(root / (label + '_VALIDATION.json'), dict(status=row['validation_status'],
                errors=errors, canonical_source=canonicals[request['request_id']]['source_record_path']))
            update_counts(progress, rows, active_attempt)
            write_json(root / 'STATUS.json', progress)
            if errors:
                raise RuntimeError('Native evaluation correctness failed: ' + repr(errors))
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error=repr(error))
        raise
    finally:
        update_counts(progress, rows, active_attempt)
        progress['evaluation_wall_s'] = time.perf_counter() - started
        write_json(root / 'STATUS.json', progress)
    return progress


def audit_stage(folder, requests, checkpoints, *, canonicals, evidence_root=None):
    """Reconstruct valid rows from raw records, not the saved validity flag alone."""
    phase, arm, event = "final", "AREPO", 58
    folder = Path(folder)
    errors, validated, progress = [], [], {}
    try:
        checkpoints = {method: str(path) for method, path in checkpoints.items()}
        planned = validate_inputs(requests, checkpoints, canonicals)
        reader = ArchiveReader(evidence_root or folder)
        protocol = json.loads((folder / 'PROTOCOL.json').read_text())
        progress = json.loads((folder / 'STATUS.json').read_text())
        rows = [json.loads(line) for line in (folder / 'MEASUREMENTS.jsonl').read_text().splitlines()]
        if (protocol['schedule'] != planned or protocol['checkpoints'] != checkpoints
                or protocol['capture'] is not False or protocol['extra_warmup_generations'] != 0
                or protocol['historical_call_gate'] is not False
                or json.loads((folder / 'REQUESTS.json').read_text()) != requests
                or json.loads((folder / 'CANONICAL_INPUT.json').read_text()) != canonicals):
            errors.append('sealed_inputs_or_protocol_changed')
        if (len(rows) != len(planned)
                or [{key: row.get(key) for key in slot} for row, slot in zip(rows, planned)] != planned):
            errors.append('incomplete_or_changed_schedule')
        for key, expected in dict(status='completed', planned_generations=len(planned),
                attempted_generations=len(planned), returned_generations=len(planned),
                completed_valid_generations=len(planned), invalid_generations=0,
                unknown_generation_attempts=0, unattempted_generations=0,
                pending_canonical_generations=0, extra_generations=0).items():
            if progress.get(key) != expected:
                errors.append('status_' + key)
        expected_publications = []
        current = None
        for slot in planned:
            if checkpoints[slot['method']] != current:
                expected_publications.append(slot)
                current = checkpoints[slot['method']]
        activations = [json.loads(line) for line in (folder / 'ACTIVATIONS.jsonl').read_text().splitlines()]
        if len(activations) != len(expected_publications):
            errors.append('publication_count')
        for saved, slot in zip(activations, expected_publications):
            if (any(saved.get(key) != value for key, value in slot.items()) or saved['errors']
                    or publication_errors(saved, checkpoints[slot['method']])):
                errors.append('checkpoint_publication')
        for method, checkpoint in checkpoints.items():
            weights = json.loads((folder / f'WEIGHTS_{method}.json').read_text())
            if weights['errors'] or weight_check_errors(weights['native_reply'], checkpoint):
                errors.append('loaded_weights_' + method)
        grouped = defaultdict(list)
        for saved, slot in zip(rows, planned):
            ordinal, method = slot['panel_ordinal'], slot['method']
            request = requests[ordinal]
            label = f"{ordinal:04d}_r{slot['repeat']}_{method}"
            raw = json.loads((folder / (label + '.json')).read_text())
            native_path = raw['metadata']['native_trace_path']
            if '/campaign_001/' in native_path:
                reader.origin = Path(native_path.split('/campaign_001/', 1)[0])
            native = reader.read(reader.saved_path(native_path))
            if native != raw['metadata']['capture']:
                errors.append(label + ': original_native_capture_changed')
            row, invalid = measurement(raw, request, checkpoints[method], canonicals[request['request_id']], slot)
            group = grouped[request['request_id'], method]
            invalid.extend(repeat_errors([*group, row], method))
            if group and row.get('native_accepted_draft_tokens') != group[0].get('native_accepted_draft_tokens'):
                invalid.append('repeat_native_accepted_count_mismatch')
            if (invalid or saved['validation_status'] != 'valid' or saved['valid_measurement'] is not True
                    or saved['errors'] or saved['structural_errors']
                    or any(saved.get(key) != value for key, value in row.items())):
                errors.append(label + ': ' + repr(invalid or ['saved_measurement_disagreement']))
            group.append(row)
            validated.append({**saved, **row})
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        errors.append('missing_or_malformed_evidence: ' + repr(error))
    return dict(status='passed' if not errors else 'invalid_or_incomplete', rows=validated,
                errors=errors, progress=progress, phase=phase, arm=arm, event=event,
                latency_comparison=phase == 'final')
