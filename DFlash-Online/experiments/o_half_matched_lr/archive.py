"""Validate immutable PR21 native evidence before reusing its count trajectories.

The archived canonical tokens remain the output reference. This module never
generates a replacement reference or treats a public scalar table as raw data.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from online_feedback.storage import write_json
from experiments.osd_persistent.run import score_record
from experiments.fresh_confirmation.protocol import (
    COUNT_FIELDS, measurement_errors, publication_errors, record_errors,
    repeat_errors, weight_check_errors,
)
from experiments.ok_coding.prepare import canonical_digest
from experiments.k_prefix_protection_v1.contract import validate_plan
from experiments.k_prefix_protection_v1.prepare import evaluation_schedule


PR21_COMMIT = 'd10521a1aa7d256bb75b37a6aae07586ae7cc8b3'
PANEL_BINDING = '8a2d3857a54c069bce655f5dda107ce0bd6ec30ebbc2b72dce20bc8a7d05295f'
ARCHIVED_METHODS = ('F', 'K', 'K_HALF')
ALL_PR21_METHODS = ('F', 'K', 'K_HALF', 'KP')
EXPECTED_CALL_TOTALS = {
    '17': {'F': 8150, 'K': 8140, 'K_HALF': 7949},
    '29': {'F': 8150, 'K': 8163, 'K_HALF': 7916},
}


def full_trace_errors(record):
    """Check native proposals and posterior decisions, beyond scalar totals."""
    metadata = record['metadata']
    capture = metadata['capture']
    response = record['response_ids']
    errors = []
    if (capture['prompt_ids'] != record['prompt_ids']
            or capture['response_ids'] != response
            or capture['blocks'] != metadata['trace']):
        errors.append('native_capture_binding')
    for number, block in enumerate(metadata['trace']):
        proposed = block['draft_tokens']
        posterior = block['target_argmax']
        width = block['verify_size']
        progress = block['committed_progress']
        if not (type(width) is int and width == len(proposed) + 1 == len(posterior)
                and 1 <= width <= 16 and type(progress) is int and 1 <= progress <= width
                and all(type(token) is int and token >= 0 for token in [*proposed, *posterior])):
            errors.append(f'block_{number}_geometry')
            continue
        first_error = next((i for i in range(width - 1) if proposed[i] != posterior[i]), width - 1)
        accepted = first_error
        proposed_commit = proposed[:accepted] + [posterior[accepted]]
        start = len(record['prompt_ids']) + block['output_start']
        if (progress > accepted + 1
                or block['committed_tokens'] != proposed_commit[:progress]
                or block['committed_tokens'] != posterior[:progress]
                or block['positions'] != list(range(start, start + width))
                or block['accepted_draft_tokens'] != min(accepted, progress)
                or block['native_accepted_draft_tokens'] != progress - 1
                or block['raw_accepted_draft_tokens_before_stop'] != accepted
                or block['stop_clipped'] != (progress < accepted + 1)):
            errors.append(f'block_{number}_native_decision')
    return errors


def validate_record(record, request, checkpoint, canonical, *, label, session):
    """Return all structural errors while leaving raw invalid evidence intact."""
    errors = record_errors(record, request, checkpoint, canonical)
    errors.extend(full_trace_errors(record))
    if (record['record_id'] != label or record['request_index'] != request['panel_ordinal']
            or record['session_id'] != session):
        errors.append('request_identity')
    return errors


class ArchiveReader:
    """Read each authoritative file once and retain its exact source identity."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}
        self.origin = None
        self.native_locations = {}
        index = self.root / 'audit_inputs/NATIVE_TRACE_REFERENCES.json'
        if index.exists():
            for row in self.read(index)['references']:
                self.native_locations[row['recorded_native_path']] = self.root / row['local_relative_path']

    def read(self, path, *, lines=False):
        path = Path(path)
        payload = path.read_bytes()
        self.files[str(path)] = {
            'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        if lines:
            return [json.loads(line) for line in payload.splitlines() if line]
        return json.loads(payload)

    def saved_path(self, recorded):
        """Resolve a recorded path, including an explicitly indexed local mirror."""
        recorded = str(recorded)
        if recorded in self.native_locations:
            return self.native_locations[recorded]
        path = Path(recorded)
        if path.is_absolute() and self.origin is not None:
            try:
                return self.root / path.relative_to(self.origin)
            except ValueError:
                if path.is_file():
                    return path
                raise ValueError('Missing explicitly referenced native evidence: ' + recorded)
        return path if path.is_absolute() else self.root / path


def _require(value, message):
    if not value:
        raise ValueError(message)


def audit_archive(pr21_root, output_path=None):
    """Audit every PR21 repeat and native trace, returning count-only reuse data.

The returned canonical objects and source path/hash records are unchanged.
The parent campaign additionally verifies model/state tensors and runtime code.
An optional receipt is saved on both success and failure; no model work occurs.
"""
    reader = ArchiveReader(pr21_root)
    result = {'status': 'running', 'source_root': str(reader.root), 'errors': [],
              'raw_generations': 0, 'native_blocks': 0, 'extra_model_work': 0}
    try:
        plan = reader.read(reader.root / 'sealed/PLAN.json')
        validate_plan(plan)
        _require(plan['execution_source']['commit'] == PR21_COMMIT, 'PR21 execution source changed')
        requests = plan['evaluation_requests']
        _require(plan['training_binding']['evaluation_binding'] == PANEL_BINDING
                 and canonical_digest(requests) == PANEL_BINDING, 'Exact PR21 panel binding changed')
        canonical_path = reader.root / 'campaign_001/EVALUATION_CANONICAL.json'
        canonicals = reader.read(canonical_path)
        _require(set(canonicals) == {r['request_id'] for r in requests}, 'Canonical panel is incomplete')
        first_source = canonicals[requests[0]['request_id']]['source_record_path']
        reader.origin = Path(first_source.split('/campaign_001/', 1)[0])
        counts, checkpoints, totals = {}, {}, {}
        for seed in (17, 29):
            folder = reader.root / f'campaign_001/seed_{seed}/event_004/evaluation'
            protocol = reader.read(folder / 'PROTOCOL.json')
            status = reader.read(folder / 'STATUS.json')
            planned = evaluation_schedule(requests, seed=seed)
            rows = reader.read(folder / 'MEASUREMENTS.jsonl', lines=True)
            _require(protocol['schedule'] == planned and len(rows) == len(planned) == 1536,
                     f'{seed}: incomplete archived schedule')
            _require([{key: row.get(key) for key in slot} for row, slot in zip(rows, planned)] == planned,
                     f'{seed}: archived chronological slots changed')
            expected = {'F': plan['draft'], 'K': plan['original_k4'][str(seed)]['export']}
            for method in ('K_HALF', 'KP'):
                endpoint = reader.read(reader.root / f'campaign_001/seed_{seed}/{method}/events/event_004/RESULT.json')
                _require(endpoint['status'] == 'completed' and endpoint['end_step'] == 512,
                         f'{seed}/{method}: missing archived endpoint')
                expected[method] = endpoint['export_path']
            _require(protocol['checkpoints'] == expected, f'{seed}: checkpoint binding changed')
            for name, value in dict(status='completed', qualification='passed',
                    attempted_generations=1536, returned_generations=1536,
                    completed_valid_generations=1536, invalid_generations=0,
                    pending_canonical_generations=0, unknown_generation_attempts=0,
                    unattempted_generations=0).items():
                _require(status[name] == value, f'{seed}: archived status {name}')
            gate = reader.read(folder / 'QUALIFICATION.json')
            _require(gate['status'] == 'passed' and gate['requests'] == 8
                     and gate['generations'] == 96 and gate['extra_generations'] == 0,
                     f'{seed}: archived correctness gate')
            for method in ALL_PR21_METHODS:
                weights = reader.read(folder / f'WEIGHTS_{method}.json')
                _require(not weights['errors'] and not weight_check_errors(weights['native_reply'], expected[method]),
                         f'{seed}/{method}: archived loaded-weight equality')
            for activation in reader.read(folder / 'ACTIVATIONS.jsonl', lines=True):
                _require(not activation['errors'] and not publication_errors(activation, expected[activation['method']]),
                         f'{seed}: archived checkpoint publication')
            grouped = defaultdict(list)
            for row, slot in zip(rows, planned):
                request = requests[slot['panel_ordinal']]
                reference = canonicals[request['request_id']]
                label = f"{slot['panel_ordinal']:04d}_r{slot['repeat']}_{slot['method']}"
                raw_path = folder / (label + '.json')
                _require(reader.saved_path(row['record_path']).resolve() == raw_path.resolve(),
                         label + ': archived raw path changed')
                record = reader.read(raw_path)
                errors = validate_record(record, request, expected[slot['method']], reference,
                    label=label, session='k_prefix_protection_v1_evaluation')
                measured = score_record(record, request_id=request['request_id'],
                                        repeat=slot['repeat'], method=slot['method'])
                measured.update(finish_reason=record['finish_reason'],
                                cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
                errors.extend(measurement_errors(measured, request, reference))
                _require(not errors and row['validation_status'] == 'valid'
                         and row['valid_measurement'] is True and not row['errors']
                         and not row['structural_errors'], label + ': invalid archived record ' + str(errors))
                _require(all(row[key] == value for key, value in measured.items()),
                         label + ': archived measurement differs from raw record')
                source_path = record['metadata']['native_trace_path']
                native = reader.read(reader.saved_path(source_path))
                _require(native == record['metadata']['capture'], label + ': original native capture differs')
                _require(reference['source_cell'] == [17, 4] and reference['source_method'] == 'F'
                         and reference['source_repeat'] == 0 and reference['F_checkpoint'] == plan['draft'],
                         label + ': canonical source changed')
                source = reader.root / f"campaign_001/seed_17/event_004/evaluation/{slot['panel_ordinal']:04d}_r0_F.json"
                _require(reader.saved_path(reference['source_record_path']).resolve() == source.resolve(),
                         label + ': canonical original path changed')
                if slot['method'] == 'F':
                    _require(all(measured[key] == reference['F_counts'][key] for key in COUNT_FIELDS),
                             label + ': global F count mismatch')
                grouped[(slot['method'], request['request_id'])].append(measured)
                result['raw_generations'] += 1
                result['native_blocks'] += len(record['metadata']['trace'])
            counts[str(seed)] = {method: {} for method in ARCHIVED_METHODS}
            for (method, request_id), group in grouped.items():
                _require(len(group) == 3 and not repeat_errors(group, method),
                         f'{seed}/{method}/{request_id}: archived repeats disagree')
                if method in ARCHIVED_METHODS:
                    counts[str(seed)][method][request_id] = {key: group[0][key] for key in COUNT_FIELDS}
            totals[str(seed)] = {method: sum(row['verification_calls'] for row in values.values())
                                 for method, values in counts[str(seed)].items()}
            _require(totals[str(seed)] == EXPECTED_CALL_TOTALS[str(seed)], f'{seed}: archived panel totals changed')
            checkpoints[str(seed)] = {method: expected[method] for method in ARCHIVED_METHODS}
        result.update(status='passed', source_commit=PR21_COMMIT, panel_binding=PANEL_BINDING,
                      requests=requests, canonicals=canonicals, counts=counts, checkpoints=checkpoints,
                      call_totals=totals, canonical_path=str(canonical_path),
                      canonical_identity=reader.files[str(canonical_path)],
                      count_trajectory_repeat=0, latency_reuse=False)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        result.update(status='failed', errors=[repr(error)])
        raise
    finally:
        result['files'] = reader.files
        if output_path is not None:
            write_json(Path(output_path), result)
    return result
