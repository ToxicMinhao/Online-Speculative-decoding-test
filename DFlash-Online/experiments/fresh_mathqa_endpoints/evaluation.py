"""Fresh MathQA comparison of three frozen event-58 serving exports.

Adapted narrowly from PR54 experiments/full_c_gsm8k/evaluation.py at execution
b4459820b3822c3d760bb3fd880e4d139fa2bbc2. Native measurement, publication,
weight and raw-trace validation remain imported from their qualified modules.
Fresh sealed canonical provenance replaces reused-panel count dependencies.
One execution only: 144 activations, 288 warmups, 9,216 scored generations.
"""
from collections import defaultdict
from itertools import permutations
import hashlib
import json
from pathlib import Path
import time

from online_feedback.storage import _temporary_file, write_json
from experiments.fresh_confirmation.evaluate import append_json
from experiments.fresh_confirmation.protocol import publication_errors, repeat_errors, weight_check_errors
from experiments.persistent_prefix_withholding import evaluation as native

METHODS = ("O58", "AREPO58", "KPLUS_C58")
ORDER_PERMUTATIONS = tuple(sorted(permutations(METHODS)))
PANEL_SIZE, BLOCK_SIZE, REPEATS = 1024, 64, 3
ACTIVATIONS = 144
SCORED, WARMUPS, TOTAL = 9216, 288, 9504
DATASETS = ("mathqa",)
FRESH_PROVENANCE_FIELDS = ("producer_weight_identity", "target_identity", "tokenizer_identity", "runtime_identity")


def dataset_contract(dataset):
    if dataset != "mathqa":
        raise ValueError("This wrapper requires the sealed fresh MathQA panel")
    return dict(dataset=dataset, event=58, scope="FRESH_MATHQA_C_AREPO_OSD_ENDPOINT",
                session="fresh_mathqa_c_arepo_osd_event58")


label_for = native.label_for
boundary_errors = native.boundary_errors



def evaluation_schedule(requests):
    if (len(requests) != PANEL_SIZE
            or [r["panel_ordinal"] for r in requests] != list(range(PANEL_SIZE))):
        raise ValueError("Require the sealed 1024-request panel in its original order")
    schedule = []
    for repeat in range(REPEATS):
        for block in range(16):
            order = ORDER_PERMUTATIONS[(16 * repeat + block) % 6]
            for position, method in enumerate(order):
                schedule.append(dict(activation=len(schedule), repeat=repeat,
                    block_ordinal=block, position=position, method=method,
                    request_ordinals=list(range(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE))))
    return schedule


def validate_references(requests, canonicals, *, reference_checkpoint,
                        producer_weight_identity=None, required_provenance=False,
                        canonical_provenance=None):
    """Bind target-verifier outputs and their declared producer before scoring."""
    native.validate_references(requests, canonicals)
    if not reference_checkpoint:
        raise ValueError("Require an authenticated native-verifier producer")
    observed_provenance = None
    for request in requests:
        reference = canonicals[request["request_id"]]
        if (reference.get("reference_kind") != "native_greedy_target_verifier"
                or reference.get("reference_checkpoint") != str(reference_checkpoint)):
            raise ValueError("Canonical reference producer differs")
        for key in ("request_id", "source_id", "stop_token_ids", "max_new_tokens"):
            if reference.get(key) != request[key]:
                raise ValueError("Canonical request binding differs: " + key)
        payload = {k: v for k, v in reference.items() if k != "binding_digest"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        if reference["binding_digest"] != digest:
            raise ValueError("Canonical reference binding digest differs from its contents")
        if required_provenance:
            if (reference.get("native_checkpoint") != str(reference_checkpoint)
                    or any(not isinstance(reference.get(k), dict) or not reference[k]
                           for k in FRESH_PROVENANCE_FIELDS)):
                raise ValueError("Fresh canonical target/tokenizer/runtime/producer identity is incomplete")
            provenance = {k:reference[k] for k in FRESH_PROVENANCE_FIELDS}
            if observed_provenance is not None and provenance != observed_provenance:
                raise ValueError("Fresh canonical producer provenance changed within the panel")
            observed_provenance = provenance
            if (producer_weight_identity is not None
                    and reference['producer_weight_identity'] != producer_weight_identity):
                raise ValueError("Fresh canonical producer checkpoint identity differs")
            if canonical_provenance is not None and any(
                    reference[k] != canonical_provenance[k] for k in
                    ('target_identity', 'tokenizer_identity', 'runtime_identity')):
                raise ValueError("Fresh canonical target/tokenizer/runtime identity differs from the sealed plan")


def validate_checkpoint_identities(checkpoints, checkpoint_identities):
    if set(checkpoints) != set(METHODS) or len(set(checkpoints.values())) != 3:
        raise ValueError("Require three distinct O58/AREPO58/KPLUS_C58 endpoints")
    if (set(checkpoint_identities) != set(METHODS) or any(
            set(v) != {'bytes', 'sha256'} or type(v['bytes']) is not int or v['bytes'] <= 0
            or not isinstance(v['sha256'], str) or len(v['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in v['sha256'])
            for v in checkpoint_identities.values())):
        raise ValueError('Require authenticated complete checkpoint file identities')
    if len({v['sha256'] for v in checkpoint_identities.values()}) != 3:
        raise ValueError('The three required endpoint weights must have distinct identities')


def validate_inputs(requests, canonicals, warmups, warmup_canonicals, reference_checkpoints,
                    *, checkpoint_identities=None, checkpoints=None, canonical_provenance=None):
    if checkpoint_identities is not None and (
            not isinstance(canonical_provenance, dict)
            or any(not isinstance(canonical_provenance.get(k), dict) or not canonical_provenance[k]
                   for k in ('target_identity', 'tokenizer_identity', 'runtime_identity'))):
        raise ValueError('Require authenticated target/tokenizer/runtime canonical provenance')
    if set(reference_checkpoints) != {'scored', 'warmup'}:
        raise ValueError('Require separate authenticated scored/warmup reference producers')
    if checkpoints is not None and reference_checkpoints['scored'] != checkpoints['O58']:
        raise ValueError('Fresh canonicals must be produced by the authenticated O58 checkpoint')
    validate_references(requests, canonicals, reference_checkpoint=reference_checkpoints['scored'],
        producer_weight_identity=None if checkpoint_identities is None else checkpoint_identities['O58'],
        required_provenance=True, canonical_provenance=canonical_provenance)
    validate_references(warmups, warmup_canonicals, reference_checkpoint=reference_checkpoints['warmup'])
    if (len(requests) != PANEL_SIZE or len(warmups) != 2
            or [r['panel_ordinal'] for r in requests] != list(range(PANEL_SIZE))
            or [r["panel_ordinal"] for r in warmups] != [0, 1]
            or len({r["source_id"] for r in requests + warmups}) != PANEL_SIZE + 2
            or len({r["request_id"] for r in requests + warmups}) != PANEL_SIZE + 2
            or len({tuple(r["prompt_ids"]) for r in requests + warmups}) != PANEL_SIZE + 2):
        raise ValueError("Require 1024 distinct scored inputs and two separate fixed warmups")



def measure(record, request, checkpoint, canonical, slot, physical_gpu_uuid, *, dataset):
    contract = dataset_contract(dataset)
    row, errors = native.measure(record, request, checkpoint, canonical, slot,
                                 physical_gpu_uuid, session=contract["session"])
    row.update(dataset=dataset, scope=contract["scope"],
               canonical_binding_digest=canonical["binding_digest"])
    return row, errors


def _repeat_errors(previous, row):
    errors = repeat_errors([*previous, row], row["method"])
    if previous:
        for key in ("native_accepted_draft_tokens", "native_acceptance_histogram"):
            if row.get(key) != previous[0].get(key):
                errors.append("repeat_" + key + "_mismatch")
    return errors


def execution_slots(schedule, requests, warmups, *, dataset):
    contract = dataset_contract(dataset)
    for entry in schedule:
        for phase, request in [*(('warmup', r) for r in warmups),
                               *(('scored', requests[i]) for i in entry['request_ordinals'])]:
            slot = {k: v for k, v in entry.items() if k != "request_ordinals"}
            slot.update(phase=phase, event=contract["event"], dataset=dataset, panel_ordinal=request["panel_ordinal"],
                        request_id=request["request_id"])
            yield slot, request


def evaluate_final(backend, requests, checkpoints, root, *, dataset, canonicals,
                   warmup_requests, warmup_canonicals, schedule, reference_checkpoints,
                   physical_gpu_uuid, boundary_check, execution_commit, checkpoint_identities, canonical_provenance=None):
    """Preserve each native return before checking it; stop without retries."""
    contract = dataset_contract(dataset)
    if any(row.get("dataset") != dataset for row in requests + warmup_requests):
        raise ValueError("Evaluation inputs must belong to the declared dataset")
    if schedule != evaluation_schedule(requests):
        raise ValueError("The sealed three-arm schedule changed")
    checkpoints = {m: str(path) for m, path in checkpoints.items()}
    validate_checkpoint_identities(checkpoints, checkpoint_identities)
    validate_inputs(requests, canonicals, warmup_requests, warmup_canonicals, reference_checkpoints,
        checkpoint_identities=checkpoint_identities, checkpoints=checkpoints,
        canonical_provenance=canonical_provenance)
    if (not physical_gpu_uuid or boundary_check is None or len(execution_commit) != 40
            or any(c not in '0123456789abcdef' for c in execution_commit)):
        raise ValueError("Require the fixed physical GPU, boundary checks and execution commit")
    root = Path(root)
    protocol = dict(schedule=schedule, checkpoints=checkpoints,
        methods=list(METHODS), **contract, capture=False,
        checkpoint_identities=checkpoint_identities,
        reference_checkpoints=reference_checkpoints, fresh_canonical_gate=True,
        canonical_provenance=canonical_provenance, canonical_producer="O58",
        physical_gpu_uuid=physical_gpu_uuid, execution_commit=execution_commit,
        planned_scored=SCORED, planned_warmups=WARMUPS)
    rows, groups, checked = [], defaultdict(list), set()
    progress = dict(status='running', planned_generations=TOTAL, attempted_generations=0,
        returned_generations=0, publications=0, logical_activations=0, activation_s=0.,
        weight_validation_s=0., optimizer_steps=0, feedback_collections=0,
        teacher_preparations=0, automatic_retry=False)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'PROTOCOL.json', protocol)
    write_json(root/'REQUESTS.json', requests)
    write_json(root/'CANONICAL.json', canonicals)
    write_json(root/'WARMUPS.json', dict(requests=warmup_requests, canonicals=warmup_canonicals))
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
        for slot, request in execution_slots(schedule, requests, warmup_requests, dataset=dataset):
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
                    name = f'WEIGHTS_{method}.json'
                    write_json(root/name, dict(native_reply=reply, errors=errors, validation_s=elapsed))
                    if errors:
                        raise RuntimeError('Published weights differ: ' + repr(errors))
                    checked.add(method)
            label = label_for(slot)
            path = root/(label+'.json')
            pending = dict(slot, label=label)
            progress['attempted_generations'] += 1
            persist()
            record = backend.generate(request['prompt_ids'], max_new_tokens=2048,
                request_id=label, session_id=contract["session"], request_index=request['panel_ordinal'],
                purpose='evaluation', save_to=path)
            progress['returned_generations'] += 1
            with _temporary_file(path) as temporary:
                temporary.write_text(json.dumps(record, allow_nan=True)+'\n')
            canonical = (warmup_canonicals if slot['phase']=='warmup' else canonicals)[request['request_id']]
            row, errors = measure(record, request, checkpoint, canonical, slot, physical_gpu_uuid, dataset=dataset)
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
    dataset = protocol['dataset']
    contract = dataset_contract(dataset)
    planned = evaluation_schedule(requests)
    validate_checkpoint_identities(protocol['checkpoints'], protocol['checkpoint_identities'])
    validate_inputs(requests, canonicals, warmups['requests'], warmups['canonicals'],
        protocol['reference_checkpoints'], checkpoint_identities=protocol['checkpoint_identities'],
        checkpoints=protocol['checkpoints'], canonical_provenance=protocol.get('canonical_provenance'))
    errors, rebuilt, groups = [], [], defaultdict(list)
    if any(row.get('dataset') != dataset for row in requests + warmups['requests']):
        errors.append('dataset_identity')
    expected_protocol = dict(schedule=planned, methods=list(METHODS), **contract, capture=False, planned_scored=SCORED, planned_warmups=WARMUPS,
        fresh_canonical_gate=True, canonical_producer="O58")
    errors.extend('protocol_'+k for k,v in expected_protocol.items() if protocol.get(k)!=v)
    commit = protocol.get('execution_commit', '')
    if (not protocol.get('physical_gpu_uuid') or len(commit) != 40
            or any(c not in '0123456789abcdef' for c in commit)):
        errors.append('execution_identity')
    if (set(protocol['checkpoints'])!=set(METHODS)
            or len(set(protocol['checkpoints'].values()))!=3):
        errors.append('endpoint_identity')
    status = read(root/'STATUS.json')
    expected_status = dict(status='completed', planned_generations=TOTAL, attempted_generations=TOTAL,
        returned_generations=TOTAL, completed_valid_generations=TOTAL, invalid_generations=0,
        scored_generations=SCORED, warmup_generations=WARMUPS, publications=ACTIVATIONS,
        logical_activations=ACTIVATIONS, unknown_generation_attempts=0, unattempted_generations=0,
        optimizer_steps=0, feedback_collections=0, teacher_preparations=0, automatic_retry=False)
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
    accepted_boundaries = []
    for row in lines(root/'BOUNDARY_CHECKS.jsonl'):
        actual = boundary_errors(row['check'], protocol['physical_gpu_uuid'])
        if row['errors'] != actual:
            errors.append('saved_boundary_validity_changed')
        errors.extend(actual)
        accepted_boundaries.append(row['activation'])
    if accepted_boundaries != [*range(ACTIVATIONS), 'after_final']:
        errors.append('boundary_schedule')
    saved_rows = lines(root/'MEASUREMENTS.jsonl')
    expected = list(execution_slots(planned, requests, warmups['requests'], dataset=dataset))
    if len(saved_rows)!=TOTAL:
        errors.append('measurement_count')
    for saved, (slot, request) in zip(saved_rows, expected):
        label = label_for(slot)
        record = read(root/(label+'.json'))
        if read(native_trace_resolver(record['metadata']['native_trace_path'])) != record['metadata']['capture']:
            errors.append(label+':native_capture_changed')
        canonical = (warmups['canonicals'] if slot['phase']=='warmup' else canonicals)[request['request_id']]
        row, invalid = measure(record, request, protocol['checkpoints'][slot['method']],
                               canonical, slot, protocol['physical_gpu_uuid'], dataset=dataset)
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
        rows=rebuilt, dataset=dataset, execution_commit=protocol['execution_commit'], extra_model_work=0)


def audit_raw_records(root, *, native_trace_resolver=None):
    """Reapply native validity to retained returns, including their separate traces."""
    try:
        return _audit_raw_records(Path(root), native_trace_resolver or Path)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        return dict(status='invalid_or_incomplete', errors=['evidence_read_or_schema:'+repr(error)],
                    rows=[], extra_model_work=0)
