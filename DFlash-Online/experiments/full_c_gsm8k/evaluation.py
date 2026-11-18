"""Fixed three-arm full-trajectory endpoint evaluation using native record validation.

Each reused dataset has 144 activations, 288 warmups and 9,216 scored executions.
All timings follow completed training on one fixed physical A800.
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

METHODS = ("O58", "KPLUS58", "KPLUS_C58")
ORDER_PERMUTATIONS = tuple(sorted(permutations(METHODS)))
PANEL_SIZE, BLOCK_SIZE, REPEATS = 1024, 64, 3
ACTIVATIONS = 144
SCORED, WARMUPS, TOTAL = 9216, 288, 9504
DATASETS = ("gsm8k", "mathqa")
EVENTS = {"gsm8k": 58, "mathqa": 58}
SCOPES = {"gsm8k": "FULL_C_REUSED_PR25_GSM8K_ENDPOINT_PANEL",
          "mathqa": "FULL_C_REUSED_PR26_MATHQA_GSM_TRAINED_SERVING_TRANSFER"}

def dataset_contract(dataset):
    if dataset not in DATASETS:
        raise ValueError("Require a separate gsm8k or mathqa dataset")
    return dict(dataset=dataset, event=EVENTS[dataset], scope=SCOPES[dataset],
                session=f"full_c_gsm8k_{dataset}_event58")

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


def validate_references(requests, canonicals, *, reference_checkpoint):
    native.validate_references(requests, canonicals)
    if not reference_checkpoint or any(
        ref.get("reference_kind") != "native_greedy_target_verifier"
        or ref.get("reference_checkpoint") != str(reference_checkpoint)
        for ref in canonicals.values()
    ):
        raise ValueError("Every reference must identify its authenticated native-verifier producer")
    for reference in canonicals.values():
        payload = {k: v for k, v in reference.items() if k != "binding_digest"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if reference["binding_digest"] != digest:
            raise ValueError("Canonical reference binding digest differs from its contents")


def validate_inputs(requests, canonicals, warmups, warmup_canonicals, reference_checkpoints):
    if set(reference_checkpoints) != {'scored', 'warmup'}:
        raise ValueError('Require separate authenticated scored/warmup reference producers')
    validate_references(requests, canonicals, reference_checkpoint=reference_checkpoints['scored'])
    validate_references(warmups, warmup_canonicals, reference_checkpoint=reference_checkpoints['warmup'])
    if (len(warmups) != 2 or [r["panel_ordinal"] for r in warmups] != [0, 1]
            or len({r["source_id"] for r in requests + warmups}) != PANEL_SIZE + 2
            or len({r["request_id"] for r in requests + warmups}) != PANEL_SIZE + 2
            or len({tuple(r["prompt_ids"]) for r in requests + warmups}) != PANEL_SIZE + 2):
        raise ValueError("Require 1024 distinct scored inputs and two separate fixed warmups")


def validate_historical_counts(requests, counts):
    if set(counts) != {r['request_id'] for r in requests}:
        raise ValueError('Historical counts must cover the entire unchanged panel')
    for item in counts.values():
        if set(item) != {'O58','KPLUS58'} or any(
                type(row.get('verification_calls')) is not int or row['verification_calls'] < 0
                for row in item.values()):
            raise ValueError('Require exact O58/KPLUS58 historical verification counts')


def historical_errors(row, counts):
    if row['phase'] == 'scored' and row['method'] in ('O58','KPLUS58'):
        if row['verification_calls'] != counts[row['request_id']][row['method']]['verification_calls']:
            return ['historical_verification_count_mismatch']
    return []


def measure(record, request, checkpoint, canonical, slot, physical_gpu_uuid, *, dataset):
    contract = dataset_contract(dataset)
    row, errors = native.measure(record, request, checkpoint, canonical, slot,
                                 physical_gpu_uuid, session=contract["session"])
    row.update(dataset=dataset, scope=contract["scope"])
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


def _identity(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(bytes=Path(path).stat().st_size, sha256=digest)


def _resume_final(root, protocol, requests, canonicals, warmups, warmup_canonicals, historical_counts):
    """Revalidate saved native returns; only a whole completed activation can resume."""
    read = lambda path: json.loads(path.read_text())
    expected = {'PROTOCOL.json':protocol, 'REQUESTS.json':requests, 'CANONICAL.json':canonicals,
        'HISTORICAL_COUNTS.json':historical_counts,
        'WARMUPS.json':dict(requests=warmups, canonicals=warmup_canonicals)}
    if any(read(root/name) != value for name,value in expected.items()):
        raise ValueError('Resume source, endpoint identities or sealed inputs changed')
    progress = read(root/'STATUS.json')
    if (progress.get('status') != 'failed' or progress.get('invalid_generations') != 0
            or progress.get('unknown_generation_attempts') != 0 or progress.get('interrupted_attempt') is not None):
        raise ValueError('Resume requires a failed administrative boundary with no invalid or unknown work')
    def lines(name):
        path = root/name
        return [json.loads(row) for row in path.read_text().splitlines()] if path.exists() else []
    rows, activations = lines('MEASUREMENTS.jsonl'), lines('ACTIVATIONS.jsonl')
    per_activation = BLOCK_SIZE + 2
    if (len(rows) > TOTAL or len(rows) % per_activation or len(activations) != len(rows)//per_activation
            or progress['attempted_generations'] != len(rows) or progress['returned_generations'] != len(rows)
            or progress['completed_valid_generations'] != len(rows)
            or progress['publications'] != len(activations) or progress['logical_activations'] != len(activations)):
        raise ValueError('Cannot resume a partial activation or an unreturned publication')
    complete = len(activations)
    schedule = protocol['schedule']
    for saved, entry in zip(activations, schedule):
        if (any(saved.get(k) != v for k,v in entry.items()) or saved.get('errors')
                or publication_errors(saved,protocol['checkpoints'][entry['method']])):
            raise ValueError('Completed activation publication is invalid')
    accepted_boundaries = []
    for row in lines('BOUNDARY_CHECKS.jsonl'):
        actual = boundary_errors(row['check'],protocol['physical_gpu_uuid'])
        if row.get('errors') != actual:
            raise ValueError('Saved boundary validity changed')
        if not actual:
            accepted_boundaries.append(row['activation'])
    if accepted_boundaries != list(range(complete)):
        raise ValueError('Resume must begin before the next activation boundary')
    groups = defaultdict(list)
    expected_slots = list(execution_slots(schedule, requests, warmups, dataset=protocol['dataset']))[:len(rows)]
    for saved, (slot, request) in zip(rows, expected_slots):
        record = read(root/(label_for(slot)+'.json'))
        if read(Path(record['metadata']['native_trace_path'])) != record['metadata']['capture']:
            raise ValueError('Completed native trace changed')
        reference = (warmup_canonicals if slot['phase']=='warmup' else canonicals)[request['request_id']]
        row, errors = measure(record, request, protocol['checkpoints'][slot['method']], reference,
            slot, protocol['physical_gpu_uuid'], dataset=protocol['dataset'])
        previous = groups[slot['phase'],slot['method'],request['request_id']]
        errors.extend(_repeat_errors(previous,row))
        errors.extend(historical_errors(row,historical_counts))
        if (errors or saved.get('valid_measurement') is not True or saved.get('errors')
                or any(saved.get(k) != v for k,v in row.items())):
            raise ValueError('Completed generation failed native replay audit')
        previous.append(saved)
    for method in {r['method'] for r in rows}:
        weights = read(root/f'WEIGHTS_{method}.json')
        if weights['errors'] or weight_check_errors(weights['native_reply'],protocol['checkpoints'][method]):
            raise ValueError('Completed endpoint weight check is invalid')
    attempts = root/'ATTEMPTS'
    attempts.mkdir(exist_ok=True)
    index = len(list(attempts.glob('resume_*.json')))+1
    write_json(attempts/f'resume_{index:03d}.json', dict(status='authenticated_complete_activation_resume',
        execution_commit=protocol['execution_commit'], checkpoint_identities=protocol['checkpoint_identities'],
        prior_status=progress, prior_status_identity=_identity(root/'STATUS.json'),
        completed_activations=complete, completed_generations=len(rows),
        append_only_prefixes={name:_identity(root/name) for name in
            ('ACTIVATIONS.jsonl','MEASUREMENTS.jsonl','BOUNDARY_CHECKS.jsonl') if (root/name).exists()},
        no_generations_repeated=True))
    progress = {k:v for k,v in progress.items() if k not in ('error','interrupted_attempt')}
    progress.update(status='running', resume_attempts=index)
    return rows, groups, progress, complete, index


def evaluate_final(backend, requests, checkpoints, root, *, dataset, canonicals,
                   warmup_requests, warmup_canonicals, schedule, reference_checkpoints, historical_counts,
                   physical_gpu_uuid, boundary_check, execution_commit, checkpoint_identities, resume=False):
    """Preserve each native return before checking it; stop without retries."""
    contract = dataset_contract(dataset)
    if any(row.get("dataset") != dataset for row in requests + warmup_requests):
        raise ValueError("Evaluation inputs must belong to the declared dataset")
    if schedule != evaluation_schedule(requests):
        raise ValueError("The sealed three-arm schedule changed")
    checkpoints = {m: str(path) for m, path in checkpoints.items()}
    if (set(checkpoints) != set(METHODS) or len(set(checkpoints.values())) != 3):
        raise ValueError("Require three distinct O58/KPLUS58/KPLUS_C58 endpoints")
    if (set(checkpoint_identities) != set(METHODS) or any(
            set(v) != {'bytes','sha256'} or type(v['bytes']) is not int or v['bytes'] <= 0
            or len(v['sha256']) != 64 or any(c not in '0123456789abcdef' for c in v['sha256'])
            for v in checkpoint_identities.values())):
        raise ValueError('Require authenticated complete checkpoint file identities')
    validate_inputs(requests, canonicals, warmup_requests, warmup_canonicals, reference_checkpoints)
    validate_historical_counts(requests, historical_counts)
    if (not physical_gpu_uuid or boundary_check is None or len(execution_commit) != 40
            or any(c not in '0123456789abcdef' for c in execution_commit)):
        raise ValueError("Require the fixed physical GPU, boundary checks and execution commit")
    root = Path(root)
    protocol = dict(schedule=schedule, checkpoints=checkpoints,
        methods=list(METHODS), **contract, capture=False,
        checkpoint_identities=checkpoint_identities,
        reference_checkpoints=reference_checkpoints, historical_count_gate=True,
        physical_gpu_uuid=physical_gpu_uuid, execution_commit=execution_commit,
        planned_scored=SCORED, planned_warmups=WARMUPS)
    rows, groups, checked = [], defaultdict(list), set()
    progress = dict(status='running', planned_generations=TOTAL, attempted_generations=0,
        returned_generations=0, publications=0, logical_activations=0, activation_s=0.,
        weight_validation_s=0., optimizer_steps=0, feedback_collections=0,
        teacher_preparations=0, automatic_retry=False)
    completed_activations, resume_index = 0, 0
    if resume:
        rows, groups, progress, completed_activations, resume_index = _resume_final(root, protocol,
            requests, canonicals, warmup_requests, warmup_canonicals, historical_counts)
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root/'PROTOCOL.json', protocol)
        write_json(root/'REQUESTS.json', requests)
        write_json(root/'CANONICAL.json', canonicals)
        write_json(root/'HISTORICAL_COUNTS.json', historical_counts)
        write_json(root/'WARMUPS.json', dict(requests=warmup_requests, canonicals=warmup_canonicals))
    previous_wall_s = progress.get('evaluation_wall_s',0.)
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
            if slot['activation'] < completed_activations:
                continue
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
                    if resume_index and (root/name).exists():
                        name = f'WEIGHTS_{method}_resume_{resume_index:03d}.json'
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
            errors.extend(historical_errors(row, historical_counts))
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
        progress['evaluation_wall_s'] = previous_wall_s+time.perf_counter()-started
        persist()
    return progress


def evaluate_qualification(backend, plan, checkpoints, root, *, boundary_check=None):
    """Exactly two sealed panel sentinels × three checkpoints × two repeats per dataset.

    This is a correctness check, without extra warmups or latency inference. The
    caller charges every backend generation to its cumulative qualification ledger.
    """
    from .runtime import boundary, read
    from experiments.kplus_full_gsm8k.preflight import timing_barrier
    checkpoints = {method: str(path) for method, path in checkpoints.items()}
    if set(checkpoints) != set(METHODS) or len(set(checkpoints.values())) != 3:
        raise ValueError('Require both retained controls and one disposable active-C export')
    gpu = plan['devices']['A']
    if boundary_check is None:
        def boundary_check():
            resource = boundary(plan)
            quiet = timing_barrier(backend.worker.pid)
            return dict(status='passed', physical_gpu_uuid=gpu, both_cards_checked=True,
                other_gpu_idle=True, host_quiet=True, storage_admitted=True,
                time_admitted=True, resources=resource, quiet=quiet)
    data = {}
    for dataset in DATASETS:
        spec = plan['datasets'][dataset]
        requests = read(spec['requests'])['requests']
        if evaluation_schedule(requests) != read(spec['schedule']):
            raise ValueError('Qualification sentinel panel order differs from its sealed schedule')
        canonicals, counts = read(spec['canonicals']), read(spec['historical_counts'])
        validate_references(requests, canonicals, reference_checkpoint=spec['reference_checkpoints']['scored'])
        validate_historical_counts(requests, counts)
        data[dataset] = (requests[:2], canonicals, counts)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'PROTOCOL.json', dict(execution_commit=plan['execution_commit'],
        checkpoints=checkpoints, physical_gpu_uuid=gpu, planned_generations=24,
        warmups=0, latency_inference=False, sentinel_selection='first two original panel requests',
        sentinels={d:[r['request_id'] for r in values[0]] for d,values in data.items()}))
    progress = dict(status='running', planned_generations=24, attempted_generations=0,
        returned_generations=0, invalid_generations=0, publications=0, activation_s=0.,
        warmup_generations=0, automatic_retry=False)
    rows, groups, checked = [], defaultdict(list), set()
    pending, started = None, time.perf_counter()
    def persist():
        progress.update(completed_valid_generations=sum(r['valid_measurement'] for r in rows),
            invalid_generations=sum(not r['valid_measurement'] for r in rows),
            unknown_generation_attempts=int(pending is not None), interrupted_attempt=pending,
            unattempted_generations=24-progress['attempted_generations'])
        write_json(root/'STATUS.json', progress)
    try:
        for dataset, (requests, canonicals, counts) in data.items():
            dsroot = root/dataset
            dsroot.mkdir()
            contract = dataset_contract(dataset)
            for repeat in range(2):
                for position, method in enumerate(METHODS):
                    check = boundary_check()
                    errors = boundary_errors(check, gpu)
                    append_json(root/'BOUNDARY_CHECKS.jsonl', dict(dataset=dataset, repeat=repeat,
                        method=method, check=check, errors=errors))
                    if errors:
                        raise RuntimeError('Qualification boundary failed: '+repr(errors))
                    receipt = backend.publish(checkpoints[method])
                    errors = publication_errors(receipt, checkpoints[method])
                    progress['publications'] += 1
                    progress['activation_s'] += receipt['activation_s']
                    append_json(root/'ACTIVATIONS.jsonl', dict(dataset=dataset, repeat=repeat,
                        method=method, **receipt, errors=errors))
                    if errors:
                        raise RuntimeError('Qualification publication failed: '+repr(errors))
                    if method not in checked:
                        reply = backend.check_weights(checkpoints[method])
                        errors = weight_check_errors(reply, checkpoints[method])
                        write_json(root/f'WEIGHTS_{method}.json', dict(native_reply=reply, errors=errors))
                        if errors:
                            raise RuntimeError('Qualification weight mismatch: '+repr(errors))
                        checked.add(method)
                    for request in requests:
                        slot = dict(phase='scored', method=method, repeat=repeat, event=58, dataset=dataset,
                            panel_ordinal=request['panel_ordinal'], request_id=request['request_id'],
                            activation=progress['publications']-1, block_ordinal=0, position=position)
                        label = label_for(slot)
                        path = dsroot/(label+'.json')
                        pending = dict(dataset=dataset, label=label)
                        progress['attempted_generations'] += 1
                        persist()
                        record = backend.generate(request['prompt_ids'], max_new_tokens=2048,
                            request_id=label, session_id=contract['session'], request_index=request['panel_ordinal'],
                            purpose='evaluation', save_to=path)
                        progress['returned_generations'] += 1
                        with _temporary_file(path) as temporary:
                            temporary.write_text(json.dumps(record, allow_nan=True)+'\n')
                        row, errors = measure(record, request, checkpoints[method], canonicals[request['request_id']],
                            slot, gpu, dataset=dataset)
                        previous = groups[dataset, method, request['request_id']]
                        errors.extend(_repeat_errors(previous, row))
                        errors.extend(historical_errors(row, counts))
                        row.update(errors=sorted(set(errors)), valid_measurement=not errors,
                            validation_status='invalid' if errors else 'valid', record_path=str(path))
                        rows.append(row)
                        previous.append(row)
                        pending = None
                        append_json(root/'MEASUREMENTS.jsonl', row)
                        write_json(dsroot/(label+'_VALIDATION.json'), dict(status=row['validation_status'], errors=row['errors']))
                        persist()
                        if errors:
                            raise RuntimeError('Qualification native correctness failure: '+repr(errors))
        progress['status'] = 'passed'
    except BaseException as error:
        progress.update(status='failed', error=repr(error))
        raise
    finally:
        progress['qualification_wall_s'] = time.perf_counter()-started
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
    validate_inputs(requests, canonicals, warmups['requests'], warmups['canonicals'], protocol['reference_checkpoints'])
    historical_counts = read(root/'HISTORICAL_COUNTS.json')
    validate_historical_counts(requests, historical_counts)
    errors, rebuilt, groups = [], [], defaultdict(list)
    if any(row.get('dataset') != dataset for row in requests + warmups['requests']):
        errors.append('dataset_identity')
    expected_protocol = dict(schedule=planned, methods=list(METHODS), **contract, capture=False, planned_scored=SCORED, planned_warmups=WARMUPS,
        historical_count_gate=True)
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
        for path in root.glob(f'WEIGHTS_{method}_resume_*.json'):
            saved = read(path)
            if saved['errors'] or weight_check_errors(saved['native_reply'],protocol['checkpoints'][method]):
                errors.append('resumed_weight_identity_'+method)
    covered_boundary_bytes = 0
    attempts = sorted((root/'ATTEMPTS').glob('resume_*.json')) if (root/'ATTEMPTS').exists() else []
    for attempt_path in attempts:
        attempt = read(attempt_path)
        if (attempt['status'] != 'authenticated_complete_activation_resume'
                or attempt['execution_commit'] != protocol['execution_commit']
                or attempt['checkpoint_identities'] != protocol['checkpoint_identities']
                or attempt['no_generations_repeated'] is not True
                or attempt['completed_generations'] != attempt['completed_activations']*(BLOCK_SIZE+2)):
            errors.append('resume_identity')
        for name, identity in attempt['append_only_prefixes'].items():
            with (root/name).open('rb') as stream:
                prefix = stream.read(identity['bytes'])
            if len(prefix) != identity['bytes'] or hashlib.sha256(prefix).hexdigest() != identity['sha256']:
                errors.append('resumed_prefix_changed:'+name)
            if name == 'BOUNDARY_CHECKS.jsonl':
                covered_boundary_bytes = max(covered_boundary_bytes,identity['bytes'])
    if status.get('resume_attempts',0) != len(attempts):
        errors.append('resume_attempt_count')
    accepted_boundaries, offset = [], 0
    for rawline in (root/'BOUNDARY_CHECKS.jsonl').read_bytes().splitlines(keepends=True):
        offset += len(rawline)
        row = json.loads(rawline)
        actual = boundary_errors(row['check'], protocol['physical_gpu_uuid'])
        if row['errors'] != actual:
            errors.append('saved_boundary_validity_changed')
        if actual:
            if offset > covered_boundary_bytes:
                errors.extend(actual)
        else:
            accepted_boundaries.append(row['activation'])
    if accepted_boundaries != [*range(ACTIVATIONS),'after_final']:
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
        invalid.extend(historical_errors(row, historical_counts))
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
