"""Eight fixed shared-feedback events, followed by one balanced endpoint.

Serving, immutable packet preparation, pass allocation and native training are
separate phases. No scientific operation is retried after an exception.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from online_feedback.storage import write_json, _temporary_file
from experiments.fresh_confirmation.protocol import record_errors, measurement_errors
from experiments.o_half_matched_lr.archive import full_trace_errors
from experiments.ok_coding.campaign import WorkLedger, read_work, require_complete_work
from experiments.ok_events_5_8.campaign import publish_checked
from experiments.ok_events_5_8.storage import identity
from experiments.osd_persistent.run import score_record
from experiments.kw_v1.evaluate import safe_scalars, append_json
from .contract import ARMS, EVENTS, BUDGET, serving_order
from .selection import ReplayRecord, ReservoirMemory, prompt_descriptor, build_event_plan
from .storage import admit, check_growth, require_write_space, retire_owned, ROW_BYTES

SERVING_LIMITS = dict(pre_update_generations=3072, capture_parity_generations=4,
                      terminal_preparation_forwards=1024)


def read(path):
    return json.loads(Path(path).read_text())


def save_rows(path, rows):
    with _temporary_file(path) as temporary:
        temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def resource_check(root, plan, progress):
    receipt = check_growth(root, plan['storage'])
    append_json(root / 'campaign_001/RESOURCE_CHECKS.jsonl', dict(
        phase=progress.get('phase'), event=progress.get('current_event'), **receipt))
    progress['costs']['resource_check_s'] = progress['costs'].get('resource_check_s', 0.) + receipt['resource_check_wall_s']
    return receipt


def check_stream_record(record, request, checkpoint, *, capture, canonical=None):
    """Apply the inherited native validator, with the declared capture contract."""
    errors = record_errors(record, request, str(checkpoint), canonical)
    if capture:
        errors = [error for error in errors if error not in ('feedback_capture_or_purpose', 'capture_contract')]
        evidence = record['metadata']['capture']
        if (record['purpose'] != 'serving' or not record['metadata']['feedback_capture_in_timed_request']
                or evidence['collect_features'] is not True or evidence['native_capture_validated'] is not True
                or evidence['feature_bytes'] <= 0):
            errors.append('required_native_capture')
    errors.extend(full_trace_errors(record))
    if (record['record_id'] != request['request_id'] or record['request_index'] != request['request_index']
            or record['session_id'] != f"event_{request['event']:03d}"):
        errors.append('original_request_identity')
    row = score_record(record, request_id=request['request_id'], repeat=0, method='K' if capture else 'evaluation')
    row.update(finish_reason=record['finish_reason'], cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'])
    errors.extend(measurement_errors(row, request, canonical))
    return sorted(set(errors))


def compare_stream(records, event):
    """Same canonical outputs; neutral events additionally require equal counts."""
    errors = []
    for request_id, methods in records.items():
        if set(methods) != set(ARMS):
            errors.append((request_id, 'incomplete_methods'))
            continue
        reference = methods['K']
        for arm in ('U', 'R'):
            if any(methods[arm][key] != reference[key] for key in ('prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids')):
                errors.append((request_id, arm + '_canonical_mismatch'))
            if event <= 2 and methods[arm]['metadata']['verification_calls'] != reference['metadata']['verification_calls']:
                errors.append((request_id, arm + '_neutral_count_mismatch'))
    return errors


def execute_worker(root, plan, arm, operation, event=0):
    """Fresh Python owns one trainer; subprocess exit precedes later serving."""
    folder = root / 'campaign_001/arms' / arm
    folder.mkdir(parents=True, exist_ok=True)
    label = 'initial' if operation == 'initialize' else f'event_{event:03d}'
    command = [plan['training_python'], '-u', '-m', 'experiments.k_workload_replay_v1.worker',
               '--root', str(root), '--arm', arm, '--operation', operation, '--event', str(event)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan['training_gpu']))
    started = time.perf_counter()
    with (folder / (label + '.log')).open('x') as stream:
        process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        receipt = dict(command=command, pid=process.pid, start_epoch=time.time(), automatic_retry=False)
        write_json(folder / (label + '_PROCESS.json'), receipt)
        try:
            returncode = process.wait(timeout=4 * 3600)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            receipt.update(returncode=process.poll(), wall_s=time.perf_counter() - started)
            write_json(folder / (label + '_PROCESS.json'), receipt)
    if returncode:
        raise RuntimeError(f'{arm} {label} worker stopped; preserve its log and partial work')
    output = folder / ('INITIAL_WORKER.json' if operation == 'initialize'
                       else f'events/event_{event:03d}/WORKER.json')
    result = read(output)
    if result['status'] != 'passed':
        raise RuntimeError('Worker exit lacks a verified scientific boundary')
    return result


def prepare_packet(backend, ledger, record, request, event_root, root, plan):
    """Validate a durable packet before retiring its redundant native tensors."""
    import torch
    from online_feedback.training import validate_feature, validate_target_binding
    path = event_root / 'features' / (request['request_id'] + '.pt')
    if len(record['response_ids']) < 2:
        raise ValueError('Fixed response has fewer than two tokens; no terminal forward or replacement is allowed')
    length = len(record['prompt_ids']) + len(record['response_ids'])
    require_write_space(root, plan['storage'], length * ROW_BYTES * 2)
    prepared = ledger.call('terminal_preparation_forwards', backend.prepare_feedback, record, path)
    if prepared['status'] != 'prepared':
        raise ValueError('This fixed response is ineligible; replacement or truncation is forbidden')
    started = time.perf_counter()
    packet = torch.load(path, map_location='cpu', weights_only=False)
    metadata = validate_feature(packet, max_length=2272, hidden_width=20480)
    validate_target_binding(packet)
    if (packet['record_id'] != request['request_id'] or packet['prompt_length'] != len(request['prompt_ids'])
            or packet['input_ids'][0].tolist() != record['prompt_ids'] + record['response_ids']
            or packet['target_last_hidden_states'].shape[-1] != 4096
            or packet['hidden_states'].dtype != torch.bfloat16
            or packet['target_last_hidden_states'].dtype != torch.bfloat16):
        raise ValueError('Prepared feature identity, precision or canonical conditioning differs')
    descriptor_started = time.perf_counter()
    descriptor, descriptor_receipt = prompt_descriptor(packet['target_last_hidden_states'], packet['prompt_length'])
    descriptor_s = time.perf_counter() - descriptor_started
    del packet
    content = identity(path)
    binding = dict(record_id=request['request_id'], global_ordinal=request['global_ordinal'],
                   path=str(path), sha256=content['sha256'])
    replay = ReplayRecord(request['request_id'], request['global_ordinal'], request['event'], descriptor)
    packet_receipt = dict(status='passed', binding=binding, metadata=metadata, identity=content,
        descriptor=asdict(replay), descriptor_checks=descriptor_receipt, descriptor_s=descriptor_s,
        validation_io_s=time.perf_counter()-started-descriptor_s, preparation=prepared)
    receipt_path = event_root / 'packets' / (request['request_id'] + '.json')
    write_json(receipt_path, packet_receipt)
    for kind, raw in (('native', record['metadata']['capture']['feature_path']),
                      ('terminal', prepared['terminal_feature']['tensor_path'])):
        retire_owned(raw, root=root,
            receipt_path=root / 'campaign_001/RETIREMENTS' / f"{request['request_id']}_{kind}.json",
            reason='Validated immutable prepared packet and compact native conditioning evidence retained',
            retained_evidence=[path, receipt_path, path.with_suffix('.json')])
    return binding, replay, packet_receipt


def serve_event(backend, ledger, requests, checkpoints, event_root, root, plan, progress):
    event = requests[0]['event']
    records, rows, bindings, descriptors, preparations, parity = {}, [], {}, [], [], []
    event_root.mkdir(parents=True, exist_ok=False)
    try:
        for arm in serving_order(event):
            publication = publish_checked(backend, checkpoints[arm], event_root / f'PUBLICATION_{arm}.json')
            progress['costs']['activation_s'] += publication['publication_and_validation_s']
            for index, request in enumerate(requests):
                resource_check(root, plan, progress)
                require_write_space(root, plan['storage'], (len(request['prompt_ids']) + 2048) * ROW_BYTES * 3)
                path = event_root / 'responses' / arm / (request['request_id'] + '.json')
                record = ledger.call('pre_update_generations', backend.generate, request['prompt_ids'],
                    max_new_tokens=2048, request_id=request['request_id'], session_id=f'event_{event:03d}',
                    request_index=request['request_index'], purpose='serving' if arm == 'K' else 'evaluation', save_to=path)
                group = records.setdefault(request['request_id'], {})
                reference = next(iter(group.values()), None)
                try:
                    errors = check_stream_record(record, request, checkpoints[arm], capture=arm == 'K', canonical=reference)
                except (KeyError, TypeError, ValueError, IndexError) as error:
                    invalid = dict(event=event, method=arm, request_id=request['request_id'],
                        domain=request['domain'], capture=arm == 'K', record_path=str(path),
                        valid_measurement=False, errors=['malformed_native_record: ' + repr(error)])
                    rows.append(invalid)
                    raise ValueError(invalid['errors']) from error
                row = score_record(record, request_id=request['request_id'], repeat=0, method=arm)
                row.update(event=event, domain=request['domain'], global_ordinal=request['global_ordinal'],
                    capture=arm == 'K', record_path=str(path), finish_reason=record['finish_reason'],
                    cache_maintenance_s=record['metadata']['capture']['cache_maintenance_s'],
                    capture_host_s=record['metadata']['capture'].get('capture_host_s', 0.),
                    feature_persistence_s=record['metadata']['capture'].get('feature_persistence_s', 0.),
                    valid_measurement=not errors, errors=errors)
                rows.append(safe_scalars(row))
                group[arm] = record
                save_rows(event_root / 'MEASUREMENTS.jsonl', rows)
                if errors:
                    raise ValueError('Pre-update native validity failed: ' + repr(errors))
                if arm == 'K':
                    binding, descriptor, receipt = prepare_packet(backend, ledger, record, request, event_root, root, plan)
                    bindings[request['request_id']] = binding
                    descriptors.append(descriptor)
                    preparations.append(receipt)
                    for key in ('teacher_preparation_s', 'feature_persistence_s'):
                        progress['costs'][key] += receipt['preparation'][key]
                    progress['costs']['descriptor_s'] += receipt['descriptor_s']
                    if event in (1, 3) and index < 2:
                        parity_path = event_root / 'parity' / (request['request_id'] + '.json')
                        control = ledger.call('capture_parity_generations', backend.generate, request['prompt_ids'],
                            max_new_tokens=2048, request_id=request['request_id'], session_id=f'event_{event:03d}',
                            request_index=request['request_index'], purpose='evaluation', save_to=parity_path)
                        errors = check_stream_record(control, request, checkpoints[arm], capture=False, canonical=record)
                        if control['metadata']['verification_calls'] != record['metadata']['verification_calls']:
                            errors.append('capture_mode_verification_count')
                        parity.append(dict(request_id=request['request_id'], errors=errors,
                                           capture_on_path=str(path), capture_off_path=str(parity_path)))
                        write_json(event_root / 'PARITY.json', parity)
                        if errors:
                            raise ValueError('Capture-mode parity failed: ' + repr(errors))
                progress.update(current_event=event, current_serving_arm=arm, event_served=len(rows))
                write_json(root / 'campaign_001/STATUS.json', progress)
        errors = compare_stream(records, event)
        invalid_ids = {row[0] for row in errors}
        for row in rows:
            if row['request_id'] in invalid_ids:
                row.update(valid_measurement=False, errors=row['errors'] + ['cross_arm_event_gate'])
        write_json(event_root / 'SERVING_CHECK.json', dict(status='failed' if errors else 'passed',
            errors=errors, requests=len(records), generations=len(rows), neutral_counts_required=event <= 2))
        if errors:
            raise ValueError('Cross-arm output/neutral-count check failed: ' + repr(errors))
        write_json(event_root / 'POOL.json', dict(immutable=True, bindings=bindings,
            preparations=preparations, teacher_terminal_forwards=len(preparations)))
        return bindings, sorted(descriptors, key=lambda row: row.global_ordinal)
    finally:
        save_rows(event_root / 'MEASUREMENTS.jsonl', rows)


def retire_previous_exports(root, checkpoints, event_root):
    """All preceding checkpoints are now finished serving; states remain durable."""
    if event_root.name == 'event_001':
        return
    for arm, export in checkpoints.items():
        for path in sorted(Path(export).iterdir()):
            if not path.is_file():
                raise ValueError('Unexpected directory in a new-run serving export')
            # Preserve small configs/receipts; only the replaceable weights matter.
            if path.suffix == '.safetensors':
                retire_owned(path, root=root,
                    receipt_path=root / 'campaign_001/RETIREMENTS' / f'{event_root.name}_{arm}_preceding_export.json',
                    reason='All preceding-checkpoint serving/validity complete; full own state retained',
                    retained_evidence=[event_root / 'SERVING_CHECK.json'])


def verify_neutral(root, event_root):
    import torch
    from .training import compare_neutral_states
    base = root / 'campaign_001/arms'
    first = torch.load(read(base / 'K/boundaries/event_001.json')['state'], map_location='cpu', weights_only=False)
    receipt = {}
    for arm in ('U', 'R'):
        other = torch.load(read(base / arm / 'boundaries/event_001.json')['state'], map_location='cpu', weights_only=False)
        receipt[arm] = compare_neutral_states(first, other)
        del other
    write_json(event_root / 'NEUTRAL_STATE.json', dict(status='passed', comparisons=receipt))
    del first
    gc.collect()


def publish_exports(root, plan, event, progress):
    from .training import materialize_export
    checkpoints = {}
    for arm in ARMS:
        base = root / 'campaign_001/arms' / arm
        boundary_path = base / 'boundaries' / f'event_{event:03d}.json'
        boundary = read(boundary_path)
        export = base / 'trainer/events' / f'event_{event:03d}/export'
        require_write_space(root, plan['storage'], 2 * plan['storage']['weight_bytes'])
        receipt = materialize_export(boundary['state'], plan['configs'][arm], export,
                                     receipt_path=export.parent / 'EXPORT_CHECK.json')
        progress['costs']['export_s'] += receipt['export_and_verify_s']
        boundary.update(checkpoint_path=str(export), export_status='materialized',
                        weights_identity=identity(export / 'model.safetensors'))
        write_json(boundary_path, boundary)
        checkpoints[arm] = str(export)
        gc.collect()
    return checkpoints


def run(root):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .evaluation import evaluate_final, audit_final
    from .report import build_report
    root = Path(root).absolute()
    plan = read(root / 'sealed/PLAN.json')
    inputs = read(plan['requests'])
    canonicals = read(plan['canonicals'])
    if plan['budget'] != BUDGET or len(inputs['training']) != 1024 or len(inputs['evaluation']) != 256:
        raise ValueError('Require the complete sealed eight-event protocol')
    campaign = root / 'campaign_001'
    campaign.mkdir(exist_ok=False)
    admit(root, plan['storage'], campaign / 'STORAGE_ADMISSION.json')
    started = time.perf_counter()
    progress = dict(status='running', phase='initializing', execution_source=plan['execution_source'],
        budget=BUDGET, events_completed=0, automatic_retry=False, costs={key: 0. for key in (
            'initialization_s', 'backend_loading_s', 'training_worker_wall_s', 'activation_s',
            'teacher_preparation_s', 'feature_persistence_s', 'descriptor_s', 'selection_s',
            'memory_maintenance_s', 'export_s', 'evaluation_wall_s')}, resource_checks=[])
    ledger = WorkLedger(campaign / 'WORK.jsonl', SERVING_LIMITS)
    backend = None
    memory, all_bindings, all_descriptors = ReservoirMemory(), {}, {}
    checkpoints = {arm: plan['draft'] for arm in ARMS}
    try:
        for arm in ARMS:
            result = execute_worker(root, plan, arm, 'initialize')
            progress['costs']['initialization_s'] += result['worker_wall_s']
        backend = SGLangDFlashBackend(plan['target'], plan['draft'], worker_python=plan['worker_python'],
            output_root=campaign / 'native', serving_gpu=str(plan['serving_gpu']), context_limit=4096, block_size=16)
        progress['costs']['backend_loading_s'] = backend.loading_s
        for event in range(1, EVENTS + 1):
            event_started = time.perf_counter()
            event_root = campaign / 'events' / f'event_{event:03d}'
            progress.update(phase='pre_update_serving', current_event=event)
            progress['resource_checks'].append(resource_check(root, plan, progress))
            requests = inputs['training'][(event-1)*128:event*128]
            fresh_bindings, fresh_records = serve_event(backend, ledger, requests, checkpoints,
                                                         event_root, root, plan, progress)
            all_bindings.update(fresh_bindings)
            all_descriptors.update({row.record_id: asdict(row) for row in fresh_records})
            write_json(campaign / 'ALL_DESCRIPTORS.json', all_descriptors)
            write_json(event_root / 'MEMORY_BEFORE.json', memory.state_dict())
            selection = build_event_plan(event, fresh_records, memory.records)
            # Domain annotations are joined only after the selector returns.
            domains = {row['request_id']: row['domain'] for row in inputs['training']}
            selection['reporting_only_domain_counts'] = {
                arm: dict(Counter(domains[rid] for rid in selection['pass_record_ids'][arm][1])) for arm in ARMS}
            selection['reporting_only_memory_domains'] = dict(Counter(domains[row.record_id] for row in memory.records))
            write_json(event_root / 'SELECTION.json', selection)
            pass_bindings = {arm: [[all_bindings[rid] for rid in rows] for rows in selection['pass_record_ids'][arm]]
                             for arm in ARMS}
            write_json(event_root / 'PASS_BINDINGS.json', pass_bindings)
            progress['costs']['selection_s'] += selection['cost']['total_wall_seconds']
            retire_previous_exports(root, checkpoints, event_root)
            verified = set()
            progress['phase'] = 'training'
            for arm in ARMS:
                progress['current_training_arm'] = arm
                write_json(campaign / 'STATUS.json', progress)
                require_write_space(root, plan['storage'], int(3.15 * plan['storage']['weight_bytes']))
                result = execute_worker(root, plan, arm, 'train', event)
                progress['costs']['training_worker_wall_s'] += result['worker_wall_s']
                verified.add(arm)
                if event > 1:
                    base = campaign / 'arms' / arm
                    previous = read(base / 'boundaries' / f'event_{event-1:03d}.json')
                    if identity(previous['state']) != previous['state_identity']:
                        raise ValueError('Preceding state changed before verified successor retirement')
                    retire_owned(previous['state'], root=root,
                        receipt_path=campaign / 'RETIREMENTS' / f'event_{event:03d}_{arm}_predecessor_state.json',
                        reason='Verified complete own successor; earlier serving finished; exact old resume retired',
                        retained_evidence=[base / 'boundaries' / f'event_{event:03d}.json',
                                           base / 'trainer/events' / f'event_{event-1:03d}/RESULT.json'])
                progress['resource_checks'].append(resource_check(root, plan, progress))
            if event == 1:
                verify_neutral(root, event_root)
            maintenance = time.perf_counter()
            insertion = memory.insert_after_event(event, fresh_records, verified_arms=verified)
            write_json(event_root / 'MEMORY_INSERTION.json', insertion)
            write_json(event_root / 'MEMORY_AFTER.json', memory.state_dict())
            write_json(campaign / 'MEMORY.json', memory.state_dict())
            retained = {row.record_id for row in memory.records}
            for rid, binding in all_bindings.items():
                path = Path(binding['path'])
                if rid not in retained and path.exists():
                    if identity(path)['sha256'] != binding['sha256']:
                        raise ValueError('Evicted packet no longer matches its immutable identity')
                    retire_owned(path, root=root,
                        receipt_path=campaign / 'RETIREMENTS' / (rid + '_prepared.json'),
                        reason='All three event commits verified; Algorithm R excludes packet from next memory',
                        retained_evidence=[event_root / 'MEMORY_AFTER.json', path.with_suffix('.json')])
            progress['costs']['memory_maintenance_s'] += time.perf_counter() - maintenance
            checkpoints = publish_exports(root, plan, event, progress)
            progress['resource_checks'].append(resource_check(root, plan, progress))
            progress.update(events_completed=event, phase='idle_boundary')
            write_json(event_root / 'COMPLETE.json', dict(status='passed', event=event, verified_arms=sorted(verified),
                checkpoints=checkpoints, event_wall_s=time.perf_counter()-event_started))
            write_json(campaign / 'STATUS.json', progress)
        require_complete_work(read_work(campaign / 'WORK.jsonl', SERVING_LIMITS), SERVING_LIMITS)
        progress['adaptation_ready_wall_s'] = time.perf_counter() - started
        progress['phase'] = 'final_evaluation'
        progress['resource_checks'].append(resource_check(root, plan, progress))
        write_json(campaign / 'STATUS.json', progress)
        from .preflight import timing_barrier
        def evaluation_boundary():
            resource = resource_check(root, plan, progress)
            barrier = timing_barrier(backend.worker.pid)
            progress['resource_checks'].append(dict(phase='final_evaluation', resource=resource, timing=barrier))
            write_json(campaign / 'STATUS.json', progress)
            return dict(resource=resource, timing=barrier)

        evaluation = evaluate_final(backend, inputs['evaluation'], checkpoints,
            campaign / 'final_evaluation', canonicals=canonicals, boundary_check=evaluation_boundary)
        progress['costs']['evaluation_wall_s'] = evaluation['evaluation_wall_s']
        audit = audit_final(campaign / 'final_evaluation', inputs['evaluation'], checkpoints, canonicals=canonicals)
        write_json(campaign / 'FINAL_AUDIT.json', audit)
        if audit['status'] != 'passed':
            raise ValueError('Final raw-evidence validation failed')
        progress.update(status='completed', phase='finished')
    except BaseException as error:
        progress.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        cleanup_error = None
        try:
            if backend is not None:
                backend.close()
        except BaseException as error:
            cleanup_error = error
            progress.update(status='failed', cleanup_error=repr(error))
        progress['campaign_wall_s'] = time.perf_counter()-started
        progress['serving_work'] = read_work(campaign / 'WORK.jsonl', SERVING_LIMITS)
        write_json(campaign / 'STATUS.json', progress)
        try:
            report = build_report(root)
            if progress['status'] == 'completed' and report['status'] != 'valid_complete':
                raise RuntimeError('Completed scientific work did not pass report/evidence accounting: ' + repr(report['errors']))
        except BaseException as error:
            write_json(campaign / 'REPORT_FAILURE.json', dict(error=repr(error), traceback=traceback.format_exc()))
            if progress['status'] == 'completed':
                raise
        if cleanup_error is not None:
            raise cleanup_error
    return progress


if __name__ == '__main__':
    def terminate_owned_work(signum, frame):
        # Let worker/serving finally blocks close their owned process groups.
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_owned_work)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    print(json.dumps(run(parser.parse_args().root), indent=2))
