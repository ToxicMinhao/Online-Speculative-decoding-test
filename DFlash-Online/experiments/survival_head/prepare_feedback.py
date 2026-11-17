"""Rebuild the retired event-five pool only after exact two-packet qualification.

The caller admits this separate post-PR36 work before invoking prepare(plan).
There are 128 target-path replays (two disposable comparisons and 126 missing
packets), zero newly generated answers, and one terminal-row forward per replay.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import traceback

from experiments.ok_coding.storage import identity
from online_feedback.presentation_schedule import PresentationSchedule
from online_feedback.sglang_backend import SGLangDFlashBackend
from online_feedback.storage import write_json
from .data import (feedback_requests, read, seal_shared_packets,
                   verify_shared_packets)
from .feedback import (expected_reconstruction_work, reconstruct_feedback,
                       require_reconstruction_qualification, source_contract_sha256,
                       validate_saved_record)


def _copy_exact(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if identity(source) != expected:
        raise ValueError('Original feedback artifact identity changed: ' + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    if identity(destination) != expected:
        raise ValueError('Pilot-owned feedback copy differs: ' + str(destination))
    return dict(source=str(source), destination=str(destination), **expected)


def _identity_fields(value):
    return {key: value[key] for key in ('bytes', 'sha256')}


def _feature_metadata(record, path):
    length, prompt = len(record['prompt_ids']) + len(record['response_ids']), len(record['prompt_ids'])
    return dict(record_id=record['record_id'], length=length, prompt_length=prompt,
        response_tokens=length - prompt, adjacent_response_positions=length - prompt - 1,
        path=str(path))


def authenticate_inputs(plan):
    """Read-only, complete source validation before any model can be loaded."""
    for key in ('prior_pool', 'original_schedule'):
        if identity(plan[key]) != plan[key + '_identity']:
            raise ValueError('Sealed original input changed: ' + key)
    pool, schedule = read(plan['prior_pool']), read(plan['original_schedule'])
    requests = feedback_requests(plan)
    rows = pool.get('records', [])
    expected_ids = [request['request_id'] for request in requests]
    if (pool.get('dataset') != 'gsm8k' or pool.get('event') != 5
            or pool.get('producer') != 'KPLUS4' or len(rows) != 128
            or [row['record_id'] for row in rows] != expected_ids):
        raise ValueError('Original event-five pool identities/order differ')
    if len(plan['retained_packet_paths']) != 2:
        raise ValueError('Require exactly the two retained prepared packets')
    origin = plan.get('origin_export') or plan['datasets']['gsm8k']['origin_export']
    records, contracts, work = [], [], []
    for index, (row, request) in enumerate(zip(rows, requests)):
        canonical = row['canonical']
        if identity(canonical['path']) != _identity_fields(canonical):
            raise ValueError('Original canonical JSON identity changed')
        record = read(canonical['path'])
        validate_saved_record(record)
        if (record['record_id'] != request['request_id']
                or record['prompt_ids'] != request['prompt_ids']
                or record['max_new_tokens'] != request['max_new_tokens']
                or record['checkpoint_id'] != origin):
            raise ValueError('Original canonical feedback differs from its selected input/origin')
        if index < 2 and identity(plan['retained_packet_paths'][index]) != _identity_fields(row['feature']):
            raise ValueError('Retained prepared packet differs from the original pool identity')
        records.append(record)
        contracts.append(source_contract_sha256(record))
        work.append(expected_reconstruction_work(record))
    expected_schedule = dict(status='completed', dataset='gsm8k', method='KPLUS',
        event_id='event_005', start_step=512, end_step=640,
        optimizer_steps=128, passes=2, response_presentations=256)
    if any(schedule.get(key) != value for key, value in expected_schedule.items()):
        raise ValueError('Original ordinary KPLUS5 schedule has a different event boundary')
    PresentationSchedule(schedule, expected_ids, 2)
    features = [_feature_metadata(record, row['feature']['path']) for record, row in zip(records, rows)]
    if schedule['features'] != features:
        raise ValueError('Original schedule feature identities, lengths or paths differ from its pool')
    totals = {key: sum(row[key] for row in work) for key in work[0]}
    return dict(pool=pool, schedule=schedule, requests=requests, records=records,
        source_contracts=contracts, expected_work=totals,
        qualification_record_ids=expected_ids[:2])


def rebound_schedule(schedule, records, feature_paths):
    """Only paths change; ordinary presentation order and anchor draws stay exact."""
    if len(records) != len(feature_paths) or len(records) != 128:
        raise ValueError('Require the complete pilot-owned shared pool')
    result = deepcopy(schedule)
    replacement = [_feature_metadata(record, str(Path(path).absolute()))
                   for record, path in zip(records, feature_paths)]
    for old, new in zip(result['features'], replacement):
        if {k: v for k, v in old.items() if k != 'path'} != {k: v for k, v in new.items() if k != 'path'}:
            raise ValueError('Cannot rebind changed feature metadata')
    result['features'] = replacement
    comparison = deepcopy(result)
    comparison['features'] = deepcopy(schedule['features'])
    if comparison != schedule:
        raise ValueError('Presentation schedule rebind changed more than paths')
    PresentationSchedule(result, [r['record_id'] for r in records], 2, features=replacement)
    return result


def _observed_work(rows):
    fields = ('prefill_attempts', 'prefill_forwards', 'verification_attempts',
              'verification_forwards', 'terminal_attempts', 'terminal_forwards')
    totals = dict.fromkeys(fields, 0)
    for row in rows:
        path = Path(row['feature_path']).with_suffix('.reconstruction.json')
        if not path.exists():
            continue
        receipt = read(path)
        replay = receipt.get('replay', {})
        for key in fields:
            totals[key] += receipt.get(key, 0) if key.startswith('terminal_') else replay.get(key, 0)
    totals.update(new_generations=0, draft_forwards=0,
        target_forwards=totals['prefill_forwards'] + totals['verification_forwards'] + totals['terminal_forwards'])
    return totals


def retire_redundant_capture(feature_path, reconstruction, *, root):
    """Remove only new raw copies whose exact rows survive in this prepared file."""
    import torch

    root, feature = Path(root).resolve(), Path(feature_path).absolute()
    receipt_path = feature.with_suffix('.redundancy.json')
    if receipt_path.exists() or reconstruction['status'] != 'passed':
        raise ValueError('Retirement requires a newly passed reconstruction')
    capture = Path(reconstruction['replay']['record']['metadata']['capture']['feature_path'])
    terminal = Path(reconstruction['preparation']['terminal_feature']['tensor_path'])
    if (capture != feature.with_suffix('.replay.capture.pt')
            or terminal != feature.with_suffix('.terminal.pt')):
        raise ValueError('Retire only this reconstruction\'s two newly created raw copies')
    for path in (feature, capture, terminal, receipt_path):
        if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('Redundant capture lies outside the pilot-owned feedback root')
    payload = torch.load(feature, map_location='cpu', weights_only=False)
    raw = torch.load(capture, map_location='cpu', weights_only=False)
    last = torch.load(terminal, map_location='cpu', weights_only=False)
    ids = payload['input_ids'][0].tolist()
    if (raw['input_ids'] != ids[:-1] or last['input_ids'] != ids
            or last['positions'] != [len(ids) - 1]
            or last['final_feature_logits_used_for_labels'] is not False):
        raise ValueError('Raw capture/terminal conditioning differs; retain both files')
    for key in ('hidden_states', 'target_last_hidden_states'):
        for original, retained in ((raw[key], payload[key][0, :-1]), (last[key], payload[key][0, -1:])):
            if (original.dtype != retained.dtype or original.shape != retained.shape
                    or not torch.equal(original.contiguous().view(torch.uint8),
                                       retained.contiguous().view(torch.uint8))):
                raise ValueError('Raw feature rows differ from prepared copy; retain both files')
    retained_identity = identity(feature)
    receipt = dict(status='validated_before_removal',
        retained=dict(path=str(feature), **retained_identity),
        candidates=[dict(path=str(path), **identity(path)) for path in (capture, terminal)],
        reason='Exact BF16 feature rows retained without modification in the prepared packet',
        proof='Both auxiliary and normalized final hidden tensors compared bitwise; canonical inputs and terminal position checked',
        compact_capture_and_terminal_JSON_retained=True,
        lost='Standalone redundant raw tensor paths; no feature values lost, replay rows remain in the prepared packet',
        removed_paths=[], free_bytes_before=shutil.disk_usage(root).free)
    write_json(receipt_path, receipt)
    try:
        for row in receipt['candidates']:
            if identity(feature) != retained_identity or identity(row['path']) != _identity_fields(row):
                raise ValueError('Retained or redundant bytes changed after validation')
            Path(row['path']).unlink()
            receipt['removed_paths'].append(row['path'])
            write_json(receipt_path, receipt)
        receipt['status'] = 'completed'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        receipt['free_bytes_after'] = shutil.disk_usage(root).free
        write_json(receipt_path, receipt)
    return receipt


def prepare(plan, *, backend_factory=None):
    """Authenticated copies -> two exact comparisons -> missing126 -> sealed pool."""
    authenticated = authenticate_inputs(plan)
    root = Path(plan['root']).absolute() / 'feedback'
    root.mkdir(parents=True, exist_ok=False)
    destination = root / 'PREPARATION.json'
    receipt = dict(status='running', automatic_retry=False, new_generations=0,
        source_pool=dict(path=plan['prior_pool'], **plan['prior_pool_identity']),
        original_schedule=dict(path=plan['original_schedule'], **plan['original_schedule_identity']),
        expected_work=authenticated['expected_work'], reconstruction_attempts=[],
        copied_records=[], reused_packets=[], original_missing_packet_byte_comparison='unavailable_126_retired',
        source_contracts=authenticated['source_contracts'])
    backend, failure = None, None
    started = time.perf_counter()
    feature_paths, canonical_paths, qualification = [], [], []
    write_json(destination, receipt)
    try:
        for index, (row, record) in enumerate(zip(authenticated['pool']['records'], authenticated['records'])):
            # An ordinal controls filenames; arbitrary request IDs never become paths.
            canonical = root / 'records' / f'{index:03d}.json'
            feature = root / 'features' / f'{index:03d}.pt'
            receipt['copied_records'].append(_copy_exact(row['canonical']['path'], canonical,
                                                        _identity_fields(row['canonical'])))
            canonical_paths.append(canonical)
            feature_paths.append(feature)
            if index < 2:
                receipt['reused_packets'].append(_copy_exact(plan['retained_packet_paths'][index], feature,
                                                           _identity_fields(row['feature'])))
        _copy_exact(plan['original_schedule'], root / 'ORIGINAL_PRESENTATION_SCHEDULE.json',
                    plan['original_schedule_identity'])
        write_json(root / 'TARGET_WORK_PLAN.json', dict(status='sealed_before_worker_startup',
            work=authenticated['expected_work'], qualification_replays=2, missing_packet_replays=126,
            generations=0, source_contracts=authenticated['source_contracts']))
        write_json(destination, receipt)
        factory = backend_factory or SGLangDFlashBackend
        origin = plan.get('origin_export') or plan['datasets']['gsm8k']['origin_export']
        backend = factory(plan['target'], origin, worker_python=plan['worker_python'],
            output_root=root / 'native', serving_gpu=str(plan['serving_gpu']),
            context_limit=4096, block_size=16, worker_module='online_feedback.survival_feedback_worker')
        receipt['loading_s'] = backend.loading_s
        for index, record in enumerate(authenticated['records']):
            if index == 2:
                receipt['qualification'] = require_reconstruction_qualification(
                    qualification, authenticated['qualification_record_ids'])
                write_json(root / 'RECONSTRUCTION_QUALIFICATION.json', receipt['qualification'])
            feature = root / 'qualification' / f'{index:03d}.pt' if index < 2 else feature_paths[index]
            row = dict(record_id=record['record_id'], feature_path=str(feature),
                scope='retained_packet_qualification' if index < 2 else 'missing_packet_reconstruction',
                status='attempted')
            receipt['reconstruction_attempts'].append(row)
            write_json(destination, receipt)
            result = reconstruct_feedback(backend, record, feature,
                retained_packet=feature_paths[index] if index < 2 else None)
            redundancy = retire_redundant_capture(feature, result, root=root)
            row.update(status='passed', reconstruction_receipt=str(feature.with_suffix('.reconstruction.json')))
            row['redundancy_receipt'] = str(feature.with_suffix('.redundancy.json'))
            row['removed_redundant_paths'] = redundancy['removed_paths']
            if index < 2:
                qualification.append(result)
            receipt['observed_work'] = _observed_work(receipt['reconstruction_attempts'])
            write_json(destination, receipt)
        # Verify source invariance again after all target work; originals are never rewritten.
        authenticate_inputs(plan)
        import torch
        from online_feedback.training import validate_feature, validate_target_binding
        for record, feature in zip(authenticated['records'], feature_paths):
            packet = torch.load(feature, map_location='cpu', weights_only=False)
            validate_target_binding(packet)
            actual = {**validate_feature(packet, max_length=4096), 'path': str(feature)}
            if (actual != _feature_metadata(record, feature)
                    or packet['hidden_states'].dtype != torch.bfloat16
                    or packet['target_last_hidden_states'].dtype != torch.bfloat16):
                raise ValueError('Prepared packet tensor metadata differs from original schedule')
        schedule = rebound_schedule(authenticated['schedule'], authenticated['records'], feature_paths)
        write_json(root / 'PRESENTATION_SCHEDULE.json', schedule)
        write_json(root / 'SCHEDULE_REBIND.json', dict(status='passed',
            original_identity=plan['original_schedule_identity'],
            rebound_identity=identity(root / 'PRESENTATION_SCHEDULE.json'),
            changes='feature paths only; original metadata, response order and sampled anchors unchanged',
            original_sampler_is_not_replaced=True, original_schedule_path=str(root / 'ORIGINAL_PRESENTATION_SCHEDULE.json')))
        pool = seal_shared_packets(feature_paths, canonical_paths, output=root / 'POOL.json',
            expected_request_ids=[r['record_id'] for r in authenticated['records']],
            provenance=dict(original_pool=receipt['source_pool'], copied_retained_packets=2,
                exactly_qualified_reconstruction_packets=126, new_generations=0,
                reconstructed_full_old_byte_comparison='unavailable_126_retired',
                reconstruction_qualification=str(root / 'RECONSTRUCTION_QUALIFICATION.json')))
        verify_shared_packets(pool)
        observed = _observed_work(receipt['reconstruction_attempts'])
        for key in ('prefill_forwards', 'verification_forwards', 'terminal_forwards', 'target_forwards'):
            if observed[key] != authenticated['expected_work'][key]:
                raise ValueError('Target reconstruction work did not reconcile: ' + key)
        receipt.update(status='completed', pool=str(root / 'POOL.json'), observed_work=observed,
            presentation_schedule=str(root / 'PRESENTATION_SCHEDULE.json'), packets=128,
            reused_original_packet_bytes=2, reconstructed_missing_packets=126)
    except BaseException as error:
        failure = error
        receipt.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as error:
                receipt.update(status='failed', cleanup_error=repr(error))
                if failure is None:
                    receipt['wall_s'] = time.perf_counter() - started
                    write_json(destination, receipt)
                    raise
        receipt['observed_work'] = _observed_work(receipt['reconstruction_attempts'])
        receipt['wall_s'] = time.perf_counter() - started
        write_json(destination, receipt)
    return receipt


def run(root):
    # A direct CLI invocation is a model-work entry point too. Do not rely on
    # an outer supervisor having checked PR36/identity/capacity earlier.
    from .preflight import verify
    verify(root, scope='qualification')
    return prepare(read(Path(root) / 'sealed/PLAN.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    args = parser.parse_args()
    print(json.dumps(run(args.root)))
