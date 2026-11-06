"""Reconstruct six fixed feedback records through the qualified native verifier.

No answer is generated here. The target replays the saved prefill and every
saved verification input, including rejected proposals. Its posterior and
committed tokens must match before a reconstructed feature can enter training.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback

from experiments.block_depth.reconstruct import validate_source
from experiments.o_half_matched_lr.archive import full_trace_errors
from experiments.ok_coding.campaign import WorkLedger, read_work
from online_feedback.sglang_backend import SGLangDFlashBackend
from online_feedback.storage import write_json


SOURCE_IDS = ('train_00072', 'train_00121', 'train_00004',
              'train_00037', 'train_00041', 'train_00060')
LIMITS = dict(native_reconstructions=6, terminal_preparations=6)
BINDING_FIELDS = ('record_id', 'prompt_length', 'response_length', 'input_ids',
                  'target_argmax_ids', 'target_layer_ids', 'feature_shape',
                  'label_binding_scope', 'response_only_loss',
                  'final_feature_logits_used_for_labels')


def _read(path):
    return json.loads(Path(path).read_text())


def _identity(path):
    payload = Path(path).read_bytes()
    return dict(path=str(Path(path).absolute()), bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest())


def _check_anchors(row, source, ordinal):
    draws = row['sampled_anchors']
    if len(draws) != 1:
        raise ValueError('Each retained presentation must contain one anchor draw')
    positions, keep = draws[0]['positions'], draws[0]['keep_mask']
    if len(positions) != 1 or len(keep) != 1 or len(positions[0]) != len(keep[0]):
        raise ValueError('Retained anchors must have the original single-response shape')
    if any(type(value) is not bool for value in keep[0]):
        raise ValueError('Anchor membership must retain Boolean values')
    active = [value for value, enabled in zip(positions[0], keep[0]) if enabled]
    prompt_length = len(source['prompt_ids'])
    response_length = len(source['response_ids'])
    if (not active or len(active) > 512 or active != sorted(set(active))
            or any(type(value) is not int or not prompt_length <= value < prompt_length + response_length - 1
                   for value in active)
            or row['active_anchors'] != len(active)
            or row['prompt_tokens'] != prompt_length or row['response_tokens'] != response_length):
        raise ValueError('Retained anchor eligibility/exposure differs from the exact response')
    observation = row['response_observation']
    if (observation['record_id'] != row['record_id'] or observation['event_id'] != 'event_001'
            or observation['pass_index'] != 0 or observation['presentation_ordinal'] != ordinal
            or observation['sampled_anchors'] != draws):
        raise ValueError('Retained presentation and anchor-observation identities differ')


def select_sources(source_campaign_path):
    """Seal the first six presentations, without sorting or sampling responses."""
    arm = Path(source_campaign_path).absolute() / 'arms/A'
    result_path = arm / 'trainer/events/event_001/RESULT.json'
    result = _read(result_path)
    records = deepcopy(result['records'][:6])
    if (result['status'] != 'completed' or result['passes'] != 2
            or result['response_presentations'] != 256 or len(result['records']) != 256
            or tuple(row['record_id'] for row in records) != SOURCE_IDS
            or any(row['pass'] != 0 for row in records)):
        raise ValueError('Use exactly the first six retained event-1, pass-1 presentations')
    sources, bindings, identities = [], [], [_identity(result_path)]
    loop = arm / 'events/event_001/loop'
    for ordinal, row in enumerate(records):
        record_id = row['record_id']
        source_path = loop / 'serving' / (record_id + '.json')
        binding_path = loop / 'features' / (record_id + '.json')
        source, binding = _read(source_path), _read(binding_path)
        validate_source(source)
        errors = full_trace_errors(source)
        if errors:
            raise ValueError('Retained native trace is invalid: ' + ', '.join(errors))
        metadata = source['metadata']
        if (source['record_id'] != record_id or source['purpose'] != 'serving'
                or source['max_new_tokens'] != 2048 or metadata['block_size'] != 16
                or metadata['context_limit'] != 4096 or metadata['runtime'] != 'pinned_SGLang_FA3'
                or len(source['prompt_ids']) + 2048 > 4096 or len(source['response_ids']) < 2
                or any(block['verify_size'] != 16 for block in metadata['trace'])):
            raise ValueError('Retained response does not use the qualified native serving contract')
        _check_anchors(row, source, ordinal)
        if (binding['record_id'] != record_id
                or binding['input_ids'] != source['prompt_ids'] + source['response_ids']
                or binding['target_argmax_ids'] != source['response_ids']
                or binding['response_only_loss'] is not True
                or binding['label_binding_scope'] != 'native_prefill_and_committed_verification'
                or binding['final_feature_logits_used_for_labels'] is not False):
            raise ValueError('Retained feature receipt does not bind this canonical response')
        sources.append(source)
        bindings.append(binding)
        identities.extend((_identity(source_path), _identity(binding_path)))
    return dict(records=records, sources=sources, original_bindings=bindings,
        pairs=[[0, 1], [2, 3], [4, 5]], source_identities=identities,
        selection='first_six_event_001_pass_0_presentations', new_generations=0,
        expected_native_prefill_forwards=6,
        expected_native_verification_forwards=sum(len(s['metadata']['trace']) for s in sources),
        expected_terminal_preparation_forwards=6,
        reconstruction_limit='Native posterior/commit equivalence; retired feature bytes cannot be compared')


def _native_work(output, selection, ledger):
    """Retain known partial forwards and distinguish failed operations from unattempted ones."""
    work = read_work(ledger.path, LIMITS)
    work['unattempted'] = {key: LIMITS[key] - work['attempts'][key] for key in LIMITS}
    work.update(new_generations=0, native_prefill_forwards=0, native_verification_forwards=0,
                posterior_positions_checked=0, terminal_preparation_forwards=0,
                active_or_unknown_forwards=[])
    for ordinal, row in enumerate(selection['records']):
        receipt_path = output / 'reconstruction' / (row['record_id'] + '.json')
        if receipt_path.exists():
            receipt = _read(receipt_path)
            for field in ('prefill_forwards', 'verification_forwards'):
                work['native_' + field] += receipt.get(field, 0)
            work['posterior_positions_checked'] += receipt.get('posterior_positions_checked', 0)
            if receipt.get('active_forward') is not None:
                work['active_or_unknown_forwards'].append(dict(record_id=row['record_id'],
                    operation='reconstruction', detail=receipt['active_forward']))
        elif ordinal < work['attempts']['native_reconstructions']:
            work['active_or_unknown_forwards'].append(dict(record_id=row['record_id'],
                operation='reconstruction', detail='No native receipt returned'))
        terminal_path = output / 'features' / (row['record_id'] + '.terminal.json')
        if terminal_path.exists():
            work['terminal_preparation_forwards'] += _read(terminal_path)['target_forward_count']
        elif ordinal < work['attempts']['terminal_preparations']:
            work['active_or_unknown_forwards'].append(dict(record_id=row['record_id'],
                operation='terminal_preparation', detail='No completed target-forward receipt'))
    return work


def prepare(config, source_campaign_path, output_dir):
    """Return six prepared feature paths and exact retained presentation records.

    Required config: target_path, draft_path (released), worker_python. Optional
    serving_gpu defaults to 0. The worker's depth-routing controller remains
    inactive: only its already-qualified native reconstruction RPC is used.
    """
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=False)
    selection = select_sources(source_campaign_path)
    selection['runtime'] = dict(target_path=str(config['target_path']),
        draft_path=str(config['draft_path']), worker_python=str(config['worker_python']),
        serving_gpu=str(config.get('serving_gpu', '0')), context_limit=4096, block_size=16,
        worker_module='experiments.block_depth.worker', additional_generation_warmups=0)
    write_json(output / 'SELECTION.json', selection)
    ledger = WorkLedger(output / 'WORK.jsonl', LIMITS)
    result = dict(status='running', feature_paths=[], records=selection['records'],
        pairs=selection['pairs'], source_identities=selection['source_identities'],
        selection_path=str(output / 'SELECTION.json'), preparations=[],
        costs=dict(loading_s=0.0, reconstruction_s=0.0, teacher_preparation_s=0.0,
                   feature_persistence_s=0.0))
    backend = None
    started = time.perf_counter()
    try:
        backend = SGLangDFlashBackend(config['target_path'], config['draft_path'],
            worker_python=config['worker_python'], output_root=output / 'native_worker',
            serving_gpu=str(config.get('serving_gpu', '0')), context_limit=4096,
            block_size=16, reasoning='off', worker_module='experiments.block_depth.worker')
        result['costs']['loading_s'] = backend.loading_s
        for source, binding in zip(selection['sources'], selection['original_bindings']):
            record_id = source['record_id']
            receipt_path = output / 'reconstruction' / (record_id + '.json')
            reply = ledger.call('native_reconstructions', backend._call, 'depth_rpc',
                name='depth_reconstruct', arguments=dict(record=source, outputfile=str(receipt_path)))
            receipt = reply['result']
            if (receipt['status'] != 'passed' or receipt['new_generations'] != 0
                    or receipt['prefill_forwards'] != 1
                    or receipt['verification_forwards'] != len(source['metadata']['trace'])
                    or receipt['posterior_positions_checked'] != 16 * len(source['metadata']['trace'])):
                raise ValueError('Native reconstruction did not validate every retained posterior')
            restored = receipt['record']
            for field in ('record_id', 'prompt_ids', 'response_ids', 'finish_reason', 'stop_token_ids'):
                if restored[field] != source[field]:
                    raise ValueError('Native reconstruction changed canonical ' + field)
            feature_path = output / 'features' / (record_id + '.pt')
            prepared = ledger.call('terminal_preparations', backend.prepare_feedback, restored, feature_path)
            if prepared['status'] != 'prepared' or any(prepared[key] != binding[key] for key in BINDING_FIELDS):
                raise ValueError('Reconstructed feature binding differs from the retained preparation')
            result['feature_paths'].append(str(feature_path))
            result['preparations'].append(prepared)
            result['costs']['reconstruction_s'] += receipt['wall_s']
            for field in ('teacher_preparation_s', 'feature_persistence_s'):
                result['costs'][field] += prepared[field]
            write_json(output / 'PREPARED.json', result)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
        except BaseException as error:
            result['cleanup_error'] = repr(error)
            if result['status'] != 'failed':
                result['status'] = 'failed'
                raise
        finally:
            result['work'] = _native_work(output, selection, ledger)
            result['costs']['wall_s'] = time.perf_counter() - started
            write_json(output / 'PREPARED.json', result)
    return result
