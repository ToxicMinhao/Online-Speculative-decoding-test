"""The fixed 60-generation native gate and 2,048-generation AR/F precheck.

Backend lifetimes, resource admission and occupied-card clocks are supplied by
the campaign. All generation and teacher calls pass through its durable ledger.
This module never creates an optimizer, changes a checkpoint or retries a slot.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from online_feedback.native import validate_generation
from online_feedback.storage import write_json

from .contract import digest
from .validity import validate_ar_output, validate_canonical, validate_measurement


def qualification_slots(inputs):
    """Seal 60 initial executions; four publication checks follow training qualification."""
    reserved, warmups = inputs['qualification_requests'], inputs['warmup_requests']
    if len(reserved) != 16 or len(warmups) != 2:
        raise ValueError('Qualification requires the sealed 16 inputs and two warmups')
    short = min(range(16), key=lambda i: (len(reserved[i]['prompt_ids']), i))
    long = min((i for i in range(16) if i != short),
               key=lambda i: (-len(reserved[i]['prompt_ids']), i))
    chosen = [reserved[short], reserved[long]]
    slots = []

    def add(card, method, request, cap, repeat, phase):
        label = f'{card}_{method}_{phase}_{request["request_index"]}_{cap}_{repeat}'
        slots.append(dict(slot_id=label, card=card, method=method, phase=phase,
            request_id=request['request_id'], max_new_tokens=cap, repeat=repeat,
            collect_features=method == 'F' and phase == 'reserved',
            request_index=request['request_index']))

    for card in ('A', 'B'):
        for method in ('AR', 'F'):
            if card == 'B':
                for request in chosen:
                    add(card, method, request, 2048, 0, 'cross_card')
                continue
            for request in reserved:
                add(card, method, request, 2048, 0, 'reserved')
            for request in warmups:
                add(card, method, request, 2048, 0, 'warmup')
            for request in chosen:
                add(card, method, request, 2048, 1, 'normal_repeat')
            for cap in (1, 17):
                for repeat in range(2):
                    for request in chosen:
                        add(card, method, request, cap, repeat, 'short_cap')
    if len(slots) != 60 or len({row['slot_id'] for row in slots}) != 60:
        raise ValueError('Qualification slot accounting changed')
    return slots


def _record_errors(record, request, slot, checkpoint):
    errors = []
    expected = dict(record_id=slot['slot_id'], prompt_ids=request['prompt_ids'],
        checkpoint_id=checkpoint, max_new_tokens=slot['max_new_tokens'],
        stop_token_ids=request['stop_token_ids'], request_index=request['request_index'],
        purpose='serving' if slot.get('collect_features') else 'evaluation')
    for key, value in expected.items():
        if record.get(key) != value:
            errors.append('identity:' + key)
    try:
        validate_measurement(record, arm=slot['method'])
        if slot['method'] == 'AR':
            _, finish = validate_ar_output(request['prompt_ids'], record['response_ids'],
                record['metadata']['native_meta_info'], max_new_tokens=slot['max_new_tokens'],
                stop_token_ids=request['stop_token_ids'])
            if finish != record['finish_reason']:
                errors.append('semantic_termination')
        else:
            _, finish = validate_generation(request['prompt_ids'],
                request['prompt_ids'] + record['response_ids'], record['metadata']['trace'],
                max_new_tokens=slot['max_new_tokens'], stop_tokens=request['stop_token_ids'])
            if finish != record['finish_reason']:
                errors.append('semantic_termination')
        if record.get('validation', {}).get('status') == 'failed':
            errors.append('backend_rejected_record')
    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        errors.append(str(error))
    return errors


def _compare(record, reference, *, compare_counts):
    validate_canonical(record, reference)
    if compare_counts:
        for key in ('verification_calls', 'accepted_draft_tokens', 'committed_progress'):
            if record['metadata'][key] != reference['metadata'][key]:
                raise ValueError('Repeated native count changed: ' + key)


def _run_slots(plan, requests, slots, base, *, ledger, backend_factory,
               close_backend, boundary, category, references=None, prepare_features=False):
    """Record each attempted/completed/invalid/unattempted slot before stopping."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=False)
    records, features = {}, {}
    work = {slot['slot_id']: dict(slot, status='unattempted') for slot in slots}
    write_json(base / 'SLOTS.json', slots)
    write_json(base / 'WORK.json', work)
    write_json(base / 'PROVENANCE.json', dict(execution_commit=plan['execution_commit'],
        canonical_provenance=plan['canonical_provenance'],
        checkpoints=plan['checkpoints'], devices=plan['devices']))
    backend, identity = None, None
    try:
        for slot in slots:
            boundary()
            current = (slot['card'], slot['method'])
            if current != identity:
                if backend is not None:
                    closing, backend = backend, None
                    close_backend(closing, base / f'{identity[0]}_{identity[1]}')
                engine_root = base / f'{slot["card"]}_{slot["method"]}'
                backend = backend_factory(slot['method'], plan['devices'][slot['card']], engine_root)
                identity = current
            request = requests[slot['request_id']]
            path = base / 'records' / (slot['slot_id'] + '.json')
            work[slot['slot_id']].update(status='started', started_epoch=time.time())
            write_json(base / 'WORK.json', work)
            try:
                kind = category(slot) if callable(category) else category
                record = ledger.call(kind, slot['slot_id'], backend.generate,
                    request['prompt_ids'], max_new_tokens=slot['max_new_tokens'],
                    request_id=slot['slot_id'], session_id=plan['session'],
                    request_index=request['request_index'],
                    purpose='serving' if slot.get('collect_features') else 'evaluation', save_to=path)
                write_json(path, record)
                records[slot['slot_id']] = record
                work[slot['slot_id']].update(status='completed', record_path=str(path))
                write_json(base / 'WORK.json', work)
                checkpoint = plan['target'] if slot['method'] == 'AR' else plan['checkpoints']['F']
                errors = _record_errors(record, request, slot, checkpoint)
                prior = [other for label, other in records.items() if label != slot['slot_id']
                         and other['prompt_ids'] == request['prompt_ids']
                         and other['max_new_tokens'] == slot['max_new_tokens']]
                try:
                    for previous in prior:
                        same_method = previous['checkpoint_id'] == record['checkpoint_id']
                        _compare(record, previous, compare_counts=same_method)
                    if references is not None:
                        _compare(record, references[slot['request_id']], compare_counts=False)
                except (ValueError, KeyError, TypeError) as error:
                    errors.append(str(error))
                work[slot['slot_id']].update(status='invalid' if errors else 'valid', errors=errors)
                write_json(base / 'WORK.json', work)
                if errors:
                    raise ValueError('Serving gate failed: ' + repr(errors))
                if prepare_features and slot.get('collect_features'):
                    from .runtime import require_eligible_response
                    require_eligible_response(record)
                    feature_path = base / 'features' / (slot['slot_id'] + '.pt')
                    receipt = ledger.call({'teacher_preparation': 1}, slot['slot_id'],
                        backend.prepare_feedback, record, feature_path)
                    features[slot['request_id']] = dict(record_path=str(path), receipt=receipt,
                        source_slot=slot['slot_id'])
                    write_json(base / 'FEATURES.json', features)
            except BaseException as error:
                if work[slot['slot_id']]['status'] == 'started':
                    # The durable call ledger determines unknown vs completed
                    # work; a missing return must never be called unattempted.
                    work[slot['slot_id']]['status'] = 'failed_return_unknown'
                work[slot['slot_id']]['error'] = repr(error)
                write_json(base / 'WORK.json', work)
                raise
    finally:
        if backend is not None:
            close_backend(backend, base / f'{identity[0]}_{identity[1]}')
    return records, features


def qualify_card(plan, inputs, card, base, *, ledger, backend_factory, close_backend, boundary):
    slots = [slot for slot in qualification_slots(inputs) if slot['card'] == card]
    requests = {row['request_id']: row for role in ('qualification', 'warmup')
                for row in inputs[role + '_requests']}
    records, features = _run_slots(plan, requests, slots, base, ledger=ledger,
        backend_factory=backend_factory, close_backend=close_backend, boundary=boundary,
        category={'generation': 1}, prepare_features=True)
    result = dict(status='passed_local_card', card=card, executions=len(records),
                  teacher_receipts=len(features), cross_card_gate_pending=True)
    write_json(Path(base) / 'RESULT.json', result)
    return result


def merge_qualification(plan, inputs, card_roots, destination):
    """Require all 60 slots including cross-card and capture-on/off parity."""
    expected = qualification_slots(inputs)
    records, by_input, warmups = {}, {}, {}
    for slot in expected:
        root = Path(card_roots[slot['card']])
        work = json.loads((root / 'WORK.json').read_text())
        if work[slot['slot_id']]['status'] != 'valid':
            raise ValueError('Qualification slot is not valid: ' + slot['slot_id'])
        record = json.loads((root / 'records' / (slot['slot_id'] + '.json')).read_text())
        key = (slot['request_id'], slot['max_new_tokens'])
        for previous in by_input.get(key, []):
            _compare(record, previous,
                     compare_counts=record['checkpoint_id'] == previous['checkpoint_id'])
        by_input.setdefault(key, []).append(record)
        records[slot['slot_id']] = record
        if slot['phase'] == 'warmup' and slot['method'] == 'AR':
            request = next(row for row in inputs['warmup_requests']
                           if row['request_id'] == slot['request_id'])
            warmups[slot['request_id']] = make_reference(plan, request, record)
    result = dict(status='passed_initial_serving', executions=60,
        publication_generations_remaining=4,
        physical_cards=plan['devices'], work_slots_digest=digest(expected),
        record_digest=digest(records), cross_card_parity=True,
        capture_on_off_parity=True, short_caps=[1, 17], normal_cap=2048,
        warmup_canonicals=warmups, execution_commit=plan['execution_commit'])
    write_json(destination, result)
    return result


def panel_precheck(plan, inputs, card, base, *, ledger, backend_factory, close_backend,
                   boundary, shard_count=2):
    """One AR canonical and one F parity generation per assigned sealed input."""
    if shard_count not in (1, 2) or card not in ('A', 'B') or (shard_count == 1 and card != 'A'):
        raise ValueError('Use one qualified serving card or the qualified two-card split')
    panel = inputs['evaluation_requests']
    if len(panel) != 1024:
        raise ValueError('The AR/F precheck requires the complete sealed 1,024-input panel')
    selected = [row for index, row in enumerate(panel)
                if shard_count == 1 or index % 2 == int(card == 'B')]
    slots = [dict(slot_id=f'{card}_{method}_precheck_{row["request_index"]}',
                 card=card, method=method, phase='panel_precheck',
                 request_id=row['request_id'], request_index=row['request_index'],
                 max_new_tokens=2048, repeat=0, collect_features=False)
             for method in ('AR', 'F') for row in selected]
    records, _ = _run_slots(plan, {row['request_id']: row for row in selected}, slots, base,
        ledger=ledger, backend_factory=backend_factory, close_backend=close_backend,
        boundary=boundary, category=lambda slot: {('canonical_generation' if slot['method'] == 'AR'
                                                 else 'frozen_precheck_generation'): 1})
    requests = {row['request_id']: row for row in selected}
    canonicals = {slot['request_id']: make_reference(plan, requests[slot['request_id']],
                                                    records[slot['slot_id']])
        for slot in slots if slot['method'] == 'AR'}
    write_json(Path(base) / 'CANONICALS.json', canonicals)
    result = dict(status='passed', card=card, shard_count=shard_count,
                  requests=len(selected), executions=len(slots), canonical_digest=digest(canonicals))
    write_json(Path(base) / 'RESULT.json', result)
    return result


def merge_panel(inputs, shard_roots, destination):
    canonicals = {}
    for root in shard_roots:
        root = Path(root)
        result = json.loads((root / 'RESULT.json').read_text())
        if result['status'] != 'passed':
            raise ValueError('An AR/F precheck shard has not passed')
        shard = json.loads((root / 'CANONICALS.json').read_text())
        if canonicals.keys() & shard.keys():
            raise ValueError('Repeated canonical request across shards')
        canonicals.update(shard)
    expected = {row['request_id'] for row in inputs['evaluation_requests']}
    if set(canonicals) != expected or len(canonicals) != 1024:
        raise ValueError('Canonical coverage differs from the sealed evaluation panel')
    write_json(destination, canonicals)
    return dict(status='passed', canonicals=1024, parity_generations=1024,
                canonical_digest=digest(canonicals))


def make_reference(plan, request, record):
    reference = dict(request_id=request['request_id'], prompt_ids=record['prompt_ids'],
        response_ids=record['response_ids'], finish_reason=record['finish_reason'],
        stop_token_ids=record['stop_token_ids'], max_new_tokens=record['max_new_tokens'],
        reference_kind='target_only_AR', verified=True,
        provenance=plan['canonical_provenance'], execution_commit=plan['execution_commit'],
        source_record_id=record['record_id'], source_checkpoint=record['checkpoint_id'])
    reference['binding_digest'] = digest(reference)
    return reference


def publication_gate(plan, inputs, initial_card_root, export, base, *, ledger,
                     backend_factory, close_backend, boundary):
    """Four included F checks: trained export, then reload of that same export."""
    from experiments.fresh_confirmation.protocol import publication_errors, weight_check_errors

    base = Path(base)
    base.mkdir(parents=True, exist_ok=False)
    chosen_ids = [slot['request_id'] for slot in qualification_slots(inputs)
                  if slot['card'] == 'B' and slot['method'] == 'AR']
    requests = {row['request_id']: row for row in inputs['qualification_requests']}
    references = {}
    for slot in qualification_slots(inputs):
        if slot['card'] == 'A' and slot['method'] == 'AR' and slot['phase'] == 'reserved':
            path = Path(initial_card_root) / 'records' / (slot['slot_id'] + '.json')
            references[slot['request_id']] = json.loads(path.read_text())
    work = {f'publication_{repeat}_{index}': dict(status='unattempted')
            for repeat in range(2) for index in range(2)}
    write_json(base / 'WORK.json', work)
    backend, records = None, {}
    try:
        boundary()
        backend = backend_factory('F', plan['devices']['A'], base / 'engine')
        for repeat in range(2):
            paths = [export] if repeat == 0 else [plan['checkpoints']['F'], export]
            for ordinal, path in enumerate(paths):
                boundary()
                receipt = backend.publish(path)
                errors = publication_errors(receipt, str(path))
                write_json(base / f'PUBLICATION_{repeat}_{ordinal}.json', dict(receipt=receipt, errors=errors))
                if errors:
                    raise ValueError('Qualification export publication failed: ' + repr(errors))
                weights = backend.check_weights(path)
                errors = weight_check_errors(weights, str(path))
                write_json(base / f'WEIGHTS_{repeat}_{ordinal}.json', dict(receipt=weights, errors=errors))
                if errors:
                    raise ValueError('Qualification export tensor reload failed: ' + repr(errors))
            for index, request_id in enumerate(chosen_ids):
                boundary()
                request = requests[request_id]
                label = f'publication_{repeat}_{index}'
                slot = dict(slot_id=label, method='F', max_new_tokens=2048, collect_features=False)
                path = base / (label + '.json')
                work[label]['status'] = 'started'
                write_json(base / 'WORK.json', work)
                try:
                    record = ledger.call({'generation': 1}, label, backend.generate,
                        request['prompt_ids'], max_new_tokens=2048, request_id=label,
                        session_id=plan['session'], request_index=request['request_index'],
                        purpose='evaluation', save_to=path)
                    write_json(path, record)
                    work[label]['status'] = 'completed'
                    write_json(base / 'WORK.json', work)
                    errors = _record_errors(record, request, slot, str(export))
                    try:
                        _compare(record, references[request_id], compare_counts=False)
                        if repeat:
                            _compare(record, records[request_id], compare_counts=True)
                    except (ValueError, KeyError, TypeError) as error:
                        errors.append(str(error))
                    work[label].update(status='invalid' if errors else 'valid', errors=errors)
                    write_json(base / 'WORK.json', work)
                    if errors:
                        raise ValueError('Publication output parity failed: ' + repr(errors))
                    records[request_id] = record
                except BaseException as error:
                    if work[label]['status'] == 'started':
                        work[label]['status'] = 'failed_return_unknown'
                    work[label]['error'] = repr(error)
                    write_json(base / 'WORK.json', work)
                    raise
    finally:
        if backend is not None:
            close_backend(backend, base / 'engine')
    result = dict(status='passed', executions=4, export=str(export),
                  exact_output_and_reload_counts=True, no_additional_optimizer_steps=True)
    write_json(base / 'RESULT.json', result)
    return result
