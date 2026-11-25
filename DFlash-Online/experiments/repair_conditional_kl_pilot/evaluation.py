"""Fixed BASE/PAIR timing on one card using the retained frozen-F canonicals."""
from collections import Counter, defaultdict
from pathlib import Path
import time

from online_feedback.storage import write_json
from experiments.kw_v1.evaluate import append_json
from experiments.fresh_confirmation.protocol import publication_errors, weight_check_errors
from experiments.opencode_native_comparison.contract import validate_requests
from experiments.opencode_native_comparison.evaluation import (
    COUNT_FIELDS, SESSION, check_reference, label_for, measure,
)

METHODS = ('BASE', 'PAIR')
REQUESTS, REPEATS, GROUP_SIZE = 128, 3, 64


def frozen_reference(value):
    """Accept the original source bundle or its authenticated frozen-F member."""
    return value['F'] if set(value) == {'AR', 'F'} else value


def evaluation_schedule(requests, *, methods=METHODS):
    if len(methods) != 2 or len(set(methods)) != 2:
        raise ValueError("Require two distinct explicit endpoint labels")
    validate_requests(requests, count=REQUESTS, role='evaluation')
    schedule = []
    for repeat in range(REPEATS):
        for group in range(REQUESTS // GROUP_SIZE):
            shift = (repeat + group) % len(methods)
            for position, method in enumerate(methods[shift:] + methods[:shift]):
                schedule.append(dict(activation=len(schedule), repeat=repeat,
                    group=group, position=position, method=method, warmups=2,
                    request_ids=[r['request_id'] for r in
                                 requests[group * GROUP_SIZE:(group + 1) * GROUP_SIZE]]))
    return schedule


def schedule_summary(schedule, *, methods=METHODS):
    coverage, first, positions = Counter(), Counter(), Counter()
    for start in range(0, len(schedule), 2):
        batch = schedule[start:start + 2]
        if (len(batch) != 2 or {row['method'] for row in batch} != set(methods)
                or batch[0]['request_ids'] != batch[1]['request_ids']
                or len(batch[0]['request_ids']) != GROUP_SIZE):
            raise ValueError('Each timing group requires the same 64 requests in both arms')
        for position, entry in enumerate(batch):
            if (entry['activation'] != start + position or entry['position'] != position
                    or entry['repeat'] != batch[0]['repeat']
                    or entry['group'] != batch[0]['group'] or entry['warmups'] != 2):
                raise ValueError('Timing activation identity changed')
            positions[entry['method'], position] += 1
            for request_id in entry['request_ids']:
                coverage[request_id, entry['repeat'], entry['method']] += 1
                if position == 0:
                    first[entry['repeat'], entry['method']] += 1
    if (len(schedule) != 12 or len(coverage) != 768
            or any(value != 1 for value in coverage.values())
            or any(first[repeat, method] != 64 for repeat in range(REPEATS) for method in methods)
            or any(positions[method, position] != 3 for method in methods for position in (0, 1))):
        raise ValueError('Fixed two-arm coverage or per-repeat position balance changed')
    return dict(activations=12, scored_generations=768, warmup_generations=24,
        groups=6, requests=128, repeats=3, group_size=64,
        first_position_requests_per_repeat={method: [first[r, method] for r in range(REPEATS)]
                                           for method in methods})


def execute(plan, root, requests, warmups, canonicals, warmup_canonicals,
            checkpoints, *, activate, boundary, ledger, methods=METHODS):
    """Use native measurement checks without changing the historical coordinator.

    ``activate(method, checkpoint, slot)`` returns a backend and publication
    receipt. ``boundary(slot)`` must enforce a quiet second card. References
    retain their original provenance; no replacement outputs are generated.
    The inherited record session is kept solely for native validator parity;
    this pilot's immutable protocol and unique root identify the experiment.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    schedule = evaluation_schedule(requests, methods=methods)
    balance = schedule_summary(schedule, methods=methods)
    if set(checkpoints) != set(methods) or len(set(map(str, checkpoints.values()))) != 2:
        raise ValueError('Require the two distinct explicitly labeled endpoint exports')
    validate_requests(warmups, count=2, role='warmup')
    all_requests = requests + warmups
    if (len({r['request_id'] for r in all_requests}) != 130
            or len({tuple(r['prompt_ids']) for r in all_requests}) != 130):
        raise ValueError('The panel and original two warmups must be disjoint')
    references = []
    for group, source in ((requests, canonicals), (warmups, warmup_canonicals)):
        if set(source) != {r['request_id'] for r in group}:
            raise ValueError('Incomplete frozen-F canonical coverage')
        selected = {key: frozen_reference(value) for key, value in source.items()}
        for request in group:
            check_reference(request, selected[request['request_id']],
                            plan['canonical_provenance'], method='F')
        references.append(selected)
    panel_references, warmup_references = references
    write_json(root / 'PROTOCOL.json', dict(schema='repair_conditional_kl_evaluation_v1',
        schedule=schedule, balance=balance, checkpoints=checkpoints,
        execution_commit=plan['execution_commit'], gpu=plan['devices']['A'],
        canonical_provenance=plan['canonical_provenance'], native_record_session=SESSION,
        canonical_policy='Both speculative arms exactly match retained frozen F',
        feedback_capture=False, initial_prefill_token_excluded_from_committed_progress=True,
        native_and_stop_clipped_acceptance_separate=True))
    write_json(root / 'REFERENCES.json', dict(panel=panel_references, warmup=warmup_references))
    write_json(root / 'INPUTS.json', dict(panel=requests, warmup=warmups))
    by_id = {row['request_id']: row for row in requests}
    grouped, checked = defaultdict(list), set()
    progress = dict(status='running', planned=792, attempted=0, returned=0,
        completed_valid=0, invalid=0, pending=None, automatic_retry=False)
    started = time.perf_counter()

    def persist():
        progress.update(unattempted=progress['planned'] - progress['attempted'],
            unknown=int(progress['pending'] is not None), wall_s=time.perf_counter() - started)
        write_json(root / 'STATUS.json', progress)

    persist()
    try:
        for entry in schedule:
            append_json(root / 'BOUNDARIES.jsonl', boundary(entry))
            method, checkpoint = entry['method'], checkpoints[entry['method']]
            backend, activation = activate(method, checkpoint, entry)
            append_json(root / 'ACTIVATIONS.jsonl', {**entry, **activation})
            if activation.get('publication') is not None:
                errors = publication_errors(activation['publication'], checkpoint)
                if errors:
                    raise ValueError('Native publication failed: ' + repr(errors))
            if method not in checked:
                receipt = backend.check_weights(checkpoint)
                errors = weight_check_errors(receipt, checkpoint)
                write_json(root / f'WEIGHTS_{method}.json', dict(receipt=receipt, errors=errors))
                if errors:
                    raise ValueError('Loaded endpoint differs: ' + repr(errors))
                checked.add(method)
            slots = [('warmup', r) for r in warmups] + [
                ('scored', by_id[key]) for key in entry['request_ids']]
            for phase, request in slots:
                slot = dict(activation=entry['activation'], repeat=entry['repeat'],
                    group=entry['group'], method=method, phase=phase,
                    request_ordinal=request['request_index'])
                label = label_for(slot)
                path = root / 'raw' / (label + '.json')
                progress['attempted'] += 1
                progress['pending'] = slot
                persist()
                record = ledger.call({phase + '_generation': 1}, label, backend.generate,
                    request['prompt_ids'], max_new_tokens=2048, request_id=label,
                    session_id=SESSION, request_index=request['request_index'],
                    purpose='evaluation', save_to=path)
                # Keep a returned invalid record before touching any validator.
                write_json(path, record)
                progress['returned'] += 1
                progress['pending'] = None
                persist()
                canonical = (warmup_references if phase == 'warmup' else panel_references)[request['request_id']]
                try:
                    row, errors = measure(record, request, checkpoint, canonical, slot)
                except (KeyError, TypeError, ValueError, IndexError, RuntimeError) as error:
                    row = dict(slot, request_id=request['request_id'], checkpoint_id=checkpoint)
                    errors = ['malformed_returned_record:' + repr(error)]
                row['physical_gpu_uuid'] = plan['devices']['A']
                previous = grouped[phase, method, request['request_id']]
                consistency = (*COUNT_FIELDS, 'finish_reason', 'checkpoint_id')
                if not errors and any(tuple(old[key] for key in consistency) !=
                                      tuple(row[key] for key in consistency) for old in previous):
                    errors.append('repeat_count_mismatch')
                row.update(valid_measurement=not errors, errors=sorted(set(errors)), raw_path=str(path))
                previous.append(row)
                append_json(root / 'MEASUREMENTS.jsonl', row)
                progress['invalid' if errors else 'completed_valid'] += 1
                persist()
                if errors:
                    raise ValueError('Pilot native validity failure: ' + repr(errors))
        append_json(root / 'BOUNDARIES.jsonl', boundary({'activation': 'finished'}))
        progress['status'] = 'completed'
    except BaseException as error:
        progress.update(status='failed', error=repr(error))
        raise
    finally:
        persist()
    return progress
