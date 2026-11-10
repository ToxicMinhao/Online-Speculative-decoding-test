"""One first-transition experiment per sealed request, with no training.

The backend supplies native walks and reconstructs source states. This module
owns qualification, exact comparisons, the four-cell table and stop handling.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from online_feedback.storage import write_json
from .analysis import find_transition, request_row, summarize, write_tables, render_report
from .budget import BudgetExceeded
from .contract import CHECKPOINTS, DATASETS, neutral_boundaries
from .data import check_receipt, read

# These are exact scientific trace fields. Timing and allocation addresses are
# deliberately not part of token/progress parity.
ROUND_FIELDS = ('anchor_token', 'output_start', 'draft_tokens', 'target_argmax',
                'committed_tokens', 'committed_progress', 'finished', 'positions')


def compare_walk(actual, expected, *, start_round=0):
    """Same-checkpoint reconstruction must preserve every proposal, not just output."""
    errors = []
    for field in ('prompt_ids', 'response_ids', 'finish_reason', 'max_new_tokens', 'stop_token_ids'):
        if actual.get(field) != expected.get(field):
            errors.append(field)
    observed = actual['metadata']['trace']
    wanted = expected['metadata']['trace'][start_round:]
    if len(observed) != len(wanted) or actual['metadata']['verification_calls'] != len(wanted):
        errors.append('verification_calls')
    for index, (left, right) in enumerate(zip(observed, wanted), start=start_round+1):
        for field in ROUND_FIELDS:
            if left[field] != right[field]:
                errors.append(f'round_{index}.{field}')
    if errors:
        raise ValueError('Native reconstruction mismatch: ' + ', '.join(errors[:24]))
    return dict(status='passed', suffix_rounds=len(wanted), all_proposals_exact=True,
                committed_progress_exact=True, tokens_and_termination_exact=True)


def check_crossed(result, canonical, source_record, source_round):
    record = result['record']
    offset = source_record['metadata']['trace'][source_round-1]['output_start']
    offset += source_record['metadata']['trace'][source_round-1]['committed_progress'] + 1
    trace = record['metadata']['trace']
    valid = (record['response_ids'] == canonical['response_ids']
             and record['finish_reason'] == canonical['finish_reason']
             and record['max_new_tokens'] == source_record['max_new_tokens']
             and record['prompt_ids'] == source_record['prompt_ids']
             and record['stop_token_ids'] == source_record['stop_token_ids'])
    position = offset - 1
    for round_index, block in enumerate(trace):
        progress = block['committed_progress']
        if (type(progress) is not int or not 1 <= progress <= 16
                or block['finished'] != (round_index == len(trace)-1)
                or block['output_start'] != position
                or block['anchor_token'] != canonical['response_ids'][position]
                or block['committed_tokens'] != canonical['response_ids'][position+1:position+progress+1]):
            valid = False
        position += progress
    valid = valid and position == len(canonical['response_ids'])-1
    calls = record['metadata']['verification_calls']
    valid = (valid and calls == len(trace) and result['verification_calls'] == calls
             and (calls >= 1 if offset < len(canonical['response_ids']) else calls == 0))
    if not valid:
        raise ValueError('Crossed suffix differs from canonical tokens, original budget or native progress')
    return dict(status='completed', verification_calls=calls,
                output_matches_canonical=True, termination_matches=True)


def counted(budget, kind, function, **metadata):
    ticket = budget.attempt(kind, **metadata)
    try:
        result = function()
    except BaseException as error:
        budget.fail(ticket, error)
        raise
    budget.complete(ticket)
    return result


def load_originals(item, *, local=False):
    originals = {}
    for checkpoint in CHECKPOINTS:
        binding = item['native'][checkpoint]
        field = 'local_path' if local else 'path'
        check_receipt(binding, field)
        originals[checkpoint] = read(binding[field])
    return originals


def run_campaign(plan, backend, budget, output, *, local_records=False):
    """Return completed or incomplete evidence; never retry or change the sample."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rows = [request_row(r['dataset'], r['request_id'], r['selection_rank']) for r in plan['requests']]
    checks = []
    gates = {dataset: False for dataset in DATASETS}
    status = dict(status='running', completed_requests=0, qualified_datasets=gates,
                  failure=None, work=budget.summary(), new_optimizer_steps=0)
    started = time.monotonic()
    # Populate original endpoint differences for all selected IDs even if the
    # native work later stops. Crossed missing cells remain explicit.
    originals = []
    try:
        for index, item in enumerate(plan['requests']):
            pair = load_originals(item, local=local_records)
            originals.append(pair)
            rows[index] = request_row(item['dataset'], item['request_id'], item['selection_rank'],
                                      pair['KPLUS4'], pair['KPLUS6'])
        write_json(root/'STATUS.json', status)
        for index, item in enumerate(plan['requests']):
            budget.check_time()
            dataset, request_id = item['dataset'], item['request_id']
            pair = originals[index]
            transition = find_transition(pair['KPLUS4'], pair['KPLUS6'])
            request_root = root/'raw'/dataset/f"{item['selection_rank']:03d}"
            request_root.mkdir(parents=True)
            walks, handles = {}, {}
            for checkpoint in CHECKPOINTS:
                retained = pair[checkpoint]
                boundaries = neutral_boundaries([
                    {'terminal': row['finished']} for row in retained['metadata']['trace']
                ]) if item['selection_rank'] == 0 else []
                if item['selection_rank'] == 0 and not boundaries:
                    raise RuntimeError(f'No nonterminal neutral boundary for {dataset}/{checkpoint}')
                capture = set(boundaries)
                divergence = transition['divergence_round']
                if divergence is not None and not retained['metadata']['trace'][divergence-1]['finished']:
                    capture.add(divergence)
                checkpoint_path = plan['checkpoints'][dataset][checkpoint]['export']
                request = dict(item['request'], canonical=item['canonical'])
                result = counted(budget, 'original_completion',
                    lambda: backend.native_walk(request, checkpoint_path, sorted(capture)),
                    dataset=dataset, request_id=request_id, checkpoint=checkpoint)
                write_json(request_root/f'{checkpoint}_native.json', result['record'])
                parity = compare_walk(result['record'], retained)
                walks[checkpoint] = result['record']
                handles[checkpoint] = result['successors']
                checks.append(dict(dataset=dataset,request_id=request_id,checkpoint=checkpoint,
                                   kind='original_walk',**parity))
                for boundary in boundaries:
                    handle = handles[checkpoint][str(boundary)]
                    suffix = counted(budget,'neutral_suffix',
                        lambda: backend.continue_from(handle,checkpoint_path,'neutral_suffix'),
                        dataset=dataset,request_id=request_id,checkpoint=checkpoint,boundary=boundary)
                    write_json(request_root/f'{checkpoint}_neutral_{boundary}.json',suffix)
                    parity = compare_walk(suffix['record'],retained,start_round=boundary)
                    reconstruction = suffix['reconstruction']
                    if reconstruction.get('status') != 'passed':
                        raise ValueError('Neutral constructor did not validate native state reconstruction')
                    checks.append(dict(dataset=dataset,request_id=request_id,checkpoint=checkpoint,
                        kind='neutral_suffix',boundary=boundary,reconstruction=reconstruction,**parity))
            if item['selection_rank'] == 0:
                gates[dataset] = True
            if not gates[dataset]:
                raise RuntimeError('Crossed replay requires this dataset\'s neutral gate')
            crossed = {}
            for name, source, destination, terminal in (
                ('C46','KPLUS6','KPLUS4',transition.get('terminal6')),
                ('C64','KPLUS4','KPLUS6',transition.get('terminal4')),
            ):
                if transition['divergence_round'] is None or terminal:
                    continue
                handle = handles[source][str(transition['divergence_round'])]
                checkpoint_path = plan['checkpoints'][dataset][destination]['export']
                suffix = counted(budget,'crossed_suffix',
                    lambda: backend.continue_from(handle,checkpoint_path,'crossed_suffix'),
                    dataset=dataset,request_id=request_id,cell=name)
                write_json(request_root/f'{name}.json',suffix)
                crossed[name] = check_crossed(suffix,item['canonical'],walks[source],transition['divergence_round'])
                rows[index] = request_row(dataset,request_id,item['selection_rank'],
                    walks['KPLUS4'],walks['KPLUS6'],crossed=crossed)
            rows[index] = request_row(dataset,request_id,item['selection_rank'],
                                     walks['KPLUS4'],walks['KPLUS6'],crossed=crossed)
            backend.release_request()
            status['completed_requests'] += 1
            status['work'] = budget.summary()
            write_json(root/'ROWS.json',rows)
            write_json(root/'NATIVE_CHECKS.json',checks)
            write_json(root/'STATUS.json',status)
        status['status'] = 'completed'
    except BaseException as error:
        status['status'] = 'incomplete' if isinstance(error,BudgetExceeded) else 'technically_incomplete'
        status['failure'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        status.update(work=budget.summary(), wall_s=time.monotonic()-started)
        write_json(root/'ROWS.json',rows)
        write_json(root/'NATIVE_CHECKS.json',checks)
        write_json(root/'STATUS.json',status)
        summaries = {d:summarize([row for row in rows if row["dataset"] == d],dataset=d) for d in DATASETS}
        if status['status'] != 'completed':
            for summary in summaries.values():
                summary['execution_status'] = status['status']
                summary['status'] = status['status']
                summary.pop('strata', None)
                summary['recommendation'] = 'INCOMPLETE; no go/no-go conclusion from selectively completed work'
        write_tables(root/'report', rows, summaries)
        (root/'report/REPORT.md').write_text(render_report(summaries, execution_commit=plan.get('execution_commit', 'UNSEALED'), work=status['work']))
    return status
