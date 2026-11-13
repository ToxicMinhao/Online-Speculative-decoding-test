"""K6 natural histories, sealed K5→K6 tags, then same-start endpoint replay.

No training or timing happens here. PR43's native cache reconstruction is reused
unchanged, including opening starts, bitwise compaction and neutral parity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from online_feedback.storage import write_json
from experiments.verification_transition.data import identity, read
from experiments.verification_transition.budget import WorkBudget, BudgetService
from experiments.lost_repair.run import block_observation, counted
from experiments.longer_run_headroom.pairs import neutral_parity
from experiments.repair_history_margin.run import validate_parent, validate_commits
from .evaluation import METHODS


def outcome_row(start, observation, block, canonical, method):
    """Keep exact-slot relapse, earlier failure and direct agreement distinct."""
    a = observation['accepted']
    j = start['repair_slot']
    old = start['a6']
    return dict(request_id=start['request_id'], canonical_start=start['canonical_start'],
        method=method, accepted=a, parent_accepted=old,
        valid_proposals=observation['valid_proposals'], repair_slot=j,
        genuine_rejection=observation['genuine_rejection'],
        exact_slot_relapse=(a == j-1 and observation['genuine_rejection']) if j else None,
        earlier_break=a < j-1 if j else None,
        accepts_through=a >= j if j else None,
        direct_canonical_agreement=(block['draft_tokens'][j-1] == canonical[j-1]) if j else None,
        parent_first_blocker=old+1 if start['genuine_k6_rejection'] else None,
        passes_parent_first_blocker=a > old if start['genuine_k6_rejection'] else None,
        acquired_prefix_occurrences=max(0, a-old), lost_prefix_occurrences=max(0, old-a),
        net_prefix_occurrences=a-old, valid=True)


def run(plan, exports, references, output):
    from experiments.repair_history_margin.native_worker import MarginBackend

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    panel = read(plan['panel_manifest'])['requests']
    by_id = {row['request_id']:row for row in panel}
    requests = [by_id[rid] for rid in plan['path_request_ids']]
    if len(requests) != min(64, len(panel)):
        raise ValueError('Fixed-start subset differs from the pre-training seal')
    budget = WorkBudget(output/'WORK.jsonl', gpu_seconds=plan['fixed_start_seconds_limit'],
        limits=dict(model_load=1, parent_walk=len(requests),
                    neutral_block=len(requests)*2048, historical_block=len(requests)*2048,
                    outcome_block=3*len(requests)*2048, draft_block=2_000_000))
    service = BudgetService(budget)
    budget.start_gpu(1)
    starts, outcomes, states, captures = [], [], {}, {}
    backend = None
    status = dict(status='running', phase='pre_endpoint', parent_requests=0,
                  planned_parent_requests=len(requests), endpoint_blocks=0)
    began = time.perf_counter()
    try:
        backend = counted(budget, 'model_load', lambda: MarginBackend(plan['target_path'],
            plan['checkpoints']['K6']['export'], worker_python=plan['worker_python'],
            output_root=output/'native', budget_connection=service.connection))
        for supplied in requests:
            request = dict(supplied, canonical=references[supplied['request_id']])
            rid = request['request_id']
            folder = output/'requests'/rid
            folder.mkdir(parents=True)
            record, captured = counted(budget, 'parent_walk', lambda: backend.capture(
                request, plan['checkpoints']['K6']['export'], folder), request_id=rid)
            write_json(folder/'PARENT_K6.json', record)
            canonical = dict(references[rid], max_new_tokens=2048,
                             stop_token_ids=request['stop_token_ids'])
            parity = validate_parent(record, canonical, None)
            write_json(folder/'PARENT_PARITY.json', parity)
            if parity['status'] != 'passed' or len(captured) != len(record['metadata']['trace']):
                raise ValueError('K6 capture differs from canonical output or complete starts')
            write_json(folder/'COMPACT.json', backend.compact_request(rid))
            captures[rid] = record
            request_starts = []
            for ordinal, parent in enumerate(record['metadata']['trace']):
                score = block_observation(parent, record)
                row = dict(request_id=rid, canonical_start=parent['output_start'],
                    round_ordinal=ordinal, a6=score['accepted'],
                    valid_proposals=score['valid_proposals'],
                    genuine_k6_rejection=score['genuine_rejection'], repair_slot=None)
                states[rid, parent['output_start']] = captured[str(ordinal)]
                request_starts.append(row)
            for checkpoint in ('K6', 'K5'):
                for row, parent in zip(request_starts, record['metadata']['trace']):
                    observed = counted(budget, 'neutral_block' if checkpoint == 'K6' else 'historical_block',
                        lambda: backend.one_block(states[rid, row['canonical_start']],
                                                   plan['checkpoints'][checkpoint]['export']),
                        request_id=rid, canonical_start=row['canonical_start'], method=checkpoint)
                    write_json(folder/f"{row['round_ordinal']:04d}_{checkpoint}.json", observed)
                    block = observed['blocks'][0]
                    validate_commits(block, record)
                    if (block['output_start'], block['anchor_token'], block['positions']) != (
                            parent['output_start'], parent['anchor_token'], parent['positions']):
                        raise ValueError('Fixed-start conditioning changed')
                    score = block_observation(block, record)
                    if checkpoint == 'K6':
                        check = neutral_parity(parent, block)
                        write_json(folder/f"{row['round_ordinal']:04d}_PARITY.json", check)
                        if check['status'] != 'passed':
                            raise ValueError('K6 neutral proposals/commits/clipping differ')
                        row['parity_valid'] = True
                    else:
                        row['a5'] = score['accepted']
                        row['genuine_k5_rejection'] = score['genuine_rejection']
                        if score['genuine_rejection'] and row['a6'] > row['a5']:
                            row['repair_slot'] = row['a5']+1
            starts.extend(request_starts)
            status['parent_requests'] += 1
            write_json(output/'STATUS.json', status)
        write_json(output/'PRE_ENDPOINT_STARTS.json', starts)
        seal = identity(output/'PRE_ENDPOINT_STARTS.json')
        write_json(output/'TAG_SEAL.json', dict(status='sealed', identity=seal,
            endpoint_loaded=False, requests=len(requests), starts=len(starts),
            repairs=sum(row['repair_slot'] is not None for row in starts)))
        status['phase'] = 'endpoint_replay'
        # All pre-update histories and repair labels are now immutable.
        for method in METHODS:
            if identity(output/'PRE_ENDPOINT_STARTS.json') != seal:
                raise ValueError('Held-out repair labels changed')
            for row in starts:
                rid, offset = row['request_id'], row['canonical_start']
                observed = counted(budget, 'outcome_block', lambda: backend.one_block(
                    states[rid, offset], exports[method]), method=method,
                    request_id=rid, canonical_start=offset)
                write_json(output/'requests'/rid/f'{offset:05d}_{method}.json', observed)
                block = observed['blocks'][0]
                record = captures[rid]
                parent = record['metadata']['trace'][row['round_ordinal']]
                validate_commits(block, record)
                if (block['output_start'], block['anchor_token'], block['positions']) != (
                        parent['output_start'], parent['anchor_token'], parent['positions']):
                    raise ValueError('Endpoint draft was conditioned on a different prefix')
                score = block_observation(block, record)
                result = outcome_row(row, score, block, record['response_ids'][offset+1:], method)
                outcomes.append(result)
                with (output/'OUTCOMES.jsonl').open('a') as stream:
                    stream.write(json.dumps(result, allow_nan=False)+'\n')
                status['endpoint_blocks'] += 1
            write_json(output/'STATUS.json', status)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(output/'STARTS.json', starts)
        write_json(output/'OUTCOMES.json', outcomes)
        write_json(output/'STATUS.json', status)
        try:
            if backend is not None:
                backend.close()
        except BaseException as error:
            status.update(status='failed',cleanup_error=repr(error))
            raise
        finally:
            budget.end_gpu()
            status.update(wall_s=time.perf_counter()-began, work=budget.summary())
            write_json(output/'STATUS.json', status)
            service.close()
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--exports', required=True)
    parser.add_argument('--references', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(read(args.plan), read(args.exports), read(args.references), args.output)


if __name__ == '__main__':
    main()
