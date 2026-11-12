"""Read-only native-state/accounting audit; no model or production-code imports.

The author implemented the native observer. This is a separate ledger/evidence
reduction, with authorship overlap, not an independent GPU reproduction.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


FIELDS = ('output_start', 'anchor_token', 'draft_tokens', 'target_argmax',
          'verify_size', 'accepted_draft_tokens', 'native_accepted_draft_tokens',
          'raw_accepted_draft_tokens_before_stop', 'committed_progress',
          'committed_tokens', 'positions', 'stop_clipped', 'finished')


def read(path):
    return json.loads(Path(path).read_text())


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []

    def equal(self, name, actual, expected):
        self.count += 1
        if actual != expected:
            self.failures.append(dict(name=name, actual=actual, expected=expected))

    def true(self, name, value):
        self.equal(name, bool(value), True)


def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def audit(folder):
    folder = Path(folder)
    checks = Checks()
    plan, result, budget = (read(folder/name) for name in ('PLAN.json', 'RESULT.json', 'BUDGET.json'))
    preflight, postflight = (read(folder/name) for name in ('PREFLIGHT.json', 'POSTFLIGHT.json'))
    seal = read(folder/'PRE_K6_SEAL.json')
    panel = plan['requests']
    requests = len(panel)
    starts = read(folder/'STARTS.json')
    outcomes = read(folder/'OUTCOMES.json')
    comparisons = read(folder/'PRE_K6.json')
    ledger = [json.loads(line) for line in (folder/'WORK.jsonl').read_text().splitlines() if line]
    checks.equal('result_completed', result['status'], 'completed')
    checks.equal('preflight_passed', preflight['status'], 'passed')
    checks.equal('postflight_passed', postflight['status'], 'passed')
    checks.equal('source_immutable', postflight['source_unchanged'], True)
    checks.equal('inputs_checkpoints_immutable', postflight['inputs_and_checkpoints_unchanged'], True)
    checks.equal('execution_commit', result['execution_commit'], plan['execution_commit'])
    checks.equal('preflight_execution_commit', preflight['execution_commit'], plan['execution_commit'])
    checks.equal('optimizer_steps_zero', result['optimizer_steps'], 0)
    checks.equal('backward_calls_zero', result['backward_calls'], 0)
    checks.equal('timed_endpoints_zero', result['timed_endpoints'], 0)
    checks.equal('physical_gpu_index', result['physical_gpu_index'], plan['physical_gpu_index'])
    checks.equal('physical_gpu_uuid', result['physical_gpu_uuid'], plan['physical_gpu_uuid'])
    selected = [row for row in preflight['gpu_inventory'] if row['index'] == plan['physical_gpu_index']]
    checks.equal('one_selected_physical_gpu', len(selected), 1)
    checks.equal('preflight_physical_gpu_uuid', selected[0]['uuid'], plan['physical_gpu_uuid'])
    for arm, config in plan['checkpoints'].items():
        path = str(Path(config['export'])/'model.safetensors')
        checks.equal('authenticated_weight:'+arm, preflight['checked_files'][path], config['weights_identity'])
    for name, expected in seal['files'].items():
        checks.equal('pre_K6_sealed_file:'+name, identity(folder/name), expected)
    checks.equal('K6_unloaded_at_seal', seal['k6_loaded'], False)

    pending, attempted, completed, failed = {}, Counter(), Counter(), Counter()
    first_outcome_index = None
    first_outcome_completed_before = None
    startup = Counter()
    nonstartup = Counter()
    occupied = []
    gpu_end = []
    observed_gpu_seconds = []
    publication_rows = []
    for index, row in enumerate(ledger):
        kind = row['kind']
        if 'gpu_seconds' in row and kind != 'contract':
            observed_gpu_seconds.append(row['gpu_seconds'])
        if kind == 'gpu_start':
            occupied.append(row['occupied_gpus'])
        elif kind == 'gpu_end':
            gpu_end.append(row['gpu_seconds'])
        status = row.get('status')
        if status == 'attempted':
            ticket = row['ticket']
            checks.true('unique_ticket:'+str(ticket), ticket not in pending)
            pending[ticket] = kind
            attempted[kind] += 1
            metadata = row.get('metadata', {})
            (startup if metadata.get('startup', False) else nonstartup)[kind] += 1
            if kind == 'model_load' and metadata.get('operation') == 'native_draft_publication':
                publication_rows.append(index)
            if kind == 'outcome_block':
                checks.equal('outcome_seal:'+str(ticket), metadata['pre_k6_sha256'], seal['files']['PRE_K6.json']['sha256'])
                if first_outcome_index is None:
                    first_outcome_index = index
                    first_outcome_completed_before = dict(completed)
        elif status in ('completed', 'failed'):
            ticket = row['ticket']
            checks.equal('ticket_kind:'+str(ticket), pending.pop(ticket, None), kind)
            (completed if status == 'completed' else failed)[kind] += 1
    checks.equal('single_GPU_interval', occupied, [1])
    checks.equal('one_GPU_end', len(gpu_end), 1)
    checks.true('monotone_gpu_clock', all(a <= b for a,b in zip(observed_gpu_seconds, observed_gpu_seconds[1:])))
    checks.equal('final_pending_empty', pending, {})
    checks.equal('final_failed_empty', dict(failed), {})
    checks.equal('final_unknown_empty', budget['unknown'], {})
    checks.equal('ledger_attempts', dict(attempted), budget['attempted'])
    checks.equal('ledger_completions', dict(completed), budget['completed'])
    checks.equal('ledger_failures', dict(failed), budget['failed'])
    checks.equal('result_budget', result['work'], budget)
    checks.true('within_three_hour_cap', budget['gpu_seconds'] < 10800)
    checks.equal('three_hour_contract', budget['gpu_seconds_limit'], 10800)
    checks.equal('end_clock_matches_summary', gpu_end[0], budget['gpu_seconds'])
    prohibited = ('optimizer_step', 'backward', 'feedback_generation', 'timing_sweep', 'teacher_preparation')
    for kind in prohibited:
        checks.equal('forbidden_work:'+kind, attempted[kind], 0)

    totals = Counter()
    for item in panel:
        request_id = item['request_id']
        current = folder/'requests'/request_id
        parent = read(current/'PARENT.json')
        parity = read(current/'PARENT_PARITY.json')
        compact = read(current/'COMPACT_STATES.json')
        margins = read(current/'MARGINS.json')
        trace = parent['metadata']['trace']
        totals['requests'] += 1
        totals['parent_rounds'] += len(trace)
        checks.equal(request_id+':parent_parity', parity['status'], 'passed')
        checks.equal(request_id+':archived_trace_available', parity['archived_trace_available'], True)
        checks.equal(request_id+':parent_parity_count', len(parity['block_parity']), len(trace))
        checks.true(request_id+':all_archived_block_parities', all(row['status']=='passed' for row in parity['block_parity']))
        checks.equal(request_id+':native_verification_calls', parent['metadata']['verification_calls'], len(trace))
        checks.equal(request_id+':head_observation_calls', margins['observed_head_calls'], len(trace))
        checks.equal(request_id+':compaction_passed', compact['status'], 'passed')
        checks.equal(request_id+':compaction_states', compact['states'], len(trace))
        checks.equal(request_id+':cache_prefix_exact', compact['cache_prefixes_exact'], True)
        checks.true(request_id+':compaction_nonexpansion', compact['after_bytes'] <= compact['before_bytes'])
        totals['compacted_cache_bytes'] += compact['after_bytes']
        totals['cache_prefix_checks'] += compact['prefixes_checked']
        totals['metadata_bytes'] += compact['metadata_offload']['tensor_bytes']
        totals['GPU_metadata_bytes_offloaded'] += compact['metadata_offload']['device_bytes_released']
        for ordinal, block in enumerate(trace):
            for arm in ('K5', 'K4'):
                replay = read(current/f'{ordinal:04d}_{arm}.json')
                checks.equal(f'{request_id}:{ordinal}:{arm}:status', replay['status'], 'completed')
                checks.equal(f'{request_id}:{ordinal}:{arm}:one_block', len(replay['blocks']), 1)
                checks.equal(f'{request_id}:{ordinal}:{arm}:original_cap', replay['original_cap'], parent['max_new_tokens'])
                checks.equal(f'{request_id}:{ordinal}:{arm}:cap_unchanged', replay['cap_unchanged'], True)
                checks.equal(f'{request_id}:{ordinal}:{arm}:canonical', replay['output_matches_canonical_prefix'], True)
                reconstruction = replay['reconstruction']
                checks.equal(f'{request_id}:{ordinal}:{arm}:reconstruction', reconstruction['status'], 'passed')
                checks.equal(f'{request_id}:{ordinal}:{arm}:target_cache', reconstruction['target_prefix_cache_exact'], True)
                checks.equal(f'{request_id}:{ordinal}:{arm}:draft_rebuilt', reconstruction['destination_draft_rebuilt'], True)
                checks.equal(f'{request_id}:{ordinal}:{arm}:no_transplant', reconstruction['source_draft_cache_transplanted'], False)
                if arm == 'K5':
                    checks.equal(f'{request_id}:{ordinal}:neutral_cache', reconstruction['neutral_draft_prefix_cache_exact'], True)
                    for name in FIELDS:
                        checks.equal(f'{request_id}:{ordinal}:neutral:'+name, replay['blocks'][0][name], block[name])
                    totals['neutral_blocks'] += 1
                else:
                    totals['historical_blocks'] += 1
    for row in outcomes:
        request = next(item for item in panel if item['source_id']==row['source_id'])
        native = read(folder/'requests'/request['request_id']/f"K6_{row['start_offset']:05d}.json")
        checks.equal(f"K6:{row['source_id']}:{row['start_offset']}:reconstruction", native['reconstruction']['status'], 'passed')
        checks.equal(f"K6:{row['source_id']}:{row['start_offset']}:target_cache", native['reconstruction']['target_prefix_cache_exact'], True)
        checks.equal(f"K6:{row['source_id']}:{row['start_offset']}:no_transplant", native['reconstruction']['source_draft_cache_transplanted'], False)
        checks.equal(f"K6:{row['source_id']}:{row['start_offset']}:canonical", native['output_matches_canonical_prefix'], True)
        checks.equal(f"K6:{row['source_id']}:{row['start_offset']}:cap", native['cap_unchanged'], True)
    checks.equal('all_requests_completed', totals['requests'], requests)
    checks.equal('all_starts_count', totals['parent_rounds'], len(starts))
    expected = dict(parent_walk=requests, neutral_block=totals['parent_rounds'],
                    historical_block=totals['parent_rounds'], outcome_block=len(outcomes))
    for kind, value in expected.items():
        checks.equal('work_count:'+kind, completed[kind], value)
    required = {(r['source_id'], r['start_offset']) for r in comparisons['required_k6_starts']}
    observed = {(r['source_id'], r['start_offset']) for r in outcomes}
    checks.equal('unique_K6_outcomes', len(outcomes), len(observed))
    checks.equal('all_required_K6_starts', sorted(observed), sorted(required))
    checks.equal('initialization_and_publication_count', completed['model_load'],
                 1 + 2 * requests + int(bool(required)))
    nonstartup_blocks = totals['parent_rounds'] * 3 + len(outcomes)
    checks.equal('nonstartup_draft_blocks', nonstartup['draft_block'], nonstartup_blocks)
    checks.equal('nonstartup_target_calls', nonstartup['target_call'], nonstartup_blocks + requests)
    if first_outcome_index is not None:
        checks.equal('all_parents_before_K6', first_outcome_completed_before.get('parent_walk',0), requests)
        checks.equal('all_neutrals_before_K6', first_outcome_completed_before.get('neutral_block',0), totals['parent_rounds'])
        checks.equal('all_K4_before_K6', first_outcome_completed_before.get('historical_block',0), totals['parent_rounds'])
        after = [index for index in publication_rows if index > first_outcome_index]
        checks.equal('one_publication_after_outcome_phase_begins', len(after), 1)
        checks.true('first_K6_publication_inside_outcome_admission', bool(after))
    return dict(status='passed' if not checks.failures else 'failed', checks=checks.count,
        failures=checks.failures, execution_commit=plan['execution_commit'],
        ledger_counts=dict(completed), startup_counts=dict(startup), nonstartup_counts=dict(nonstartup),
        native_totals=dict(totals), gpu_seconds=budget['gpu_seconds'],
        physical_gpu_index=plan['physical_gpu_index'], physical_gpu_uuid=plan['physical_gpu_uuid'],
        chronology=dict(first_outcome_ledger_row=first_outcome_index,
            completed_before_first_outcome=first_outcome_completed_before,
            seal_file_sha256=seal['files']['PRE_K6.json']['sha256'],
            evidence_scope='Sealed-hash outcome admission follows every K4/K5 completion. The pinned runner binds the sole subsequent publication to K6; publication ledger rows do not independently timestamp the checkpoint path.'),
        report_scope='Native completion and accounting only; this audit does not assert that the report-rendering wrapper exited successfully.',
        disclosure='Separate read-only ledger/native reduction by the native-observer implementer; authorship overlap. No independent GPU replication or new model work.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    result=audit(args.campaign)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('status','checks','gpu_seconds','ledger_counts')}))
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':
    raise SystemExit(main())
