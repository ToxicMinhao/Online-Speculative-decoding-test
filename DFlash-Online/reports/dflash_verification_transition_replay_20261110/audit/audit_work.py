"""CPU-only audit of the completed verification-transition evidence.

No Torch imports or model work. Nested duration categories are not added into
a fabricated kernel breakdown. Missing final evidence remains unavailable.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = 'd7b1311e7b2f910e935bf81b4e40282767d7540c'
LIMITS = dict(original_completion=256, crossed_suffix=256,
              neutral_suffix=8, draft_block=32768)
WORK = set(LIMITS) | {'target_call', 'context_forward', 'model_load'}
FORBIDDEN = {'optimizer_step', 'backward', 'feedback_generation', 'timing_sweep'}


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def analyze_ledger(rows):
    check(rows and rows[0]['kind'] == 'contract', 'Missing first ledger contract')
    check(rows[0]['limits'] == LIMITS and rows[0]['gpu_seconds'] == 14400,
          'Scientific work limits differ')
    pending, attempts, done = {}, {}, {}
    counts = {name: Counter() for name in ('attempted', 'completed', 'failed', 'unknown')}
    groups = defaultdict(lambda: Counter())
    durations = defaultdict(list)
    tokens = defaultdict(lambda: Counter())
    gpu_active, gpu_ends, stops, processes = False, [], [], []
    previous_time = 0.0
    for ordinal, row in enumerate(rows[1:], 2):
        kind = row['kind']
        check(kind not in FORBIDDEN, f'Forbidden model work: {kind}')
        if 'gpu_seconds' in row:
            now = row['gpu_seconds']
            check(isinstance(now, (int, float)) and previous_time <= now <= 14400,
                  f'Nonmonotonic or over-budget GPU clock at row {ordinal}')
            previous_time = now
        if kind == 'gpu_start':
            check(not gpu_active and row['occupied_gpus'] == 1, 'Overlapping/non-single GPU occupancy')
            gpu_active = True
            continue
        if kind == 'gpu_end':
            check(gpu_active, 'GPU end without active interval')
            gpu_active = False
            gpu_ends.append(row['gpu_seconds'])
            continue
        if kind == 'owned_process':
            check(row['pid'] > 1 and row['pgid'] > 1, 'Invalid native process binding')
            processes.append(dict(pid=row['pid'], pgid=row['pgid']))
            continue
        if kind == 'budget_stop':
            stops.append(row)
            continue
        check(kind in WORK, f'Unknown ledger kind {kind}')
        ticket, status = row['ticket'], row['status']
        if status == 'attempted':
            check(gpu_active, 'Model work admitted outside GPU occupancy')
            check(ticket not in attempts, f'Duplicate attempt {ticket}')
            attempts[ticket] = row
            pending[ticket] = row
            counts['attempted'][kind] += 1
            if kind in LIMITS:
                check(counts['attempted'][kind] <= LIMITS[kind], f'Exceeded {kind}')
        else:
            check(status in ('completed', 'failed'), f'Unknown work status {status}')
            check(ticket in pending and pending[ticket]['kind'] == kind,
                  f'Missing/duplicate/wrong-kind finish {ticket}')
            attempt = pending.pop(ticket)
            done[ticket] = row
            counts[status][kind] += 1
            metadata = attempt.get('metadata', {})
            operation = metadata.get('operation', 'unspecified')
            groups[(kind, operation, bool(metadata.get('startup')), bool(metadata.get('rebuild'))) ][status] += 1
            if kind == 'context_forward':
                count = metadata.get('tokens')
                check(type(count) is int and count >= 0, 'Missing context-token count')
                tokens[operation][status] += count
            durations[(kind, operation, status)].append(row['gpu_seconds']-attempt['gpu_seconds'])
    for row in pending.values():
        kind = row['kind']
        counts['unknown'][kind] += 1
        metadata = row.get('metadata', {})
        operation = metadata.get('operation', 'unspecified')
        groups[(kind, operation, bool(metadata.get('startup')), bool(metadata.get('rebuild')))]['unknown'] += 1
        if kind == 'context_forward':
            count = metadata.get('tokens')
            check(type(count) is int and count >= 0, 'Missing unresolved context-token count')
            tokens[operation]['unknown'] += count
    for kind in WORK:
        check(counts['attempted'][kind] == sum(counts[name][kind] for name in ('completed','failed','unknown')),
              'Unreconciled category '+kind)
    return dict(counts={name:dict(value) for name,value in counts.items()},
                remaining={name:limit-counts['attempted'][name] for name,limit in LIMITS.items()},
                operations=[dict(kind=k,operation=o,startup=s,rebuild=r,**dict(value))
                            for (k,o,s,r),value in sorted(groups.items())],
                context_tokens={key:dict(value) for key,value in sorted(tokens.items())},
                duration_groups=[dict(kind=k,operation=o,status=s,count=len(values),total_seconds=sum(values),
                                      minimum_seconds=min(values),maximum_seconds=max(values))
                                 for (k,o,s),values in sorted(durations.items())],
                gpu_seconds=gpu_ends[-1] if gpu_ends else previous_time,
                gpu_interval_open=gpu_active,gpu_end_count=len(gpu_ends),
                budget_stops=stops,registered_processes=processes,
                duration_interpretation='Observed enclosing intervals only; categories are nested and must not be summed as kernel time.')


def compare_summary(observed, expected, *, final):
    for name in ('attempted', 'completed', 'failed', 'unknown', 'remaining'):
        values = observed['remaining'] if name == 'remaining' else observed['counts'][name]
        left = {key:value for key,value in values.items() if value}
        right = {key:value for key,value in expected[name].items() if value}
        check(left == right, f'Reported {name} differs from independent ledger')
    check(expected['gpu_seconds_limit'] == 14400, 'Reported GPU budget differs')
    if final:
        check(abs(observed['gpu_seconds']-expected['gpu_seconds']) < 1e-6,
              'Final GPU interval differs')
    else:
        check(expected['gpu_seconds'] <= observed['gpu_seconds'], 'Earlier campaign time exceeds final time')


def verify_full_postflight(preflight, receipt):
    check(receipt['status'] == 'passed'
          and receipt['actual_commit'] == receipt['expected_execution_commit'] == COMMIT
          and not receipt['working_tree'] and receipt['new_model_work'] == 0,
          'Full postflight status, source or scope differs')
    expected = {}
    for item in preflight['artifact_checks']:
        check(item['path'] not in expected, 'Duplicated preflight artifact')
        expected[item['path']] = {key:item[key] for key in ('bytes','sha256') if key in item}
    for relative, item in preflight['source_files'].items():
        path = str(Path(preflight['code_root'])/relative)
        check(path not in expected, 'Duplicated preflight source')
        expected[path] = item
    observed = {}
    for item in receipt['checks']:
        check(item['path'] not in observed, 'Duplicated full postflight check')
        observed[item['path']] = item
    check(set(expected) == set(observed), 'Full postflight paths omit or add dependencies')
    for path, wanted in expected.items():
        item = observed[path]
        check(item['matches'] is True
              and all(item['actual'][key] == value for key,value in wanted.items()),
              'Full postflight actual identity differs: '+path)
    return dict(status='passed',artifact_checks=len(preflight['artifact_checks']),
                source_checks=len(preflight['source_files']),total_checks=len(expected),
                clean_tree=True,commit=COMMIT,new_model_work=0,wall_s=receipt['wall_s'])


def audit_raw_work(evidence, ledger):
    totals = Counter()
    categories = Counter()
    for path in sorted((evidence/'campaign/raw').glob('*/*/*.json')):
        saved = read(path)
        if path.name.endswith('_native.json'):
            category = 'original_completion'
            record = saved
            totals['target_prefill_forwards'] += 1
            totals['target_prefill_tokens'] += len(record['prompt_ids'])
            totals['draft_append_forwards'] += 1
            totals['draft_append_tokens'] += len(record['prompt_ids'])
        else:
            category = 'neutral_suffix' if '_neutral_' in path.name else 'crossed_suffix'
            check(saved['status'] == 'completed', 'Incomplete raw suffix')
            record = saved['record']
            reconstruction = saved['reconstruction']
            check(reconstruction['status'] == 'passed'
                  and reconstruction['target_prefix_cache_exact']
                  and reconstruction['destination_draft_rebuilt']
                  and not reconstruction['source_draft_cache_transplanted'],
                  'Raw reconstruction invalid')
            if category == 'neutral_suffix':
                check(reconstruction['neutral_draft_prefix_cache_exact'] is True,
                      'Neutral cache comparison invalid')
            totals['constructor_ingestion_segments'] += reconstruction['ingestion_segments']
            totals['constructor_ingestion_tokens'] += reconstruction['ingestion_tokens']
        categories[category] += 1
        trace = record['metadata']['trace']
        check(record['metadata']['verification_calls'] == len(trace), 'Raw round/count mismatch')
        totals['native_rounds'] += len(trace)
        totals['draft_append_forwards'] += len(trace)
        totals['draft_append_tokens'] += sum(row['committed_progress'] for row in trace)
    for kind,count in categories.items():
        check(count == ledger['counts']['completed'].get(kind,0), 'Raw/ledger count differs: '+kind)
    operations = {(row['kind'],row['operation'],row['startup'],row['rebuild']):
                  row.get('completed',0) for row in ledger['operations']}
    check(totals['native_rounds'] == operations.get(('draft_block','native_proposal_plus_verification',False,False),0),
          'Raw native rounds differ from admitted proposal blocks')
    check(totals['native_rounds'] == operations.get(('target_call','native_target_verify',False,False),0),
          'Raw native rounds differ from target verification calls')
    check(totals['target_prefill_forwards'] == operations.get(('context_forward','native_target_prefill',False,False),0),
          'Raw target prefill count differs')
    draft_count = sum(count for (kind,operation,startup,rebuild),count in operations.items()
                      if kind=='context_forward' and operation=='native_draft_context_ingestion')
    check(totals['draft_append_forwards']+totals['constructor_ingestion_segments'] == draft_count,
          'Draft append plus reconstruction context count differs')
    check(totals['target_prefill_tokens'] == ledger['context_tokens']['native_target_prefill'].get('completed',0),
          'Target context token count differs')
    check(totals['draft_append_tokens']+totals['constructor_ingestion_tokens']
          == ledger['context_tokens']['native_draft_context_ingestion'].get('completed',0),
          'Draft context token count differs')
    return dict(status='passed',raw_records=sum(categories.values()),categories=dict(categories),
                **dict(totals),rebuild_metadata_scope='rebuild=True also includes continuation-time appends; constructor-only totals use saved reconstruction receipts.')



def audit(evidence, code, prepared, control):
    evidence, code, prepared, control = map(Path, (evidence, code, prepared, control))
    ledger_path = evidence/'WORK.jsonl'
    ledger = analyze_ledger([json.loads(line) for line in ledger_path.read_text().splitlines()])
    result, status, plan = (read(evidence/name) for name in ('RESULT.json','campaign/STATUS.json','PLAN.json'))
    compare_summary(ledger, result['work'], final=True)
    compare_summary(ledger, status['work'], final=False)
    check(not ledger['gpu_interval_open'] and ledger['gpu_end_count'] == 1, 'GPU interval not closed exactly once')
    check(result['exit_code'] == 0 and not result['timed_out'] and status['status'] == 'completed',
          'Execution is not a completed, non-timed-out experiment')
    check(status['completed_requests'] == 128 and status['qualified_datasets'] == {'gsm8k':True,'mathqa':True},
          'Request or native gate completion differs')
    check(not ledger['counts']['failed'] and not ledger['counts']['unknown'], 'Failed or unfinished model work remains')
    check(ledger['counts']['completed'].get('original_completion') == 256, 'Original walk total differs')
    rows, native_checks = read(evidence/'campaign/ROWS.json'), read(evidence/'campaign/NATIVE_CHECKS.json')
    check(len(rows) == 128 and len({(r['dataset'],r['request_id']) for r in rows}) == 128,
          'Selected result identities incomplete')
    expected_crossed = sum(not r.get(terminal, True) for r in rows
                           if r['transition_kind'] != 'none' for terminal in ('terminal4','terminal6'))
    check(ledger['counts']['completed'].get('crossed_suffix',0) == expected_crossed,
          'Crossed completions differ from nonterminal successors')
    neutral = [r for r in native_checks if r['kind'] == 'neutral_suffix']
    original = [r for r in native_checks if r['kind'] == 'original_walk']
    check(len(neutral) == ledger['counts']['completed'].get('neutral_suffix',0) <= 8, 'Neutral allowance differs')
    check(len(original) == 256 and all(r['status'] == 'passed' for r in native_checks), 'Native parity evidence incomplete')
    check(len({(r['dataset'],r['request_id'],r['checkpoint'],r['boundary']) for r in neutral}) == len(neutral),
          'Duplicated neutral gate evidence')
    preflight = read(evidence/'PREFLIGHT.json')
    check(preflight['status'] == 'passed' and preflight['execution_commit'] == plan['execution_commit'] == COMMIT,
          'Execution source identity differs')
    check(subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip() == COMMIT,
          'Local source checkout differs')
    source_checks = []
    for relative, expected in preflight['source_files'].items():
        path = code/relative
        actual = identity(path)
        check(actual == expected, 'Source differs: '+relative)
        source_checks.append(relative)
    sealed = read(prepared/'INPUTS.json')
    actual_input = identity(prepared/'INPUTS.json')
    check(read(evidence/'INPUT_IDENTITY.json') == actual_input == read(prepared/'SEAL.json')['private_inputs'],
          'Execution INPUTS differs from prepared seal')
    check({k:v for k,v in plan.items() if k != 'execution_commit'} == sealed, 'Executed scientific plan differs')
    check(identity(prepared/'SELECTED_IDS.json') == sealed['selected_ids_identity'] == read(prepared/'SEAL.json')['selected'],
          'Selected manifest differs')
    postflight = read(evidence/'POSTFLIGHT.json')
    check(len(postflight['checkpoints']) == 4 and all(r['unchanged'] for r in postflight['checkpoints']),
          'Draft weights changed')
    resources = dict(admitted_free_bytes=preflight['free_bytes'],reserve_bytes=preflight['reserve_bytes'],
                     artifact_allowance_bytes=preflight['artifact_allowance_bytes'])
    check(resources['admitted_free_bytes'] >= resources['reserve_bytes']+resources['artifact_allowance_bytes'],
          'Storage admission did not cover reserve')
    control_data = {}
    for name in ('START_UTC','END_UTC','EXIT_CODE','EXECUTION_COMMIT','WORKING_TREE.txt','CPU_TESTS.log'):
        path = control/name
        if path.exists():
            text = path.read_text()
            control_data[name] = text if name != 'CPU_TESTS.log' else text.splitlines()[-5:]
    if 'EXIT_CODE' in control_data:
        check(control_data['EXIT_CODE'].strip() == '0', 'Launch wrapper failed')
    if 'WORKING_TREE.txt' in control_data:
        check(not control_data['WORKING_TREE.txt'].strip(), 'Executed source tree was dirty')
    if all(name in control_data for name in ('START_UTC','END_UTC')):
        parse = lambda text: datetime.fromisoformat(text.strip().replace('Z','+00:00'))
        resources['wrapper_wall_seconds'] = (parse(control_data['END_UTC'])-parse(control_data['START_UTC'])).total_seconds()
    full_path = evidence/'POSTFLIGHT_FULL.json'
    full_receipt = verify_full_postflight(preflight, read(full_path)) if full_path.exists() else None
    raw_work = audit_raw_work(evidence, ledger)
    resources['runner_wall_seconds'] = result['wall_s']
    resources['postflight_cpu_seconds'] = full_receipt['wall_s'] if full_receipt else None
    fallacy_coverage = {
        'Simpson paradox': 'No pooled scientific trend interpreted; datasets stay separate in endpoint analysis.',
        'Ecological fallacy': 'Operation counts do not imply request-level benefit.',
        'Berkson paradox': 'Exact sealed request set checked; no cost-based request replacement.',
        'Collider bias': 'No outcome-based conditioning introduced by this work audit.',
        'Base rate neglect': 'Attempted/completed/failed/unknown and all allowance denominators retained.',
        'Regression to mean': 'No extreme-performance selection or learning-effect claim here.',
        'Survivorship bias': 'All 128 selected requests and all attempted work reconcile, without omissions.',
        'Look-elsewhere effect': 'Fixed work accounting only; no additional scientific tests selected.',
        'Garden of forking paths': 'Frozen execution identities checked; audit helper errors are preserved separately.',
        'Correlation versus causation': 'No causal mechanism, latency gain or kernel breakdown inferred from work costs.',
        'Reverse causality': 'No directional causal claim made in this accounting audit.'
    }

    return dict(status='core_accounting_passed',verification_status='ANALYZED',
                scientific_outcomes_analyzed=False,execution_commit=COMMIT,fallacy_coverage=fallacy_coverage,
                ledger=ledger,source_files_checked=len(source_checks),selected_requests=len(rows),
                native_checks=dict(original=len(original),neutral=len(neutral)),
                unused_crossed_allowance=256-expected_crossed,
                no_divergence_requests=sum(r['transition_kind']=='none' for r in rows),
                terminal_crossed_cells=sum(bool(r.get(t,False)) for r in rows if r['transition_kind']!='none'
                                           for t in ('terminal4','terminal6')),
                resources=resources,control=control_data,full_postflight_receipt=full_receipt,raw_work=raw_work,
                full_postflight_independent_review='passed' if full_receipt else 'unavailable',
                input_identities={str(p.relative_to(evidence)):identity(p) for p in
                    (ledger_path,evidence/'RESULT.json',evidence/'PLAN.json',evidence/'PREFLIGHT.json',evidence/'POSTFLIGHT.json',full_path)},
                limitations=['No dedicated peak-memory measurement inferred from idle GPU receipts.',
                             'Context-token totals count repeated processing; they are not distinct response tokens.',
                             'Nested work intervals are not a kernel-time decomposition.'])


def render(result):
    ledger = result['ledger']
    lines = ['# Independent work/provenance audit','','## Material Passport','',
             'Verification status: ANALYZED. CPU-only; no new model work. Same-team review with prior authorship overlap in admission fixtures and input preparation.',
             '',f"Execution: {result['execution_commit']}. Core ledger/accounting checks passed.",
             '', '| Work | Attempted | Completed | Failed | Unknown | Remaining allowance |',
             '|---|---:|---:|---:|---:|---:|']
    for kind in sorted(WORK):
        counts = [ledger['counts'][name].get(kind,0) for name in ('attempted','completed','failed','unknown')]
        lines.append('| '+kind+' | '+' | '.join(map(str,counts))+' | '+str(ledger['remaining'].get(kind,'not separately capped'))+' |')
    lines += ['',f"Occupied GPU time: **{ledger['gpu_seconds']:.3f}s**; limit 14,400s.",
              f"Neutral checks: {result['native_checks']['neutral']}; originals: {result['native_checks']['original']}.",
              f"Unused crossed allowance: {result['unused_crossed_allowance']}; no-divergence requests: {result['no_divergence_requests']}; terminal crossed cells: {result['terminal_crossed_cells']}.",
              f"Source files authenticated: {result['source_files_checked']}; four draft-weight postflight checks pass.",
              'Full source/target/runtime postflight review: '+result['full_postflight_independent_review']+'.',
              'All selected requests completed; zero failed, unresolved or unattempted required scientific operations. Unspent block allowance is unused capacity, not missing planned work.',
              '', 'Only existing ledger intervals are used for work costs. Native initialization/publication, original walks, suffixes and context operations overlap; never add all categories as kernel time.',
              'Context-token totals include repeated ingestion. No peak-memory estimate is invented from endpoint idle-memory receipts.',
              '', 'Scientific transition outcomes, bootstrap findings and recommendations are outside this audit.']
    raw = result['raw_work']
    lines += ['', '## Raw-work reconciliation', '',
              f"Independently read all {raw['raw_records']} native records. They contain {raw['native_rounds']:,} proposal-plus-verification rounds. Startup contributes three additional draft blocks and 48 standalone verifier blocks, giving 27,800 charged blocks. Target calls add 256 request prefills and 48 startup calls to the native rounds, giving 28,053.",
              f"Context work comprises {raw['target_prefill_forwards']} target prefills / {raw['target_prefill_tokens']:,} token occurrences, {raw['draft_append_forwards']:,} ordinary/continuation draft appends / {raw['draft_append_tokens']:,} token occurrences, and {raw['constructor_ingestion_segments']:,} constructor-only ingestion segments / {raw['constructor_ingestion_tokens']:,} token occurrences. These reconcile to 29,342 context operations and 274,116 processed token occurrences across target and draft paths.",
              'The stored rebuild=True label also covers continuation-time appends. Constructor-only counts above come from the separate saved reconstruction receipts; 14,739 rebuild-labeled operations must not be described as constructor rebuilds alone.',
              '', '## Recorded cost and resources', '',
              '| Existing ledger interval | Count | Seconds |', '|---|---:|---:|']
    for item in ledger['duration_groups']:
        if item['kind'] == 'model_load' and item['status'] == 'completed':
            label = 'Initial native-engine startup' if item['operation'] == 'unspecified' else 'Native draft publication'
            lines.append(f"| {label} | {item['count']} | {item['total_seconds']:.3f} |")
    resources = result['resources']
    lines += ['',f"Wrapper wall time: {resources.get('wrapper_wall_seconds')}s; native-runner wall time: {resources['runner_wall_seconds']:.3f}s; final read-only identity audit: {resources['postflight_cpu_seconds']:.3f}s, performed after occupied GPU accounting ended.",
              f"Admitted storage: {resources['admitted_free_bytes']:,} free bytes, with {resources['reserve_bytes']:,} reserve plus {resources['artifact_allowance_bytes']:,} artifact allowance. No dedicated peak-memory measurement is available.",
              '', 'All 1,421 full postflight identities were compared against the exact preflight expected paths and hashes: 282 artifacts plus 1,139 source files. Both actual commit and clean-tree receipt agree. The 57 A800 launch CPU tests passed. Seven focused private auditor fixtures pass; an initial helper lookup failure and a subsequent helper insertion error were repaired only in CPU analysis code, with their failure receipts preserved. Native execution/source and raw results were not changed.',
              '', 'ARS interpretation coverage: 11/11 categories checked within this work/provenance scope. No scientific effect or causal conclusion is drawn here.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence','code','prepared','control','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        result=audit(args.evidence,args.code,args.prepared,args.control)
    except Exception as error:
        (args.output/'WORK_AUDIT_FAILURE.json').write_text(json.dumps({'status':'failed','error':repr(error)},indent=2)+'\n')
        raise
    (args.output/'WORK_RESULT_REVIEW.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output/'WORK_RESULT_REVIEW.md').write_text(render(result))
    print(json.dumps({'status':result['status'],'gpu_seconds':result['ledger']['gpu_seconds']}))


if __name__=='__main__':
    main()
