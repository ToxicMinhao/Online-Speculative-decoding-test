"""Prespecified unchanged-C transfer analysis for three fixed MathQA endpoints."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding.analysis import COUNTS, TIMES, compare, method_totals
from .evaluation import METHODS, PANEL_SIZE, REPEATS, SCOPE, audit_raw_records

BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 1717, 10000


def reduce_requests(rows):
    """Counts are one validated trajectory; timing is the three-repeat mean."""
    groups, identities = defaultdict(list), {}
    for row in rows:
        if row.get('phase') == 'warmup':
            continue
        if (row.get('phase')!='scored' or row.get('method') not in METHODS
                or row.get('event')!=6 or row.get('scope')!=SCOPE
                or row.get('valid_measurement') is not True or row.get('errors')
                or row.get('validation_status')!='valid'):
            raise ValueError('Only valid event-six N/C/S scored rows enter analysis')
        ordinal = row['panel_ordinal']
        identity = row['source_id'], row['request_id']
        if ordinal in identities and identities[ordinal]!=identity:
            raise ValueError('Panel identity changed between measurements')
        identities[ordinal] = identity
        groups[ordinal,row['method']].append(row)
    if (sorted(identities)!=list(range(PANEL_SIZE))
            or len({v[0] for v in identities.values()})!=PANEL_SIZE
            or len({v[1] for v in identities.values()})!=PANEL_SIZE):
        raise ValueError('Missing or duplicate fixed-panel requests')
    paired = []
    for ordinal, (source_id,request_id) in sorted(identities.items(), key=lambda item:item[1]):
        result = dict(panel_ordinal=ordinal, source_id=source_id, request_id=request_id)
        for method in METHODS:
            records = sorted(groups[ordinal,method],key=lambda row:row['repeat'])
            if [row['repeat'] for row in records]!=list(range(REPEATS)):
                raise ValueError('Every request requires exactly three repeats per arm')
            first = records[0]
            for key in COUNTS:
                if (type(first.get(key)) is not int or first[key]<0
                        or any(row[key]!=first[key] for row in records)):
                    raise ValueError('Invalid or inconsistent native counts')
            calls,progress,native = (first[k] for k in ('verification_calls','committed_progress','native_accepted_draft_tokens'))
            if (native!=progress-calls or progress!=first['output_tokens']-1
                    or first['accepted_draft_tokens']>progress or (calls==0 and progress!=0)):
                raise ValueError('Native acceptance/progress identities failed')
            if (first['finish_reason'] not in ('eos','length')
                    or first['cap_hit']!=(first['finish_reason']=='length')
                    or (first['cap_hit'] and first['output_tokens']!=2048)
                    or any(row[k]!=first[k] for row in records for k in
                           ('checkpoint_id','physical_gpu_uuid','finish_reason','cap_hit','native_acceptance_histogram'))):
                raise ValueError('Endpoint, device, termination or histogram changed')
            if any(not np.isfinite(row[k]) or row[k]<0 or (k!='cache_maintenance_s' and row[k]==0)
                   for row in records for k in TIMES):
                raise ValueError('Invalid request latency')
            histogram = first['native_acceptance_histogram']
            if (any(not 0<=int(k)<=15 or type(v) is not int or v<0 for k,v in histogram.items())
                    or sum(histogram.values())!=calls
                    or sum(int(k)*v for k,v in histogram.items())!=native):
                raise ValueError('Native histogram differs from its counts or block size')
            result[method] = {k:first[k] for k in COUNTS}
            result[method].update({k:float(np.mean([r[k] for r in records])) for k in TIMES})
            result[method].update(checkpoint_id=first['checkpoint_id'],cap_hit=first['cap_hit'],
                finish_reason=first['finish_reason'],native_acceptance_histogram=histogram,
                repeat_request_latency_s=[r['request_latency_s'] for r in records])
        if len({(result[m]['output_tokens'],result[m]['finish_reason']) for m in METHODS})!=1:
            raise ValueError('Paired output length or termination differs')
        paired.append(result)
    if len({r['physical_gpu_uuid'] for r in rows})!=1:
        raise ValueError('All serving rows must use one fixed physical GPU')
    return paired


def decision(comparisons):
    """Apply the ordered handoff gates; an unresolved interval is not equality."""
    native = comparisons['C_vs_N']
    scalar = comparisons['C_vs_S']
    timing, acceptance = native['time_reduction'], native['acceptance_improvement']
    positive_lower = lambda metric: metric['ci95'] is not None and metric['ci95'][0] > 0
    positive_point = lambda metric: metric['point'] is not None and metric['point'] > 0
    checks = dict(
        time_point_at_least_half_percent=timing['point'] is not None and timing['point'] >= .005,
        time_lower_bound_positive=positive_lower(timing),
        acceptance_lower_bound_positive=positive_lower(acceptance),
        every_repeat_faster=len(native['repeat_time_reductions']) == REPEATS
            and all(value > 0 for value in native['repeat_time_reductions']),
        C_vs_S_time_point_positive=positive_point(scalar['time_reduction']),
        C_vs_S_acceptance_point_positive=positive_point(scalar['acceptance_improvement']),
    )
    native_gate = all(checks[key] for key in (
        'time_point_at_least_half_percent', 'time_lower_bound_positive',
        'acceptance_lower_bound_positive', 'every_repeat_faster'))
    comparator_gate = checks['C_vs_S_time_point_positive'] and checks['C_vs_S_acceptance_point_positive']
    harmful = [key for key in ('time_reduction', 'acceptance_improvement')
               if native[key]['ci95'] is not None and native[key]['ci95'][1] < 0]
    if native_gate and comparator_gate:
        outcome = 'TRANSFER_GATE_PASSED'
    elif harmful:
        outcome = 'CLEAR_HARM_ON_THIS_COMPARISON'
    elif native_gate and not comparator_gate:
        outcome = 'NATIVE_GAIN_BUT_COMPARATOR_GATE_FAILED'
    elif timing['ci95'] is not None and timing['ci95'][1] < .005:
        outcome = 'PRACTICAL_TIME_GAIN_NOT_SUPPORTED'
    else:
        outcome = 'UNRESOLVED_OR_MIXED'
    return dict(outcome=outcome, checks=checks, harmful_primary_metrics=harmful,
        automatic_followup_authorized=False,
        scope='Fixed unchanged-C MathQA transfer; no equivalence, seed robustness, or replacement of native K+ claimed.')


def summarize(rows):
    paired = reduce_requests(rows)
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0, len(paired), size=(BOOTSTRAP_DRAWS, len(paired)))
    comparisons = {}
    for candidate, reference in (('C', 'N'), ('C', 'S'), ('S', 'N')):
        effect = compare(paired, candidate, reference, draws)
        effect['net_calls_saved'] = effect['gross_calls_saved'] - effect['gross_calls_added']
        comparisons[f'{candidate}_vs_{reference}'] = effect
    scored = [row for row in rows if row['phase'] == 'scored']

    def totals(selected):
        return {method: {key: sum(row[key] for row in selected if row['method'] == method)
                         for key in (*COUNTS, *TIMES)} for method in METHODS}

    source_order = [dict(source_id=row['source_id'], request_id=row['request_id']) for row in paired]
    order_bytes = json.dumps(source_order, sort_keys=True, separators=(',', ':')).encode()
    return dict(status='completed', scope=SCOPE, request_count=len(paired), count_trajectory_repeat=0,
        method_totals={method: method_totals(paired, method) for method in METHODS},
        comparisons=comparisons, primary_comparison='C_vs_N', required_comparator='C_vs_S',
        context_comparison='S_vs_N', decision=decision(comparisons), paired_requests=paired,
        repeat_totals={str(i): totals([row for row in scored if row['repeat'] == i]) for i in range(REPEATS)},
        block_views={f'repeat_{i}_block_{j}': totals([
            row for row in scored if row['repeat'] == i and row['block_ordinal'] == j])
            for i in range(REPEATS) for j in range(4)},
        bootstrap=dict(draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, generator='PCG64', method='linear',
            order='lexicographic (source_id, request_id)',
            unit='whole paired request, all three arms and all repeat times together',
            common_draws_all_comparisons=True, source_order=source_order,
            source_order_sha256=hashlib.sha256(order_bytes).hexdigest(),
            draw_matrix_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()),
        limitations=[
            'This fixed MathQA transfer does not revise the earlier GSM8K C-versus-S failure.',
            'Audited-unexposed means unexposed within enumerated inventories, not universal freshness.',
            'Request-bootstrap uncertainty conditions on these checkpoints and does not establish training-seed robustness.',
            'Identical-output acceptance and verification-call changes are linked, not independent mechanism evidence.',
            'C-versus-S point estimates passing the gate do not establish superiority if their intervals cross zero.',
            'Serving comparison does not establish training-inclusive speedup or an adaptive-controller benefit.',
            'Every result leaves native K+ as the primary baseline; no automatic follow-up is authorized.'])


def _csv(path, fields, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)


def write_report(root, *, native_trace_resolver=None):
    """Write compact evidence only after the complete raw-record validity gate."""
    root=Path(root)
    audit=audit_raw_records(root,native_trace_resolver=native_trace_resolver)
    write_json(root/'RAW_AUDIT.json',{k:v for k,v in audit.items() if k!='rows'})
    if audit['status']!='passed':
        result=dict(status='INVALID_OR_INCOMPLETE',scope=SCOPE,errors=audit['errors'])
        write_json(root/'SUMMARY.json',result)
        return result
    rows=audit['rows']
    result=summarize(rows)
    result['execution_commit']=audit['execution_commit']
    write_json(root/'SUMMARY.json',result)
    fields=['source_id','request_id','panel_ordinal','method','repeat','block_ordinal','position',
        'activation','physical_gpu_uuid','checkpoint_id','finish_reason','cap_hit',*COUNTS,*TIMES]
    for phase,name in [('scored','ALL_REPEATS.csv'),('warmup','WARMUPS.csv')]:
        _csv(root/name,fields,[r for r in rows if r['phase']==phase])
    paired_rows=[dict(source_id=r['source_id'],request_id=r['request_id'],panel_ordinal=r['panel_ordinal'],
                     method=m,**r[m]) for r in result['paired_requests'] for m in METHODS]
    _csv(root/'PER_REQUEST.csv',['source_id','request_id','panel_ordinal','method',*COUNTS,*TIMES],paired_rows)
    _csv(root/'REPEAT_TOTALS.csv',['repeat','method',*COUNTS,*TIMES],
        [dict(repeat=i,method=m,**result['repeat_totals'][str(i)][m]) for i in range(REPEATS) for m in METHODS])
    trace_rows=[]
    for row in rows:
        record=json.loads(Path(row['record_path']).read_text())
        for index,block in enumerate(record['metadata']['trace']):
            trace_rows.append(dict(label=row['label'],phase=row['phase'],method=row['method'],repeat=row['repeat'],
                request_id=row['request_id'],verification_ordinal=index,output_start=block['output_start'],
                native_accepted_draft_tokens=block['native_accepted_draft_tokens'],
                accepted_draft_tokens=block['accepted_draft_tokens'],committed_progress=block['committed_progress'],
                stop_clipped=block['stop_clipped']))
    _csv(root/'TRACE_COUNTS.csv',['label','phase','method','repeat','request_id','verification_ordinal',
        'output_start','native_accepted_draft_tokens','accepted_draft_tokens','committed_progress','stop_clipped'],trace_rows)
    write_json(root/'VALIDITY.json',dict(status='passed',execution_commit=audit['execution_commit'],
        rows=[{k:r[k] for k in ('label','phase','method','request_id','repeat','validation_status','errors')} for r in rows],
        returned=len(rows),invalid=0,unknown=0,unattempted=0,private_raw_records_retained=True))
    status=json.loads((root/'STATUS.json').read_text())
    write_json(root/'EVALUATION_COSTS.json',dict(**status,
        cost_scope='Evaluation wrapper includes checkpoint publication, validation, warmups, native generation and evidence IO.',
        loading_cost='Backend construction is outside this function and must be recorded by the task runner.',
        nested_timers_not_additive=True,
        measured_time_by_phase={phase:{key:sum(row[key] for row in rows if row['phase']==phase)
            for key in TIMES} for phase in ('warmup','scored')},
        serving_summed_request_means={m:result['method_totals'][m]['request_latency_s'] for m in METHODS}))
    return result
