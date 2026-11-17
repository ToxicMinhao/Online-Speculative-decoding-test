"""The fixed C/N and C/S development decision; no selection or tuning."""
from pathlib import Path
import csv
import hashlib
import numpy as np

from online_feedback.storage import write_json
from experiments.persistent_prefix_withholding import analysis as prior
from experiments.persistent_prefix_withholding.evaluation import audit_raw_records

METHODS = ('N', 'C', 'S')
CONTRASTS = (('C', 'N'), ('C', 'S'), ('S', 'N'))


def decision(comparisons):
    cn, cs = comparisons['C_vs_N'], comparisons['C_vs_S']
    metrics = ('time_reduction', 'acceptance_improvement')
    checks = dict(
        time_point_at_least_half_percent=cn['time_reduction']['point'] >= .005,
        time_lower_bound_positive=cn['time_reduction']['ci95'] is not None and cn['time_reduction']['ci95'][0] > 0,
        acceptance_lower_bound_positive=cn['acceptance_improvement']['ci95'] is not None and cn['acceptance_improvement']['ci95'][0] > 0,
        all_repeats_faster=all(x > 0 for x in cn['repeat_time_reductions']),
        cs_both_points_positive=all(cs[key]['point'] is not None and cs[key]['point'] > 0 for key in metrics))
    passes = all(checks.values())
    if passes:
        reason = 'Prespecified screen passed; independent confirmation needs a separate design.'
    elif any(cn[key]['point'] is not None and cn[key]['point'] < 0 for key in metrics):
        reason = 'Negative estimate on a primary C/N metric.'
    elif 0 <= cn['time_reduction']['point'] < .005:
        reason = 'Time point estimate below the practical threshold.'
    else:
        reason = 'Insufficient precision, repeat consistency or advantage over fixed S.'
    return dict(outcome='ELIGIBLE_FOR_SEPARATE_CONFIRMATION' if passes else 'CLOSE_FIXED_CANDIDATE_AT_THIS_SCREEN',
        checks=checks, reason=reason, automatic_confirmation=False,
        cs_intervals_resolved=all(cs[key]['ci95'] is not None and cs[key]['ci95'][0] > 0 for key in metrics))


def summarize(rows):
    paired = prior.reduce_requests(rows, methods=METHODS)
    draws = np.random.Generator(np.random.PCG64(1708)).integers(0, len(paired), size=(10000, len(paired)))
    comparisons = {a+'_vs_'+b: prior.compare(paired, a, b, draws) for a, b in CONTRASTS}
    scored = [row for row in rows if row['phase'] == 'scored']
    def totals(selected):
        return {m: {k: sum(r[k] for r in selected if r['method'] == m)
                    for k in (*prior.COUNTS, *prior.TIMES)} for m in METHODS}
    return dict(scope=prior.SCOPE, request_count=len(paired),
        method_totals={m: prior.method_totals(paired, m) for m in METHODS},
        comparisons=comparisons, decision=decision(comparisons), paired_requests=paired,
        repeat_totals={str(i): totals([r for r in scored if r['repeat'] == i]) for i in range(3)},
        block_views={f'repeat_{i}_block_{j}': totals([r for r in scored if r['repeat'] == i and r['block_ordinal'] == j])
                     for i in range(3) for j in range(4)},
        bootstrap=dict(draws=10000, seed=1708, generator='PCG64', method='linear',
            order='lexicographic source_id', unit='whole paired request, all repeats together',
            draw_matrix_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()),
        limitations=['One training origin and reused development population.',
            'C/N call and acceptance changes are linked views, not independent evidence.',
            'Beating fixed S does not isolate allocation shape, confidence or update size.',
            'No automatic follow-up, OSD comparison, tuning or generalization claim.'])


def write_report(root, *, native_trace_resolver=None):
    root = Path(root)
    audit = audit_raw_records(root, native_trace_resolver=native_trace_resolver)
    write_json(root/'RAW_AUDIT.json', {k: v for k, v in audit.items() if k != 'rows'})
    if audit['status'] != 'passed':
        result = dict(status='INVALID_OR_INCOMPLETE', errors=audit['errors'])
        write_json(root/'SUMMARY.json', result)
        return result
    result = summarize(audit['rows'])
    result.update(status='completed', execution_commit=audit['execution_commit'])
    write_json(root/'SUMMARY.json', result)
    fields = ['source_id','request_id','panel_ordinal','method','repeat','block_ordinal',
        'position','activation','physical_gpu_uuid','checkpoint_id','finish_reason','cap_hit',
        *prior.COUNTS,*prior.TIMES]
    for phase, name in (('scored','ALL_REPEATS.csv'),('warmup','WARMUPS.csv')):
        with (root/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in audit['rows'] if row['phase']==phase)
    fields=['source_id','request_id','panel_ordinal','method',*prior.COUNTS,*prior.TIMES]
    with (root/'PER_REQUEST.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in result['paired_requests']:
            for m in METHODS:
                writer.writerow({**{k:row[k] for k in fields[:3]},'method':m,
                    **{k:row[m][k] for k in (*prior.COUNTS,*prior.TIMES)}})
    return result
