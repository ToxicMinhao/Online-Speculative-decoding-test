"""Fixed four-endpoint comparison and the user's observed0.1% rule."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.opencode_native_comparison.contract import digest
from experiments.repair_conditional_kl_pilot.analysis import _reduce_requests, TIMES
from experiments.opencode_native_comparison.evaluation import COUNT_FIELDS
from experiments.frontier_preservation_pilot.point3_analysis import exported_parameters_identity
from online_feedback.storage import write_json
from .serving import METHODS,ORDERS

DRAWS,SEED=10000,20261126
CONTRASTS=(('TAIL','BASE'),('BASE','AREPO'),('TAIL','AREPO'),('BASE','F'),('TAIL','F'))


def complete_accounting(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['activation']].append(row)
    if len(rows)!=12384 or set(groups)!=set(range(48)):
        raise ValueError('Require all12288 scored executions and96 warmups')
    warmups={r['request_id'] for r in rows if r['phase']=='warmup'}
    panel_groups={}
    for activation,records in groups.items():
        repeat,group=divmod(activation//4,4)
        method=ORDERS[(repeat+group)%4][activation%4]
        if any((r['repeat'],r['group'],r['method'])!=(repeat,group,method) for r in records):
            raise ValueError('A row changed its balanced slot')
        w=[r for r in records if r['phase']=='warmup'];s=[r for r in records if r['phase']=='scored']
        if len(w)!=2 or len(warmups)!=2 or {r['request_id'] for r in w}!=warmups or len(s)!=256 or len({r['request_id'] for r in s})!=256:
            raise ValueError('A planned warmup or scored request is missing/duplicated')
        chosen=tuple(r['request_id'] for r in s)
        if group in panel_groups and chosen!=panel_groups[group]:raise ValueError('Sealed group order changed')
        panel_groups[group]=chosen
    ids=[key for chosen in panel_groups.values() for key in chosen]
    if len(set(ids))!=1024 or set(ids)&warmups:raise ValueError('Panel groups/warmups overlap')
    return dict(returned=12384,valid=12384,scored=12288,warmups=96,planned_activations=48)


def _interval(point,samples,primary=False):
    if not np.isfinite(samples).all() or not np.isfinite(point):
        raise ValueError('Undefined endpoint comparison')
    return dict(value=float(point),ci95=[float(v) for v in np.quantile(samples,[.025,.975],method='linear')],
                status='prespecified_primary' if primary else 'descriptive_unadjusted')


def summarize(rows,*,expected_requests=1024):
    paired=_reduce_requests(rows,expected_requests=expected_requests,methods=METHODS)
    draws=np.random.Generator(np.random.PCG64(SEED)).integers(0,len(paired),size=(DRAWS,len(paired)))
    values={m:{k:np.asarray([r[m][k] for r in paired],dtype=np.float64) for k in (*TIMES,*COUNT_FIELDS)} for m in METHODS}
    totals={}
    sampled={}
    for method,metrics in values.items():
        total={k:float(v.sum()) for k,v in metrics.items()}
        calls=total['verification_calls']
        for output,key in [('native_accepted_per_call','native_accepted_draft_tokens'),('stop_clipped_accepted_per_call','accepted_draft_tokens'),('committed_per_call','committed_progress')]:
            total[output]=total[key]/calls if calls else None
        total['tokens_per_s']=total['output_tokens']/total['request_latency_s']
        total['repeat_totals']={k:[sum(r[method]['repeat_times'][k][repeat] for r in paired) for repeat in range(3)] for k in TIMES}
        total['termination_counts']=dict(Counter(r[method]['finish_reason'] for r in paired))
        lengths=metrics['output_tokens']
        total['output_length']=dict(min=int(lengths.min()),max=int(lengths.max()),mean=float(lengths.mean()),median=float(np.median(lengths)))
        totals[method]=total
        sampled[method]={key:metrics[key][draws].sum(1) for key in (*TIMES,'verification_calls')}
    contrasts={}
    for candidate,baseline in CONTRASTS:
        result=dict(candidate=candidate,baseline=baseline)
        for key in TIMES:
            # Positive degradation always means the candidate takes longer.
            result[key+'_degradation_pct']=_interval(100*(totals[candidate][key]/totals[baseline][key]-1),
                100*(sampled[candidate][key]/sampled[baseline][key]-1),
                primary=(candidate,baseline,key)==('TAIL','BASE','request_latency_s'))
            delta=values[baseline][key]-values[candidate][key]
            result[key+'_request_outcomes']=dict(wins=int((delta>0).sum()),regressions=int((delta<0).sum()),ties=int((delta==0).sum()))
        result['verification_call_reduction_pct']=_interval(100*(1-totals[candidate]['verification_calls']/totals[baseline]['verification_calls']),
            100*(1-sampled[candidate]['verification_calls']/sampled[baseline]['verification_calls']))
        delta=values[baseline]['verification_calls']-values[candidate]['verification_calls']
        result.update(wins=int((delta>0).sum()),regressions=int((delta<0).sum()),ties=int((delta==0).sum()),
            gross_calls_saved=int(np.maximum(delta,0).sum()),gross_calls_added=int(np.maximum(-delta,0).sum()),net_calls_saved=int(delta.sum()))
        contrasts[candidate+'_vs_'+baseline]=result
    ids=[r['request_id'] for r in paired]
    return dict(requests=len(paired),totals=totals,contrasts=contrasts,paired=paired,
        generation_accounting=complete_accounting(rows) if expected_requests==1024 else None,
        bootstrap=dict(draws=DRAWS,generator='PCG64',seed=SEED,input_order=ids,input_order_digest=digest(ids),
            draw_sha256=hashlib.sha256(draws.astype('<i8',copy=False).tobytes()).hexdigest(),
            percentiles='linear 2.5/97.5',resampling_unit='paired whole request, after averaging three repeats',
            shared_indices_across_endpoints_metrics=True,scope='Fixed four endpoints; one training seed and dataset'),
        sign_convention='Positive timing degradation is slower; positive call reduction is fewer calls')


def selection_decision(report,*,complete,identical_weights=False,identical_trajectories=False):
    if not complete:return dict(status='BLOCKED',selected=None,reason='Incomplete or invalid full comparison')
    value=report['contrasts']['TAIL_vs_BASE']['request_latency_s_degradation_pct']
    point,(_,upper)=value['value'],value['ci95']
    return dict(status='completed',selected='TAIL' if point<=.1 else 'BASE',
        practical_rule_met=point<=.1,degradation_bound_supported=upper<=.1,speed_improvement_supported=upper<0,
        observed_degradation_pct=point,ci95=value['ci95'],threshold_pct=.1,
        identical_serving_weights=identical_weights,identical_native_trajectories=identical_trajectories,
        mechanism_benefit_established=False if identical_weights or identical_trajectories else None,
        interpretation=('Acceptable under the user\'s observed rule;0.1% bound remains unresolved'
            if point<=.1 and upper>.1 else 'Apply the separate observed, bound and speed criteria'),
        call_count_is_adoption_gate=False,automatic_followup=False)


def _csv(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_tables(public,rows,report):
    public=Path(public);public.mkdir(parents=True,exist_ok=True)
    ids={r['request_id']:i for i,r in enumerate(report['paired'])}
    _csv(public/'PER_REQUEST.csv',[dict(request_ordinal=i,**{m+'_'+k:r[m][k] for m in METHODS for k in (*TIMES,*COUNT_FIELDS,'finish_reason')}) for i,r in enumerate(report['paired'])])
    for phase,name in [('scored','ALL_REPEATS.csv'),('warmup','WARMUPS.csv')]:
        lookup=ids if phase=='scored' else {k:i for i,k in enumerate(sorted({r['request_id'] for r in rows if r['phase']=='warmup'}))}
        _csv(public/name,[dict(request_ordinal=lookup[r['request_id']],**{k:r[k] for k in ('activation','repeat','group','method',*TIMES,*COUNT_FIELDS,'finish_reason','valid_measurement')}) for r in rows if r['phase']==phase])
    write_json(public/'SERVING_SUMMARY.json',{k:v for k,v in report.items() if k not in ('paired','bootstrap')})
    write_json(public/'BOOTSTRAP.json',{k:v for k,v in report['bootstrap'].items() if k!='input_order'})


def run_analysis(plan,public):
    root=Path(plan['root'])
    rows=[json.loads(line) for line in (root/'evaluation/MEASUREMENTS.jsonl').read_text().splitlines()]
    report=summarize(rows)
    retained=json.loads((root/'evaluation/SUMMARY.json').read_text())
    report['serving_costs']=retained['serving_costs']
    report['identical_native_trajectories']=retained['identical_native_trajectories']
    status=json.loads((root/'evaluation/STATUS.json').read_text())
    parameters=json.loads((root/'SERVING_PARAMETERS.json').read_text())
    complete=(status['status']=='completed' and status['completed_valid']==12384 and not any(status[k] for k in ('invalid','unknown','unattempted')))
    canonical=json.loads((root/'canonicals/STATUS.json').read_text())
    complete=complete and canonical['status']=='completed' and canonical['completed']==1024 and not any(canonical[k] for k in ('invalid','unknown','unattempted'))
    for arm in ('BASE','TAIL'):
        path=plan.get('endpoint_status_paths',{}).get(arm,root/'campaign_001/arms'/arm/'STATUS.json')
        state=json.loads(Path(path).read_text())
        complete=complete and state['status']=='completed' and state['events_completed']==58 and state['global_step']==7424 and state['presentations']==14848
    report['decision']=selection_decision(report,complete=complete,
        identical_weights=parameters['BASE']['sha256']==parameters['TAIL']['sha256'],
        identical_trajectories=json.loads((root/'evaluation/SUMMARY.json').read_text())['identical_native_trajectories'])
    report['parameter_identities']=parameters
    report['execution_commit']=plan['execution_commit']
    write_json(root/'FINAL_ANALYSIS.json',report)
    write_tables(public,rows,report)
    return report
