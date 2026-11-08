"""All-six-trajectory pilot report, retaining partial work and raw validity failures."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from online_feedback.storage import write_json
from .contract import ARMS, DATASETS, NEW_BUDGET, arm_limits, read_requests, trajectory_root
from .diagnostics import summarize_diagnostics
from .evaluation import audit_stage
from .statistics import COUNT_FIELDS, TIMINGS, reduce_stage, summarize, recommend


def read(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def pooled_observations(observations):
    """Retain all additive counters and pool ratio numerators/denominators."""
    if not observations:
        return dict(response_count=0, sum_metrics={}, ratio_metrics={})
    sums, ratios = {}, {}
    for row in observations:
        for key,value in row['sum_metrics'].items():
            if not math.isfinite(float(value)):
                raise ValueError('Nonfinite training diagnostic')
            sums[key] = sums.get(key,0.) + value
        for key,(numerator,denominator) in row['ratio_metrics'].items():
            if not all(math.isfinite(float(x)) for x in (numerator,denominator)) or denominator < 0:
                raise ValueError('Invalid training diagnostic ratio')
            totals = ratios.setdefault(key,[0.,0.])
            totals[0] += numerator
            totals[1] += denominator
    # KPLUS keeps its original metrics. Role mass and D/Z can still be computed
    # from the same native selection/weight counters without another model pass.
    if all(k in sums for k in ('available_position_weight','selected_position_weight','kplus_W1','kplus_W2')):
        errors = sums['kplus_W1']+sums['kplus_W2']
        sums['derived_original_error_weight'] = errors
        sums['derived_original_prefix_weight'] = sums['selected_position_weight']-errors
        ratios['derived_pooled_full_to_selected'] = [sums['available_position_weight']+1e-6*len(observations),
            sums['selected_position_weight']+1e-6*len(observations)]
    return dict(response_count=len(observations), sum_metrics=sums,
        ratio_metrics={k:dict(numerator=n,denominator=d,value=n/d if d else None) for k,(n,d) in ratios.items()},
        actual_mean_response_loss=sum(row['loss'] for row in observations)/len(observations))


def training_summary(root, dataset, arm):
    from .training import validate_event_receipt
    base = trajectory_root(root,dataset,arm)
    status = read(base/'STATUS.json')
    errors, event_rows, pass_rows, all_observations, all_windows = [], [], [], [], []
    limits = arm_limits(arm)
    work = status.get('actual_work',{})
    if (status.get('status') != 'completed' or status.get('events_completed') != 4
            or work.get('attempts') != limits or work.get('completed') != limits
            or any(work.get('unknown',{}).values())):
        errors.append('trajectory_incomplete_or_extra_work')
    for event in range(1,5):
        name = f'event_{event:03d}'
        event_dir = base/'trainer/events'/name
        try:
            result = read(event_dir/'RESULT.json')
            validate_event_receipt(result,arm=arm,dataset=dataset,event_number=event)
            boundary = read(base/'boundaries'/f'{name}.json')
            if any(boundary.get(k) != v for k,v in dict(status='passed',event=event,dataset=dataset,arm=arm).items()):
                raise ValueError('Scientific trajectory boundary identity differs')
            observations = [row['response_observation'] for row in result['records']]
            if observations != read_lines(event_dir/'RESPONSE_OBSERVATIONS.jsonl'):
                raise ValueError('Event and original-forward observations disagree')
            windows = read_lines(event_dir/'WINDOW_OBSERVATIONS.jsonl')
            if len(windows) != 128 or any(not math.isfinite(w['gradient_norm']) for w in windows):
                raise ValueError('Optimizer window diagnostics incomplete/nonfinite')
            all_observations.extend(observations)
            all_windows.extend(windows)
            event_rows.append(dict(dataset=dataset,arm=arm,event=event,
                resources=read(base/'events'/name/'RESOURCE_COSTS.json'),
                objective_work=result['objective_work'], boundary=boundary,
                optimizer_windows=len(windows), clipped_windows=sum(w['gradient_norm']>1 for w in windows),
                pre_clip_gradient_norm_sum=sum(w['gradient_norm'] for w in windows),
                pre_clip_gradient_norm_max=max(w['gradient_norm'] for w in windows)))
            for pass_index in (0,1):
                selected = [row for row in observations if row['pass_index']==pass_index]
                if len(selected) != 128:
                    raise ValueError('Training pass does not contain128 original observations')
                pass_rows.append(dict(dataset=dataset,arm=arm,event=event,pass_index=pass_index,
                                      **pooled_observations(selected)))
        except (OSError,KeyError,ValueError,TypeError,RuntimeError) as error:
            errors.append(f'{name}: {error!r}')
    return dict(status='passed' if not errors else 'invalid_or_incomplete', dataset=dataset,arm=arm,
        errors=errors, progress=status, actual_work=work, limits=limits,
        unattempted={k:limits[k]-work.get('attempts',{}).get(k,0) for k in limits},
        exposure=pooled_observations(all_observations), pass_rows=pass_rows,event_rows=event_rows,
        optimizer_window_count=len(all_windows), clipped_window_count=sum(w['gradient_norm']>1 for w in all_windows),
        clipped_window_fraction=(sum(w['gradient_norm']>1 for w in all_windows)/len(all_windows) if all_windows else None),
        gradient_norm_scope='Native pre-clipping global norm, one observation per Adam window',
        component_scope='Native K+ forward-KL diagnostics are distinct from the actual hybrid loss.',
        KPLUS_diagnostic_limit='D/Z is derived from native chunk sums; cap and teacher-CE discrepancies are unavailable because KPLUS evaluates neither CE nor reverse KL.')


def independent_scalar_audit(rows, summary):
    """Second calculation from reconstructed native rows; no shared reducer/effect code."""
    methods, checks = ('KPLUS','MIX','ROLE'), 0
    requests = sorted({row['panel_ordinal'] for row in rows})
    if len(requests) != 256:
        raise ValueError('Audit requires all256 paired requests')
    arrays = {}
    for method in methods:
        arrays[method] = {}
        for field in (*COUNT_FIELDS,*TIMINGS):
            values = []
            for request in requests:
                repeats = [row[field] for row in rows if row['panel_ordinal']==request and row['method']==method]
                values.append(sum(repeats)/len(repeats) if field in TIMINGS else repeats[0])
            arrays[method][field] = np.asarray(values,dtype=np.float64)
            if float(np.sum(values)) != summary['method_totals'][method][field]:
                raise ValueError('Independent method total differs: '+method+'/'+field)
            checks += 1
    draws = np.random.Generator(np.random.PCG64(20261107)).integers(0,256,size=(10000,256))
    for candidate,reference in (('ROLE','KPLUS'),('MIX','KPLUS'),('ROLE','MIX')):
        reported = summary['comparisons'][candidate+'_vs_'+reference]
        for field,key in (('verification_calls','call_reduction'), *((t,t+'_reduction') for t in TIMINGS)):
            left,right = arrays[candidate][field],arrays[reference][field]
            denominator = right[draws].sum(axis=1)
            valid = denominator>0
            effects = 100*(1-left[draws].sum(axis=1)[valid]/denominator[valid])
            point = 100*(1-left.sum()/right.sum()) if right.sum()>0 else None
            ci = np.percentile(effects,[2.5,97.5],method='linear').tolist() if len(effects) else None
            if (reported[key]['point_pct'] != point or reported[key]['ci95_pct'] != ci
                    or reported[key]['undefined_bootstrap_draws'] != int((~valid).sum())):
                raise ValueError('Independent bootstrap differs: '+candidate+'/'+reference+'/'+field)
            checks += 3
        delta = arrays[reference]['verification_calls']-arrays[candidate]['verification_calls']
        for key,value in dict(request_wins=int(sum(delta>0)),request_regressions=int(sum(delta<0)),
                              request_ties=int(sum(delta==0))).items():
            if reported[key] != value:
                raise ValueError('Independent request call comparison differs')
            checks += 1
    return dict(status='passed',scalar_checks=checks,source='Raw-native-reconstructed executions',
        calculation='Separate request aggregation and ratio bootstrap implementation',
        limitation='Same-team CPU recalculation, not independent GPU replication.')


def build_report(root, output=None):
    root = Path(root)
    output = Path(output) if output else root/'results'
    output.mkdir(parents=True,exist_ok=False)
    plan = read(root/'sealed/PLAN.json')
    campaign_path = root/'campaign_001/STATUS.json'
    campaign = read(campaign_path) if campaign_path.exists() else dict(status='unavailable')
    errors,training,stages,diagnostics,audits,raw_rows,paired_rows = [],{},{},{},{},[],[]
    for dataset in DATASETS:
        item = plan['datasets'][dataset]
        requests = read_requests(item['panel_manifest'],'test')
        canonicals = read(item['references'])
        statuses = {}
        for arm in ARMS:
            key = dataset+'/'+arm
            try:
                training[key] = training_summary(root,dataset,arm)
                statuses[arm] = training[key]['progress']
                errors.extend(key+': '+error for error in training[key]['errors'])
            except (OSError,KeyError,ValueError,TypeError,RuntimeError) as error:
                training[key] = dict(status='unavailable',dataset=dataset,arm=arm,error=repr(error))
                errors.append(key+': '+repr(error))
            try:
                base = root/'campaign_001/diagnostics'/dataset
                diagnostics[key] = summarize_diagnostics(read(base/'INITIAL/event_000/RESULT.json'),
                    read(base/arm/'event_002/RESULT.json'),read(base/arm/'event_004/RESULT.json'))
            except (OSError,KeyError,ValueError,TypeError,RuntimeError) as error:
                diagnostics[key] = dict(status='invalid_or_incomplete',error=repr(error))
                errors.append(key+'/diagnostics: '+repr(error))
        for event in (2,4):
            key = dataset+f'/event_{event:03d}'
            folder = root/'campaign_001/evaluation'/key
            returned = read_lines(folder/'MEASUREMENTS.jsonl')
            raw_rows.extend(returned)
            checkpoints = {arm:statuses.get(arm,{}).get('event_exports',{}).get(str(event),'missing:'+arm) for arm in ARMS}
            schedule = item.get('evaluation_schedules',{}).get(str(event))
            audit = audit_stage(folder,requests,checkpoints,canonicals=canonicals,dataset=dataset,event=event,
                                schedule=schedule,evidence_root=root)
            audits[key] = {k:v for k,v in audit.items() if k!='rows'}
            if audit['status'] == 'passed':
                try:
                    stages[key] = summarize(audit['rows'],dataset=dataset,event=event)
                    audits[key]['separate_calculation'] = independent_scalar_audit(audit['rows'],stages[key])
                    paired_rows.extend(dict(dataset=dataset,event=event,request_id=row['request_id'],panel_ordinal=row['panel_ordinal'],
                        **{arm+'_'+k:v for arm in ARMS for k,v in row[arm].items()}) for row in reduce_stage(audit['rows'],dataset=dataset,event=event))
                except (KeyError,ValueError,TypeError,RuntimeError) as error:
                    errors.append(key+'/statistics: '+repr(error))
            else:
                errors.extend(key+': '+error for error in audit['errors'])
    qualification_path = root/'qualification/QUALIFICATION.json'
    qualification = read(qualification_path) if qualification_path.exists() else dict(status='unavailable')
    if qualification.get('status') != 'passed':
        errors.append('qualification_not_passed')
    if campaign.get('status') != 'completed':
        errors.append('campaign_not_completed')
    pools = {}
    for dataset in DATASETS:
        path = root/'campaign_001/diagnostics'/dataset/'pool/POOL.json'
        pools[dataset] = read(path) if path.exists() else dict(status='unavailable')
        if pools[dataset].get('status') != 'completed' or pools[dataset].get('completed_preparations') != 16:
            errors.append(dataset+': diagnostic_preparations_incomplete')
    endpoints = {d:stages[d+'/event_004'] for d in DATASETS if d+'/event_004' in stages}
    recommendation = recommend(endpoints if not errors else {})
    summary = dict(status='valid_complete' if not errors else 'invalid_or_incomplete',
        material_passport=dict(workflow='ARS experiment-agent run -> validate',readiness='ANALYZED' if not errors else 'PARTIAL',
            review_scope='Native record reconstruction plus separately calculated CPU audit; not independent GPU replication'),
        execution_source=plan['execution_source'],binding=plan['binding'],budget=NEW_BUDGET,
        training=training,stages=stages,diagnostics=diagnostics,diagnostic_preparation=pools,
        audits=audits,campaign=campaign,qualification=qualification,recommendation=recommendation,errors=errors)
    write_json(output/'SUMMARY.json',summary)
    write_json(output/'AUDIT.json',audits)
    write_json(output/'TRAINING.json',training)
    write_json(output/'DIAGNOSTICS.json',diagnostics)
    write_csv(output/'ALL_REPEATS.csv',raw_rows)
    write_csv(output/'PAIRED_REQUESTS.csv',paired_rows)
    write_csv(output/'TRAINING_EXPOSURE.csv',[row for cell in training.values() for row in cell.get('pass_rows',[])])
    write_csv(output/'EVENT_COSTS.csv',[row for cell in training.values() for row in cell.get('event_rows',[])])
    (output/'REPORT.md').write_text(render_report(summary)+'\n')
    return summary


def render_report(summary):
    lines = ['# K+ role-dependent loss early-adaptation pilot','','## Material Passport','',
        'ARS experiment-agent run -> validate; status **'+summary['status']+'**. '
        'All six trajectories and both planned checkpoints are reported. The audit recalculates retained '
        'native evidence on CPU; it is not independent GPU replication.','',
        'Execution commit: `'+summary['execution_source']['commit']+'`; seal `'+summary['binding']+'`.','',
        'KPLUS retains its qualified forward KL. MIX uses the selected-position 0.8 teacher-argmax CE / '
        '0.2 released-position-clipped reverse-KL mixture. ROLE uses forward KL on matching prefixes and '
        'the mixture at the first error. Selection, redistributed weights, native BF16 training and serving '
        'are held fixed. Each arm collects its own feedback from an independent released initialization.','',
        'This tests early adaptation after four events, not the ranking after a 58-event campaign or the '
        'full AdaFlash system. Both development panels were previously used. Stage 1 / project None remains unchanged.']
    for dataset in DATASETS:
        for event in (2,4):
            stage = summary['stages'].get(dataset+f'/event_{event:03d}')
            lines += ['',f'## {dataset}: event {event}', '']
            if stage is None:
                lines += ['Incomplete or invalid; no comparison is inferred.']
                continue
            lines += ['| Arm | Verification calls | Request seconds | Native generation seconds | Matched / call | Progress / call | Cap requests |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for arm in ARMS:
                value = stage['method_totals'][arm]
                lines += [f"| {arm} | {value['verification_calls']} | {value['request_latency_s']:.6f} | {value['native_generate_s']:.6f} | {value['matched_accepted_tokens_per_call']:.6f} | {value['committed_progress_per_call']:.6f} | {value['cap_hit_requests']} |"]
            lines += ['','| Contrast | Request-time reduction % [95% CI] | Call reduction % [95% CI] | Call wins / regressions / ties | Time wins / regressions / ties |',
                      '|---|---:|---:|---:|---:|']
            for name,effect in stage['comparisons'].items():
                t,c = effect['request_latency_s_reduction'],effect['call_reduction']
                lines += [f"| {name} | {t['point_pct']:.6f} {t['ci95_pct']} | {c['point_pct']:.6f} {c['ci95_pct']} | {effect['request_wins']} / {effect['request_regressions']} / {effect['request_ties']} | {effect['request_latency_wins']} / {effect['request_latency_regressions']} / {effect['request_latency_ties']} |"]
    lines += ['','## Fixed-context explanation','',
        '| Dataset / arm | Initial / event2 / event4 prefix tokens | Event2 / event4 useful repairs | Broken prefixes event2 / event4 | Persistent repairs | Event2 gained / persisted / lost tokens |',
        '|---|---:|---:|---:|---:|---:|']
    for key,value in summary['diagnostics'].items():
        if value.get('status') != 'passed':
            lines += ['| '+key+' | incomplete | | | | |']
            continue
        t = value['totals']
        lines += [f"| {key} | {t.get('initial_prefix',0)} / {t.get('event2_prefix',0)} / {t.get('event4_prefix',0)} | {t.get('event2_repair_with_prefix_intact',0)} / {t.get('event4_repair_with_prefix_intact',0)} | {t.get('event2_broken_prefix',0)} / {t.get('event4_broken_prefix',0)} | {t.get('repair_persists',0)} | {t.get('event2_gained_prefix_tokens',0)} / {t.get('persisted_gain_tokens',0)} / {t.get('lost_gain_tokens',0)} |"]
    lines += ['','A useful repair retains the entire earlier canonical prefix. Diagnostics use identical fixed contexts, '
        'target features, anchors and masks; RNG and model modes are restored. They are local explanations, not serving '
        'speedup estimates or proof of causality. TRAINING.json retains additive role masses, selected fractions, D/Z, '
        'priority fractions, component losses, cap counts, teacher/canonical discrepancies and numerical counters; '
        'ratios pool raw numerators and denominators. EVENT_COSTS.csv retains pre-clipping norms and clipping counts.','',
        '## Work and costs','',
        '| Trajectory | Events complete | Work status | Optimizer windows | Clipped windows |',
        '|---|---:|---|---:|---:|']
    for dataset in DATASETS:
        for arm in ARMS:
            value = summary['training'][dataset+'/'+arm]
            lines += [f"| {dataset}/{arm} | {value.get('progress',{}).get('events_completed',0)} | {value['status']} | {value.get('optimizer_window_count',0)} | {value.get('clipped_window_count',0)} |"]
    lines += ['','The fixed production budget is 3,072 feedback generations, 24 events, 3,072 Adam steps, 6,144 response '
        'forward/backwards and 4,608 timed completions. At most 512 missing references, 4 disposable feedback generations, '
        '8 warmup/qualification completions and 6 disposable steps remain separate. Fixed-context diagnostics cost at most '
        '224 draft response forwards and 32 response feature preparations. Target prefill, native verification replay and '
        'terminal forwards are counted separately in SUMMARY.json; preparation count is not physical-forward count. '
        'Startup/compilation, activation, teacher preparation, persistence and nested timers remain separate. '
        'Unknown, invalid and unattempted work stays in original status records and SUMMARY.json.','',
        'Exact canonical output and termination, including all cap cases, are required for every timed execution. '
        'Counts use one repeat after consistency validation. Event 4 timings average the two repeats within request; '
        'event2 has one repeat. The bootstrap uses 10,000 paired-request draws, NumPy PCG64 seed 20261107, ratio of summed '
        'measurements and linear 2.5/97.5 percentiles. ROLE/KPLUS event4 is primary separately for each dataset; '
        'event2 and multiple secondary intervals are descriptive. There is no pooled cross-dataset efficacy claim.','',
        '## Decision and limits','',summary['recommendation']['next_action'],'',
        'One training seed and reused development inputs limit inference. Matched online recipes do not imply '
        'identical feedback features. No training-inclusive or asynchronous deployment speedup is established. '
        'The pilot stops here; a longer run or another variant needs a separate research decision.']
    if summary['errors']:
        lines += ['','No complete efficacy conclusion is supported because evidence is incomplete or invalid.','',
                  *['- '+error for error in summary['errors']]]
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = build_report(args.root,args.output)
    print(json.dumps(dict(status=result['status'],errors=result['errors'])))
    raise SystemExit(0 if result['status']=='valid_complete' else 1)
