"""CPU-only ABT evidence report; incomplete datasets never receive inference.

Recompute endpoint validity from native records and paired statistics from the
two individual timing repeats. Source-fit diagnostics cannot establish a serving
mechanism, and no result triggers a new experiment or tuning run.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .contract import ARMS, DATASETS, EVENTS, QUALIFICATION_CASES, ANALYSIS, budgets
from .campaign import tasks, validate_update
from .evaluation import audit_stage
from .statistics import summarize, reduce_stage, recommend


def file_identity(path):
    payload=Path(path).read_bytes()
    return dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())


class Evidence:
    def __init__(self,root,plan):
        self.root,self.plan=Path(root),plan
        self.files,self.missing,self.errors={},{},[]

    def remember(self,path,observed):
        name=str(path)
        if name in self.files and self.files[name]!=observed:
            self.errors.append(dict(path=name,error='Input changed between reads'))
        self.files.setdefault(name,observed)

    def local(self,path):
        path=Path(path)
        if path.is_absolute():
            try: return self.root/path.relative_to(self.plan['root'])
            except ValueError: return path
        return self.root/path

    def read(self,path,*,lines=False):
        path=self.local(path)
        if not path.is_file():
            self.missing[str(path)]='No retained receipt; not imputed as zero or success.'
            return None
        raw=path.read_bytes()
        self.remember(path,dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        try:
            return [json.loads(row) for row in raw.splitlines() if row.strip()] if lines else json.loads(raw)
        except (ValueError,TypeError) as error:
            self.errors.append(dict(path=str(path),error=repr(error)))
            return None

    def remember_tree(self,path):
        """Hash compact records only; never load weights or tensor diagnostics."""
        path=self.local(path)
        for file in sorted(path.rglob('*')) if path.exists() else []:
            if file.is_file() and file.suffix in ('.json','.jsonl','.csv'):
                self.remember(file,file_identity(file))

    def stability(self):
        changed=[]
        for name,expected in self.files.items():
            path=Path(name)
            if not path.is_file() or file_identity(path)!=expected: changed.append(name)
        return dict(status='passed' if not changed else 'changed_during_analysis',changed=changed)


def dump_json(path,value):
    def safe(item):
        if isinstance(item,float) and not math.isfinite(item):
            return {'nonfinite_saved_value':repr(item)}
        if isinstance(item,dict): return {key:safe(v) for key,v in item.items()}
        if isinstance(item,list): return [safe(v) for v in item]
        return item
    Path(path).write_text(json.dumps(safe(value),indent=2,allow_nan=False)+'\n')


def dump_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or ['status'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()})


def flattened(value):
    if isinstance(value,(list,tuple)):
        return [number for item in value for number in flattened(item)]
    if isinstance(value,(int,float)) and math.isfinite(value): return [float(value)]
    raise ValueError('Nonfinite or malformed diagnostic numeric value')


def radius_summary(value):
    values=flattened(value)
    return dict(count=len(values),mean=sum(values)/len(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None,
                total=sum(values))


def qualification(evidence):
    admission=evidence.read('qualification/ADMISSION.json')
    records=[]
    for dataset in DATASETS:
        for case in QUALIFICATION_CASES:
            base=Path('qualification')/dataset/case
            value=evidence.read(base/'RESULT.json')
            row=dict(scope='qualification',dataset=dataset,arm=case,
                     status='no_receipt' if value is None else value.get('status','unknown'))
            if value is not None:
                row.update(error=value.get('error'),work=value.get('work'),wall_s=value.get('wall_s'),
                    two_view_installed=value.get('two_view_installed'),
                    neutral_exact=value.get('neutral_exact'),native_difference=value.get('native_difference'))
                if case=='CLEAN2':
                    difference=value.get('native_difference')
                    if difference is not None:
                        row.update(clean2_native_exact=difference.get('exact_equal'),
                            differing_tensors=len(difference.get('tensor_differences',[])),
                            differing_coordinates=sum(x.get('differing_coordinates',0)
                                for x in difference.get('tensor_differences',[])),
                            metadata_differences=len(difference.get('metadata_differences',[])))
                if case in ARMS:
                    served=evidence.read(base/'SERVING_CHECK.json')
                    row['serving_status']='no_receipt' if served is None else served.get('status','unknown')
            records.append(row)
    return admission,records


def training(evidence):
    results,valid,costs,windows,views,diagnostics,statuses={},{},[],[],[],[],[]
    for dataset in DATASETS:
        for event in EVENTS:
            for arm in ARMS:
                key=(dataset,event,arm)
                base=Path('campaign_001')/dataset/f'event_{event:03d}'/arm
                result=evidence.read(base/'RESULT.json'); status=evidence.read(base/'STATUS.json')
                progress=dict(scope='training',dataset=dataset,event=event,arm=arm,
                    status=(status or result or {}).get('status','no_receipt'),
                    error=(status or result or {}).get('error'))
                errors=[]
                if result is not None:
                    results[key]=result
                    try:
                        baseline=results.get((dataset,event,'KPLUS')) if arm!='KPLUS' else None
                        if arm!='KPLUS' and baseline is None: raise ValueError('Missing contemporaneous KPLUS schedule')
                        validate_update(result,dataset=dataset,event=event,arm=arm,baseline=baseline)
                        if result['boundary']['status']!='passed' or result['boundary']['export_exact'] is not True:
                            raise ValueError('Incomplete decoded boundary/export receipt')
                        for identity_key in ('state_identity','weights_identity'):
                            if len(result[identity_key]['sha256'])!=64 or result[identity_key]['bytes']<=0:
                                raise ValueError('Missing final boundary identity')
                        valid[key]=True
                    except (KeyError,TypeError,ValueError) as error:
                        errors.append(repr(error));valid[key]=False
                    saved=result.get('costs',{})
                    times=[row.get('wall_s') for row in result.get('diagnostic_windows',[])]
                    diagnostic_s=(sum(times) if len(times)==4 and all(isinstance(v,(int,float))
                        and math.isfinite(v) and v>=0 for v in times) else None)
                    total=saved.get('training_s')
                    intrinsic=(total-diagnostic_s if isinstance(total,(int,float))
                        and math.isfinite(total) and diagnostic_s is not None else None)
                    if intrinsic is None: errors.append('Missing/nonfinite complete training or diagnostic timing')
                    if intrinsic is not None and intrinsic<0: errors.append('Diagnostic time exceeds enclosing training time')
                    for name,value in saved.items():
                        if isinstance(value,(int,float)):
                            costs.append(dict(dataset=dataset,event=event,arm=arm,scope='event_training_publication',component=name,value=value))
                    costs.extend([dict(dataset=dataset,event=event,arm=arm,scope='recomputed_nonoverlapping_training',
                        component='diagnostic_observer_s',value=diagnostic_s),
                        dict(dataset=dataset,event=event,arm=arm,scope='recomputed_nonoverlapping_training',
                             component='intrinsic_training_s',value=intrinsic)])
                    for diagnostic in result.get('diagnostic_windows',[]):
                        if len(diagnostic.get('rows',[]))!=2:
                            errors.append('Expected two fixed post-update response observations')
                        for row in diagnostic.get('rows',[]):
                            required=('record_id','clean_fixed_support_loss_before','clean_fixed_support_loss_after',
                                'blocks','old_matching_prefix_tokens','new_matching_prefix_tokens','net_matching_prefix_change',
                                'old_first_errors','first_errors_repaired_with_prefix_intact','new_earlier_errors','old_prefix_tokens_lost')
                            if any(key not in row for key in required):
                                errors.append('Incomplete fixed diagnostic observation')
                            diagnostics.append(dict(dataset=dataset,event=event,arm=arm,
                                window=diagnostic.get('window'),status=diagnostic.get('status'),
                                rng_restored_exact=diagnostic.get('rng_restored_exact'),**row))
                if status is not None:
                    for name in ('wall_s','loading_s'):
                        costs.append(dict(dataset=dataset,event=event,arm=arm,scope='worker_envelope',component=name,value=status.get(name)))
                    startup=status.get('startup',{})
                    costs.append(dict(dataset=dataset,event=event,arm=arm,scope='discarded_startup',component='wall_s',value=startup.get('wall_s')))
                    progress['work']=status.get('work')
                resource=evidence.read(base/'RESOURCE_COSTS.json')
                if resource is not None:
                    for name,value in resource.items():
                        if isinstance(value,(int,float)):
                            costs.append(dict(dataset=dataset,event=event,arm=arm,scope='resource_profile',component=name,value=value))
                observations=base/'trainer/events'/f'event_{event:03d}'
                window_rows=evidence.read(observations/'WINDOW_OBSERVATIONS.jsonl',lines=True)
                if window_rows is not None:
                    for row in window_rows:
                        norm=row.get('gradient_norm')
                        if not isinstance(norm,(int,float)) or not math.isfinite(norm) or norm<0:
                            errors.append('Missing/nonfinite preclip gradient norm')
                        windows.append(dict(dataset=dataset,event=event,arm=arm,**row,
                            preclip_norm_gt_1=(norm>1.) if isinstance(norm,(int,float)) else None))
                    progress['observed_optimizer_windows']=len(window_rows)
                    if valid.get(key) and len(window_rows)!=128:
                        errors.append('Expected 128 compact optimizer-window observations')
                elif valid.get(key): errors.append('Missing compact optimizer-window observations')
                if arm!='KPLUS':
                    view_rows=evidence.read(observations/'VIEW_OBSERVATIONS.jsonl',lines=True)
                    if view_rows is not None:
                        for row in view_rows:
                            try:
                                value=dict(dataset=dataset,event=event,arm=arm,
                                    record_id=row['record_id'],ordinary_presentation=row['ordinary_presentation'],
                                    clean_loss=row['clean_loss'],second_loss=row['second_loss'],
                                    teacher_loss_change=row['second_loss']-row['clean_loss'],
                                    clean_prefix_positions=row['clean_prefix_positions'],second_prefix_positions=row['second_prefix_positions'],
                                    matching_prefix_change=row['second_prefix_positions']-row['clean_prefix_positions'])
                                if not all(isinstance(value[key],(int,float)) and math.isfinite(value[key])
                                           for key in ('clean_loss','second_loss','teacher_loss_change')):
                                    raise ValueError('Nonfinite two-view loss')
                                for name,array in row.get('radii',{}).items():
                                    for metric,number in radius_summary(array).items(): value[name+'_'+metric]=number
                                views.append(value)
                            except (KeyError,TypeError,ValueError) as error:
                                errors.append('Malformed augmentation observation: '+repr(error))
                        progress['observed_two_view_presentations']=len(view_rows)
                        if valid.get(key) and len(view_rows)!=256: errors.append('Expected 256 augmentation observations')
                    elif valid.get(key): errors.append('Missing augmentation observations')
                if errors: valid[key]=False
                progress['audit_errors']=errors
                progress['receipt_valid']=bool(valid.get(key))
                statuses.append(progress)
    return results,valid,costs,windows,views,diagnostics,statuses


def endpoint(evidence,dataset,results,training_valid,admission):
    """A failed/absent fourth arm or native record suppresses that dataset's CI."""
    folder=evidence.root/'campaign_001/evaluation'/dataset
    status=evidence.read(folder/'STATUS.json')
    available=evidence.read(folder/'MEASUREMENTS.jsonl',lines=True) or []
    if not isinstance(available,list): available=[]
    receipt=dict(dataset=dataset,status='invalid_or_incomplete',saved_status=status,inference_permitted=False)
    evidence.remember_tree(folder)
    evidence.remember_tree(folder.parent/(dataset+'_native'))
    if (admission or {}).get('status')!='passed':
        receipt['errors']=['Native qualification admission is absent or failed'];return receipt,[],available
    if not all(training_valid.get((dataset,event,arm),False) for event in EVENTS for arm in ARMS):
        receipt['errors']=['Every event/arm needs complete valid training evidence'];return receipt,[],available
    try:
        item=evidence.plan['datasets'][dataset]
        requests=evidence.read(item['panel_manifest'])['requests']
        canonicals=evidence.read(item['references'])
        if {r.get('binding_digest') for r in canonicals.values()}!={evidence.plan['binding']}:
            raise ValueError('Canonical references belong to a different campaign binding')
        checkpoints={arm:results[dataset,6,arm]['checkpoint_path'] for arm in ARMS}
        audit=audit_stage(folder,requests,checkpoints,canonicals=canonicals,dataset=dataset,event=6,
                          schedule=item['evaluation_schedule'],evidence_root=evidence.root)
        receipt['native_audit']={k:v for k,v in audit.items() if k!='rows'}
        if audit['status']!='passed':
            receipt['errors']=audit['errors'];return receipt,[],available
        stats=summarize(audit['rows'],dataset=dataset,event=6)
        paired=reduce_stage(audit['rows'],dataset=dataset,event=6)
        receipt.update(status='passed',inference_permitted=True,summary=stats)
        return receipt,paired,audit['rows']
    except (KeyError,TypeError,ValueError,OSError) as error:
        receipt['errors']=[repr(error)];return receipt,[],available


def contrast_rows(endpoints):
    rows=[]
    for dataset,item in endpoints.items():
        if item['status']!='passed': continue
        for contrast,value in item['summary']['comparisons'].items():
            calls,time=value['call_reduction'],value['request_latency_s_reduction']
            rows.append(dict(dataset=dataset,contrast=contrast,
                calls_candidate_sum=calls['candidate_sum'],calls_reference_sum=calls['reference_sum'],
                calls_ratio=calls['candidate_sum']/calls['reference_sum'],call_reduction_pct=calls['point_pct'],
                call_ci95_low_pct=calls['ci95_pct'][0],call_ci95_high_pct=calls['ci95_pct'][1],
                calls_saved=value['gross_calls_saved'],calls_added=value['gross_calls_added'],
                call_wins=value['request_wins'],call_regressions=value['request_regressions'],call_ties=value['request_ties'],
                time_candidate_sum_s=time['candidate_sum'],time_reference_sum_s=time['reference_sum'],
                time_ratio=time['candidate_sum']/time['reference_sum'],time_reduction_pct=time['point_pct'],
                time_ci95_low_pct=time['ci95_pct'][0],time_ci95_high_pct=time['ci95_pct'][1],
                time_wins=value['request_latency_wins'],time_regressions=value['request_latency_regressions'],time_ties=value['request_latency_ties']))
    return rows


def render(summary,contrasts,costs,windows,views,diagnostics,qualification_rows):
    lines=['# K+ adversarial block training: bounded events 5–6', '',
        f"**Evidence status: {summary['status']}.** {summary['conclusion']['outcome']}.", '',
        '## Material Passport', '', '- Origin Skill: ARS-Codex 3.24.2 / experiment-agent',
        '- Origin Mode: validate', '- Verification Status: ANALYZED (CPU evidence reconstruction; no model rerun)',
        '- Version Label: kplus_abt_report_v1', '',
        f"Execution source: `{summary['execution_source']['commit']}`; binding `{summary['binding']}`.",
        'Each dataset restores its complete PR32 KPLUS4 state, then continues four arms from 512 to 768 Adam steps. '
        'KPLUS4 supplies event-5 feedback and this campaign’s KPLUS5 supplies event-6 feedback. '
        'The arms share packets, ordinary order and anchors; this is not four independent feedback trajectories.', '',
        'Rho is fixed at 0.01. KPLUS uses one view; CLEAN2, RANDOM and ADV use two equally weighted views with one native '
        'clip/Adam update per ordinary two-response window. Serving remains clean, greedy BF16 block-16, concurrency one, '
        'context 4096 and 2048 generated tokens or EOS. The user-scoped D24 exception does not establish a FreeLB reference reproduction.', '',
        '## Completeness and failures', '',
        f"Valid arm-events: {summary['valid_arm_events']}/16. Dataset endpoint states: "
        +', '.join(f"{d}: {v['status']}" for d,v in summary['endpoints'].items())+'.',
        'All expected tasks and qualification cases remain in `RUN_STATUS.csv`, including failures and absent receipts. '
        'Absence is not imputed as zero work. A dataset with incomplete training or failed native output/termination audit receives no paired inference.', '',
        '## Endpoint comparisons', '',
        'Positive reduction means fewer calls or less complete-request time. Each time is summed after averaging the two repeats within each request. '
        'Ratios are ratios of sums, not means of request ratios. Calls use one validated deterministic trajectory per request.', '',
        '| Dataset / ADV versus | Calls reduction % [95% CI] | Time reduction % [95% CI] | Call wins/regressions/ties |',
        '|---|---:|---:|---:|']
    for row in contrasts:
        lines.append(f"| {row['dataset']} / {row['contrast'].split('_vs_')[1]} | "
            f"{row['call_reduction_pct']:.4f} [{row['call_ci95_low_pct']:.4f}, {row['call_ci95_high_pct']:.4f}] | "
            f"{row['time_reduction_pct']:.4f} [{row['time_ci95_low_pct']:.4f}, {row['time_ci95_high_pct']:.4f}] | "
            f"{row['call_wins']}/{row['call_regressions']}/{row['call_ties']} |")
    if not contrasts: lines.append('| No complete valid dataset | unavailable | unavailable | unavailable |')
    lines.extend(['',f"Paired bootstrap: {ANALYSIS['resamples']:,} PCG64 samples, seed {ANALYSIS['seed']}, linear percentiles, "
        'shared request draws across all three contrasts. ADV/KPLUS is primary; CLEAN2 and RANDOM are mechanism controls. '
        'Intervals are unadjusted and conditional on one continuing training seed and historically reused development panels. '
        'Timing repeats are not independent requests or training seeds. No pooled favorable number or equivalence claim is made.', '',
        '## Training cost and numerical observations', '',
        '| Dataset / arm | Intrinsic training s | Diagnostic observer s | Observed windows with preclip norm >1 |',
        '|---|---:|---:|---:|'])
    for dataset in DATASETS:
        for arm in ARMS:
            matching=[r for r in costs if r['dataset']==dataset and r['arm']==arm and r['scope']=='recomputed_nonoverlapping_training']
            def total(component):
                values=[r['value'] for r in matching if r['component']==component and r['value'] is not None]
                return f'{sum(values):.3f} ({len(values)}/2 events)' if values else 'unavailable'
            win=[r for r in windows if r['dataset']==dataset and r['arm']==arm]
            clip=f"{sum(r['preclip_norm_gt_1'] is True for r in win)}/{len(win)}" if win else 'unavailable'
            lines.append(f"| {dataset} / {arm} | {total('intrinsic_training_s')} | {total('diagnostic_observer_s')} | {clip} |")
    lines.extend(['','Intrinsic training subtracts only measured diagnostic observer time from its enclosing training interval. '
        'Worker loading, discarded startup, checkpoint save/export, shared collection and endpoint activation remain separate in `TRAINING_COSTS.csv` '
        'and `RUN_STATUS.csv`; nested timers must not be added. The clipping column counts reported preclip norms above the native threshold 1, '
        'not a different optimizer setting. Serving speed is not adaptation-inclusive speedup.', '',
        '| Dataset | CLEAN2 exact native comparison | Different tensors / coordinates |', '|---|---|---:|'])
    for dataset in DATASETS:
        row=next(r for r in qualification_rows if r['dataset']==dataset and r['arm']=='CLEAN2')
        exact=row.get('clean2_native_exact','unavailable')
        lines.append(f"| {dataset} | {exact} | {row.get('differing_tensors','unavailable')} / {row.get('differing_coordinates','unavailable')} |")
    lines.extend(['','CLEAN2 is the actual two-view BF16 measurement; mathematical redundancy does not imply bitwise identity. '
        'Disabled and zero-radius checks are separate exact-native gates. Full numerical differences remain in qualification evidence.', '',
        '| Dataset / arm | Recorded two-view responses | Mean clean→second teacher-loss change | Mean intended / realized block norm |',
        '|---|---:|---:|---:|'])
    for dataset in DATASETS:
        for arm in ARMS[1:]:
            selected=[r for r in views if r['dataset']==dataset and r['arm']==arm]
            delta=sum(r['teacher_loss_change'] for r in selected)/len(selected) if selected else None
            def weighted(name):
                count=sum(r.get(name+'_count',0) for r in selected)
                return sum(r.get(name+'_total',0) for r in selected)/count if count else None
            def fmt(number): return 'unavailable' if number is None else f'{number:.6g}'
            lines.append(f"| {dataset} / {arm} | {len(selected)}/512 | {fmt(delta)} | {fmt(weighted('intended_norm'))} / {fmt(weighted('realized_norm'))} |")
    lines.extend(['','Intended FP32 radii and realized BF16 input changes are distinct. `AUGMENTATION.csv` also preserves coverage, '
        'zero-gradient/no-eligible blocks and clean-to-second prefix changes. Loss increase under a finite nonlinear attack is an observation, '
        'not an automatic software-validity test or evidence of clean serving benefit.', '',
        '## Fixed clean post-update diagnostics', '',
        '| Dataset / arm | Recorded response observations | First errors repaired with prefix intact | New earlier errors | Net prefix change |',
        '|---|---:|---:|---:|---:|'])
    for dataset in DATASETS:
        for arm in ARMS:
            selected=[r for r in diagnostics if r['dataset']==dataset and r['arm']==arm and r['status']=='completed']
            def count(name): return sum(r.get(name,0) for r in selected) if selected else 'unavailable'
            lines.append(f"| {dataset} / {arm} | {len(selected)}/16 | {count('first_errors_repaired_with_prefix_intact')} | "
                         f"{count('new_earlier_errors')} | {count('net_matching_prefix_change')} |")
    lines.extend(['','Windows 1/32/65/128 are fixed in both events and every arm. These are the exact fitting responses at fixed anchors, '
        'with clean support reused after the committed update. They are overlapping sampled-block occurrences on divergent training states, '
        'not held-out transfer, independent tokens or unique causal attribution of endpoint changes. See `DIAGNOSTICS.csv` for every observation.', '',
        '## Interpretation and limits', '',
        'All four arms and both datasets must remain visible. ADV must earn improvement over CLEAN2 and RANDOM before the direction itself '
        'is supported. Similar or unresolved differences do not establish equivalence. Better perturbed loss without better clean serving '
        'does not meet the project objective. This short shared-feedback continuation does not establish the full 58-event ranking, '
        'superadditivity, training-seed robustness or deployment performance.', '',
        'ARS statistical/methodological scan: 11/11 categories addressed in `FALLACY_CHECKS.json`, with remaining panel-selection and '
        'multiple-comparison limitations explicit. This is a bounded report review, not an external independent replication. '
        'No automatic radius sweep, method adoption, longer run or research-stage advance follows. Stage 1 / project None remains unchanged.', '',
        'Tables: `COMPARISONS.csv`, `PAIRED_REQUESTS.csv`, `TIMING_REPEATS.csv`, `TRAINING_COSTS.csv`, `OPTIMIZER_WINDOWS.csv`, '
        '`AUGMENTATION.csv`, `DIAGNOSTICS.csv`, `QUALIFICATION.json`, `RUN_STATUS.csv`, `SUMMARY.json`, `INPUT_MANIFEST.json`.'])
    return '\n'.join(lines)+'\n'


def build(root,output):
    root,output=Path(root),Path(output)
    plan=json.loads((root/'sealed/PLAN.json').read_text())
    if plan.get('budgets')!=budgets() or plan.get('analysis')!=ANALYSIS:
        raise ValueError('Report requires the sealed, unchanged ABT budgets and analysis')
    if output.exists(): raise FileExistsError('A report is immutable; choose a separate output directory')
    output.mkdir(parents=True)
    evidence=Evidence(root,plan);evidence.read('sealed/PLAN.json')
    admission,qual_rows=qualification(evidence)
    if admission is not None and (admission.get('binding')!=plan['binding']
            or admission.get('execution_commit')!=plan['execution_source']['commit']):
        admission=dict(admission,status='invalid_binding')
    if admission is not None and admission.get('status')=='passed':
        checks=('native_neutral_exact','two_view_execution','mask_scope','perturbation_gradient',
                'bf16_realization','same_clean_selection','one_adam_per_window','serving','resource','complete_state')
        if (any(admission.get('checks',{}).get(name) is not True for name in checks)
                or any(row['status']!='completed' for row in qual_rows)
                or any(row.get('serving_status')!='passed' for row in qual_rows if row['arm'] in ARMS)):
            admission=dict(admission,status='incomplete_qualification_evidence')
    results,valid,costs,windows,views,diagnostics,statuses=training(evidence)
    controller=evidence.read('campaign_001/STATUS.json')
    postflight=evidence.read('POSTFLIGHT_PILOT.json')
    exit_code=evidence.read('PILOT_EXIT_CODE')
    phase_receipts={}
    for path in ('qualification/STATUS.json','qualification_feedback/PREPARATION.json',
                 'PREFLIGHT_QUALIFICATION.json','POSTFLIGHT_QUALIFICATION.json','PREFLIGHT_PILOT.json'):
        value=evidence.read(path)
        phase_receipts[path]=value
        statuses.append(dict(scope=path,status=(value or {}).get('status','no_receipt'),
            error=(value or {}).get('error'),attempted=(value or {}).get('attempted'),
            completed=(value or {}).get('completed'),work=(value or {}).get('work')))
    endpoints,paired,repeats={},{},{}
    for dataset in DATASETS:
        endpoints[dataset],paired[dataset],repeats[dataset]=endpoint(evidence,dataset,results,valid,admission)
        statuses.append(dict(scope='endpoint',dataset=dataset,status=endpoints[dataset]['status'],errors=endpoints[dataset].get('errors')))
        for event in EVENTS:
            collection=evidence.read(Path('campaign_001')/dataset/f'event_{event:03d}'/'shared/COLLECTION.json')
            statuses.append(dict(scope='shared_feedback',dataset=dataset,event=event,
                status=(collection or {}).get('status','no_receipt'),error=(collection or {}).get('error'),
                work=(collection or {}).get('work'),wall_s=(collection or {}).get('wall_s'),
                loading_s=(collection or {}).get('loading_s')))
        execution=evidence.read(Path('campaign_001/evaluation')/(dataset+'_WORKER.json'))
        statuses.append(dict(scope='evaluation_worker',dataset=dataset,
            status=(execution or {}).get('status','no_receipt'),wall_s=(execution or {}).get('wall_s'),
            loading_s=(execution or {}).get('loading_s'),error=(execution or {}).get('error')))
    stability=evidence.stability()
    if stability['status']!='passed':
        for dataset in DATASETS:
            endpoints[dataset]=dict(status='changed_during_analysis',inference_permitted=False,
                errors=['Consumed input changed; create a stable evidence snapshot first'])
            paired[dataset]=[]
    complete=(sum(valid.values())==16 and all(v['status']=='passed' for v in endpoints.values())
        and (controller or {}).get('status')=='completed' and (postflight or {}).get('status')=='passed'
        and exit_code==0 and not evidence.errors)
    conclusion=recommend({d:v['summary'] for d,v in endpoints.items() if v['status']=='passed'})
    if not complete:
        conclusion=dict(outcome='partial_or_incomplete_evidence; inspect dataset-specific valid endpoints',automatic_followup_authorized=False)
    summary=dict(status='complete' if complete else 'partial_or_incomplete',execution_source=plan['execution_source'],
        binding=plan['binding'],valid_arm_events=sum(valid.values()),expected_tasks=tasks(),controller=controller,
        postflight=postflight,pilot_exit_code=exit_code,phase_receipts=phase_receipts,
        endpoints=endpoints,conclusion=conclusion,input_stability=stability,
        model_work=0,analysis_source_identity=file_identity(__file__),automatic_followup_authorized=False)
    contrasts=contrast_rows(endpoints)
    paired_rows=[]
    for dataset,records in paired.items():
        for row in records:
            base={k:v for k,v in row.items() if k not in ARMS}
            for arm in ARMS: paired_rows.append(dict(base,arm=arm,**row[arm]))
    statuses.extend(qual_rows)
    statuses.append(dict(scope='controller',status=(controller or {}).get('status','no_receipt'),
        error=(controller or {}).get('error'),attempted_task=(controller or {}).get('attempted_task'),
        unattempted_tasks=(controller or {}).get('unattempted_tasks')))
    repeat_rows=[dict(r,analysis_usable=endpoints[ds]['status']=='passed')
                 for ds in DATASETS for r in repeats[ds] if isinstance(r,dict)]
    for name,rows in [('COMPARISONS.csv',contrasts),('PAIRED_REQUESTS.csv',paired_rows),
            ('TIMING_REPEATS.csv',repeat_rows),('TRAINING_COSTS.csv',costs),
            ('OPTIMIZER_WINDOWS.csv',windows),('AUGMENTATION.csv',views),('DIAGNOSTICS.csv',diagnostics),('RUN_STATUS.csv',statuses)]:
        dump_csv(output/name,rows)
    dump_json(output/'SUMMARY.json',summary)
    dump_json(output/'QUALIFICATION.json',dict(admission=admission,cases=qual_rows))
    dump_json(output/'INPUT_MANIFEST.json',dict(files=evidence.files,missing=evidence.missing,
        parse_errors=evidence.errors,stability=stability,model_work=0))
    checks=[('Simpson paradox','Datasets remain separate; no pooled estimate.'),
        ('Ecological fallacy','Paired requests are the uncertainty unit; block counts are descriptive.'),
        ('Berkson selection bias','Historically used input-selected development panels limit generalization.'),
        ('Collider bias','No outcome-conditioned filtering; invalid datasets suppress inference.'),
        ('Base rate neglect','Wins/regressions/ties and complete denominators remain visible.'),
        ('Regression to mean','Contemporaneous controls from the same prescribed origin; no worst-case selection.'),
        ('Survivorship bias','Every expected case/task retained, including failures and missing evidence.'),
        ('Look-elsewhere effect','Three fixed contrasts per dataset; unadjusted intervals, no familywise claim.'),
        ('Forking paths','Sealed ANALYSIS seed/unit/statistic; no outcome-dependent alternate analysis.'),
        ('Correlation versus causation','Fitting diagnostics cannot uniquely explain serving changes.'),
        ('Reverse causality','Training precedes endpoint evaluation, without claiming diagnostic mediation.')]
    dump_json(output/'FALLACY_CHECKS.json',dict(coverage='11/11',scope='Report design and retained-evidence interpretation; not proof that all biases are absent',
        checks=[dict(category=k,disposition=v) for k,v in checks]))
    (output/'REPORT.md').write_text(render(summary,contrasts,costs,windows,views,diagnostics,qual_rows))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=build(args.root,args.output)
    print(json.dumps(dict(status=result['status'],valid_arm_events=result['valid_arm_events'],model_work=0)))
