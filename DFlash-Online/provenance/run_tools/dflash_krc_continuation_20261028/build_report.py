"""Local KRC2 evidence summarizer; no model, training or remote execution."""
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys

HERE=Path(__file__).resolve().parent
ARMS=('K2','KRC2')
TIMERS=('request_latency_s','native_generate_s','cache_maintenance_s')
COUNTS=('verification_calls','accepted_draft_tokens','committed_progress','output_tokens')
FORWARDS=('original_response_forwards','pre_reference_response_forwards','post_k_reference_response_forwards',
          'post_c_reference_response_forwards','post_k_learning_response_forwards','post_c_learning_response_forwards')


def read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def rows(path):
    return list(csv.DictReader(path.read_text().splitlines())) if path.is_file() else []


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_csv(path,records):
    keys=list(dict.fromkeys(key for row in records for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(records)


def number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def show(value,digits=4):
    return f'{value:.{digits}f}' if number(value) else 'unavailable'


def reduction(reference,candidate):
    return 100*(1-candidate/reference) if reference else None


def compare(reference,candidate):
    delta=[left-right for left,right in zip(reference,candidate,strict=True)]
    return dict(reference_calls=sum(reference),candidate_calls=sum(candidate),
        call_reduction_pct=reduction(sum(reference),sum(candidate)),
        improved=sum(x>0 for x in delta),regressed=sum(x<0 for x in delta),tied=sum(x==0 for x in delta),
        gross_saved=sum(max(x,0) for x in delta),gross_added=sum(max(-x,0) for x in delta),net_saved=sum(delta))


def panel_summary(raw,name):
    directory=raw/'evaluation'/name
    source=read(directory/'report/SUMMARY.json',{})
    per_request=rows(directory/'report/PER_REQUEST.csv')
    valid=[row for row in per_request if row['valid']=='True']
    totals={}
    for arm in ARMS:
        totals[arm]={key:sum(int(row[f'{arm}_{key}']) for row in valid) for key in COUNTS}
        for timer in TIMERS:
            totals[arm]['summed_mean_'+timer]=sum(float(row[f'{arm}_mean_{timer}']) for row in valid)
        if source:
            for key,value in totals[arm].items():
                expected=source[arm][key]
                if (key in COUNTS and value!=expected) or (key not in COUNTS and abs(value-expected)>1e-8):
                    raise ValueError(f'{name} paired CSV/summary mismatch: {arm}/{key}')
    call_values={arm:[int(row[f'{arm}_verification_calls']) for row in valid] for arm in ARMS}
    history={arm:[int(row[f'{arm}_historical_verification_calls']) for row in valid]
             for arm in ('K1','KRC1')}
    comparisons={
        'KRC2_vs_K2':compare(call_values['K2'],call_values['KRC2']),
        'KRC1_vs_K1':compare(history['K1'],history['KRC1']),
        'KRC2_vs_KRC1':compare(history['KRC1'],call_values['KRC2']),
        'K2_vs_K1':compare(history['K1'],call_values['K2'])}
    old,new=comparisons['KRC1_vs_K1']['call_reduction_pct'],comparisons['KRC2_vs_K2']['call_reduction_pct']
    repeat_rows=rows(directory/'report/REPEAT_TOTALS.csv');repeats=[]
    for repeat in range(3):
        selected={row['method']:row for row in repeat_rows if int(row['repeat'])==repeat}
        if set(selected)!=set(ARMS):continue
        result=dict(repeat=repeat)
        for arm in ARMS:
            result[arm+'_requests']=int(selected[arm]['requests'])
            for timer in TIMERS:result[arm+'_'+timer]=float(selected[arm][timer])
        result['request_time_reduction_pct']=reduction(result['K2_request_latency_s'],result['KRC2_request_latency_s'])
        repeats.append(result)
    execution=read(directory/'STATUS.json',source.get('execution',{}))
    activations=[json.loads(line) for line in (directory/'ACTIVATIONS.jsonl').read_text().splitlines()] if (directory/'ACTIVATIONS.jsonl').is_file() else []
    activation_s=sum(row['activation_s'] for row in activations)
    if execution and abs(activation_s-execution.get('activation_s',0))>1e-8:
        raise ValueError(name+' activation records and total disagree')
    result=dict(name=name,valid_requests=len(valid),planned_requests=len(per_request),totals=totals,
        comparisons=comparisons,relative_advantage_change_percentage_points=None if old is None or new is None else new-old,
        request_time_reduction_pct=reduction(totals['K2']['summed_mean_request_latency_s'],totals['KRC2']['summed_mean_request_latency_s']),
        repeats=repeats,execution=execution,activation_count=len(activations),activation_s=activation_s,
        native_weight_validation_s=execution.get('weight_validation_s'),canonical_outputs=source.get('canonical_outputs'),
        canonical_scope='retained F' if name=='development' else 'retained first K1 on this panel',
        historical_count_gate='K2 development counts' if name=='development' else 'none; K1 is not a K2 count reference')
    return result,per_request,repeat_rows,activations


def correction_steps(manager):
    result=[]
    for source in manager.get('steps',[]):
        value=source.get('correction') or {}
        coords=source.get('integrity',{}).get('step_coordinates',{})
        post=source.get('post_K_totals') or {}
        row=dict(event_local_step=source['step'],global_step=coords.get('global_step_after_native_Adam',128+source['step']),
            window_status=source['status'],correction_status=value.get('status'),
            active=value.get('status')=='applied',BF16_nonzero_correction=number(value.get('realized_correction_norm')) and value['realized_correction_norm']>0,
            learning_record_ids='|'.join(source.get('record_ids',[])),
            reference_record_ids=None,protection_role='R_reference',
            V_and_post_K_totals_role='R_reference',window_wall_s=source.get('window_wall_s'),
            integrity_s=source.get('integrity_s'))
        for key in ('V','global_protected_weight','gradient_norm_squared_fp32','coefficient_fp32',
                    'proposed_correction_norm','realized_correction_norm','K_update_norm',
                    'proposed_correction_to_K_norm_ratio','realized_correction_to_K_norm_ratio'):
            row[key]=value.get(key)
        row.update({'post_K_reference_'+key:value for key,value in post.items()})
        row.update({'cost_'+key:value for key,value in source.get('costs',{}).items()})
        row.update({'work_'+key:value for key,value in source.get('work',{}).items()})
        result.append(row)
    return result


def flatten(value):
    if isinstance(value,list):
        return [item for child in value for item in flatten(child)]
    return [value]


def reference_exposure(raw):
    result=[]
    for path in sorted((raw/'correction').glob('window_*.json')):
        window=read(path,{})
        details=window.get('details',{});binding=details.get('schedule_binding',{})
        reference=details.get('reference',{});metadata=reference.get('responses',[])
        post=(reference.get('post_k') or {}).get('totals',{})
        result.append(dict(event_local_step=window['summary']['step'],global_step=128+window['summary']['step'],
            learning_record_ids='|'.join(binding.get('learning_record_ids',[])),
            reference_record_ids='|'.join(binding.get('reference_record_ids',[])),
            pass_index=binding.get('pass_index'),reference_batch_index=binding.get('reference_batch_index'),
            captured_reference_responses=len(metadata),
            sampled_valid_anchors=sum(sum(bool(v) for v in flatten(row['keep'])) for row in metadata),
            available_reference_proposal_occurrences=sum(sum(v>0 for v in flatten(row['validity_weights'])) for row in metadata),
            protected_positions=post.get('protected_positions'),protected_weight=post.get('protected_weight'),
            strict_margin_violations=post.get('strict_margin_violations'),wrong_at_tie_positions=post.get('wrong_at_tie_positions'),
            pre_reference_integrity_passed=bool(details.get('reference_pre_integrity')) and all(details['reference_pre_integrity'].values())))
    return result


def training_summary(event,manager,costs,steps,references=()):
    counts=manager.get('counts',{});work=manager.get('work',{});components=manager.get('costs',{})
    complete=[row for row in steps if row['window_status']=='completed'];active=[row for row in complete if row['active']]
    ratios=[row['realized_correction_to_K_norm_ratio'] for row in active if number(row['realized_correction_to_K_norm_ratio'])]
    presentations=[]
    for ordinal,source in enumerate(event.get('records',[])):
        anchors=source.get('sampled_anchors',[])
        row={key:source.get(key) for key in ('record_id','pass','optimizer_step_after','optimizer_stepped','learning_rate','training_s','metrics_scope')}
        row.update(presentation_ordinal=ordinal,event_local_optimizer_step=(ordinal//2)+1,
            sampled_anchor_occurrences=sum(sum(bool(x) for x in flatten(anchor['keep_mask'])) for anchor in anchors))
        row.update(source.get('metrics',{}));presentations.append(row)
    boundary=[row for row in event.get('records',[]) if row['optimizer_stepped']]
    exposure={key:sum(row['metrics'][key] for row in boundary) for key in event.get('selection_totals',{})}
    if exposure!=event.get('selection_totals',{}):raise ValueError('Boundary-only selection/exposure sums differ')
    objective_work=event.get('objective_work',costs.get('objective_work',{}))
    reference_ids={identity for row in references for identity in row['reference_record_ids'].split('|') if identity}
    reference_presentations=sum(row['captured_reference_responses'] for row in references)
    summary=dict(event_id=event.get('event_id'),start_step=event.get('start_step'),end_step=event.get('end_step'),
        optimizer_steps=event.get('optimizer_steps'),presentations=event.get('response_presentations'),
        correction_counts=counts,correction_work=work,forward_calls=manager.get('forward_calls',{}),
        objective_work=objective_work,
        reference_exposure=dict(captured_presentations=reference_presentations,
            distinct_response_ids=len(reference_ids),
            pre_update_response_forwards=work.get('pre_reference_response_forwards',0),
            post_K_response_forwards=work.get('post_k_reference_response_forwards',0),
            combined_phase_response_forwards=work.get('pre_reference_response_forwards',0)+work.get('post_k_reference_response_forwards',0),
            scope='A reference presentation is one retained R response/window. Its pre-update and post-K evaluations are two phase forwards, not two distinct examples.'),
        work_counter_scopes={
            'response_forwards':'Complete response-level draft evaluations; independent of chunk/module invocation counters.',
            'pre_reference_chunks':'Reference vocabulary-projection chunks in no-grad capture.',
            'post_k_reference_chunks':'Objective-function entries, including checkpoint recomputation during reference backward; not unique logical blocks.',
            'learning_objective_chunks_at_probe_steps':'Sparse optional A-probe instrumentation; zero means disabled, not zero K-objective work.',
            'forward_calls':'Actual module invocations by phase, including recomputation. learning_K/vocabulary_head includes draft and teacher projections, not just logical K chunks.',
            'objective_work':'Existing K-objective timers, including teacher-head projection and selection, nested in training/window costs; these are not new teacher-feedback collection.'},
        core_response_forwards=sum(work.get(key,0) for key in FORWARDS),
        core_backward_equivalents=counts.get('K_backward_completed',0)+work.get('violation_backward_completed',0),
        diagnostic_response_forwards=sum(work.get(key,0) for key in FORWARDS[3:]),
        first_window=manager.get('first_window'),diagnostic_migration=manager.get('diagnostic_migration'),
        selection_totals=exposure,selection_scope='Optimizer-boundary accumulation metrics counted once; reference exposure is separate.',
        completed_windows=len(complete),applied=len(active),realized_nonzero=sum(row['BF16_nonzero_correction'] for row in complete),
        applied_rounded_zero=sum(not row['BF16_nonzero_correction'] for row in active),
        positive_V=sum(number(row['V']) and row['V']>0 for row in complete),
        zero_V=sum(number(row['V']) and row['V']==0 for row in complete),
        correction_branches={value:sum(row['correction_status']==value for row in steps) for value in sorted({row['correction_status'] for row in steps if row['correction_status']})},
        median_active_realized_to_K_norm_ratio=median(ratios) if ratios else None,
        source_response_anchors=sum(row['sampled_anchor_occurrences'] for row in presentations),
        costs=dict(event_wall_s=costs.get('event_wall_s'),training_windows_s=event.get('costs',{}).get('training_s'),
            intrinsic_reference_correction_s=sum(components.get(key,0) for key in ('pre_reference_s','post_k_s','correction_s')),
            pre_reference_s=components.get('pre_reference_s'),post_K_reference_s=components.get('post_k_s'),
            correction_s=components.get('correction_s'),
            teacher_head_s=objective_work.get('teacher_head_s'),
            first_error_selection_s=objective_work.get('selection_s'),
            integrity_s=sum(components.get(key,0) for key in ('integrity_s','pre_reference_integrity_s','first_window_gate_s')),
            checkpoint_save_s=event.get('costs',{}).get('checkpoint_save_s'),export_s=event.get('costs',{}).get('export_s'),
            startup_s=costs.get('startup',{}).get('wall_s'),snapshot_validation_s=costs.get('snapshot_validation_s'),
            loading_and_live_resume_s=costs.get('loading_and_live_resume_s'),backend_loading_s=costs.get('backend_loading_s'),
            peak_allocated_bytes=costs.get('peak_allocated_bytes'),peak_reserved_bytes=costs.get('peak_reserved_bytes'),
            actual_campaign_wall_s=costs.get('actual_campaign_wall_s'),
            nesting='Component timers are included in their enclosing event/window times; never add them twice.'),
        measurement_limit='V and reference-prefix statistics are measured after K and before correction. No post-correction replay is run; they do not measure the correction’s actual final prefix effect.')
    return summary,presentations


def render(summary):
    training=summary['training'];status=summary['status'];panels=summary['panels']
    text=['# KRC event-2 continuation versus retained K2','',
          f"Execution: **{status.get('status','unavailable')}**. This report keeps the two fixed panels separate.",'',
          'KRC2 resumes its own corrected KRC1 weights, Adam state/counters and RNG at step 128. '
          'It uses the retained shared event-2 feedback and original ordinary order/anchors, then stops at step 256. '
          'No new feedback or K2 training is performed. Reference correction is unchanged; optional learning-prefix and post-correction probes are disabled.','',
          '| Panel | K2 calls | KRC2 calls | KRC2/K2 reduction | Improved / regressed / tied | Gross calls saved / added |',
          '|---|---:|---:|---:|---:|---:|']
    if status.get('status') == 'completed' and all(
        p['comparisons']['KRC2_vs_K2']['net_saved'] > 0
        and p['comparisons']['KRC2_vs_KRC1']['net_saved'] > 0
        and p['relative_advantage_change_percentage_points'] < 0
        for p in panels.values()
    ):
        text[4:4] = [
            '**Answer: KRC’s small advantage persists after the second event, and KRC itself improves on both panels. '
            'However, the advantage over K shrinks: ordinary K gains more from this additional feedback window.** '
            'This supports persistence through the two tested events, not a growing advantage or a generally established online-adaptation method.',
            '',
        ]
    for name,p in panels.items():
        c=p['comparisons']['KRC2_vs_K2']
        text.append(f"| {name} | {c['reference_calls']} | {c['candidate_calls']} | {show(c['call_reduction_pct'])}% | {c['improved']} / {c['regressed']} / {c['tied']} | {c['gross_saved']} / {c['gross_added']} |")
    text.extend(['','| Panel | Historical KRC1/K1 advantage | KRC2/K2 advantage | Change (percentage points) | KRC2/KRC1 call reduction | K2/K1 call reduction |',
                 '|---|---:|---:|---:|---:|---:|'])
    for name,p in panels.items():
        c=p['comparisons']
        text.append(f"| {name} | {show(c['KRC1_vs_K1']['call_reduction_pct'])}% | {show(c['KRC2_vs_K2']['call_reduction_pct'])}% | {show(p['relative_advantage_change_percentage_points'])} | {show(c['KRC2_vs_KRC1']['call_reduction_pct'])}% | {show(c['K2_vs_K1']['call_reduction_pct'])}% |")
    text.extend(['','The relative advantage over K and the change in KRC’s own call count are distinct. '
        'The historical entries above use counts on the same valid request cohort; historical-versus-fresh latency is not interpreted as learning progress.','',
        '| Panel | Arm | Complete-request time (s) | Native generation (s) | Cache maintenance (s) | Accepted draft tokens |',
        '|---|---|---:|---:|---:|---:|'])
    for name,p in panels.items():
        for arm,value in p['totals'].items():
            text.append(f"| {name} | {arm} | {show(value['summed_mean_request_latency_s'],6)} | {show(value['summed_mean_native_generate_s'],6)} | {show(value['summed_mean_cache_maintenance_s'],6)} | {value['accepted_draft_tokens']} |")
    text.extend(['','Request totals sum each request’s three-repeat mean. Native generation and cache maintenance are nested components of complete request latency, not additional costs.','',
        '| Panel | Repeat | K2 complete time (s) | KRC2 complete time (s) | Reduction |','|---|---:|---:|---:|---:|'])
    for name,p in panels.items():
        for row in p['repeats']:
            text.append(f"| {name} | {row['repeat']} | {show(row['K2_request_latency_s'],6)} | {show(row['KRC2_request_latency_s'],6)} | {show(row['request_time_reduction_pct'])}% |")
    text.extend(['','## Restoration, training and correction','',
        f"Exact live resume: **{summary['resume'].get('exact_resume','unavailable')}**. "
        f"Training: **{training.get('optimizer_steps')} new Adam steps / {training.get('presentations')} ordinary presentations**, "
        f"global steps {training.get('start_step')}→{training.get('end_step')}. "
        f"Actual core work: {training['core_response_forwards']} response forwards / {training['core_backward_equivalents']} backward equivalents; "
        f"optional diagnostic forwards: {training['diagnostic_response_forwards']}.",'',
        f"Reference exposure: **{training['reference_exposure']['captured_presentations']} presentations of "
        f"{training['reference_exposure']['distinct_response_ids']} distinct saved responses**, evaluated with "
        f"{training['reference_exposure']['pre_update_response_forwards']} pre-update and "
        f"{training['reference_exposure']['post_K_response_forwards']} post-K forwards "
        f"({training['reference_exposure']['combined_phase_response_forwards']} reference phase forwards total). "
        'The two phases do not double the number of examples. These reference occurrences are additional training exposure.','',
        f"Correction decisions: {training['applied']} applied; {training['realized_nonzero']} have a nonzero BF16-realized displacement; "
        f"{training['applied_rounded_zero']} applied corrections round to zero. Branch counts: `{json.dumps(training['correction_branches'],sort_keys=True)}`. "
        f"Median active correction-to-K norm ratio: {show(training['median_active_realized_to_K_norm_ratio'],6)}.",'',
        training['measurement_limit'],'',
        'Response-forward totals are separate from actual chunk and module invocation counts in `WORK_COUNTERS.csv`. '
        'Post-K reference chunk entries include checkpoint recomputation. The learning-phase vocabulary-head counter includes '
        'both draft and teacher projections plus recomputation; it is not a count of unique training blocks. '
        'Zero sparse learning-probe chunk counts mean that optional instrumentation was disabled, not that K had no loss work.','',
        'Ordinary supervised exposure uses boundary accumulation metrics once. Reference examples are additional training exposure, not held-out evaluation. '
        'Detailed counts, every correction decision, original anchors and reference bindings remain in the machine-readable tables.','',
        '| Cost | Seconds |','|---|---:|'])
    for key,value in training['costs'].items():
        if key.endswith('_s'):text.append(f'| {key} | {show(value,3)} |')
    allocated=training['costs']['peak_allocated_bytes'];reserved=training['costs']['peak_reserved_bytes']
    text.extend(['',f"Peak allocated/reserved GPU memory: {show(allocated/1e9 if number(allocated) else None,3)} / {show(reserved/1e9 if number(reserved) else None,3)} GB.",'',
        'Training, persistence, intrinsic correction and integrity timers are nested. Startup is separately counted as two disposable forward/backwards and zero Adam steps. '
        'Feedback/teacher features are reused; their historical preparation cost is not erased. No historical K training time is presented as a fresh matched overhead control.','',
        '| Panel | Valid / invalid / unknown / unattempted | Publications | Publication time (s) | Weight checks (s) |','|---|---:|---:|---:|---:|'])
    for name,p in panels.items():
        e=p['execution'];v=[e.get(key,'unavailable') for key in ('completed_valid_generations','invalid_generations','unknown_generation_attempts','unattempted_generations')]
        text.append(f"| {name} | {' / '.join(map(str,v))} | {p['activation_count']} | {show(p['activation_s'],3)} | {show(p['native_weight_validation_s'],3)} |")
    text.extend(['','Development output validity uses retained F tokens/termination and retained K2 counts. Additional-panel validity uses retained first-K1 tokens/termination, with no historical K2 count gate. '
        'Both arms require repeat-consistent counts. The first eight development requests provide the included 48-generation gate. '
        'Loading/publication are outside request latency; benchmark switching is not charged once per deployed request.','',
        'This is a shared-feedback continuation, not two independently measured online deployments. '
        'The measurements concern these checkpoints, two feedback windows and fixed panels; they do not establish seed robustness or training-inclusive acceleration. '
        'No later event, retuning or new comparison is authorized by this report.','',
        'One inherited runtime-compatibility metadata label still names the earlier K/G-disabled/G-enabled qualification trials. '
        'That label is stale: the actual startup and campaign receipts for this run record two disposable response forward/backwards, zero startup Adam steps, and the single resumed KRC2 trajectory. '
        'The original metadata is preserved; it is not counted as additional execution.','',
        'Evidence: `analysis/SUMMARY.json`, `TRAINING_SUMMARY.json`, `PRESENTATIONS.csv`, `CORRECTION_STEPS.csv`, '
        '`REFERENCE_EXPOSURE.csv`, `WORK_COUNTERS.csv`, panel-specific request/repeat tables; raw resume checks, schedule checks, '
        'per-window records and native outputs remain under `raw/campaign_001`.'])
    if summary.get('historical_source_bindings'):
        text.extend(['', 'Execution retains the qualified KRC1 source versions. The earlier O/K run used pre-guard versions '
            'of `objective.py` and `training.py`; their original bytes and the current-version mapping are retained in '
            '`HISTORICAL_SOURCE_BINDINGS.json` and the source diff. The first preparation stopped at this version check '
            'before GPU work; resolving the provenance mapping did not change the K loss or KRC correction. '
            'Exact KRC1 restoration is checked, without claiming full-trajectory bitwise equality with historical K2. '
            'K2/KRC2 serving timings above were measured freshly together.'])
    return '\n'.join(text)+'\n'


def build(root=HERE):
    root=Path(root);raw=root/'raw/campaign_001';output=root/'analysis'
    if not raw.is_dir():raise FileNotFoundError('Raw continuation evidence has not arrived: '+str(raw))
    output.mkdir(parents=True,exist_ok=True)
    status=read(raw/'STATUS.json',{});costs=read(raw/'COSTS.json',{})
    manager=read(raw/'CORRECTION_EVIDENCE.json',costs.get('correction',{}))
    event=read(raw/'KRC2/events/event_002/RESULT.json',{})
    references=reference_exposure(raw);reference_by_step={row['event_local_step']:row for row in references}
    steps=correction_steps(manager)
    for row in steps:
        reference=reference_by_step.get(row['event_local_step'])
        if reference is not None:
            if row['learning_record_ids']!=reference['learning_record_ids']:
                raise ValueError('Learning/reference role binding differs in correction summary')
            row['reference_record_ids']=reference['reference_record_ids']
    training,presentations=training_summary(event,manager,costs,steps,references)
    panels={}
    for name in ('development','additional'):
        summary,per_request,repeat_rows,activations=panel_summary(raw,name);panels[name]=summary
        destination=output/name;destination.mkdir(exist_ok=True)
        write(destination/'SUMMARY.json',summary)
        write_csv(destination/'PER_REQUEST.csv',per_request)
        write_csv(destination/'REPEAT_TOTALS.csv',repeat_rows)
        write(destination/'ACTIVATIONS.json',activations)
        source=raw/'evaluation'/name/'report/ALL_REPEATS.csv'
        if source.is_file():(destination/'ALL_REPEATS.csv').write_bytes(source.read_bytes())
    historical_bindings=read(root/'raw/HISTORICAL_SOURCE_BINDINGS.json',
                             read(root/'HISTORICAL_SOURCE_BINDINGS.json',[]))
    summary=dict(status=status,historical_source_bindings=historical_bindings,
        resume=read(raw/'LIVE_RESUME.json',{}),
        snapshot_check=read(raw/'RESUME_SNAPSHOT.json',{}),bindings=read(raw/'REFERENCE_SCHEDULE_CHECK.json',{}),
        training_check=read(raw/'TRAINING_CHECK.json',{}),correction_check=read(raw/'CORRECTION_CHECK.json',{}),
        training=training,startup=costs.get('startup',{}),startup_work=costs.get('startup_work_counts',{}),panels=panels)
    write(output/'SUMMARY.json',summary);write(output/'TRAINING_SUMMARY.json',training)
    write_csv(output/'CORRECTION_STEPS.csv',steps);write_csv(output/'PRESENTATIONS.csv',presentations)
    write_csv(output/'REFERENCE_EXPOSURE.csv',references)
    counters=[]
    for name,value in manager.get('work',{}).items():
        kind='response_forward' if name in FORWARDS else 'objective_entry' if name.endswith('_chunks') or name=='learning_objective_chunks_at_probe_steps' else 'other_work'
        counters.append(dict(counter_family=kind,counter=name,value=value,
            scope=training['work_counter_scopes'].get(name,training['work_counter_scopes']['response_forwards'] if kind=='response_forward' else 'Exact retained work counter; not a response count.')))
    for name,value in manager.get('forward_calls',{}).items():
        counters.append(dict(counter_family='module_invocation',counter=name,value=value,scope=training['work_counter_scopes']['forward_calls']))
    write_csv(output/'WORK_COUNTERS.csv',counters)
    (root/'REPORT.md').write_text(render(summary))
    return summary


if __name__=='__main__':
    result=build(Path(sys.argv[1]) if len(sys.argv)>1 else HERE)
    print(json.dumps({'status':result['status'].get('status'),
                      'panels':{name:value['comparisons'] for name,value in result['panels'].items()}},indent=2))
