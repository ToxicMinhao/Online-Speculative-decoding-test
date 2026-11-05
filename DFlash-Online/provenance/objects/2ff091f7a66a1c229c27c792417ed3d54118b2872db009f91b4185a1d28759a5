"""Independent retained-data KRC2_OFF audit. No Torch, model execution, network or raw writes."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ARMS = ('KRC2_OFF', 'KRC2')
COUNTS = ('verification_calls', 'accepted_draft_tokens', 'committed_progress', 'output_tokens')
TIMERS = ('request_latency_s', 'native_generate_s', 'cache_maintenance_s')


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(ok, issues, name):
    if not ok:
        issues.append(name)


def close(a, b):
    return type(a) in (int, float) and type(b) in (int, float) and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def compare(actual, expected, issues, label):
    if isinstance(expected, dict):
        check(isinstance(actual, dict), issues, label + ': mapping required')
        if isinstance(actual, dict):
            for k, v in expected.items():
                compare(actual.get(k), v, issues, label + '.' + k)
    else:
        check(close(actual, expected) if type(expected) in (int, float) else actual == expected,
              issues, label + ': differs')


def mapped(path, raw):
    """Read existing local mirrors without rewriting any retained remote path."""
    path = str(path)
    source = '/data/sources/minhao/DFlash-Online/'
    runs = '/data/sources/minhao/runs/'
    candidates = [Path(path)]
    if path.startswith(source):
        relative = path[len(source):]
        candidates += [HERE / 'worktree' / relative]
        if relative.startswith('training/'):
            candidates += [Path('/home/minhao2/DFlash-Research') / relative]
    elif path.startswith(runs):
        name, relative = path[len(runs):].split('/', 1)
        base = HERE.parent / name
        candidates += [base / 'raw' / relative, base / relative, base / 'final_review/raw' / relative]
        if name == HERE.name:
            candidates.insert(0, raw / relative)
    return next((p for p in candidates if p.is_file()), None)


def functions_from(path, names, namespace=None):
    namespace = {} if namespace is None else namespace
    nodes = [n for n in ast.parse(Path(path).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def native_helpers():
    path = HERE.parent / 'dflash_prefix_ranking_one_event_20261025/audit_results.py'
    spec = importlib.util.spec_from_file_location('retained_trace_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = functions_from(HERE / 'worktree/online_feedback/native.py', ['validate_generation'])
    return module.trace_record, native['validate_generation']


def slots(requests):
    return [(r['request_id'], repeat, arm) for r in requests for repeat in range(3)
            for arm in (ARMS if (r['panel_ordinal'] + repeat) % 2 == 0 else ARMS[::-1])]


def pool(rows, requests):
    """Counts once per request; timing is the mean of the three repeats."""
    values, totals = {}, {}
    for request in requests:
        for arm in ARMS:
            selected = [r for r in rows if r['request_id'] == request['request_id'] and r['method'] == arm]
            if len(selected) != 3 or {r['repeat'] for r in selected} != {0, 1, 2}:
                raise ValueError('Incomplete repeat set')
            first = next(r for r in selected if r['repeat'] == 0)
            values[request['request_id'], arm] = {**{k: first[k] for k in COUNTS},
                **{k: sum(r[k] for r in selected) / 3 for k in TIMERS}}
    for arm in ARMS:
        total = {k: sum(values[r['request_id'], arm][k] for r in requests) for k in COUNTS}
        for timer in TIMERS:
            total['summed_mean_' + timer] = sum(values[r['request_id'], arm][timer] for r in requests)
            total['all_repeats_' + timer] = sum(r[timer] for r in rows if r['method'] == arm)
        total['calls_per_1000_output_tokens'] = 1000 * total['verification_calls'] / total['output_tokens']
        total['accepted_draft_tokens_per_call'] = total['accepted_draft_tokens'] / total['verification_calls']
        total['committed_progress_per_call'] = total['committed_progress'] / total['verification_calls']
        totals[arm] = total
    comparison = dict(candidate='KRC2', reference='KRC2_OFF')
    for label, metric, reduction in [('call', 'verification_calls', 'call_reduction_pct'),
                                     ('latency', 'request_latency_s', 'request_time_reduction_pct')]:
        before = [values[r['request_id'], 'KRC2_OFF'][metric] for r in requests]
        after = [values[r['request_id'], 'KRC2'][metric] for r in requests]
        delta = [b-a for b, a in zip(before, after)]
        comparison[reduction] = 100 * (1-sum(after)/sum(before))
        comparison[label + '_request_directions'] = dict(improved=sum(d>0 for d in delta), regressed=sum(d<0 for d in delta), tied=sum(d==0 for d in delta))
        comparison[label + '_gross_saved'] = sum(max(0,d) for d in delta)
        comparison[label + '_gross_extra'] = sum(max(0,-d) for d in delta)
        comparison[label + '_net_saved'] = sum(delta)
    return totals, comparison, values


def audit_panel(raw, panel, issues):
    name = panel['name']
    directory = raw / 'campaign_001/evaluation' / name
    requests = [dict(r, panel_ordinal=r.get('panel_ordinal', i), expected_control_calls=(None if panel['expected_control_counts'] is None else panel['expected_control_counts'][r['request_id']]))
                for i, r in enumerate(panel['requests'])]
    retained_name = 'dflash_krc_event_20261027' if name=='development' else 'dflash_krc_confirmation_20261028'
    retained_root = HERE.parent / retained_name / 'raw/campaign_001/evaluation'
    retained_canonical = read(retained_root/'CANONICAL.json')
    retained_rows = {r['request_id']:r for r in lines(retained_root/'MEASUREMENTS.jsonl')
                     if r['method']=='KRC' and r['repeat']==0}
    for request in requests:
        retained = retained_canonical[request['request_id']]
        original_row = retained_rows[request['request_id']]
        original_native = read(retained_root/Path(original_row['record_path']).name)
        check(request['response_ids']==retained['response_ids'] and
              request['finish_reason']==retained['finish_reason'] and
              request['prompt_ids']==original_native['prompt_ids'],
              issues,name+'/'+request['request_id']+': original canonical/input/count provenance')
    rows, protocol, status = lines(directory/'MEASUREMENTS.jsonl'), read(directory/'PROTOCOL.json'), read(directory/'STATUS.json')
    checkpoints = protocol['checkpoints']
    check(read(directory/'REQUESTS.json') == requests, issues, name + ': sealed inputs unchanged')
    check(read(directory/'EXPECTED_CONTROL_COUNTS.json') == panel['expected_control_counts'], issues, name + ': control count identity')
    check(read(directory/'HISTORICAL_COUNTS.json') == panel['historical_counts'], issues, name + ': historical count identity')
    audit_historical_inputs(panel, retained_root, issues)
    compare(protocol, dict(arms=list(ARMS), comparisons=[['KRC2','KRC2_OFF']], repeats=3,
        rotation='(panel_ordinal + repeat) % 2', count_trajectory_repeat=0,
        retained_KRC2_count_gate=True, capture=False, extra_warmup_generations=0), issues, name + ': protocol')
    expected = slots(requests)
    check([(r['request_id'],r['repeat'],r['method']) for r in rows] == expected and len(rows)==384,
          issues, name + ': exact 384 chronological slots')
    for repeat in range(3):
        first = [arm for i,(_,r,arm) in enumerate(expected) if r==repeat and i%2==0]
        check(Counter(first)==Counter(KRC2_OFF=32,KRC2=32), issues, name + ': balanced repeat order')
    compare(status, dict(status='completed',planned_generations=384,attempted_generations=384,
        returned_generations=384,completed_valid_generations=384,invalid_generations=0,
        pending_canonical_generations=0,unknown_generation_attempts=0,unattempted_generations=0,
        optimizer_steps=0,feedback_collections=0,automatic_retry=False,interrupted_attempt=None), issues,name+': status')
    if name=='development':
        compare(read(directory/'QUALIFICATION.json'),dict(status='passed',requests=8,generations=48,
            arms_with_native_equality=sorted(ARMS),performance_win_required=False,extra_generations=0),issues,'included first-eight gate')
        check(status['qualification']=='passed' and status['qualification_requests']==8,issues,'development gate status')
    else:
        check(not (directory/'QUALIFICATION.json').exists() and status['qualification_requests']==0
            and status['qualification']=='not_required_prior_panel_gate', issues,'no repeated additional-panel qualification')
    canonical = read(directory/'CANONICAL.json')
    expected_canonical = {r['request_id']:dict(response_ids=r['response_ids'],finish_reason=r['finish_reason'],
        source_method=r['canonical_source_method'],source='retained_panel_canonical') for r in requests}
    check(canonical==expected_canonical,issues,name+': retained canonical binding')
    by_id = {r['request_id']:r for r in requests}
    trace_record, validate_generation = native_helpers()
    native_paths, total_blocks, checked = set(),0,[]
    histograms = {arm:Counter() for arm in ARMS}
    for row in rows:
        req = by_id[row['request_id']]
        arm, repeat = row['method'],row['repeat']
        label = f"{req['panel_ordinal']:04d}_r{repeat}_{arm}"
        record = read(directory/(label+'.json'))
        compare(row,dict(label=label,validation_status='valid',valid_measurement=True,
            structural_errors=[],errors=[]),issues,name+'/'+label)
        check(Path(row['record_path']).name==label+'.json',issues,label+': record path')
        derived = trace_record(record,req,checkpoints[arm],issues,name+'/'+label)
        meta,capture = record['metadata'],record['metadata']['capture']
        blocks = meta['trace']
        check(record['record_id']==label and record['request_index']==req['panel_ordinal'] and
            record['session_id']=='krc_off_evaluation',issues,label+': native identity')
        check(capture['blocks']==blocks and capture['prompt_ids']==req['prompt_ids'] and
            capture['response_ids']==capture['target_argmax_ids']==record['response_ids'] and
            capture['covered_token_ids']==req['prompt_ids']+record['response_ids'][:-1] and
            capture['covered_rows']==len(capture['covered_token_ids']),issues,label+': capture consistency')
        check(capture['status']=='passed' and capture['block_size']==16 and
            capture['feature_persistence_s']==0 and not capture.get('feature_path') and not meta.get('feature_path'),issues,label+': no feedback capture')
        actual,finish = validate_generation(req['prompt_ids'],record['prompt_ids']+record['response_ids'],blocks,max_new_tokens=2048,stop_tokens=req['stop_token_ids'])
        check(actual==req['response_ids'] and finish==req['finish_reason'],issues,label+': native verifier canonical')
        for i,b in enumerate(blocks):
            check(b['native_accepted_draft_tokens']==max(b['committed_progress']-1,0) and
                b['stop_clipped']==(b['committed_progress']<b['raw_accepted_draft_tokens_before_stop']+1) and
                b['finished']==(i==len(blocks)-1),issues,label+': block counters/termination')
            if repeat==0: histograms[arm][b['accepted_draft_tokens']] +=1
        for key in COUNTS[:-1]+TIMERS[:2]:
            check(close(row[key],meta[key]),issues,label+': native '+key)
        check(row['output_tokens']==len(actual) and row['prompt_tokens']==len(req['prompt_ids']) and
            row['finish_reason']==finish and row['cap_hit']==(finish=='length'),issues,label+': row outputs')
        check(close(row['cache_maintenance_s'],capture['cache_maintenance_s']),issues,label+': cache timer')
        check(all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>=0 for k in TIMERS),issues,label+': finite timers')
        check(row['native_generate_s']+row['cache_maintenance_s']<=row['request_latency_s']+1e-9,issues,label+': nested timers fit total')
        if arm=='KRC2':
            check(derived['verification_calls']==req['expected_control_calls'],issues,label+': retained KRC2 count')
        native_path = raw/'campaign_001/native'/Path(meta['native_trace_path']).name
        check(read(native_path)==capture,issues,label+': raw worker capture')
        check(native_path.name not in native_paths,issues,label+': unique generation capture')
        native_paths.add(native_path.name)
        total_blocks += len(blocks)
        validation=read(directory/(label+'_VALIDATION.json'))
        compare(validation,dict(label=label,request_id=req['request_id'],method=arm,repeat=repeat,
            status='valid',valid=True,errors=[],canonical_source='retained_panel_canonical',
            canonical_method=req['canonical_source_method']),issues,label+': validity receipt')
        checked.append({**row,**{k:derived[k] for k in COUNTS}})
    check({p.name for p in directory.iterdir() if re.fullmatch(r'\d{4}_r[012]_(?:KRC2_OFF|KRC2)\.json',p.name)}=={r['label']+'.json' for r in rows},issues,name+': no extra complete-response records')
    events=lines(directory/'VALIDATION_EVENTS.jsonl')
    check(len(events)==384 and all(e['status']=='valid' and not e['errors'] for e in events),issues,name+': 384 validity events')
    for req in requests:
        for arm in ARMS:
            repeats=[r for r in checked if r['request_id']==req['request_id'] and r['method']==arm]
            check(len({tuple(r[k] for k in COUNTS) for r in repeats})==1,issues,name+'/'+req['request_id']+'/'+arm+': repeat counts')
    totals,comparison,values=pool(checked,requests)
    # Compare reported aggregates only after independently reconstructing all raw evidence.
    summary=read(directory/'report/SUMMARY.json')
    compare(summary,dict(planned_requests=64,valid_requests=64,invalid_or_incomplete_requests=0,
        returned_measurements=384,valid_timed_generations=384,primary_comparison='KRC2_vs_KRC2_OFF',
        timing_repeats_per_method_request=3,**totals,comparisons={'KRC2_vs_KRC2_OFF':comparison}),issues,name+': reported aggregates')
    with (directory/'report/PER_REQUEST.csv').open(newline='') as f: per_request=list(csv.DictReader(f))
    check([r['request_id'] for r in per_request]==[r['request_id'] for r in requests],issues,name+': reported request order')
    for row in per_request:
        check(row['valid']==row['complete']=='True' and row['errors']=='',issues,name+': request validity')
        for arm in ARMS:
            for key in COUNTS:
                check(int(row[arm+'_'+key])==values[row['request_id'],arm][key],issues,name+': per-request '+key)
            for key in TIMERS:
                check(close(float(row[arm+'_mean_'+key]),values[row['request_id'],arm][key]),issues,name+': per-request '+key)
    repeat_totals=[]
    for repeat in range(3):
        by_arm={arm:{k:sum(r[k] for r in checked if r['method']==arm and r['repeat']==repeat) for k in TIMERS} for arm in ARMS}
        repeat_totals.append(dict(repeat=repeat,**by_arm,request_time_reduction_pct=100*(1-by_arm['KRC2']['request_latency_s']/by_arm['KRC2_OFF']['request_latency_s'])))
    historical = count_history(values, requests, panel['historical_counts'])
    compare(summary['count_learning_curve'], historical['comparisons'], issues, name+': count-only learning curve')
    publication=audit_publication(directory,requests,checkpoints,status,issues)
    return dict(records=len(rows),requests=len(requests),native_captures=sorted(native_paths),
        retained_source=str(retained_root),
        reconstructed_native_blocks=total_blocks,totals=totals,comparison=comparison,
        accepted_count_histograms_repeat0={a:dict(histograms[a]) for a in ARMS},
        repeat_totals=repeat_totals,publication=publication,status=status,history=historical)


def audit_publication(directory,requests,checkpoints,status,issues):
    helper=functions_from(HERE/'worktree/experiments/osd_components/protocol.py',
        ['publication_errors','weight_check_errors'],{'BLOCK_SIZE':16})
    activations=lines(directory/'ACTIVATIONS.jsonl')
    previous=None;expected=[]
    by_id={r['request_id']:r for r in requests}
    for rid,repeat,arm in slots(requests):
        if arm!=previous: expected.append((by_id[rid]['panel_ordinal'],repeat,arm));previous=arm
    check([(r['panel_ordinal'],r['repeat'],r['method']) for r in activations]==expected,issues,str(directory)+': publication order')
    for row in activations:
        check(not row['errors'] and not helper['publication_errors'](row,checkpoints[row['method']]),issues,'native publication validity')
    activation_s=sum(r['activation_s'] for r in activations)
    check(close(activation_s,status['activation_s']),issues,'publication time accounting')
    weight_s=0
    for arm in ARMS:
        receipt=read(directory/('WEIGHTS_'+arm+'.json'))
        check(not receipt['errors'] and not helper['weight_check_errors'](receipt['native_reply'],checkpoints[arm]),issues,arm+': loaded tensor equality')
        weight_s+=receipt['validation_s']
    check(close(weight_s,status['weight_validation_s']),issues,'weight validation cost')
    return dict(activations=len(activations),activation_s=activation_s,weight_checks=2,weight_validation_s=weight_s)


def count_history(values, requests, historical):
    """Context uses paired counts, never historical-versus-fresh latency."""
    counts = {arm: {r['request_id']: values[r['request_id'], arm]['verification_calls']
                    for r in requests} for arm in ARMS}
    counts.update(historical)
    comparisons = {}
    for candidate, reference in (('KRC2_OFF', 'KRC1'), ('KRC2', 'KRC1'), ('KRC2_OFF', 'K2')):
        before = [counts[reference][r['request_id']] for r in requests]
        after = [counts[candidate][r['request_id']] for r in requests]
        differences = [b-a for b,a in zip(before, after)]
        comparisons[candidate+'_vs_'+reference] = dict(
            candidate_calls=sum(after), reference_calls=sum(before),
            call_reduction_pct=100*(1-sum(after)/sum(before)),
            request_directions=dict(improved=sum(d>0 for d in differences),
                regressed=sum(d<0 for d in differences), tied=sum(d==0 for d in differences)),
            gross_saved=sum(max(d,0) for d in differences),
            gross_extra=sum(max(-d,0) for d in differences))
    return dict(comparisons=comparisons)


def audit_historical_inputs(panel, retained_root, issues):
    original = lines(retained_root/'MEASUREMENTS.jsonl')
    krc1 = {r['request_id']:r['verification_calls'] for r in original
            if r['method']=='KRC' and r['repeat']==0}
    check(panel['historical_counts']['KRC1'] == krc1, issues,
          panel['name']+': original KRC1 counts')
    prior = lines(HERE.parent/'dflash_krc_continuation_20261028/raw/campaign_001/evaluation'
                  /panel['name']/'MEASUREMENTS.jsonl')
    for arm in ('K2','KRC2'):
        expected={r['request_id']:r['verification_calls'] for r in prior
                  if r['method']==arm and r['repeat']==0}
        actual = panel['historical_counts']['K2'] if arm=='K2' else panel['expected_control_counts']
        check(actual == expected, issues,panel['name']+': retained '+arm+' counts')


def correction_helpers():
    path=HERE.parent/'dflash_krc_event_20261027/audit.py'
    spec=importlib.util.spec_from_file_location('prior_independent_correction_audit',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_training(raw, plan, issues):
    helper = correction_helpers()
    root = raw/'campaign_001'
    event = read(root/'KRC2_OFF/events/event_002/RESULT.json')
    ordinary = read(mapped(plan['event2_ordinary_receipt'],raw))
    retained_k = read(mapped(plan['event2_k_receipt'],raw))
    old_krc = read(mapped(plan['krc1_receipt'],raw))
    krc2 = read(mapped(plan['krc2_receipt'],raw))
    report = read(root/'OFF_EVIDENCE.json')
    schedule = read(mapped(plan['presentation_schedule'],raw))
    check(schedule == ordinary, issues, 'sealed schedule is the retained ordinary event-2 receipt')
    check(event['features']==ordinary['features']==retained_k['features']==krc2['features'],
          issues,'same immutable event-2 pool across O/K/KRC2/OFF')
    compare(event,dict(status='completed',event_id='event_002',start_step=128,end_step=256,
        optimizer_steps=128,response_presentations=256,passes=2,objective='osd_first_error',
        reduction='response_mean',training_variant='KRC2_OFF'),issues,'event-2 receipt')
    feature_ids={f['record_id'] for f in event['features']}
    check(len(feature_ids)==len(event['features'])==128,issues,'128 distinct feedback responses')
    evaluation=[r for p in plan['panels'] for r in p['requests']]
    check(not feature_ids.intersection(r['request_id'] for r in evaluation),issues,'evaluation IDs excluded from event-2 feedback')
    evaluation_prompts={tuple(r['prompt_ids']) for r in evaluation}
    response_root=HERE.parent/'dflash_ok_continuation_20261027/raw/campaign_001/event_002/responses'
    responses=[read(path) for path in sorted(response_root.glob('*.json'))]
    check(len(responses)==128 and not evaluation_prompts.intersection(tuple(r['prompt_ids']) for r in responses),
          issues,'retained event-2 prompt tokens exclude both evaluation panels')
    capped=next(f for f in event['features'] if f['record_id']=='train_00187')
    check(capped['response_tokens']==2048 and capped['length']==2133,issues,'capped response preserved')
    exposure=helper.TRACE.training_exposure(event,2,'K',issues,ordinary)
    check(len(event['records'])==256 and all(
        all(a[k]==b[k] for k in ('record_id','pass','sampled_anchors','optimizer_step_after','optimizer_stepped'))
        for a,b in zip(event['records'],ordinary['records'],strict=True)),
        issues,'ordinary RNG sampling naturally matches all event-2 presentations/anchors')
    snapshot=read(root/'RESUME_SNAPSHOT.json')
    compare(snapshot,dict(status='passed',completed_event=1,objective='osd_first_error',global_step=128,
        presentations=256,horizon=7424,draft_tensors=58,optimizer_parameters=58,adam_step=128,
        cuda_rng_states=1,scientific_extra_steps=0,export_exact=True,state_path=plan['krc1_state'],
        export_path=old_krc['checkpoint_path']),issues,'complete own corrected KRC1 snapshot')
    resume=read(root/'LIVE_RESUME.json')
    compare(resume,dict(status='passed',exact_resume=True,mismatches=[],mismatch_count=0,
        global_step=128,presentations=256,no_rng_reset=True,scientific_extra_steps=0),issues,'exact live model/Adam/RNG resume')
    migration=read(root/'off/CONTRACT_MIGRATION.json')
    check(migration==report['migration'] and migration['changed_contract_keys']==['reference_correction_event']
          and not migration['reseeded'] and not migration['optimizer_reset']
          and not migration['ordinary_K_settings_changed'],issues,'only correction removal after strict restoration')
    before=dict(migration['before']);after=dict(migration['after'])
    before.pop('reference_correction_event');disabled=after.pop('reference_correction_event')
    check(before==after,issues,'ordinary K contract unchanged')
    compare(disabled,dict(arm='KRC2_OFF',base_objective='osd_first_error',reference_processing=False,
        protected_prefix_capture=False,reference_gradients=False,direct_correction=False,probe_steps=[],
        extra_optimizer_steps=0),issues,'explicit correction-disabled contract')
    first=read(root/'off/FIRST_A.json')
    check(first==report['first_A'] and first['status']=='passed' and not first['differences']
          and all(first['checks'].values()),issues,'included first actual A check')
    compare(first,dict(event_local_step=1,global_step_before=128,global_step_after=129,
        optimizer_counters=[129.],extra_optimizer_steps=0,extra_forwards=0),issues,'first-A global accounting')
    fields=('record_id','pass','sampled_anchors','optimizer_step_after','optimizer_stepped','learning_rate','metrics','metrics_scope')
    for index in range(2):
        for field in fields:
            check(first['records'][index][field]==event['records'][index][field]==krc2['records'][index][field],
                  issues,'first-A retained pre-correction equality: '+str(index)+'.'+field)
    check(first['retained_records']==krc2['records'][:2],issues,'first-A gate references actual KRC2 first pair')
    compare(report,dict(arm='KRC2_OFF',status='completed',start_step=128,global_step=256,
        start_global_step=128,end_global_step=256,event_local_windows=128,
        extra_qualification_optimizer_steps=0,direct_corrections=0,replay_updates=0),issues,'OFF boundary and prohibited-work accounting')
    expected_counts=dict(response_forwards_attempted=256,response_forwards_completed=256,
        backward_attempted=256,backward_completed=256,Adam_attempted=128,Adam_completed=128,
        presentations_completed=256,draft_forwards=256)
    compare(report['counts'],expected_counts,issues,'actual ordinary call counters')
    check(all(value==0 for key,value in report['counts'].items() if key.endswith('_failed')),
          issues,'no failed ordinary operation')
    work=dict(response_forwards=256,K_backwards=256,Adam_steps=128,presentations=256,
        reference_forwards=0,reference_backwards=0,protected_prefix_captures=0,
        direct_corrections=0,probe_forwards=0)
    compare(report['work'],work,issues,'256 forwards/backwards with zero reference/probe work')
    check(all(value==0 for value in report['zero_reference_work'].values()),issues,'no hidden reference work')
    journal=lines(root/'off/PRESENTATIONS.jsonl')
    check(len(journal)==256,issues,'256 durable presentation records')
    for actual,retained in zip(journal,event['records'],strict=True):
        check(all(actual[field]==retained[field] for field in fields),issues,'observed presentation equals exported event')
    check(event['correction_removal']==report,issues,'recovery metadata binds correction-removal report')
    compare(read(root/'TRAINING_CHECK.json'),dict(status='passed'),issues,'training receipt gate')
    compare(read(root/'OFF_CHECK.json'),dict(passed=True,core_response_forwards=256,
        core_backwards=256,reference_operations=0,direct_corrections=0),issues,'OFF budget gate')
    startup=read(root/'startup/STARTUP.json')
    compare(startup,dict(status='passed',response_evaluations=2,backwards=2,optimizer_steps=0,
        exact_initial_state_restored=True,separate_disposable_runtime=True,
        runtime_closed_before_K_construction=True),issues,'separate disposable startup')
    check(not (root/'correction').exists(),issues,'no correction directory or optional prefix probes')
    return dict(exposure=exposure,counts=report['counts'],work=report['work'],core_response_forwards=256,
        core_backwards=256,reference_operations=0,direct_corrections=0,diagnostic_forwards=0,
        resume_snapshot=snapshot,live_resume=resume,first_A=first,migration=migration,startup=startup,
        event_costs=event['costs'],off_costs=report['costs'])


def audit_integrity(raw, plan, issues):
    seal=read(raw/'sealed/SEAL.json');pre=read(raw/'PREFLIGHT.json');post=read(raw/'POSTFLIGHT.json')
    check(seal['sealed_before_training'],issues,'source and dependencies sealed before training')
    for label,receipt in (('preflight',pre),('postflight',post)):
        check(receipt['status']=='passed' and receipt['files_checked']==len(seal['files']),issues,label+': every sealed file checked')
    storage=pre['storage']
    check(storage['admitted'] and storage['free_bytes']>=storage['required_bytes']
        and storage['reserve_bytes']==8*2**30
        and storage['host_available_bytes']>=storage['host_required_bytes']==72*2**30,
        issues,'peak disk and host-memory admission')
    runtime=read(raw/'sealed/RUNTIME_PIN.json')
    for receipt in (pre,post):
        compare(receipt['runtime'],{k:runtime[k] for k in ('python_version','torch','transformers','cuda','native_runtime_sha256')},issues,'pinned native runtime')
        compare(receipt['training_runtime'],dict(torch='2.13.0+cu130',transformers='5.12.1'),issues,'pinned training runtime')
    local_count=0;remote_only=[]
    for name,identity in seal['files'].items():
        local=mapped(name,raw)
        if local is None:
            remote_only.append(name)
            continue
        check(local.stat().st_size==identity['bytes'] and digest(local)==identity['sha256'],issues,'local sealed identity: '+str(local))
        local_count+=1
    bindings=read(HERE.parent/'dflash_krc_continuation_20261028/raw/HISTORICAL_SOURCE_BINDINGS.json')
    check({Path(b['original_path']).name for b in bindings}=={'objective.py','training.py'},issues,'two explicit old/current source bindings')
    for binding in bindings:
        archived=mapped(binding['retained_original'],raw)
        current=mapped(binding['original_path'],raw)
        check(archived is not None and digest(archived)==binding['original_identity']['sha256'],issues,'exact archived O/K source bytes')
        check(current is not None and digest(current)==binding['execution_identity']['sha256'],issues,'unchanged qualified current execution source')
        parent_seal=read(mapped(binding['execution_pin'],raw))
        check(parent_seal['files'][binding['original_path']]==binding['execution_identity'],issues,'current source bound to qualified KRC1')
    check(int((raw/'EXIT_CODE').read_text())==0,issues,'supervisor exit0')
    status=read(raw/'campaign_001/STATUS.json');costs=read(raw/'campaign_001/COSTS.json')
    compare(status,dict(status='completed',phase='finished',global_optimizer_step=256,new_optimizer_steps=128,
        new_presentations=256,event_local_windows_completed=128,new_feedback_generations=0,
        new_qualification_optimizer_steps=0,new_qualification_presentations=0,automatic_retry=False,
        actual_new_Adam_steps_completed=128,actual_global_Adam_step=256,actual_K_presentations_completed=256),issues,'completed bounded campaign')
    compare(status['evaluation_totals'],dict(planned_generations=768,attempted_generations=768,
        returned_generations=768,completed_valid_generations=768,invalid_generations=0,
        unknown_generation_attempts=0,unattempted_generations=0),issues,'768 native executions including qualification')
    check(all(type(costs[k]) in (int,float) and math.isfinite(costs[k]) and costs[k]>0 for k in
        ('event_wall_s','loading_and_live_resume_s','snapshot_validation_s','backend_loading_s','actual_campaign_wall_s')),issues,'finite separate campaign costs')
    engine=read(raw/'campaign_001/native/engine_settings.json')
    compare(engine,dict(model_path=plan['target'],speculative_draft_model_path=plan['draft'],
        speculative_algorithm='DFLASH',speculative_num_draft_tokens=16,dtype='bfloat16',attention_backend='fa3',
        tp_size=1,max_running_requests=1,context_length=4112,chunked_prefill_size=8192,
        random_seed=17,skip_server_warmup=True),issues,'qualified inference engine')
    compare(read(raw/'campaign_001/TRAINER_CLOSED.json'),dict(closed=True,failed=False,
        global_step=256,presentations=512,completed_events=['event_001','event_002'],
        reduction='response_mean',objective='osd_first_error',accumulation_remainder=0,horizon=7424),issues,'trainer closed before evaluation')
    retrieval=read(raw/'RETRIEVAL_MANIFEST.json');retrieved_bytes=0
    for name,identity in retrieval.items():
        local=raw/name
        check(local.is_file() and local.stat().st_size==identity['bytes'] and digest(local)==identity['sha256'],issues,'retrieved identity: '+name)
        retrieved_bytes+=identity['bytes']
    return dict(sealed_files=len(seal['files']),local_files_rehashed=local_count,
        remote_only_files=len(remote_only),retrieved_files_rehashed=len(retrieval),retrieved_bytes=retrieved_bytes,
        source_integrity_basis='All local counterparts rehashed; remote pre/postflight binds nonlocal model/optimizer tensors.',
        historical_source_bindings=bindings,preflight=pre,postflight=post,costs=costs,status=status)


def run_audit(raw):
    raw=Path(raw);issues=[]
    result=dict(verification_status='ANALYZED',scope='KRC correction-removal continuation and two separate fixed panels',
        auditor_training_steps=0,auditor_generations=0,issues=issues,
        provenance='Independent same-team retained-data and source audit, not blinded external review.')
    try:
        plan=read(raw/'sealed/PLAN.json')
        result['training']=audit_training(raw,plan,issues)
        panels={p['name']:audit_panel(raw,p,issues) for p in plan['panels']}
        result['panels']=panels
        captures=[p for panel in panels.values() for p in panel['native_captures']]
        check(len(captures)==len(set(captures))==768 and set(captures)==
            {p.name for p in (raw/'campaign_001/native').glob('request_*.json')},issues,'exact 768 unique worker captures across both panels')
        result['native_blocks']=sum(p['reconstructed_native_blocks'] for p in panels.values())
        result['integrity']=audit_integrity(raw,plan,issues)
    except (OSError,KeyError,TypeError,ValueError,IndexError,RuntimeError,StopIteration) as error:
        issues.append(type(error).__name__+': '+str(error))
    result['status']='passed' if not issues else 'issues_found'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,default=HERE/'raw')
    args=parser.parse_args()
    result=run_audit(args.raw)
    (HERE/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],issues=result['issues'],native_blocks=result.get('native_blocks'),
        panels={name:p['comparison'] for name,p in result.get('panels',{}).items()}),indent=2))
    raise SystemExit(bool(result['issues']))
