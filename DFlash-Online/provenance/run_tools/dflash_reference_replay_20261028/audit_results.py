"""Independent retained-data KR audit. No Torch, model execution, network or raw writes."""
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
ARMS = ('KR', 'KRC')
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
    comparison = dict(candidate='KR', reference='KRC')
    for label, metric, reduction in [('call', 'verification_calls', 'call_reduction_pct'),
                                     ('latency', 'request_latency_s', 'request_time_reduction_pct')]:
        before = [values[r['request_id'], 'KRC'][metric] for r in requests]
        after = [values[r['request_id'], 'KR'][metric] for r in requests]
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
    requests = [dict(r, panel_ordinal=r.get('panel_ordinal', i), retained_krc_calls=panel['retained_krc_counts'][r['request_id']])
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
              request['prompt_ids']==original_native['prompt_ids'] and
              request['retained_krc_calls']==original_row['verification_calls'],
              issues,name+'/'+request['request_id']+': original canonical/input/count provenance')
    rows, protocol, status = lines(directory/'MEASUREMENTS.jsonl'), read(directory/'PROTOCOL.json'), read(directory/'STATUS.json')
    checkpoints = protocol['checkpoints']
    check(read(directory/'REQUESTS.json') == requests, issues, name + ': sealed inputs unchanged')
    check(read(directory/'RETAINED_KRC_COUNTS.json') == panel['retained_krc_counts'], issues, name + ': retained count identity')
    compare(protocol, dict(arms=list(ARMS), comparisons=[['KR','KRC']], repeats=3,
        rotation='(panel_ordinal + repeat) % 2', count_trajectory_repeat=0,
        historical_KRC_count_gate=True, capture=False, extra_warmup_generations=0), issues, name + ': protocol')
    expected = slots(requests)
    check([(r['request_id'],r['repeat'],r['method']) for r in rows] == expected and len(rows)==384,
          issues, name + ': exact 384 chronological slots')
    for repeat in range(3):
        first = [arm for i,(_,r,arm) in enumerate(expected) if r==repeat and i%2==0]
        check(Counter(first)==Counter(KR=32,KRC=32), issues, name + ': balanced repeat order')
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
            record['session_id']=='reference_replay_evaluation',issues,label+': native identity')
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
        if arm=='KRC':check(derived['verification_calls']==req['retained_krc_calls'],issues,label+': retained KRC count')
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
    check({p.name for p in directory.iterdir() if re.fullmatch(r'\d{4}_r[012]_(?:KR|KRC)\.json',p.name)}=={r['label']+'.json' for r in rows},issues,name+': no extra complete-response records')
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
        returned_measurements=384,valid_timed_generations=384,primary_comparison='KR_vs_KRC',
        timing_repeats_per_method_request=3,**totals,comparisons={'KR_vs_KRC':comparison}),issues,name+': reported aggregates')
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
        repeat_totals.append(dict(repeat=repeat,**by_arm,request_time_reduction_pct=100*(1-by_arm['KR']['request_latency_s']/by_arm['KRC']['request_latency_s'])))
    publication=audit_publication(directory,requests,checkpoints,status,issues)
    return dict(records=len(rows),requests=len(requests),native_captures=sorted(native_paths),
        retained_source=str(retained_root),
        reconstructed_native_blocks=total_blocks,totals=totals,comparison=comparison,
        accepted_count_histograms_repeat0={a:dict(histograms[a]) for a in ARMS},
        repeat_totals=repeat_totals,publication=publication,status=status)


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


def active_anchors(row):
    for draw in row['sampled_anchors']:
        for positions,keep in zip(draw['positions'],draw['keep_mask'],strict=True):
            for position,valid in zip(positions,keep,strict=True):
                if valid:yield position


def audit_training(raw,plan,issues):
    root=raw/'campaign_001';event=root/'KR/events/event_001'
    receipt=read(event/'RESULT.json');schedule=read(raw/'sealed/REPLAY_SCHEDULE.json')
    original=read(mapped(plan['ordinary_receipt'],raw));retained_k=read(mapped(plan['k_receipt'],raw))
    manifest=read(mapped(plan['manifest'],raw))
    training_ids={r['request_id'] for r in manifest['train_requests']}
    training_prompts={tuple(r['prompt_ids']) for r in manifest['train_requests']}
    evaluation_requests=[r for p in plan['panels'] for r in p['requests']]
    check(len(training_ids)==len(training_prompts)==128 and
        not training_ids.intersection(r['request_id'] for r in evaluation_requests) and
        not training_prompts.intersection(tuple(r['prompt_ids']) for r in evaluation_requests),
        issues,'evaluation inputs excluded from fixed training pool by IDs and tokens')
    compare(receipt,dict(status='completed',training_variant='KR',objective='osd_first_error',passes=2,
        response_presentations=512,optimizer_steps=256,start_step=0,end_step=256,reduction='response_mean'),issues,'KR training receipt')
    check(receipt['features']==schedule['features']==original['features']==retained_k['features'],issues,'one unchanged 128-feedback feature pool')
    check(len(receipt['records'])==512 and len(schedule['updates'])==256,issues,'512 presentations/256 bindings')
    counts=Counter();anchors=Counter();available=Counter();exposure=Counter();features={r['record_id']:r for r in receipt['features']}
    # Rebuild A/R schedule from the original O receipt, independently of its builder.
    for update_index,update in enumerate(schedule['updates']):
        step=update_index+1;pass_index=update_index//128;pair=(update_index%128)//2
        role='A' if update_index%2==0 else 'R';source_pair=pair if role=='A' else (pair+1)%64
        compare(update,dict(optimizer_step=step,outer_window=pass_index*64+pair+1,role=role,
            pass_index=pass_index,batch_index=pair,source_pair_index=source_pair),issues,f'update {step}: schedule')
        expected_original=original['records'][pass_index*128+source_pair*2:pass_index*128+source_pair*2+2]
        for offset,(binding,old) in enumerate(zip(update['records'],expected_original,strict=True)):
            row=receipt['records'][update_index*2+offset]
            for key in ('record_id','pass','sampled_anchors'):
                check(row[key]==binding[key]==old[key],issues,f'update {step}: own original '+key)
            compare(row,dict(role=role,outer_window=pass_index*64+pair+1,
                scheduled_optimizer_step=step,optimizer_stepped=offset==1,
                optimizer_step_after=step-int(offset==0),learning_rate=0.0003 if offset else None,
                metrics_scope='accumulation_window' if offset else 'presentation'),issues,f'update {step}: row accounting')
            check(all(type(v) in (int,float) and math.isfinite(v) for v in row['metrics'].values()),issues,'finite training metrics')
            counts[row['record_id'],role]+=1
            feature=features[row['record_id']]
            for anchor in active_anchors(row):
                anchors[role]+=1
                check(feature['prompt_length']<=anchor<feature['length']-1,issues,'eligible retained anchor')
                available[role]+=sum(feature['prompt_length']<=anchor+j<feature['length'] for j in range(1,16))
            if offset==1:
                for key,value in row['metrics'].items():
                    if key in ('available_position_occurrences','selected_position_occurrences','supervised_position_exposure','no_error_blocks','empty_blocks') or key.startswith('first_error_position_'):
                        exposure[role,key]+=value
    check(len(features)==128 and all(counts[r,role]==2 for r in features for role in ('A','R')),issues,'each response exactly two A and two R presentations')
    check(anchors==Counter(A=65096,R=65096),issues,'130192 original-anchor occurrences')
    for role in ('A','R'):
        check(exposure[role,'available_position_occurrences']==available[role],issues,role+': full-position exposure reconstruction')
        check(exposure[role,'selected_position_occurrences']==exposure[role,'supervised_position_exposure']<=available[role],issues,role+': selected exposure')
        check(sum(v for (r,k),v in exposure.items() if r==role and (k.startswith('first_error_position_') or k in ('no_error_blocks','empty_blocks')))==anchors[role],issues,role+': block outcomes sum to anchors')
    replay=receipt['reference_replay'];work=replay['counts']
    expected_work=dict(response_forwards_attempted=512,response_forwards_completed=512,backward_attempted=512,
        backward_completed=512,Adam_attempted=256,Adam_completed=256,presentations_completed=512,
        updates_completed=256,outer_windows_completed=128,shuffle_rng_draws=2,anchor_rng_draws=512,first_A_checks_completed=1)
    for role in ('A','R'):
        expected_work.update({role+'_presentations_completed':256,role+'_response_forwards_completed':256,role+'_backward_completed':256,role+'_Adam_completed':128})
    compare(work,expected_work,issues,'actual training work')
    check(not any(v for k,v in work.items() if k.endswith('_failed')),issues,'no failed core work')
    compare(replay,dict(status='completed',arm='KR',global_step=256,presentations=512,correction_steps=0,protected_capture=False,reference_model=False),issues,'ordinary-only KR')
    check([u['native_Adam_step'] for u in replay['updates']]==list(range(1,257)),issues,'continuous native Adam counters')
    check([u['role'] for u in replay['updates']]==['A','R']*128,issues,'continuous A/R optimizer order')
    check([p['pass_index'] for p in replay['pass_draws']]==[0,1]
        and replay['pass_draws'][0]['ordinary_shuffle_matches_sealed'],issues,'ordinary RNG draw at each pass and exact first shuffle')
    check(all(row['anchor_rng_draw_matches_sealed']==[True] for row in receipt['records'][:2]),issues,'first-A sampler RNG matches retained draws')
    check(all(len(row['anchor_rng_draw_matches_sealed'])==1 for row in receipt['records']),issues,'one ordinary sampler RNG draw per actual presentation')
    first_fields=('record_id','pass','sampled_anchors','optimizer_stepped','optimizer_step_after','learning_rate','metrics','metrics_scope')
    check(all(a[k]==b[k] for a,b in zip(receipt['records'][:2],retained_k['records'][:2],strict=True) for k in first_fields),issues,'exact retained first-A training records')
    first=read(root/'FIRST_A_REPRODUCTION.json')
    check(first['exact_equal'] and not first['tensor_differences'] and not first['metadata_differences'],issues,'first-A comparison receipt')
    ar=read(event/'FIRST_AR_WINDOW.json')
    compare(ar,dict(status='passed',native_Adam_steps=[1,2],first_A_record_gate_passed=True,
        continuous_optimizer=True,rng_restored_between_updates=False,additional_forwards=0,additional_optimizer_steps=0),issues,'first actual A/R window')
    compare(ar['actual_work'],ar['expected_work'],issues,'first A/R work')
    init=read(root/'INITIALIZATION.json')
    compare(init,dict(exact_pinned_initial_state=True,full_rng_restored_after_construction=True,fresh_optimizer=True,inherited_qualification_learning=False),issues,'fresh KR initialization')
    for name in ('FRESH_CORE.json','INITIAL_VS_PIN.json'):
        exact=read(root/'initialization'/name)
        check(exact['exact_equal'] and not exact['tensor_differences'] and not exact['metadata_differences'],issues,'initial tensors/RNG '+name)
    startup=read(root/'startup/STARTUP.json')
    compare(startup,dict(status='passed',response_evaluations=2,backwards=2,optimizer_steps=0,
        exact_initial_state_restored=True,separate_disposable_runtime=True,runtime_closed_before_K_construction=True),issues,'separate startup')
    compare(receipt['objective_work'],dict(reference_setup_s=0.,reference_forward_s=0.,reference_head_s=0.,projection_s=0.),issues,'no protection/reference/correction work')
    # Journals are separate durable evidence, not additional training presentations.
    check(lines(event/'REPLAY_PRESENTATIONS.jsonl')==receipt['records'],issues,'durable presentation journal equals receipt')
    compare(read(root/'TRAINING_CHECK.json'),dict(passed=True),issues,'campaign training gate')
    return dict(presentations=512,optimizer_steps=256,distinct_responses=128,
        anchors=dict(anchors),total_anchors=sum(anchors.values()),available_positions=dict(available),
        exposure={role:{k:v for (r,k),v in exposure.items() if r==role} for role in ('A','R')},
        actual_work=work,first_A=first['exact_equal'],first_AR=ar['status'],fresh_initialization=init,
        startup=startup,costs=receipt['costs'],objective_work=receipt['objective_work'],replay_costs=replay['costs'],
        first_A_scope='Exact initial tensors/state plus exact first-A records; not retained post-step tensor equality.',
        rng_scope='Source and native receipts show one continuous RNG stream; only initial and first-A draws must match the retained path.')


def audit_integrity(raw,plan,issues):
    seal=read(raw/'sealed/SEAL.json');pre=read(raw/'PREFLIGHT.json');post=read(raw/'POSTFLIGHT.json')
    check(seal['sealed_before_training'] and len(seal['files'])==722,issues,'722 sealed source/input files before training')
    for name,receipt in [('preflight',pre),('postflight',post)]:
        check(receipt['status']=='passed' and receipt['files_checked']==len(seal['files']),issues,name+': complete sealed files check')
    storage=pre['storage']
    check(storage['admitted'] and storage['free_bytes']>=storage['required_bytes']
        and storage['reserve_bytes']==8*2**30,issues,'peak storage admission and unchanged reserve')
    runtime=read(raw/'sealed/RUNTIME_PIN.json')
    for receipt in (pre,post):
        compare(receipt['runtime'],{k:runtime[k] for k in ('python_version','torch','transformers','cuda','native_runtime_sha256')},issues,'pinned native runtime')
        check(receipt['training_runtime']['torch']=='2.13.0+cu130' and
            receipt['training_runtime']['transformers']=='5.12.1',issues,'pinned training runtime')
    local_count=0;remote_only=[]
    for name,identity in seal['files'].items():
        local=mapped(name,raw)
        if local is None:remote_only.append(name);continue
        check(local.stat().st_size==identity['bytes'] and digest(local)==identity['sha256'],issues,'sealed local identity: '+str(local))
        local_count+=1
    check(int((raw/'EXIT_CODE').read_text())==0,issues,'supervisor exit0')
    status=read(raw/'campaign_001/STATUS.json');costs=read(raw/'campaign_001/COSTS.json')
    compare(status,dict(status='completed',phase='finished',main_optimizer_steps=256,main_presentations=512,
        outer_windows_completed=128,new_feedback_generations=0,new_qualification_optimizer_steps=0,
        new_qualification_presentations=0,automatic_retry=False),issues,'completed campaign budget')
    compare(status['evaluation_totals'],dict(planned_generations=768,attempted_generations=768,
        returned_generations=768,completed_valid_generations=768,invalid_generations=0,
        unknown_generation_attempts=0,unattempted_generations=0),issues,'total native budget')
    check(all(type(costs[k]) in (int,float) and math.isfinite(costs[k]) and costs[k]>0 for k in
        ('event_wall_s','loading_and_initialization_s','backend_loading_s','actual_campaign_wall_s')),issues,'finite campaign costs')
    engine=read(raw/'campaign_001/native/engine_settings.json')
    compare(engine,dict(model_path=plan['target'],speculative_draft_model_path=plan['draft'],
        speculative_algorithm='DFLASH',speculative_num_draft_tokens=16,dtype='bfloat16',attention_backend='fa3',
        tp_size=1,max_running_requests=1,context_length=4112,chunked_prefill_size=8192,
        random_seed=17,skip_server_warmup=True),issues,'qualified native runtime')
    compare(read(raw/'campaign_001/TRAINER_CLOSED.json'),dict(closed=True,failed=False,
        global_step=256,presentations=512,completed_events=['event_001'],
        reduction='response_mean',objective='osd_first_error',accumulation_remainder=0,horizon=7424),issues,'training closed with continuous qualified horizon')
    retrieval=read(raw/'RETRIEVAL_MANIFEST.json')
    retrieved_bytes=0
    for name,identity in retrieval.items():
        local=raw/name
        check(local.is_file() and local.stat().st_size==identity['bytes'] and digest(local)==identity['sha256'],issues,'retrieved identity: '+name)
        retrieved_bytes+=identity['bytes']
    return dict(sealed_files=len(seal['files']),local_files_rehashed=local_count,remote_only_files=len(remote_only),
        retrieved_files_rehashed=len(retrieval),retrieved_bytes=retrieved_bytes,
        source_integrity_basis='Every local counterpart rehashed, including unchanged author training; remote pre/postflight for nonlocal tensors.',
        preflight=pre,postflight=post,costs=costs,status=status)


def run_audit(raw):
    raw=Path(raw);issues=[]
    result=dict(verification_status='ANALYZED',scope='one ordinary reference-replay event and two separate fixed panels',
        auditor_training_steps=0,auditor_generations=0,issues=issues,
        provenance='Independent same-team retained-data/source audit, not blinded external review.')
    try:
        plan=read(raw/'sealed/PLAN.json')
        result['training']=audit_training(raw,plan,issues)
        panels={p['name']:audit_panel(raw,p,issues) for p in plan['panels']}
        result['panels']=panels
        all_paths=[p for panel in panels.values() for p in panel['native_captures']]
        check(len(all_paths)==len(set(all_paths))==768 and set(all_paths)=={p.name for p in (raw/'campaign_001/native').glob('request_*.json')},issues,'exact 768 worker captures across both panels')
        result['native_blocks']=sum(p['reconstructed_native_blocks'] for p in panels.values())
        result['integrity']=audit_integrity(raw,plan,issues)
    except (OSError,KeyError,TypeError,ValueError,IndexError,RuntimeError,StopIteration) as error:
        issues.append(type(error).__name__+': '+str(error))
    result['status']='passed' if not issues else 'issues_found'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,default=HERE/'raw');args=parser.parse_args()
    result=run_audit(args.raw)
    (HERE/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],issues=result['issues'],native_blocks=result.get('native_blocks'),
        panels={name:p['comparison'] for name,p in result.get('panels',{}).items()}),indent=2))
    raise SystemExit(bool(result['issues']))
