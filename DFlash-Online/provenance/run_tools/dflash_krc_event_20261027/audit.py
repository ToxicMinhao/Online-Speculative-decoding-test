"""Independent KRC record audit: no model, training, generation or remote work."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
SPEC = importlib.util.spec_from_file_location('kc_record_audit',
    RESEARCH/'dflash_kc_event_20261027/audit.py')
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
TRACE = BASE.TRACE
load, lines, check, compare, near = TRACE.load, TRACE.lines, TRACE.check, TRACE.compare, TRACE.near
ARMS = ('K', 'KRC')


def expected_schedule(ordinary):
    """Rebuild reference pairs independently, without calling the executor helper."""
    windows = []
    for pass_index in range(2):
        rows = ordinary['records'][128*pass_index:128*(pass_index+1)]
        for batch in range(64):
            next_batch = (batch + 1) % 64
            a, r = rows[2*batch:2*batch+2], rows[2*next_batch:2*next_batch+2]
            binding = lambda entries: [{k: row[k] for k in ('record_id','pass','sampled_anchors')}
                                       for row in entries]
            windows.append(dict(step=pass_index*64+batch+1, pass_index=pass_index,
                batch_index=batch, reference_batch_index=next_batch,
                learning_record_ids=[row['record_id'] for row in a],
                reference_record_ids=[row['record_id'] for row in r],
                learning=binding(a), reference=binding(r)))
    return windows


def role_summary(details, role, step, eos, issues, *, collect_blocks):
    totals, blocks = BASE.score_window(details, step, eos, issues, collect_blocks=collect_blocks)
    # The shared scorer calls the corrected local state KC; give it this run's
    # actual KRC name in the derived table. It executes no checkpoint or model.
    totals['KRC'] = totals.pop('KC', None)
    blocks = [{('KRC'+key[2:] if key.startswith('KC_') else key): value
               for key,value in row.items()} for row in blocks]
    for row in blocks:
        row['role'] = role
    result = dict(step=step, role=role, K=totals['K'], KRC=totals['KRC'],
                  K_V=details['post_k']['V'])
    if details['post_c'] is not None:
        result.update(KRC_V=details['post_c']['V'],
                      repair_retention=details['post_c']['repair_totals'])
        changes = [row['KRC_prefix_length']-row['K_prefix_length'] for row in blocks]
        result.update(improved=sum(x>0 for x in changes), regressed=sum(x<0 for x in changes),
            tied=sum(x==0 for x in changes), added_prefix_tokens=sum(max(x,0) for x in changes),
            lost_post_K_prefix_tokens=sum(max(-x,0) for x in changes))
    return result, blocks


def training(evidence, output):
    root = evidence.raw/'campaign_001'
    event = evidence.read(root/'KRC/events/event_001/RESULT.json')
    report = evidence.read(root/'CORRECTION_EVIDENCE.json')
    schedule = evidence.read(evidence.raw/'sealed/REFERENCE_SCHEDULE.json')
    if event is None or report is None or schedule is None:
        return {'status': 'unavailable'}
    ordinary = load(RESEARCH/'dflash_osd_one_event_20261025/final_review/raw/campaign_001/trainer/events/event_001/RESULT.json')
    original_manifest = load(RESEARCH/'dflash_osd_one_event_20261025/sealed/MANIFEST.json')
    expected = expected_schedule(ordinary)
    check(schedule['windows'] == expected and schedule['features'] == ordinary['features'],
          evidence.issues, 'Exact independently reconstructed cyclic schedule')
    evaluation_ids = {row['request_id'] for row in original_manifest['eval_requests']}
    for window in expected:
        check(not set(window['learning_record_ids']) & set(window['reference_record_ids']),
              evidence.issues, 'Disjoint learning/reference pair')
        check(not set(window['reference_record_ids']) & evaluation_ids,
              evidence.issues, 'No evaluation reference exposure')
    result = TRACE.training_exposure(event, 1, 'K', evidence.issues, ordinary)
    result['arm'] = 'KRC: K objective with separate reference-response correction'
    compare(report['counts'], dict(K_backward_attempted=256,K_backward_completed=256,K_backward_failed=0,
        Adam_attempted=128,Adam_completed=128,Adam_failed=0,
        presentations_completed=256,windows_completed=128), evidence.issues, 'Exact main counts')
    compare(report['work'], dict(original_response_forwards=256,pre_reference_response_forwards=256,
        post_k_reference_response_forwards=256,post_c_reference_response_forwards=6,
        post_k_learning_response_forwards=6,post_c_learning_response_forwards=6,
        violation_backward_failed=0), evidence.issues, 'Exact 786 core forwards including18diagnostics')
    check(report['work']['violation_backward_completed'] <= 256 and
          report['work']['direct_corrections'] <= 128, evidence.issues, 'Bounded additional backwards/corrections')
    check([s['step'] for s in report['steps']] == list(range(1,129)), evidence.issues, '128 decisions')
    check([p['step'] for p in report['probes']] == [1,64,128], evidence.issues, 'Three probe steps')
    eos = BASE.eos_ids(evidence)
    rows, blocks, probes = [], [], []
    summed_work, summed_costs = {}, {}
    reference_anchors = reference_available = reference_protected = 0
    for summary in report['steps']:
        step = summary['step']
        window = evidence.read(root/f'correction/window_{step:03d}.json')
        if window is None:
            continue
        check(window['summary'] == summary, evidence.issues, f'Step {step} persisted summary')
        details, bound = window['details'], expected[step-1]
        check(details['schedule_binding'] == bound, evidence.issues, f'Step {step} binding')
        check(summary['status']=='completed' and summary['actual_Adam_steps_completed']==step and
              summary['trainer_step_before_return']==step-1, evidence.issues, f'Step {step} native accounting')
        check(all(details['reference_pre_integrity'].values()), evidence.issues, f'Step {step} pre-R integrity')
        for phase, checks in summary['integrity'].items():
            check(checks is True or all(checks.values()), evidence.issues, f'Step {step} {phase} integrity')
        check(details['reference']['correction'] == details['correction'] and
              details['reference']['post_k'] == details['post_k'], evidence.issues,
              f'Step {step} reference alone supplies correction')
        for role in ('reference', 'learning'):
            data = details.get(role)
            if data is None:
                check(role=='learning' and step not in (1,64,128), evidence.issues,
                      f'Step {step} only sparse learning probes absent')
                continue
            check([row['record_id'] for row in data['responses']] == bound[role+'_record_ids'],
                  evidence.issues, f'Step {step} {role} response identities')
            for source, binding in zip(data['responses'], bound[role]):
                draw, = binding['sampled_anchors']
                check(source['anchors']==draw['positions'] and source['keep']==draw['keep_mask'],
                      evidence.issues, f'Step {step} {role} own per-pass anchors')
                if role=='reference':
                    reference_anchors += sum(source['keep'][0])
                    reference_available += sum(BASE.valid_length(labels,validity,eos)
                        for labels,validity in zip(source['labels'][0],source['validity_weights'][0]))
            scored, table = role_summary(data,role,step,eos,evidence.issues,collect_blocks=step in (1,64,128))
            if role=='reference':
                reference_protected += scored['K']['protected_positions']
            if step in (1,64,128):
                probes.append(scored)
                blocks.extend(table)
        rows.append(dict(step=step,**details['correction'],**summary['costs'],integrity_s=summary['integrity_s']))
        for key,value in details['work'].items():
            summed_work[key]=summed_work.get(key,0)+value
        for key,value in summary['costs'].items():
            summed_costs[key]=summed_costs.get(key,0)+value
        summed_costs['integrity_s']=summed_costs.get('integrity_s',0)+summary['integrity_s']
        if step in (1,64,128):
            saved_probe=next(p for p in report['probes'] if p['step']==step)
            check(saved_probe=={'step':step,**details},evidence.issues,f'Step {step} probe identity')
        if step==1:
            prior=load(RESEARCH/'dflash_prefix_correction_step_20261027/raw/execution_001/RESULT.json')['core']
            check(details['learning']['responses']==prior['responses'] and
                  details['learning']['post_k']['responses']==prior['post_k']['responses'],
                  evidence.issues,'Exact first learning-pair ordinary K reproduction only')
    check(summed_work==report['work'],evidence.issues,'Window work sums')
    check(all(near(v,report['costs'][k]) for k,v in summed_costs.items()),evidence.issues,'Window cost sums')
    BASE.write_csv(output/'CORRECTION_STEPS.csv',rows)
    BASE.write_csv(output/'DUAL_ROLE_PROBE_BLOCKS.csv',blocks)
    result.update(correction_steps=len(rows),active=sum(r['status']=='applied' for r in rows),
        work=report['work'],counts=report['counts'],correction_costs=report['costs'],
        probes=probes,event_costs=event['costs'],reference_anchors=reference_anchors,
        reference_available_occurrences=reference_available,reference_protected_occurrences=reference_protected,
        eos_token_ids=sorted(eos))
    return result


def evaluation(evidence):
    root = evidence.raw / 'campaign_001/evaluation'
    requests = evidence.read(root / 'REQUESTS.json')
    protocol = evidence.read(root / 'PROTOCOL.json')
    if requests is None or protocol is None:
        return {'status':'unavailable'}
    original=load(RESEARCH/'dflash_osd_one_event_20261025/sealed/MANIFEST.json')['eval_requests']
    check(requests == [dict(r,panel_ordinal=r.get('panel_ordinal',i)) for i,r in enumerate(original)],evidence.issues,'Exact original64 requests/canonicalF')
    checkpoints=protocol['checkpoints']
    retained_k=load(RESEARCH/'dflash_first_error_execution_20261026/raw/campaign_001/K/events/event_001/RESULT.json')
    trained_kc=evidence.read(evidence.raw/'campaign_001/KRC/events/event_001/RESULT.json')
    if trained_kc:
        check(checkpoints=={'K':retained_k['checkpoint_path'],'KRC':trained_kc['checkpoint_path']},
              evidence.issues,'Exact retained K1/new KRC export identities')
    expected=[(r,n,a) for r in requests for n in range(3)
              for a in (ARMS if (r['panel_ordinal']+n)%2==0 else ARMS[::-1])]
    rows=lines(root/'MEASUREMENTS.jsonl')
    check(len(rows)==384 and [(r['request_id'],r['repeat'],r['method']) for r in rows] ==
          [(r['request_id'],n,a) for r,n,a in expected],evidence.issues,'Exact384 alternating slots')
    valid_rows=[]
    for serial,(row,(request,repeat,arm)) in enumerate(zip(rows,expected),1):
        label=f"{request['panel_ordinal']:04d}_r{repeat}_{arm}"
        record=evidence.read(root/(label+'.json'))
        if record is None:continue
        before=len(evidence.issues)
        counts=TRACE.native_record(record,request,checkpoints[arm],capture=False,
            session='reference_correction_evaluation',label=label,index=request['panel_ordinal'],issues=evidence.issues)
        TRACE.bind_native(evidence,record,serial)
        compare(row,dict(**counts,label=label,validation_status='valid',valid_measurement=True,
                         structural_errors=[],errors=[],finish_reason=request['finish_reason']),evidence.issues,label+' row')
        for name in ('request_latency_s','native_generate_s'):
            check(row[name]==record['metadata'][name],evidence.issues,label+' '+name)
        if len(evidence.issues)==before:valid_rows.append(row)
    summary=evidence.read(root/'report/SUMMARY.json')
    cohort=[]
    for request in requests:
        own=[r for r in valid_rows if r['request_id']==request['request_id']]
        complete=len(own)==6 and {(r['method'],r['repeat']) for r in own}=={(a,n) for a in ARMS for n in range(3)}
        consistent=all(len({tuple(r[k] for k in TRACE.COUNTS) for r in own if r['method']==a})<=1 for a in ARMS)
        check(consistent,evidence.issues,request['request_id']+' repeat counts')
        if complete and consistent:cohort.extend(own)
    totals={}
    for arm in ARMS:
        own=[r for r in cohort if r['method']==arm]
        totals[arm]={k:sum(r[k] for r in own if r['repeat']==0) for k in TRACE.COUNTS}
        totals[arm]['summed_mean_request_latency_s']=sum(r['request_latency_s'] for r in own)/3
        if summary:
            for name,value in totals[arm].items():check(near(value,summary[arm][name]),evidence.issues,'Pooled '+arm+'/'+name)
    pairs=[(next(r for r in cohort if r['request_id']==req['request_id'] and r['method']=='K' and r['repeat']==0),
            next(r for r in cohort if r['request_id']==req['request_id'] and r['method']=='KRC' and r['repeat']==0))
           for req in requests if sum(r['request_id']==req['request_id'] for r in cohort)==6]
    directions={name:sum((k['verification_calls']>g['verification_calls'] if name=='improved' else
                         k['verification_calls']<g['verification_calls'] if name=='regressed' else
                         k['verification_calls']==g['verification_calls']) for k,g in pairs)
                for name in ('improved','regressed','tied')}
    ratios={name:100*(1-totals['KRC'][key]/totals['K'][key]) if totals['K'][key] else None
            for name,key in [('call_reduction_pct','verification_calls'),('request_time_reduction_pct','summed_mean_request_latency_s')]}
    if summary:
        for name,value in ratios.items():check(near(value,summary['comparisons']['KRC_vs_K'][name]),evidence.issues,'Pooled ratio '+name)
        check(directions==summary['comparisons']['KRC_vs_K']['call_request_directions'],evidence.issues,'Request directions')
    activations = lines(root / 'ACTIVATIONS.jsonl')
    last = None
    expected_activation = []
    for request,repeat,arm in expected:
        if arm != last:
            expected_activation.append((request['panel_ordinal'], repeat, arm))
            last = arm
    check([(r['panel_ordinal'],r['repeat'],r['method']) for r in activations]==expected_activation,evidence.issues,'Publication schedule')
    for row in activations:
        check(not row['errors'],evidence.issues,'Publication errors')
        TRACE.publication(row,checkpoints[row['method']],evidence.issues,'Publication')
    for arm in ARMS:
        record=evidence.read(root/f'WEIGHTS_{arm}.json')
        if record:TRACE.tensor_equality(record['native_reply'],checkpoints[arm],evidence.issues,arm+' weights')
    status=evidence.read(root/'STATUS.json')
    if status:
        compare(status,dict(status='completed',planned_generations=384,attempted_generations=384,
            returned_generations=384,completed_valid_generations=384,invalid_generations=0,
            unknown_generation_attempts=0,unattempted_generations=0,optimizer_steps=0,feedback_collections=0),evidence.issues,'Evaluation accounting')
        check(near(status['activation_s'],sum(r['activation_s'] for r in activations)),evidence.issues,'Activation cost once')
    return dict(valid_requests=len(pairs),returned=len(rows),native_blocks=evidence.native_blocks,
                totals=totals,comparisons=ratios,directions=directions,publications=len(activations),execution=status)


def audit(root=HERE):
    root=Path(root); output=root/'analysis'; output.mkdir(exist_ok=True)
    evidence=TRACE.Evidence(root/'raw');raw=evidence.raw
    retrieval=evidence.read(raw/'RETRIEVAL_MANIFEST.json')
    for name,expected in (retrieval or {}).items():
        path=raw/name
        if not path.is_file():
            evidence.missing.append(str(path));continue
        actual=TRACE.identity(path)
        check(actual['bytes']==expected.get('bytes',expected.get('size')) and
              actual['sha256']==expected['sha256'],evidence.issues,'Retrieval '+name)
    seal=evidence.read(raw/'sealed/SEAL.json');plan=evidence.read(raw/'sealed/PLAN.json')
    if seal and plan:
        bindings={
            plan['ordinary_receipt']:RESEARCH/'dflash_osd_one_event_20261025/final_review/raw/campaign_001/trainer/events/event_001/RESULT.json',
            plan['manifest']:RESEARCH/'dflash_osd_one_event_20261025/sealed/MANIFEST.json',
            plan['k_receipt']:RESEARCH/'dflash_first_error_execution_20261026/raw/campaign_001/K/events/event_001/RESULT.json',
            plan['reference_diagnostic']:RESEARCH/'dflash_prefix_correction_step_20261027/raw/execution_001/RESULT.json'}
        for remote,local in bindings.items():
            check(TRACE.identity(local)==seal['files'][remote],evidence.issues,'Original input '+remote)
    for name in ('PREFLIGHT.json','POSTFLIGHT.json'):
        record=evidence.read(raw/name)
        if record and seal:
            check(record['status']=='passed' and record['files_checked']==len(seal['files']),
                  evidence.issues,name+' unchanged seal')
    status=evidence.read(raw/'campaign_001/STATUS.json')
    costs=evidence.read(raw/'campaign_001/COSTS.json')
    completed=status and status['status']=='completed'
    if completed:
        compare(status,dict(main_optimizer_steps=128,main_presentations=256,
            native_adam_steps_completed=128,native_K_presentations_completed=256,
            new_qualification_optimizer_steps=0,new_qualification_presentations=0,
            new_feedback_generations=0),evidence.issues,'Main budget')
        first=evidence.read(raw/'campaign_001/FIRST_WINDOW_REPRODUCTION.json')
        check(first and first['exact_equal'],evidence.issues,'First learning-pair native gate')
        for name in ('INITIAL_VS_PIN','FRESH_CORE'):
            record=evidence.read(raw/f'campaign_001/initialization/{name}.json')
            check(record and record['exact_equal'],evidence.issues,'Exact fresh state '+name)
    if costs:
        check(costs['new_qualification_optimizer_steps']==0,evidence.issues,'No extra qualification steps')
        if 'startup' in costs:
            compare(costs['startup'],dict(status='passed',optimizer_steps=0,response_evaluations=2,
                backwards=2,exact_initial_state_restored=True),evidence.issues,'Separate startup')
    train=evidence.section('Training',lambda:training(evidence,output)) if completed else {}
    serving=evidence.section('Evaluation',lambda:evaluation(evidence)) if completed else {}
    if costs and train.get('event_costs'):
        check(train['event_costs']==costs['event'],evidence.issues,'Event costs match')
        nested=train['correction_costs']
        components=('pre_reference_s','post_k_s','correction_s','post_c_reference_s',
            'post_k_learning_s','post_c_learning_s','pre_reference_integrity_s','integrity_s',
            'first_learning_gate_s')
        check(sum(nested.get(k,0) for k in components)<=costs['event']['training_s']<=
              costs['event_wall_s']<=costs['actual_campaign_wall_s'],evidence.issues,'Nested cost accounting')
    result=dict(status=('passed' if completed and not evidence.issues and not evidence.missing
                        else 'failed_or_incomplete'),experiment_status=status,issues=evidence.issues,
        missing=evidence.missing,verified_retrieved_files=len(retrieval or {}),training=train,
        evaluation=serving,costs=costs,
        scope='Independent compact-evidence reconstruction only; no new model work.',
        interpretation='References are additional training exposure. Reused development panel. Historical KC is not a fresh paired timing control.')
    (root/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    result=audit(sys.argv[1] if len(sys.argv)>1 else HERE)
    print(json.dumps({k:v for k,v in result.items() if k not in ('training','evaluation','costs')},indent=2))
    sys.exit(0 if result['status']=='passed' else 1)
