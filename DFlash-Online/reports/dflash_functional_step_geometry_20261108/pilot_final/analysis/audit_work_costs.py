"""Reconcile finalized FSG pilot work and cost receipts using CPU JSON only."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import time

COMMIT = '6440e8ff5af1a845f830d65c884f6711d7e4a687'
ARMS = ('KPLUS', 'FSG', 'SCALAR')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:json.dumps(v, sort_keys=True) if isinstance(v, (dict,list)) else v
                         for k,v in row.items()} for row in rows)


def read_ordinary_ledger(rows, limits):
    started = {k:set() for k in limits}
    completed = {k:set() for k in limits}
    for row in rows:
        kind, number, phase = row['kind'], row['ordinal'], row['phase']
        assert kind in limits and type(number) is int and 1 <= number <= limits[kind]
        if phase == 'started':
            assert number not in started[kind] and number == len(started[kind])+1
            started[kind].add(number)
        else:
            assert phase == 'completed' and number in started[kind] and number not in completed[kind]
            completed[kind].add(number)
    result = {k:dict(attempted=len(started[k]),completed=len(completed[k]),
        unknown=len(started[k]-completed[k]),failed=0,maximum=limits[k]) for k in limits}
    assert all(v['attempted'] == v['completed'] == v['maximum'] for v in result.values())
    return result


def read_diagnostic_ledger(rows):
    limits = dict(adam=0,ordinary_forward=0,ordinary_backward=0,additional_forward=40,
                  additional_backward=4,serving_generation=0)
    counts, pending = Counter(), None
    for row in rows:
        kind,status = row['kind'],row['status']
        assert kind in limits and status in ('attempted','completed','failed')
        if status == 'attempted':
            assert pending is None and counts[kind,status] < limits[kind]
            pending = kind
        else:
            assert pending == kind
            pending = None
        counts[kind,status] += 1
    assert pending is None
    result = {k:dict(attempted=counts[k,'attempted'],completed=counts[k,'completed'],failed=counts[k,'failed'],
        unknown=counts[k,'attempted']-counts[k,'completed']-counts[k,'failed'],maximum=v) for k,v in limits.items()}
    assert all(v['failed']==v['unknown']==0 and v['attempted']==v['completed'] for v in result.values())
    return result


def audit(gsm8k_root, mathqa_root, output):
    before = time.perf_counter()
    roots = dict(gsm8k=Path(gsm8k_root),mathqa=Path(mathqa_root))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    inputs={}; work=[]; costs=[]; states=[]; physical=[]; feedback=[]
    def read(dataset,relative,lines=False):
        p=roots[dataset]/relative; data=p.read_bytes()
        inputs[dataset+':'+str(relative)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        return [json.loads(line) for line in data.splitlines() if line] if lines else json.loads(data)
    def add_work(dataset,event,arm,scope,counts,window=None):
        for kind, values in counts.items():
            work.append(dict(dataset=dataset,event=event,arm=arm,scope=scope,window=window,kind=kind,**values))
    def add_cost(dataset,event,arm,scope,component,value,source):
        assert type(value) in (float,int) and math.isfinite(value) and value>=0,(source,component,value)
        costs.append(dict(dataset=dataset,event=event,arm=arm,scope=scope,component=component,value=value,
            source=str(source),unit='bytes' if component.endswith('_bytes') else 'seconds',
            aggregation='component-specific; nested/overlapping scopes must not be summed'))
    def add_numeric_costs(dataset,event,arm,scope,value,source,prefix=''):
        for key,item in value.items():
            if type(item) in (int,float) and (key.endswith('_s') or key.endswith('_bytes')):
                add_cost(dataset,event,arm,scope,prefix+key,item,source)
    bindings=[]
    for dataset in roots:
        plan=read(dataset,'sealed/PLAN.json');assert plan['execution_source']['commit']==COMMIT
        bindings.append(plan['binding'])
        for event in (5,6):
            base=Path(f'campaign_001/{dataset}/event_{event:03d}')
            for arm in ARMS:
                directory=base/arm
                result=read(dataset,directory/'RESULT.json');status=read(dataset,directory/'STATUS.json')
                startup=read(dataset,directory/'startup/RESULT.json')
                restoration=read(dataset,directory/'startup/EXACT_RESTORATION.json')
                boundary=read(dataset,directory/'BOUNDARY_CHECK.json');resume=read(dataset,directory/'trainer/RESUME_CHECK.json')
                assert result['status']==status['status']==startup['status']=='completed'
                assert restoration['exact_equal'] and startup['optimizer_steps']==0
                assert boundary['status']==resume['status']==boundary['live']['status']=='passed'
                assert boundary['export_exact'] and boundary['historical_receipts_unchanged']
                assert resume['exact_resume'] and boundary['live']['exact_resume']
                assert resume['global_step']==(event-1)*128 and boundary['global_step']==event*128
                assert resume['seed_reset'] is False and resume['no_rng_reset'] is True
                assert not resume['mismatches'] and not boundary['live']['mismatches']
                assert result['optimizer_steps']==128 and result['response_presentations']==256
                assert result['start_step']==resume['global_step'] and result['end_step']==boundary['global_step']
                states.append(dict(dataset=dataset,event=event,arm=arm,start_step=resume['global_step'],end_step=boundary['global_step'],
                    exact_resume=True,export_exact=True,history_unchanged=True,seed_reset=False,
                    state_identity=result['state_identity'],weights_identity=result['weights_identity']))
                for name,limits,scope in (('WORK.jsonl',dict(response_forwards=256,response_backwards=256,optimizer_steps=128),'ordinary_main'),
                        ('startup/WORK.jsonl',dict(response_forwards=2,response_backwards=2,optimizer_steps=0),'disposable_startup')):
                    add_work(dataset,event,arm,scope,read_ordinary_ledger(read(dataset,directory/name,True),limits))
                add_numeric_costs(dataset,event,arm,'event_training_and_publication',result['costs'],directory/'RESULT.json')
                add_numeric_costs(dataset,event,arm,'worker_envelope',status,directory/'STATUS.json')
                add_numeric_costs(dataset,event,arm,'startup',startup,directory/'startup/RESULT.json')
                resource=read(dataset,directory/'RESOURCE_COSTS.json')
                add_numeric_costs(dataset,event,arm,'observed_event_memory_peak',resource,directory/'RESOURCE_COSTS.json')
                physical.append(dict(dataset=dataset,event=event,arm=arm,scope='event_resource_and_compiler_observer',
                    counters=resource,note='Profiler overhead counters describe this observer only; compilation timings are nested and must not be added to event time.'))
                objective=result['objective_work']
                physical.append(dict(dataset=dataset,event=event,arm=arm,scope='ordinary_physical_observer_including_auxiliary_activity',
                    counters=objective.get('work'),note='Overlapping physical instrumentation; not additional response-level work.'))
                physical.append(dict(dataset=dataset,event=event,arm=arm,scope='startup_physical',counters=startup['physical_work']))
                if arm!='KPLUS':
                    geometry=objective['functional_geometry'];gwork=geometry['work'];last=geometry['steps'][-1]['work_cumulative']
                    assert all(gwork[key]==value for key,value in last.items())
                    assert gwork['captured_ordinary_responses']==256
                    count={}
                    for kind,maximum in (('additional_response_forwards',768),('additional_backwards',256)):
                        attempted,completed,failed=(gwork[kind+'_'+s] for s in ('attempted','completed','failed'))
                        assert attempted==completed<=maximum and failed==0
                        count[kind]=dict(attempted=attempted,completed=completed,failed=failed,unknown=0,maximum=maximum)
                    add_work(dataset,event,arm,'main_auxiliary',count)
                    physical.append(dict(dataset=dataset,event=event,arm=arm,scope='main_geometry_physical',counters=gwork))
                    add_numeric_costs(dataset,event,arm,'nested_geometry',geometry['costs'],directory/'RESULT.json')
                if arm=='FSG':
                    observed=read(dataset,directory/'diagnostics/RESULTS.json')
                    assert [r['window'] for r in observed]==[1,32,65,128]
                    for window in (1,32,65,128):
                        location=directory/f'diagnostics/window_{window:03d}'
                        diagnostic=read(dataset,location/'RESULT.json')
                        assert diagnostic['status']=='completed' and diagnostic['restoration_exact'] and diagnostic['physical_restore_hidden_exact']
                        counts=read_diagnostic_ledger(read(dataset,location/'MODEL_WORK.jsonl',True))
                        assert counts['additional_forward']['completed']==diagnostic['response_forwards']
                        assert counts['additional_backward']['completed']==diagnostic['backwards']
                        add_work(dataset,event,arm,'diagnostic',counts,window)
                        physical.append(dict(dataset=dataset,event=event,arm=arm,window=window,
                            scope='diagnostic_physical',counters=diagnostic['actual_work']))
                        add_numeric_costs(dataset,event,arm,'nested_diagnostic_window_'+str(window),diagnostic['costs'],location/'RESULT.json')
            shared=base/'shared';collection=read(dataset,shared/'COLLECTION.json');pool=read(dataset,shared/'POOL.json')
            assert collection['status']=='completed' and len(collection['records'])==len(pool['records'])==128
            reused=2 if event==5 else 0
            assert collection['reused_feedback']==reused
            add_work(dataset,event,'shared','new_feedback',read_ordinary_ledger(read(dataset,shared/'WORK.jsonl',True),
                dict(feedback_generations=128-reused,teacher_preparations=128-reused)))
            add_numeric_costs(dataset,event,'shared','collection_envelope',collection,shared/'COLLECTION.json')
            add_numeric_costs(dataset,event,'shared','nested_collection_publication',collection['publication'],shared/'COLLECTION.json')
            totals=Counter()
            for ordinal,row in enumerate(collection['records']):
                prepared=row['preparation'];assert prepared['status']=='prepared' and not prepared['label_mismatches']
                # Reused packets retain their historical preparation receipts. Keep their costs separate.
                category='reused_historical' if ordinal<reused else 'new'
                for key in ('teacher_preparation_s','feature_persistence_s'):
                    totals[category+'_'+key]+=prepared[key]
                totals[category+'_supervised_positions']+=prepared['supervised_positions']
                totals[category+'_teacher_terminal_forwards']+=prepared['terminal_feature']['target_forward_count']
                if ordinal>=reused:
                    raw=read(dataset,shared/'records'/Path(row['record_path']).name)
                    for key in ('request_latency_s','native_generate_s'):
                        totals['new_'+key]+=raw['metadata'][key]
                    totals['new_cache_maintenance_s']+=raw['metadata']['capture']['cache_maintenance_s']
            feedback.append(dict(dataset=dataset,event=event,shared_records=128,reused_records=reused,new_records=128-reused,**totals))
            for key,value in totals.items():
                if key.endswith('_s'):
                    add_cost(dataset,event,'shared','collection_record_components',key,value,shared/'COLLECTION.json')
        evaluation=read(dataset,f'campaign_001/evaluation/{dataset}/STATUS.json')
        assert evaluation['status']=='completed'
        for key in ('planned_generations','attempted_generations','returned_generations','completed_valid_generations'):
            assert evaluation[key]==1536
        for key in ('invalid_generations','unknown_generation_attempts','unattempted_generations','pending_canonical_generations',
                    'optimizer_steps','feedback_collections','teacher_preparations','extra_generations'):
            assert evaluation[key]==0
        add_work(dataset,6,'all','endpoint',dict(serving_generation=dict(attempted=1536,completed=1536,failed=0,unknown=0,maximum=1536)))
        add_numeric_costs(dataset,6,'all','evaluation_envelope',evaluation,f'campaign_001/evaluation/{dataset}/STATUS.json')
        # Both compact worker envelopes were authenticated in the final snapshot.
        worker=read('mathqa',f'campaign_001/evaluation/{dataset}_WORKER.json')
        assert worker['status']=='completed'
        add_numeric_costs(dataset,6,'all','evaluation_worker',worker,f'campaign_001/evaluation/{dataset}_WORKER.json')
        summary=read(dataset,f'campaign_001/evaluation/{dataset}/SUMMARY.json')
        for arm,totals in summary['method_totals'].items():
            add_numeric_costs(dataset,6,arm,'summed_request_means_serving',totals,f'campaign_001/evaluation/{dataset}/SUMMARY.json')
    assert len(set(bindings))==1
    aggregate=defaultdict(Counter)
    for row in work:
        for key in ('attempted','completed','failed','unknown','maximum'):
            aggregate[row['scope'],row['kind']][key]+=row[key]
    totals=[dict(scope=scope,kind=kind,**v) for (scope,kind),v in aggregate.items()]
    campaign=read('mathqa','campaign_001/STATUS.json');postflight=read('mathqa','POSTFLIGHT.json')
    assert campaign['status']=='completed' and len(campaign['completed_tasks'])==18
    assert campaign['attempted_task'] is None and campaign['unattempted_tasks']==0 and postflight['status']=='passed'
    exit_code=(roots['mathqa']/'EXIT_CODE').read_text().strip();assert exit_code=='0'
    summary=dict(status='passed',execution_commit=COMMIT,binding=bindings[0],event_count=len(states),
        main_adam_steps=aggregate['ordinary_main','optimizer_steps']['completed'],
        main_ordinary_forwards=aggregate['ordinary_main','response_forwards']['completed'],
        main_ordinary_backwards=aggregate['ordinary_main','response_backwards']['completed'],
        main_auxiliary_forwards=aggregate['main_auxiliary','additional_response_forwards']['completed'],
        main_auxiliary_backwards=aggregate['main_auxiliary','additional_backwards']['completed'],
        diagnostic_forwards=aggregate['diagnostic','additional_forward']['completed'],
        diagnostic_backwards=aggregate['diagnostic','additional_backward']['completed'],
        startup_forwards=aggregate['disposable_startup','response_forwards']['completed'],
        startup_backwards=aggregate['disposable_startup','response_backwards']['completed'],
        new_feedback=aggregate['new_feedback','feedback_generations']['completed'],
        reused_feedback=sum(r['reused_records'] for r in feedback),
        serving_generations=aggregate['endpoint','serving_generation']['completed'],
        failed_work=sum(v['failed'] for v in aggregate.values()),unknown_work=sum(v['unknown'] for v in aggregate.values()),
        final_cumulative_steps_per_trajectory=768,initial_cumulative_steps_per_trajectory=512,
        campaign_wall_s=campaign['wall_s'],postflight_status=postflight['status'],exit_code=exit_code,
        analysis_wall_s=time.perf_counter()-before,new_model_work=0)
    write_csv(output/'WORK_BY_SCOPE.csv',work);write_csv(output/'WORK_TOTALS.csv',totals)
    write_csv(output/'COST_COMPONENTS.csv',costs);write_csv(output/'SHARED_FEEDBACK_COSTS.csv',feedback)
    write_json(output/'STATE_CONTINUITY.json',states);write_json(output/'PHYSICAL_COUNTERS.json',physical)
    write_json(output/'WORK_SUMMARY.json',summary);write_json(output/'INPUTS.json',inputs)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gsm8k',required=True);p.add_argument('--mathqa',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();audit(args.gsm8k,args.mathqa,args.output)
