"""Fixed O/K/KW events9--12, original native interfaces, no automatic retry."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_events_5_8.campaign import collect_pool as _native_collect_pool, verify_pool, identity, publish_checked
from .prepare import NEW_BUDGET, QUALIFICATION_BUDGET, ARMS
from .resume import (OKContinuationTrainer, load_snapshot, verify_live_resume,
                     validate_event_receipt, verify_saved_boundary)
ARM_OBJECTIVES={'O':'osd','K':'osd_first_error','KW':'osd_first_error'}


def validate_plan(plan):
    from .evaluation_protocol import validate_panels
    if plan.get('new_budget') != NEW_BUDGET or plan.get('qualification_budget') != QUALIFICATION_BUDGET:
        raise ValueError('Fixed production or qualification budget differs')
    if [w['event_index'] for w in plan['windows']] != [9,10,11,12]:
        raise ValueError('Only ordered events9--12 are authorized')
    if set(plan['initial']) != set(ARMS): raise ValueError('Require original own O8/K8 and exact K8-to-KW source')
    for arm,source in plan['initial'].items():
        if not all(source.get(k) for k in ('state','export','receipt','state_identity')):
            raise ValueError('Incomplete original state binding: '+arm)
    kw=dict(plan['initial']['KW']); migration=kw.pop('migration',None)
    if kw!=plan['initial']['K'] or migration!='exact_complete_K8_to_KW_v1':
        raise ValueError('KW must migrate one exact complete K8 parent')
    if (plan.get('stop_step')!=1536 or plan.get('inherited_horizon')!=7424
            or plan.get('no_automatic_retry') is not True or plan.get('evaluation_events')!=[10,12]
            or plan.get('generation_warmups')!=0):
        raise ValueError('Preserve fixed stop,horizon,endpoints,and no retries/warmups')
    panel_check=validate_panels(plan['panels'])
    eval_ids={r['request_id'] for p in plan['panels'] for r in p['requests']}
    eval_prompts={tuple(r['prompt_ids']) for p in plan['panels'] for r in p['requests']}
    ids,prompts,indices=set(),set(),[]
    for window in plan['windows']:
        if window['event_id']!=f"event_{window['event_index']:03d}" or len(window['requests'])!=128:
            raise ValueError('Require next128 eligible chronological prompts per event')
        for row in window['requests']:
            rid,prompt=row['request_id'],tuple(row['prompt_ids'])
            if rid in ids|eval_ids or prompt in prompts|eval_prompts:raise ValueError('Training/evaluation overlap')
            if (row['max_new_tokens']!=2048 or not prompt or len(prompt)+2048>2272
                    or any(type(token) is not int or token<0 for token in prompt)):
                raise ValueError('Invalid unchanged full-length training prompt')
            ids.add(rid);prompts.add(prompt);indices.append(row['dataset_index'])
    if (indices!=sorted(set(indices)) or indices[0]<plan['source_cursor_before']
            or plan['source_cursor_after']!=indices[-1]+1):
        raise ValueError('Chronological source cursor binding differs')
    return dict(status='passed',windows=[9,10,11,12],new_budget=NEW_BUDGET,
        own_start_steps=1024,own_end_steps=1536,panels=panel_check)


def require_qualification(root):
    record=json.loads((Path(root)/'IMPLEMENTATION_QUALIFICATION.json').read_text())
    if record.get('status')!='passed':raise ValueError('Bounded full-model implementation qualification did not pass')
    plan=json.loads((Path(root)/'sealed/PLAN.json').read_text())
    inputs=json.loads((Path(root)/'sealed/QUALIFICATION_INPUTS.json').read_text())
    expected=dict(source_k8_sha256=plan['initial']['K']['state_identity']['sha256'],
        qualification_inputs_sha256=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        execution_commit=plan['execution_source']['commit'])
    if record.get('bindings')!=expected:
        raise ValueError('Qualification source/input/execution binding differs from sealed production plan')
    for key,ceiling in QUALIFICATION_BUDGET.items():
        value=record.get('work',{}).get(key)
        if type(value) is not int or not 0<=value<=ceiling:
            raise ValueError('Qualification counter missing or outside ceiling: '+key)
    for key in ('full_model_memory_checked','native_neutral_exact','preview_isolation'):
        if record.get(key) is not True:raise ValueError('Implementation qualification gate missing: '+key)
    return record

def collect_pool(backend,window,checkpoint,output,progress,save_progress,storage):
    pool,costs=_native_collect_pool(backend,window,checkpoint,output,progress,save_progress,storage)
    started=time.perf_counter();sources={}
    for path in sorted((output/'responses').glob('*.json')):
        record=json.loads(path.read_text());capture=Path(record['metadata']['capture']['feature_path'])
        sources[str(path)]=identity(path);sources[str(capture)]=identity(capture)
    for path in sorted((output/'features').glob('*.terminal.json')):
        record=json.loads(path.read_text());tensor=Path(record['tensor_path'])
        sources[str(path)]=identity(path);sources[str(tensor)]=identity(tensor)
    pool['retained_source_identities']=sources
    costs['source_retention_hash_s']=time.perf_counter()-started
    write_json(output/'POOL.json',pool);write_json(output/'COSTS.json',costs)
    return pool,costs


def train_event(plan, arm, event_index, parent, pool, root, ordinary=None):
    """Exactly restore this arm, add its next128 steps and close before serving."""
    import torch
    from online_feedback.kw_training import KWContinuationTrainer, verify_kw_live_resume
    output = root / arm
    audit_root = root / f'resume_{event_index:03d}_{arm}'
    audit_root.mkdir()
    started = time.perf_counter()
    if torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    state, receipt = load_snapshot(parent['state'], parent['export'], objective=ARM_OBJECTIVES[arm],
        config_text=Path(plan['config']).read_text(), completed_event=event_index - 1,
        algorithm_arm=arm if arm != 'KW' or event_index > 9 else 'K')
    receipt['state_identity'] = identity(parent['state'])
    write_json(audit_root / 'SAVED.json', receipt)
    # The initial parent must not already contain any newly selected request.
    used = {feature['record_id'] for event in state['events'] for feature in event['features']}
    if used & set(pool['record_ids']):
        raise ValueError('New feedback reuses a request from this own arm history')
    trainer = None
    try:
        trainer_class = KWContinuationTrainer if arm == 'KW' else OKContinuationTrainer
        trainer = trainer_class(plan['config'], output, resume_state=parent['state'],
                                         arm=arm, completed_event=event_index - 1)
        verifier = verify_kw_live_resume if arm == 'KW' else verify_live_resume
        verifier(trainer, state, receipt_path=audit_root / 'LIVE.json')
        del state
        loading_s = time.perf_counter() - started
        loading_peak_allocated = torch.cuda.max_memory_allocated()
        loading_peak_reserved = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = trainer.train_event(f'event_{event_index:03d}', pool['features'], passes=2,
                                     presentation_schedule=ordinary)
        event_wall_s = time.perf_counter() - started
        check = validate_event_receipt(result, event_number=event_index, objective=ARM_OBJECTIVES[arm],
            reference_schedule=ordinary, trainer_status=trainer.status())
        write_json(audit_root / 'EVENT_CHECK.json', check)
        # Every boundary remains complete until its verified successor.
        # Hard links alias latest without another tensor allocation.
        if event_index in (9, 10, 11, 12):
            retained = trainer.events_root / f'event_{event_index:03d}' / 'training_state.pt'
            os.link(trainer.latest, retained)
            result['latest_state_path'] = str(trainer.latest)
            result['full_state_path'] = str(retained)
            result['state_retention'] = 'immutable hard link; latest is atomically replaced'
            write_json(retained.parent / 'RESULT.json', result)
        saved, saved_check = load_snapshot(result['full_state_path'], result['export_path'],
            objective=ARM_OBJECTIVES[arm], config_text=Path(plan['config']).read_text(), completed_event=event_index, algorithm_arm=arm)
        write_json(audit_root / 'SAVED_ENDPOINT.json', saved_check)
        verify_saved_boundary(trainer, saved, audit_root / 'LIVE_BOUNDARY.json')
        del saved
        retained_identities = {result['full_state_path']: identity(result['full_state_path'])}
        retained_identities.update({str(path): identity(path) for path in sorted(Path(result['export_path']).iterdir())
                                   if path.is_file()})
        write_json(audit_root / 'STATE_IDENTITIES.json', retained_identities)
        cost = dict(loading_and_resume_validation_s=loading_s, event_wall_s=event_wall_s,
            **result['costs'], objective_work=result['objective_work'],
            controller_work=result.get('controller_work', result.get('kw', {})),
            loading_peak_allocated_bytes=loading_peak_allocated,
            loading_peak_reserved_bytes=loading_peak_reserved,
            peak_scope='Training-only peak after loading; loading peaks separate; persistence checks add no model passes.',
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            saved_state_validation_s=saved_check.get('wall_s'))
        write_json(audit_root / 'COSTS.json', cost)
        return result, cost
    except BaseException as error:
        failure = dict(error=repr(error), automatic_retry=False)
        if trainer is not None:
            failure['status'] = trainer.status()
        write_json(audit_root / 'FAILED.json', failure)
        raise
    finally:
        if trainer is not None:
            original_failure = (audit_root / 'FAILED.json').exists()
            try:
                trainer.close()
            except BaseException as error:
                write_json(audit_root / 'CLEANUP_ERROR.json', dict(error=repr(error)))
                if not original_failure:
                    raise
            finally:
                write_json(audit_root / 'CLOSED.json', trainer.status())
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def reconcile_work(output,progress):
    latest={a:1024 for a in ARMS};presentations={a:2048 for a in ARMS}
    previews=credits=0
    attempts=dict(previews=0,live_steps=0,credits=0)
    completed=dict(previews=0,live_steps=0,credits=0)
    for event in (9,10,11,12):
        for arm in ARMS:
            folder=Path(output)/f'resume_{event:03d}_{arm}'
            state=None
            if (folder/'CLOSED.json').exists():state=json.loads((folder/'CLOSED.json').read_text())
            elif (folder/'FAILED.json').exists():state=json.loads((folder/'FAILED.json').read_text()).get('status')
            if state is not None:latest[arm],presentations[arm]=state['global_step'],state['presentations']
        # Window rows are emitted after every committed KW step. A failed live
        # optimizer call may remain unknown and is explicitly disclosed below.
        path=Path(output)/'KW'/'events'/f'event_{event:03d}'/'KW_WINDOWS.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row=json.loads(line)
                    previews+=row.get('optimizer_previews',2)
                    credits+=int(row.get('credit_applied',row.get('credit') is not None))
        work_path=Path(output)/'KW'/'events'/f'event_{event:03d}'/'KW_WORK.jsonl'
        if work_path.exists():
            for line in work_path.read_text().splitlines():
                if not line.strip():continue
                row=json.loads(line);kind=row['kind']
                if row['phase']=='started':attempts[kind]+=1
                elif row['phase']=='completed':completed[kind]+=1
                elif row['phase']=='completed_pair' and kind=='previews':completed[kind]+=2
                else:raise ValueError('Unknown KW work-ledger phase')
    if any(completed[k]>attempts[k] for k in attempts):
        raise ValueError('KW completed work exceeds recorded attempts')
    runtime_steps=sum(latest.values())-3072
    accounted_steps=runtime_steps+max(0,completed['live_steps']-(latest['KW']-1024))
    progress.update(arm_counters=latest,arm_presentations=presentations,
        optimizer_steps=accounted_steps,presentations=sum(presentations.values())-6144,
        optimizer_previews=attempts['previews'],credit_opportunities=attempts['credits'],
        kw_completed_window_previews=previews,kw_completed_window_credits=credits,
        kw_work_attempts=attempts,kw_work_completed_lower_bound=completed,
        kw_work_unknown={k:attempts[k]-completed[k] for k in attempts},
        optimizer_steps_runtime_counter=runtime_steps)
    evaluations={}
    for event in (10,12):
        for panel in ('development','additional','fresh','final'):
            path=Path(output)/f'event_{event:03d}'/'evaluation'/panel/'STATUS.json'
            if path.exists():evaluations[f'{event}/{panel}']=json.loads(path.read_text())
    progress['evaluations']=evaluations
    for suffix,key in (('returned','returned_generations'),('attempted','attempted_generations'),
        ('unknown','unknown_generation_attempts'),('valid','completed_valid_generations'),
        ('invalid','invalid_generations'),('pending','pending_canonical_generations')):
        progress['evaluation_'+suffix]=sum(r.get(key,0) for r in evaluations.values())
    progress['feedback_unknown']=int(progress.get('active_feedback') is not None)
    progress['evaluation_unattempted']=4032-progress['evaluation_attempted']
    progress['feedback_unattempted']=512-progress['feedback_attempted']
    progress['training_interrupted']=progress['status']=='failed' and progress['phase'].startswith('training_')
    progress['training_counter_scope']='Completed calls from receipts; interrupted in-flight arithmetic remains unknown, never assumed absent.'


def validate_completed_work(progress):
    """Require exact completed work, rather than only charged attempt totals."""
    expected=dict(optimizer_steps=1536,presentations=3072,feedback_returned=512,
        evaluation_returned=4032,evaluation_valid=4032,optimizer_previews=1024,credit_opportunities=504)
    if any(progress.get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed production work differs from fixed budget: '+str({k:progress.get(k) for k in expected}))
    if progress['arm_counters']!={a:1536 for a in ARMS}:
        raise ValueError('Require complete own final12 states')
    expected_kw=dict(previews=1024,live_steps=512,credits=504)
    for field in ('kw_work_attempts','kw_work_completed_lower_bound'):
        actual=progress.get(field)
        if (not isinstance(actual,dict) or actual!=expected_kw
                or any(type(value) is not int for value in actual.values())):
            raise ValueError('KW completed work is not exact: '+field)
    unknown=progress.get('kw_work_unknown')
    if (not isinstance(unknown,dict) or set(unknown)!=set(expected_kw)
            or any(type(value) is not int or value!=0 for value in unknown.values())):
        raise ValueError('KW work remains unknown; completion is invalid')


def load_boundary_for_resume(plan,output,event):
    """Explicit operator resume only after a complete event; never replay work."""
    if event not in (9,10,11):raise ValueError('Recover only complete event9,10,11; event12 is terminal')
    output=Path(output)
    prior=json.loads((output/'STATUS.json').read_text())
    if prior.get('events_completed')!=list(range(9,event+1)):
        raise ValueError('Resume does not match the last complete committed event')
    if prior.get('active_feedback') is not None or prior['feedback_attempted']!=(event-8)*128:
        raise ValueError('Subsequent feedback attempt exists; do not regenerate or discard partial work')
    if any((output/f'event_{n:03d}').exists() for n in range(event+1,13)):
        raise ValueError('Subsequent event artifacts exist; recovery requires explicit failure analysis')
    retention=output/'retention'/f'event_{event:03d}_RETENTION.json'
    if not retention.exists() or json.loads(retention.read_text())['status']!='completed':
        raise ValueError('Complete idle event-boundary retention receipt required')
    parents=json.loads((output/'CURRENT_PARENTS.json').read_text())
    for arm in ARMS:
        audit=output/f'resume_{event:03d}_{arm}'
        for path,expected in json.loads((audit/'STATE_IDENTITIES.json').read_text()).items():
            if identity(path)!=expected:raise ValueError('Resume endpoint changed: '+path)
        state,check=load_snapshot(parents[arm]['state'],parents[arm]['export'],objective=ARM_OBJECTIVES[arm],
            config_text=Path(plan['config']).read_text(),completed_event=event,algorithm_arm=arm)
        del state
        write_json(audit/'EXPLICIT_RESUME_CHECK.json',check)
    # Preserve stop/failure history before modifying current status.
    prior_path=output/f'STATUS_BEFORE_EXPLICIT_RESUME_{event:03d}.json'
    if prior_path.exists():raise ValueError('Boundary was already resumed; no automatic repeated recovery')
    write_json(prior_path,prior)
    return parents,prior


def run(root,*,resume_after=None):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from online_feedback.qualify_sglang import check_recipe
    from .evaluate import evaluate
    from .storage import StorageLifecycle
    root=Path(root);plan=json.loads((root/'sealed/PLAN.json').read_text())
    plan_check=validate_plan(plan);qualification=require_qualification(root);check_recipe(plan['config'])
    output=root/'campaign_001';backend=None;started=time.perf_counter()
    if resume_after is None:
        output.mkdir(exist_ok=False)
        parents={a:dict(s) for a,s in plan['initial'].items()}
        progress=dict(status='running',phase='loading_native',feedback_attempted=0,feedback_returned=0,
            optimizer_steps=0,presentations=0,evaluation_returned=0,active_feedback=None,
            arm_counters={a:1024 for a in ARMS},events_completed=[],automatic_retry=False,
            extra_optimizer_steps=0,extra_generations=0,additional_controller_model_passes=0)
        costs={'execution_source':plan.get('execution_source'),'qualification':qualification}
    else:
        parents,progress=load_boundary_for_resume(plan,output,resume_after)
        costs=json.loads((output/'COSTS.json').read_text())
        costs.setdefault('previous_process_segment_wall_s',[]).append(costs.get('campaign_wall_s',0))
        progress.update(status='running',phase='loading_native',explicit_resume_after=resume_after)
    def save():write_json(output/'STATUS.json',progress)
    save();write_json(output/'PLAN_CHECK.json',plan_check)
    try:
        storage=StorageLifecycle(plan,output,resume=resume_after is not None)
        backend=SGLangDFlashBackend(plan['target'],plan['draft'],worker_python=plan['worker_python'],
            output_root=output/('native' if resume_after is None else f'native_resume_{resume_after:03d}'),
            serving_gpu='0',context_limit=4096,block_size=16)
        costs['backend_loading_s']=backend.loading_s
        for window in plan['windows']:
            number=window['event_index']
            if resume_after is not None and number<=resume_after:continue
            progress.update(event=number,phase='collecting_feedback');save()
            pool,pool_cost=collect_pool(backend,window,parents['O']['export'],output/window['event_id'],progress,save,storage)
            costs[f'event_{number:03d}_feedback']=pool_cost;ordinary=None
            for arm in ARMS:
                progress['phase']='training_'+arm;save()
                verify_pool(pool);storage.admit_training(arm,number,pool,parents)
                result,cost=train_event(plan,arm,number,parents[arm],pool,output,ordinary=ordinary)
                if arm=='O':ordinary=result
                verify_pool(pool)
                parents[arm]=dict(state=result['full_state_path'],export=result['export_path'],
                    receipt=str(output/arm/'events'/f'event_{number:03d}'/'RESULT.json'))
                progress['optimizer_steps']+=128;progress['presentations']+=256
                progress['arm_counters'][arm]=result['end_step'];save()
                costs[f'event_{number:03d}_{arm}']=cost;write_json(output/'COSTS.json',costs)
            progress['events_completed'].append(number);save()
            if number in (10,12):
                for panel in plan['panels']:
                    if panel['name']=='final' and number!=12:continue
                    progress['phase']=f"evaluation_{number:03d}_{panel['name']}";save()
                    status=evaluate(backend,panel['requests'],{a:parents[a]['export'] for a in ARMS},
                        output/f'event_{number:03d}'/'evaluation'/panel['name'],event_number=number,
                        panel_name=panel['name'],canonical_mode=panel['canonical_mode'],
                        qualification_requests=8 if number==10 and panel['name']=='development' else 0)
                    progress['evaluation_returned']+=status['returned_generations'];save()
            progress['phase']=f'retention_{number:03d}';save()
            storage.after_event(number,parents,pool,evaluated=number in (10,12))
            write_json(output/'CURRENT_PARENTS.json',parents)
            progress['phase']=f'idle_boundary_{number:03d}';save()
        storage.finalize(parents);reconcile_work(output,progress)
        validate_completed_work(progress)
        progress.update(status='completed',phase='finished')
    except BaseException as error:
        progress.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        cleanup_error=None
        if backend is not None:
            try:backend.close()
            except BaseException as error:
                progress['cleanup_error']=repr(error)
                if progress['status']!='failed':progress.update(status='failed',error=repr(error));cleanup_error=error
        reconcile_work(output,progress)
        costs['campaign_wall_s']=time.perf_counter()-started
        costs['accounting']='Shared O feedback once. Three native update trajectories; KW adds isolated optimizer previews and delayed training credit, no model pass. Nested timers are not additive. No transfers/training during serving. All failed/partial work retained; no automatic retry.'
        write_json(output/'COSTS.json',costs);save()
        if cleanup_error:raise cleanup_error


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root')
    parser.add_argument('--resume-after',type=int,choices=(9,10,11));args=parser.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    run(args.root,resume_after=args.resume_after)
