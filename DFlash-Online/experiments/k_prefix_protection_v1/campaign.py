"""Four own trajectories, then two fixed serving cells; no collection or retries."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.attempts import reserve_directory, reserve_receipt
from experiments.ok_coding.campaign import fresh_worker_environment, read_work, require_complete_work
from experiments.ok_events_5_8.storage import identity
from .contract import ARMS, SEEDS, validate_plan
from .storage import admit


def read(path):
    return json.loads(Path(path).read_text())


def limits(arm):
    result=dict(forwards=256,backwards=256,optimizer_steps=128)
    if arm=='KP': result['reference_forwards']=256
    return result


def job_paths(root,seed,arm,event):
    base=Path(root)/'campaign_001'/f'seed_{seed}'
    return base/f'resume_{event:03d}_{arm}', base/arm/'events'/f'event_{event:03d}'


def completed_job(root,plan,seed,arm,event, *, decode=False):
    from experiments.ok_coding.training import validate_event_receipt, _export_equal
    audit,endpoint=job_paths(root,seed,arm,event)
    worker=read(audit/'WORKER.json')
    if worker['status']!='completed' or read(audit/'EXEC.json')['returncode']!=0:
        raise ValueError('Incomplete work cannot be retried: '+str(audit))
    require_complete_work(read_work(audit/'WORK.jsonl',limits(arm)),limits(arm))
    result=read(endpoint/'RESULT.json')
    ordinary=read(plan['original_schedules'][str(seed)][str(event)]['path'])
    validate_event_receipt(result,event_number=event,objective='osd_first_error',reference_schedule=ordinary)
    saved,live,closed=read(audit/'SAVED_ENDPOINT.json'),read(audit/'LIVE_BOUNDARY.json'),read(audit/'CLOSED.json')
    if (saved['status']!='passed' or not saved['export_exact'] or live['exact_resume'] is not True
            or closed['global_step']!=event*128 or closed['presentations']!=event*256):
        raise ValueError('Saved/live/closed scientific boundary disagrees')
    for path,expected in read(audit/'STATE_IDENTITIES.json').items():
        if identity(path)!=expected: raise ValueError('Saved endpoint changed: '+path)
    if decode:
        import torch
        from .training_worker import validate_own
        state=torch.load(endpoint/'training_state.pt',map_location='cpu',weights_only=False)
        validate_own(state,plan,plan['configs'][str(seed)][arm],arm,seed,event)
        _export_equal(state,endpoint/'export')
    return result,read(audit/'COSTS.json')


def inspect_boundary(root,plan):
    """Establish complete chronology before giving any remaining-capacity credit."""
    completed=[]
    initial=set()
    gap=False
    evaluation_complete=[]
    for seed in SEEDS:
        for event in range(1,5):
            for arm in ARMS:
                audit,endpoint=job_paths(root,seed,arm,event)
                present=audit.exists() or endpoint.exists()
                if not present:
                    gap=True
                    continue
                if gap: raise ValueError('Training work exists after a chronology gap')
                completed_job(root,plan,seed,arm,event,decode=True)
                completed.append([seed,arm,event])
                initial.add((seed,arm))
        evaluation=Path(root)/'campaign_001'/f'seed_{seed}/event_004/evaluation'
        if evaluation.exists():
            from .report import audit_cell
            rows,errors=audit_cell(root,plan,seed,4)
            if errors or len(rows)!=128: raise ValueError('Partial/invalid evaluation cannot be retried')
            evaluation_complete.append(seed)
    if 17 not in evaluation_complete and any(item[0]==29 for item in completed):
        raise ValueError('Later seed exists before required earlier evaluation')
    return dict(completed_jobs=completed,completed_initials=len(initial),
                evaluation_complete=evaluation_complete)


def dispatch_training(root,plan,seed,arm,event):
    audit,_=job_paths(root,seed,arm,event)
    audit.mkdir(parents=True,exist_ok=False)
    environment=fresh_worker_environment(audit/'caches')
    command=[plan['training_python'],'-u','-m','experiments.k_prefix_protection_v1.training_worker',
             str(root),'--seed',str(seed),'--arm',arm,'--event',str(event)]
    write_json(audit/'JOB.json',dict(seed=seed,arm=arm,event=event,command=command,
                                   execution_source=plan['execution_source']))
    started=time.perf_counter()
    with (audit/'WORKER.log').open('w') as stream:
        result=subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT)
    write_json(audit/'EXEC.json',dict(command=command,returncode=result.returncode,
               wall_s=time.perf_counter()-started,automatic_retry=False))
    if result.returncode: raise RuntimeError('Training worker failed: '+str(audit))
    return completed_job(root,plan,seed,arm,event)


def reconcile(root):
    totals={phase:{kind:0 for kind in ('forwards','backwards','optimizer_steps','reference_forwards')}
            for phase in ('attempts','completed','unknown')}
    counters,presentations,evaluations={},{},{}
    for seed in SEEDS:
        for arm in ARMS:
            key=f'{seed}/{arm}'
            counters[key]=presentations[key]=0
            for event in range(1,5):
                audit,_=job_paths(root,seed,arm,event)
                for phase,values in read_work(audit/'WORK.jsonl',limits(arm)).items():
                    for kind,value in values.items(): totals[phase][kind]+=value
                if (audit/'CLOSED.json').exists():
                    row=read(audit/'CLOSED.json')
                    counters[key]=row['global_step']
                    presentations[key]=row['presentations']
        path=Path(root)/'campaign_001'/f'seed_{seed}/event_004/evaluation/STATUS.json'
        if path.exists(): evaluations[f'{seed}/4']=read(path)
    result=dict(work=totals,arm_counters=counters,arm_presentations=presentations,evaluations=evaluations,
                optimizer_steps=totals['completed']['optimizer_steps'],presentations=sum(presentations.values()))
    for suffix,key in (('attempted','attempted_generations'),('returned','returned_generations'),
        ('valid','completed_valid_generations'),('unknown','unknown_generation_attempts'),
        ('invalid','invalid_generations'),('pending','pending_canonical_generations')):
        result['evaluation_'+suffix]=sum(row.get(key,0) for row in evaluations.values())
    return result


def require_complete(progress):
    require_complete_work(progress['work'],dict(forwards=4096,backwards=4096,optimizer_steps=2048,reference_forwards=2048))
    if (any(v!=512 for v in progress['arm_counters'].values())
            or any(v!=1024 for v in progress['arm_presentations'].values())):
        raise ValueError('Incomplete own training trajectory')
    for key,value in dict(attempted=3072,returned=3072,valid=3072,unknown=0,invalid=0,pending=0).items():
        if progress['evaluation_'+key]!=value: raise ValueError('Incomplete/invalid serving budget')


def run(root, *, resume=False):
    from online_feedback.sglang_backend import SGLangDFlashBackend
    from .evaluation import evaluate
    root=Path(root).resolve()
    plan=read(root/'sealed/PLAN.json')
    validate_plan(plan)
    if read(root/'qualification_001/QUALIFICATION.json')['status']!='passed':
        raise ValueError('Production requires the sole successful numerical qualification')
    from .preflight import verify
    # Source and reused artifacts are checked before any fresh or resumed model.
    source_receipt=reserve_receipt(root/'source_verification')
    verify(root,postflight=True,receipt_path=source_receipt,phase='source_before_production')
    output=root/'campaign_001'
    if output.exists() and not resume: raise FileExistsError('Explicit complete-boundary recovery required')
    boundary=inspect_boundary(root,plan) if resume else dict(completed_jobs=[],completed_initials=0,evaluation_complete=[])
    receipt=reserve_receipt(root/'production_admission')
    admission=admit(root,completed_jobs=len(boundary['completed_jobs']),
                    completed_initials=boundary['completed_initials'],qualification_complete=True,
                    receipt=str(receipt.relative_to(root)))
    output.mkdir(exist_ok=resume)
    progress=dict(status='running',phase='training',automatic_retry=False,
                  execution_source=plan['execution_source'],explicit_resume=resume)
    costs=read(output/'COSTS.json') if (output/'COSTS.json').exists() else {}
    started=time.perf_counter()
    try:
        for seed in SEEDS:
            for event in range(1,5):
                for arm in ARMS:
                    progress.update(seed=seed,event=event,arm=arm,phase='training')
                    write_json(output/'STATUS.json',{**progress,**reconcile(root)})
                    if [seed,arm,event] in boundary['completed_jobs']:
                        _,job_costs=completed_job(root,plan,seed,arm,event)
                    else:
                        _,job_costs=dispatch_training(root,plan,seed,arm,event)
                    costs[f'{seed}/{event}/{arm}']=job_costs
                    write_json(output/'COSTS.json',costs)
            if seed in boundary['evaluation_complete']: continue
            progress.update(phase='evaluation',seed=seed,event=4)
            write_json(output/'STATUS.json',{**progress,**reconcile(root)})
            attempt=reserve_directory(output/'native_attempts')
            backend=None
            attempt_status=dict(status='starting',native_path=str(attempt/'native'),seed=seed,
                                execution_source=plan['execution_source'])
            write_json(attempt/'STATUS.json',attempt_status)
            try:
                backend=SGLangDFlashBackend(plan['target'],plan['draft'],worker_python=plan['worker_python'],
                    output_root=attempt/'native',serving_gpu='0',context_limit=4096,block_size=16)
                costs[f'{seed}/native_loading_s']=backend.loading_s
                checkpoints={'F':plan['draft'],'K':plan['original_k4'][str(seed)]['export']}
                for arm in ARMS:
                    _,endpoint=job_paths(root,seed,arm,4)
                    checkpoints[arm]=str(endpoint/'export')
                evaluation=output/f'seed_{seed}/event_004/evaluation'
                evaluation.parent.mkdir(parents=True,exist_ok=True)
                costs[f'{seed}/evaluation']=evaluate(backend,plan['evaluation_requests'],checkpoints,
                    evaluation,seed=seed,event=4,canonical_path=output/'EVALUATION_CANONICAL.json')
                attempt_status['status']='completed'
            except BaseException as error:
                attempt_status.update(status='failed',error=repr(error))
                raise
            finally:
                if backend is not None: backend.close()
                write_json(attempt/'STATUS.json',attempt_status)
                write_json(output/'COSTS.json',costs)
        progress.update(reconcile(root))
        require_complete(progress)
        progress.update(status='completed',phase='finished')
    except BaseException as error:
        progress.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        progress.update(reconcile(root),wall_s=time.perf_counter()-started)
        write_json(output/'STATUS.json',progress)
        write_json(output/'COSTS.json',costs)
    return progress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args.root,resume=args.resume)

if __name__=='__main__':
    main()
