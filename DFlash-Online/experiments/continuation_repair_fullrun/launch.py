"""Serial qualification, BASE58, TAIL58 and the fixed fresh four-endpoint panel."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.frontier_preservation_pilot.qualify import verify_source
from experiments.frontier_preservation_pilot.point1_prepare import gpu_inventory
from experiments.persistent_prefix_withholding.launch import child_environment, check_compilers
from experiments.bounded_prefix_allocation.launch import stop_owned
from .contract import ARMS
from .runtime import (read,write_json,initialize,clock_status,
                      storage,forecast,LIMITS,begin_card,end_card)


def check_clock(plan):
    clock=clock_status(plan['root'])
    if clock['elapsed_s']>=LIMITS['elapsed_s'] or clock['occupied_s']>=LIMITS['occupied_card_s']:
        raise TimeoutError('Original full-run time ceiling reached')
    return clock


def qualification_attempt(root, phase):
    """Only one localized retry is shared by both qualification phases."""
    paths=list((Path(root)/'processes').glob('qualification_*.json'))
    failed=sum(read(path).get('exit_code') != 0 for path in paths)
    if failed>=2:
        raise RuntimeError('The single overall qualification retry is exhausted')
    return 1+sum(path.name.startswith(phase+'_') for path in paths)


def child(plan_path,plan,phase,*,arm=None,resume=False,attempt=1):
    root=Path(plan['root'])
    name=phase+('_'+arm if arm else '')+f'_{attempt:03d}'
    if gpu_inventory()['processes']:
        raise RuntimeError('An existing GPU process prevents the next fixed phase')
    command=[plan['training_python'],'-u','-m','experiments.continuation_repair_fullrun.entry',
             '--plan',str(plan_path),'--phase',phase,'--attempt',str(attempt)]
    if arm:command.extend(['--arm',arm])
    if resume:command.append('--resume')
    slot='B' if phase in ('train','qualification_training') else 'A'
    slots=('A','B') if phase=='train' else (slot,)
    environment=child_environment(plan,slot)
    environment['PATH']=str(Path(plan['zstd']).parent)+os.pathsep+environment['PATH']
    record=dict(phase=phase,arm=arm,attempt=attempt,command=command,devices=[plan['devices'][s] for s in slots])
    process=None;owned=[]
    try:
        for card in slots:
            begin_card(root,name+'_'+card,plan['devices'][card]);owned.append(card)
        with (root/'logs'/f'{name}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=plan['source_root'],env=environment,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record.update(pid=process.pid,started_epoch=time.time())
            write_json(root/'processes'/f'{name}.json',record)
            next_storage=time.monotonic()
            while process.poll() is None:
                check_clock(plan)
                if time.monotonic()>=next_storage:
                    observed=storage(plan)
                    write_json(root/f'STORAGE.{os.getpid()}.json',observed)
                    next_storage=time.monotonic()+30
                time.sleep(1)
            if process.returncode:
                raise RuntimeError('Full-run phase failed; preserved original evidence: '+name)
    except BaseException as error:
        record['error']=repr(error)
        raise
    finally:
        try:
            if process is not None:stop_owned(process)
        except BaseException as error:
            record['cleanup_error']=repr(error)
            raise
        finally:
            code=None if process is None else process.poll()
            # A failed termination keeps its clock interval open for reconciliation.
            confirmed=process is None or code is not None
            record.update(process_created=process is not None,exit_code=code,
                finished_epoch=time.time() if confirmed else None,termination_confirmed=confirmed)
            try:
                write_json(root/'processes'/f'{name}.json',record)
            finally:
                if confirmed:
                    for card in owned:end_card(root,name+'_'+card)

    result=read(root/'phases'/name/'STATUS.json')
    if result['status']!='completed':raise ValueError('Child lacks completed phase receipt')
    return result


def remaining_forecast(plan):
    root=Path(plan['root']);fixed=plan['forecast'];counts=[]
    rate=fixed['full_arm_wall_s']/58
    for arm in ARMS:
        path=root/'campaign_001/arms'/arm/'STATUS.json'
        status=read(path) if path.exists() else {}
        count=status.get('events_completed',0)
        if count:
            rate=max(rate,status.get('worker_wall_s',0.)/count)
        counts.append(count)
    training=sum(max(0,58-count) for count in counts)*rate
    if (root/'evaluation/SUMMARY.json').exists():
        evaluation=0.
    else:
        work=read(root/'WORK.json')['completed']
        done=sum(work.get('production.'+key,0) for key in ('canonical_generation','scored_generation','warmup_generation'))
        evaluation=fixed['final_evaluation_wall_s']*max(0.,1-done/(1024+12288+96))
    qualification=0. if (root/'qualification/QUALIFICATION.json').exists() else fixed['qualification_wall_s']
    return dict(wall_s=training+evaluation+qualification+fixed['audit_wall_s'],
                card_s=2*training+evaluation+qualification)


def run(plan_path,*,resume=False):
    plan_path=Path(plan_path);plan=read(plan_path);root=Path(plan['root'])
    check_compilers();verify_source(plan)
    for name in ('logs','processes','qualification'): (root/name).mkdir(exist_ok=True)
    if not resume:
        if (root/'STATUS.json').exists():raise FileExistsError('Preserve existing attempt; use explicit continuation')
        from .preflight import verify
        verify(plan)
        initialize(root)
    elif not (root/'CLOCK.json').exists():
        raise ValueError('Continuation must keep the original model-load clock')
    report=read(root/'STATUS.json') if resume else dict(status='running',phases=[],started_epoch=time.time())
    report.update(status='running',execution_commit=plan['execution_commit'])
    write_json(root/'STATUS.json',report)
    try:
        estimates=remaining_forecast(plan)
        forecast(plan,remaining_wall_s=estimates['wall_s'],remaining_card_s=estimates['card_s'],
            destination=root/'forecasts'/('RESUME_'+str(time.time_ns())+'.json' if resume else 'INITIAL.json'))
        qualified=root/'qualification/QUALIFICATION.json'
        if not qualified.exists():
            for phase in ('qualification_training','qualification_serving'):
                receipt=root/'qualification'/('TRAINING.json' if phase.endswith('training') else 'SERVING.json')
                if receipt.exists() and read(receipt)['status']=='passed':continue
                attempt=qualification_attempt(root,phase)
                result=child(plan_path,plan,phase,attempt=attempt)
                report['phases'].append(result);write_json(root/'STATUS.json',report)
            training,serving=read(root/'qualification/TRAINING.json'),read(root/'qualification/SERVING.json')
            if training['status']!='passed' or serving['status']!='passed':
                raise ValueError('Every integration correctness gate must pass')
            work=read(root/'WORK.json')
            if work['pending']:raise ValueError('Qualification contains unreconciled unknown work')
            write_json(qualified,dict(status='passed',training=training,serving=serving,
                work=work,execution_commit=plan['execution_commit'],efficacy_required=False))
            from .qualification import retire_disposable_states
            retire_disposable_states(plan,training)
        for arm in ARMS:
            path=root/'campaign_001/arms'/arm/'STATUS.json'
            if path.exists() and read(path)['status']=='completed':continue
            attempts=len(list((root/'processes').glob('train_'+arm+'_*.json')))
            result=child(plan_path,plan,'train',arm=arm,resume=path.exists(),attempt=attempts+1)
            report['phases'].append(result);write_json(root/'STATUS.json',report)
        if not (root/'evaluation/SUMMARY.json').exists():
            attempts=len(list((root/'processes').glob('evaluate_*.json')))
            result=child(plan_path,plan,'evaluate',resume=resume,attempt=attempts+1)
            report['phases'].append(result);write_json(root/'STATUS.json',report)
        report.update(status='completed',finished_epoch=time.time())
    except BaseException as error:
        report.update(status='blocked',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['clock']=clock_status(root)
        report['work']=read(root/'WORK.json')
        write_json(root/'STATUS.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(args.plan,resume=args.resume)
