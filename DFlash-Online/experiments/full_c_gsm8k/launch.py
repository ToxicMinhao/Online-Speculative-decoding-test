"""Qualify once, admit the full fixed cost, then run one C58 trajectory."""
import argparse
from pathlib import Path
import subprocess
import shutil
import time
import traceback

from online_feedback.storage import write_json
from experiments.bounded_prefix_allocation.launch import stop_owned
from experiments.persistent_prefix_withholding.launch import child_environment, check_compilers
from .runtime import initialize_work, read, enforce_clock, require_time_for, boundary
from .contract import PRODUCTION, QUALIFICATION


def child(plan_path,plan,phase,*,resume=False):
    root=Path(plan['root'])
    slot='B' if phase in ('qualification_training','train') else 'A'
    command=[plan['training_python'],'-u','-m','experiments.full_c_gsm8k.entry',
        '--plan',str(plan_path),'--phase',phase]
    if resume:command.append('--resume')
    suffix=('_resume_'+str(len(list((root/'logs').glob(phase+'_resume_*')))+1)) if resume else ''
    name=phase+suffix
    receipt=dict(command=command,started_epoch=time.time(),phase=phase)
    with (root/'logs'/(name+'.log')).open('x') as log:
        process=subprocess.Popen(command,cwd=plan['source_root'],env=child_environment(plan,slot),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt['pid']=process.pid
        write_json(root/'processes'/(name+'.json'),receipt)
        try:
            storage_checked=time.monotonic()
            while process.poll() is None:
                enforce_clock(plan)
                if time.monotonic()-storage_checked >= 60:
                    # A statvfs query does not hash/copy artifacts or disturb timed
                    # serving; the full remaining-write admission runs at boundaries.
                    free=shutil.disk_usage(root).free
                    if free < plan['storage']['reserve_bytes']:
                        raise RuntimeError('Shared storage reserve lost during execution')
                    if phase=='train':
                        boundary(plan)
                    storage_checked=time.monotonic()
                time.sleep(1)
            if process.returncode:
                raise RuntimeError('Phase failed; no automatic scientific retry: '+phase)
        finally:
            stop_owned(process)
            receipt.update(exit_code=process.poll(),finished_epoch=time.time())
            write_json(root/'processes'/(name+'.json'),receipt)
    result=read(root/'phases'/phase/'STATUS.json')
    if result['status']!='completed':
        raise ValueError('Phase has no complete result: '+phase)
    return result


def forecast(plan,qualification=None):
    """Historical whole-phase costs plus measured unchanged-C update cost."""
    prices=dict(plan['estimated_seconds'])
    if qualification is not None:
        cases=qualification['result']['cases']
        pairs={r['case']:r['pair_wall_excluding_observers_s'] for r in cases}
        ratio=max(1.,pairs['active']/pairs['native'])
        # The complete original synchronous PR25 trajectory includes collection,
        # teacher preparation, publication and I/O; increase only its train component.
        prices['train']=max(prices['train'],31887.92+19272.24*(ratio-1.))
    elapsed=sum(prices.values())
    cards=2*prices['train']+prices['final_gsm8k']+prices['final_mathqa']
    return dict(prices=prices,admission=require_time_for(plan,elapsed,cards),
        interpretation='Planning estimate, not a guaranteed upper bound; no endpoint or recipe changes')


def final_audit(plan):
    from .analysis import write_report
    root=Path(plan['root']);began=time.perf_counter()
    result={}
    for dataset in ('gsm8k','mathqa'):
        enforce_clock(plan)
        summary=write_report(root/'evaluation'/dataset)
        if summary['status']!='completed':
            raise ValueError('Raw endpoint record audit failed: '+dataset)
        result[dataset]=dict(status='passed',report_root=str(root/'evaluation'/dataset),
                             contrasts=summary['comparisons'])
    receipt=dict(status='passed',execution_commit=plan['execution_commit'],datasets=result,
        wall_s=time.perf_counter()-began,extra_model_work=0,clock=enforce_clock(plan))
    write_json(root/'FINAL_AUDIT.json',receipt)
    return receipt


def run(plan_path,*,resume=False):
    plan_path=Path(plan_path);plan=read(plan_path);root=Path(plan['root'])
    path=root/'LAUNCH.json'
    if path.exists() and not resume:
        raise FileExistsError('Original attempt is immutable; reconcile completed work before explicit resume')
    if resume:
        return resume_pending(plan_path,plan)
    (root/'logs').mkdir(exist_ok=True)
    report=dict(status='cpu_preflight',execution_commit=plan['execution_commit'],
        binding=plan['binding'],started_epoch=time.time(),completed=[],automatic_retry=False)
    write_json(path,report)
    try:
        check_compilers()
        with (root/'logs/CPU_PREFLIGHT.log').open('x') as log:
            subprocess.run([plan['training_python'],'-u','-m','experiments.full_c_gsm8k.preflight',
                '--plan',str(plan_path)],cwd=plan['source_root'],env=child_environment(plan),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        if read(root/'CPU_PREFLIGHT.json')['status']!='passed':
            raise ValueError('CPU preflight has not passed')
        initialize_work(root)
        report['preliminary_forecast']=forecast(plan)
        report['status']='qualification';write_json(path,report)
        trained=child(plan_path,plan,'qualification_training')
        report['completed'].append('qualification_training');write_json(path,report)
        served=child(plan_path,plan,'qualification_serving')
        report['completed'].append('qualification_serving');write_json(path,report)
        work=read(root/'WORK.json')
        if work['pending'] or work['attempted']!=QUALIFICATION or work['completed']!=QUALIFICATION:
            raise ValueError('Bounded qualification accounting does not reconcile')
        qualification=dict(status='passed',training=trained,serving=served,work=work,
            execution_commit=plan['execution_commit'],binding=plan['binding'])
        write_json(root/'QUALIFICATION.json',qualification)
        from .storage import retire_qualification
        report['qualification_retirement']=retire_qualification(plan,qualification)
        report['resource_admission']=forecast(plan,trained)
        write_json(root/'RESOURCE_ADMISSION.json',dict(report['resource_admission'],storage=boundary(plan)))
        report['status']='production';write_json(path,report)
        for phase in ('train','final_gsm8k','final_mathqa'):
            report['active_phase']=phase;write_json(path,report)
            child(plan_path,plan,phase)
            report['completed'].append(phase);write_json(path,report)
        work=read(root/'WORK.json')
        expected={k:PRODUCTION[k]+QUALIFICATION[k] for k in PRODUCTION}
        if work['pending'] or work['attempted']!=expected or work['completed']!=expected:
            raise ValueError('Fixed campaign accounting does not reconcile')
        report.update(status='completed',work=work,audit=final_audit(plan),postflight=boundary(plan))
    except BaseException as error:
        report.update(status='stopped',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        report['calendar_wall_s']=time.time()-report['started_epoch']
        write_json(path,report)
    return report


def resume_pending(plan_path,plan):
    """Continue only known unfinished phases, never repeat qualification or science."""
    root=Path(plan['root']);original=read(root/'LAUNCH.json')
    if (original['execution_commit']!=plan['execution_commit'] or original['binding']!=plan['binding']
            or read(root/'QUALIFICATION.json')['status']!='passed'):
        raise ValueError('A validated original checkout and completed qualification are required')
    work=read(root/'WORK.json')
    if work['pending'] or work['attempted']!=work['completed']:
        raise ValueError('Unknown work requires explicit reconciliation; no automatic replay')
    folder=root/'resumes';folder.mkdir(exist_ok=True)
    path=folder/f'RESUME_{len(list(folder.glob("RESUME_*.json")))+1:03d}.json'
    report=dict(status='running',execution_commit=plan['execution_commit'],binding=plan['binding'],
        original_launch=str(root/'LAUNCH.json'),started_epoch=time.time(),completed=[])
    write_json(path,report)
    try:
        enforce_clock(plan);boundary(plan)
        prices=read(root/'RESOURCE_ADMISSION.json')['prices']
        remaining=dict(prices)
        for phase in ('train','final_gsm8k','final_mathqa'):
            status_path=root/'phases'/phase/'STATUS.json'
            if status_path.exists() and read(status_path)['status']=='completed':remaining[phase]=0.
        train_status=root/'campaign_001/arms/KPLUS_C/STATUS.json'
        if remaining['train'] and train_status.exists():
            completed=read(train_status)['events_completed']
            remaining['train']*=max(0.,(58-completed)/58)
        report['resume_admission']=require_time_for(plan,sum(remaining.values()),
            2*remaining['train']+remaining['final_gsm8k']+remaining['final_mathqa'])
        write_json(path,report)
        for phase in ('train','final_gsm8k','final_mathqa'):
            status_path=root/'phases'/phase/'STATUS.json'
            prior=read(status_path) if status_path.exists() else None
            if prior is not None and prior['status']=='completed':
                report['completed'].append(dict(phase=phase,reused=True));continue
            child(plan_path,plan,phase,resume=prior is not None)
            report['completed'].append(dict(phase=phase,reused=False));write_json(path,report)
        work=read(root/'WORK.json')
        expected={k:PRODUCTION[k]+QUALIFICATION[k] for k in PRODUCTION}
        # Reloading a complete recovery constructs a runtime, never another
        # scientific update. It is separately visible and still bounded.
        expected['training_initialization']=work['completed']['training_initialization']
        if (work['pending'] or work['attempted']!=expected or work['completed']!=expected
                or not 2<=expected['training_initialization']<=5):
            raise ValueError('Resumed fixed campaign does not reconcile')
        report.update(status='completed',work=work,audit=final_audit(plan),postflight=boundary(plan))
    except BaseException as error:
        report.update(status='stopped',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        report['wall_s']=time.time()-report['started_epoch'];write_json(path,report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    run(args.plan,resume=args.resume)
