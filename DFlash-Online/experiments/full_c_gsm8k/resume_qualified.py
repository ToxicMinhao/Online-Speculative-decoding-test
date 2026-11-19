"""Resume production after the qualified run's elapsed-forecast stop only."""
import argparse
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.persistent_prefix_withholding.launch import child_environment, check_compilers
from .contract import PRODUCTION, QUALIFICATION, plan_binding
from .runtime import ELAPSED_AMENDMENT, LIMITS, read, boundary, enforce_clock
from .launch import child, forecast, final_audit
from .retry_initial import SOURCE_FIELDS, STORAGE_SOURCE_FIELDS

OPERATIONAL_CHANGES = {
    'experiments/full_c_gsm8k/runtime.py',
    'experiments/full_c_gsm8k/resume_qualified.py',
    'tests/full_c_gsm8k/test_runtime.py',
    'tests/full_c_gsm8k/test_resume_qualified.py',
}


def source_continuity(original, revised, preflight):
    if identity(original['source_manifest']) != preflight['source']['manifest_identity']:
        raise ValueError('Qualified source manifest changed after preflight')
    old,new=read(original['source_manifest']),read(revised['source_manifest'])
    if (old['dirty'] or new['dirty'] or old['commit'] != original['execution_commit']
            or new['commit'] != revised['execution_commit']):
        raise ValueError('Source manifests must identify the published clean checkouts')
    changed=[]
    for name in sorted(set(old['files'])|set(new['files'])):
        if old['files'].get(name) == new['files'].get(name):
            continue
        report=(name.startswith('reports/dflash_full_c_gsm8k_20261118/')
                and Path(name).suffix in ('.md','.json','.csv'))
        if name not in OPERATIONAL_CHANGES and not report:
            raise ValueError('Qualified scientific source changed: '+name)
        changed.append(name)
    return dict(status='passed',changed_files=changed,
        unchanged_scientific_files=len(old['files'])-len(set(changed)&set(old['files'])))


def validate_continuation(plan):
    root=Path(plan['root']);qualified=read(root/'QUALIFICATION.json')
    original=read(qualified['active_plan']);stopped=read(root/'initial_retry_001/LAUNCH.json')
    if (qualified['status'] != 'passed' or qualified['training']['status'] != 'completed'
            or qualified['serving']['status'] != 'completed'
            or qualified['training']['result']['status'] != 'passed'
            or qualified['execution_commit'] != original['execution_commit']
            or qualified['binding'] != original['binding']
            or stopped['status'] != 'stopped'
            or stopped['completed'] != ['qualification_training','qualification_serving']
            or 'Complete fixed campaign no longer fits:' not in stopped.get('error','')):
        raise ValueError('Require completed qualification and the retained forecast-only stop')
    if plan_binding(original) != original['binding'] or plan_binding(plan) != plan['binding']:
        raise ValueError('Qualified or continuation plan binding changed')
    if 'elapsed_budget_amendment' in original or plan.get('elapsed_budget_amendment') != ELAPSED_AMENDMENT:
        raise ValueError('Continuation requires exactly the authorized elapsed-only amendment')
    skip=SOURCE_FIELDS|{'binding','storage','elapsed_budget_amendment'}
    if {k:v for k,v in original.items() if k not in skip} != {k:v for k,v in plan.items() if k not in skip}:
        raise ValueError('Continuation changes the qualified science or forecast')
    storage=lambda x:{k:v for k,v in x['storage'].items() if k not in STORAGE_SOURCE_FIELDS}
    if storage(original) != storage(plan):
        raise ValueError('Storage capacity, reserve or sizing changed')
    if (not set(original['storage']['external_roots']) <= set(plan['storage']['external_roots'])
            or str(Path(plan['source_root']).parent) not in plan['storage']['external_roots']
            or any(plan['storage'][key] != plan[key] for key in ('source_root','source_manifest','execution_commit'))):
        raise ValueError('Prior/current source allocation must remain authenticated and charged')
    work,clock=read(root/'WORK.json'),read(root/'CLOCK.json')
    prior=qualified['prior_work']['completed']
    expected={k:prior[k]+QUALIFICATION[k] for k in QUALIFICATION}
    if (work != qualified['work'] or work['limits'] != LIMITS or work['pending']
            or work['attempted'] != expected or work['completed'] != expected or clock['active']
            or clock['first_model_load_epoch'] != stopped['clock']['first_model_load_epoch']
            or (root/'campaign_001').exists()
            or any((root/'phases'/name).exists() for name in ('train','final_gsm8k','final_mathqa'))):
        raise ValueError('Continuation requires reconciled qualification and no started production')
    source=source_continuity(original,plan,read(root/'initial_retry_001/CPU_PREFLIGHT.json'))
    return dict(status='passed',qualification_reused=True,qualification_identity=identity(root/'QUALIFICATION.json'),
        qualified_commit=qualified['execution_commit'],qualified_binding=qualified['binding'],
        execution_commit=plan['execution_commit'],binding=plan['binding'],source=source,
        original_clock=clock,work_before=work,elapsed_budget_amendment=dict(ELAPSED_AMENDMENT))


def run(plan_path):
    plan_path=Path(plan_path);plan=read(plan_path);root=Path(plan['root'])
    checked=validate_continuation(plan)
    base=root/'production_resume_001';base.mkdir(exist_ok=False)
    path=base/'LAUNCH.json'
    report={**checked,'status':'cpu_preflight','started_epoch':time.time(),'completed':[],
            'qualification_optimizer_steps_repeated':0,'automatic_retry':False}
    write_json(path,report)
    protected={name:identity(root/name) for name in ('WORK.json','CLOCK.json','QUALIFICATION.json')}
    try:
        check_compilers();enforce_clock(plan)
        with (base/'CPU_PREFLIGHT.log').open('x') as log:
            subprocess.run([plan['training_python'],'-u','-m','experiments.full_c_gsm8k.preflight',
                '--plan',str(plan_path),'--output',str(base/'CPU_PREFLIGHT.json')],cwd=plan['source_root'],
                env=child_environment(plan),stdout=log,stderr=subprocess.STDOUT,check=True)
        if (read(base/'CPU_PREFLIGHT.json')['status'] != 'passed'
                or any(identity(root/name) != value for name,value in protected.items())):
            raise ValueError('Continuation preflight failed or altered qualified evidence/work/clock')
        qualification=read(root/'QUALIFICATION.json')
        report['resource_admission']=forecast(plan,qualification['training'])
        resources=dict(report['resource_admission'],storage=boundary(plan))
        if (root/'RESOURCE_ADMISSION.json').exists():
            raise FileExistsError('Do not overwrite a prior production admission')
        write_json(root/'RESOURCE_ADMISSION.json',resources)
        report['status']='production';write_json(path,report)
        for phase in ('train','final_gsm8k','final_mathqa'):
            report['active_phase']=phase;write_json(path,report)
            child(plan_path,plan,phase)
            report['completed'].append(phase);write_json(path,report)
        work=read(root/'WORK.json')
        expected={k:checked['work_before']['completed'][k]+PRODUCTION[k] for k in PRODUCTION}
        if work['pending'] or work['attempted'] != expected or work['completed'] != expected:
            raise ValueError('Production did not preserve all prior cumulative work')
        report.update(status='completed',work=work,audit=final_audit(plan),postflight=boundary(plan))
    except BaseException as error:
        report.update(status='stopped',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        report['wall_s']=time.time()-report['started_epoch'];write_json(path,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    run(parser.parse_args().plan)
