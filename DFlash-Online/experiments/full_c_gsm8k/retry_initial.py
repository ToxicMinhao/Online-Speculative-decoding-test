"""Explicit retry of the diagnosed, zero-update initial-state failure only.

The old attempt remains evidence. This entry never resets WORK/CLOCK, changes
scientific settings, or automatically retries another failure.
"""
import argparse
from pathlib import Path
import subprocess
import time
import traceback

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.persistent_prefix_withholding.launch import child_environment, check_compilers
from .contract import PRODUCTION, QUALIFICATION, plan_binding
from .runtime import LIMITS, read, boundary, enforce_clock
from .launch import child, forecast, final_audit


SOURCE_FIELDS = {'source_root', 'source_manifest', 'execution_commit', 'execution_source'}
STORAGE_SOURCE_FIELDS = {'source_root', 'source_manifest', 'execution_commit',
                         'external_roots', 'prior_source_manifests'}


def validate_retry(original, revised):
    """Permit only source staging changes, with the fixed science kept exact."""
    if plan_binding(original) != original['binding'] or plan_binding(revised) != revised['binding']:
        raise ValueError('Original or revised sealed plan binding differs')
    scientific = lambda x: {k:v for k,v in x.items() if k not in SOURCE_FIELDS | {'binding','storage'}}
    if scientific(original) != scientific(revised):
        raise ValueError('Initial retry may not change the scientific plan or resource forecast')
    storage = lambda x: {k:v for k,v in x['storage'].items() if k not in STORAGE_SOURCE_FIELDS}
    if storage(original) != storage(revised):
        raise ValueError('Initial retry may not change resource limits or storage sizing')
    prior_roots = set(original['storage']['external_roots'])
    roots = set(revised['storage']['external_roots'])
    if not prior_roots <= roots or str(Path(revised['source_root']).parent) not in roots:
        raise ValueError('Prior and current source allocations must remain charged')
    for key in ('source_root','source_manifest','execution_commit'):
        if revised['storage'][key] != revised[key]:
            raise ValueError('Storage source identity differs from the revised execution')
    root = Path(original['root'])
    launched = read(root/'LAUNCH.json')
    failed = read(root/'qualification/training_B/DETAILS.json')
    phase = read(root/'phases/qualification_training/STATUS.json')
    if (launched['status'] != 'stopped' or launched['completed']
            or launched['execution_commit'] != original['execution_commit']
            or launched['binding'] != original['binding']
            or phase['status'] != 'failed' or failed['status'] != 'failed'
            or failed['phase'] != 'loading' or failed['cases']
            or 'Independent arm initialization differs' not in failed.get('error','')):
        raise ValueError('Only the retained initial-state failure is eligible')
    if any((root/path).exists() for path in ('QUALIFICATION.json','campaign_001','qualification/training_B/trainer/initial_state.pt')):
        raise ValueError('Completed qualification or production work must not be repeated')
    work,clock = read(root/'WORK.json'),read(root/'CLOCK.json')
    counts=work['completed']
    if (work['limits'] != LIMITS or work['pending'] or work['attempted'] != counts
            or set(counts) != set(LIMITS) or not 1 <= counts['training_initialization'] <= 3
            or any(value for key,value in counts.items() if key != 'training_initialization')
            or clock['active']):
        raise ValueError('Retry requires known constructor-only work and idle clock')
    return dict(status='passed',original_binding=original['binding'],binding=revised['binding'],
        original_commit=original['execution_commit'],execution_commit=revised['execution_commit'],
        prior_work=work,clock=clock,scientific_plan_unchanged=True)


def preserve_failed_paths(root, destination):
    """Move only the failed attempt's occupied paths; retain byte identities."""
    names = ('phases/qualification_training','qualification/training_B',
             'logs/qualification_training.log','processes/qualification_training.json')
    rows=[]
    for name in names:
        source=root/name
        if not source.exists():
            raise FileNotFoundError(source)
        files=sorted(p for p in source.rglob('*') if p.is_file()) if source.is_dir() else [source]
        if any(p.is_symlink() for p in files):
            raise ValueError('Failed-attempt evidence contains an unexpected link')
        rows.extend(dict(original_path=str(p),retained_path=str(destination/p.relative_to(root)),
                         identity=identity(p)) for p in files)
    write_json(destination/'PRESERVATION.json',dict(status='planned',files=rows))
    for name in names:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        (root/name).rename(target)
    if any(identity(row['retained_path']) != row['identity'] for row in rows):
        raise ValueError('Retained failed-attempt bytes changed')
    write_json(destination/'PRESERVATION.json',dict(status='passed',files=rows))
    return rows


def run(plan_path):
    plan_path=Path(plan_path);plan=read(plan_path);root=Path(plan['root'])
    original=read(root/'sealed/PLAN.json')
    checked=validate_retry(original,plan)
    attempt=root/'initial_retry_001';attempt.mkdir(exist_ok=False)
    receipt=attempt/'LAUNCH.json'
    report={**checked,'status':'cpu_preflight','started_epoch':time.time(),
            'completed':[],'automatic_retry':False}
    write_json(receipt,report)
    before={name:identity(root/name) for name in ('WORK.json','CLOCK.json')}
    try:
        check_compilers();enforce_clock(plan)
        with (attempt/'CPU_PREFLIGHT.log').open('x') as log:
            subprocess.run([plan['training_python'],'-u','-m','experiments.full_c_gsm8k.preflight',
                '--plan',str(plan_path),'--output',str(attempt/'CPU_PREFLIGHT.json')],
                cwd=plan['source_root'],env=child_environment(plan),stdout=log,stderr=subprocess.STDOUT,check=True)
        if read(attempt/'CPU_PREFLIGHT.json')['status'] != 'passed':
            raise ValueError('Revised source/input preflight did not pass')
        report['preserved_failure']=preserve_failed_paths(root,attempt/'prior_failure')
        if any(identity(root/name) != value for name,value in before.items()):
            raise ValueError('Preparation changed the original cumulative ledger or clock')
        report.update(status='qualification',ledgers_unchanged_before_retry=True)
        write_json(receipt,report)
        trained=child(plan_path,plan,'qualification_training')
        report['completed'].append('qualification_training');write_json(receipt,report)
        served=child(plan_path,plan,'qualification_serving')
        report['completed'].append('qualification_serving');write_json(receipt,report)
        prior=checked['prior_work']['completed']
        work=read(root/'WORK.json');expected={k:prior[k]+QUALIFICATION[k] for k in prior}
        if work['pending'] or work['attempted'] != expected or work['completed'] != expected:
            raise ValueError('Retry qualification lost cumulative failed-attempt accounting')
        qualified=dict(status='passed',training=trained,serving=served,work=work,
            execution_commit=plan['execution_commit'],binding=plan['binding'],prior_work=checked['prior_work'],
            original_failure=str(attempt/'prior_failure'),active_plan=str(plan_path))
        write_json(root/'QUALIFICATION.json',qualified)
        from .storage import retire_qualification
        report['qualification_retirement']=retire_qualification(plan,qualified)
        report['resource_admission']=forecast(plan,trained)
        write_json(root/'RESOURCE_ADMISSION.json',dict(report['resource_admission'],storage=boundary(plan)))
        report['status']='production';write_json(receipt,report)
        for phase in ('train','final_gsm8k','final_mathqa'):
            report['active_phase']=phase;write_json(receipt,report)
            child(plan_path,plan,phase)
            report['completed'].append(phase);write_json(receipt,report)
        work=read(root/'WORK.json')
        expected={k:prior[k]+QUALIFICATION[k]+PRODUCTION[k] for k in prior}
        if work['pending'] or work['attempted'] != expected or work['completed'] != expected:
            raise ValueError('Fixed campaign lost cumulative prior-work accounting')
        report.update(status='completed',work=work,audit=final_audit(plan),postflight=boundary(plan))
    except BaseException as error:
        report.update(status='stopped',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        report['wall_s']=time.time()-report['started_epoch'];write_json(receipt,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    run(parser.parse_args().plan)
