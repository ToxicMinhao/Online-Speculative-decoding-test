"""One authorized production continuation after the passed qualification stop.

The original plan, qualification, launch receipt, work ledger and clock remain
in place. Only the explicit elapsed-time policy changes; no qualification or
canonical warmup generation is repeated.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

from online_feedback.storage import write_json
from experiments.bounded_prefix_allocation.preflight import verify_source
from .prepare import identity, load_inputs, plan_binding
from .preflight import runtime_inventory
from .runtime import read, exclusive, elapsed_limit, boundary, clock_status, LIMITS
from . import launch

COMPLETED_QUALIFICATION = dict(adam=12,response_forward=28,response_backward=28,
    training_initialization=2,feedback_generation=0,teacher_preparation=0,
    canonical_generation=4,qualification_generation=16,warmup_generation=0,scored_generation=0)
OPERATIONAL_FIELDS = {
    'execution_commit','source_root','source_manifest','allocation_roots',
    'original_execution_commit','original_scientific_binding','elapsed_time_authorization',
    'adopted_qualification_path','resume_from_qualified_stop',
}
ORCHESTRATION_FILES = {f'experiments/c_paired_confirmation/{name}.py'
                      for name in ('launch','runtime','task_runner','resume')}


def pinned(node):
    if set(node)!={'path','identity'} or identity(node['path'])!=node['identity']:
        raise ValueError('Retained evidence identity differs')
    return read(node['path'])


def source_compatibility(original, plan):
    """Every old/new scientific source is unchanged outside four reviewed files."""
    before,after=read(original['source_manifest']),read(plan['source_manifest'])
    if before['commit']!=original['execution_commit'] or before['dirty']:
        raise ValueError('Original source manifest is not the qualified clean checkout')
    protected=('online_feedback/','training/','inference/','experiments/')
    changed=[]
    for name in sorted(set(before['files'])|set(after['files'])):
        if name.startswith(protected) and before['files'].get(name)!=after['files'].get(name):
            if name not in ORCHESTRATION_FILES:
                raise ValueError('Qualified scientific source changed: '+name)
            changed.append(name)
    return dict(status='passed',changed_orchestration_files=changed,
        original_execution_commit=original['execution_commit'],execution_commit=plan['execution_commit'],
        source=verify_source(plan))


def required_retained_files(original):
    root=Path(original['root'])
    paths={root/name for name in ('sealed/PLAN.json','sealed/SEAL.json','LAUNCH.json','WORK.json',
        'CLOCK.json','QUALIFICATION.json','CPU_PREFLIGHT.json')}
    paths.add(Path(original['source_manifest']))
    for dataset,sub in original['datasets'].items():
        for kind,slot in (('training','A'),('serving','A'),('serving','B')):
            base=root/'qualification'/f'{dataset}_{kind}_{slot}'
            paths.add(base/'STATUS.json')
            if kind=='training':paths.add(base/'DETAILS.json')
            else:
                paths.update((base/'RESULT.json',base/'calls/RESULT.json'))
        paths.update((root/'tasks'/f'{dataset}_warmup_references'/'RESULT.json',
            Path(sub['warmup_references']),Path(sub['warmup_references']).parent/'WARMUP_SEAL.json'))
        for reference in read(sub['warmup_references']).values():
            path=Path(reference['source_record_path']);paths.add(path)
            paths.add(Path(read(path)['metadata']['native_trace_path']))
    return {str(path) for path in paths}


def authenticate_resume(plan):
    """CPU-only provenance and absence-of-production checks; no tensors loaded."""
    proof=plan['resume_from_qualified_stop'];original=pinned(proof['original_plan'])
    authority=pinned(proof['authorization']);root=Path(plan['root'])
    if elapsed_limit(plan) is not None or authority.get('elapsed_time_authorization')!=plan['elapsed_time_authorization']:
        raise ValueError('Require explicit matching completion authorization for elapsed time only')
    if original['binding']!=plan_binding(original):
        raise ValueError('Original sealed plan binding differs')
    if (plan['original_execution_commit']!=original['execution_commit']
            or plan['original_scientific_binding']!=original['binding']
            or plan['binding']!=original['binding'] or plan['execution_commit']==original['execution_commit']):
        raise ValueError('Original/current execution provenance is missing or conflated')
    left={k:v for k,v in original.items() if k not in OPERATIONAL_FIELDS}
    right={k:v for k,v in plan.items() if k not in OPERATIONAL_FIELDS}
    if left!=right or plan['limits']!=LIMITS:
        raise ValueError('Resume changed the scientific plan or another budget')
    if (not set(original['allocation_roots']).issubset(plan['allocation_roots'])
            or str(root) not in plan['allocation_roots'] or plan['source_root'] not in plan['allocation_roots']):
        raise ValueError('Resume allocation omitted an existing charged root')
    adopted=Path(plan['adopted_qualification_path'])
    if adopted.name!='QUALIFICATION.json' or adopted.parent.parent!=root or not adopted.parent.name.startswith('resume_'):
        raise ValueError('Adopted qualification must have its own resume evidence directory')
    expected=proof['retained_files']
    if not required_retained_files(original).issubset(expected):
        raise ValueError('Missing required retained qualification or stop evidence')
    for path,value in expected.items():
        if identity(path)!=value:raise ValueError('Retained evidence changed: '+path)
    if (root/'PRODUCTION_STARTED.json').exists() or (root/'RESOURCE_ADMISSION.json').exists():
        raise ValueError('Production or a previous admission already started')
    actual_tasks={p.name for p in (root/'tasks').iterdir() if p.is_dir()}
    if actual_tasks!={d+'_warmup_references' for d in ('gsm8k','mathqa')}:
        raise ValueError('Production task evidence already exists')
    prior=read(root/'LAUNCH.json');work=read(root/'WORK.json');clock=read(root/'CLOCK.json')
    if (prior['status']!='stopped' or prior['completed']!=['cpu_preflight','gsm8k_qualification','mathqa_qualification']
            or 'Remaining fixed campaign does not fit original clocks:' not in prior.get('error','')):
        raise ValueError('Only the fully qualified pre-production forecast stop may continue')
    if (work['limits']!=LIMITS or work['attempted']!=COMPLETED_QUALIFICATION
            or work['completed']!=COMPLETED_QUALIFICATION or work['pending'] or clock['active']):
        raise ValueError('Unexpected production, failed, pending or active work')
    if (clock['completed_card_s']!=prior['clock']['occupied_s']
            or read(root/'CPU_PREFLIGHT.json')['status']!='passed'):
        raise ValueError('Prior clock or completed CPU preflight differs')
    old_gate=read(root/'QUALIFICATION.json')
    if old_gate!=launch.qualification_gate(original):
        raise ValueError('Passed qualification does not reproduce from its retained receipts')
    for dataset,sub in original['datasets'].items():
        result=read(root/'tasks'/f'{dataset}_warmup_references'/'RESULT.json')
        if (result['status']!='completed' or result['reference_count']!=2
                or result['execution_commit']!=original['execution_commit'] or result['binding']!=original['binding']):
            raise ValueError('Warmup canonical qualification is incomplete')
        seal=read(Path(sub['warmup_references']).parent/'WARMUP_SEAL.json')
        if (seal['execution_commit']!=original['execution_commit'] or seal['binding']!=original['binding']
                or seal['status']!='passed' or seal['checkpoint']!=sub['origin_export']
                or seal['checkpoint_identity']!=sub['origin_weights_identity'] or seal['reference_count']!=2
                or seal['files']!={sub['warmup_references']:identity(sub['warmup_references'])}):
            raise ValueError('Original canonical warmup seal changed')
    for path,expected_identity in original['input_identities'].items():
        if identity(path)!=expected_identity:raise ValueError('Qualified input dependency changed: '+path)
    load_inputs(original)
    source=source_compatibility(original,plan)
    runtime=runtime_inventory(plan)
    return dict(status='passed',binding=original['binding'],original_execution_commit=original['execution_commit'],
        execution_commit=plan['execution_commit'],original_clock=clock,consumed_work=work,
        retained_evidence_count=len(expected),source=source,runtime=runtime,qualification=old_gate,new_model_work=0)


def adopt_qualification(plan, validated):
    receipt=dict(validated['qualification'],execution_commit=plan['execution_commit'],
        original_execution_commit=plan['original_execution_commit'],original_scientific_binding=plan['binding'],
        reused_without_new_model_work=True,original_qualification=dict(
            path=str(Path(plan['root'])/'QUALIFICATION.json'),identity=identity(Path(plan['root'])/'QUALIFICATION.json')),
        resume_authorization=plan['resume_from_qualified_stop']['authorization'])
    path=Path(plan['adopted_qualification_path'])
    if path.exists():raise FileExistsError('Never overwrite adopted qualification evidence')
    write_json(path,receipt)
    return receipt


def run(plan_path):
    plan_path=Path(plan_path);plan=read(plan_path);root=Path(plan['root'])
    status_path=Path(plan['adopted_qualification_path']).parent/'RESUME_LAUNCH.json'
    with exclusive(root/'RESUME.lock'):
        if status_path.exists():raise FileExistsError('No automatic continuation retry')
        receipt=dict(status='cpu_authentication',binding=plan['binding'],execution_commit=plan['execution_commit'],
            original_execution_commit=plan['original_execution_commit'],started_epoch=time.time(),completed=[],
            qualification_repeated=False,automatic_retry=False)
        write_json(status_path,receipt)
        try:
            launch.check_compilers()
            validated=authenticate_resume(plan)
            write_json(status_path.parent/'AUTHENTICATION.json',validated)
            receipt['authentication']=str(status_path.parent/'AUTHENTICATION.json')
            adopt_qualification(plan,validated)
            resources=launch.admission(plan,launch.production_groups())
            write_json(root/'RESOURCE_ADMISSION.json',resources);write_json(plan_path,plan)
            receipt.update(status='production',resources=resources)
            write_json(status_path,receipt)
            launch.execute_production(plan_path,plan,receipt,status_path,resources)
        except BaseException as error:
            receipt.update(status='stopped',error=repr(error),traceback=traceback.format_exc());raise
        finally:
            receipt['wall_s']=time.time()-receipt['started_epoch'];receipt['clock']=clock_status(root)
            write_json(status_path,receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    args=parser.parse_args();run(args.plan)


if __name__=='__main__':main()
