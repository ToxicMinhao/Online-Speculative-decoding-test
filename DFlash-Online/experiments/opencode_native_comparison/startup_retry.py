"""Explicit recovery of a failed engine startup that executed no scientific work.

The original clock, work ledger, inputs and compiler baseline stay in place.
This is not a general interrupted-training resume or an automatic retry.
"""
import argparse
import copy
from pathlib import Path
import shutil

from online_feedback.storage import write_json
from experiments.ok_coding.storage import identity
from experiments.bounded_prefix_allocation.preflight import verify_source
from .contract import digest
from .runtime import LIMITS, read

ARCHIVE = ('LAUNCH.json', 'LAUNCH_PROCESS.json', 'CPU_PREFLIGHT.json',
           'PROTECTED_DEPENDENCIES.json', 'PROTECTED_RECHECKS.jsonl',
           'launcher.log', 'qualification', 'phases', 'logs', 'processes')
SCOPE = 'retry_startup_qualification_once_then_unchanged_campaign_if_gates_pass'


def validate_origin(plan, authorization, launch, work, clock):
    if authorization.get('status') != 'authorized' or authorization.get('scope') != SCOPE:
        raise ValueError('Explicit startup retry authorization is required')
    if plan['binding'] != digest({k:v for k,v in plan.items() if k != 'binding'}):
        raise ValueError('Original sealed plan changed')
    if (launch['status'] != 'failed' or launch['completed']
            or launch.get('preflight', {}).get('status') != 'passed'):
        raise ValueError('Require a failed startup after passed CPU preflight')
    if (work['limits'] != LIMITS or work['pending'] or work['failures']
            or work['next_operation'] != 0
            or any(work[k] != {key:0 for key in LIMITS} for k in ('attempted','completed'))):
        raise ValueError('Startup recovery requires exactly zero scientific work')
    if clock['active'] or clock['completed_card_s'] < 0 or clock['first_model_load_epoch'] <= 0:
        raise ValueError('Original clock must be complete with no active leases')


def prepare(plan_path, source_root, source_manifest, authorization):
    plan_path, source_root = Path(plan_path), Path(source_root).resolve()
    source_manifest, authorization = Path(source_manifest).resolve(), Path(authorization).resolve()
    old = read(plan_path)
    root = Path(old['root'])
    archive = root/'attempts/failed_startup_001'
    destination = root/'sealed/PLAN_RETRY_001.json'
    receipt_path = root/'STARTUP_RETRY_001.json'
    if archive.exists() or destination.exists() or receipt_path.exists():
        raise FileExistsError('This startup recovery was already prepared or attempted')
    clock, work = read(root/'CLOCK.json'), read(root/'WORK.json')
    validate_origin(old, read(authorization), read(root/'LAUNCH.json'), work, clock)
    source = read(source_manifest)
    verify_source(dict(source_root=str(source_root), source_manifest=str(source_manifest),
                       execution_commit=source['commit']))
    from .protected import check
    check(root, full_hash=True, purpose='before_authorized_startup_retry')
    new = copy.deepcopy(old)
    new.update(source_root=str(source_root), source_manifest=str(source_manifest),
               execution_commit=source['commit'], execution_source=dict(commit=source['commit'],dirty=False))
    if str(source_root.parent) not in new['allocation_roots']:
        new['allocation_roots'].append(str(source_root.parent))
    new['startup_retry'] = dict(authorization=str(authorization), authorization_identity=identity(authorization),
        prior_plan=str(plan_path), prior_plan_identity=identity(plan_path),
        prior_execution_commit=old['execution_commit'], archive=str(archive),
        receipt=str(receipt_path), failed_startup_card_s=clock['completed_card_s'],
        original_piecewise_forward_count='unknown additional initialization')
    new['binding'] = digest({k:v for k,v in new.items() if k != 'binding'})
    archive.mkdir(parents=True)
    for name in ARCHIVE:
        path = root/name
        if path.exists():
            path.rename(archive/name)
    for name in ('CLOCK.json','WORK.json'):
        shutil.copy2(root/name, archive/name)
    shutil.copy2(plan_path, archive/'PLAN.json')
    shutil.copy2(old['source_manifest'], archive/'SOURCE.json')
    files = {str(p.relative_to(archive)):identity(p) for p in sorted(archive.rglob('*')) if p.is_file()}
    write_json(destination,new)
    write_json(receipt_path,dict(status='prepared_once',new_plan=str(destination),
        new_plan_identity=identity(destination),original_clock=clock,original_work=work,
        archived_files=files,archive=str(archive),execution_commit=source['commit'],
        authorization_identity=identity(authorization),scientific_work_repeated=0,
        original_startup_cost_preserved=True,unknown_piecewise_work_preserved=True))
    return dict(plan=str(destination),receipt=str(receipt_path))


def validate_ready(plan, receipt_path):
    root = Path(plan['root'])
    receipt = read(receipt_path)
    retry = plan['startup_retry']
    if (str(Path(receipt_path).resolve()) != retry['receipt']
            or receipt['status'] != 'prepared_once'
            or receipt['new_plan_identity'] != identity(receipt['new_plan'])
            or read(receipt['new_plan']) != plan
            or identity(retry['authorization']) != retry['authorization_identity']
            or identity(retry['prior_plan']) != retry['prior_plan_identity']
            or read(root/'CLOCK.json') != receipt['original_clock']
            or read(root/'WORK.json') != receipt['original_work']):
        raise ValueError('Startup recovery identities, original clock or work changed')
    archive = Path(receipt['archive'])
    for name,pin in receipt['archived_files'].items():
        if identity(archive/name) != pin:
            raise ValueError('Failed startup evidence changed: '+name)
    validate_origin(read(retry['prior_plan']),read(retry['authorization']),
                    read(archive/'LAUNCH.json'),receipt['original_work'],receipt['original_clock'])
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan-path','source-root','source-manifest','authorization'):
        parser.add_argument('--'+key,required=True)
    print(prepare(**vars(parser.parse_args())))
