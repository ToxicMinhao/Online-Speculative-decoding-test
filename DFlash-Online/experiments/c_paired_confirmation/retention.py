"""Retire only verified new derived exports and disposable qualification payloads."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import time

from online_feedback.storage import write_json
from .prepare import identity
from .runtime import read, clock_status


def require_quiet_publication(plan):
    if clock_status(plan['root'])['active']:
        raise RuntimeError('Retirement/reconstruction waits for owned model workers to exit')


def unlink_manifest(plan, paths, *, receipt_path, reason, retained, verification):
    """Exact enumerated new files only; no globs or historical-root retirement."""
    require_quiet_publication(plan)
    root=Path(plan['root']).resolve();paths=[Path(p) for p in paths]
    receipt=dict(status='checking',reason=reason,retained_counterparts=retained,
        verification=verification,started_epoch=time.time(),paths=[],free_before=shutil.disk_usage(root).free)
    if not paths or len(set(paths))!=len(paths):
        raise ValueError('Retirement requires an explicit distinct file list')
    for path in paths:
        if not path.is_relative_to(root) or path.resolve()!=path or not path.is_file():
            raise ValueError('Only regular newly-created run files may be retired')
        stat=path.stat()
        if stat.st_nlink!=1:raise ValueError('Hardlinked payload requires separate admission')
        receipt['paths'].append(dict(path=str(path),identity=identity(path),device=stat.st_dev,
            inode=stat.st_ino,links=stat.st_nlink,allocated_bytes=stat.st_blocks*512))
    # No scientific child remains by the shared lease/process exit boundary.
    # Also check the launcher and remaining descendants for an open/mapped file.
    import psutil
    wanted={str(p) for p in paths};parent=psutil.Process()
    receipt['live_use_checked_pids']=[]
    for proc in [parent,*parent.children(recursive=True)]:
        receipt['live_use_checked_pids'].append(proc.pid)
        live=[e.path.removesuffix(' (deleted)') for e in proc.open_files()]
        live += [e.path.removesuffix(' (deleted)') for e in proc.memory_maps(grouped=False)]
        if wanted.intersection(live):raise RuntimeError('New derived payload is still live')
    receipt['status']='verified_before_deletion';write_json(receipt_path,receipt)
    for row in receipt['paths']:
        path=Path(row['path']);stat=path.stat()
        if (stat.st_dev,stat.st_ino,stat.st_size)!=(row['device'],row['inode'],row['identity']['bytes']):
            raise ValueError('Retirement target changed since verification')
        path.unlink()
        row['removed']=True;write_json(receipt_path,receipt)
    receipt.update(status='completed',finished_epoch=time.time(),
        logical_bytes_removed=sum(r['identity']['bytes'] for r in receipt['paths']),
        allocated_bytes_released=sum(r['allocated_bytes'] for r in receipt['paths']),
        free_after=shutil.disk_usage(root).free)
    receipt['observed_free_change_bytes']=receipt['free_after']-receipt['free_before']
    write_json(receipt_path,receipt)
    return receipt


def retire_endpoint_export(plan,dataset,task):
    from .export import verify_derived_export
    sub=plan['datasets'][dataset];root=Path(plan['root'])
    result=read(root/'tasks'/f'{dataset}_{task}'/'RESULT.json')
    if result['status']!='completed':raise ValueError('Cannot retire an incomplete endpoint')
    saved=result['endpoint'];path=Path(saved['export'])
    if identity(saved['state'])!=saved['state_identity']:
        raise ValueError('Complete retained recovery changed')
    if task.endswith('first'):
        for consumer in (task[0]+'_second',task[0]+'_collect'):
            if read(root/'tasks'/f'{dataset}_{consumer}'/'STATUS.json')['status']!='completed':
                raise ValueError('Serving export still has an uncompleted consumer')
    elif dataset!='gsm8k' or task not in ('N_second','C_second') or plan['storage']['retain_final_exports']:
        raise ValueError('Only the explicitly priced GSM final copies can be temporarily retired')
    qualifier=read(root/'qualification'/f'{dataset}_training_A'/'STATUS.json')
    if qualifier['status']!='passed' or qualifier['result']['cpu_export']['status']!='passed':
        raise ValueError('CPU export reconstruction has not qualified')
    qualified=qualifier['result']
    command=dict(state_path=saved['state'],draft_config_path=qualified['draft_config'],
        output_dir=saved['export'],embedding_source=qualified['embedding_source'],
        embedding_key=qualified['embedding_key'])
    verification=verify_derived_export(saved['state'],saved['export'],reconstruction=command,
        receipt_path=root/'retirement'/f'{dataset}_{task}_DERIVATION.json')
    files=[str(p) for p in sorted(path.iterdir()) if p.is_file()]
    if any(not p.is_file() for p in path.iterdir()):
        raise ValueError('Unexpected nested export layout; retain and reprice')
    return unlink_manifest(plan,files,receipt_path=root/'retirement'/f'{dataset}_{task}_EXPORT.json',
        reason='Qualified derived serving copy; complete state and metadata retained',
        retained=[dict(path=saved['state'],identity=saved['state_identity'])],verification=verification)


def reconstruct_final_export(plan,dataset,task):
    """Do this before any latency sweep; no inference or optimizer execution."""
    from .export import reconstruct_export,export_manifest
    from experiments.ok_continuation.resume import differences
    root=Path(plan['root']);retired=read(root/'retirement'/f'{dataset}_{task}_EXPORT.json')
    if retired['status']!='completed':raise ValueError('Missing verified derived-copy retirement')
    check=retired['verification'];command=check['reconstruction'];out=Path(command['output_dir'])
    if any(out.iterdir()):raise ValueError('Reconstruction destination is not empty')
    require_quiet_publication(plan)
    for row in retired['retained_counterparts']:
        if identity(row['path'])!=row['identity']:raise ValueError('Retained reconstruction source changed')
    out.rmdir();began=time.perf_counter();reconstruct_export(**command)
    actual=export_manifest(out);errors=differences(actual,check['manifest'])
    receipt=dict(status='passed' if not errors else 'failed',wall_s=time.perf_counter()-began,
        errors=errors,manifest=actual,command=command,additional_gpu_work=0,additional_training=0)
    result=read(root/'tasks'/f'{dataset}_{task}'/'RESULT.json')
    receipt['weights_file_identity']=identity(out/'model.safetensors')
    if receipt['weights_file_identity']!=result['endpoint']['weights_identity']:
        receipt['status']='failed';receipt['errors'].append('Serving file identity differs from original export')
    write_json(root/'retirement'/f'{dataset}_{task}_RECONSTRUCTION.json',receipt)
    if receipt['status']!='passed':raise ValueError('CPU serving reconstruction differs')
    return receipt
